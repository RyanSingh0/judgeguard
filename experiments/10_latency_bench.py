#!/usr/bin/env python
"""Latency, measured rather than assumed.

The distinction this experiment exists to defend:

* an **async evaluator** scores after the fact for dashboards and regression
  tracking, and latency is irrelevant;
* an **inline guardrail** blocks a bad response before the user sees it, and has
  a hard millisecond budget.

They are not interchangeable. Building an async evaluator where a guardrail was
needed means harmful output reaches users while the evaluation is still running.

So: a real closed-loop benchmark of the student path, warm and under concurrency,
reported as p50/p95/p99 against an explicit 150 ms p99 budget. The LLM judge
latencies are quoted from the same run for contrast.

    python experiments/10_latency_bench.py --requests 2000 --concurrency 8
"""

from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from _common import apply_quick_isolation, banner, base_parser, corpus, done

from judgeguard.config import get_settings
from judgeguard.degrade.text import build_variants
from judgeguard.distill.features import render_example
from judgeguard.distill.train import Student
from judgeguard.store import RESULTS_DIR, load, save


def percentiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)

    def q(p: float) -> float:
        if not s:
            return float("nan")
        k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[k]

    return {
        "n": len(s),
        "mean": statistics.fmean(s) if s else float("nan"),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": s[-1] if s else float("nan"),
    }


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = apply_quick_isolation(ap.parse_args())
    n_req = 200 if args.quick else args.requests
    budget = get_settings().guard_p99_budget_ms

    t0 = banner(f"10  latency: guardrail path against a {budget:.0f} ms p99 budget")
    student = Student.load(RESULTS_DIR / "student_model.joblib")

    items, by_id = corpus(60, args.seed)
    variants = build_variants(items, seed=args.seed)
    payloads = [(by_id[v.item_id].question, v.text, by_id[v.item_id].context) for v in variants] + [
        (i.question, i.reference, i.context) for i in items
    ]

    # Warm-up: first calls pay for lazy imports and cold caches, and including
    # them would flatter or wreck the percentiles depending on the machine.
    for q, a, ctx in payloads[:25]:
        student.guard(q, a, ctx)

    def one(k: int) -> float:
        q, a, ctx = payloads[k % len(payloads)]
        t = time.perf_counter()
        student.guard(q, a, ctx)
        return (time.perf_counter() - t) * 1000

    serial = [one(k) for k in range(n_req)]

    # Concurrency sweep. One p99 number is a machine fingerprint; the useful
    # engineering output is the concurrency at which the budget still holds,
    # because that is what sizes the deployment. The guardrail is CPU-bound, so
    # past the core count the queue -- not the model -- owns the tail.
    sweep: dict[str, dict] = {}
    max_ok = 0
    for workers in args.concurrency_levels:
        n = max(200, n_req // 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            t_start = time.perf_counter()
            lat = list(pool.map(one, range(n)))
            wall = time.perf_counter() - t_start
        stats = percentiles(lat)
        stats["throughput_rps"] = n / max(wall, 1e-9)
        stats["within_budget"] = bool(stats["p99"] <= budget)
        sweep[str(workers)] = stats
        if stats["within_budget"]:
            max_ok = workers
        print(
            f"    concurrency {workers:>3}  p50 {stats['p50']:7.2f}  p95 {stats['p95']:7.2f}  "
            f"p99 {stats['p99']:7.2f} ms  {stats['throughput_rps']:6.0f} rps  "
            f"{'within budget' if stats['within_budget'] else 'OVER BUDGET'}"
        )

    p_conc = sweep[str(args.concurrency)] if str(args.concurrency) in sweep else percentiles(serial)
    wall = 1.0 / max(p_conc.get("throughput_rps", 1.0), 1e-9) * max(n_req, 1)

    # Batched scoring: the async-evaluator path, where throughput is what matters.
    batch_texts = [render_example(q, a, ctx) for q, a, ctx in payloads[:512]]
    t = time.perf_counter()
    student.predict_proba(batch_texts)
    batch_ms = (time.perf_counter() - t) * 1000

    p_serial = percentiles(serial)

    judge_latency = {}
    try:
        cost = load("07_cost_accuracy")
        judge_latency = {r["judge"]: r["mean_latency_ms"] for r in cost["rows"]}
    except FileNotFoundError:
        pass

    payload = {
        "experiment": "10_latency_bench",
        "question": "Does the guardrail path fit inside its latency budget, and what does it trade?",
        "budget_p99_ms": budget,
        # This experiment measures a real model on real hardware. The simulator
        # never enters the guardrail path, so this artefact is not watermarked
        # even when the rest of the battery is running in simulated mode.
        "real_measurement": True,
        "featurizer": student.featurizer_kind,
        "threshold": student.threshold,
        "cpu_count": os.cpu_count(),
        "serial": p_serial,
        "concurrency_sweep": sweep,
        "max_concurrency_within_budget": max_ok,
        f"concurrent_{args.concurrency}_workers": p_conc,
        "throughput_rps_at_concurrency": p_conc.get("throughput_rps"),
        "batched_512_total_ms": batch_ms,
        "batched_per_item_ms": batch_ms / 512,
        "meets_budget_serial": bool(p_serial["p99"] <= budget),
        "meets_budget_concurrent": bool(p_conc["p99"] <= budget),
        "headroom_ms": budget - p_serial["p99"],
        "llm_judge_mean_latency_ms": judge_latency,
        # Single-stream against single-stream: the LLM judge latencies are
        # per-request means, so comparing them to a concurrent p50 would be
        # flattering and wrong.
        "speedup_vs_cheapest_judge_serial": (
            min(judge_latency.values()) / max(p_serial["p50"], 1e-9) if judge_latency else None
        ),
        "speedup_vs_cheapest_judge_at_concurrency": (
            min(judge_latency.values()) / max(p_conc["p50"], 1e-9) if judge_latency else None
        ),
        "interpretation": (
            "The guardrail path is the distilled student only. No LLM call happens inside the "
            "request, which is the only way a hard budget can be honoured. The LLM judge stays "
            "available on the async /evaluate path, where latency is a throughput question "
            "rather than a user-facing one. The sweep is the deployable result: the budget holds "
            f"up to concurrency {max_ok} on {os.cpu_count()} cores, and past that the queue owns "
            "the tail, so the fix is horizontal replicas rather than a smaller model."
        ),
        "caveat": (
            "Measured on the machine that produced results/. Absolute numbers are hardware "
            "dependent; the ratio between the guardrail path and the judge path is not."
        ),
    }
    print(
        f"    serial       p50 {p_serial['p50']:7.2f}  p95 {p_serial['p95']:7.2f}  "
        f"p99 {p_serial['p99']:7.2f} ms"
    )
    print(
        f"    budget {budget:.0f} ms p99 -> per-request "
        f"{'MET' if payload['meets_budget_serial'] else 'MISSED'} "
        f"({payload['headroom_ms']:+.1f} ms headroom); holds up to concurrency {max_ok} "
        f"on {os.cpu_count()} cores"
    )
    done(t0, save("10_latency_bench", payload))


if __name__ == "__main__":
    main()
