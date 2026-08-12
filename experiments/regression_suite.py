#!/usr/bin/env python
"""The CI eval gate. Fails the build when evaluation quality regresses.

Eval-driven development means treating evaluation like a test suite: a change
that quietly makes the guardrail worse should break the build the same way a
failing unit test does. Almost no portfolio project has this, and it is the
single most current practice in the repo.

The gate is deliberately cheap -- a small deterministic slice against the
simulator -- so it runs on every push in under a minute with no API key and no
quota. It checks behaviour, not vibes:

  1. the guardrail's accuracy against degradation ground truth
  2. its calibration error
  3. its p99 latency against the configured budget
  4. structural invariants that must hold for the method to mean anything

Exit code 1 on any failure.

    python experiments/regression_suite.py --fail-under 0.85
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict

import numpy as np
from _common import REPO, apply_quick_isolation, corpus

from judgeguard.config import get_settings
from judgeguard.degrade.text import ERROR_DEGRADATIONS, PROBE_DEGRADATIONS, build_variants
from judgeguard.distill.features import render_example
from judgeguard.distill.train import Student, expected_calibration_error
from judgeguard.store import RESULTS_DIR, load, save

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Gate:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(
        self, name: str, ok: bool, observed: object, expected: str, *, gating: bool = True
    ) -> None:
        """Record a check. ``gating=False`` reports it without failing the build.

        Used in smoke mode for the checks that are meaningless at tiny sample
        sizes -- a guardrail trained on 80 examples has no stable precision, and
        failing the build on it would train people to ignore the gate.
        """
        self.checks.append(
            {
                "name": name,
                "passed": bool(ok),
                "observed": observed,
                "expected": expected,
                "gating": gating,
            }
        )
        mark = (
            (f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}")
            if gating
            else (f"{GREEN}pass{RESET}" if ok else f"{YELLOW}warn{RESET}")
        )
        print(f"  [{mark}] {name:<44} {observed}   (expected {expected})")

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"] and c["gating"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fail-under",
        type=float,
        default=0.85,
        help="minimum guardrail accuracy against ground truth",
    )
    ap.add_argument("--max-ece", type=float, default=0.15)
    ap.add_argument("--items", type=int, default=120)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke mode: structural + latency checks only, quality bounds relaxed",
    )
    ap.add_argument("--seed", type=int, default=get_settings().seed)
    args = apply_quick_isolation(ap.parse_args())

    if args.quick:
        # A student trained on 80 examples is genuinely bad, and gating it at
        # production bounds would fail every smoke run for the right reason at the
        # wrong time. Structural invariants and latency still apply -- those are
        # sample-size independent and are what a smoke run is actually for.
        args.fail_under = min(args.fail_under, 0.55)
        args.items = min(args.items, 40)

    print("\n=== JudgeGuard eval gate ===")
    if args.quick:
        print("    smoke mode: quality bounds relaxed; structural + latency checks unchanged")
    g = Gate()
    t0 = time.perf_counter()

    # ---------------------------------------------------- structural invariants
    items, by_id = corpus(args.items, args.seed)
    variants = build_variants(items, seed=args.seed)
    g.check(
        "corpus is non-degenerate",
        len({i.reference for i in items}) == len(items),
        f"{len({i.reference for i in items})} unique / {len(items)}",
        "all distinct",
    )
    g.check(
        "every variant carries an audit trail",
        all(len(v.edit) > 20 for v in variants),
        "all",
        "non-empty edit description",
    )
    probes = [v for v in variants if not v.introduces_error]
    g.check(
        "verbosity probe introduces no error",
        all(v.degradation in PROBE_DEGRADATIONS for v in probes) and len(probes) > 0,
        f"{len(probes)} probe variants",
        "only the verbosity probe",
    )
    g.check(
        "error variants differ from their reference",
        all(v.text != by_id[v.item_id].reference for v in variants if v.introduces_error),
        "all differ",
        "no accidental no-ops",
    )
    g.check(
        "degradation set complete",
        sorted({v.degradation for v in variants})
        == sorted(ERROR_DEGRADATIONS + PROBE_DEGRADATIONS),
        len({v.degradation for v in variants}),
        f"{len(ERROR_DEGRADATIONS) + len(PROBE_DEGRADATIONS)} types",
    )

    # ------------------------------------------------------------ guardrail gate
    model_path = RESULTS_DIR / "student_model.joblib"
    if not model_path.exists():
        print(f"  [{RED}FAIL{RESET}] student model missing — run experiments/09_distill.py")
        return 1
    student = Student.load(model_path)

    # The guardrail is evaluated on the protocol it is deployed under, not on the
    # battery's enumeration. Scoring it against the raw 1:15 cross-product would
    # measure a task it was deliberately not built for -- and would report ~0.48
    # for a model that never blocks a correct answer.
    rng = random.Random(args.seed)
    by_item: dict[str, dict[str, list]] = defaultdict(lambda: {"error": [], "probe": []})
    for v in variants:
        by_item[v.item_id]["error" if v.introduces_error else "probe"].append(v)

    texts: list[str] = []
    truth: list[int] = []
    degs: list[str] = []
    for i in items:
        texts.append(render_example(i.question, i.reference, i.context))
        truth.append(1)
        degs.append("reference")
        for v in rng.sample(by_item[i.id]["error"], min(1, len(by_item[i.id]["error"]))):
            texts.append(render_example(i.question, v.text, i.context))
            truth.append(0)
            degs.append(v.degradation)

    y = np.asarray(truth)
    p = student.predict_proba(texts)
    pred = (p >= student.threshold).astype(int)
    acc = float((pred == y).mean())
    ece, _ = expected_calibration_error(y, p)
    blocked = pred == 0
    block_precision = float((blocked & (y == 0)).sum() / max(blocked.sum(), 1))
    false_block = float(((y == 1) & blocked).sum() / max((y == 1).sum(), 1))
    caught = float(((y == 0) & blocked).sum() / max((y == 0).sum(), 1))

    g.check(
        "guardrail accuracy vs ground truth",
        acc >= args.fail_under,
        f"{acc:.4f}",
        f">= {args.fail_under}",
    )
    # The one a guardrail lives or dies by: never block a correct answer.
    # Guardrail quality has no stable estimate on an 80-example smoke student, so
    # in quick mode these are reported and not gated. Structural invariants and
    # latency are sample-size independent and stay gating throughout.
    quality = not args.quick
    min_precision = 0.60 if args.quick else 0.90
    max_false_block = 0.35 if args.quick else 0.05
    min_caught = 0.15 if args.quick else 0.35
    g.check(
        "guardrail block precision",
        block_precision >= min_precision,
        f"{block_precision:.4f}",
        f">= {min_precision:.2f}",
        gating=quality,
    )
    g.check(
        "guardrail false-block rate",
        false_block <= max_false_block,
        f"{false_block:.4f}",
        f"<= {max_false_block:.2f}",
        gating=quality,
    )
    g.check(
        "guardrail catches bad output",
        caught >= min_caught,
        f"{caught:.4f}",
        f">= {min_caught:.2f}",
        gating=quality,
    )
    g.check("guardrail calibration error", ece <= args.max_ece, f"{ece:.4f}", f"<= {args.max_ece}")

    # Documented blind spots. This check fires in BOTH directions on purpose: a
    # regression is a bug, and a silent improvement means docs/findings.md and
    # docs/limitations.md now describe a model that no longer exists.
    for spot in ("numeric_swap", "topic_drift"):
        flags = [int(pred[k] == y[k]) for k, dg in enumerate(degs) if dg == spot]
        rate = sum(flags) / max(len(flags), 1)
        g.check(
            f"documented blind spot: {spot}",
            rate <= 0.25,
            f"{rate:.4f} detected",
            "<= 0.25 (documented as structural)",
        )

    # -------------------------------------------------------------- latency gate
    budget = get_settings().guard_p99_budget_ms
    for _ in range(20):
        student.guard(items[0].question, items[0].reference, items[0].context)
    lat = []
    for i in items[:100]:
        t = time.perf_counter()
        student.guard(i.question, i.reference, i.context)
        lat.append((time.perf_counter() - t) * 1000)
    p99 = float(np.percentile(lat, 99))
    g.check("guardrail p99 latency", p99 <= budget, f"{p99:.2f} ms", f"<= {budget:.0f} ms")

    # ------------------------------------------------- headline claims still hold
    try:
        d = load("01_discrimination")
        for judge in d["judges"]:
            est = d["per_judge"][judge]["overall_accuracy"]
            g.check(
                f"CI reported for {judge}",
                est["hi"] > est["lo"] or est["n"] < 3,
                f"[{est['lo']:.3f}, {est['hi']:.3f}]",
                "a real interval",
            )
        tr = load("08_trajectory_blindness")
        worst = max(
            tr["judges"],
            key=lambda j: tr["per_judge"][j]["by_degradation"]["wrong_argument"]["miss_rate"],
        )
        miss = tr["per_judge"][worst]["by_degradation"]["wrong_argument"]["miss_rate"]
        g.check(
            "argument-blindness finding present",
            miss > 0.0,
            f"{miss:.3f} miss rate ({worst})",
            "> 0",
        )
    except FileNotFoundError as exc:
        g.check(
            "battery results committed",
            False,
            str(exc),
            "results/*.json present",
            gating=not args.quick,
        )

    payload = {
        "gate": "regression_suite",
        "passed": not g.failed,
        "thresholds": {
            "fail_under": args.fail_under,
            "max_ece": args.max_ece,
            "p99_budget_ms": budget,
        },
        "measurements": {
            "accuracy": acc,
            "block_precision": block_precision,
            "false_block_rate": false_block,
            "bad_output_caught": caught,
            "ece": ece,
            "p99_ms": p99,
        },
        "checks": g.checks,
        "seconds": time.perf_counter() - t0,
    }
    save("regression_gate", payload)

    n_fail = len(g.failed)
    print(
        f"\n  {len(g.checks) - n_fail}/{len(g.checks)} checks passed in {time.perf_counter() - t0:.1f}s"
    )
    if n_fail:
        print(f"  {RED}gate failed:{RESET} " + ", ".join(c["name"] for c in g.failed))
        return 1
    print(f"  {GREEN}gate passed{RESET}  ({REPO.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
