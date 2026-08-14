#!/usr/bin/env python
"""THE MAIN FIGURE: where does each judge go blind?

Pointwise scoring of the reference and of every error-introducing variant. A
judge is credited only when it scores the reference strictly above the broken
copy. Ties count as failures, which is the conservative choice and is stated
rather than hidden.

Sweeping severity turns "judges are unreliable" into a curve with an x-axis, and
the per-degradation breakdown says *which* errors a judge cannot see -- which is
the operationally useful part.

    python experiments/01_discrimination.py --items 500
"""

from __future__ import annotations

from collections import defaultdict

from _common import (
    apply_quick_isolation,
    banner,
    base_parser,
    corpus,
    done,
    resolve_judges,
    run_main,
)

from judgeguard.degrade.text import ERROR_DEGRADATIONS, SEVERITIES, build_variants
from judgeguard.judges.run import ScoreTask, run_scores
from judgeguard.stats.intervals import bca_ci
from judgeguard.stats.tests import severity_monotonicity
from judgeguard.store import save
from judgeguard.telemetry import tracking_run

CONFIG = "cot"


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--config", default=CONFIG, choices=["vague", "rubric", "cot"])
    args = apply_quick_isolation(ap.parse_args())
    n_items = 40 if args.quick else args.items
    judges = resolve_judges(args.judges)

    t0 = banner("01  discrimination: accuracy vs severity")
    items, by_id = corpus(n_items, args.seed)
    variants = build_variants(items, seed=args.seed, degradations=ERROR_DEGRADATIONS)

    per_judge: dict[str, dict] = {}
    records: list[dict] = []

    with tracking_run(
        "judgeguard",
        f"01_discrimination_{args.config}",
        {"items": n_items, "config": args.config, "judges": judges},
    ) as run:
        for judge in judges:
            ref_tasks = [
                ScoreTask(
                    uid=f"{i.id}:reference",
                    item=i,
                    answer=i.reference,
                    degradation="reference",
                    severity=0.0,
                    len_ratio=1.0,
                )
                for i in items
            ]
            var_tasks = [
                ScoreTask(
                    uid=v.uid,
                    item=by_id[v.item_id],
                    answer=v.text,
                    degradation=v.degradation,
                    severity=v.severity,
                    len_ratio=v.len_ratio,
                )
                for v in variants
            ]
            ref_j = run_scores(judge, args.config, ref_tasks)
            var_j = run_scores(judge, args.config, var_tasks)
            ref_score = {j.uid.split(":")[0]: j.score for j in ref_j}

            correct_flat: list[int] = []
            by_deg: dict[str, list[int]] = defaultdict(list)
            by_deg_sev: dict[tuple[str, float], list[int]] = defaultdict(list)
            sev_x: dict[str, list[float]] = defaultdict(list)
            sev_y: dict[str, list[float]] = defaultdict(list)
            ties = 0
            parse_fail = 0

            for j in var_j:
                item_id = j.uid.split(":")[0]
                r, d = ref_score.get(item_id), j.score
                if r is None or d is None:
                    parse_fail += 1
                    continue
                ok = int(d < r)
                ties += int(d == r)
                correct_flat.append(ok)
                by_deg[j.degradation].append(ok)
                by_deg_sev[(j.degradation, j.severity)].append(ok)
                sev_x[j.degradation].append(j.severity)
                sev_y[j.degradation].append(d)
                records.append(
                    {
                        "uid": j.uid,
                        "judge": judge,
                        "config": args.config,
                        "degradation": j.degradation,
                        "severity": j.severity,
                        "reference_score": r,
                        "variant_score": d,
                        "correct": ok,
                        "latency_ms": j.latency_ms,
                        "cost_usd": j.cost_usd,
                        "prompt_tokens": j.prompt_tokens,
                        "completion_tokens": j.completion_tokens,
                    }
                )

            overall = bca_ci(correct_flat, seed=args.seed)
            per_judge[judge] = {
                "overall_accuracy": overall.as_dict(),
                "n_pairs": len(correct_flat),
                "tie_rate": ties / max(len(correct_flat), 1),
                "parse_failure_rate": parse_fail / max(len(var_j), 1),
                "mean_reference_score": float(
                    sum(v for v in ref_score.values() if v is not None) / max(len(ref_score), 1)
                ),
                "by_degradation": {
                    d: {
                        **bca_ci(v, seed=args.seed).as_dict(),
                        "monotonicity": severity_monotonicity(sev_x[d], sev_y[d]),
                    }
                    for d, v in sorted(by_deg.items())
                },
                "by_degradation_severity": {
                    f"{d}@{s:g}": bca_ci(v, seed=args.seed).as_dict()
                    for (d, s), v in sorted(by_deg_sev.items())
                },
                "cost_usd_total": sum(j.cost_usd for j in ref_j + var_j),
                "latency_ms_mean": sum(j.latency_ms for j in var_j) / max(len(var_j), 1),
            }
            print(
                f"    {judge:<14} accuracy {overall.pct()}  "
                f"worst: {min(by_deg, key=lambda d: sum(by_deg[d]) / len(by_deg[d]))}"
            )
            run.log_metrics({f"{judge}_accuracy": overall.value})

    payload = {
        "experiment": "01_discrimination",
        "question": "Can each judge rank a known-good answer above a deliberately broken copy?",
        "config": args.config,
        "judges": judges,
        "n_items": len(items),
        "degradations": ERROR_DEGRADATIONS,
        "severities": list(SEVERITIES),
        "scoring_rule": "correct iff score(reference) > score(variant); ties count as failures",
        "per_judge": per_judge,
        "records": records,
    }
    done(t0, save("01_discrimination", payload))


if __name__ == "__main__":
    run_main(main)
