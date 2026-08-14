#!/usr/bin/env python
"""Position bias, and what swap-and-average actually buys.

Every comparison runs in both orderings. The disagreement rate between them is
not a proxy for position bias -- it *is* position bias, measured directly.

Four accuracy views are reported, and the gap between the first two is the point:
a harness that always shows the reference first reports a number inflated by the
bias it failed to control for.

    python experiments/02_position_bias.py --items 250
"""

from __future__ import annotations

from collections import defaultdict

from _common import (
    N_PAIRWISE_DEFAULT,
    apply_quick_isolation,
    banner,
    base_parser,
    corpus,
    done,
    resolve_judges,
    run_main,
    severities_from,
)

from judgeguard.degrade.text import ERROR_DEGRADATIONS, build_variants
from judgeguard.judges.run import PairTask, run_pairwise
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.store import save
from judgeguard.telemetry import tracking_run


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.set_defaults(items=N_PAIRWISE_DEFAULT)
    args = apply_quick_isolation(ap.parse_args())
    n_items = 30 if args.quick else args.items
    judges = resolve_judges(args.judges)

    t0 = banner("02  position bias and the swap protocol")
    items, by_id = corpus(n_items, args.seed)
    variants = build_variants(
        items,
        seed=args.seed,
        degradations=ERROR_DEGRADATIONS,
        severities=severities_from(args),
    )
    tasks = [
        PairTask(
            uid=v.uid,
            item=by_id[v.item_id],
            reference=by_id[v.item_id].reference,
            degraded=v.text,
            degradation=v.degradation,
            severity=v.severity,
            len_ratio=v.len_ratio,
        )
        for v in variants
    ]

    per_judge: dict[str, dict] = {}
    records: list[dict] = []
    with tracking_run(
        "judgeguard", "02_position_bias", {"items": n_items, "judges": judges}
    ) as run:
        for judge in judges:
            out = run_pairwise(judge, "pairwise", tasks)
            first = [o.correct_ref_first for o in out]
            degf = [o.correct_deg_first for o in out]
            rand = [o.correct_single_random for o in out]
            swap = [o.correct_swapped for o in out]
            incons = [0 if o.consistent else 1 for o in out]
            tie = [1 if o.forward == "tie" or o.reverse == "tie" else 0 for o in out]

            by_deg = defaultdict(list)
            for o in out:
                by_deg[o.degradation].append(0 if o.consistent else 1)

            inflation = paired_bootstrap_diff(first, rand, seed=args.seed)
            mitigation = paired_bootstrap_diff(swap, rand, seed=args.seed)

            per_judge[judge] = {
                "n_pairs": len(out),
                "inconsistency_rate": bca_ci(incons, seed=args.seed).as_dict(),
                "accuracy_reference_first": bca_ci(first, seed=args.seed).as_dict(),
                "accuracy_degraded_first": bca_ci(degf, seed=args.seed).as_dict(),
                "accuracy_random_order": bca_ci(rand, seed=args.seed).as_dict(),
                "accuracy_swap_and_average": bca_ci(swap, seed=args.seed).as_dict(),
                "tie_rate": sum(tie) / max(len(tie), 1),
                "parse_failure_rate": sum(0 if o.parsed_ok else 1 for o in out) / max(len(out), 1),
                "fixed_order_inflation": inflation.as_dict(),
                "swap_vs_single_call": mitigation.as_dict(),
                "inconsistency_by_degradation": {
                    d: bca_ci(v, seed=args.seed).as_dict() for d, v in sorted(by_deg.items())
                },
                "cost_usd_total": sum(o.cost_usd for o in out),
                "calls_per_comparison": 2,
            }
            records.extend(
                {
                    "uid": o.uid,
                    "judge": judge,
                    "degradation": o.degradation,
                    "severity": o.severity,
                    "forward": o.forward,
                    "reverse": o.reverse,
                    "consistent": int(o.consistent),
                    "correct_ref_first": o.correct_ref_first,
                    "correct_swapped": o.correct_swapped,
                }
                for o in out
            )
            print(
                f"    {judge:<14} order disagreement {100 * sum(incons) / len(incons):5.1f}%   "
                f"ref-first {100 * sum(first) / len(first):5.1f}%  ->  "
                f"random-order {100 * sum(rand) / len(rand):5.1f}%  ->  "
                f"swapped {100 * sum(swap) / len(swap):5.1f}%"
            )
            run.log_metrics({f"{judge}_inconsistency": sum(incons) / len(incons)})

    payload = {
        "experiment": "02_position_bias",
        "question": "How often does a judge change its mind when the two options are swapped?",
        "protocol": (
            "Each pair is judged twice: reference-first and degraded-first. Verdicts are "
            "normalised into REF/DEG space. Disagreement between the two orderings is the "
            "position-bias metric. Swap-and-average scores a disagreement as a miss."
        ),
        "judges": judges,
        "n_items": len(items),
        "per_judge": per_judge,
        "records": records,
    }
    done(t0, save("02_position_bias", payload))


if __name__ == "__main__":
    run_main(main)
