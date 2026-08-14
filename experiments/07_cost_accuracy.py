#!/usr/bin/env python
"""Accuracy per dollar and per second. The argument for distilling at all.

Accuracy alone picks the biggest model every time. A judge that is three points
better at thirty times the price is usually the wrong choice, and saying so out
loud is the whole point of this table.

Costs are computed at published list prices even though every call in this
project runs on a free tier -- reporting $0 would make the comparison
meaningless. That substitution is stated, not buried.

Depends on: 01_discrimination.json, 02_position_bias.json.

    python experiments/07_cost_accuracy.py
"""

from __future__ import annotations

from _common import apply_quick_isolation, banner, base_parser, done, run_main

from judgeguard.config import load_registry
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.store import load, save


def main() -> None:
    ap = base_parser(__doc__ or "")
    args = apply_quick_isolation(ap.parse_args())

    t0 = banner("07  cost / accuracy / latency frontier")
    disc = load("01_discrimination")
    reg = load_registry()

    per_judge_correct: dict[str, list[int]] = {}
    per_judge_uid: dict[str, list[str]] = {}
    for r in disc["records"]:
        per_judge_correct.setdefault(r["judge"], []).append(r["correct"])
        per_judge_uid.setdefault(r["judge"], []).append(r["uid"])

    rows = []
    for judge in disc["judges"]:
        stats = disc["per_judge"][judge]
        spec = reg.by_alias(judge)
        acc = stats["overall_accuracy"]
        cost = stats["cost_usd_total"]
        n = stats["n_pairs"]
        correct_n = acc["value"] * n
        rows.append(
            {
                "judge": judge,
                "provider": spec.provider,
                "family": spec.family,
                "tier": spec.tier,
                "accuracy": acc,
                "total_cost_usd_list_price": cost,
                "cost_per_1k_judgments_usd": 1000 * cost / max(n, 1),
                "cost_per_correct_judgment_usd": cost / max(correct_n, 1e-9),
                "mean_latency_ms": stats["latency_ms_mean"],
                "judgments_per_second_single_stream": 1000 / max(stats["latency_ms_mean"], 1e-9),
                "accuracy_per_dollar": acc["value"] / max(cost, 1e-12),
            }
        )

    frontier = max(rows, key=lambda r: r["accuracy"]["value"])
    best_open = max(
        (r for r in rows if r["tier"] == "open-weight"),
        key=lambda r: r["accuracy"]["value"],
        default=None,
    )
    headline = None
    if best_open and best_open["judge"] != frontier["judge"]:
        gap = paired_bootstrap_diff(
            per_judge_correct[best_open["judge"]],
            per_judge_correct[frontier["judge"]],
            seed=args.seed,
        )
        open_cost = best_open["cost_per_1k_judgments_usd"]
        front_cost = frontier["cost_per_1k_judgments_usd"]
        ratio = open_cost / max(front_cost, 1e-12)  # >1 means the open model costs MORE
        cheapest = min(rows, key=lambda r: r["cost_per_correct_judgment_usd"])
        headline = {
            "frontier_judge": frontier["judge"],
            "best_open_weight_judge": best_open["judge"],
            "accuracy_retained_pct": 100
            * best_open["accuracy"]["value"]
            / frontier["accuracy"]["value"],
            "accuracy_gap": gap.as_dict(),
            "open_cost_over_frontier_cost": ratio,
            "cheapest_per_correct_judgment": cheapest["judge"],
            "latency_ratio_frontier_over_open": frontier["mean_latency_ms"]
            / max(best_open["mean_latency_ms"], 1e-9),
            "sentence": (
                f"{best_open['judge']} retained "
                f"{100 * best_open['accuracy']['value'] / frontier['accuracy']['value']:.1f}% of "
                f"{frontier['judge']}'s discrimination accuracy while costing "
                f"{ratio:.1f}x as much per 1k judgments at list prices"
                if ratio >= 1
                else f"{best_open['judge']} retained "
                f"{100 * best_open['accuracy']['value'] / frontier['accuracy']['value']:.1f}% of "
                f"{frontier['judge']}'s discrimination accuracy at {1 / ratio:.1f}x lower cost per 1k judgments"
            ),
            "note": (
                "The open-weight cost advantage that motivates most judge-selection advice does "
                "not survive contact with 2026 list prices. Whichever way this lands, no LLM "
                "judge in the panel is cheap enough or fast enough to run inline -- which is the "
                "argument for distillation, not for picking a different judge."
            ),
        }

    # Cost of the swap protocol: 2x the calls, and here is what it bought.
    swap_cost = None
    try:
        pos = load("02_position_bias")
        swap_cost = {
            judge: {
                "extra_calls_multiplier": 2,
                "accuracy_gain_over_single_call": pos["per_judge"][judge]["swap_vs_single_call"],
                "inconsistency_rate": pos["per_judge"][judge]["inconsistency_rate"],
            }
            for judge in pos["judges"]
        }
    except FileNotFoundError:
        pass

    pareto = []
    for r in sorted(rows, key=lambda r: r["cost_per_1k_judgments_usd"]):
        if not pareto or r["accuracy"]["value"] > pareto[-1]["accuracy"]["value"]:
            pareto.append(r)

    for r in rows:
        print(
            f"    {r['judge']:<14} acc {100 * r['accuracy']['value']:5.1f}%   "
            f"${r['cost_per_1k_judgments_usd']:7.3f}/1k   "
            f"{r['mean_latency_ms']:6.0f} ms   "
            f"${r['cost_per_correct_judgment_usd'] * 1000:6.3f}/1k correct"
        )
    if headline:
        print(f"    -> {headline['sentence']}")

    payload = {
        "experiment": "07_cost_accuracy",
        "question": "What does each accuracy point actually cost, in dollars and in milliseconds?",
        "pricing_note": (
            "All calls ran on free tiers. Costs use each provider's published list price so the "
            "comparison is meaningful; free-tier cost would be $0 for every row."
        ),
        "rows": rows,
        "pareto_frontier": [r["judge"] for r in pareto],
        "headline": headline,
        "swap_protocol_economics": swap_cost,
        "accuracy_ci_note": "Accuracy intervals are the BCa intervals from 01_discrimination.",
        "n_pairs": {j: len(v) for j, v in per_judge_correct.items()},
        "sanity": {j: bca_ci(v, seed=args.seed).as_dict() for j, v in per_judge_correct.items()},
    }
    done(t0, save("07_cost_accuracy", payload))


if __name__ == "__main__":
    run_main(main)
