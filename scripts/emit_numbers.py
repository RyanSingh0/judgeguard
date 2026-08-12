#!/usr/bin/env python
"""Print every headline figure, derived from results/*.json.

Single source of truth for the prose. Run this, then write the docs from its
output -- never the other way round.

    python scripts/emit_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
L = lambda n: json.loads((R / f"{n}.json").read_text(encoding="utf-8"))  # noqa: E731
pc = lambda e, d=1: f"{100 * e['value']:.{d}f}% [{100 * e['lo']:.{d}f}, {100 * e['hi']:.{d}f}]"  # noqa: E731


def main() -> None:
    d0, d1, d2, d3 = (
        L("degraded_set"),
        L("01_discrimination"),
        L("02_position_bias"),
        L("03_verbosity_bias"),
    )
    d4, d5, d6 = L("04_self_enhancement"), L("05_rubric_ablation"), L("06_self_consistency")
    d7, d8, d9, d10 = (
        L("07_cost_accuracy"),
        L("08_trajectory_blindness"),
        L("09_distill"),
        L("10_latency_bench"),
    )
    J = d1["judges"]

    print(f"PROVENANCE  mode={d1['provenance']['mode']}  seed={d1['provenance']['seed']}")
    print(f"COUNTS      {d0['counts']}")
    print(
        f"POWER       n_required={d0['power_analysis']['n_for_independent_samples']['n_required']}  "
        f"mcnemar={d0['power_analysis']['n_for_paired_mcnemar_at_20pct_discordant']}"
    )

    print("\n== 01 DISCRIMINATION ==")
    for j in J:
        p = d1["per_judge"][j]
        by = p["by_degradation"]
        print(
            f"  {j:13} {pc(p['overall_accuracy'])}  tie={100 * p['tie_rate']:.1f}%  "
            + "  ".join(f"{k}={100 * v['value']:.1f}" for k, v in by.items())
        )
        ns = p["by_degradation_severity"]
        print(
            "       numeric_swap by sev: "
            + "  ".join(f"{s}={pc(ns[f'numeric_swap@{s}'])}" for s in ("0.2", "0.5", "0.9"))
        )
        print(f"       rho(numeric_swap)={by['numeric_swap']['monotonicity']['spearman_rho']:.3f}")

    print("\n== 02 POSITION ==")
    for j in J:
        a = d2["per_judge"][j]
        print(
            f"  {j:13} inconsistency={pc(a['inconsistency_rate'])}  ref_first={pc(a['accuracy_reference_first'])}  "
            f"random={pc(a['accuracy_random_order'])}  swap={pc(a['accuracy_swap_and_average'])}"
        )
        i = a["fixed_order_inflation"]
        print(
            f"       inflation={100 * i['diff']:+.1f} [{100 * i['lo']:+.1f}, {100 * i['hi']:+.1f}] p={i['p_value']:.4f}   "
            f"swap-random={100 * (a['accuracy_swap_and_average']['value'] - a['accuracy_random_order']['value']):+.1f} pp"
        )

    print("\n== 03 VERBOSITY (max severity) ==")
    for j in J:
        for c in d3["configs"]:
            e = d3["per_judge"][j][c]
            s = max(e)
            v = e[s]
            print(
                f"  {j:13} {c:7} ratio={v['mean_length_ratio']:.2f}  "
                f"delta={v['delta_points']['diff']:+.2f} [{v['delta_points']['lo']:+.2f}, {v['delta_points']['hi']:+.2f}]  "
                f"scored_higher={100 * v['share_scored_higher']['value']:.1f}%"
            )

    print("\n== 04 SELF-ENHANCEMENT ==")
    for j in d4["judges"]:
        e = d4["per_judge"][j]["self_enhancement_points"]
        if e:
            print(
                f"  {j:13} {e['diff']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]  p={e['p_value']:.4f}  sig={bool(e['significant'])}"
            )

    print("\n== 05 RUBRIC ABLATION ==")
    for j in J:
        print(f"  {j:13} " + "  ".join(f"{c}={pc(d5['accuracy'][j][c])}" for c in d5["configs"]))
    for c in d5["comparisons"]:
        if c["baseline"] == "vague" and c["candidate"] == "cot":
            dd = c["delta_accuracy"]
            print(
                f"       {c['judge']:13} cot-vague={100 * dd['diff']:+.1f} [{100 * dd['lo']:+.1f}, {100 * dd['hi']:+.1f}]  "
                f"holm_p={c['p_adjusted_holm']:.2e}  extra_tokens={c['extra_tokens']:,}  "
                f"pts/1k_tok={c['accuracy_points_per_1k_extra_tokens']:.4f}"
            )
    print(
        f"  multiplicity: {d5['multiplicity']['n_significant']}/{d5['multiplicity']['n_tests']} survive Holm"
    )

    print("\n== 06 SELF-CONSISTENCY ==")
    for j in J:
        a = d6["per_judge"][j]
        print(
            f"  {j:13} alpha={a['krippendorff_alpha']:.3f}  kappa={a['mean_pairwise_kappa']:.3f}  "
            f"two_run={100 * a['two_run_decision_disagreement']:.1f}%  any_of_5={pc(a['accept_reject_flip_rate'])}  "
            f"score_sd={a['mean_score_sd']:.2f}"
        )

    print("\n== 07 COST ==")
    for r in d7["rows"]:
        print(
            f"  {r['judge']:13} acc={pc(r['accuracy'])}  ${r['cost_per_1k_judgments_usd']:.4f}/1k  "
            f"{r['mean_latency_ms']:.0f} ms  ${r['cost_per_correct_judgment_usd'] * 1000:.4f}/1k-correct  tier={r['tier']}"
        )
    print(f"  pareto={d7['pareto_frontier']}")
    print(f"  headline: {d7['headline']['sentence']}")
    print(f"  cheapest_per_correct={d7['headline']['cheapest_per_correct_judgment']}")

    print("\n== 08 TRAJECTORY ==")
    for j in d8["judges"]:
        p = d8["per_judge"][j]
        print(
            f"  {j:13} overall={pc(p['overall_detection'])}  n={p['n_comparisons']}  "
            f"length_bias={p['trajectory_length_bias_points']['diff']:+.2f} "
            f"[{p['trajectory_length_bias_points']['lo']:+.2f}, {p['trajectory_length_bias_points']['hi']:+.2f}]"
        )
        for v in p["by_degradation"].values():
            print(
                f"       {v['label']:28} detect={pc(v['detection_accuracy'])}  miss={100 * v['miss_rate']:.1f}%"
            )

    print("\n== 09 DISTILL ==")
    a = d9["accuracy_against_ground_truth"]
    print(
        f"  teacher={d9['teacher']['judge']}/{d9['teacher']['config']}  labels={d9['teacher']['labels_generated']}"
    )
    print(
        f"  majority={100 * a['majority_class_baseline']:.1f}%  "
        f"teacher@rubric={pc(a['teacher_at_rubric_threshold'])}  "
        f"teacher@oracle={pc(a['teacher_at_oracle_threshold'])} (cut {a['teacher_at_oracle_threshold']['cut']:.1f})  "
        f"student={pc(a['student'])}"
    )
    g = a["student_minus_teacher_oracle_cut"]
    print(
        f"  gap={100 * g['diff']:+.1f} [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}] p={g['p_value']:.4f} sig={bool(g['significant'])}"
    )
    print(
        f"  recovery={100 * (a['teacher_at_oracle_threshold']['value'] - a['teacher_at_rubric_threshold']['value']):.0f} pts"
    )
    op = d9["operating_point"]
    print(
        f"  op: threshold={op['threshold']:.2f}  block_precision={100 * op['block_precision']:.1f}%  "
        f"false_block={100 * op['false_block_rate']:.1f}%  caught={100 * op['bad_output_caught']:.1f}%  "
        f"block_rate={100 * op['block_rate']:.1f}%  false_allow={100 * op['false_allow_rate']:.1f}%"
    )
    c = d9["calibration"]
    print(
        f"  ECE vs truth: {c['ece_vs_truth_before_recalibration']:.3f} -> {c['ece_vs_truth_after_recalibration']:.3f}  "
        f"(n_cal={c['n_calibration_items']})  ECE vs teacher={c['ece_vs_teacher_labels']:.3f}  brier={c['brier']:.3f}"
    )
    f = d9["fidelity_to_teacher"]
    print(f"  agreement={100 * f['agreement']:.1f}%  auc={f['auc_against_teacher_labels']:.3f}")
    for k, v in f["student_accuracy_by_degradation"].items():
        t = f["teacher_accuracy_by_degradation"].get(k, {})
        print(
            f"       {k:14} student={100 * v['value']:5.1f}% (n={v['n']:<4}) teacher={100 * t.get('value', float('nan')):5.1f}%"
        )
    print(f"  probe_allow_rate={100 * d9['verbosity_probe_holdout']['allow_rate']:.0f}%")
    e = d9["economics"]
    print(
        f"  student={e['student_mean_latency_ms']:.2f} ms  teacher={e['teacher_mean_latency_ms']:.0f} ms  "
        f"speedup={e['latency_speedup']:.0f}x"
    )
    ca = d9["cascade"]
    print(f"  judge_only={pc(ca['judge_only_accuracy'])}")
    for r in ca["rows"]:
        print(
            f"       escalate={100 * r['escalation_rate']:5.1f}%  acc={pc(r['combined_accuracy'])}  "
            f"${r['cost_per_1k_usd']:.4f}/1k  {r['mean_latency_ms']:.0f} ms"
        )
    print(f"  recommended: {ca['sentence']}")

    print("\n== 10 LATENCY ==")
    print("  serial: " + "  ".join(f"{k}={v:.2f}" for k, v in d10["serial"].items() if k != "n"))
    print(
        f"  budget={d10['budget_p99_ms']:.0f} ms  headroom={d10['headroom_ms']:.0f} ms  "
        f"max_concurrency={d10['max_concurrency_within_budget']}  cpus={d10['cpu_count']}  "
        f"speedup_serial={d10['speedup_vs_cheapest_judge_serial']:.0f}x  batched={d10['batched_per_item_ms']:.3f} ms/item"
    )
    for k, v in sorted(d10["concurrency_sweep"].items(), key=lambda kv: int(kv[0])):
        print(
            f"       c={k:<3} p99={v['p99']:7.2f} ms  {v['throughput_rps']:6.0f} rps  ok={v['within_budget']}"
        )


if __name__ == "__main__":
    main()
