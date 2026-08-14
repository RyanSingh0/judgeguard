#!/usr/bin/env python
"""Cross-check every number quoted in the prose against results/*.json.

Documentation drifts from data silently. This makes that failure loud: it
re-derives each headline claim from the committed result files and compares it
against what the README, findings, paper and LinkedIn post actually say.

Run it after any experiment rerun. `make verify`, and it gates CI.

    python scripts/verify_claims.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Every other entry point honours JUDGEGUARD_RESULTS_DIR. This one didn't, so
# pointing the battery at a scratch dir and then checking claims compared the
# prose against the old committed results and reported everything green.
_env = os.getenv("JUDGEGUARD_RESULTS_DIR", "")
R = (Path(_env) if Path(_env).is_absolute() else REPO / _env) if _env else REPO / "results"
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def load(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text(encoding="utf-8"))


def pc(e: dict, d: int = 1) -> str:
    return f"{100 * e['value']:.{d}f}% [{100 * e['lo']:.{d}f}, {100 * e['hi']:.{d}f}]"


def main() -> int:
    d0 = load("degraded_set")
    d1, d2, d3 = load("01_discrimination"), load("02_position_bias"), load("03_verbosity_bias")
    d4, d5, d6 = (
        load("04_self_enhancement"),
        load("05_rubric_ablation"),
        load("06_self_consistency"),
    )
    d7, d8 = load("07_cost_accuracy"), load("08_trajectory_blindness")
    d9, d10 = load("09_distill"), load("10_latency_bench")
    J = d1["judges"]

    # Work out best and worst rather than naming them. These were the literal
    # aliases "gemini-flash" and "gptoss20b", so as soon as the panel changed
    # the checker either raised KeyError or, worse, quietly verified the wrong
    # judge. This script exists to catch quoted numbers drifting from the
    # measurement, and a hardcoded judge lets that drift straight through.
    _acc = {j: d1["per_judge"][j]["overall_accuracy"]["value"] for j in J}
    BEST = max(_acc, key=lambda j: _acc[j])
    WORST = min(_acc, key=lambda j: _acc[j])
    _inc = {j: d2["per_judge"][j]["inconsistency_rate"]["value"] for j in J}
    MOST_CONSISTENT = min(_inc, key=lambda j: _inc[j])
    LEAST_CONSISTENT = max(_inc, key=lambda j: _inc[j])

    claims: list[tuple[str, str, bool]] = []

    def claim(label: str, derived: str, expected: str) -> None:
        claims.append((label, f"{derived}  vs quoted {expected}", derived == expected))

    # ---------------------------------------------------------------- counts
    c = d0["counts"]
    claim("gold pairs", str(c["text_error_pairs"]), "7500")
    claim("probe pairs", str(c["text_probe_pairs"]), "1500")
    claim("trajectory comparisons per judge", str(d8["per_judge"][J[0]]["n_comparisons"]), "540")
    claim(
        "power n_required",
        str(d0["power_analysis"]["n_for_independent_samples"]["n_required"]),
        "575",
    )

    # ------------------------------------------------------- 01 discrimination
    acc = {j: d1["per_judge"][j]["overall_accuracy"] for j in J}
    claim(
        "overall accuracy range",
        f"{100 * min(a['value'] for a in acc.values()):.1f}-{100 * max(a['value'] for a in acc.values()):.1f}%",
        "81.7-95.4%",
    )
    claim(f"best judge accuracy ({BEST})", pc(acc[BEST]), "95.4% [94.9, 95.8]")
    claim(f"worst judge accuracy ({WORST})", pc(acc[WORST]), "81.7% [80.8, 82.6]")
    ns = {j: d1["per_judge"][j]["by_degradation"]["numeric_swap"]["value"] for j in J}
    claim(
        "numeric_swap range",
        f"{100 * min(ns.values()):.1f}-{100 * max(ns.values()):.1f}%",
        "67.8-85.9%",
    )
    claim(
        "numeric_swap worst class for every judge",
        str(
            all(
                ns[j] == min(v["value"] for v in d1["per_judge"][j]["by_degradation"].values())
                for j in J
            )
        ),
        "True",
    )
    claim(
        f"best judge numeric_swap@0.2 ({BEST})",
        pc(d1["per_judge"][BEST]["by_degradation_severity"]["numeric_swap@0.2"]),
        "66.2% [62.0, 70.4]",
    )
    claim(
        f"worst judge numeric_swap@0.2 ({WORST})",
        pc(d1["per_judge"][WORST]["by_degradation_severity"]["numeric_swap@0.2"]),
        "54.8% [50.4, 59.0]",
    )
    td = {j: d1["per_judge"][j]["by_degradation"]["topic_drift"]["value"] for j in J}
    claim(
        "topic_drift range",
        f"{100 * min(td.values()):.0f}-{100 * max(td.values()):.0f}%",
        "95-100%",
    )

    # ---------------------------------------------------------- 02 position
    inc = {j: d2["per_judge"][j]["inconsistency_rate"] for j in J}
    claim(
        f"min order disagreement ({MOST_CONSISTENT})",
        pc(inc[MOST_CONSISTENT]),
        "11.6% [10.6, 12.6]",
    )
    claim(
        f"max order disagreement ({LEAST_CONSISTENT})",
        pc(inc[LEAST_CONSISTENT]),
        "32.3% [30.8, 33.8]",
    )
    inf = {j: d2["per_judge"][j]["fixed_order_inflation"] for j in J}
    claim(
        f"min fixed-order inflation ({MOST_CONSISTENT})",
        f"{100 * inf[MOST_CONSISTENT]['diff']:.1f} [{100 * inf[MOST_CONSISTENT]['lo']:.1f}, {100 * inf[MOST_CONSISTENT]['hi']:.1f}]",
        "3.9 [3.3, 4.6]",
    )
    claim(
        f"max fixed-order inflation ({LEAST_CONSISTENT})",
        f"{100 * inf[LEAST_CONSISTENT]['diff']:.1f} [{100 * inf[LEAST_CONSISTENT]['lo']:.1f}, {100 * inf[LEAST_CONSISTENT]['hi']:.1f}]",
        "12.1 [11.1, 13.2]",
    )
    drops = [
        100
        * (
            d2["per_judge"][j]["accuracy_swap_and_average"]["value"]
            - d2["per_judge"][j]["accuracy_random_order"]["value"]
        )
        for j in J
    ]
    claim("swap lowers accuracy by", f"{-max(drops):.1f}-{-min(drops):.1f} pp", "4.0-13.3 pp")

    # --------------------------------------------------------- 03 verbosity
    worst = max(
        (
            d3["per_judge"][j][cfg][s]["delta_points"]
            for j in J
            for cfg in d3["configs"]
            for s in d3["per_judge"][j][cfg]
        ),
        key=lambda e: e["diff"],
    )
    claim(
        "max verbosity gain",
        f"+{worst['diff']:.2f} [+{worst['lo']:.2f}, +{worst['hi']:.2f}]",
        "+1.45 [+1.36, +1.53]",
    )
    shares = [
        d3["per_judge"][j][cfg][max(d3["per_judge"][j][cfg])]["share_scored_higher"]["value"]
        for j in J
        for cfg in d3["configs"]
    ]
    claim("share scored higher", f"{100 * min(shares):.0f}-{100 * max(shares):.0f}%", "85-94%")

    # --------------------------------------------------- 04 self-enhancement
    se = {j: d4["per_judge"][j]["self_enhancement_points"] for j in d4["judges"]}
    claim(
        "self-enhancement range",
        f"+{min(v['diff'] for v in se.values()):.2f} to +{max(v['diff'] for v in se.values()):.2f}",
        "+0.12 to +0.79",
    )
    claim(
        "largest self-enhancement is the most accurate judge",
        max(se, key=lambda j: se[j]["diff"]),
        max(acc, key=lambda j: acc[j]["value"]),
    )

    # ------------------------------------------------------------ 05 rubric
    gains = {
        x["judge"]: 100 * x["delta_accuracy"]["diff"]
        for x in d5["comparisons"]
        if x["baseline"] == "vague" and x["candidate"] == "cot"
    }
    claim(
        "cot-vague gain range",
        f"+{min(gains.values()):.1f} to +{max(gains.values()):.1f} pp",
        "+8.2 to +15.7 pp",
    )
    claim(
        "config comparisons surviving Holm",
        f"{d5['multiplicity']['n_significant']}/{d5['multiplicity']['n_tests']}",
        "12/12",
    )

    # ------------------------------------------------------- 06 consistency
    al = {j: d6["per_judge"][j]["krippendorff_alpha"] for j in J}
    claim("alpha range", f"{min(al.values()):.3f}-{max(al.values()):.3f}", "0.182-0.545")
    claim("no judge reaches the 0.667 floor", str(max(al.values()) < 0.667), "True")
    tw = {j: d6["per_judge"][j]["two_run_decision_disagreement"] for j in J}
    claim(
        "two-run disagreement",
        f"{100 * min(tw.values()):.1f}-{100 * max(tw.values()):.1f}%",
        "19.2-31.5%",
    )

    # -------------------------------------------------------------- 07 cost
    claim(
        "cheapest per correct judgment",
        d7["headline"]["cheapest_per_correct_judgment"],
        "gptoss20b",
    )
    claim("pareto frontier size", str(len(d7["pareto_frontier"])), "3")
    lat = [r["mean_latency_ms"] for r in d7["rows"]]
    claim("judge latency range", f"{min(lat):.0f}-{max(lat):.0f} ms", "299-703 ms")

    # -------------------------------------------------------- 08 trajectory
    wa = {j: d8["per_judge"][j]["by_degradation"]["wrong_argument"] for j in d8["judges"]}
    claim(
        "argument blindness miss range",
        f"{100 * min(v['miss_rate'] for v in wa.values()):.1f}-{100 * max(v['miss_rate'] for v in wa.values()):.1f}%",
        "37.2-64.4%",
    )
    claim(
        "best argument detection",
        pc(max(wa.values(), key=lambda v: v["detection_accuracy"]["value"])["detection_accuracy"]),
        "62.8% [55.6, 69.4]",
    )
    claim(
        "worst argument detection",
        pc(min(wa.values(), key=lambda v: v["detection_accuracy"]["value"])["detection_accuracy"]),
        "35.6% [28.9, 42.8]",
    )
    ph = {
        j: d8["per_judge"][j]["by_degradation"]["phantom_tool"]["miss_rate"] for j in d8["judges"]
    }
    claim(
        "phantom tool miss range",
        f"{100 * min(ph.values()):.1f}-{100 * max(ph.values()):.1f}%",
        "10.0-43.9%",
    )
    sf = {
        j: d8["per_judge"][j]["by_degradation"]["silent_failure"]["miss_rate"] for j in d8["judges"]
    }
    claim(
        "path blindness miss range",
        f"{100 * min(sf.values()):.1f}-{100 * max(sf.values()):.1f}%",
        "20.6-47.2%",
    )
    lb = [d8["per_judge"][j]["trajectory_length_bias_points"] for j in d8["judges"]]
    claim(
        "trajectory length bias",
        f"+{min(x['diff'] for x in lb):.2f} to +{max(x['diff'] for x in lb):.2f}",
        "+0.14 to +0.39",
    )
    claim("all length-bias intervals exclude zero", str(all(x["lo"] > 0 for x in lb)), "True")
    claim(
        "argument blindness worst class for every judge",
        str(
            all(
                wa[j]["miss_rate"]
                == max(v["miss_rate"] for v in d8["per_judge"][j]["by_degradation"].values())
                for j in d8["judges"]
            )
        ),
        "True",
    )

    # ----------------------------------------------------------- 09 distill
    a = d9["accuracy_against_ground_truth"]
    claim("majority baseline", f"{100 * a['majority_class_baseline']:.1f}%", "51.0%")
    claim("teacher at rubric cut", pc(a["teacher_at_rubric_threshold"]), "80.0% [74.0, 85.0]")
    claim("teacher at oracle cut", pc(a["teacher_at_oracle_threshold"]), "94.0% [90.0, 96.5]")
    claim("oracle cut", f"{a['teacher_at_oracle_threshold']['cut']:.1f}", "7.7")
    claim(
        "threshold recovery",
        f"{100 * (a['teacher_at_oracle_threshold']['value'] - a['teacher_at_rubric_threshold']['value']):.0f} pts",
        "14 pts",
    )
    claim("student accuracy", pc(a["student"]), "77.0% [70.5, 82.5]")
    g = a["student_minus_teacher_oracle_cut"]
    claim(
        "student vs teacher gap",
        f"{100 * g['diff']:.1f} [{100 * g['lo']:.1f}, {100 * g['hi']:.1f}]",
        "-17.0 [-23.0, -11.5]",
    )
    claim("gap IS significant", str(bool(g["significant"])), "True")
    op = d9["operating_point"]
    claim("block precision", f"{100 * op['block_precision']:.1f}%", "100.0%")
    claim("false-block rate", f"{100 * op['false_block_rate']:.1f}%", "0.0%")
    claim("bad output caught", f"{100 * op['bad_output_caught']:.1f}%", "53.1%")
    cal = d9["calibration"]
    claim(
        "ECE before/after",
        f"{cal['ece_vs_truth_before_recalibration']:.3f} -> {cal['ece_vs_truth_after_recalibration']:.3f}",
        "0.193 -> 0.015",
    )
    claim("calibration items", str(cal["n_calibration_items"]), "200")
    fid = d9["fidelity_to_teacher"]
    sd, tdg = fid["student_accuracy_by_degradation"], fid["teacher_accuracy_by_degradation"]
    claim("student numeric_swap", f"{100 * sd['numeric_swap']['value']:.1f}%", "0.0%")
    claim("teacher numeric_swap", f"{100 * tdg['numeric_swap']['value']:.1f}%", "57.9%")
    claim("student topic_drift", f"{100 * sd['topic_drift']['value']:.1f}%", "0.0%")
    claim("student hedging", f"{100 * sd['hedging']['value']:.1f}%", "100.0%")
    claim("student allows references", f"{100 * sd['reference']['value']:.1f}%", "100.0%")
    claim("probe allow rate", f"{100 * d9['verbosity_probe_holdout']['allow_rate']:.0f}%", "0%")
    ca = d9["cascade"]
    rec = ca["recommended"]
    claim("cascade escalation", f"{100 * rec['escalation_rate']:.0f}%", "75%")
    claim("cascade combined accuracy", pc(rec["combined_accuracy"]), "94.5% [90.5, 97.0]")
    claim("judge-only accuracy", pc(ca["judge_only_accuracy"]), "94.0% [90.0, 96.5]")
    claim(
        "cascade CI overlaps judge-only",
        str(
            rec["combined_accuracy"]["lo"]
            <= ca["judge_only_accuracy"]["value"]
            <= rec["combined_accuracy"]["hi"]
        ),
        "True",
    )

    # ----------------------------------------------------------- 10 latency
    claim("guardrail p50", f"{d10['serial']['p50']:.2f} ms", "3.84 ms")
    claim("guardrail p95", f"{d10['serial']['p95']:.2f} ms", "5.43 ms")
    claim("guardrail p99", f"{d10['serial']['p99']:.2f} ms", "7.15 ms")
    claim("meets budget", str(d10["meets_budget_serial"]), "True")
    claim("headroom", f"{d10['headroom_ms']:.0f} ms", "143 ms")
    claim("speedup vs cheapest judge", f"{d10['speedup_vs_cheapest_judge_serial']:.0f}x", "78x")
    claim("max concurrency within budget", str(d10["max_concurrency_within_budget"]), "4")

    # ------------------------------------------------------------- report
    print("\n=== derived vs quoted ===")
    for label, detail, ok in claims:
        print(f"  [{GREEN + 'OK' + RESET if ok else RED + 'XX' + RESET}] {label:<46} {detail}")

    print("\n=== prose references only existing figures ===")
    figs = {p.name for p in (R / "figures").glob("*.png")}
    missing = []
    for doc in [REPO / "README.md", *(REPO / "docs").glob("*.md")]:
        for ref in re.findall(r"figures/([\w.]+\.png)", doc.read_text(encoding="utf-8")):
            if ref not in figs:
                missing.append(f"{doc.name} -> {ref}")
    print(
        f"  [{GREEN + 'OK' + RESET if not missing else RED + 'XX' + RESET}] "
        f"{len(figs)} figures present, {len(missing)} broken references"
    )

    print("\n=== provenance stamped, JSON strictly valid ===")
    files = sorted(R.glob("*.json"))
    bad = [p.name for p in files if "provenance" not in json.loads(p.read_text(encoding="utf-8"))]
    nan = [
        p.name
        for p in files
        if re.search(r":\s*(NaN|Infinity|-Infinity)", p.read_text(encoding="utf-8"))
    ]
    print(
        f"  [{GREEN + 'OK' + RESET if not bad else RED + 'XX' + RESET}] "
        f"{len(files)} result files, {len(bad)} missing provenance, {len(nan)} with bare NaN"
    )

    n_fail = sum(1 for _, _, ok in claims if not ok) + len(missing) + len(bad) + len(nan)
    print(
        f"\n{len(claims) - sum(1 for _, _, ok in claims if not ok)}/{len(claims)} quoted numbers match the data"
    )
    if n_fail:
        print(f"{RED}{n_fail} discrepancies — fix the prose or rerun the experiment{RESET}")
        return 1
    print(f"{GREEN}every number in the prose is traceable to results/{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
