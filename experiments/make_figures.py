#!/usr/bin/env python
"""Regenerate every figure from results/*.json.

Figures are built only from committed result files, never from a live run. That
keeps them reproducible by a reader with no API key, and it means a figure can
never quietly disagree with the JSON the README cites.

Any figure built from simulated provenance is watermarked. That is not decoration
-- a chart screenshotted out of context is exactly how a simulated prior becomes
someone's cited finding.

    python experiments/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import banner, rel

from judgeguard.store import RESULTS_DIR, load

FIGDIR = RESULTS_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

INK = "#16202e"
GRID = "#dfe4ea"
PALETTE = ["#1b6ca8", "#e08a1e", "#2a9d5c", "#c0392b", "#7d5ba6", "#4a4a4a"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "legend.frameon": False,
    }
)


def _watermark(fig, payload: dict) -> None:
    mode = payload.get("provenance", {}).get("mode", "unknown")
    if payload.get("real_measurement"):
        # The guardrail benchmark runs a real model on real hardware; stamping
        # it SIMULATED would be its own kind of dishonesty.
        fig.text(
            0.99,
            0.005,
            "guardrail latency measured on real hardware; quoted LLM judge latencies "
            f"come from the {mode} provider",
            fontsize=6.5,
            color="#8a8f98",
            ha="right",
        )
        return
    if mode == "simulated":
        fig.text(
            0.5,
            0.5,
            "SIMULATED",
            fontsize=54,
            color="#c0392b",
            alpha=0.10,
            ha="center",
            va="center",
            rotation=28,
            weight="bold",
            zorder=0,
        )
        fig.text(
            0.99,
            0.005,
            "provenance: simulated provider — not a measurement of a real model",
            fontsize=6.5,
            color="#8a8f98",
            ha="right",
        )
    else:
        fig.text(
            0.99,
            0.005,
            f"provenance: {mode} | git {payload.get('provenance', {}).get('git_sha', '?')}",
            fontsize=6.5,
            color="#8a8f98",
            ha="right",
        )


def _save(fig, name: str, payload: dict) -> Path:
    _watermark(fig, payload)
    p = FIGDIR / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    {rel(p)}")
    return p


def _band(ax, x, mid, lo, hi, color, label):
    ax.plot(x, mid, "-o", color=color, label=label, ms=3.5, lw=1.6)
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)


# ------------------------------------------------------------------ figure 01
def fig_discrimination() -> None:
    d = load("01_discrimination")
    degs = d["degradations"]
    sevs = d["severities"]
    fig, axes = plt.subplots(1, len(degs), figsize=(3.1 * len(degs), 3.3), sharey=True)
    for ax, deg in zip(axes, degs, strict=True):
        for c, judge in zip(PALETTE, d["judges"], strict=False):
            bd = d["per_judge"][judge]["by_degradation_severity"]
            mid = [bd[f"{deg}@{s:g}"]["value"] for s in sevs]
            lo = [bd[f"{deg}@{s:g}"]["lo"] for s in sevs]
            hi = [bd[f"{deg}@{s:g}"]["hi"] for s in sevs]
            _band(ax, sevs, mid, lo, hi, c, judge)
        ax.axhline(0.5, color="#b0b6bf", ls=":", lw=1)
        ax.set_title(deg.replace("_", " "))
        ax.set_xlabel("degradation severity")
        ax.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("discrimination accuracy")
    axes[0].text(
        0.02, 0.52, "chance", fontsize=7, color="#8a8f98", transform=axes[0].get_yaxis_transform()
    )
    axes[-1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(
        "Where each judge goes blind — accuracy vs degradation severity (95% BCa CI)",
        y=1.04,
        fontsize=12,
        weight="bold",
    )
    _save(fig, "fig_01_discrimination.png", d)


# ------------------------------------------------------------------ figure 02
def fig_position_bias() -> None:
    d = load("02_position_bias")
    judges = d["judges"]
    keys = [
        ("accuracy_reference_first", "reference shown first"),
        ("accuracy_random_order", "random order (1 call)"),
        ("accuracy_swap_and_average", "swap-and-average (2 calls)"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.26
    for i, (k, label) in enumerate(keys):
        xs = [j + (i - 1) * w for j in range(len(judges))]
        vals = [d["per_judge"][j][k]["value"] for j in judges]
        err = [
            [v - d["per_judge"][j][k]["lo"] for v, j in zip(vals, judges, strict=True)],
            [d["per_judge"][j][k]["hi"] - v for v, j in zip(vals, judges, strict=True)],
        ]
        ax1.bar(
            xs,
            vals,
            w,
            label=label,
            color=PALETTE[i],
            yerr=err,
            capsize=2,
            error_kw={"lw": 0.8, "ecolor": "#6b7280"},
        )
    ax1.set_xticks(range(len(judges)))
    ax1.set_xticklabels(judges, fontsize=8)
    ax1.set_ylabel("pairwise accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.set_title("Fixed-order evaluation reports a number it did not earn")

    inc = [d["per_judge"][j]["inconsistency_rate"]["value"] for j in judges]
    lo = [
        v - d["per_judge"][j]["inconsistency_rate"]["lo"] for v, j in zip(inc, judges, strict=True)
    ]
    hi = [
        d["per_judge"][j]["inconsistency_rate"]["hi"] - v for v, j in zip(inc, judges, strict=True)
    ]
    ax2.barh(
        judges,
        inc,
        color=PALETTE[3],
        xerr=[lo, hi],
        capsize=2,
        error_kw={"lw": 0.8, "ecolor": "#6b7280"},
    )
    ax2.set_xlabel("verdict changes when the two options are swapped")
    ax2.set_title("Position-bias disagreement rate")
    ax2.invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig_02_position_bias.png", d)


# ------------------------------------------------------------------ figure 03
def fig_verbosity() -> None:
    d = load("03_verbosity_bias")
    judges, configs = d["judges"], d["configs"]
    fig, axes = plt.subplots(1, len(configs), figsize=(3.4 * len(configs), 3.3), sharey=True)
    for ax, config in zip(axes, configs, strict=True):
        for c, judge in zip(PALETTE, judges, strict=False):
            entry = d["per_judge"][judge][config]
            x = [entry[s]["mean_length_ratio"] for s in sorted(entry)]
            mid = [entry[s]["delta_points"]["diff"] for s in sorted(entry)]
            lo = [entry[s]["delta_points"]["lo"] for s in sorted(entry)]
            hi = [entry[s]["delta_points"]["hi"] for s in sorted(entry)]
            _band(ax, x, mid, lo, hi, c, judge)
        ax.axhline(0, color=INK, lw=1.1)
        ax.set_title(f"{config} prompt")
        ax.set_xlabel("length ratio vs reference")
    axes[0].set_ylabel("score change from padding alone (points)")
    axes[0].text(
        0.02,
        0.03,
        "correct answer: 0",
        fontsize=7,
        color="#8a8f98",
        transform=axes[0].get_yaxis_transform(),
    )
    axes[-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Verbosity bias — padding a CORRECT answer with content-free filler",
        y=1.04,
        fontsize=12,
        weight="bold",
    )
    _save(fig, "fig_03_verbosity_bias.png", d)


# ------------------------------------------------------------------ figure 04
def fig_self_enhancement() -> None:
    d = load("04_self_enhancement")
    judges = [j for j in d["judges"] if d["per_judge"][j].get("self_enhancement_points")]
    vals = [d["per_judge"][j]["self_enhancement_points"]["diff"] for j in judges]
    lo = [
        v - d["per_judge"][j]["self_enhancement_points"]["lo"]
        for v, j in zip(vals, judges, strict=True)
    ]
    hi = [
        d["per_judge"][j]["self_enhancement_points"]["hi"] - v
        for v, j in zip(vals, judges, strict=True)
    ]
    colors = [
        PALETTE[2] if d["per_judge"][j]["self_enhancement_points"]["significant"] else "#9aa3ad"
        for j in judges
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(
        judges,
        vals,
        color=colors,
        xerr=[lo, hi],
        capsize=3,
        error_kw={"lw": 0.9, "ecolor": "#4a4a4a"},
    )
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlabel("extra points given to own-family output (leave-one-out vs peer judges)")
    ax.set_title("Self-enhancement bias\ngrey = not distinguishable from noise")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig_04_self_enhancement.png", d)


# ------------------------------------------------------------------ figure 05
def fig_rubric() -> None:
    d = load("05_rubric_ablation")
    judges, configs = d["judges"], d["configs"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.26
    for i, config in enumerate(configs):
        xs = [j + (i - 1) * w for j in range(len(judges))]
        vals = [d["accuracy"][j][config]["value"] for j in judges]
        err = [
            [v - d["accuracy"][j][config]["lo"] for v, j in zip(vals, judges, strict=True)],
            [d["accuracy"][j][config]["hi"] - v for v, j in zip(vals, judges, strict=True)],
        ]
        ax.bar(
            xs,
            vals,
            w,
            label=config,
            color=PALETTE[i],
            yerr=err,
            capsize=2,
            error_kw={"lw": 0.8, "ecolor": "#6b7280"},
        )
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, fontsize=8)
    ax.set_ylabel("discrimination accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    n_sig = d["multiplicity"]["n_significant"]
    ax.set_title(
        f"Prompt structure vs discrimination — {n_sig}/{d['multiplicity']['n_tests']} "
        f"differences survive Holm-Bonferroni"
    )
    fig.tight_layout()
    _save(fig, "fig_05_rubric_ablation.png", d)


# ------------------------------------------------------------------ figure 06
def fig_self_consistency() -> None:
    d = load("06_self_consistency")
    judges = d["judges"]
    alpha = [d["per_judge"][j]["krippendorff_alpha"] for j in judges]
    flip = [d["per_judge"][j]["accept_reject_flip_rate"]["value"] for j in judges]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))
    ax1.barh(judges, alpha, color=PALETTE[0])
    ax1.axvline(0.667, color=PALETTE[3], ls="--", lw=1.2)
    ax1.text(0.667, -0.45, " alpha = 0.667\n conventional floor", fontsize=7, color=PALETTE[3])
    ax1.set_xlabel("Krippendorff's alpha across 5 reruns")
    ax1.set_title("Agreement of a judge with itself")
    ax1.invert_yaxis()
    ax2.barh(judges, flip, color=PALETTE[3])
    ax2.set_xlabel("share of items where accept/reject flipped across reruns")
    ax2.set_title("The noise floor for every other comparison")
    ax2.invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig_06_self_consistency.png", d)


# ------------------------------------------------------------------ figure 07
def fig_cost_accuracy() -> None:
    d = load("07_cost_accuracy")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, r in enumerate(d["rows"]):
        x = r["cost_per_1k_judgments_usd"]
        y = r["accuracy"]["value"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - r["accuracy"]["lo"]], [r["accuracy"]["hi"] - y]],
            fmt="o",
            color=PALETTE[i % len(PALETTE)],
            ms=9,
            capsize=3,
            ecolor="#6b7280",
            elinewidth=0.9,
        )
        ax.annotate(
            f"{r['judge']}\n{r['mean_latency_ms']:.0f} ms",
            (x, y),
            textcoords="offset points",
            xytext=(9, -4),
            fontsize=7.5,
        )
    pf = [r for r in d["rows"] if r["judge"] in d["pareto_frontier"]]
    pf.sort(key=lambda r: r["cost_per_1k_judgments_usd"])
    ax.plot(
        [r["cost_per_1k_judgments_usd"] for r in pf],
        [r["accuracy"]["value"] for r in pf],
        "--",
        color="#9aa3ad",
        lw=1.1,
        zorder=0,
    )
    ax.set_xlabel("USD per 1,000 judgments (published list prices)")
    ax.set_ylabel("discrimination accuracy")
    ax.set_title("Accuracy is not the only axis")
    fig.tight_layout()
    _save(fig, "fig_07_cost_accuracy.png", d)


# ------------------------------------------------------------------ figure 08
def fig_trajectory() -> None:
    d = load("08_trajectory_blindness")
    judges = d["judges"]
    degs = sorted(d["per_judge"][judges[0]]["by_degradation"])
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    w = 0.8 / len(judges)
    for i, judge in enumerate(judges):
        vals = [d["per_judge"][judge]["by_degradation"][g]["miss_rate"] for g in degs]
        xs = [k + (i - (len(judges) - 1) / 2) * w for k in range(len(degs))]
        ax.bar(xs, vals, w, label=judge, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(range(len(degs)))
    ax.set_xticklabels(
        [d["per_judge"][judges[0]]["by_degradation"][g]["label"].replace(" ", "\n") for g in degs],
        fontsize=8,
    )
    ax.set_ylabel("miss rate (failed to prefer the correct trajectory)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncols=2)
    ax.set_title("Agent-trajectory blindness — failure classes output-only evaluation cannot see")
    fig.tight_layout()
    _save(fig, "fig_08_trajectory_blindness.png", d)


# ------------------------------------------------------------------ figure 09
def fig_reliability() -> None:
    d = load("09_distill")
    cal = d["calibration"]
    fid = d["fidelity_to_teacher"]
    curve = [b for b in cal["reliability_curve"] if b["n"] > 0]
    before = [b for b in cal.get("reliability_curve_before", []) if b["n"] > 0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 3.7))

    # -- calibration
    ax1.plot([0, 1], [0, 1], "--", color="#9aa3ad", lw=1, label="perfect calibration")
    if before:
        ax1.plot(
            [b["confidence"] for b in before],
            [b["accuracy"] for b in before],
            "-o",
            color=PALETTE[3],
            ms=4,
            lw=1.3,
            alpha=0.85,
            label=f"distilled only (ECE {cal['ece_vs_truth_before_recalibration']:.3f})",
        )
    ax1.plot(
        [b["confidence"] for b in curve],
        [b["accuracy"] for b in curve],
        "-o",
        color=PALETTE[2],
        ms=5,
        label=f"+ free recalibration (ECE {cal['ece_vs_truth_after_recalibration']:.3f})",
    )
    ax1.set_xlabel("predicted probability the answer is acceptable")
    ax1.set_ylabel("observed frequency")
    ax1.set_title("Distillation inherits calibration error")
    ax1.legend(fontsize=7, loc="upper left")

    # -- where the student inherits, and where it cannot follow
    sd = fid.get("student_accuracy_by_degradation", {})
    td = fid.get("teacher_accuracy_by_degradation", {})
    keys = [k for k in sd if k != "reference"]
    keys.sort(key=lambda k: sd[k]["value"])
    w = 0.38
    xs = range(len(keys))
    ax2.bar(
        [x - w / 2 for x in xs],
        [td.get(k, {}).get("value", 0) for k in keys],
        w,
        label="teacher (LLM judge)",
        color=PALETTE[1],
    )
    ax2.bar(
        [x + w / 2 for x in xs],
        [sd[k]["value"] for k in keys],
        w,
        label="student (3 ms)",
        color=PALETTE[0],
    )
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("detection accuracy")
    ax2.legend(fontsize=7.5)
    ax2.set_title("The student's blind spots are structural")

    # -- cascade
    rows = d.get("cascade", {}).get("rows", [])
    jo = d.get("cascade", {}).get("judge_only_accuracy", {})
    if rows:
        esc = [100 * r["escalation_rate"] for r in rows]
        acc = [r["combined_accuracy"]["value"] for r in rows]
        lo = [r["combined_accuracy"]["lo"] for r in rows]
        hi = [r["combined_accuracy"]["hi"] for r in rows]
        order = sorted(range(len(esc)), key=lambda i: esc[i])
        esc = [esc[i] for i in order]
        acc = [acc[i] for i in order]
        lo = [lo[i] for i in order]
        hi = [hi[i] for i in order]
        ax3.plot(esc, acc, "-o", color=PALETTE[0], ms=5)
        ax3.fill_between(esc, lo, hi, color=PALETTE[0], alpha=0.14, lw=0)
        if jo:
            ax3.axhline(jo["value"], color=PALETTE[3], ls="--", lw=1.3)
            ax3.text(
                2,
                jo["value"] + 0.008,
                f"judge on everything: {100 * jo['value']:.1f}%",
                fontsize=7.5,
                color=PALETTE[3],
            )
        rec = d["cascade"].get("recommended")
        if rec:
            ax3.scatter(
                [100 * rec["escalation_rate"]],
                [rec["combined_accuracy"]["value"]],
                s=110,
                facecolor="none",
                edgecolor=PALETTE[2],
                lw=2,
                zorder=5,
            )
        ax3.set_xlabel("% of traffic escalated to the LLM judge")
        ax3.set_ylabel("combined accuracy")
        ax3.set_title("Cascade: the student absorbs what it can see")
    fig.tight_layout()
    _save(fig, "fig_09_reliability.png", d)


# ------------------------------------------------------------------ figure 10
def fig_latency() -> None:
    d = load("10_latency_bench")
    budget = d["budget_p99_ms"]
    sweep = d.get("concurrency_sweep") or {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6))

    labels = ["p50", "p90", "p95", "p99"]
    serial = [d["serial"][k] for k in labels]
    ax1.bar(labels, serial, 0.55, color=PALETTE[0])
    for i, v in enumerate(serial):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(budget, color=PALETTE[3], ls="--", lw=1.4)
    ax1.text(
        3.4, budget * 1.02, f"budget {budget:.0f} ms", color=PALETTE[3], fontsize=8, ha="right"
    )
    ax1.set_ylabel("guardrail latency (ms)")
    ax1.set_ylim(0, max(budget * 1.25, max(serial) * 1.3))
    ax1.set_title("Per-request, single stream")

    if sweep:
        xs = sorted(int(k) for k in sweep)
        p99 = [sweep[str(x)]["p99"] for x in xs]
        rps = [sweep[str(x)]["throughput_rps"] for x in xs]
        ok = [sweep[str(x)]["within_budget"] for x in xs]
        ax2.plot(xs, p99, "-o", color=PALETTE[0], ms=5, label="p99 latency")
        ax2.scatter(
            [x for x, o in zip(xs, ok, strict=True) if not o],
            [v for v, o in zip(p99, ok, strict=True) if not o],
            color=PALETTE[3],
            zorder=5,
            s=70,
            label="over budget",
        )
        ax2.axhline(budget, color=PALETTE[3], ls="--", lw=1.4)
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([str(x) for x in xs])
        ax2.set_xlabel(f"concurrent requests ({d.get('cpu_count', '?')} cores)")
        ax2.set_ylabel("p99 latency (ms)")
        twin = ax2.twinx()
        twin.plot(xs, rps, ":s", color=PALETTE[2], ms=4, label="throughput")
        twin.set_ylabel("requests / second", color=PALETTE[2])
        twin.grid(False)
        ax2.set_title(f"Budget holds to concurrency {d.get('max_concurrency_within_budget', '?')}")
        ax2.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_10_latency.png", d)


# --------------------------------------------------------------- headline
def fig_headline() -> None:
    tr = load("08_trajectory_blindness")
    disc = load("01_discrimination")
    judges = tr["judges"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))

    degs = disc["degradations"]
    worst = {}
    for judge in judges:
        bd = disc["per_judge"][judge]["by_degradation"]
        worst[judge] = [bd[g]["value"] for g in degs]
    for i, judge in enumerate(judges):
        ax1.plot(
            range(len(degs)), worst[judge], "-o", color=PALETTE[i % len(PALETTE)], label=judge, ms=4
        )
    ax1.set_xticks(range(len(degs)))
    ax1.set_xticklabels([g.replace("_", "\n") for g in degs], fontsize=8)
    ax1.set_ylim(0.4, 1.03)
    ax1.axhline(0.5, color="#b0b6bf", ls=":", lw=1)
    ax1.set_ylabel("discrimination accuracy")
    ax1.set_title("Text output: judges are weakest on wrong numbers")
    ax1.legend(fontsize=7.5, loc="lower left")

    tdegs = sorted(tr["per_judge"][judges[0]]["by_degradation"])
    w = 0.8 / len(judges)
    for i, judge in enumerate(judges):
        vals = [tr["per_judge"][judge]["by_degradation"][g]["miss_rate"] for g in tdegs]
        xs = [k + (i - (len(judges) - 1) / 2) * w for k in range(len(tdegs))]
        ax2.bar(xs, vals, w, color=PALETTE[i % len(PALETTE)], label=judge)
    ax2.set_xticks(range(len(tdegs)))
    ax2.set_xticklabels(
        [
            tr["per_judge"][judges[0]]["by_degradation"][g]["label"].replace(" ", "\n")
            for g in tdegs
        ],
        fontsize=8,
    )
    ax2.set_ylabel("miss rate")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Agent trajectories: and blindest to malformed tool arguments")
    fig.suptitle(
        "JudgeGuard — measured judge failure modes, 95% BCa intervals throughout",
        y=1.05,
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_00_headline.png", tr)


FIGURES = [
    fig_headline,
    fig_discrimination,
    fig_position_bias,
    fig_verbosity,
    fig_self_enhancement,
    fig_rubric,
    fig_self_consistency,
    fig_cost_accuracy,
    fig_trajectory,
    fig_reliability,
    fig_latency,
]


def main() -> None:
    banner("figures")
    made = 0
    for fn in FIGURES:
        try:
            fn()
            made += 1
        except FileNotFoundError as exc:
            print(f"    skipped {fn.__name__}: {exc}")
    print(f"    {made}/{len(FIGURES)} figures written to {rel(FIGDIR)}")


if __name__ == "__main__":
    main()
