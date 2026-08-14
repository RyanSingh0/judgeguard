#!/usr/bin/env python
"""Assemble the static Space in site/.

Hugging Face's free tier serves static files only, so the demo has no backend.
That turns out to suit this project: the distilled guardrail is a sparse dot
product plus a step function, so it runs client-side at the same speed and with
the same numbers (see site/parity.test.js). The LLM `/evaluate` path is dropped
rather than faked -- a static page cannot hold an API key, and pretending
otherwise would be the kind of thing this repo exists to complain about.

    python scripts/build_site.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from judgeguard.data.load import load_items
from judgeguard.degrade.text import hedging, numeric_swap, omission, verbosity
from judgeguard.store import json_safe

SITE = REPO / "site"
RESULTS = REPO / "results"

FIGURES = [
    ("fig_00_headline.png", "Headline: where judges fail"),
    ("fig_01_discrimination.png", "Discrimination vs degradation severity"),
    ("fig_02_position_bias.png", "Position bias and the swap protocol"),
    ("fig_03_verbosity_bias.png", "Verbosity bias, in points"),
    ("fig_08_trajectory_blindness.png", "Agent-trajectory blindness"),
    ("fig_05_rubric_ablation.png", "Does prompt structure buy accuracy?"),
    ("fig_04_self_enhancement.png", "Self-enhancement bias"),
    ("fig_06_self_consistency.png", "Judges vs themselves"),
    ("fig_07_cost_accuracy.png", "Accuracy is not the only axis"),
    ("fig_09_reliability.png", "Student calibration, blind spots, cascade"),
    ("fig_10_latency.png", "Guardrail latency vs budget"),
]


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def build_summary() -> dict:
    d1, d2 = load("01_discrimination"), load("02_position_bias")
    d8, d9, d10 = load("08_trajectory_blindness"), load("09_distill"), load("10_latency_bench")

    wa = {j: d8["per_judge"][j]["by_degradation"]["wrong_argument"] for j in d8["judges"]}
    worst = max(wa, key=lambda j: wa[j]["miss_rate"])
    best = min(wa, key=lambda j: wa[j]["miss_rate"])
    pos = max(d2["judges"], key=lambda j: d2["per_judge"][j]["inconsistency_rate"]["value"])
    a = d9["accuracy_against_ground_truth"]
    cal = d9["calibration"]
    op = d9["operating_point"]

    findings = [
        {
            "id": "argument_blindness",
            "headline": (
                f"Judges missed malformed tool arguments in "
                f"{100 * wa[best]['miss_rate']:.1f}%–{100 * wa[worst]['miss_rate']:.1f}% of agent "
                f"trajectories (best detection {100 * wa[best]['detection_accuracy']['value']:.1f}%, "
                f"95% CI [{100 * wa[best]['detection_accuracy']['lo']:.1f}, "
                f"{100 * wa[best]['detection_accuracy']['hi']:.1f}])"
            ),
            "why": "A failure class output-only evaluation cannot reach by construction.",
        },
        {
            "id": "position_bias",
            "headline": (
                f"The weakest judge reversed its verdict on "
                f"{100 * d2['per_judge'][pos]['inconsistency_rate']['value']:.1f}% of comparisons "
                f"purely because the two options were swapped; showing the reference first "
                f"over-reports accuracy by up to "
                f"{100 * d2['per_judge'][pos]['fixed_order_inflation']['diff']:.1f} points"
            ),
            "why": "The number a fixed-order harness reports is inflated by a bias it did not control.",
        },
        {
            "id": "calibration",
            "headline": (
                f"The best judge ranks at "
                f"{100 * load('01_discrimination')['per_judge'][d1['judges'][0]]['overall_accuracy']['value']:.1f}% "
                f"but classifies at {100 * a['teacher_at_rubric_threshold']['value']:.1f}% at its own "
                f"rubric's cut-off — moving the threshold recovers "
                f"{100 * a['teacher_at_oracle_threshold']['value']:.1f}%"
            ),
            "why": "Ranking well and deciding well are different properties, measured differently.",
        },
    ]

    return {
        "provenance": load("01_discrimination")["provenance"],
        "findings": findings,
        "guardrail": {
            "block_precision": op["block_precision"],
            "false_block_rate": op["false_block_rate"],
            "bad_output_caught": op["bad_output_caught"],
            "threshold": op["threshold"],
            "ece_before": cal["ece_vs_truth_before_recalibration"],
            "ece_after": cal["ece_vs_truth_after_recalibration"],
            "student_accuracy": a["student"],
            "teacher_accuracy": a["teacher_at_oracle_threshold"],
            "p50_ms": d10["serial"]["p50"],
            "p99_ms": d10["serial"]["p99"],
            "budget_ms": d10["budget_p99_ms"],
        },
        "judges": d1["judges"],
        "figures": [{"file": f.replace(".png", ".svg"), "title": t} for f, t in FIGURES],
    }


def build_examples() -> dict:
    item = load_items(500)[7]
    return {
        "question": item.question,
        "context": item.context,
        "candidates": [
            {
                "label": "reference (known good)",
                "expect": "allow",
                "note": "The correct answer. A guardrail that blocks this is worse than useless.",
                "text": item.reference,
            },
            {
                "label": "hedging (vague, uncheckable)",
                "expect": "block",
                "note": "Committed values replaced by vagueness. Caught by absolute surface signature.",
                "text": hedging(item, 0.6, seed=1).text,
            },
            {
                "label": "omission (a required fact removed)",
                "expect": "block",
                "note": "Caught via coverage of the source record's content words.",
                "text": omission(item, 0.5, seed=1).text,
            },
            {
                "label": "numeric swap — the known blind spot",
                "expect": "allow (wrongly)",
                "note": (
                    "One number changed, and it contradicts the SOURCE RECORD above. The "
                    "guardrail scores this identically to the reference: a bag-of-n-grams model "
                    "cannot check a value against a record. This is what the cascade escalates "
                    "to the LLM judge — and what the judges themselves miss up to 42% of the time."
                ),
                "text": numeric_swap(item, 0.5, seed=1).text,
            },
            {
                "label": "padded reference (no error, just 3x longer)",
                "expect": "block",
                "note": (
                    "Mirror image of the judges' bias: they scored padding UP by as much as 1.45 "
                    "points, the student blocks it outright. Neither is neutral about length."
                ),
                "text": verbosity(item, 0.9, seed=1).text,
            },
        ],
    }


def main() -> None:
    data = SITE / "data"
    figs = SITE / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    (data / "summary.json").write_text(
        json.dumps(json_safe(build_summary()), separators=(",", ":")), encoding="utf-8"
    )
    (data / "examples.json").write_text(
        json.dumps(build_examples(), separators=(",", ":")), encoding="utf-8"
    )

    # Remove stale assets first. PNGs left behind here would be pushed to the
    # Space and rejected: Hugging Face blocks binary files above a size
    # threshold and wants Xet storage for them.
    stale = 0
    for old in figs.glob("*.png"):
        try:
            old.unlink()
            stale += 1
        except OSError as exc:
            print(f"    ! could not remove {old.name}: {exc}")
    if stale:
        print(f"  removed {stale} stale PNG(s) from site/figures/")

    # SVG, not PNG. Vector stays sharp under the viewer's zoom, and being text it
    # pushes to a Static Space over plain git -- Hugging Face rejects binaries
    # above a size threshold and wants Xet storage for them.
    n = 0
    for f, _ in FIGURES:
        src = (RESULTS / "figures" / f).with_suffix(".svg")
        if src.exists():
            shutil.copy2(src, figs / src.name)
            n += 1

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"  site/ assembled: {n} figures, {size / 1e6:.2f} MB total")
    for p in sorted(SITE.iterdir()):
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
