#!/usr/bin/env python
"""Verbosity bias, measured in points.

The padded variant is the *unmodified reference* plus content-free filler. No
fact is added, removed or altered, so the correct score change is exactly zero.
Whatever the judge does instead is the bias, and it is denominated in the judge's
own scale rather than in a correlation coefficient -- which is what makes it
actionable ("padding to 2x buys you +0.6 points").

    python experiments/03_verbosity_bias.py --items 500
"""

from __future__ import annotations

from _common import (
    apply_quick_isolation,
    banner,
    base_parser,
    corpus,
    done,
    resolve_judges,
    run_main,
)

from judgeguard.degrade.text import SEVERITIES, verbosity
from judgeguard.judges.prompts import POINTWISE_CONFIGS
from judgeguard.judges.run import ScoreTask, run_scores
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.store import save
from judgeguard.telemetry import tracking_run


def main() -> None:
    ap = base_parser(__doc__ or "")
    args = apply_quick_isolation(ap.parse_args())
    n_items = 40 if args.quick else args.items
    judges = resolve_judges(args.judges)
    configs = list(POINTWISE_CONFIGS)

    t0 = banner("03  verbosity bias: does padding raise the score?")
    items, _ = corpus(n_items, args.seed)
    padded = {s: [verbosity(i, s, seed=args.seed) for i in items] for s in SEVERITIES}

    per_judge: dict[str, dict] = {}
    with tracking_run("judgeguard", "03_verbosity_bias", {"items": n_items}) as run:
        for judge in judges:
            entry: dict[str, dict] = {}
            for config in configs:
                ref_tasks = [
                    ScoreTask(uid=f"{i.id}:reference", item=i, answer=i.reference, len_ratio=1.0)
                    for i in items
                ]
                ref_j = run_scores(judge, config, ref_tasks)
                ref_by = {j.uid.split(":")[0]: j.score for j in ref_j}
                by_sev: dict[float, dict] = {}
                for sev in SEVERITIES:
                    vs = padded[sev]
                    pad_tasks = [
                        ScoreTask(
                            uid=v.uid,
                            item=next(i for i in items if i.id == v.item_id),
                            answer=v.text,
                            degradation="verbosity",
                            severity=sev,
                            len_ratio=v.len_ratio,
                        )
                        for v in vs
                    ]
                    pad_j = run_scores(judge, config, pad_tasks)
                    a, b, ratios = [], [], []
                    for j, v in zip(pad_j, vs, strict=True):
                        r = ref_by.get(v.item_id)
                        if r is None or j.score is None:
                            continue
                        a.append(j.score)
                        b.append(r)
                        ratios.append(v.len_ratio)
                    diff = paired_bootstrap_diff(a, b, seed=args.seed)
                    mean_ratio = sum(ratios) / max(len(ratios), 1)
                    by_sev[f"{sev:g}"] = {
                        "mean_length_ratio": mean_ratio,
                        "delta_points": diff.as_dict(),
                        "points_per_doubling": diff.diff / max((mean_ratio - 1.0), 1e-9)
                        if mean_ratio > 1
                        else 0.0,
                        "share_scored_higher": bca_ci(
                            [int(x > y) for x, y in zip(a, b, strict=True)], seed=args.seed
                        ).as_dict(),
                        "padded_mean_score": sum(a) / max(len(a), 1),
                        "reference_mean_score": sum(b) / max(len(b), 1),
                    }
                entry[config] = by_sev
            per_judge[judge] = entry
            worst = max(
                entry[c][f"{s:g}"]["delta_points"]["diff"] for c in configs for s in SEVERITIES
            )
            print(f"    {judge:<14} max score gain from pure padding: {worst:+.2f} points")
            run.log_metrics({f"{judge}_max_verbosity_gain": worst})

    payload = {
        "experiment": "03_verbosity_bias",
        "question": "Does padding a CORRECT answer with content-free filler raise its score?",
        "control": "The padded text is the unmodified reference plus filler. Correct delta is 0.",
        "judges": judges,
        "configs": configs,
        "n_items": len(items),
        "per_judge": per_judge,
    }
    done(t0, save("03_verbosity_bias", payload))


if __name__ == "__main__":
    run_main(main)
