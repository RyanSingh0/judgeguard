#!/usr/bin/env python
"""Build and freeze the degraded dataset.

Committing the generated set means a reader can audit every gold pair and
reproduce every downstream number without regenerating anything -- and without
an API key. The power analysis is written alongside it so the sample size can be
defended as a design decision rather than a budget artefact.

    python experiments/00_build_dataset.py --items 500
"""

from __future__ import annotations

from _common import apply_quick_isolation, banner, base_parser, corpus, done, run_main

from judgeguard.agent.rollout import build_trajectories, trajectory_stats
from judgeguard.data.corpus import corpus_stats
from judgeguard.degrade.text import (
    ERROR_DEGRADATIONS,
    PROBE_DEGRADATIONS,
    SEVERITIES,
    build_variants,
)
from judgeguard.degrade.trajectory import (
    TRAJ_ERROR_DEGRADATIONS,
    TRAJ_PROBE_DEGRADATIONS,
    build_trajectory_variants,
)
from judgeguard.stats.power import assess, n_for_mcnemar
from judgeguard.store import save


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--trajectories", type=int, default=60)
    args = apply_quick_isolation(ap.parse_args())
    n_items = 40 if args.quick else args.items
    n_traj = 12 if args.quick else args.trajectories

    t0 = banner("00  build dataset")
    items, _ = corpus(n_items, args.seed)
    variants = build_variants(items, seed=args.seed)
    trajs = build_trajectories(n_traj, seed=args.seed)
    tvariants = build_trajectory_variants(trajs, seed=args.seed)

    n_pairs = sum(1 for v in variants if v.introduces_error)
    power = {
        "design": "paired: every judge sees every item, so McNemar/paired bootstrap apply",
        "target_effect_pp": 7.0,
        "n_for_independent_samples": assess(n_pairs, 0.80, 0.73).as_dict(),
        "n_for_paired_mcnemar_at_20pct_discordant": n_for_mcnemar(0.20, 1.6),
        "achieved": assess(n_pairs, 0.80, 0.73).as_dict(),
        "note": (
            "Sample size is set by the paired design, not by free-tier quota. "
            "At n pairs above, a 7 percentage-point difference between two judge "
            "configurations is detectable at 80% power, alpha 0.05."
        ),
    }

    payload = {
        "corpus": corpus_stats(items),
        "trajectories": trajectory_stats(trajs),
        "degradations": {
            "text_error": ERROR_DEGRADATIONS,
            "text_probe": PROBE_DEGRADATIONS,
            "trajectory_error": TRAJ_ERROR_DEGRADATIONS,
            "trajectory_probe": TRAJ_PROBE_DEGRADATIONS,
            "severities": list(SEVERITIES),
        },
        "counts": {
            "items": len(items),
            "text_variants": len(variants),
            "text_error_pairs": n_pairs,
            "text_probe_pairs": len(variants) - n_pairs,
            "trajectory_tasks": len(trajs),
            "trajectory_variants": len(tvariants),
        },
        "power_analysis": power,
        "items": [i.model_dump() for i in items],
        "variants": [v.model_dump() for v in variants],
        "reference_trajectories": [t.model_dump() for t in trajs],
        "trajectory_variants": [t.model_dump() for t in tvariants],
    }
    path = save("degraded_set", payload)

    print(
        f"    {len(items)} items x {len(ERROR_DEGRADATIONS) + len(PROBE_DEGRADATIONS)} degradations "
        f"x {len(SEVERITIES)} severities = {len(variants)} text variants"
    )
    print(f"    {len(trajs)} trajectories -> {len(tvariants)} trajectory variants")
    print(f"    audit sample: {variants[0].edit}")
    done(t0, path)


if __name__ == "__main__":
    run_main(main)
