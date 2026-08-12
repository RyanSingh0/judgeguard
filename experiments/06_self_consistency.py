#!/usr/bin/env python
"""How often does a judge disagree with itself?

Five independent reruns of the same judge on the same items at temperature 0.7.
Krippendorff's alpha rather than pairwise kappa, because there are five raters
here and averaging ten kappas throws away information.

Self-consistency puts a ceiling on everything else in the battery: a judge that
cannot reproduce its own verdict cannot be meaningfully compared to another
judge, and any accuracy difference smaller than its own run-to-run variance is
not a difference.

    python experiments/06_self_consistency.py --items 200 --replicates 5
"""

from __future__ import annotations

import statistics

from _common import apply_quick_isolation, banner, base_parser, corpus, done, resolve_judges

from judgeguard.degrade.text import ERROR_DEGRADATIONS, build_variants
from judgeguard.judges.run import ScoreTask, run_scores
from judgeguard.stats.intervals import bca_ci
from judgeguard.stats.tests import judge_agreement, krippendorff_alpha_nominal
from judgeguard.store import save
from judgeguard.telemetry import tracking_run

CONFIG = "rubric"


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.set_defaults(items=200)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = apply_quick_isolation(ap.parse_args())
    n_items = 25 if args.quick else args.items
    reps = 3 if args.quick else args.replicates
    judges = resolve_judges(args.judges)

    t0 = banner(f"06  self-consistency ({reps} reruns at temperature {args.temperature})")
    items, by_id = corpus(n_items, args.seed)
    variants = build_variants(items, seed=args.seed, degradations=ERROR_DEGRADATIONS)
    tasks = [
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

    per_judge: dict[str, dict] = {}
    with tracking_run("judgeguard", "06_self_consistency", {"replicates": reps}) as run:
        for judge in judges:
            runs: list[list[float | None]] = []
            for r in range(reps):
                out = run_scores(judge, CONFIG, tasks, temperature=args.temperature, replicate=r)
                runs.append([j.score for j in out])

            # Discretise onto the rubric's own bands before measuring agreement.
            def band(s: float | None) -> str | None:
                if s is None:
                    return None
                return (
                    "excellent"
                    if s >= 9
                    else "good"
                    if s >= 7
                    else "adequate"
                    if s >= 5
                    else "poor"
                    if s >= 3
                    else "bad"
                )

            banded = [[band(s) for s in r] for r in runs]
            alpha = krippendorff_alpha_nominal(banded)
            kappas = [
                judge_agreement(list(banded[i]), list(banded[j]))
                for i in range(reps)
                for j in range(i + 1, reps)
            ]
            per_item_sd = [
                statistics.pstdev([r[k] for r in runs if r[k] is not None])
                for k in range(len(tasks))
                if sum(1 for r in runs if r[k] is not None) > 1
            ]
            flip = [
                int(len({band(r[k]) for r in runs if r[k] is not None}) > 1)
                for k in range(len(tasks))
            ]
            # The band flip rate is granular; the accept/reject flip rate is the
            # one that changes what a guardrail does, so both are reported.
            decision_flip = [
                int(len({int(r[k] >= 7.0) for r in runs if r[k] is not None}) > 1)
                for k in range(len(tasks))
            ]
            # Flip rates across five runs compound: "at least one disagreement in
            # five" is not the same statistic as "two runs disagree". Both are
            # reported, because only the second is directly interpretable.
            pair_disagree = []
            for i in range(reps):
                for j in range(i + 1, reps):
                    d = [
                        int((runs[i][k] >= 7.0) != (runs[j][k] >= 7.0))
                        for k in range(len(tasks))
                        if runs[i][k] is not None and runs[j][k] is not None
                    ]
                    pair_disagree.append(sum(d) / max(len(d), 1))

            per_judge[judge] = {
                "replicates": reps,
                "temperature": args.temperature,
                "krippendorff_alpha": alpha,
                "mean_pairwise_kappa": sum(kappas) / max(len(kappas), 1),
                "verdict_flip_rate": bca_ci(flip, seed=args.seed).as_dict(),
                "accept_reject_flip_rate": bca_ci(decision_flip, seed=args.seed).as_dict(),
                "two_run_decision_disagreement": sum(pair_disagree) / max(len(pair_disagree), 1),
                "mean_score_sd": sum(per_item_sd) / max(len(per_item_sd), 1),
                "max_score_sd": max(per_item_sd) if per_item_sd else 0.0,
            }
            print(
                f"    {judge:<14} alpha={alpha:.3f}  kappa={per_judge[judge]['mean_pairwise_kappa']:.3f}  "
                f"two-run accept/reject disagreement "
                f"{100 * sum(pair_disagree) / len(pair_disagree):.1f}%  "
                f"(any-of-5 flip {100 * sum(decision_flip) / len(decision_flip):.1f}%)"
            )
            run.log_metrics({f"{judge}_alpha": alpha})

    payload = {
        "experiment": "06_self_consistency",
        "question": "Run the same judge five times on the same item. How often does it disagree with itself?",
        "config": CONFIG,
        "judges": judges,
        "n_items": len(items),
        "n_pairs": len(tasks),
        "per_judge": per_judge,
        "interpretation": (
            "The two-run disagreement rate is the noise floor for every other comparison in the "
            "battery: an accuracy gap narrower than this is not measurable with one run."
        ),
    }
    done(t0, save("06_self_consistency", payload))


if __name__ == "__main__":
    main()
