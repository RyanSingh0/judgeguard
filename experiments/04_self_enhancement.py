#!/usr/bin/env python
"""Self-enhancement bias: does a judge favour its own model family?

The trap in this measurement is confounding. If family X's answers score highest
across the whole panel, X's answers may simply be better. So the statistic is
leave-one-out: for every response, a judge's score is compared against the mean
score the *other* judges gave the same response. Response quality then cancels,
and what remains is judge-specific preference.

    self_enhancement(j) = mean_over_own_family_responses  [ s_j - mean_{j' != j} s_j' ]
                        - mean_over_other_family_responses[ s_j - mean_{j' != j} s_j' ]

Needs at least two model families to mean anything; the free tiers supply four.

    python experiments/04_self_enhancement.py --items 300
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

from judgeguard.config import load_registry
from judgeguard.data.generate import generate_response
from judgeguard.judges.run import ScoreTask, run_scores
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.store import save
from judgeguard.telemetry import tracking_run

CONFIG = "rubric"


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.set_defaults(items=300)
    args = apply_quick_isolation(ap.parse_args())
    n_items = 30 if args.quick else args.items
    judges = resolve_judges(args.judges)
    reg = load_registry()
    generators = [g for g in reg.generators if g in reg.aliases]

    t0 = banner("04  self-enhancement bias (leave-one-out)")
    items, _ = corpus(n_items, args.seed)

    responses: dict[str, dict[str, str]] = {
        g: {i.id: generate_response(g, i, seed=args.seed) for i in items} for g in generators
    }

    # scores[judge][generator][item_id]
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with tracking_run("judgeguard", "04_self_enhancement", {"items": n_items}) as run:
        for judge in judges:
            for gen in generators:
                gen_family = reg.by_alias(gen).family
                tasks = [
                    ScoreTask(
                        uid=f"{i.id}:{gen}",
                        item=i,
                        answer=responses[gen][i.id],
                        degradation="generated",
                        generator_family=gen_family,
                        len_ratio=len(responses[gen][i.id]) / max(len(i.reference), 1),
                    )
                    for i in items
                ]
                for j in run_scores(judge, CONFIG, tasks):
                    if j.score is not None:
                        scores[judge][gen][j.uid.split(":")[0]] = j.score

        per_judge: dict[str, dict] = {}
        for judge in judges:
            jf = reg.by_alias(judge).family
            own_dev: list[float] = []
            other_dev: list[float] = []
            by_gen: dict[str, list[float]] = {}
            for gen in generators:
                gf = reg.by_alias(gen).family
                devs = []
                for i in items:
                    mine = scores[judge][gen].get(i.id)
                    peers = [
                        scores[o][gen].get(i.id)
                        for o in judges
                        if o != judge and scores[o][gen].get(i.id) is not None
                    ]
                    if mine is None or not peers:
                        continue
                    devs.append(mine - sum(peers) / len(peers))
                by_gen[gen] = devs
                (own_dev if gf == jf else other_dev).extend(devs)

            effect = (
                paired_bootstrap_diff(
                    own_dev[: min(len(own_dev), len(other_dev))],
                    other_dev[: min(len(own_dev), len(other_dev))],
                    seed=args.seed,
                )
                if own_dev and other_dev
                else None
            )

            per_judge[judge] = {
                "family": jf,
                "leave_one_out_deviation_by_generator": {
                    g: bca_ci(v, seed=args.seed).as_dict() for g, v in by_gen.items() if v
                },
                "own_family_deviation": bca_ci(own_dev, seed=args.seed).as_dict()
                if own_dev
                else None,
                "other_family_deviation": bca_ci(other_dev, seed=args.seed).as_dict()
                if other_dev
                else None,
                "self_enhancement_points": effect.as_dict() if effect else None,
            }
            if effect:
                print(
                    f"    {judge:<14} favours own family by {effect.diff:+.2f} points  "
                    f"[{effect.lo:+.2f}, {effect.hi:+.2f}]  "
                    f"{'significant' if effect.significant else 'not distinguishable from noise'}"
                )
                run.log_metrics({f"{judge}_self_enhancement": effect.diff})

    payload = {
        "experiment": "04_self_enhancement",
        "question": "Does a judge score its own family's output higher than peers do?",
        "estimator": "leave-one-out deviation from the peer-judge mean on the same response",
        "config": CONFIG,
        "judges": judges,
        "generators": generators,
        "families": {a: reg.by_alias(a).family for a in set(judges) | set(generators)},
        "n_items": len(items),
        "per_judge": per_judge,
    }
    done(t0, save("04_self_enhancement", payload))


if __name__ == "__main__":
    run_main(main)
