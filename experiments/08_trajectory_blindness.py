#!/usr/bin/env python
"""Agent-trajectory blindness. The part output-only evaluation cannot reach.

Same construction as the text battery, moved one level up: break the agent's
*process* rather than its answer, and see whether the judge notices.

Two of the four degradations are the interesting ones, and both are chosen
because they are what actually goes wrong in production:

* ``wrong_argument`` -- the correct tool called with a wrong value. The
  transcript reads as valid, so a judge with no execution model has nothing to
  catch.
* ``phantom_tool``   -- a call to a tool that was never declared. The action
  space is printed in the prompt, so failing this is a failure to check
  something the judge was explicitly given.

``silent_failure`` produces the right answer via a wrong path: by construction,
output-only evaluation scores it perfect. ``length_padding`` is the probe -- more
steps, same correct answer, nothing wrong.

A negative result here, measured rigorously, is a real contribution. If the
judges are bad at this, the README says so loudly.

    python experiments/08_trajectory_blindness.py --trajectories 60
"""

from __future__ import annotations

from collections import defaultdict

from _common import apply_quick_isolation, banner, base_parser, done, resolve_judges

from judgeguard.agent.rollout import build_trajectories
from judgeguard.data.schema import Item
from judgeguard.degrade.trajectory import (
    TRAJ_ERROR_DEGRADATIONS,
    TRAJ_SEVERITIES,
    build_trajectory_variants,
    length_padding,
)
from judgeguard.judges.run import PairTask, ScoreTask, run_pairwise, run_scores
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.store import save
from judgeguard.telemetry import tracking_run

BLINDNESS_LABEL = {
    "phantom_tool": "hallucination blindness",
    "wrong_argument": "argument blindness",
    "silent_failure": "path blindness",
    "length_padding": "trajectory-length bias (probe)",
}


def _shim(task_id: str) -> Item:
    """Trajectory prompts carry their own header; the Item is a carrier only."""
    return Item(id=task_id, domain="agent", context="", question="", reference="")


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--trajectories", type=int, default=60)
    args = apply_quick_isolation(ap.parse_args())
    n_traj = 12 if args.quick else args.trajectories
    judges = resolve_judges(args.judges)

    t0 = banner("08  agent-trajectory blindness")
    trajs = build_trajectories(n_traj, seed=args.seed)
    by_id = {t.task_id: t for t in trajs}
    variants = build_trajectory_variants(trajs, seed=args.seed)
    err = [v for v in variants if v.introduces_error]

    pair_tasks = [
        PairTask(
            uid=v.uid,
            item=_shim(v.task_id),
            reference=by_id[v.task_id].render(),
            degraded=v.trajectory.render(),
            kind="trajectory",
            degradation=v.degradation,
            severity=v.severity,
            len_ratio=v.len_ratio,
        )
        for v in err
    ]
    padded = [length_padding(t, s, seed=args.seed) for t in trajs for s in TRAJ_SEVERITIES]

    per_judge: dict[str, dict] = {}
    with tracking_run("judgeguard", "08_trajectory", {"trajectories": n_traj}) as run:
        for judge in judges:
            out = run_pairwise(judge, "pairwise", pair_tasks)
            by_deg: dict[str, list[int]] = defaultdict(list)
            by_deg_sev: dict[tuple[str, float], list[int]] = defaultdict(list)
            incons: dict[str, list[int]] = defaultdict(list)
            for o in out:
                by_deg[o.degradation].append(o.correct_swapped)
                by_deg_sev[(o.degradation, o.severity)].append(o.correct_swapped)
                incons[o.degradation].append(0 if o.consistent else 1)

            # Trajectory-length bias: score the reference, then the padded copy.
            ref_j = run_scores(
                judge,
                "rubric",
                [
                    ScoreTask(
                        uid=f"{t.task_id}:reference",
                        item=_shim(t.task_id),
                        answer=t.render(),
                        kind="trajectory",
                    )
                    for t in trajs
                ],
            )
            ref_by = {j.uid.split(":")[0]: j.score for j in ref_j}
            pad_j = run_scores(
                judge,
                "rubric",
                [
                    ScoreTask(
                        uid=v.uid,
                        item=_shim(v.task_id),
                        answer=v.trajectory.render(),
                        kind="trajectory",
                        degradation="length_padding",
                        severity=v.severity,
                        len_ratio=v.len_ratio,
                    )
                    for v in padded
                ],
            )
            a, b = [], []
            for j, v in zip(pad_j, padded, strict=True):
                r = ref_by.get(v.task_id)
                if r is not None and j.score is not None:
                    a.append(j.score)
                    b.append(r)
            length_bias = paired_bootstrap_diff(a, b, seed=args.seed)

            blind = {
                d: {
                    "label": BLINDNESS_LABEL[d],
                    "detection_accuracy": bca_ci(v, seed=args.seed).as_dict(),
                    "miss_rate": 1 - bca_ci(v, seed=args.seed).value,
                    "order_inconsistency": bca_ci(incons[d], seed=args.seed).as_dict(),
                    "by_severity": {
                        f"{s:g}": bca_ci(by_deg_sev[(d, s)], seed=args.seed).as_dict()
                        for s in TRAJ_SEVERITIES
                        if by_deg_sev[(d, s)]
                    },
                }
                for d, v in sorted(by_deg.items())
            }
            per_judge[judge] = {
                "n_comparisons": len(out),
                "overall_detection": bca_ci(
                    [o.correct_swapped for o in out], seed=args.seed
                ).as_dict(),
                "by_degradation": blind,
                "trajectory_length_bias_points": length_bias.as_dict(),
            }
            worst = min(blind, key=lambda d: blind[d]["detection_accuracy"]["value"])
            print(
                f"    {judge:<14} worst: {BLINDNESS_LABEL[worst]:<24} "
                f"missed {100 * blind[worst]['miss_rate']:5.1f}% of cases;  "
                f"length bias {length_bias.diff:+.2f} pts"
            )
            run.log_metrics(
                {f"{judge}_trajectory_detection": per_judge[judge]["overall_detection"]["value"]}
            )

    payload = {
        "experiment": "08_trajectory_blindness",
        "question": "Do judges detect process errors that never surface in the final answer?",
        "why_it_matters": (
            "Malformed tool arguments and calls to undefined tools are the failures that break "
            "agents in production, and they are invisible to output-only evaluation by "
            "construction. silent_failure makes that explicit: the final answer is correct."
        ),
        "judges": judges,
        "n_trajectories": len(trajs),
        "degradations": TRAJ_ERROR_DEGRADATIONS,
        "severities": list(TRAJ_SEVERITIES),
        "scoring_rule": "swap-and-average pairwise: correct only if both orderings prefer the reference",
        "per_judge": per_judge,
    }
    done(t0, save("08_trajectory_blindness", payload))


if __name__ == "__main__":
    main()
