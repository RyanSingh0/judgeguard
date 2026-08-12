"""The four agent-trajectory degradations.

This is the transfer that makes the project a contribution rather than a
replication: degradation-based gold labelling exists for text, but the same
construction applied to an agent's *process* exposes failure classes that
output-only evaluation cannot see by construction.

* ``length_padding``  -- BIAS PROBE. Redundant but harmless steps, correct
  answer. Nothing is wrong. If the judge prefers it, that is trajectory-length
  bias, measured in points.
* ``phantom_tool``    -- a call to a tool that was never defined. Tests whether
  the judge validates the action space at all.
* ``wrong_argument``  -- correct tool, wrong argument value, downstream results
  recomputed so the transcript stays internally consistent. This is the hardest
  and the most production-relevant: the trajectory *looks* right.
* ``silent_failure``  -- correct final answer reached by an incorrect path.
  Output-only evaluation scores this perfect, by definition.
"""

from __future__ import annotations

import random

from judgeguard.data.schema import Step, Trajectory, TrajectoryVariant

TRAJ_SEVERITIES: tuple[float, ...] = (0.2, 0.5, 0.9)

_PHANTOM_TOOLS = ["web_search", "sql_query", "currency_api", "weather_lookup", "geocode"]

_REDUNDANT = [
    (
        "lookup",
        {"key": "handling_fee_usd"},
        "412",
        "Retrieved for completeness; not used downstream.",
    ),
    (
        "lookup",
        {"key": "insurance_pct"},
        "1.8",
        "Retrieved for context; does not affect the result.",
    ),
    ("calculator", {"expr": "1 * 1"}, "1", "Sanity check of the arithmetic backend."),
    (
        "lookup",
        {"key": "customs_flat_usd"},
        "265",
        "Noted for the record; not part of this calculation.",
    ),
]


def _rng(traj: Trajectory, name: str, sev: float, seed: int) -> random.Random:
    return random.Random(f"{traj.task_id}|{name}|{sev}|{seed}")


def _variant(
    traj: Trajectory,
    new: Trajectory,
    name: str,
    sev: float,
    edit: str,
    *,
    introduces_error: bool = True,
) -> TrajectoryVariant:
    return TrajectoryVariant(
        task_id=traj.task_id,
        variant_id=f"{name}@{sev:g}",
        degradation=name,
        severity=sev,
        trajectory=new,
        edit=edit,
        introduces_error=introduces_error,
        len_ratio=round(len(new.render()) / max(len(traj.render()), 1), 4),
    )


def length_padding(traj: Trajectory, severity: float, *, seed: int = 0) -> TrajectoryVariant:
    """BIAS PROBE: more steps, same correct answer, no new information."""
    rng = _rng(traj, "length_padding", severity, seed)
    k = max(1, round(severity * 4))
    extra = [
        Step(tool=t, args=dict(a), result=r, reasoning=why)
        for t, a, r, why in rng.sample(_REDUNDANT, min(k, len(_REDUNDANT)))
    ]
    steps = list(traj.steps)
    for s in extra:
        steps.insert(rng.randint(0, len(steps)), s)
    new = Trajectory(
        task_id=traj.task_id, task=traj.task, steps=steps, final_answer=traj.final_answer
    )
    return _variant(
        traj,
        new,
        "length_padding",
        severity,
        f"Inserted {len(extra)} redundant but harmless step(s). The final answer is unchanged "
        f"and correct; only the path got longer.",
        introduces_error=False,
    )


def phantom_tool(traj: Trajectory, severity: float, *, seed: int = 0) -> TrajectoryVariant:
    """Call a tool that does not exist in the declared action space."""
    rng = _rng(traj, "phantom_tool", severity, seed)
    k = max(1, round(severity * 2))
    steps = list(traj.steps)
    names = rng.sample(_PHANTOM_TOOLS, min(k, len(_PHANTOM_TOOLS)))
    for name in names:
        pos = rng.randint(0, max(len(steps) - 1, 0))
        steps.insert(
            pos,
            Step(
                tool=name,
                args={"query": traj.task[:48]},
                result="(result returned)",
                reasoning="Consulted an external source for supporting context.",
            ),
        )
    new = Trajectory(
        task_id=traj.task_id, task=traj.task, steps=steps, final_answer=traj.final_answer
    )
    return _variant(
        traj,
        new,
        "phantom_tool",
        severity,
        f"Inserted {len(names)} call(s) to undefined tool(s) ({', '.join(names)}). The declared "
        f"action space contains only calculator, lookup, date_diff and unit_convert.",
    )


def wrong_argument(traj: Trajectory, severity: float, *, seed: int = 0) -> TrajectoryVariant:
    """Correct tool, wrong argument. The transcript stays internally consistent."""
    rng = _rng(traj, "wrong_argument", severity, seed)
    steps = [s.model_copy(deep=True) for s in traj.steps]
    idx = [
        i
        for i, s in enumerate(steps)
        if s.tool in ("calculator", "unit_convert", "date_diff", "lookup")
    ]
    if not idx:
        return length_padding(traj, severity, seed=seed)
    k = max(1, round(severity * min(2, len(idx))))
    targets = rng.sample(idx, k)
    notes = []
    for i in targets:
        s = steps[i]
        if s.tool == "calculator":
            before = s.args["expr"]
            toks = before.split()
            num_pos = [j for j, t in enumerate(toks) if _isnum(t)]
            if num_pos:
                j = rng.choice(num_pos)
                old = float(toks[j])
                new_v = old * (1 + (0.05 + 0.5 * severity) * rng.choice((-1, 1)))
                toks[j] = f"{new_v:.4f}".rstrip("0").rstrip(".")
                s.args["expr"] = " ".join(toks)
                notes.append(f"step {i + 1} calculator arg {old:g} -> {toks[j]}")
        elif s.tool == "unit_convert":
            old = s.args.get("frm", "kg")
            s.args["frm"] = {
                "kg": "lb",
                "lb": "kg",
                "usd": "eur",
                "eur": "usd",
                "km": "mi",
                "mi": "km",
            }.get(str(old), "lb")
            notes.append(f"step {i + 1} unit_convert frm {old} -> {s.args['frm']}")
        elif s.tool == "date_diff":
            old = str(s.args["d2"])
            y, m, d = (int(x) for x in old.split("-"))
            s.args["d2"] = f"{y:04d}-{m:02d}-{min(28, max(1, d + rng.choice((-3, 5)))):02d}"
            notes.append(f"step {i + 1} date_diff d2 {old} -> {s.args['d2']}")
        else:
            old = s.args["key"]
            s.args["key"] = (
                "warehouse_b_capacity" if old != "warehouse_b_capacity" else "warehouse_c_capacity"
            )
            notes.append(f"step {i + 1} lookup key {old} -> {s.args['key']}")
    new = Trajectory(
        task_id=traj.task_id, task=traj.task, steps=steps, final_answer=traj.final_answer
    )
    return _variant(
        traj,
        new,
        "wrong_argument",
        severity,
        f"Passed a wrong argument to a correct tool ({'; '.join(notes)}). Tool names, step count "
        f"and prose are untouched, so the trajectory reads as valid.",
    )


def silent_failure(traj: Trajectory, severity: float, *, seed: int = 0) -> TrajectoryVariant:
    """Right answer, wrong path. Output-only evaluation scores this perfect."""
    rng = _rng(traj, "silent_failure", severity, seed)
    steps = [s.model_copy(deep=True) for s in traj.steps]
    k = max(1, round(severity * min(2, len(steps))))
    targets = rng.sample(range(len(steps)), k)
    notes = []
    for i in targets:
        s = steps[i]
        try:
            val = float(s.result)
            s.result = f"{val * (1 + (0.1 + 0.6 * severity)):.4f}".rstrip("0").rstrip(".")
        except ValueError:
            s.result = "(unavailable)"
        s.reasoning = (s.reasoning or "") + " Value carried forward."
        notes.append(f"step {i + 1} result corrupted")
    new = Trajectory(
        task_id=traj.task_id, task=traj.task, steps=steps, final_answer=traj.final_answer
    )
    return _variant(
        traj,
        new,
        "silent_failure",
        severity,
        f"Corrupted {len(notes)} intermediate result(s) ({'; '.join(notes)}) while leaving the "
        f"final answer at its correct value. The stated answer no longer follows from the steps.",
    )


def _isnum(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


TRAJECTORY_DEGRADATIONS = {
    "length_padding": length_padding,
    "phantom_tool": phantom_tool,
    "wrong_argument": wrong_argument,
    "silent_failure": silent_failure,
}

TRAJ_ERROR_DEGRADATIONS = [k for k in TRAJECTORY_DEGRADATIONS if k != "length_padding"]
TRAJ_PROBE_DEGRADATIONS = ["length_padding"]


def build_trajectory_variants(
    trajs: list[Trajectory],
    *,
    severities: tuple[float, ...] = TRAJ_SEVERITIES,
    seed: int = 20260731,
) -> list[TrajectoryVariant]:
    out: list[TrajectoryVariant] = []
    for t in trajs:
        for fn in TRAJECTORY_DEGRADATIONS.values():
            for sev in severities:
                out.append(fn(t, sev, seed=seed))
    return out


__all__ = [
    "TRAJECTORY_DEGRADATIONS",
    "TRAJ_ERROR_DEGRADATIONS",
    "TRAJ_PROBE_DEGRADATIONS",
    "TRAJ_SEVERITIES",
    "build_trajectory_variants",
    "length_padding",
    "phantom_tool",
    "silent_failure",
    "wrong_argument",
]
