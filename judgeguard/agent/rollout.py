"""Task generation and reference-trajectory rollout.

Every task is a small deterministic program over the four tools. The rollout
executes it for real, so the recorded trajectory is not a plausible-looking
transcript -- it is the correct one, with every intermediate value computed by
the tool that claims to have computed it. That is what makes a degraded copy
provably wrong.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from judgeguard.agent.tools import STATIC_TABLE, call
from judgeguard.data.schema import Step, Trajectory

_WAREHOUSES = ["warehouse_a_capacity", "warehouse_b_capacity", "warehouse_c_capacity"]


def _d(rng: random.Random) -> tuple[str, str]:
    start = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    end = start + timedelta(days=rng.randint(3, 120))
    return start.isoformat(), end.isoformat()


def _t_demurrage(rng: random.Random, tid: str) -> Trajectory:
    d1, d2 = _d(rng)
    days = call("date_diff", {"d1": d1, "d2": d2})
    rate = call("lookup", {"key": "demurrage_rate_usd_per_day"})
    total = call("calculator", {"expr": f"{days} * {rate}"})
    steps = [
        Step(
            tool="date_diff",
            args={"d1": d1, "d2": d2},
            result=days,
            reasoning="Length of the detention window in whole days.",
        ),
        Step(
            tool="lookup",
            args={"key": "demurrage_rate_usd_per_day"},
            result=rate,
            reasoning="Daily demurrage rate from the reference table.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{days} * {rate}"},
            result=total,
            reasoning="Days multiplied by the daily rate.",
        ),
    ]
    task = (
        f"A container was detained from {d1} to {d2}. Using the published demurrage rate, "
        f"what is the total demurrage charge in USD?"
    )
    return Trajectory(task_id=tid, task=task, steps=steps, final_answer=f"USD {total}")


def _t_capacity_convert(rng: random.Random, tid: str) -> Trajectory:
    wh = rng.choice(_WAREHOUSES)
    cap = call("lookup", {"key": wh})
    pw = call("lookup", {"key": "pallet_weight_kg"})
    kg = call("calculator", {"expr": f"{cap} * {pw}"})
    lb = call("unit_convert", {"value": kg, "frm": "kg", "to": "lb"})
    steps = [
        Step(
            tool="lookup",
            args={"key": wh},
            result=cap,
            reasoning="Pallet capacity of the requested warehouse.",
        ),
        Step(
            tool="lookup",
            args={"key": "pallet_weight_kg"},
            result=pw,
            reasoning="Mass of a single loaded pallet.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{cap} * {pw}"},
            result=kg,
            reasoning="Total mass at full capacity, in kilograms.",
        ),
        Step(
            tool="unit_convert",
            args={"value": kg, "frm": "kg", "to": "lb"},
            result=lb,
            reasoning="Convert to pounds as the question requires.",
        ),
    ]
    task = (
        f"If {wh.replace('_capacity', '').replace('_', ' ')} is filled to its pallet capacity, "
        f"what is the total stored mass in pounds?"
    )
    return Trajectory(task_id=tid, task=task, steps=steps, final_answer=f"{lb} lb")


def _t_fuel_carbon(rng: random.Random, tid: str) -> Trajectory:
    fleet = call("lookup", {"key": "fleet_size"})
    km = call("lookup", {"key": "avg_route_km"})
    cons = call("lookup", {"key": "consumption_l_per_100km"})
    litres = call("calculator", {"expr": f"{fleet} * {km} * {cons} / 100"})
    carbon = call("lookup", {"key": "carbon_kg_per_litre"})
    kg = call("calculator", {"expr": f"{litres} * {carbon}"})
    steps = [
        Step(
            tool="lookup",
            args={"key": "fleet_size"},
            result=fleet,
            reasoning="Vehicles in service.",
        ),
        Step(
            tool="lookup",
            args={"key": "avg_route_km"},
            result=km,
            reasoning="Average distance per vehicle.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{fleet} * {km} * {cons} / 100"},
            result=litres,
            reasoning="Fleet fuel burn for one route cycle, in litres.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{litres} * {carbon}"},
            result=kg,
            reasoning="Litres multiplied by the emission factor.",
        ),
    ]
    task = "What are the total CO2 emissions in kilograms for one full fleet route cycle?"
    return Trajectory(task_id=tid, task=task, steps=steps, final_answer=f"{kg} kg CO2")


def _t_rework_cost(rng: random.Random, tid: str) -> Trajectory:
    ppm = call("lookup", {"key": "defect_rate_ppm"})
    batch = call("lookup", {"key": "batch_size"})
    units = call("calculator", {"expr": f"{batch} * {ppm} / 1000000"})
    cost = call("lookup", {"key": "rework_cost_usd_per_unit"})
    total = call("calculator", {"expr": f"{units} * {cost}"})
    steps = [
        Step(
            tool="lookup",
            args={"key": "defect_rate_ppm"},
            result=ppm,
            reasoning="Defect rate in parts per million.",
        ),
        Step(
            tool="lookup", args={"key": "batch_size"}, result=batch, reasoning="Units in the batch."
        ),
        Step(
            tool="calculator",
            args={"expr": f"{batch} * {ppm} / 1000000"},
            result=units,
            reasoning="Expected defective units in this batch.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{units} * {cost}"},
            result=total,
            reasoning="Defective units multiplied by unit rework cost.",
        ),
    ]
    task = "What is the expected rework cost in USD for one production batch?"
    return Trajectory(task_id=tid, task=task, steps=steps, final_answer=f"USD {total}")


def _t_storage_window(rng: random.Random, tid: str) -> Trajectory:
    d1, d2 = _d(rng)
    days = call("date_diff", {"d1": d1, "d2": d2})
    rate = call("lookup", {"key": "storage_usd_per_pallet_day"})
    pallets = str(rng.randint(40, 900))
    total = call("calculator", {"expr": f"{days} * {rate} * {pallets}"})
    steps = [
        Step(
            tool="date_diff",
            args={"d1": d1, "d2": d2},
            result=days,
            reasoning="Storage window in days.",
        ),
        Step(
            tool="lookup",
            args={"key": "storage_usd_per_pallet_day"},
            result=rate,
            reasoning="Daily storage rate per pallet.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{days} * {rate} * {pallets}"},
            result=total,
            reasoning="Days times rate times pallet count.",
        ),
    ]
    task = (
        f"{pallets} pallets were stored from {d1} to {d2}. What is the total storage charge in USD?"
    )
    return Trajectory(task_id=tid, task=task, steps=steps, final_answer=f"USD {total}")


def _t_labour_convert(rng: random.Random, tid: str) -> Trajectory:
    rate = call("lookup", {"key": "labour_rate_usd_per_hour"})
    shift = call("lookup", {"key": "shift_hours"})
    crews = str(rng.randint(2, 14))
    usd = call("calculator", {"expr": f"{rate} * {shift} * {crews}"})
    eur = call("unit_convert", {"value": usd, "frm": "usd", "to": "eur"})
    steps = [
        Step(
            tool="lookup",
            args={"key": "labour_rate_usd_per_hour"},
            result=rate,
            reasoning="Hourly labour rate.",
        ),
        Step(
            tool="lookup",
            args={"key": "shift_hours"},
            result=shift,
            reasoning="Hours in a standard shift.",
        ),
        Step(
            tool="calculator",
            args={"expr": f"{rate} * {shift} * {crews}"},
            result=usd,
            reasoning="Cost of one shift across all crews, in USD.",
        ),
        Step(
            tool="unit_convert",
            args={"value": usd, "frm": "usd", "to": "eur"},
            result=eur,
            reasoning="Convert to euro as the question requires.",
        ),
    ]
    task = f"What does one standard shift cost in euro when {crews} crews are rostered?"
    return Trajectory(task_id=tid, task=task, steps=steps, final_answer=f"EUR {eur}")


_TEMPLATES = [
    _t_demurrage,
    _t_capacity_convert,
    _t_fuel_carbon,
    _t_rework_cost,
    _t_storage_window,
    _t_labour_convert,
]


def build_trajectories(n: int = 60, seed: int = 20260731) -> list[Trajectory]:
    """Reference trajectories, balanced across templates, 3-4 tool calls each."""
    rng = random.Random(seed)
    out: list[Trajectory] = []
    for i in range(n):
        tpl = _TEMPLATES[i % len(_TEMPLATES)]
        out.append(tpl(rng, f"traj-{i:03d}"))
    return out


def trajectory_stats(trajs: list[Trajectory]) -> dict[str, object]:
    steps = [len(t.steps) for t in trajs]
    tools = sorted({s.tool for t in trajs for s in t.steps})
    return {
        "n_tasks": len(trajs),
        "tools_used": tools,
        "table_keys": len(STATIC_TABLE),
        "mean_steps": round(sum(steps) / max(len(steps), 1), 2),
        "min_steps": min(steps) if steps else 0,
        "max_steps": max(steps) if steps else 0,
    }


__all__ = ["build_trajectories", "trajectory_stats"]
