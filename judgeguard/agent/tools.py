"""Four deterministic tools. Scope discipline is the point.

This is not a framework and not a general agent. It exists solely to produce
reference trajectories that can be broken in controlled ways -- the same trick
as the text degradations, moved one level up from the output to the process.

Determinism is what makes it work: because every tool is a pure function over a
fixed table, the correct trajectory for a task is unambiguous and needs no human
to label it.

``calculator`` evaluates arithmetic through a whitelisted AST walk rather than
``eval``. A judge-reliability project that shells arbitrary strings into ``eval``
would be a poor advertisement for the author.
"""

from __future__ import annotations

import ast
import operator as op
from datetime import date, datetime
from typing import Any

# --------------------------------------------------------------------- calculator
_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
MAX_EXPONENT = 8


class ToolError(ValueError):
    """Raised when a tool is called with arguments it cannot honour."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        if isinstance(node.op, ast.Pow):
            right = _eval_node(node.right)
            if abs(right) > MAX_EXPONENT:
                raise ToolError("exponent too large")
            return _BINOPS[type(node.op)](_eval_node(node.left), right)
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ToolError(f"disallowed expression node: {type(node).__name__}")


def calculator(expr: str) -> str:
    """Evaluate an arithmetic expression. Whitelisted AST only, never ``eval``."""
    if len(expr) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ToolError(f"unparseable expression: {expr!r}") from exc
    value = _eval_node(tree)
    return f"{value:.4f}".rstrip("0").rstrip(".") if value % 1 else str(int(value))


# ------------------------------------------------------------------------ lookup
STATIC_TABLE: dict[str, str] = {
    "warehouse_a_capacity": "4200",
    "warehouse_b_capacity": "6150",
    "warehouse_c_capacity": "2875",
    "demurrage_rate_usd_per_day": "38.5",
    "handling_fee_usd": "412",
    "pallet_weight_kg": "27.5",
    "container_teu_weight_kg": "21600",
    "fuel_price_usd_per_litre": "1.42",
    "fleet_size": "58",
    "avg_route_km": "312",
    "consumption_l_per_100km": "28.4",
    "carbon_kg_per_litre": "2.68",
    "labour_rate_usd_per_hour": "31.75",
    "shift_hours": "8",
    "defect_rate_ppm": "740",
    "batch_size": "12500",
    "rework_cost_usd_per_unit": "6.4",
    "storage_usd_per_pallet_day": "2.15",
    "insurance_pct": "1.8",
    "customs_flat_usd": "265",
}


def lookup(key: str) -> str:
    if key not in STATIC_TABLE:
        raise ToolError(f"unknown key {key!r}")
    return STATIC_TABLE[key]


# --------------------------------------------------------------------- date_diff
def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def date_diff(d1: str, d2: str) -> str:
    """Whole days from d1 to d2 (may be negative)."""
    return str((_parse(d2) - _parse(d1)).days)


# ------------------------------------------------------------------ unit_convert
CONVERSIONS: dict[tuple[str, str], float] = {
    ("kg", "lb"): 2.2046226218,
    ("lb", "kg"): 0.45359237,
    ("km", "mi"): 0.6213711922,
    ("mi", "km"): 1.609344,
    ("l", "gal"): 0.2641720524,
    ("gal", "l"): 3.785411784,
    ("usd", "eur"): 0.92,
    ("eur", "usd"): 1.0869565217,
    ("t", "kg"): 1000.0,
    ("kg", "t"): 0.001,
    ("h", "min"): 60.0,
    ("min", "h"): 1 / 60.0,
}


def unit_convert(value: float | str, frm: str, to: str) -> str:
    key = (str(frm).lower(), str(to).lower())
    if key not in CONVERSIONS:
        raise ToolError(f"no conversion {frm} -> {to}")
    out = float(value) * CONVERSIONS[key]
    return f"{out:.4f}".rstrip("0").rstrip(".")


TOOLS: dict[str, Any] = {
    "calculator": calculator,
    "lookup": lookup,
    "date_diff": date_diff,
    "unit_convert": unit_convert,
}

TOOL_SIGNATURES = {
    "calculator": "calculator(expr: str) -> number",
    "lookup": "lookup(key: str) -> value from a fixed reference table",
    "date_diff": "date_diff(d1: 'YYYY-MM-DD', d2: 'YYYY-MM-DD') -> whole days",
    "unit_convert": "unit_convert(value: number, frm: str, to: str) -> converted number",
}


def call(tool: str, args: dict[str, Any]) -> str:
    if tool not in TOOLS:
        raise ToolError(f"undefined tool {tool!r}")
    return str(TOOLS[tool](**args))


__all__ = [
    "CONVERSIONS",
    "STATIC_TABLE",
    "TOOLS",
    "TOOL_SIGNATURES",
    "ToolError",
    "calculator",
    "call",
    "date_diff",
    "lookup",
    "unit_convert",
]
