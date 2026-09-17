"""Bounded replay of four pure tools; never execute arbitrary trace code or URLs."""

from __future__ import annotations

import inspect
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from judgeguard.agent.tools import TOOLS, call


class AuditStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str = Field(min_length=1, max_length=80)
    args: dict[str, Any]
    result: str = Field(max_length=2000)

    @field_validator("args")
    @classmethod
    def bounded_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 8:
            raise ValueError("At most eight arguments per step")
        for key, item in value.items():
            if len(key) > 80 or type(item) not in (str, int, float):
                raise ValueError("Arguments must be short names and scalar strings or numbers")
            if isinstance(item, str) and len(item) > 512:
                raise ValueError("Argument string exceeds 512 characters")
            if isinstance(item, (int, float)) and (abs(item) > 1e12 or not math.isfinite(item)):
                raise ValueError("Numeric argument out of bounds")
        return value


class AuditTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=4000)
    steps: list[AuditStep] = Field(min_length=1, max_length=40)
    final_answer: str = Field(max_length=4000)
    expected_final_answer: str | None = Field(default=None, max_length=4000)


def audit_trace(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        if len(payload) > 100_000:
            raise ValueError("Trace exceeds 100,000 characters")
        trace = AuditTrace.model_validate_json(payload)
    else:
        trace = AuditTrace.model_validate(payload)
    rows = []
    seen = set()
    warnings = []
    for index, step in enumerate(trace.steps, 1):
        row = {
            "step": index,
            "tool": step.tool,
            "recorded": step.result,
            "replayed": None,
            "status": "passed",
            "detail": "Recorded result matches replay.",
        }
        signature = json.dumps([step.tool, step.args], sort_keys=True)
        if signature in seen:
            warnings.append(
                f"Step {index} repeats an earlier call; redundancy is not a factual error."
            )
        seen.add(signature)
        if step.tool not in TOOLS:
            row.update(
                status="unsupported_tool",
                detail="Tool is outside the four-tool allowlist; not executed.",
            )
        else:
            try:
                inspect.signature(TOOLS[step.tool]).bind(**step.args)
                value = call(step.tool, step.args)
                # These tools return numeric strings. Reject overflow/non-finite output.
                if not math.isfinite(float(value)):
                    raise ValueError("Tool result is non-finite")
                row["replayed"] = value
                if value.strip() != step.result.strip():
                    row.update(
                        status="result_mismatch",
                        detail="Recorded output differs from executing the supplied arguments.",
                    )
            except (ValueError, TypeError, ArithmeticError, RecursionError, AttributeError) as exc:
                row.update(status="invalid_call", detail=f"{type(exc).__name__}: {str(exc)[:180]}")
        rows.append(row)
    failures = sum(r["status"] != "passed" for r in rows)
    outcome = (
        None
        if trace.expected_final_answer is None
        else trace.final_answer.strip() == trace.expected_final_answer.strip()
    )
    status = (
        "failed"
        if failures or outcome is False
        else ("passed" if outcome is True else "consistent")
    )
    return {
        "status": status,
        "steps_checked": len(rows),
        "step_failures": failures,
        "answer_matches_expected": outcome,
        "oracle_source": "user-supplied expected answer" if outcome is not None else "none",
        "steps": rows,
        "warnings": warnings,
        "method": "Deterministic independent tool replay; no LLM or network calls.",
        "scope": "Checks tool membership, arguments, recorded results, and optional exact expected-answer match. Does not establish task appropriateness, missing steps, causal dependencies, or safety of arbitrary agents. Reference-table values are fixed demo values; currency conversions are not live rates.",
    }


def trace_examples() -> dict[str, dict[str, Any]]:
    clean: dict[str, Any] = {
        "task": "Compute the mass in kg when warehouse A is filled to pallet capacity.",
        "steps": [
            {"tool": "lookup", "args": {"key": "warehouse_a_capacity"}, "result": "4200"},
            {"tool": "lookup", "args": {"key": "pallet_weight_kg"}, "result": "27.5"},
            {"tool": "calculator", "args": {"expr": "4200 * 27.5"}, "result": "115500"},
        ],
        "final_answer": "115500 kg",
        "expected_final_answer": "115500 kg",
    }
    examples = {"Valid execution": clean}
    for name in (
        "Wrong argument, correct final answer",
        "Fabricated tool result",
        "Undeclared tool",
        "Wrong final answer",
        "Redundant but valid call",
    ):
        examples[name] = json.loads(json.dumps(clean))
    examples["Wrong argument, correct final answer"]["steps"][2]["args"]["expr"] = "4200 * 28.5"
    examples["Fabricated tool result"]["steps"][0]["result"] = "9999"
    examples["Undeclared tool"]["steps"][0] = {
        "tool": "web_search",
        "args": {"query": "warehouse capacity"},
        "result": "4200",
    }
    examples["Wrong final answer"]["final_answer"] = "125500 kg"
    examples["Redundant but valid call"]["steps"].append(clean["steps"][0].copy())
    return examples
