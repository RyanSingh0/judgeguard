"""Request/response contracts for the service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GuardRequest(BaseModel):
    question: str = Field(default="", max_length=8000)
    context: str = Field(default="", max_length=40000)
    answer: str = Field(..., min_length=1, max_length=40000)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class GuardResponse(BaseModel):
    decision: Literal["allow", "block"]
    probability_good: float
    threshold: float
    latency_ms: float
    budget_ms: float
    within_budget: bool
    reason: str
    signals: dict[str, float]
    path: Literal["inline-guardrail"] = "inline-guardrail"


class EvaluateRequest(BaseModel):
    question: str = Field(default="", max_length=8000)
    context: str = Field(default="", max_length=40000)
    answer: str = Field(..., min_length=1, max_length=40000)
    judge: str | None = None
    config: Literal["vague", "rubric", "cot"] = "cot"


class EvaluateResponse(BaseModel):
    judge: str
    config: str
    score: float | None
    parsed_ok: bool
    reasoning: str
    latency_ms: float
    cost_usd: float
    model_served: str
    simulated: bool
    judge_reliability: dict[str, Any]
    path: Literal["async-evaluator"] = "async-evaluator"


class HealthResponse(BaseModel):
    status: str
    version: str
    provider_mode: str
    student_loaded: bool
    results_available: list[str]
    uptime_seconds: float


__all__ = [
    "EvaluateRequest",
    "EvaluateResponse",
    "GuardRequest",
    "GuardResponse",
    "HealthResponse",
]
