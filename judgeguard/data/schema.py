"""Typed records for everything that flows through the harness.

The unusual field is ``Item.facts``. Most QA corpora give you a question and a
free-text gold answer, which makes controlled degradation approximate: you can
delete a sentence but you cannot say precisely what information was destroyed.
Here each reference answer is *composed* from an explicit fact list, so every
perturbation is exact, reversible and auditable -- ``Variant.edit`` records the
literal change and ``Variant.facts_removed`` / ``facts_corrupted`` record its
semantic footprint.

That is what lets the gold label be defended: "known worse" is not a vibe, it is
"this variant asserts 47 where the reference asserts 42".
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FactKind(str, Enum):
    NUMERIC = "numeric"
    ENTITY = "entity"
    DATE = "date"
    CLAIM = "claim"


class Fact(BaseModel):
    key: str
    value: str
    kind: FactKind = FactKind.CLAIM
    unit: str = ""
    required: bool = True
    clause: str = "{value}."  # sentence template; the reference is the join of these

    def render(self) -> str:
        return f"{self.value}{(' ' + self.unit) if self.unit else ''}"

    def sentence(self) -> str:
        return self.clause.format(value=self.render())


class Item(BaseModel):
    id: str
    domain: str
    context: str
    question: str
    reference: str
    facts: list[Fact] = Field(default_factory=list)

    def numeric_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.kind is FactKind.NUMERIC]


class Variant(BaseModel):
    """A deliberately-broken version of an item's reference answer."""

    item_id: str
    variant_id: str
    degradation: str
    severity: float
    text: str
    edit: str  # one-sentence audit trail
    facts_removed: list[str] = Field(default_factory=list)
    facts_corrupted: list[str] = Field(default_factory=list)
    introduces_error: bool = True  # False for the verbosity probe
    len_ratio: float = 1.0  # len(variant) / len(reference)

    @property
    def uid(self) -> str:
        return f"{self.item_id}:{self.variant_id}"


class Pair(BaseModel):
    """Gold-ordered pair. ``better`` is the reference, by construction."""

    uid: str
    item: Item
    better: str
    worse: str
    degradation: str
    severity: float
    len_ratio: float = 1.0
    kind: Literal["text", "trajectory"] = "text"


class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str
    reasoning: str = ""


class Trajectory(BaseModel):
    task_id: str
    task: str
    steps: list[Step]
    final_answer: str

    def render(self) -> str:
        lines = [f"TASK: {self.task}"]
        for i, s in enumerate(self.steps, 1):
            arg_str = ", ".join(f"{k}={v!r}" for k, v in s.args.items())
            lines.append(f"STEP {i}: call {s.tool}({arg_str}) -> {s.result}")
            if s.reasoning:
                lines.append(f"        rationale: {s.reasoning}")
        lines.append(f"FINAL ANSWER: {self.final_answer}")
        return "\n".join(lines)


class TrajectoryVariant(BaseModel):
    task_id: str
    variant_id: str
    degradation: str
    severity: float
    trajectory: Trajectory
    edit: str
    introduces_error: bool = True
    len_ratio: float = 1.0

    @property
    def uid(self) -> str:
        return f"{self.task_id}:{self.variant_id}"


class Judgment(BaseModel):
    """One judge decision, with everything needed to audit or re-derive it."""

    uid: str
    judge: str
    config: str
    task: Literal["score", "pairwise"]
    kind: Literal["text", "trajectory"] = "text"
    degradation: str = "none"
    severity: float = 0.0
    score: float | None = None
    verdict: str | None = None
    correct: int | None = None
    raw_text: str = ""
    parsed_ok: bool = True
    model_served: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    simulated: bool = False
    replicate: int = 0
    #: Set when the provider call failed after all retries. The item is excluded
    #: from accuracy rather than silently counted as a miss.
    error: str = ""


__all__ = [
    "Fact",
    "FactKind",
    "Item",
    "Judgment",
    "Pair",
    "Step",
    "Trajectory",
    "TrajectoryVariant",
    "Variant",
]
