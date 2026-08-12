"""Batch judge execution, and the position-swap protocol.

The swap protocol is non-negotiable for pairwise judging. Every comparison is
run in both orderings and the disagreement rate between them *is* the
position-bias metric -- not a proxy for it, the thing itself. Accuracy is then
reported with and without swap-and-average so the standard mitigation can be
priced rather than assumed.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from judgeguard.config import get_settings, load_registry
from judgeguard.data.schema import Item, Judgment
from judgeguard.judges.parse import parse_score, parse_verdict
from judgeguard.judges.prompts import SYSTEM, pairwise_prompt, score_prompt
from judgeguard.providers.registry import complete, resolve
from judgeguard.telemetry import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ScoreTask:
    uid: str
    item: Item
    answer: str
    kind: str = "text"
    degradation: str = "none"
    severity: float = 0.0
    len_ratio: float = 1.0
    generator_family: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PairTask:
    uid: str
    item: Item
    reference: str
    degraded: str
    kind: str = "text"
    degradation: str = "none"
    severity: float = 0.0
    len_ratio: float = 1.0
    generator_family: str = ""


@dataclass(slots=True)
class PairOutcome:
    """One comparison, judged in both orderings.

    Four accuracy views, because which one you report changes the story:

    ``correct_ref_first``      reference shown as option A. This is the number a
                               naive harness reports, and position bias inflates
                               it whenever the reference is systematically first.
    ``correct_deg_first``      the same comparison with the order flipped. The
                               gap against the previous line is the inflation.
    ``correct_single_random``  one ordering chosen by a seeded coin flip -- the
                               unbiased single-call estimate.
    ``correct_swapped``        swap-and-average: correct only if both orderings
                               agree *and* agree on the reference. An ordering
                               disagreement is scored as a miss rather than
                               quietly broken in the judge's favour.
    """

    uid: str
    judge: str
    config: str
    kind: str
    degradation: str
    severity: float
    forward: str | None  # verdict with REF shown first: "REF"|"DEG"|"tie"|None
    reverse: str | None  # verdict with DEG shown first, normalised back
    consistent: bool
    verdict: str  # "REF" | "DEG" | "tie" | "inconsistent"
    correct_ref_first: int
    correct_deg_first: int
    correct_single_random: int
    correct_swapped: int
    latency_ms: float
    cost_usd: float
    tokens: int
    parsed_ok: bool


def _meta(
    t: ScoreTask | PairTask, judge: str, config: str, replicate: int, **more: Any
) -> dict[str, Any]:
    fam = resolve(judge).family
    return {
        "uid": t.uid,
        "judge": judge,
        "config": config,
        "kind": t.kind,
        "degradation": t.degradation,
        "severity": t.severity,
        "len_ratio": t.len_ratio,
        "same_family": bool(t.generator_family) and t.generator_family == fam,
        "replicate": replicate,
        **more,
    }


def _cost(judge: str, prompt_tokens: int, completion_tokens: int) -> float:
    return resolve(judge).cost_usd(prompt_tokens, completion_tokens)


def _map(fn: Any, jobs: list[Any], workers: int | None = None) -> list[Any]:
    workers = workers or get_settings().max_concurrency
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs))


# ------------------------------------------------------------------ pointwise
def run_scores(
    judge: str,
    config: str,
    tasks: list[ScoreTask],
    *,
    temperature: float = 0.0,
    replicate: int = 0,
) -> list[Judgment]:
    def one(t: ScoreTask) -> Judgment:
        prompt = score_prompt(config, t.item, t.answer, kind=t.kind)
        # Reasoning models need room to think before they will emit SCORE:.
        # Measured live: qwen3.6-27b returns "" at 128 tokens and parses at 512.
        budget = resolve(judge).token_budget(320 if config == "cot" else 160)
        c = complete(
            judge,
            prompt,
            temperature=temperature,
            max_tokens=budget,
            system=SYSTEM,
            meta=_meta(t, judge, config, replicate, task="score"),
        )
        score = parse_score(c.text)
        return Judgment(
            uid=t.uid,
            judge=judge,
            config=config,
            task="score",
            kind=t.kind,
            degradation=t.degradation,
            severity=t.severity,
            score=score,
            raw_text=c.text,
            parsed_ok=score is not None,
            model_served=c.model_served,
            prompt_tokens=c.prompt_tokens,
            completion_tokens=c.completion_tokens,
            latency_ms=c.latency_ms,
            cost_usd=_cost(judge, c.prompt_tokens, c.completion_tokens),
            simulated=c.simulated,
            replicate=replicate,
        )

    return _map(one, tasks)


# ------------------------------------------------------------------- pairwise
_FWD = {"A": "REF", "B": "DEG", "tie": "tie"}
_REV = {"A": "DEG", "B": "REF", "tie": "tie"}


def run_pairwise(
    judge: str,
    config: str = "pairwise",
    tasks: list[PairTask] | None = None,
    *,
    temperature: float = 0.0,
    replicate: int = 0,
) -> list[PairOutcome]:
    """Run every comparison in both orderings.

    ``correct_forward``  - accuracy using only the first ordering (the naive
                           protocol most projects ship).
    ``correct_swapped``  - accuracy after swap-and-average, where an ordering
                           disagreement is scored as a miss rather than quietly
                           resolved. The gap between the two is what the
                           mitigation actually buys.
    """
    tasks = tasks or []

    def one(t: PairTask) -> PairOutcome:
        p_fwd = pairwise_prompt(config, t.item, t.reference, t.degraded, kind=t.kind)
        p_rev = pairwise_prompt(config, t.item, t.degraded, t.reference, kind=t.kind)
        budget = resolve(judge).token_budget(224)
        c_fwd = complete(
            judge,
            p_fwd,
            temperature=temperature,
            max_tokens=budget,
            system=SYSTEM,
            meta=_meta(t, judge, config, replicate, task="pairwise", first_is_reference=True),
        )
        c_rev = complete(
            judge,
            p_rev,
            temperature=temperature,
            max_tokens=budget,
            system=SYSTEM,
            meta=_meta(t, judge, config, replicate, task="pairwise", first_is_reference=False),
        )
        v_fwd_raw, v_rev_raw = parse_verdict(c_fwd.text), parse_verdict(c_rev.text)
        fwd = _FWD.get(v_fwd_raw) if v_fwd_raw else None
        rev = _REV.get(v_rev_raw) if v_rev_raw else None
        consistent = fwd is not None and fwd == rev
        verdict = fwd if consistent else "inconsistent"
        # Seeded coin flip, so the "one call only" estimate is reproducible.
        pick_first = (
            int(hashlib.sha256(f"{t.uid}|{judge}|{config}|{replicate}".encode()).hexdigest(), 16)
            % 2
        ) == 0
        single = fwd if pick_first else rev
        tokens = (
            c_fwd.prompt_tokens
            + c_fwd.completion_tokens
            + c_rev.prompt_tokens
            + c_rev.completion_tokens
        )
        return PairOutcome(
            uid=t.uid,
            judge=judge,
            config=config,
            kind=t.kind,
            degradation=t.degradation,
            severity=t.severity,
            forward=fwd,
            reverse=rev,
            consistent=consistent,
            verdict=verdict or "inconsistent",
            correct_ref_first=int(fwd == "REF"),
            correct_deg_first=int(rev == "REF"),
            correct_single_random=int(single == "REF"),
            correct_swapped=int(consistent and fwd == "REF"),
            latency_ms=c_fwd.latency_ms + c_rev.latency_ms,
            cost_usd=_cost(judge, c_fwd.prompt_tokens, c_fwd.completion_tokens)
            + _cost(judge, c_rev.prompt_tokens, c_rev.completion_tokens),
            tokens=tokens,
            parsed_ok=fwd is not None and rev is not None,
        )

    return _map(one, tasks)


def panel() -> list[str]:
    return list(load_registry().panel)


__all__ = ["PairOutcome", "PairTask", "ScoreTask", "panel", "run_pairwise", "run_scores"]
