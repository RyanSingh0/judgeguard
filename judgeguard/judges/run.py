"""Batch judge execution and the position-swap protocol.

Every pairwise comparison runs in both orderings. The disagreement rate between
them is the position-bias metric itself, not a stand-in for it. Accuracy gets
reported with and without swap-and-average so the cost of the usual mitigation
is measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from judgeguard.config import get_settings, load_registry
from judgeguard.data.schema import Item, Judgment
from judgeguard.judges.parse import parse_score, parse_verdict
from judgeguard.judges.prompts import SYSTEM, pairwise_prompt, score_prompt
from judgeguard.providers.limits import QuotaExhaustedError
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
    ``correct_single_random``  one ordering picked by a seeded coin flip. The
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


# Each provider spells "you hit the token cap" differently. OpenAI-compatible
# APIs say length, Gemini says MAX_TOKENS, and an empty Gemini candidate comes
# back as EMPTY.
_TRUNCATED = {"length", "max_tokens", "empty"}


def _truncated(finish_reason: str | None) -> bool:
    return (finish_reason or "").strip().lower() in _TRUNCATED


# A live battery is tens of thousands of calls over hours. One malformed
# response should cost one data point, not the whole run. But a systemic failure
# like a revoked key or a retired model still has to stop things, rather than
# quietly writing out a results file full of holes.
MAX_FAILURE_RATE = 0.20
MIN_FAILURES_BEFORE_ABORT = 25


class BatchAbortedError(RuntimeError):
    """Too many calls failed; the run is not worth continuing."""


class JudgeUnavailableError(RuntimeError):
    """This judge is out of daily quota. Skip it, the others are fine.

    Killing the whole run because one of four judges hit its limit throws away
    the three that didn't. The cache makes it free to fill the gap in tomorrow.
    """


class _Failures:
    """Per-judge failure counters.

    This used to be a plain module dict written from 8 worker threads.
    `d[k] += 1` is load/add/store in CPython so concurrent increments lose
    counts, and these counts decide whether to abort. It also pooled all four
    judges into one denominator, so a judge failing every call could stay under
    the 20% threshold because its peers were healthy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_judge: dict[str, dict[str, Any]] = {}

    def _slot(self, judge: str) -> dict[str, Any]:
        return self._by_judge.setdefault(
            judge, {"failed": 0, "total": 0, "errors": Counter(), "quota_exhausted": False}
        )

    def record(self, judge: str, exc: BaseException | None) -> tuple[int, int]:
        with self._lock:
            s = self._slot(judge)
            s["total"] += 1
            if exc is not None:
                s["failed"] += 1
                s["errors"][type(exc).__name__] += 1
            return s["failed"], s["total"]

    def mark_quota(self, judge: str) -> None:
        with self._lock:
            self._slot(judge)["quota_exhausted"] = True

    def reset(self) -> None:
        with self._lock:
            self._by_judge.clear()

    def report(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {"by_judge": {}}
            tf = tt = 0
            for j, s in self._by_judge.items():
                tf += s["failed"]
                tt += s["total"]
                out["by_judge"][j] = {
                    "failed_calls": s["failed"],
                    "total_calls": s["total"],
                    "failure_rate": s["failed"] / s["total"] if s["total"] else 0.0,
                    "errors": dict(s["errors"]),
                    "quota_exhausted": s["quota_exhausted"],
                }
            out["failed_calls"] = tf
            out["total_calls"] = tt
            out["failure_rate"] = tf / tt if tt else 0.0
            return out


FAILURES = _Failures()


def _map(fn: Any, jobs: list[Any], workers: int | None = None, judge: str = "?") -> list[Any]:
    """Run `fn` over `jobs`, tolerating individual failures.

    Failures come back as (None, error_string), counted per judge. The caller
    turns them into judgements carrying the error so the results can tell a
    dead HTTP call apart from a judge that answered unparseably. The old version
    lumped both into parse_failure_rate.
    """
    workers = workers or get_settings().max_concurrency
    stop = threading.Event()

    def guarded(job: Any) -> tuple[Any, str]:
        if stop.is_set():
            return None, "skipped: judge unavailable"
        try:
            out = fn(job)
        except QuotaExhaustedError as exc:
            FAILURES.mark_quota(judge)
            FAILURES.record(judge, exc)
            stop.set()
            return None, f"QuotaExhaustedError: {exc}"[:300]
        except Exception as exc:
            failed, total = FAILURES.record(judge, exc)
            log.warning(
                "call_failed",
                judge=judge,
                error=f"{type(exc).__name__}: {exc}"[:300],
                uid=getattr(job, "uid", "?"),
                failures=failed,
            )
            if failed >= MIN_FAILURES_BEFORE_ABORT and failed / max(total, 1) > MAX_FAILURE_RATE:
                stop.set()
                raise BatchAbortedError(
                    f"{judge}: {failed} of {total} calls failed "
                    f"(> {MAX_FAILURE_RATE:.0%}). Last error: {type(exc).__name__}: {exc}"
                ) from exc
            return None, f"{type(exc).__name__}: {exc}"[:300]
        FAILURES.record(judge, None)
        return out, ""

    if workers <= 1 or len(jobs) <= 1:
        results = [guarded(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(guarded, jobs))
    if stop.is_set() and FAILURES.report()["by_judge"].get(judge, {}).get("quota_exhausted"):
        raise JudgeUnavailableError(
            f"{judge}: daily quota exhausted after "
            f"{FAILURES.report()['by_judge'][judge]['total_calls']} calls this run. "
            f"Re-run tomorrow; completed work is cached."
        )
    return results


def failure_report() -> dict[str, Any]:
    """Per-judge failure accounting, written into every results file."""
    return FAILURES.report()


def reset_failures() -> None:
    FAILURES.reset()


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
        # Reasoning models need room to think before they emit SCORE:.
        # qwen3.6-27b returns "" at 128 tokens and parses fine at 512.
        #
        # The cot base is 512, not 320. Checked live on 2026-08-13:
        # llama-3.3-70b spends ~380 tokens reasoning before writing SCORE:, so
        # at 320 every call truncated with finish_reason="length" and parsed
        # nothing. That reads as a judge that can't follow the format, and I
        # would have reported it as one.
        budget = resolve(judge).token_budget(512 if config == "cot" else 160)
        c = complete(
            judge,
            prompt,
            temperature=temperature,
            max_tokens=budget,
            system=SYSTEM,
            meta=_meta(t, judge, config, replicate, task="score"),
        )
        score = parse_score(c.text)
        # One retry at double the budget. A judge cut off mid-sentence has told
        # us about our token cap, not about its judgement. Capped at one retry:
        # if it still can't finish at 2x, that's a genuine format failure and
        # worth reporting as one.
        if score is None and _truncated(c.finish_reason):
            log.warning("budget_retry", judge=judge, config=config, budget=budget, uid=t.uid)
            c = complete(
                judge,
                prompt,
                temperature=temperature,
                max_tokens=budget * 2,
                system=SYSTEM,
                meta=_meta(t, judge, config, replicate, task="score", retry="budget"),
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

    out = _map(one, tasks, judge=judge)
    return [
        j
        if j is not None
        else Judgment(
            uid=t.uid,
            judge=judge,
            config=config,
            task="score",
            kind=t.kind,
            degradation=t.degradation,
            severity=t.severity,
            score=None,
            parsed_ok=False,
            raw_text="<call failed>",
            # Lets downstream analysis separate "the transport broke" from
            # "the judge said something unparseable". Only the second is a
            # property of the judge and belongs in a parse-failure rate.
            error=err,
            replicate=replicate,
        )
        for (j, err), t in zip(out, tasks, strict=True)
    ]


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

    out = _map(one, tasks, judge=judge)
    return [
        o
        if o is not None
        else PairOutcome(
            uid=t.uid,
            judge=judge,
            config=config,
            kind=t.kind,
            degradation=t.degradation,
            severity=t.severity,
            forward=None,
            reverse=None,
            consistent=False,
            verdict="inconsistent",
            correct_ref_first=0,
            correct_deg_first=0,
            correct_single_random=0,
            correct_swapped=0,
            latency_ms=0.0,
            cost_usd=0.0,
            tokens=0,
            parsed_ok=False,
        )
        for (o, _err), t in zip(out, tasks, strict=True)
    ]


def panel() -> list[str]:
    return list(load_registry().panel)


__all__ = [
    "BatchAbortedError",
    "JudgeUnavailableError",
    "PairOutcome",
    "PairTask",
    "ScoreTask",
    "failure_report",
    "panel",
    "reset_failures",
    "run_pairwise",
    "run_scores",
]
