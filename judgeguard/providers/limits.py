"""Client-side rate limiting and daily quota tracking.

My first live run died after 33 calls on a Gemini 429 that I'd treated as a
temporary rate limit. It was the daily quota. Two problems:

- Nothing was pacing the client. 8 threads firing as fast as the pool allowed.
  Groq's free tier gives qwen3.6-27b 8,000 tokens/min and a reasoning judgement
  costs about 2,000, so ~4 calls/min is the real ceiling.
- The quota got retried like a rate limit: 6 attempts, backoff up to 60s, then
  failed anyway. Wasted about 3 minutes per call on something that couldn't
  have worked until midnight Pacific.

So: pace requests against whatever limits the provider reports, and remember
what's been spent today so a resumed run knows where it stands.

Notes to self:

- Limits are learned from response headers where possible, not hardcoded. Groq
  and OpenRouter send x-ratelimit-* on every response. Gemini sends nothing, so
  that one comes from models.yaml.
- Daily counters persist to disk, keyed on the reset boundary (midnight Pacific
  for Gemini, rolling 86,400s for Groq). A run spread over several days picks up
  where it left off.
- One lock per bucket. The worker threads both read and increment these, and
  `d[k] += 1` isn't atomic in CPython.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from judgeguard.telemetry import get_logger

log = get_logger(__name__)

# Gemini's daily quota resets at midnight Pacific. Using -8 year-round means
# the reset lands up to an hour late during PDT. That's the safe way to be
# wrong: it never assumes quota we don't have.
_PACIFIC_OFFSET = timedelta(hours=-8)


def _pacific_day() -> str:
    return (datetime.now(UTC) + _PACIFIC_OFFSET).strftime("%Y-%m-%d")


class QuotaExhaustedError(RuntimeError):
    """Daily budget spent. Retrying today won't help.

    Kept separate from RateLimitError. A rate limit means slow down; a quota
    means stop and come back tomorrow. Treating them the same is what crashed
    the first run.
    """

    def __init__(self, model: str, spent: int, limit: int | None, resets: str) -> None:
        self.model = model
        self.spent = spent
        self.limit = limit
        self.resets = resets
        super().__init__(
            f"daily quota exhausted for {model}: {spent} requests spent"
            + (f" of {limit}" if limit else "")
            + f"; resets {resets}"
        )


@dataclass
class _Bucket:
    """Limits and spend for one (provider, model) pair.

    rpm/tpm are per-minute ceilings, enforced by sleeping. rpd is a daily
    ceiling, enforced by raising. None means we haven't seen a value yet and
    nothing is enforced; better to run the first call unpaced than guess.
    """

    model: str
    rpm: float | None = None
    tpm: float | None = None
    rpd: int | None = None
    _req_times: list[float] = field(default_factory=list)
    _tok_times: list[tuple[float, int]] = field(default_factory=list)
    # Recent per-call token totals, used to size the next reservation.
    _observed: list[int] = field(default_factory=list)
    day: str = field(default_factory=_pacific_day)
    # Set when the provider itself said the allowance is gone. Day-scoped, so it
    # clears on its own at the reset boundary.
    exhausted_day: str = ""
    day_requests: int = 0
    day_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _roll_day(self) -> None:
        today = _pacific_day()
        if today != self.day:
            self.day = today
            self.day_requests = 0
            self.day_tokens = 0

    def _prune(self, now: float) -> None:
        cut = now - 60.0
        self._req_times = [t for t in self._req_times if t > cut]
        self._tok_times = [(t, n) for t, n in self._tok_times if t > cut]

    def estimate(self, prompt_tokens: int, max_completion: int) -> int:
        """How many tokens to hold for a call we haven't made yet.

        Reserving max_tokens looks safe and costs a lot. qwen3.6-27b gets a
        5,120 token cot budget and actually spends 1,300-2,300. Against 8,000
        TPM that's ~1 call/min instead of ~3, which over a 4,500-call phase adds
        up to days.

        Once there's enough history, reserve the 90th percentile of what this
        model actually spends. Before that, use the full budget. Every call
        settles against real usage afterwards so a run of long answers still
        slows things down.
        """
        with self.lock:
            if len(self._observed) >= 8:
                ranked = sorted(self._observed)
                p90 = ranked[min(len(ranked) - 1, int(0.9 * len(ranked)))]
                # Never reserve more than the caller could possibly use.
                return int(prompt_tokens + min(max_completion, p90 * 1.15))
        return prompt_tokens + max_completion

    def settle(self, reserved: int, actual: int) -> None:
        """Swap a reservation for what the call actually cost.

        Without this the per-minute window stays inflated and we throttle
        against tokens nobody spent.
        """
        with self.lock:
            self._observed.append(actual)
            if len(self._observed) > 64:
                del self._observed[:-64]
            for i in range(len(self._tok_times) - 1, -1, -1):
                ts, n = self._tok_times[i]
                if n == reserved:
                    self._tok_times[i] = (ts, actual)
                    self.day_tokens += actual - reserved
                    return

    def acquire(self, est_tokens: int, *, reserve: float = 0.9) -> None:
        """Block until this call fits inside the per-minute limits.

        `reserve` holds us at 90% of the published ceiling. Our window boundary
        and theirs won't line up exactly, and running 10% slow costs much less
        than a burst of 429s.
        """
        deadline = time.monotonic() + 300.0
        while True:
            with self.lock:
                self._roll_day()
                if self.exhausted_day == self.day:
                    raise QuotaExhaustedError(
                        self.model, self.day_requests, None, "midnight Pacific"
                    )
                if self.rpd is not None and self.day_requests >= self.rpd:
                    raise QuotaExhaustedError(
                        self.model, self.day_requests, self.rpd, "midnight Pacific"
                    )
                now = time.monotonic()
                self._prune(now)
                wait = 0.0
                if self.rpm and len(self._req_times) >= self.rpm * reserve:
                    wait = max(wait, 60.0 - (now - self._req_times[0]) + 0.05)
                if self.tpm:
                    used = sum(n for _, n in self._tok_times)
                    if used + est_tokens > self.tpm * reserve and self._tok_times:
                        wait = max(wait, 60.0 - (now - self._tok_times[0][0]) + 0.05)
                if wait <= 0:
                    self._req_times.append(now)
                    self._tok_times.append((now, est_tokens))
                    self.day_requests += 1
                    self.day_tokens += est_tokens
                    return
            if time.monotonic() > deadline:
                # Five minutes of waiting on a per-minute window means the
                # limits we learned are wrong, not that we're busy. Go ahead and
                # let the provider's 429 decide.
                log.warning("limiter_stuck", model=self.model, rpm=self.rpm, tpm=self.tpm)
                return
            time.sleep(min(wait, 5.0))

    def observe(self, *, rpm: float | None, tpm: float | None, rpd: int | None) -> None:
        """Limits the provider reported. These win over anything in config."""
        with self.lock:
            if rpm:
                self.rpm = rpm
            if tpm:
                self.tpm = tpm
            if rpd:
                self.rpd = rpd

    def seed(self, *, rpm: float | None, tpm: float | None, rpd: int | None) -> None:
        """Limits from models.yaml, applied only where we know nothing.

        Gemini's generateContent sends no x-ratelimit-* headers at all (checked
        against a live 200). So a header-only limiter never paces it, and at 5
        RPM on the free tier that's a 429 on nearly every call.

        Config fills the gap but never overrides. Whatever the provider says at
        runtime wins; the YAML is just a snapshot I took on some past date.
        """
        with self.lock:
            if rpm and self.rpm is None:
                self.rpm = rpm
            if tpm and self.tpm is None:
                self.tpm = tpm
            if rpd and self.rpd is None:
                self.rpd = rpd

    def mark_exhausted(self) -> None:
        """Provider says today's allowance is gone.

        Flags the day, and deliberately does NOT write rpd. The count we hit the
        wall at is only "what was left when this run started", not the published
        cap. On 2026-08-14 gemini-flash-lite stopped at 284 because preflight and
        some manual testing had already spent part of the day. Recording 284 as
        the permanent limit would have capped every future run at 284 even if the
        real allowance is higher.
        """
        with self.lock:
            self._roll_day()
            self.exhausted_day = self.day

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._roll_day()
            return {
                "model": self.model,
                "rpm": self.rpm,
                "tpm": self.tpm,
                "rpd": self.rpd,
                "day": self.day,
                "day_requests": self.day_requests,
                "day_tokens": self.day_tokens,
                "exhausted_day": self.exhausted_day,
                "exhausted_today": self.exhausted_day == self.day,
                "remaining_today": 0
                if self.exhausted_day == self.day
                else ((self.rpd - self.day_requests) if self.rpd else None),
            }


class Limiter:
    """Process-wide registry of buckets, persisted across runs."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._load()

    def bucket(self, provider: str, model: str) -> _Bucket:
        key = f"{provider}:{model}"
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(model=key)
            return b

    # ------------------------------------------------------------- persistence
    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:  # corrupt state is not worth failing a run over
            return
        for key, d in raw.items():
            b = _Bucket(model=key)
            b.rpm, b.tpm, b.rpd = d.get("rpm"), d.get("tpm"), d.get("rpd")
            b.day = d.get("day", _pacific_day())
            b.exhausted_day = d.get("exhausted_day", "")
            b.day_requests = d.get("day_requests", 0)
            b.day_tokens = d.get("day_tokens", 0)
            b._roll_day()
            self._buckets[key] = b

    def save(self) -> None:
        with self._lock:
            snap = {k: v.snapshot() for k, v in self._buckets.items()}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {k: v.snapshot() for k, v in self._buckets.items()}


def parse_openai_headers(headers: Any) -> dict[str, float | int | None]:
    """Read Groq / OpenRouter x-ratelimit-* headers.

    Groq reports the daily request budget as a refill interval, not a plain
    number. One request against a 1,000 limit comes back as
    `x-ratelimit-reset-requests: 1m26.4s`, and 86,400 / 1,000 = 86.4. So a reset
    interval that's a big fraction of a day means limit-requests is per day, not
    per minute. Read it the other way and we'd pace at 1,000 requests a minute
    against a daily cap.
    """

    def _f(name: str) -> float | None:
        v = headers.get(name)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    lim_req = _f("x-ratelimit-limit-requests")
    lim_tok = _f("x-ratelimit-limit-tokens")
    reset_req = _duration(headers.get("x-ratelimit-reset-requests"))
    reset_tok = _duration(headers.get("x-ratelimit-reset-tokens"))

    rpm = tpm = None
    rpd = None
    if lim_req is not None:
        # Refill of one slot takes reset_req seconds => window = limit * reset.
        window = (reset_req or 0) * lim_req
        if window > 3600:
            rpd = int(lim_req)
        else:
            rpm = lim_req
    if lim_tok is not None:
        window_t = reset_tok or 0
        # tokens reset is reported as time-to-refill-what-was-just-used, so it is
        # tiny; treat limit-tokens as per-minute, which is what Groq documents.
        tpm = lim_tok if window_t < 60 else None
    return {"rpm": rpm, "tpm": tpm, "rpd": rpd}


def _duration(text: Any) -> float | None:
    """Parse Groq's ``1m26.4s`` / ``185ms`` / ``2.5s`` duration strings."""
    if not isinstance(text, str) or not text:
        return None
    s = text.strip()
    total = 0.0
    num = ""
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isdigit() or ch == ".":
            num += ch
            i += 1
            continue
        if s[i : i + 2] == "ms":
            total += float(num or 0) / 1000.0
            num = ""
            i += 2
            continue
        if ch == "s":
            total += float(num or 0)
            num = ""
        elif ch == "m":
            total += float(num or 0) * 60.0
            num = ""
        elif ch == "h":
            total += float(num or 0) * 3600.0
            num = ""
        i += 1
    return total or None


__all__ = ["Limiter", "QuotaExhaustedError", "parse_openai_headers"]
