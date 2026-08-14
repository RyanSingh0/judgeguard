"""Tests for rate limiting and quota accounting.

The bug these guard against cost a six-hour battery: a daily quota was retried
as if it were a per-minute rate limit. The distinction is not cosmetic, so it
gets tests.
"""

from __future__ import annotations

import threading

import pytest

from judgeguard.providers.limits import (
    Limiter,
    QuotaExhaustedError,
    _Bucket,
    _duration,
    parse_openai_headers,
)


class TestDuration:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1m26.4s", 86.4),
            ("185ms", 0.185),
            ("2.5s", 2.5),
            ("1h30m", 5400.0),
            ("", None),
            (None, None),
        ],
    )
    def test_parses_provider_duration_strings(self, text, expected):
        assert _duration(text) == expected


class TestHeaderParsing:
    def test_groq_daily_request_budget_is_not_mistaken_for_rpm(self):
        """Groq reports a 1,000/day budget as limit=1000, reset=86.4s per slot.

        Read naively that looks like 1,000 requests per minute -- three orders of
        magnitude of headroom that does not exist. The refill interval is what
        disambiguates it.
        """
        limits = parse_openai_headers(
            {
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-reset-requests": "1m26.4s",
                "x-ratelimit-reset-tokens": "90ms",
            }
        )
        assert limits["rpd"] == 1000
        assert limits["rpm"] is None
        assert limits["tpm"] == 8000

    def test_short_reset_interval_reads_as_per_minute(self):
        limits = parse_openai_headers(
            {"x-ratelimit-limit-requests": "60", "x-ratelimit-reset-requests": "1s"}
        )
        assert limits["rpm"] == 60
        assert limits["rpd"] is None

    def test_missing_headers_are_not_invented(self):
        assert parse_openai_headers({}) == {"rpm": None, "tpm": None, "rpd": None}


class TestBucket:
    def test_daily_budget_raises_rather_than_sleeping(self):
        b = _Bucket(model="groq:llama", rpd=3)
        for _ in range(3):
            b.acquire(10)
        with pytest.raises(QuotaExhaustedError) as exc:
            b.acquire(10)
        assert "3 requests spent" in str(exc.value)

    def test_provider_verdict_overrides_our_counter(self):
        b = _Bucket(model="gemini:flash")
        b.acquire(10)
        b.mark_exhausted()
        with pytest.raises(QuotaExhaustedError):
            b.acquire(10)

    def test_counters_survive_concurrent_increment(self):
        """`day_requests += 1` from N threads must not lose counts.

        The previous failure counter was a bare dict mutated from the worker
        pool. Read-modify-write on a dict value is three bytecodes in CPython,
        and the count it produced gated the abort decision.
        """
        b = _Bucket(model="x", rpd=10_000)
        threads = [
            threading.Thread(target=lambda: [b.acquire(1) for _ in range(50)]) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert b.day_requests == 400


class TestLimiterPersistence:
    def test_spend_survives_a_restart(self, tmp_path):
        """A battery spread over several days must not forget yesterday."""
        path = tmp_path / "limits.json"
        a = Limiter(path)
        b = a.bucket("groq", "llama-3.3-70b-versatile")
        b.observe(rpm=None, tpm=12000, rpd=1000)
        for _ in range(5):
            b.acquire(100)
        a.save()

        revived = Limiter(path)
        snap = revived.bucket("groq", "llama-3.3-70b-versatile").snapshot()
        assert snap["day_requests"] == 5
        assert snap["rpd"] == 1000
        assert snap["remaining_today"] == 995

    def test_corrupt_state_does_not_break_a_run(self, tmp_path):
        path = tmp_path / "limits.json"
        path.write_text("{not json", encoding="utf-8")
        assert Limiter(path).report() == {}


class TestConfigSeeding:
    """Config fills gaps; the provider always wins."""

    def test_seed_only_fills_unknown_limits(self):
        b = _Bucket(model="gemini:flash-lite")
        b.seed(rpm=15, tpm=250_000, rpd=None)
        assert (b.rpm, b.tpm, b.rpd) == (15, 250_000, None)

    def test_observed_headers_are_not_overwritten_by_config(self):
        """Groq reports its own limits; models.yaml must not clobber them.

        A YAML file is a snapshot taken by a human on some past date. A response
        header is what the provider believes right now. When they disagree the
        header is right, and the pinned-model-ID incident in this repo is the
        standing reminder of what happens when config is trusted over reality.
        """
        b = _Bucket(model="groq:qwen")
        b.observe(rpm=None, tpm=8000, rpd=1000)
        b.seed(rpm=None, tpm=999_999, rpd=999_999)
        assert b.tpm == 8000
        assert b.rpd == 1000

    def test_gemini_would_be_unpaced_without_seeding(self):
        """Google's generateContent returns no x-ratelimit-* headers at all.

        Verified against a live 200 response. So header-learned limits leave the
        bucket empty and the limiter never sleeps -- which at Gemini's 5 RPM free
        tier means a 429 on essentially every call.
        """
        b = _Bucket(model="gemini:3.6-flash")
        assert parse_openai_headers({}) == {"rpm": None, "tpm": None, "rpd": None}
        b.observe(**parse_openai_headers({}))  # type: ignore[arg-type]
        assert b.rpm is None
        b.seed(rpm=5, tpm=250_000, rpd=None)
        assert b.rpm == 5


class TestReservationSizing:
    """Reserving max_tokens throttles against tokens nobody spends."""

    def test_falls_back_to_full_budget_without_history(self):
        b = _Bucket(model="groq:qwen")
        assert b.estimate(400, 5120) == 5520

    def test_uses_observed_p90_once_there_is_history(self):
        # No tpm: this exercises the estimate, not the pacing, and a real 8,000
        # TPM ceiling would make the loop below sleep for minutes.
        b = _Bucket(model="groq:qwen")
        for _ in range(10):
            r = b.estimate(400, 5120)
            b.acquire(r)
            b.settle(r, 2000)  # what qwen actually spends on a cot judgement
        # 2000 * 1.15 + 400, not 400 + 5120: ~2.4x more calls per minute.
        assert b.estimate(400, 5120) < 3000

    def test_estimate_never_exceeds_what_the_call_could_use(self):
        b = _Bucket(model="x")
        for _ in range(10):
            r = b.estimate(50, 4096)
            b.acquire(r)
            b.settle(r, 4096)
        assert b.estimate(50, 200) <= 250

    def test_settle_returns_the_unused_reservation_to_the_window(self):
        b = _Bucket(model="x", tpm=10_000)
        r = b.estimate(100, 5000)
        b.acquire(r)
        assert b.day_tokens == 5100
        b.settle(r, 900)
        assert b.day_tokens == 900
        assert sum(n for _, n in b._tok_times) == 900
