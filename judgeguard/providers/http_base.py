"""HTTP plumbing shared by the providers: pacing, retries, error classification.

Free tiers throttle in two ways and they need different handling. A rate limit
means "too fast" and the same call works after a wait. A quota means "that's
your lot for today" and waiting does nothing.

Both come back as a 429, which is how the first live run died. `_classify`
tells them apart now, and the limiter paces against whatever ceilings the
provider publishes, so the rate-limit path rarely comes up.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from judgeguard.config import REPO_ROOT, get_settings
from judgeguard.providers.base import (
    Completion,
    ProviderError,
    RateLimitError,
    TransientError,
    approx_tokens,
)
from judgeguard.providers.limits import Limiter, QuotaExhaustedError, parse_openai_headers
from judgeguard.telemetry import get_logger

log = get_logger(__name__)

_s = get_settings()
_state = (_s.cache_dir if _s.cache_dir.is_absolute() else REPO_ROOT / _s.cache_dir) / "limits.json"
LIMITER = Limiter(_state)

# QuotaExhaustedError is left out on purpose. It isn't retryable, and putting
# it here would bring back the bug this module exists to fix.
RETRY = dict(
    retry=retry_if_exception_type((RateLimitError, TransientError, httpx.TransportError)),
    wait=wait_random_exponential(multiplier=1.5, min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)

# Substrings that mark a 429 as a spent allowance rather than a burst limit.
_QUOTA_MARKERS = (
    "exceeded your current quota",
    "perday",
    "per day",
    "requests per day",
    "quota_metric",
    "quotafailure",
    "daily",
    "insufficient_quota",
)


def _retry_after(resp: httpx.Response) -> float | None:
    """Use the provider's own retry hint if it gave one, rather than guessing."""
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    try:
        body = resp.json()
    except Exception:
        return None
    for d in (body.get("error", {}) or {}).get("details", []) or []:
        delay = d.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                pass
    return None


def _is_quota(resp: httpx.Response) -> bool:
    """Tell "slow down" apart from "come back tomorrow".

    Google puts it in the error body: QuotaFailure details, or a message naming
    a PerDay metric. OpenAI-compatible providers put it in the headers, where
    zero remaining requests plus a reset measured in hours means a daily
    allowance rather than a per-minute window.
    """
    text = resp.text[:2000].lower()
    if any(m in text for m in _QUOTA_MARKERS):
        return True
    remaining = resp.headers.get("x-ratelimit-remaining-requests")
    if remaining is not None:
        try:
            if float(remaining) <= 0:
                from judgeguard.providers.limits import _duration

                reset = _duration(resp.headers.get("x-ratelimit-reset-requests")) or 0
                return reset > 900
        except ValueError:
            pass
    return False


def _classify(resp: httpx.Response, *, model: str = "?") -> None:
    if resp.status_code == 429:
        if _is_quota(resp):
            raise QuotaExhaustedError(model, 0, None, "provider reports the allowance is spent")
        wait = _retry_after(resp)
        if wait:
            time.sleep(min(wait, 60.0))
        raise RateLimitError(f"429 rate limited: {resp.text[:300]}")
    if resp.status_code >= 500:
        raise TransientError(f"{resp.status_code}: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise ProviderError(f"{resp.status_code}: {resp.text[:300]}")


class OpenAICompatProvider:
    """Groq, Cerebras and OpenRouter all speak the OpenAI chat-completions API."""

    def __init__(self, name: str, base_url: str, api_key: str, timeout: float = 90.0) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.name == "openrouter":
            h["HTTP-Referer"] = "https://github.com/RyanSingh0/judgeguard"
            h["X-Title"] = "JudgeGuard"
        return h

    @retry(**RETRY)  # type: ignore[arg-type]
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        bucket = LIMITER.bucket(self.name, model)
        reserved = bucket.estimate(approx_tokens(prompt) + approx_tokens(system or ""), max_tokens)
        bucket.acquire(reserved)
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=body
        )
        limits = parse_openai_headers(resp.headers)
        bucket.observe(rpm=limits["rpm"], tpm=limits["tpm"], rpd=limits["rpd"])  # type: ignore[arg-type]
        try:
            _classify(resp, model=f"{self.name}:{model}")
        except QuotaExhaustedError:
            bucket.mark_exhausted()
            LIMITER.save()
            raise
        data = resp.json()
        dt = (time.perf_counter() - t0) * 1000
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        bucket.settle(
            reserved,
            usage.get("total_tokens")
            or (
                usage.get("prompt_tokens", approx_tokens(prompt))
                + usage.get("completion_tokens", approx_tokens(text))
            ),
        )
        return Completion(
            text=text,
            model_requested=model,
            model_served=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", approx_tokens(prompt)),
            completion_tokens=usage.get("completion_tokens", approx_tokens(text)),
            latency_ms=dt,
            provider=self.name,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )


class GeminiProvider:
    """Google AI Studio generateContent endpoint."""

    name = "gemini"

    def __init__(self, base_url: str, api_key: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    @retry(**RETRY)  # type: ignore[arg-type]
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        bucket = LIMITER.bucket(self.name, model)
        reserved = bucket.estimate(approx_tokens(prompt) + approx_tokens(system or ""), max_tokens)
        bucket.acquire(reserved)
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/models/{model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            json=body,
        )
        try:
            _classify(resp, model=f"gemini:{model}")
        except QuotaExhaustedError:
            bucket.mark_exhausted()
            LIMITER.save()
            raise
        data = resp.json()
        dt = (time.perf_counter() - t0) * 1000
        # A Gemini candidate doesn't always carry content. If a reasoning model
        # burns maxOutputTokens on its thinking block, or a safety filter stops
        # the response, you get a finishReason and no "content" key. Indexing
        # that blindly raises KeyError in a worker thread and kills the run.
        cands = data.get("candidates") or []
        text = ""
        finish = "EMPTY"
        if cands:
            finish = cands[0].get("finishReason", "STOP")
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts)
        elif data.get("promptFeedback", {}).get("blockReason"):
            finish = "BLOCKED:" + data["promptFeedback"]["blockReason"]
        usage = data.get("usageMetadata") or {}
        # Gemini counts thinking tokens separately from candidate tokens and
        # bills for both. On a real judge call I got candidatesTokenCount=5 and
        # thoughtsTokenCount=208. Reading only the first understates a reasoning
        # judge's output by about 40x, and that feeds straight into exp 07.
        completion = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        bucket.settle(
            reserved,
            usage.get("totalTokenCount")
            or (usage.get("promptTokenCount", approx_tokens(prompt)) + completion),
        )
        return Completion(
            text=text,
            model_requested=model,
            model_served=data.get("modelVersion", model),
            prompt_tokens=usage.get("promptTokenCount", approx_tokens(prompt)),
            completion_tokens=completion or approx_tokens(text),
            latency_ms=dt,
            provider=self.name,
            finish_reason=finish,
        )


class OllamaProvider:
    """Local models. No key, no quota, your hardware is the limit."""

    name = "ollama"

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    @retry(**RETRY)  # type: ignore[arg-type]
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Completion:
        body = {
            "model": model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/api/generate", json=body)
        _classify(resp, model=f"ollama:{model}")
        data = resp.json()
        dt = (time.perf_counter() - t0) * 1000
        text = data.get("response", "")
        return Completion(
            text=text,
            model_requested=model,
            model_served=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count", approx_tokens(prompt)),
            completion_tokens=data.get("eval_count", approx_tokens(text)),
            latency_ms=dt,
            provider=self.name,
        )


__all__ = [
    "LIMITER",
    "GeminiProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "QuotaExhaustedError",
]
