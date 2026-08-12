"""Shared HTTP plumbing: retries, backoff, error classification.

Free tiers throttle. A 429 must back off, not fail the run — losing a six-hour
battery to one rate-limit response is the most expensive bug in this repo.
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

from judgeguard.providers.base import (
    Completion,
    ProviderError,
    RateLimitError,
    TransientError,
    approx_tokens,
)
from judgeguard.telemetry import get_logger

log = get_logger(__name__)

RETRY = dict(
    retry=retry_if_exception_type((RateLimitError, TransientError, httpx.TransportError)),
    wait=wait_random_exponential(multiplier=1.5, min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


def _classify(resp: httpx.Response) -> None:
    if resp.status_code == 429:
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
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=body
        )
        _classify(resp)
        data = resp.json()
        dt = (time.perf_counter() - t0) * 1000
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
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
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/models/{model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            json=body,
        )
        _classify(resp)
        data = resp.json()
        dt = (time.perf_counter() - t0) * 1000
        cands = data.get("candidates") or []
        text = ""
        if cands:
            text = "".join(p.get("text", "") for p in cands[0]["content"].get("parts", []))
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            model_requested=model,
            model_served=data.get("modelVersion", model),
            prompt_tokens=usage.get("promptTokenCount", approx_tokens(prompt)),
            completion_tokens=usage.get("candidatesTokenCount", approx_tokens(text)),
            latency_ms=dt,
            provider=self.name,
            finish_reason=(cands[0].get("finishReason", "STOP") if cands else "EMPTY"),
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
        _classify(resp)
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


__all__ = ["GeminiProvider", "OllamaProvider", "OpenAICompatProvider"]
