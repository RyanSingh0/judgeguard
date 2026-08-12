"""Provider protocol.

The contract every backend implements. Two fields carry more weight than they
look like they should:

* ``model_served`` — what actually answered, which is not always what we asked
  for. Providers silently alias and retire model IDs; an experiment record that
  only stores the requested ID is not reproducible.
* ``meta`` on ``complete()`` — non-semantic call metadata. Real HTTP providers
  ignore it entirely. Only the deterministic simulator reads it, which is how
  the whole pipeline runs end-to-end with zero API keys without contaminating
  the prompts that real judges see.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Completion(BaseModel):
    text: str
    model_requested: str
    model_served: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    cached: bool = False
    simulated: bool = False
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Completion: ...


class ProviderError(RuntimeError):
    """Non-retryable provider failure."""


class RateLimitError(ProviderError):
    """429 / quota exhausted. Retryable with backoff."""


class TransientError(ProviderError):
    """5xx, timeout, connection reset. Retryable."""


def approx_tokens(text: str) -> int:
    """Cheap token estimate. Used only when a provider omits usage counts."""
    return max(1, len(text) // 4)


__all__ = [
    "Completion",
    "LLMProvider",
    "ProviderError",
    "RateLimitError",
    "TransientError",
    "approx_tokens",
]
