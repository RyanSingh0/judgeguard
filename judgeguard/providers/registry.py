"""Alias -> live provider (or the simulator), plus the one call everything uses.

Downstream code never imports a concrete provider. It calls
``complete("llama70b", prompt, meta=...)`` and this module decides whether that
becomes an HTTPS request to Groq or a draw from the simulator. Swapping the
whole panel from simulated to live is one environment variable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from judgeguard.config import ModelSpec, ProviderMode, get_settings, load_registry
from judgeguard.providers.base import Completion
from judgeguard.providers.cache import cached_complete
from judgeguard.providers.http_base import GeminiProvider, OllamaProvider, OpenAICompatProvider
from judgeguard.providers.simulated import SimulatedProvider
from judgeguard.telemetry import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _simulator() -> SimulatedProvider:
    return SimulatedProvider()


@lru_cache(maxsize=16)
def _live_provider(provider_name: str) -> Any:
    s = get_settings()
    reg = load_registry()
    spec = reg.providers[provider_name]
    key = s.key_for(provider_name)
    if provider_name == "gemini":
        return GeminiProvider(spec.base_url, key)
    if provider_name == "ollama":
        return OllamaProvider(s.ollama_base_url)
    return OpenAICompatProvider(provider_name, spec.base_url, key)


def resolve(alias: str) -> ModelSpec:
    return load_registry().by_alias(alias)


def is_simulated(alias: str) -> bool:
    """True when this alias will be served by the simulator."""
    s = get_settings()
    mode = s.effective_mode()
    if mode is ProviderMode.SIMULATED:
        return True
    spec = resolve(alias)
    if spec.provider == "ollama":
        return False
    return not s.key_for(spec.provider)


def mode_banner() -> str:
    s = get_settings()
    mode = s.effective_mode().value
    live = [a for a in load_registry().panel if not is_simulated(a)]
    return f"provider_mode={mode} live_models={live or 'none'}"


def complete(
    alias: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    system: str | None = None,
    meta: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> Completion:
    """The single entry point for every model call in the project."""
    spec = resolve(alias)
    meta = {**(meta or {}), "judge": alias}
    if is_simulated(alias):
        provider: Any = _simulator()
        model_id = spec.id
    else:
        provider = _live_provider(spec.provider)
        model_id = spec.id
    return cached_complete(
        provider,
        prompt,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system,
        meta=meta,
        use_cache=use_cache,
    )


__all__ = ["complete", "is_simulated", "mode_banner", "resolve"]
