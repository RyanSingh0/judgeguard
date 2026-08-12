"""Disk cache for provider calls.

Definition of done for the provider phase: rerunning the entire battery costs
zero API calls. That property is what makes it safe to iterate on analysis code
without burning a free-tier quota you cannot get back.

The key covers everything that can change the answer: provider, model, prompt,
system prompt, temperature, max_tokens, and the simulator profile version.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from diskcache import Cache

from judgeguard.config import REPO_ROOT, get_settings
from judgeguard.providers.base import Completion

_settings = get_settings()
_cache_path = (
    _settings.cache_dir if _settings.cache_dir.is_absolute() else REPO_ROOT / _settings.cache_dir
)


class _MemoryCache(dict):
    """Fallback when the cache directory cannot host a SQLite file.

    Network mounts, read-only containers and some CI sandboxes cannot run
    SQLite's WAL. Refusing to start in those environments would be the wrong
    trade -- the cache is an optimisation, not a correctness requirement -- so
    the harness degrades to process memory and says so in ``cache_stats()``.
    """

    backend = "memory"

    def clear(self) -> None:  # pragma: no cover - trivial
        super().clear()


try:
    _cache: Any = Cache(str(_cache_path), size_limit=int(4e9))
    _BACKEND = "diskcache"
except Exception:  # sqlite3.OperationalError and friends
    _cache = _MemoryCache()
    _BACKEND = "memory"

STATS = {"hits": 0, "misses": 0}


def cache_key(provider: str, model: str, prompt: str, **params: Any) -> str:
    blob = json.dumps(
        {"p": provider, "m": model, "prompt": prompt, **params}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cached_complete(
    provider: Any,
    prompt: str,
    *,
    model: str,
    use_cache: bool = True,
    **params: Any,
) -> Completion:
    meta = params.pop("meta", None)
    # The replicate index belongs in the key. At temperature > 0 the five reruns
    # of a self-consistency experiment are five independent samples, and folding
    # them onto one cache entry would silently report perfect self-agreement.
    replicate = (meta or {}).get("replicate", 0)
    key = cache_key(provider.name, model, prompt, _replicate=replicate, **params)
    if use_cache and key in _cache:
        STATS["hits"] += 1
        hit: Completion = Completion.model_validate(_cache[key])
        return hit.model_copy(update={"cached": True})
    STATS["misses"] += 1
    result = provider.complete(prompt, model=model, meta=meta, **params)
    if use_cache:
        _cache[key] = result.model_dump()
    return result


def cache_stats() -> dict[str, Any]:
    total = STATS["hits"] + STATS["misses"]
    return {
        **STATS,
        "hit_rate": (STATS["hits"] / total) if total else 0.0,
        "entries": len(_cache),
        "path": str(_cache_path),
        "backend": _BACKEND,
    }


def clear_cache() -> None:
    _cache.clear()


__all__ = ["cache_key", "cache_stats", "cached_complete", "clear_cache"]
