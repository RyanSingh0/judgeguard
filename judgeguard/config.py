"""Central configuration.

Two ideas matter here:

1. `provider_mode` decides whether the harness talks to real APIs or to the
   deterministic simulator. Every artefact the pipeline writes is stamped with
   the mode that produced it, so a simulated number can never be mistaken for a
   measured one.
2. Model IDs live in `configs/models.yaml`, never in code. Providers retire
   models without notice; the config is the record of what we asked for and
   `Completion.model_served` is the record of what actually answered.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class ProviderMode(str, Enum):
    SIMULATED = "simulated"
    LIVE = "live"
    AUTO = "auto"


class Settings(BaseSettings):
    """Environment-driven settings. See `.env.example`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    provider_mode: ProviderMode = Field(
        default=ProviderMode.SIMULATED, alias="JUDGEGUARD_PROVIDER_MODE"
    )
    cache_dir: Path = Field(default=Path(".cache/llm"), alias="JUDGEGUARD_CACHE_DIR")
    results_dir: Path = Field(default=Path("results"), alias="JUDGEGUARD_RESULTS_DIR")
    max_concurrency: int = Field(default=4, alias="JUDGEGUARD_MAX_CONCURRENCY")
    seed: int = Field(default=20260731, alias="JUDGEGUARD_SEED")
    guard_p99_budget_ms: float = Field(default=150.0, alias="JUDGEGUARD_GUARD_P99_BUDGET_MS")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    otel_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    # ---------------------------------------------------------------- helpers
    def key_for(self, provider: str) -> str:
        return {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "cerebras": self.cerebras_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": "local",
        }.get(provider, "")

    def any_live_key(self) -> bool:
        return any(self.key_for(p) for p in ("gemini", "groq", "cerebras", "openrouter"))

    def effective_mode(self) -> ProviderMode:
        if self.provider_mode is ProviderMode.AUTO:
            return ProviderMode.LIVE if self.any_live_key() else ProviderMode.SIMULATED
        return self.provider_mode

    @property
    def abs_results_dir(self) -> Path:
        p = self.results_dir
        return p if p.is_absolute() else REPO_ROOT / p


class ModelSpec(BaseModel):
    id: str
    alias: str
    family: str
    tier: str
    provider: str = ""
    #: Emits a visible thinking block before committing to an answer. Judged at a
    #: small max_tokens these models spend the entire budget reasoning and return
    #: an empty string -- which looks like a broken provider, not a token limit.
    reasoning: bool = False
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0
    list_usd_per_mtok_in: float = 0.0
    list_usd_per_mtok_out: float = 0.0

    def token_budget(self, base: int) -> int:
        """Scale the output budget for models that think before they answer.

        Measured live on 2026-07-31: qwen3.6-27b spent 1,175 output tokens on a
        single rubric judgement, of which ~1,100 were the thinking block. At 128
        tokens it returns an empty string and at 640 it truncates mid-thought --
        both of which read as a broken provider rather than a token cap. A judge
        that looks broken because of our own budget is a measurement error, not a
        model property.
        """
        return base * 10 if self.reasoning else base

    def cost_usd(
        self, prompt_tokens: int, completion_tokens: int, *, list_price: bool = True
    ) -> float:
        """Cost of one call.

        `list_price=True` uses published list prices even when the call was
        served on a free tier. Reporting free-tier cost as $0 makes the
        cost/accuracy analysis meaningless, so the headline economics use list
        price and say so.
        """
        cin = self.list_usd_per_mtok_in if list_price else self.usd_per_mtok_in
        cout = self.list_usd_per_mtok_out if list_price else self.usd_per_mtok_out
        return (prompt_tokens * cin + completion_tokens * cout) / 1_000_000


class ProviderSpec(BaseModel):
    name: str
    base_url: str
    env_key: str | None = None
    models: list[ModelSpec] = []


class ModelRegistry(BaseModel):
    providers: dict[str, ProviderSpec]
    panel: list[str]
    generators: list[str]

    def by_alias(self, alias: str) -> ModelSpec:
        for spec in self.providers.values():
            for m in spec.models:
                if m.alias == alias:
                    return m
        raise KeyError(f"unknown model alias: {alias!r}")

    def provider_of(self, alias: str) -> str:
        return self.by_alias(alias).provider

    @property
    def aliases(self) -> list[str]:
        return [m.alias for s in self.providers.values() for m in s.models]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


@lru_cache(maxsize=1)
def load_registry(path: Path | None = None) -> ModelRegistry:
    path = path or (CONFIG_DIR / "models.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    providers: dict[str, ProviderSpec] = {}
    for name, body in raw["providers"].items():
        models = [ModelSpec(provider=name, **m) for m in body.get("models", [])]
        providers[name] = ProviderSpec(
            name=name,
            base_url=_expand(body.get("base_url", "")),
            env_key=body.get("env_key"),
            models=models,
        )
    return ModelRegistry(providers=providers, panel=raw["panel"], generators=raw["generators"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "REPO_ROOT",
    "ModelRegistry",
    "ModelSpec",
    "ProviderMode",
    "ProviderSpec",
    "Settings",
    "get_settings",
    "load_registry",
]
