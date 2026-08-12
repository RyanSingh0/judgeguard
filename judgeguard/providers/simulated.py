"""Deterministic simulator standing in for a real judge.

Why this exists
---------------
A portfolio repo that only runs if the reader has four API keys does not get
run. A CI eval-gate that needs a live quota is not a gate. So the harness ships
with a generative model of judge behaviour, parameterised entirely in
``configs/simulator.yaml``, that produces responses in the exact wire format the
real judge parser consumes. Every code path -- prompting, parsing, retries,
caching, statistics, distillation, serving -- is exercised identically whether
the bytes came from Google or from this file.

What it is not
--------------
It is not evidence. Artefacts produced here carry ``provenance.mode ==
"simulated"``, figures are watermarked, and the README says so above the fold.
The parameters are priors to be overwritten by measurement, not findings.

The generative model
--------------------
Each candidate answer gets a latent perceived quality::

    latent = base
           + verbosity_pull * log2(len_ratio)      # length, independent of content
           + self_pref * 1[generator family == judge family]
           - penalty_scale * severity * detectability[degradation] * sensitivity
           + N(0, noise_sd)                        # deterministic given the seed

Discrimination failure, verbosity bias, self-enhancement bias and
position-order inconsistency are then *emergent* consequences of that single
equation rather than hand-written outcomes -- which is the only way the harness
can be said to measure anything at all, even in simulation.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from judgeguard.config import CONFIG_DIR
from judgeguard.providers.base import Completion, approx_tokens


@lru_cache(maxsize=1)
def load_profile(path: Path | None = None) -> dict[str, Any]:
    path = path or (CONFIG_DIR / "simulator.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _unit_normal(*parts: Any) -> float:
    """Deterministic standard-normal draw keyed on the arguments.

    Same inputs always give the same draw, so a rerun of the whole battery is
    bit-identical -- which is what makes the CI gate meaningful.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    u1 = int.from_bytes(h[0:8], "big") / 2**64
    u2 = int.from_bytes(h[8:16], "big") / 2**64
    u1 = min(max(u1, 1e-12), 1 - 1e-12)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class SimulatedProvider:
    """Speaks the LLMProvider protocol; reads ground truth only from ``meta``."""

    name = "simulated"

    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        self.profile = profile or load_profile()

    # ------------------------------------------------------------------ core
    def _judge_params(self, judge: str, config: str) -> dict[str, float]:
        p = self.profile
        jp = dict(p["judges"].get(judge, p["judges"]["llama70b"]))
        cp = p["configs"].get(config, p["configs"]["rubric"])
        jp["sensitivity"] *= cp.get("sensitivity", 1.0)
        jp["verbosity_pull"] *= cp.get("verbosity_pull", 1.0)
        jp["noise_sd"] *= cp.get("noise_sd", 1.0)
        return jp

    def _detect(self, kind: str, degradation: str | None) -> float:
        if not degradation or degradation in ("none", "reference"):
            return 0.0
        table = self.profile["detectability"].get(kind, {})
        return float(table.get(degradation, 0.40))

    def latent(
        self,
        *,
        judge: str,
        config: str,
        kind: str = "text",
        degradation: str | None = None,
        severity: float = 0.0,
        len_ratio: float = 1.0,
        same_family: bool = False,
        noise_key: tuple[Any, ...] = (),
        temperature: float = 0.0,
    ) -> float:
        s = self.profile["scoring"]
        jp = self._judge_params(judge, config)
        latent = float(s["reference_base"])
        latent += jp["verbosity_pull"] * math.log2(max(len_ratio, 1e-3))
        if same_family:
            latent += jp["self_pref"]
        latent -= (
            float(s["penalty_scale"])
            * float(severity)
            * self._detect(kind, degradation)
            * jp["sensitivity"]
        )
        # Deterministic idiosyncratic noise. Temperature widens it, and the
        # replicate index enters the key so repeats at temp>0 actually differ.
        sd = jp["noise_sd"] * (0.60 + 0.30 * min(max(temperature, 0.0), 1.0) / 0.7)
        latent += sd * _unit_normal(judge, config, kind, degradation, severity, *noise_key)
        return latent

    def _clip(self, x: float) -> float:
        s = self.profile["scoring"]
        return max(float(s["scale_min"]), min(float(s["scale_max"]), x))

    # -------------------------------------------------------------- protocol
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
        meta = meta or {}
        judge = meta.get("judge", model)
        config = meta.get("config", "rubric")
        task = meta.get("task", "score")
        kind = meta.get("kind", "text")

        if task == "pairwise":
            text = self._pairwise_text(meta, judge, config, kind, temperature)
        else:
            text = self._score_text(meta, judge, config, kind, temperature)

        lat = self.profile["latency"].get(judge, {"ttft_ms": 300, "tok_per_s": 200, "jitter": 0.25})
        out_tok = approx_tokens(text)
        jitter = 1.0 + lat["jitter"] * _unit_normal("lat", judge, meta.get("uid", ""), temperature)
        latency_ms = max(15.0, (lat["ttft_ms"] + 1000.0 * out_tok / lat["tok_per_s"]) * jitter)

        return Completion(
            text=text,
            model_requested=model,
            model_served=f"{model}-sim{self.profile['version']}",
            prompt_tokens=approx_tokens(prompt),
            completion_tokens=out_tok,
            latency_ms=latency_ms,
            provider=self.name,
            simulated=True,
        )

    # ------------------------------------------------------------- renderers
    def _score_text(
        self, meta: dict[str, Any], judge: str, config: str, kind: str, temperature: float
    ) -> str:
        score = self._clip(
            self.latent(
                judge=judge,
                config=config,
                kind=kind,
                degradation=meta.get("degradation"),
                severity=float(meta.get("severity", 0.0)),
                len_ratio=float(meta.get("len_ratio", 1.0)),
                same_family=bool(meta.get("same_family", False)),
                noise_key=(meta.get("uid", ""), meta.get("replicate", 0)),
                temperature=temperature,
            )
        )
        score = round(score, 1)
        if config == "cot":
            return (
                "REASONING: I check the answer against the question's requirements, look for "
                "unsupported claims, verify internal consistency of any figures, and discount "
                "length that carries no additional content.\n"
                f"SCORE: {score}"
            )
        if config == "vague":
            return f"{score}"
        return f"SCORE: {score}\nJUSTIFICATION: Assessed against the rubric bands."

    def _pairwise_text(
        self, meta: dict[str, Any], judge: str, config: str, kind: str, temperature: float
    ) -> str:
        s = self.profile["scoring"]
        jp = self._judge_params(judge, config)
        first_is_ref = bool(meta.get("first_is_reference", True))
        common = dict(
            judge=judge,
            config=config,
            kind=kind,
            same_family=bool(meta.get("same_family", False)),
            temperature=temperature,
        )
        ref_latent = self.latent(
            **common,
            degradation=None,
            severity=0.0,
            len_ratio=1.0,
            noise_key=(meta.get("uid", ""), "ref", meta.get("replicate", 0)),
        )
        deg_latent = self.latent(
            **common,
            degradation=meta.get("degradation"),
            severity=float(meta.get("severity", 0.0)),
            len_ratio=float(meta.get("len_ratio", 1.0)),
            noise_key=(meta.get("uid", ""), "deg", meta.get("replicate", 0)),
        )
        first, second = (ref_latent, deg_latent) if first_is_ref else (deg_latent, ref_latent)
        # The bias itself: whatever is shown first gets a bonus.
        first += jp["position_pull"]
        diff = first - second
        if abs(diff) < float(s["tie_threshold"]):
            verdict = "tie"
        else:
            verdict = "A" if diff > 0 else "B"
        if config == "cot":
            return f"REASONING: Comparing the two candidates on the rubric criteria.\nVERDICT: {verdict}"
        return f"VERDICT: {verdict}"


__all__ = ["SimulatedProvider", "load_profile"]
