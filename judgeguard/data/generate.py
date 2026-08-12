"""Candidate-response generation for the self-enhancement probe.

Self-enhancement bias only exists if different model families actually produce
the text being judged. Re-labelling identical text with a different "author"
measures nothing, because a judge has no author field to read -- the bias comes
from stylistic self-similarity.

In live mode each generator model is asked to answer the item, and the resulting
responses are what the panel scores. In simulated mode a family-specific,
deterministic rewrite stands in: each family has its own characteristic defect
profile and verbosity habit, so the panel is scoring genuinely different text of
genuinely different quality.

The analysis is leave-one-out precisely because the responses differ in quality:
a judge preferring its own family must be separated from its family simply
writing better answers. See ``experiments/04_self_enhancement.py``.
"""

from __future__ import annotations

from judgeguard.data.schema import Item
from judgeguard.degrade.text import hedging, omission, verbosity
from judgeguard.judges.prompts import SYSTEM
from judgeguard.providers.registry import complete, is_simulated, resolve

#: Deterministic stylistic profile per family, used only in simulated mode.
_FAMILY_STYLE: dict[str, tuple[str, float]] = {
    "gemini": ("clean", 0.0),
    "llama": ("verbose", 0.35),
    "qwen": ("hedged", 0.30),
    "gpt-oss": ("lossy", 0.35),
    "mistral": ("lossy", 0.35),
    "nemotron": ("hedged", 0.40),
    "glm": ("verbose", 0.28),
}


def generate_response(alias: str, item: Item, *, seed: int = 0) -> str:
    """The answer model ``alias`` gives for ``item``."""
    if not is_simulated(alias):
        prompt = (
            f"SOURCE CONTEXT:\n{item.context}\n\nQUESTION:\n{item.question}\n\n"
            "Answer using only the source context. Be complete and specific. "
            "Do not add information the context does not contain."
        )
        return complete(alias, prompt, temperature=0.2, max_tokens=400, system=SYSTEM).text.strip()

    family = resolve(alias).family
    if family not in _FAMILY_STYLE:
        # Two generators sharing the "clean" default emit byte-identical text,
        # which makes the leave-one-out self-enhancement estimator degenerate and
        # silently reports a 0.00 effect. Fail loudly instead.
        raise KeyError(
            f"no simulated style for family {family!r}; add one to _FAMILY_STYLE "
            f"or the self-enhancement probe will be degenerate for {alias}"
        )
    style, sev = _FAMILY_STYLE[family]
    if style == "clean" or sev == 0.0:
        return item.reference
    if style == "verbose":
        return verbosity(item, sev, seed=seed).text
    if style == "hedged":
        return hedging(item, sev, seed=seed).text
    return omission(item, sev, seed=seed).text


__all__ = ["generate_response"]
