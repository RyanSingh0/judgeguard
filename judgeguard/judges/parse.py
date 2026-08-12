"""Robust extraction of scores and verdicts from free-form judge text.

Parse failures are a measurement, not an inconvenience: a judge that cannot
follow a two-line output format is telling you something about its reliability.
So the parse rate is recorded per judge and per config and reported, rather than
quietly retried until it works.
"""

from __future__ import annotations

import re

#: Reasoning models wrap their scratchpad in a think block. A number inside it is
#: not the verdict -- stripping it first is the difference between reading a
#: model's conclusion and reading its second thoughts.
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.I | re.S)
_UNCLOSED_THINK = re.compile(r"<(think|thinking|reasoning)>.*", re.I | re.S)


def strip_reasoning(text: str) -> str:
    """Remove thinking blocks, unless that would leave nothing to parse."""
    if not text:
        return text
    stripped = _UNCLOSED_THINK.sub("", _THINK.sub("", text)).strip()
    return stripped or text


_SCORE_PATTERNS = [
    re.compile(r"SCORE\s*[:=]\s*\**\s*(-?\d+(?:\.\d+)?)", re.I),
    re.compile(r"\b(?:rating|score)\b\D{0,12}(-?\d+(?:\.\d+)?)\s*/\s*10", re.I),
    re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:/\s*10)?\s*$", re.M),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:out of|/)\s*10", re.I),
]

_VERDICT_PATTERNS = [
    re.compile(r"VERDICT\s*[:=]\s*\**\s*(A|B|TIE)\b", re.I),
    re.compile(r"\b(?:answer|trajectory|option)\s+(A|B)\s+is\s+better", re.I),
    re.compile(r"^\s*(A|B|tie)\s*$", re.I | re.M),
]


def parse_score(text: str, *, lo: float = 1.0, hi: float = 10.0) -> float | None:
    """First plausible 1-10 score, or None if the judge did not produce one."""
    if not text:
        return None
    text = strip_reasoning(text)
    for pat in _SCORE_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
            except (TypeError, ValueError):
                continue
            if lo - 1e-9 <= v <= hi + 1e-9:
                return v
    return None


def parse_verdict(text: str) -> str | None:
    """Return 'A', 'B', 'tie' or None."""
    if not text:
        return None
    text = strip_reasoning(text)
    for pat in _VERDICT_PATTERNS:
        m = pat.search(text)
        if m:
            v = m.group(1).upper()
            return "tie" if v == "TIE" else v
    return None


__all__ = ["parse_score", "parse_verdict", "strip_reasoning"]
