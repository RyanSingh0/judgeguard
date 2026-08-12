"""The six text degradations.

The methodological core of the project. Instead of paying annotators to say
which of two answers is better, start from a known-good answer and break it in a
controlled way. If a judge cannot rank the reference above a deliberately broken
variant, that is a measured failure with zero annotation cost.

Two rules govern every function here, and both are load-bearing:

1. **Every degradation must be verifiably worse, in one sentence.** If you
   cannot state why a human would prefer the reference, the degradation does not
   belong in the battery. ``Variant.edit`` carries that sentence.
2. **``verbosity`` is the exception and must be applied to the reference.** It
   introduces no error at all -- it only adds length. If a judge scores the
   padded reference *above* the clean reference, that is verbosity bias measured
   directly, in points. ``introduces_error=False`` marks it so no accuracy
   metric ever counts it as a discrimination item.

``severity`` in [0, 1] scales the magnitude of the break. Sweeping it is what
turns "judges are unreliable" into a curve with an x-axis.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from judgeguard.data.schema import Fact, FactKind, Item, Variant

SEVERITIES: tuple[float, ...] = (0.2, 0.5, 0.9)

_FILLER = [
    "It is worth emphasising that these figures were compiled with care and reflect the "
    "underlying record as reported.",
    "Readers should bear in mind that context matters when interpreting results of this kind, "
    "as with any summary of a detailed source.",
    "Taken together, the picture that emerges is one that rewards close attention to the "
    "particulars rather than a hasty reading.",
    "This is, of course, only a summary, and the full record contains the complete detail "
    "behind each of the points above.",
    "It bears repeating that the values above are stated exactly as they appear in the source "
    "material and have not been adjusted.",
    "On the whole, the reported picture is consistent with what the source describes, and the "
    "summary reflects that consistency.",
]

_FABRICATIONS = [
    "An independent replication confirmed every figure above within one percent.",
    "The result was subsequently ranked among the strongest in its category that year.",
    "A follow-up review found no methodological concerns of any kind.",
    "The findings were cited as decisive by the relevant oversight committee.",
    "A parallel cohort produced identical values under blinded conditions.",
    "The reported figures were later adopted as the reference standard for the field.",
]


def _rng(item: Item, degradation: str, severity: float, seed: int) -> random.Random:
    return random.Random(f"{item.id}|{degradation}|{severity}|{seed}")


def _compose(facts: list[Fact]) -> str:
    return " ".join(f.sentence() for f in facts)


def _variant(
    item: Item,
    degradation: str,
    severity: float,
    text: str,
    edit: str,
    *,
    removed: list[str] | None = None,
    corrupted: list[str] | None = None,
    introduces_error: bool = True,
) -> Variant:
    return Variant(
        item_id=item.id,
        variant_id=f"{degradation}@{severity:g}",
        degradation=degradation,
        severity=severity,
        text=text,
        edit=edit,
        facts_removed=removed or [],
        facts_corrupted=corrupted or [],
        introduces_error=introduces_error,
        len_ratio=round(len(text) / max(len(item.reference), 1), 4),
    )


# --------------------------------------------------------------------------- 1
def omission(item: Item, severity: float, *, seed: int = 0, **_: object) -> Variant:
    """Drop required facts. The answer stays fluent; it just stops being complete."""
    rng = _rng(item, "omission", severity, seed)
    n = len(item.facts)
    k = max(1, round(severity * (n - 1)))
    drop = set(rng.sample(range(n), k))
    kept = [f for i, f in enumerate(item.facts) if i not in drop]
    removed = [item.facts[i].key for i in sorted(drop)]
    return _variant(
        item,
        "omission",
        severity,
        _compose(kept),
        f"Removed {k} of {n} required facts ({', '.join(removed)}); the remaining text is "
        f"fluent but no longer answers the question completely.",
        removed=removed,
    )


# --------------------------------------------------------------------------- 2
def fabrication(item: Item, severity: float, *, seed: int = 0, **_: object) -> Variant:
    """Inject plausible falsehoods that the source does not support."""
    rng = _rng(item, "fabrication", severity, seed)
    k = max(1, round(severity * 3))
    picks = rng.sample(_FABRICATIONS, min(k, len(_FABRICATIONS)))
    text = _compose(item.facts) + " " + " ".join(picks)
    return _variant(
        item,
        "fabrication",
        severity,
        text.strip(),
        f"Appended {len(picks)} unsupported claim(s) absent from the source; every original "
        f"fact is intact, so the error is addition rather than loss.",
        corrupted=[f"fabricated_claim_{i}" for i in range(len(picks))],
    )


# --------------------------------------------------------------------------- 3
def numeric_swap(item: Item, severity: float, *, seed: int = 0, **_: object) -> Variant:
    """Perturb numeric values. Fluent, confident, and definitively wrong.

    The most interesting degradation in the battery, because a reference-free
    judge has no way to check the number and the text reads perfectly.
    """
    rng = _rng(item, "numeric_swap", severity, seed)
    nums = [i for i, f in enumerate(item.facts) if f.kind is FactKind.NUMERIC]
    if not nums:
        return hedging(item, severity, seed=seed)
    k = max(1, round(severity * min(3, len(nums))))
    targets = rng.sample(nums, k)
    facts = [f.model_copy(deep=True) for f in item.facts]
    corrupted, details = [], []
    for i in targets:
        f = facts[i]
        try:
            val = float(f.value)
        except ValueError:
            continue
        # Relative perturbation grows with severity: 4% at 0.2, ~55% at 0.9.
        rel = 0.02 + 0.60 * severity**1.5
        sign = rng.choice((-1, 1))
        dp = len(f.value.split(".")[1]) if "." in f.value else 0
        step = 10.0**-dp if dp else 1.0

        def _fmt(x: float, dp: int = dp) -> str:
            return f"{x:.{dp}f}" if dp else str(int(round(x)))

        # The rendered value must actually change. On small numbers a relative
        # perturbation can round straight back to the original string, which
        # would produce a "degradation" identical to the reference and quietly
        # poison that gold pair. Widen until the printed form differs.
        delta = max(abs(val) * rel, step)
        candidate = _fmt(val + delta * sign)
        guard = 0
        while candidate == f.value and guard < 12:
            delta += step
            candidate = _fmt(val + delta * sign)
            guard += 1
        if candidate == f.value:
            continue
        f.value = candidate
        corrupted.append(f.key)
        details.append(f"{f.key}: {item.facts[i].value} -> {f.value}")
    return _variant(
        item,
        "numeric_swap",
        severity,
        _compose(facts),
        f"Altered {len(corrupted)} numeric value(s) [{'; '.join(details)}]; the prose is "
        f"unchanged and reads as confidently as the reference.",
        corrupted=corrupted,
    )


# --------------------------------------------------------------------------- 4
def verbosity(item: Item, severity: float, *, seed: int = 0, **_: object) -> Variant:
    """THE BIAS PROBE. Pad the reference. Add length, add nothing else.

    ``introduces_error=False``: this variant is not worse, it is merely longer.
    A judge that raises its score has revealed verbosity bias in points.
    """
    rng = _rng(item, "verbosity", severity, seed)
    target_ratio = 1.5 + 1.5 * severity  # 1.5x .. 3.0x
    body = item.reference
    pool = _FILLER[:]
    rng.shuffle(pool)
    out = [body]
    cur = len(body)
    i = 0
    while cur < target_ratio * len(body) and i < 40:
        s = pool[i % len(pool)]
        out.append(s)
        cur += len(s) + 1
        i += 1
    text = " ".join(out)
    return _variant(
        item,
        "verbosity",
        severity,
        text,
        f"Padded the unmodified reference to {len(text) / max(len(body), 1):.2f}x length with "
        f"content-free filler. No fact was added, removed or altered.",
        introduces_error=False,
    )


# --------------------------------------------------------------------------- 5
def topic_drift(
    item: Item, severity: float, *, seed: int = 0, neighbours: list[Item] | None = None, **_: object
) -> Variant:
    """Answer a neighbouring question instead of the one that was asked."""
    rng = _rng(item, "topic_drift", severity, seed)
    pool = [n for n in (neighbours or []) if n.id != item.id and n.domain == item.domain]
    if not pool:
        pool = [n for n in (neighbours or []) if n.id != item.id]
    n = len(item.facts)
    k = max(1, round(severity * n))
    if not pool:
        kept = item.facts[: n - k]
        return _variant(
            item,
            "topic_drift",
            severity,
            _compose(kept) + " The remainder addresses a different question.",
            f"Replaced {k} clause(s) with off-question content.",
        )
    # Substituted clauses must actually differ. Two items in the same domain can
    # coincide on a value, and a "degradation" that silently returns the
    # reference verbatim would poison the gold label for that pair. The eval
    # gate checks for exactly this, and this loop is why it passes.
    facts = list(item.facts)
    swapped: set[int] = set()
    for other in rng.sample(pool, len(pool)):
        for i in rng.sample(range(n), n):
            if i in swapped:
                continue
            cand = other.facts[i % len(other.facts)]
            if cand.sentence() != item.facts[i].sentence():
                facts[i] = cand
                swapped.add(i)
            if len(swapped) >= k:
                break
        if len(swapped) >= k:
            break

    if not swapped:  # pathological pool: degrade by truncation
        return _variant(
            item,
            "topic_drift",
            severity,
            _compose(item.facts[: max(1, n - k)])
            + " The remainder addresses a different question.",
            f"No distinguishable substitute clause was available, so {k} clause(s) were replaced "
            f"with off-question filler instead.",
            corrupted=[f.key for f in item.facts[max(1, n - k) :]],
        )

    return _variant(
        item,
        "topic_drift",
        severity,
        _compose(facts),
        f"Substituted {len(swapped)} of {n} clauses with material answering a different "
        f"question (source item {other.id}); the answer drifts off the asked question.",
        corrupted=sorted(item.facts[i].key for i in swapped),
    )


# --------------------------------------------------------------------------- 6
_VAGUE = [
    "a modest amount",
    "a fairly typical level",
    "a reasonably high figure",
    "something in the usual range",
    "a number of units",
    "a noticeable share",
]


def hedging(item: Item, severity: float, *, seed: int = 0, **_: object) -> Variant:
    """Replace committed claims with vagueness. Nothing false, nothing useful."""
    rng = _rng(item, "hedging", severity, seed)
    n = len(item.facts)
    k = max(1, round(severity * n))
    targets = set(rng.sample(range(n), min(k, n)))
    facts = [f.model_copy(deep=True) for f in item.facts]
    hedged = []
    for i in sorted(targets):
        facts[i].value = rng.choice(_VAGUE)
        facts[i].unit = ""
        facts[i].clause = "It is generally understood that this was " + facts[
            i
        ].clause.lower().replace("the ", "", 1)
        hedged.append(item.facts[i].key)
    return _variant(
        item,
        "hedging",
        severity,
        _compose(facts),
        f"Replaced {len(hedged)} committed value(s) ({', '.join(hedged)}) with vague quantifiers; "
        f"the answer asserts nothing checkable.",
        corrupted=hedged,
    )


DEGRADATIONS: dict[str, Callable[..., Variant]] = {
    "omission": omission,
    "fabrication": fabrication,
    "numeric_swap": numeric_swap,
    "verbosity": verbosity,
    "topic_drift": topic_drift,
    "hedging": hedging,
}

#: Degradations that introduce a real error (used for discrimination accuracy).
ERROR_DEGRADATIONS = [k for k in DEGRADATIONS if k != "verbosity"]

#: Degradations that are pure bias probes (no error introduced).
PROBE_DEGRADATIONS = ["verbosity"]


def build_variants(
    items: list[Item],
    *,
    severities: tuple[float, ...] = SEVERITIES,
    seed: int = 20260731,
    degradations: list[str] | None = None,
) -> list[Variant]:
    """Full cross-product: items x degradation types x severity levels."""
    names = degradations or list(DEGRADATIONS)
    out: list[Variant] = []
    for item in items:
        for name in names:
            fn = DEGRADATIONS[name]
            for sev in severities:
                out.append(fn(item, sev, seed=seed, neighbours=items))
    return out


__all__ = [
    "DEGRADATIONS",
    "ERROR_DEGRADATIONS",
    "PROBE_DEGRADATIONS",
    "SEVERITIES",
    "build_variants",
    "fabrication",
    "hedging",
    "numeric_swap",
    "omission",
    "topic_drift",
    "verbosity",
]
