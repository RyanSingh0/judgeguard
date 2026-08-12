"""Featurisation for the student model.

Two backends, and the choice between them is a latency decision rather than an
accuracy one:

* ``hashing``  -- word + character n-gram hashing plus a small block of
  hand-designed structural features. Pure scikit-learn, no torch, ~1 ms per
  example on CPU. This is what the inline guardrail actually runs.
* ``minilm``   -- sentence-transformers ``all-MiniLM-L6-v2``. Better semantics,
  but the encode step alone spends more of the 150 ms p99 budget than the entire
  hashing pipeline. Available via ``pip install judgeguard[embeddings]``.

Benchmarking both and then shipping the cheaper one *because the budget said so*
is the point of the exercise.

The structural block is deliberately interpretable: hedge-word density, digit
density, filler-phrase count, type-token ratio. When the student blocks a
response, these are what let you say why.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

HEDGE_WORDS = {
    "generally",
    "roughly",
    "approximately",
    "somewhat",
    "modest",
    "typical",
    "usual",
    "noticeable",
    "some",
    "several",
    "reasonably",
    "fairly",
    "around",
    "understood",
    "believed",
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
}

FILLER_PHRASES = [
    "it is worth",
    "it bears repeating",
    "readers should",
    "taken together",
    "on the whole",
    "of course",
    "as with any",
    "close attention",
]

_WORD = re.compile(r"[a-zA-Z']+")
_NUM = re.compile(r"\d")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "for",
    "on",
    "at",
    "by",
    "with",
    "was",
    "were",
    "is",
    "are",
    "be",
    "been",
    "that",
    "this",
    "it",
    "its",
    "as",
    "from",
}


def structural_features(text: str) -> np.ndarray:
    """Twelve interpretable signals. Cheap enough to be free at request time."""
    words = _WORD.findall(text.lower())
    n = max(len(words), 1)
    chars = max(len(text), 1)
    uniq = len(set(words))
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    hedges = sum(1 for w in words if w in HEDGE_WORDS)
    filler = sum(text.lower().count(p) for p in FILLER_PHRASES)
    digits = len(_NUM.findall(text))
    return np.array(
        [
            np.log1p(len(text)),
            np.log1p(len(words)),
            uniq / n,  # type-token ratio
            hedges / n,  # hedge density
            filler,  # content-free filler count
            digits / chars,  # numeric density
            text.count(",") / sentences,
            n / sentences,  # mean sentence length
            float(any(c.isupper() for c in text[:1])),
            np.log1p(sentences),
            sum(1 for w in words if len(w) > 9) / n,
            float(bool(re.search(r"\bp\s*=\s*0", text))),
        ],
        dtype=np.float32,
    )


class HashingFeaturizer:
    """Word + char n-gram hashing, concatenated with the structural block."""

    name = "hashing"

    def __init__(self, n_word: int = 2**17, n_char: int = 2**16) -> None:
        self.word = HashingVectorizer(
            n_features=n_word,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
            lowercase=True,
            dtype=np.float32,
        )
        self.char = HashingVectorizer(
            n_features=n_char,
            alternate_sign=False,
            analyzer="char_wb",
            ngram_range=(3, 4),
            norm="l2",
            lowercase=True,
            dtype=np.float32,
        )

    @property
    def dim(self) -> int:
        return self.word.n_features + self.char.n_features + 18

    def transform(self, texts: list[str]) -> Any:
        parts = [split_example(t) for t in texts]
        answers = [a for _, _, a in parts]
        w = self.word.transform(answers)
        c = self.char.transform(answers)
        s = sparse.csr_matrix(
            np.vstack(
                [
                    np.concatenate([structural_features(a), coverage_features(ctx, q, a)])
                    for ctx, q, a in parts
                ]
            )
        )
        return sparse.hstack([w, c, s], format="csr")

    fit_transform = transform


class MiniLMFeaturizer:
    """Dense sentence embeddings. Optional dependency, materially slower."""

    name = "minilm"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(model)

    @property
    def dim(self) -> int:
        return int(self.encoder.get_sentence_embedding_dimension()) + 18

    def transform(self, texts: list[str]) -> Any:
        parts = [split_example(t) for t in texts]
        emb = self.encoder.encode(
            [a for _, _, a in parts],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        struct = np.vstack(
            [
                np.concatenate([structural_features(a), coverage_features(ctx, q, a)])
                for ctx, q, a in parts
            ]
        )
        return np.hstack([emb, struct]).astype(np.float32)

    fit_transform = transform


def get_featurizer(kind: str = "hashing") -> Any:
    if kind == "minilm":
        try:
            return MiniLMFeaturizer()
        except Exception:
            # Falling back is the right call: a demo that cannot start because
            # torch is missing is worse than a demo that starts slightly dumber.
            return HashingFeaturizer()
    return HashingFeaturizer()


def coverage_features(context: str, question: str, answer: str) -> np.ndarray:
    """Cross-features between the source and the candidate.

    A production guardrail sees the prompt, not just the response, and refusing
    to use it would be leaving the cheapest signal on the table. Coverage of the
    source's content words is what makes *omission* and *topic drift* detectable
    without a reference answer: the source says which fields were reported, and
    the answer either mentions them or does not.

    Deliberately absent: any check that a stated number is *correct*. In this
    corpus the source describes which quantities were reported but not their
    values, so numeric substitution is unverifiable from the context alone --
    exactly the blind spot the LLM judges show. The student inherits it rather
    than papering over it.
    """
    ctx = set(_WORD.findall(context.lower())) - _STOP
    ans = set(_WORD.findall(answer.lower())) - _STOP
    q = set(_WORD.findall(question.lower())) - _STOP
    if not ctx:
        return np.zeros(6, dtype=np.float32)
    inter = ctx & ans
    return np.array(
        [
            len(inter) / max(len(ctx), 1),  # source coverage  -> omission
            len(inter) / max(len(ans), 1),  # answer groundedness -> fabrication
            len(ans - ctx) / max(len(ans), 1),  # unsupported vocabulary
            len(q & ans) / max(len(q), 1),  # on-question       -> topic drift
            len(ans) / max(len(ctx), 1),  # relative vocabulary size
            float(len(_NUM.findall(answer))) / max(len(ans), 1),
        ],
        dtype=np.float32,
    )


def render_example(question: str, answer: str, context: str = "") -> str:
    """The text the student embeds. Context included: the guardrail has it."""
    head = f"CONTEXT: {context}\n" if context else ""
    return f"{head}QUESTION: {question}\n[SEP]\nANSWER: {answer}"


def split_example(text: str) -> tuple[str, str, str]:
    """Inverse of :func:`render_example`, so featurisers can use the parts."""
    context, question = "", ""
    body = text
    if body.startswith("CONTEXT: "):
        context, _, body = body[len("CONTEXT: ") :].partition("\nQUESTION: ")
        body = "QUESTION: " + body
    if body.startswith("QUESTION: "):
        question, _, body = body[len("QUESTION: ") :].partition("\n[SEP]\nANSWER: ")
    return context, question, body


__all__ = [
    "HEDGE_WORDS",
    "HashingFeaturizer",
    "MiniLMFeaturizer",
    "coverage_features",
    "get_featurizer",
    "render_example",
    "split_example",
    "structural_features",
]
