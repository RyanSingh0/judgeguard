"""Real-source numeric corruption pilot, separate from the procedural battery."""

from __future__ import annotations

import hashlib
import json
import random
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

SOURCE = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


class EvalPair(BaseModel):
    id: str
    source_id: str
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    degradation: str = "numeric_swap"
    split: str = "test"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct(self) -> EvalPair:
        if self.reference == self.candidate:
            raise ValueError("A corruption must differ from its reference")
        return self


def load_pairs(path: Path) -> list[EvalPair]:
    pairs = [
        EvalPair.model_validate_json(s)
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if not pairs or len({p.id for p in pairs}) != len(pairs):
        raise ValueError("Dataset must be nonempty with unique IDs")
    return pairs


def prepare_squad(source: Path, output: Path, n: int = 100, seed: int = 20260731) -> dict:
    """Keep numeric questions whose annotators agree; one question per passage.

    These are constructed errors on human-annotated source QA, not naturally
    occurring model failures. They test a narrow factual-support capability.
    """
    if n < 1:
        raise ValueError("n must be positive")
    raw = source.read_bytes()
    data = json.loads(raw)
    pool = []
    number = re.compile(r"\d+(?:\.\d+)?")
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if len(context) > 12000:
                continue
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                if qa.get("is_impossible") or not answers:
                    continue
                values = {a["text"].strip() for a in answers}
                if len(values) != 1:
                    continue
                ref = next(iter(values))
                if not number.fullmatch(ref):
                    continue
                if any(
                    context[a["answer_start"] : a["answer_start"] + len(a["text"])] != a["text"]
                    for a in answers
                ):
                    continue
                value = Decimal(ref)
                step = Decimal(1).scaleb(-len(ref.split(".")[1])) if "." in ref else Decimal(1)
                bad = format(value + max(step, (value * Decimal("0.1")).quantize(step)), "f")
                if bad in values or bad in context:
                    continue
                pool.append(
                    EvalPair(
                        id=qa["id"],
                        source_id=article["title"],
                        context=context,
                        question=qa["question"],
                        reference=ref,
                        candidate=bad,
                        metadata={
                            "source_url": SOURCE,
                            "license": "CC-BY-SA-4.0",
                            "source_split": "SQuAD-v2-development",
                            "answer_spans": answers,
                            "edit": f"Replaced annotated numeric answer {ref} with {bad}",
                            "label_origin": "human reference; programmatic corruption",
                            "review_status": "automatic checks only; human spot-check pending",
                        },
                    )
                )
                break
    random.Random(seed).shuffle(pool)
    selected = pool[:n]
    if len(selected) < n:
        raise ValueError(f"Only {len(selected)} eligible passages for requested {n}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(p.model_dump_json() + "\n" for p in selected), encoding="utf-8")
    manifest = {
        "source_url": SOURCE,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "license": "CC-BY-SA-4.0",
        "attribution": "SQuAD: Rajpurkar, Jia, Liang et al.; Wikipedia passages",
        "pairs": len(selected),
        "candidate_answers": 2 * len(selected),
        "source_articles": len({p.source_id for p in selected}),
        "seed": seed,
        "scope": "Numeric short-answer pilot only; not an overall RAG or agent benchmark",
        "training_policy": "Evaluation-only. Do not train or tune on this pilot.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
