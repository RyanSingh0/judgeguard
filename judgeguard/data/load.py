"""Corpus loading.

Default source is the procedural generator (no network, seeded, fact-structured).
A JSONL loader is provided so a live run can swap in a real reference corpus --
any dataset with a question, a context and a gold answer works, at the cost of
coarser degradation auditing (see ``docs/limitations.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

from judgeguard.config import get_settings
from judgeguard.data.corpus import generate_corpus
from judgeguard.data.schema import Item


def load_items(
    n: int = 500,
    *,
    seed: int | None = None,
    source: str = "procedural",
    path: Path | str | None = None,
) -> list[Item]:
    seed = seed if seed is not None else get_settings().seed
    if source == "procedural":
        return generate_corpus(n=n, seed=seed)
    if source == "jsonl":
        if path is None:
            raise ValueError("source='jsonl' requires path=")
        items: list[Item] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    items.append(Item.model_validate(json.loads(line)))
        return items[:n]
    raise ValueError(f"unknown source {source!r}")


def save_items(items: list[Item], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    return p


__all__ = ["load_items", "save_items"]
