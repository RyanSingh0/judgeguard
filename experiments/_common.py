"""Shared plumbing for the experiment scripts.

Each script under ``experiments/`` produces exactly one figure or one table and
writes one JSON file to ``results/``. Keeping that one-to-one makes every claim
in the README traceable to a file and a command.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from judgeguard.config import get_settings, load_registry
from judgeguard.data.load import load_items
from judgeguard.data.schema import Item
from judgeguard.degrade.text import build_variants
from judgeguard.providers.cache import cache_stats
from judgeguard.providers.registry import mode_banner
from judgeguard.telemetry import get_logger

log = get_logger("experiment")

N_ITEMS_DEFAULT = 500
N_PAIRWISE_DEFAULT = 250
N_TRAJ_DEFAULT = 60


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--items", type=int, default=N_ITEMS_DEFAULT, help="number of source items")
    p.add_argument("--judges", type=str, default="", help="comma-separated judge aliases")
    p.add_argument("--seed", type=int, default=get_settings().seed)
    p.add_argument("--quick", action="store_true", help="tiny run for smoke tests / CI")
    return p


def apply_quick_isolation(args: argparse.Namespace) -> argparse.Namespace:
    """Send `--quick` artefacts to `results/_smoke/` instead of `results/`.

    A 40-item smoke run that overwrites the committed 500-item results is a
    silent, expensive mistake: everything still runs, every figure still
    renders, and every number in the README quietly becomes wrong.
    """
    if getattr(args, "quick", False):
        import judgeguard.store as _store

        smoke = _store.RESULTS_DIR / "_smoke"
        _store.set_results_dir(smoke)
        print(f"    quick mode: writing to {smoke.name}/ so committed results stay intact")
    return args


def resolve_judges(arg: str) -> list[str]:
    if arg:
        return [a.strip() for a in arg.split(",") if a.strip()]
    return list(load_registry().panel)


def banner(name: str) -> float:
    print(f"\n=== {name} ===")
    print(f"    {mode_banner()}")
    return time.perf_counter()


def rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    ``JUDGEGUARD_RESULTS_DIR`` may legitimately point outside the checkout (a
    scratch dir in CI, a mounted volume in a container), and ``relative_to``
    raises rather than falling back.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def done(t0: float, path: Path) -> None:
    print(
        f"    wrote {rel(path)}  ({time.perf_counter() - t0:.1f}s, cache {cache_stats()['hits']} hits)"
    )


def corpus(n: int, seed: int) -> tuple[list[Item], dict[str, Item]]:
    items = load_items(n, seed=seed)
    return items, {i.id: i for i in items}


def variants_for(items: list[Item], seed: int, degradations: list[str] | None = None) -> list[Any]:
    return build_variants(items, seed=seed, degradations=degradations)


__all__ = [
    "N_ITEMS_DEFAULT",
    "N_PAIRWISE_DEFAULT",
    "N_TRAJ_DEFAULT",
    "REPO",
    "apply_quick_isolation",
    "banner",
    "base_parser",
    "corpus",
    "done",
    "log",
    "rel",
    "resolve_judges",
    "variants_for",
]
