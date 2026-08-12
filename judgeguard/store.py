"""Result persistence.

Every experiment writes one JSON file to `results/`, and those files are
committed. That is what makes the README, `/report` and the paper render for a
reader who has no API key and never runs anything.

A DuckDB view is layered on top so cross-experiment queries ("accuracy by judge
x degradation x severity") are one SQL statement instead of a pile of pandas.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from judgeguard.config import REPO_ROOT, get_settings

RESULTS_DIR = get_settings().abs_results_dir


def set_results_dir(path: Path | str) -> Path:
    """Redirect writes at runtime.

    Exists for one reason: a smoke run must never overwrite the committed
    results. `--quick` produces 40-item artefacts, and if those land in
    `results/` they silently replace the 500-item run that the README, the
    figures and the paper all quote -- a failure that looks like nothing at all
    until someone checks the numbers. `experiments/_common.py` redirects every
    `--quick` invocation into `results/_smoke/`.
    """
    global RESULTS_DIR
    RESULTS_DIR = Path(path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def provenance() -> dict[str, Any]:
    """Stamped onto every artefact. `mode` is the honesty flag."""
    s = get_settings()
    return {
        "mode": s.effective_mode().value,
        "seed": s.seed,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def json_safe(obj: Any) -> Any:
    """Replace NaN/Inf with null.

    ``json.dumps`` emits bare ``NaN`` by default, which is not valid JSON and
    which every strict parser -- including the browser reading ``/report`` --
    rejects. Statistics legitimately produce NaN (an undefined correlation, a
    degenerate interval), so the fix belongs at the serialisation boundary
    rather than in the statistics.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save(name: str, payload: dict[str, Any], *, subdir: str = "") -> Path:
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name if name.endswith(".json") else f"{name}.json")
    body = json_safe({"provenance": provenance(), **payload})
    path.write_text(
        json.dumps(body, indent=2, sort_keys=False, default=str, allow_nan=False), encoding="utf-8"
    )
    return path


def load(name: str, *, subdir: str = "") -> dict[str, Any]:
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    path = out_dir / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make all` (or the specific experiment) first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def exists(name: str, *, subdir: str = "") -> bool:
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    return (out_dir / (name if name.endswith(".json") else f"{name}.json")).exists()


def duckdb_con(judgments_file: str = "judgments.json") -> Any:
    """A DuckDB connection with `judgments` registered as a view.

    Example:
        con = duckdb_con()
        con.sql("select judge, degradation, avg(correct) from judgments group by 1,2")
    """
    import duckdb

    con = duckdb.connect()
    path = RESULTS_DIR / judgments_file
    if path.exists():
        con.execute(
            "create or replace view judgments as "
            "select unnest(records, recursive := true) from read_json_auto(?)",
            [str(path)],
        )
    return con


__all__ = [
    "RESULTS_DIR",
    "duckdb_con",
    "exists",
    "json_safe",
    "load",
    "provenance",
    "save",
    "set_results_dir",
]
