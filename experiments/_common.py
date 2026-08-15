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
    p.add_argument(
        "--severities",
        type=str,
        default="",
        help=(
            "comma-separated severity levels, e.g. '0.5'. Defaults to the full "
            "0.2/0.5/0.9 sweep. That sweep triples every call count, and only "
            "experiment 01 plots a severity curve, so the rest can run one mid "
            "severity without weakening anything they claim."
        ),
    )
    return p


def severities_from(args: Any, default: tuple[float, ...] | None = None) -> tuple[float, ...]:
    """Resolve ``--severities``, falling back to the module default."""
    from judgeguard.degrade.text import SEVERITIES

    if getattr(args, "severities", ""):
        return tuple(float(s) for s in args.severities.split(",") if s.strip())
    return default or SEVERITIES


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


def exhausted_today() -> set[str]:
    """Panel aliases whose daily allowance is already gone.

    Read from the persisted limiter state, so it survives across runs and across
    the separate processes each experiment runs in.
    """
    from judgeguard.providers.http_base import LIMITER
    from judgeguard.providers.registry import resolve

    report = LIMITER.report()
    out: set[str] = set()
    for alias in load_registry().panel:
        spec = resolve(alias)
        s = report.get(f"{spec.provider}:{spec.id}", {})
        if s.get("exhausted_today") or s.get("remaining_today") == 0:
            out.add(alias)
    return out


def resolve_judges(arg: str) -> list[str]:
    """The panel, minus anyone who has no allowance left today.

    Learned this the hard way on the first phase-1 run. gemini-lite sits first in
    the panel and ran out after 493 calls, which aborted experiment 01 before the
    three Groq judges were touched at all. They had 1,000 calls each sitting
    unused. One judge's small quota was blocking three healthy ones.

    So skip the exhausted judge and run the rest. The results file records who
    was skipped, and because every call is cached, tomorrow's run refills the
    gap and rewrites the file complete for a few hundred new calls.
    """
    if arg:
        return [a.strip() for a in arg.split(",") if a.strip()]
    panel = list(load_registry().panel)
    dead = exhausted_today()
    live = [j for j in panel if j not in dead]
    if dead:
        print(f"    SKIPPING (no allowance left today): {', '.join(sorted(dead))}")
        print(f"    running with {len(live)} of {len(panel)} judges; re-run tomorrow to fill in")
    if not live:
        raise SystemExit(EXIT_QUOTA)
    return live


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


# Exit code for "stopped cleanly, daily allowance ran out". Separate from 1 so
# the runner can tell "come back tomorrow" from "this is broken, read the
# traceback".
EXIT_QUOTA = 42


def record_partial(experiment: str, skipped: list[str]) -> None:
    """Note which judges are missing from a results file, where I'll see it.

    A results file with three of four judges is fine as long as it is obviously
    three of four. It is not fine if it renders a figure that looks complete.
    """
    import json

    from judgeguard.store import RESULTS_DIR

    path = RESULTS_DIR / "_partial.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if skipped:
        data[experiment] = sorted(skipped)
    else:
        data.pop(experiment, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif path.exists():
        path.unlink()


def run_main(main: Any) -> None:
    """Wrapper that handles a judge running out of allowance mid-experiment.

    First phase-1 run taught me this. gemini-lite ran dry 493 calls into
    experiment 01 and the whole thing stopped, before the three Groq judges had
    made a single call. They had 1,000 each going spare.

    So when a judge dies partway, mark it, throw away the partial work in memory
    and run the experiment again without it. The rerun is nearly free because
    every completed call is cached. Tomorrow's run picks the judge back up and
    rewrites the file with the full panel.

    Only stops outright when nobody has allowance left.
    """
    from judgeguard.judges.run import JudgeUnavailableError, failure_report, reset_failures
    from judgeguard.providers.http_base import LIMITER

    name = Path(sys.argv[0]).stem
    try:
        try:
            main()
            record_partial(name, sorted(exhausted_today() & set(load_registry().panel)))
        except JudgeUnavailableError as exc:
            LIMITER.save()
            print(f"\n    {exc}")
            dead = sorted(exhausted_today())
            print(f"    Retrying this experiment without: {', '.join(dead)}")
            print("    Everything already measured is cached, so this costs almost nothing.\n")
            reset_failures()
            main()
            record_partial(name, dead)
            print(f"\n    PARTIAL: this file has {len(dead)} judge(s) missing ({', '.join(dead)}).")
            print("    Re-run tomorrow to fill them in; the rest will come from cache.")
    except SystemExit as exc:
        if exc.code == EXIT_QUOTA:
            print("\n    Every judge is out of allowance for today. Nothing more to do.")
            print("    Re-run tomorrow; completed work is cached.")
        raise
    except JudgeUnavailableError:
        # Second judge died during the retry. Stop rather than spiral.
        LIMITER.save()
        report = failure_report()["by_judge"]
        spent = {j: s["total_calls"] for j, s in report.items() if s["quota_exhausted"]}
        print(f"\n    QUOTA STOP: another judge ran out during the retry ({spent}).")
        print("    No results file written. Re-run tomorrow.")
        sys.exit(EXIT_QUOTA)
    finally:
        LIMITER.save()


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
