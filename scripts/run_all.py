#!/usr/bin/env python
"""Run the whole pipeline, in order, with one command.

python scripts/run_all.py                 # everything
python scripts/run_all.py --stage battery # experiments 01-08 only
python scripts/run_all.py --quick         # smoke run, seconds not minutes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STAGES: dict[str, list[str]] = {
    "data": ["00_build_dataset.py"],
    "battery": [
        "01_discrimination.py",
        "02_position_bias.py",
        "03_verbosity_bias.py",
        "04_self_enhancement.py",
        "05_rubric_ablation.py",
        "06_self_consistency.py",
        "07_cost_accuracy.py",
        "08_trajectory_blindness.py",
    ],
    "ship": ["09_distill.py", "10_latency_bench.py"],
    "figures": ["make_figures.py"],
    "gate": ["regression_suite.py"],
}
ORDER = ["data", "battery", "ship", "figures", "gate"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=[*ORDER, "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--items", type=int, default=None)
    args, passthrough = ap.parse_known_args()

    stages = ORDER if args.stage == "all" else [args.stage]
    t0 = time.perf_counter()
    for stage in stages:
        for script in STAGES[stage]:
            cmd = [sys.executable, str(REPO / "experiments" / script)]
            if args.quick and script != "make_figures.py":
                cmd.append("--quick")
            if args.items and script not in ("make_figures.py", "07_cost_accuracy.py"):
                cmd += ["--items", str(args.items)]
            cmd += passthrough
            rc = subprocess.call(cmd, cwd=REPO)
            if rc != 0:
                print(f"\n!! {script} exited {rc}", file=sys.stderr)
                return rc
    print(
        f"\nAll stages complete in {time.perf_counter() - t0:.1f}s. "
        f"See results/ and docs/findings.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
