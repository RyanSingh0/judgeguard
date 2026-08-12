#!/usr/bin/env python
"""Export the distilled guardrail so it can run in a browser.

The student is a logistic model over hashed n-grams with an isotonic calibration
layer. Nothing about that requires a server: the whole thing is a sparse dot
product and a step function, which is why the guardrail can be shipped as a
static page and still be the *same model* rather than a mock-up.

What is exported:
  * the 5 cross-validated (coefficient, intercept, isotonic) triples that
    CalibratedClassifierCV averages over -- exported faithfully rather than
    collapsed, because the isotonic step is non-linear and cannot be averaged
    into a single linear model without changing the predictions;
  * the final isotonic recalibrator fitted on free degradation gold labels;
  * the feature vocabulary constants (hedge words, filler phrases, stop words)
    so the JavaScript port cannot silently drift from the Python definition;
  * a parity fixture -- texts with their Python probabilities -- so the port is
    checked rather than assumed.

    python scripts/export_web_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from judgeguard.data.load import load_items  # noqa: E402
from judgeguard.degrade.text import hedging, numeric_swap, omission, verbosity  # noqa: E402
from judgeguard.distill.features import (  # noqa: E402
    _STOP,
    FILLER_PHRASES,
    HEDGE_WORDS,
    render_example,
)
from judgeguard.distill.train import Student  # noqa: E402

OUT = REPO / "site" / "model.json"
COEF_EPS = 1e-12  # anything below this contributes nothing at float32


def sparse(coef: np.ndarray) -> dict:
    idx = np.flatnonzero(np.abs(coef) > COEF_EPS)
    return {
        "i": idx.astype(int).tolist(),
        "v": [round(float(x), 7) for x in coef[idx]],
    }


def iso(cal) -> dict:
    """An isotonic regressor is just a monotone step function: knots + values."""
    return {
        "x": [round(float(v), 7) for v in cal.X_thresholds_],
        "y": [round(float(v), 7) for v in cal.y_thresholds_],
    }


def main() -> None:
    blob = joblib.load(REPO / "results" / "student_model.joblib")
    clf, recal = blob["clf"], blob["calibrator"]

    members = []
    for cc in clf.calibrated_classifiers_:
        members.append(
            {
                "coef": sparse(cc.estimator.coef_.ravel()),
                "b": float(cc.estimator.intercept_[0]),
                "iso": iso(cc.calibrators[0]),
            }
        )

    # ---------------------------------------------------------- parity fixture
    student = Student.load(REPO / "results" / "student_model.joblib")
    items = load_items(500)
    cases = []
    for it in items[:6]:
        variants = [
            ("reference", it.reference),
            ("hedging", hedging(it, 0.6, seed=1).text),
            ("omission", omission(it, 0.5, seed=1).text),
            ("numeric_swap", numeric_swap(it, 0.5, seed=1).text),
            ("verbosity", verbosity(it, 0.9, seed=1).text),
        ]
        for label, answer in variants:
            p = float(student.predict_proba([render_example(it.question, answer, it.context)])[0])
            cases.append(
                {
                    "label": label,
                    "context": it.context,
                    "question": it.question,
                    "answer": answer,
                    "p": round(p, 10),
                }
            )

    payload = {
        "_comment": (
            "Exported by scripts/export_web_model.py. This is the SAME model the Python "
            "service serves, not a re-fit approximation. Parity is asserted by "
            "site/parity.test.js against the fixtures below."
        ),
        "version": 1,
        "threshold": float(blob["threshold"]),
        "featurizer": blob["featurizer"],
        "dims": {"word": 2**17, "char": 2**16, "extra": 18},
        "ngrams": {"word": [1, 2], "char": [3, 4]},
        "members": members,
        "recalibrator": iso(recal) if recal is not None else None,
        "vocab": {
            "hedge_words": sorted(HEDGE_WORDS),
            "filler_phrases": list(FILLER_PHRASES),
            "stop_words": sorted(_STOP),
        },
        "parity_cases": cases,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    nz = sum(len(m["coef"]["i"]) for m in members)
    print(f"  wrote {OUT.relative_to(REPO)}")
    print(f"  {len(members)} members, {nz:,} nonzero coefficients, {len(cases)} parity cases")
    print(f"  {OUT.stat().st_size / 1e6:.2f} MB uncompressed")


if __name__ == "__main__":
    main()
