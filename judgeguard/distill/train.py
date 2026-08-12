"""Distil the winning judge into a student that can run inline.

The industry pattern this implements: a cheap distilled evaluator scores high
volumes continuously, and the expensive judge is reserved for deep verification.
The student is trained on the *teacher judge's* labels, not on the degradation
ground truth -- distillation means inheriting the teacher's judgement, including
wherever the teacher is wrong. Measuring the gap to ground truth separately is
what keeps that honest.

Calibration is not decoration. A guardrail thresholds a probability, so the
number the student emits has to mean what it says: of the responses it scores
0.7, roughly 70% should be good. Isotonic calibration plus a reliability curve
and Brier/ECE is how that claim gets checked.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from judgeguard.distill.features import get_featurizer, render_example


@dataclass
class StudentReport:
    featurizer: str
    n_train: int
    n_calibration: int
    n_test: int
    accuracy: float
    auc: float
    brier: float
    ece: float  # against TEACHER labels: fidelity to the teacher
    ece_vs_truth_before: float  # against ground truth, before recalibration
    ece_vs_truth_after: float  # against ground truth, after free recalibration
    log_loss: float
    agreement_with_teacher: float
    recalibrated: bool
    fit_seconds: float
    inference_ms_mean: float
    inference_ms_p95: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """ECE plus the per-bin table the reliability curve is drawn from."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    table: list[dict[str, float]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if not n:
            table.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": 0,
                    "confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        conf = float(p[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (n / p.size) * abs(acc - conf)
        table.append(
            {"bin_lo": float(lo), "bin_hi": float(hi), "n": n, "confidence": conf, "accuracy": acc}
        )
    return float(ece), table


class Student:
    """The thing that actually serves ``POST /guard``."""

    def __init__(self, featurizer: str = "hashing", seed: int = 20260731) -> None:
        self.featurizer_kind = featurizer
        self.featurizer = get_featurizer(featurizer)
        self.seed = seed
        self.clf: Any = None
        self.calibrator: Any = None
        self.threshold: float = 0.5

    # ------------------------------------------------------------------ train
    def fit(
        self,
        texts: list[str],
        labels: list[int],
        truth: list[int] | None = None,
        *,
        test_size: float = 0.20,
        calibration_size: float = 0.20,
    ) -> tuple[StudentReport, dict[str, Any]]:
        """Fit on the teacher's labels, then recalibrate against free ground truth.

        Three splits, and the middle one is the interesting part:

        ``train``        teacher labels only. This is distillation: the student
                         learns the teacher's judgement, mistakes included.
        ``calibration``  degradation ground truth. Distillation faithfully
                         inherits the teacher's *calibration error* along with
                         its decisions, so a student fit purely on teacher
                         labels emits probabilities that mean "the teacher would
                         accept this", not "this is acceptable". Because
                         degradation labels cost nothing to produce, that
                         mismatch can be corrected with an isotonic map -- free
                         calibration is the underrated payoff of manufacturing
                         gold labels instead of buying them.
        ``test``         never touched by either fit.
        """
        X_all = self.featurizer.transform(texts)
        y_all = np.asarray(labels, dtype=int)
        t_all = np.asarray(truth if truth is not None else labels, dtype=int)
        idx = np.arange(len(labels))

        i_fit, i_test = train_test_split(
            idx, test_size=test_size, random_state=self.seed, stratify=y_all
        )
        rel_cal = calibration_size / (1.0 - test_size)
        i_tr, i_cal = train_test_split(
            i_fit, test_size=rel_cal, random_state=self.seed, stratify=y_all[i_fit]
        )

        base = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        self.clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
        t0 = time.perf_counter()
        self.clf.fit(X_all[i_tr], y_all[i_tr])
        fit_s = time.perf_counter() - t0

        raw_test = self.clf.predict_proba(X_all[i_test])[:, 1]
        y_teacher_test = y_all[i_test]
        y_truth_test = t_all[i_test]
        ece_teacher, _ = expected_calibration_error(y_teacher_test, raw_test)
        ece_truth_before, _ = expected_calibration_error(y_truth_test, raw_test)

        recalibrated = False
        if truth is not None and len(set(t_all[i_cal].tolist())) > 1:
            raw_cal = self.clf.predict_proba(X_all[i_cal])[:, 1]
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_cal, t_all[i_cal])
            recalibrated = True

        p = self._apply_calibrator(raw_test)
        ece_truth_after, table = expected_calibration_error(y_truth_test, p)
        pred = (p >= 0.5).astype(int)

        lat = self._time_inference(texts[:200] or texts)
        report = StudentReport(
            featurizer=self.featurizer_kind,
            n_train=len(i_tr),
            n_calibration=len(i_cal),
            n_test=len(i_test),
            accuracy=float((pred == y_truth_test).mean()),
            auc=float(roc_auc_score(y_teacher_test, raw_test))
            if len(set(y_teacher_test)) > 1
            else float("nan"),
            brier=float(brier_score_loss(y_truth_test, p)),
            ece=ece_teacher,
            ece_vs_truth_before=ece_truth_before,
            ece_vs_truth_after=ece_truth_after,
            log_loss=float(log_loss(y_truth_test, np.clip(p, 1e-6, 1 - 1e-6))),
            agreement_with_teacher=float(((raw_test >= 0.5).astype(int) == y_teacher_test).mean()),
            recalibrated=recalibrated,
            fit_seconds=fit_s,
            inference_ms_mean=lat["mean"],
            inference_ms_p95=lat["p95"],
        )
        extra = {
            "reliability_curve": table,
            "reliability_curve_before": expected_calibration_error(y_truth_test, raw_test)[1],
            "test_indices": i_test.tolist(),
            "calibration_indices": i_cal.tolist(),
            "test_probabilities": p.tolist(),
            "test_probabilities_uncalibrated": raw_test.tolist(),
            "test_labels": y_teacher_test.tolist(),
            "test_truth": y_truth_test.tolist(),
        }
        return report, extra

    def _apply_calibrator(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return p
        return np.clip(self.calibrator.predict(p), 0.0, 1.0)

    # -------------------------------------------------------------- inference
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.featurizer.transform(texts)
        return self._apply_calibrator(self.clf.predict_proba(X)[:, 1])

    def guard(self, question: str, answer: str, context: str = "") -> dict[str, Any]:
        t0 = time.perf_counter()
        p = float(self.predict_proba([render_example(question, answer, context)])[0])
        dt = (time.perf_counter() - t0) * 1000
        return {
            "probability_good": p,
            "decision": "allow" if p >= self.threshold else "block",
            "threshold": self.threshold,
            "latency_ms": dt,
        }

    def _time_inference(self, texts: list[str], repeats: int = 3) -> dict[str, float]:
        times = []
        for _ in range(repeats):
            for t in texts:
                t0 = time.perf_counter()
                self.predict_proba([t])
                times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times) if times else np.array([0.0])
        return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95))}

    def choose_threshold(
        self, p: np.ndarray, y: np.ndarray, *, max_false_allow: float = 0.10
    ) -> float:
        """Pick the operating point from a stated policy, not from 0.5.

        A guardrail threshold is a policy decision and 0.5 is a default, not a
        policy. Two criteria, in order:

        1. **Constraint** -- of everything the guardrail lets through, at most
           ``max_false_allow`` may be bad. This is a precision bound on ``allow``
           and it is the number a stakeholder actually signs off on.
        2. **Objective** -- among thresholds satisfying the constraint, maximise
           balanced accuracy.

        The second criterion is load-bearing. Taking the most permissive feasible
        threshold instead sounds right and is not: on a bimodal probability
        distribution it lands in the tail and blocks a fifth of *good* traffic to
        buy false-allow headroom nobody asked for. Balanced accuracy prices both
        error types symmetrically and lands the cut in the gap between the modes,
        which is also where it is robust to a shift in the base rate.
        """
        pos, neg = y == 1, y == 0
        best_thr, best_score = 0.5, -1.0
        feasible = False
        for cand in np.linspace(0.01, 0.99, 197):
            allow = p >= cand
            n_allow = int(allow.sum())
            if n_allow == 0:
                continue
            false_allow = float(((y == 0) & allow).sum() / n_allow)
            if false_allow > max_false_allow:
                continue
            tpr = float(allow[pos].mean()) if pos.any() else 0.0
            tnr = float((~allow)[neg].mean()) if neg.any() else 0.0
            score = (tpr + tnr) / 2
            if score > best_score:
                best_score, best_thr, feasible = score, float(cand), True
        if not feasible:  # constraint unsatisfiable anywhere
            best_thr = 0.5
        self.threshold = best_thr
        return best_thr

    # ------------------------------------------------------------------- i/o
    def save(self, path: Path | str) -> Path:
        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "calibrator": self.calibrator,
                "threshold": self.threshold,
                "featurizer": self.featurizer_kind,
            },
            p,
            compress=3,
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> Student:
        import joblib

        blob = joblib.load(Path(path))
        s = cls(featurizer=blob.get("featurizer", "hashing"))
        s.clf = blob["clf"]
        s.calibrator = blob.get("calibrator")
        s.threshold = float(blob.get("threshold", 0.5))
        return s


__all__ = ["Student", "StudentReport", "expected_calibration_error"]
