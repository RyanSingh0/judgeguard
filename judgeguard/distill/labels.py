"""Turn the winning judge's decisions into a supervised training set.

Deliberate design choice: the student learns the *teacher's* label, not the
degradation ground truth. Training on ground truth would produce a better
classifier and a worse experiment -- the question under test is how much of a
specific judge's behaviour survives compression, and you cannot answer that by
quietly substituting a better teacher.

Where the teacher is wrong, the student is supposed to be wrong in the same way.
Ground truth is kept alongside so the two gaps -- student-vs-teacher and
student-vs-truth -- can be reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from judgeguard.distill.features import render_example


@dataclass(slots=True)
class LabelledExample:
    uid: str
    text: str
    teacher_label: int  # 1 = teacher judged it acceptable
    truth_label: int  # 1 = it really is the reference (or a no-error probe)
    degradation: str
    severity: float
    teacher_score: float | None = None


def label_from_scores(
    records: list[dict[str, Any]],
    *,
    accept_at: float = 7.0,
) -> list[LabelledExample]:
    """Binarise pointwise teacher scores at a stated acceptance band.

    ``accept_at`` is the rubric's own boundary between "minor gaps" and "a
    material omission", so the cut point comes from the prompt rather than from
    whatever threshold made the numbers look best.
    """
    out: list[LabelledExample] = []
    for r in records:
        score = r.get("score")
        if score is None:
            continue
        out.append(
            LabelledExample(
                uid=r["uid"],
                text=render_example(r.get("question", ""), r.get("answer", "")),
                teacher_label=int(float(score) >= accept_at),
                truth_label=int(r.get("is_reference", 0)),
                degradation=r.get("degradation", "none"),
                severity=float(r.get("severity", 0.0)),
                teacher_score=float(score),
            )
        )
    return out


def to_arrays(examples: list[LabelledExample]) -> tuple[list[str], list[int], list[int]]:
    return (
        [e.text for e in examples],
        [e.teacher_label for e in examples],
        [e.truth_label for e in examples],
    )


__all__ = ["LabelledExample", "label_from_scores", "to_arrays"]
