"""The four judge configurations, escalating in structure.

| config     | shape                                   | hypothesis under test                        |
|------------|-----------------------------------------|----------------------------------------------|
| `vague`    | "Is this good? 1-10."                   | baseline; expect the worst discrimination    |
| `rubric`   | explicit criteria + score bands         | structure improves consistency               |
| `cot`      | rubric + reason-before-scoring          | CoT judging improves alignment                |
| `pairwise` | two answers, pick one                   | different failure surface; enables position  |
|            |                                         | bias measurement                             |

Note what the judge is **not** given: the reference answer. A reference-based
judge is a much easier problem and would not resemble the production setting the
guardrail targets. The judge sees the source context, the question and a
candidate, exactly as an online evaluator would.
"""

from __future__ import annotations

from judgeguard.data.schema import Item

CONFIGS: tuple[str, ...] = ("vague", "rubric", "cot", "pairwise")
POINTWISE_CONFIGS: tuple[str, ...] = ("vague", "rubric", "cot")

SYSTEM = "You are a careful evaluator. Follow the output format exactly."

_RUBRIC = """\
Score the candidate answer against these criteria:
  1. FACTUAL SUPPORT  - every claim is supported by the source context.
  2. COMPLETENESS     - all information the question asks for is present.
  3. PRECISION        - values are specific and committed, not hedged.
  4. RELEVANCE        - the answer addresses the question that was asked.
  5. EFFICIENCY       - length is justified by content; padding is a defect.

Score bands:
  9-10  complete, fully supported, precise
  7-8   minor gaps or imprecision
  5-6   a material omission or an unsupported claim
  3-4   several defects, or a claim contradicting the source
  1-2   largely wrong, evasive or off-question"""

_TRAJ_RUBRIC = """\
Score the agent trajectory against these criteria:
  1. TOOL VALIDITY    - every tool called exists in the declared action space.
  2. ARGUMENT CORRECTNESS - each call receives the right argument values.
  3. RESULT CONSISTENCY - the final answer follows from the intermediate results.
  4. NECESSITY        - every step contributes; redundant steps are a defect.
  5. GOAL ACHIEVEMENT - the task is actually completed.

Score bands:
  9-10  valid tools, correct arguments, consistent, minimal
  7-8   minor inefficiency
  5-6   a redundant step or a questionable argument
  3-4   an undefined tool, a wrong argument, or an inconsistent chain
  1-2   the trajectory does not support the stated answer"""

_ACTION_SPACE = """\
DECLARED ACTION SPACE (the only tools that exist):
  calculator(expr)              arithmetic on a numeric expression
  lookup(key)                   read a value from a fixed reference table
  date_diff(d1, d2)             whole days between two ISO dates
  unit_convert(value, frm, to)  convert between known units"""


def _header(item: Item, kind: str) -> str:
    if kind == "trajectory":
        return f"{_ACTION_SPACE}\n"
    return f"SOURCE CONTEXT:\n{item.context}\n\nQUESTION:\n{item.question}\n"


def score_prompt(config: str, item: Item, answer: str, *, kind: str = "text") -> str:
    """Pointwise scoring prompt for one candidate."""
    head = _header(item, kind)
    label = "AGENT TRAJECTORY" if kind == "trajectory" else "CANDIDATE ANSWER"
    rubric = _TRAJ_RUBRIC if kind == "trajectory" else _RUBRIC

    if config == "vague":
        return (
            f"{head}\n{label}:\n{answer}\n\n"
            "Is this a good answer? Give it a score from 1 to 10.\n"
            "Reply with the number only."
        )
    if config == "rubric":
        return (
            f"{head}\n{rubric}\n\n{label}:\n{answer}\n\n"
            "Reply in exactly this format:\n"
            "SCORE: <number 1-10>\n"
            "JUSTIFICATION: <one sentence>"
        )
    if config == "cot":
        return (
            f"{head}\n{rubric}\n\n{label}:\n{answer}\n\n"
            "Reason step by step through each criterion BEFORE committing to a score. "
            "Do not let length or confident tone substitute for content.\n\n"
            "Reply in exactly this format:\n"
            "REASONING: <your step-by-step analysis>\n"
            "SCORE: <number 1-10>"
        )
    raise ValueError(f"{config!r} is not a pointwise config")


def pairwise_prompt(config: str, item: Item, a: str, b: str, *, kind: str = "text") -> str:
    """Pairwise comparison prompt. Always run through the swap protocol."""
    head = _header(item, kind)
    label = "TRAJECTORY" if kind == "trajectory" else "ANSWER"
    rubric = _TRAJ_RUBRIC if kind == "trajectory" else _RUBRIC
    tail = (
        "Reply in exactly this format:\nREASONING: <one or two sentences>\nVERDICT: <A|B|tie>"
        if config == "cot"
        else "Reply with exactly one line:\nVERDICT: <A|B|tie>"
    )
    return (
        f"{head}\n{rubric}\n\n"
        f"{label} A:\n{a}\n\n{label} B:\n{b}\n\n"
        f"Which is better? Judge on content, not on length or tone.\n\n{tail}"
    )


__all__ = ["CONFIGS", "POINTWISE_CONFIGS", "SYSTEM", "pairwise_prompt", "score_prompt"]
