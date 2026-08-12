"""JudgeGuard — can you trust the judge?

A harness that manufactures ground truth without human annotators, probes LLM
judges for known biases with confidence intervals on every number, and distills
the winning judge into an inline guardrail with a hard latency budget.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
