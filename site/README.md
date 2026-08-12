---
title: JudgeGuard
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Can you trust the judge? The guardrail runs in your browser.
---

# JudgeGuard — can you trust the judge?

Everyone building with LLMs uses an LLM to grade the output. Almost nobody measures whether the
grader is any good.

**This page has no backend.** The distilled guardrail — a logistic model over hashed n-grams with
isotonic calibration — runs entirely in your browser, and it is the *same model* the Python service
serves: a parity test asserts agreement with scikit-learn to within 1e-6 on every case. That is
possible because the whole inference path is a sparse dot product and a step function, which is
also why it fits in a 150 ms budget with 143 ms to spare.

Click the five examples. The guardrail allows the correct answer, blocks hedging and omission,
**allows a numeric substitution at an identical probability** — a documented structural blind spot —
and blocks padded-but-correct text, which is the mirror image of the judges' verbosity bias.

**Headline finding:** judges failed to detect malformed tool arguments in 37.2%–64.4% of agent
trajectories, a failure class output-only evaluation cannot reach by construction.

📦 [Code + write-up](https://github.com/RyanSingh0/judgeguard) ·
📄 [Findings](https://github.com/RyanSingh0/judgeguard/blob/main/docs/findings.md) ·
⚠️ [Limitations](https://github.com/RyanSingh0/judgeguard/blob/main/docs/limitations.md)

Committed results come from a deterministic simulator and every figure is watermarked; see
Limitations §1.
