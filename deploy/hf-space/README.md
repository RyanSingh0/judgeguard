---
title: JudgeGuard
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
license: mit
short_description: Measuring LLM judge reliability, served as a sub-150ms guardrail
---

# JudgeGuard — can you trust the judge?

Everyone building with LLMs uses an LLM to grade the output. Almost nobody measures whether the
grader is any good.

This Space runs the live demo. Paste a response and watch two paths that are deliberately **not**
the same product:

- **`POST /guard`** — inline. Distilled classifier, no network call inside the request, hard
  p99 ≤ 150 ms budget. Measured: p50 3.8 ms.
- **`POST /evaluate`** — async. Full LLM judge, with *that judge's measured reliability* attached
  to its score.

Five example buttons show the method in three clicks: it allows a correct answer, blocks hedging
and omission, **allows a numeric substitution at an identical probability** (a documented
structural blind spot), and blocks padded-but-correct text.

**Headline finding:** judges failed to detect malformed tool arguments in 37.2%–64.4% of agent
trajectories — a failure class output-only evaluation cannot reach by construction.

📦 [Code + full write-up](https://github.com/RyanSingh0/judgeguard) ·
📄 [Findings](https://github.com/RyanSingh0/judgeguard/blob/main/docs/findings.md) ·
⚠️ [Limitations](https://github.com/RyanSingh0/judgeguard/blob/main/docs/limitations.md)

Committed results are produced by a deterministic simulator and every figure is watermarked; see
Limitations §1.
