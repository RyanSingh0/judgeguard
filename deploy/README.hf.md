---
title: JudgeGuard
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
license: mit
---

# JudgeGuard

An evaluation workbench with a measured local-model numeric QA pilot, auditable
response evidence, historical simulated experiments, and an experimental CPU guardrail.

The published real pilot uses Qwen3-4B Q4_K_M on 100 SQuAD numeric pairs. The
simulation-trained student is not a production factual verifier. The UI distinguishes
these evidence sources. No provider secret is needed to view results or run the
prototype student. With default settings, the interactive full judge is simulated.

Dataset attribution and separate CC BY-SA 4.0 terms: `data/README.md`.
Local benchmark and deployment instructions: `docs/completion.md`.
