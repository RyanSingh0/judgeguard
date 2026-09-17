---
title: JudgeGuard
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Replay agent traces and inspect real LLM judge failures
---

# JudgeGuard — Agent Evaluation Workbench

A correct answer can hide a broken trace. Replay four deterministic tools, locate
inconsistent recorded results, and inspect 200 actual local Qwen judge responses.

**Tabs:** editable trace audit, real numeric QA evidence explorer, optional live
Qwen3-0.6B judge, and release criteria. No API key is needed. Only the live judge
uses GPU quota; replay and saved evidence do not. Free queues and daily limits apply.
The live 0.6B model is separate from the measured 4B pilot below and is not a blocker.

The 100-pair Qwen pilot scored 96 correct rankings (95% article-cluster interval
92.45%–99.03%). Four ties remain visible. This is a narrow constructed-error study,
not a general agent or RAG accuracy claim. Independent label review is pending.

The experimental simulation-trained student failed promotion checks and is not used
to make decisions in this public workbench. Historical simulated findings remain
labelled in the source repository.

Source: https://github.com/RyanSingh0/judgeguard

Code MIT. SQuAD-derived data CC BY-SA 4.0: https://rajpurkar.github.io/SQuAD-explorer/
Attribution and edits: https://github.com/RyanSingh0/judgeguard/blob/main/data/README.md
