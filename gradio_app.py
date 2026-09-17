"""Evaluation workbench with optional, separately labelled ZeroGPU inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from judgeguard.agent.audit import audit_trace, trace_examples

LIVE_JUDGE = None
if os.getenv("SPACE_ID") or os.getenv("JUDGEGUARD_LIVE_DEMO") == "1":
    from judgeguard.space_judge import judge_live

    LIVE_JUDGE = judge_live

ROOT = Path(__file__).resolve().parent
BUNDLE = json.loads((ROOT / "demo-data.json").read_text(encoding="utf-8"))
EXAMPLES = trace_examples()
PILOT = BUNDLE["pilot"]
PAIRS = BUNDLE["pairs"]
RECORDS = {(r["pair_id"], r["role"]): r for r in BUNDLE["judgments"]}
PAIR_BY_ID = {p["id"]: p for p in PAIRS}
FAILURES = [
    p
    for p in PAIRS
    if RECORDS[p["id"], "reference"]["score"] <= RECORDS[p["id"], "candidate"]["score"]
]
ORDERED_PAIRS = FAILURES + [p for p in PAIRS if p not in FAILURES]

CSS = """
.gradio-container {max-width:1200px!important; margin:auto!important; font-family:Inter,system-ui,sans-serif!important}
#hero {background:linear-gradient(120deg,#10283f,#153d56);color:white;border-radius:18px;padding:30px 34px;margin:12px 0 22px}
#hero h1 {color:white;font-size:38px;letter-spacing:-1.5px;margin:0 0 10px;font-weight:750}
#hero p {color:#d5e8f1;margin:0;max-width:760px;line-height:1.6}
.eyebrow {color:#82dcc4;font:600 12px ui-monospace,monospace;letter-spacing:2px;margin-bottom:12px}
.chips {display:flex;gap:8px;margin-top:20px;flex-wrap:wrap}.chips span {border:1px solid #ffffff30;border-radius:30px;padding:5px 12px;font-size:12px;color:#edf8ff}
.result {border-radius:12px;padding:18px 22px;margin:10px 0;border:1px solid #d2dae2;background:#f3f7fa;color:#152e42!important}
.result strong {display:block;font-size:23px;margin-bottom:4px;color:inherit!important}.result.fail {border-color:#e6a5a5;background:#fff2f2;color:#832c35!important}.result.pass {border-color:#8cc9b4;background:#ecfaf4;color:#15543d!important}
.metric-row {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 20px}.metric {padding:20px;border:1px solid #dbe3e9;border-radius:12px;background:#f5f8fa;color:#17334b!important}.metric b {display:block;font-size:30px;color:#17334b!important}.metric span {font-size:12px;color:#476075!important}
@media(max-width:650px){.metric-row{grid-template-columns:repeat(2,1fr)}#hero{padding:24px}#hero h1{font-size:32px}}
"""


def sample_trace(name: str) -> str:
    return json.dumps(EXAMPLES[name], indent=2)


def run_trace(text: str):
    try:
        result = audit_trace(text)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid trace: {str(exc)[:700]}") from exc
    status = result["status"]
    label = {
        "passed": "Execution and supplied outcome match",
        "failed": "Trace check failed",
        "consistent": "Calls match; task outcome unverified",
    }[status]
    tone = "fail" if status == "failed" else "pass"
    outcome = result["answer_matches_expected"]
    answer = "not supplied" if outcome is None else ("matches" if outcome else "does not match")
    card = f'<div class="result {tone}"><strong>{label}</strong>{result["step_failures"]} inconsistent calls / {result["steps_checked"]} checked. Final answer: {answer} the expected outcome.</div>'
    rows = [
        [r["step"], r["tool"], r["recorded"], r["replayed"] or "—", r["status"], r["detail"]]
        for r in result["steps"]
    ]
    return card, rows, result


def inspect_pair(pair_id: str):
    p = PAIR_BY_ID[pair_id]
    a, b = RECORDS[pair_id, "reference"], RECORDS[pair_id, "candidate"]
    successful = a["score"] > b["score"]
    status = "Correct ranking" if successful else "Failure: tied or reversed ranking"
    tone = "pass" if successful else "fail"
    card = f'<div class="result {tone}"><strong>{status}</strong>Reference score: {a["score"]:g}/10 · Corrupted score: {b["score"]:g}/10. Saved Qwen response; no new inference.</div>'
    return (
        p["question"],
        p["context"],
        p["reference"],
        p["candidate"],
        card,
        a["completion"]["text"],
        b["completion"]["text"],
        {"pair": p, "reference_judgment": a, "candidate_judgment": b},
    )


def build_demo() -> gr.Blocks:
    a = PILOT["summary"]["discrimination_accuracy"]
    with gr.Blocks(
        title="JudgeGuard | Agent Evaluation Workbench", delete_cache=(3600, 3600)
    ) as demo:
        gr.HTML(
            '<div id="hero"><div class="eyebrow">AGENT RELIABILITY · OPEN EVIDENCE</div><h1>A correct answer can hide a broken trace.</h1><p>Replay tool calls. Inspect the failure. Then check whether the AI judge noticed.<br>JudgeGuard brings deterministic execution checks and measured model judgments into one workbench.</p><div class="chips"><span>4 replayable tools</span><span>200 real judge responses</span><span>No API key needed</span></div></div>'
        )
        with gr.Tab("01 · Audit an agent trace"):
            gr.Markdown(
                "### Find the failure behind a plausible answer\nLoad **Wrong argument, correct final answer** and run the audit. The final answer matches, but the calculation does not. These examples are constructed traces, not collected autonomous-agent runs."
            )
            with gr.Row():
                with gr.Column(scale=5):
                    selector = gr.Dropdown(
                        list(EXAMPLES),
                        value="Wrong argument, correct final answer",
                        label="Scenario",
                    )
                    trace = gr.Code(
                        value=sample_trace("Wrong argument, correct final answer"),
                        language="json",
                        lines=23,
                        label="Editable trace JSON",
                    )
                    run = gr.Button("Replay and audit trace", variant="primary")
                with gr.Column(scale=6):
                    status = gr.HTML(
                        '<div class="result"><strong>Ready to replay</strong>Every supported call is executed against the fixed tool environment.</div>'
                    )
                    table = gr.Dataframe(
                        headers=["Step", "Tool", "Recorded", "Replayed", "Check", "Detail"],
                        interactive=False,
                        wrap=True,
                        label="Execution evidence",
                    )
                    with gr.Accordion("Audit JSON and scope", open=False):
                        result = gr.JSON(label="Machine-readable audit")
                    gr.Markdown(
                        "**What this proves:** a recorded result matches its declared call; an optional final answer matches the supplied oracle.\n\n**What it does not prove:** that the task was appropriate, all required steps are present, or arbitrary external tools are safe. Calls are replayed independently. No user code, shell commands, URLs or network tools are executed."
                    )
            selector.change(sample_trace, selector, trace, api_name="load_trace")
            run.click(
                run_trace,
                trace,
                [status, table, result],
                api_name="audit_trace",
                concurrency_limit=2,
            )
        with gr.Tab("02 · Inspect the real model benchmark"):
            gr.HTML(
                f'<div class="metric-row"><div class="metric"><b>96 / 100</b><span>correct numeric rankings</span></div><div class="metric"><b>{a["lo"] * 100:.1f}–{a["hi"] * 100:.1f}%</b><span>95% article-cluster interval</span></div><div class="metric"><b>4 ties</b><span>all counted as failures</span></div><div class="metric"><b>$0</b><span>paid API charges · local GPU</span></div></div>'
            )
            gr.Markdown(
                "**Qwen3-4B Q4_K_M · 100 SQuAD numeric pairs · 28 articles · 200 saved responses.** This is a completed local-model study, not a live model call or general RAG accuracy claim. The four failures appear first. Public-benchmark contamination is possible; human review is pending."
            )
            choices = [
                (
                    f"{'FAIL · ' if p in FAILURES else 'PASS · '}{p['source_id']} — {p['question'][:85]}",
                    p["id"],
                )
                for p in ORDERED_PAIRS
            ]
            pair = gr.Dropdown(choices, value=ORDERED_PAIRS[0]["id"], label="Inspect a test case")
            first = inspect_pair(ORDERED_PAIRS[0]["id"])
            question = gr.Textbox(value=first[0], label="Question", interactive=False)
            context = gr.Textbox(value=first[1], label="Source passage", lines=6, interactive=False)
            with gr.Row():
                ref = gr.Textbox(
                    value=first[2], label="Human-annotated reference", interactive=False
                )
                bad = gr.Textbox(
                    value=first[3], label="Constructed numeric corruption", interactive=False
                )
            verdict = gr.HTML(first[4])
            with gr.Row():
                reasoning_a = gr.Textbox(
                    value=first[5],
                    label="Judge's reasoning · reference",
                    lines=5,
                    interactive=False,
                )
                reasoning_b = gr.Textbox(
                    value=first[6],
                    label="Judge's reasoning · corruption",
                    lines=5,
                    interactive=False,
                )
            with gr.Accordion("Source offsets, raw responses and provenance", open=False):
                raw = gr.JSON(first[7])
            pair.change(
                inspect_pair,
                pair,
                [question, context, ref, bad, verdict, reasoning_a, reasoning_b, raw],
                api_name="inspect_pair",
            )
        with gr.Tab("03 · Try a live model"):
            gr.Markdown(
                "### Ask a small judge, then inspect its reasoning\n"
                "**Qwen3-0.6B · live GPU inference · experimental.** This smaller model is "
                "different from the Qwen3-4B pilot. Its opinion can be wrong and does not "
                "control any blocking decision. Only this tab consumes free ZeroGPU quota; "
                "queues and daily limits apply. The other tabs work without inference.\n\n"
                "Edit the public example or enter non-sensitive material. Your inputs are "
                "processed on Hugging Face; JudgeGuard does not add them to the published dataset."
            )
            live_source = gr.Textbox(
                value="The warehouse stores 4200 pallets. Each pallet weighs 27.5 kg.",
                label="Source (up to 4000 characters)",
                lines=4,
            )
            live_question = gr.Textbox(
                value="How many pallets does the warehouse store?",
                label="Question (up to 600 characters)",
            )
            live_answer = gr.Textbox(
                value="4400 pallets", label="Answer to judge (up to 600 characters)"
            )
            live_run = gr.Button(
                "Run live judge", variant="primary", interactive=LIVE_JUDGE is not None
            )
            live_response = gr.Textbox(label="Model opinion — verify against the source", lines=5)
            live_meta = gr.JSON(label="This response's provenance")
            if LIVE_JUDGE is not None:
                live_run.click(
                    LIVE_JUDGE,
                    [live_source, live_question, live_answer],
                    [live_response, live_meta],
                    api_name="judge_live",
                    concurrency_limit=1,
                )
            else:
                gr.Markdown(
                    "Live inference is available in the hosted ZeroGPU Space. Local replay and saved evidence need no model download."
                )
        with gr.Tab("04 · Evidence and release criteria"):
            gr.Markdown(
                "## Measured, replayed, or simulated?\n\n| Component | Evidence | Status |\n|---|---|---|\n| Trace auditor | Actual deterministic tool execution | Available interactively |\n| Numeric QA pilot | 200 actual local Qwen responses | Published with raw evidence |\n| Historical four-judge study | Simulated model responses | Research archive only |\n| Distilled student | Simulation-trained classifier | **Not approved for production blocking** |\n\n### The failed gate stays visible\nThe corrected 200-item student rebuild scored **56.25% accuracy**, with **54.39% block precision** and **65% false blocks** on 80 held-out rows. It does not satisfy the existing promotion criteria. No threshold was weakened to make it pass. The workbench release tests do not certify the student.\n\n### Reproduce and extend\nRun the local model, collect checkpoints, inspect failures and version the dataset. New domains need their own labels and source-separated validation. Replay and evidence inspection need no GPU allocation. The separate live tab runs an uncalibrated Qwen3-0.6B judge on ZeroGPU, subject to free quotas. The 4B pilot remains a separately recorded study.\n\n[Source and setup](https://github.com/RyanSingh0/judgeguard) · [Full model provenance](https://github.com/RyanSingh0/judgeguard/blob/main/results/pilot_live.json) · [Raw judgments](https://github.com/RyanSingh0/judgeguard/blob/main/results/pilot-evidence/judgments.jsonl) · [Dataset attribution](https://github.com/RyanSingh0/judgeguard/blob/main/data/README.md)"
            )
            gr.JSON(PILOT, label="Published study manifest", open=False)
            gr.Markdown(
                "Code: MIT. SQuAD-derived data: CC BY-SA 4.0. Qwen weights: Apache-2.0. The hosted live tab downloads the pinned 0.6B model; the recorded pilot used 4B.\n\nBuilt by **Aryan Meena** · [GitHub](https://github.com/RyanSingh0)"
            )
    return demo


demo = build_demo()
if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=CSS,
        show_error=False,
    )
