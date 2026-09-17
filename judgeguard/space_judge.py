"""Optional genuine ZeroGPU inference. Imported only in the hosted Space."""

import time
from typing import Any, cast

import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from judgeguard.live_prompt import MODEL_ID, MODEL_REVISION, judge_messages

# ZeroGPU requires CUDA placement during startup so its emulator can capture weights.
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
MODEL = (
    cast(Any, AutoModelForCausalLM)
    .from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to("cuda")
)
MODEL.eval()


@spaces.GPU(duration=45)
def judge_live(context: str, question: str, answer: str):
    messages = judge_messages(context, question, answer)
    prompt = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = TOKENIZER(prompt, return_tensors="pt")
    prompt_tokens = inputs["input_ids"].shape[-1]
    if prompt_tokens > 3072:
        raise ValueError("Input exceeds 3072 model tokens. Shorten it; no evidence was truncated.")
    inputs = inputs.to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    ids = output[0, prompt_tokens:].tolist()
    decoded = TOKENIZER.decode(ids, skip_special_tokens=True)
    if not isinstance(decoded, str):
        raise RuntimeError("The tokenizer returned an unexpected output type.")
    text = decoded.strip()
    stopped = bool(ids and ids[-1] == TOKENIZER.eos_token_id)
    return text, {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "simulated": False,
        "cached": False,
        "device": "cuda",
        "prompt_tokens": prompt_tokens,
        "output_tokens": len(ids),
        "finish_reason": "stop" if stopped else "length",
        "inference_seconds": round(time.perf_counter() - started, 3),
        "interpretation": "Uncalibrated model opinion; not a guardrail decision or the 4B pilot.",
    }
