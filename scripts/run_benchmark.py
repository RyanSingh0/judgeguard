"""Resumable real-model benchmark: bounded calls, raw evidence, no simulation.

Use --backend transformers on a GPU notebook, or --backend api with a configured
alias (including Ollama). No model training occurs in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from judgeguard.data.pilot import EvalPair, load_pairs
from judgeguard.data.schema import Item
from judgeguard.judges.parse import parse_score
from judgeguard.judges.prompts import SYSTEM, score_prompt
from judgeguard.providers.base import Completion


class LocalJudge:
    def __init__(self, model: str, revision: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU notebook runtime for this backend, or use API/Ollama")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
        self.name = model
        self.revision = getattr(self.model.config, "_commit_hash", revision)

    def complete(self, prompt: str, max_tokens: int) -> Completion:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        n = inputs.input_ids.shape[1]
        if n + max_tokens > 8192:
            raise ValueError("Prompt exceeds pilot context budget; no silent truncation")
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        ids = output[0, n:]
        return Completion(
            text=self.tokenizer.decode(ids, skip_special_tokens=True),
            model_requested=self.name,
            model_served=f"{self.name}@{self.revision}",
            prompt_tokens=n,
            completion_tokens=len(ids),
            provider="local-transformers",
            latency_ms=(time.perf_counter() - t0) * 1000,
            finish_reason="length" if len(ids) == max_tokens else "stop",
        )


def summarize(pairs: list[EvalPair], records: list[dict]) -> dict:
    by = {(r["pair_id"], r["role"]): r for r in records}
    flags, sources = [], []
    parse_failures = 0
    for p in pairs:
        a, b = by.get((p.id, "reference")), by.get((p.id, "candidate"))
        if not a or not b:
            continue
        parsed = a["score"] is not None and b["score"] is not None
        parse_failures += not parsed
        flags.append(int(parsed and a["score"] > b["score"]))
        sources.append(p.source_id)
    interval = None
    if flags:
        groups = sorted(set(sources))
        grouped = [[x for x, s in zip(flags, sources, strict=True) if s == g] for g in groups]
        sums = np.array([sum(g) for g in grouped])
        sizes = np.array([len(g) for g in grouped])
        rng = np.random.default_rng(20260731)
        idx = rng.integers(0, len(groups), size=(10000, len(groups)))
        draws = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        interval = {
            "value": float(np.mean(flags)),
            "lo": float(lo),
            "hi": float(hi),
            "method": "article-cluster percentile bootstrap",
            "confidence": 0.95,
            "n_pairs": len(flags),
            "n_clusters": len(groups),
            "caveat": "Small or boundary samples can yield uninformative bootstrap intervals; inspect counts.",
        }
    return {
        "mode": "live",
        "planned_pairs": len(pairs),
        "completed_pairs": len(flags),
        "complete": len(flags) == len(pairs),
        "scoring_rule": "reference > corrupted; ties and parse failures count as failures",
        "discrimination_accuracy": interval,
        "pairs_with_parse_failure": parse_failures,
        "provider_calls_recorded": len(records),
        "prompt_tokens": sum(r["completion"]["prompt_tokens"] for r in records),
        "completion_tokens": sum(r["completion"]["completion_tokens"] for r in records),
        "models_served": sorted({r["completion"]["model_served"] for r in records}),
        "limitations": "Constructed numeric errors on public QA; no production, calibration, or safety guarantee.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["api", "transformers"], default="api")
    ap.add_argument(
        "--model",
        required=True,
        help="Registry alias for API; Hugging Face model ID for transformers",
    )
    ap.add_argument("--revision", default="main")
    ap.add_argument("--max-new-calls", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.max_new_calls < 0 or args.max_tokens < 32:
        ap.error("call budget must be nonnegative and token budget at least 32")
    pairs = load_pairs(args.data)
    signature = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "backend": args.backend,
        "model": args.model,
        "revision": args.revision,
        "max_tokens": args.max_tokens,
        "prompt_sha256": hashlib.sha256(
            (
                SYSTEM
                + score_prompt(
                    "rubric", Item(id="x", domain="x", context="", question="", reference=""), ""
                )
            ).encode()
        ).hexdigest(),
    }
    if args.backend == "api":
        from judgeguard.config import load_registry

        registry = load_registry()
        spec = registry.by_alias(args.model)
        signature["resolved_model"] = spec.id
        signature["provider"] = spec.provider
        signature["base_url"] = registry.providers[spec.provider].base_url
    if args.dry_run:
        print(
            json.dumps(
                {
                    **signature,
                    "pairs": len(pairs),
                    "total_calls": 2 * len(pairs),
                    "max_new_calls_this_run": args.max_new_calls,
                    "network_calls": 0,
                },
                indent=2,
            )
        )
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError(
            "Output folder belongs to different data/model/prompt settings; use a new folder"
        )
    manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    raw_path = args.out / "judgments.jsonl"
    records = (
        [json.loads(s) for s in raw_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if raw_path.exists()
        else []
    )
    if any(r["completion"]["simulated"] for r in records):
        raise ValueError("Simulated checkpoint cannot be used as live evidence")
    done = {(r["pair_id"], r["role"]) for r in records}
    local = None
    calls, status = 0, 0
    try:
        for p in pairs:
            for role, answer in (("reference", p.reference), ("candidate", p.candidate)):
                if (p.id, role) in done or calls >= args.max_new_calls:
                    continue
                item = Item(
                    id=p.id,
                    domain="numeric_qa",
                    context=p.context,
                    question=p.question,
                    reference=p.reference,
                )
                prompt = score_prompt("rubric", item, answer)
                if args.backend == "transformers":
                    local = local or LocalJudge(args.model, args.revision)
                    c = local.complete(prompt, args.max_tokens)
                else:
                    from judgeguard.providers.registry import complete, is_simulated

                    if is_simulated(args.model):
                        raise ValueError(
                            "Set JUDGEGUARD_PROVIDER_MODE=live; this benchmark rejects simulation"
                        )
                    c = complete(
                        args.model,
                        prompt,
                        system=SYSTEM,
                        max_tokens=args.max_tokens,
                        use_cache=False,
                    )
                if c.simulated:
                    raise ValueError("Refusing simulated response")
                calls += 1
                record = {
                    "pair_id": p.id,
                    "source_id": p.source_id,
                    "role": role,
                    "score": parse_score(c.text),
                    "completion": c.model_dump(),
                    "created_at": datetime.now(UTC).isoformat(),
                }
                with raw_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                records.append(record)
                print(
                    f"{len(records)}/{2 * len(pairs)} saved; {p.id} {role}: {record['score']}",
                    flush=True,
                )
    except Exception as exc:
        print(f"Stopped: {type(exc).__name__}: {str(exc)[:180]}. Completed responses are saved.")
        status = 1
    summary = summarize(pairs, records)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return status or (0 if summary["complete"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
