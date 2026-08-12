"""The service. Two paths, deliberately not interchangeable.

``POST /guard``     inline. The distilled student only. No network call happens
                    inside the request, which is the only way a hard millisecond
                    budget can be honoured. Returns allow/block plus the latency
                    it actually took and whether that was within budget.

``POST /evaluate``  asynchronous. The full LLM judge, with reasoning and with the
                    measured reliability of that judge attached, so a caller can
                    see how much to trust the score it just received.

``GET /report``     the whole bias battery as JSON, served from committed files
                    so it works with no API key and no live quota.

Confusing the two paths is a real production failure: an async evaluator
deployed where a guardrail was needed means harmful output reaches the user
while the evaluation is still in flight.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from judgeguard import __version__
from judgeguard.config import get_settings, load_registry
from judgeguard.data.schema import Item
from judgeguard.distill.features import structural_features
from judgeguard.distill.train import Student
from judgeguard.judges.parse import parse_score
from judgeguard.judges.prompts import SYSTEM, score_prompt
from judgeguard.providers.registry import complete, mode_banner
from judgeguard.serve.latency import RECORDER
from judgeguard.serve.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    GuardRequest,
    GuardResponse,
    HealthResponse,
)
from judgeguard.store import RESULTS_DIR, json_safe, load
from judgeguard.telemetry import configure_logging, get_logger

configure_logging()
log = get_logger("serve")
STATIC = Path(__file__).parent / "static"
START = time.time()

app = FastAPI(
    title="JudgeGuard",
    version=__version__,
    description=(
        "Measuring LLM judge reliability with degradation-generated gold labels, "
        "then serving the distilled judge as an inline guardrail."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_student: Student | None = None
_report_cache: dict[str, Any] = {}


def get_student() -> Student:
    global _student
    if _student is None:
        path = RESULTS_DIR / "student_model.joblib"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="Student model not found. Run `python experiments/09_distill.py` first.",
            )
        _student = Student.load(path)
        log.info(
            "student_loaded", featurizer=_student.featurizer_kind, threshold=_student.threshold
        )
    return _student


def _reliability_for(judge: str) -> dict[str, Any]:
    """What the battery measured about this judge, attached to its verdict."""
    out: dict[str, Any] = {"judge": judge}
    try:
        d = load("01_discrimination")
        pj = d["per_judge"].get(judge)
        if pj:
            worst = min(pj["by_degradation"], key=lambda g: pj["by_degradation"][g]["value"])
            out["discrimination_accuracy"] = pj["overall_accuracy"]
            out["weakest_degradation"] = {"type": worst, **pj["by_degradation"][worst]}
    except FileNotFoundError:
        pass
    try:
        p = load("02_position_bias")["per_judge"].get(judge)
        if p:
            out["position_inconsistency_rate"] = p["inconsistency_rate"]
    except FileNotFoundError:
        pass
    try:
        c = load("06_self_consistency")["per_judge"].get(judge)
        if c:
            out["accept_reject_flip_rate_at_temp_0_7"] = c["accept_reject_flip_rate"]
    except FileNotFoundError:
        pass
    out["caveat"] = (
        "These are the measured properties of this judge on this battery. A single score "
        "from it should be read with them in view."
    )
    return out


# ------------------------------------------------------------------ endpoints
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    path = STATIC / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>JudgeGuard</h1><p>UI not built.</p>")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = sorted(p.stem for p in RESULTS_DIR.glob("*.json"))
    return HealthResponse(
        status="ok",
        version=__version__,
        provider_mode=mode_banner(),
        student_loaded=(RESULTS_DIR / "student_model.joblib").exists(),
        results_available=available,
        uptime_seconds=time.time() - START,
    )


@app.post("/guard", response_model=GuardResponse)
def guard(req: GuardRequest, response: Response) -> GuardResponse:
    """Inline path. Hard budget, no network call."""
    budget = get_settings().guard_p99_budget_ms
    with RECORDER.time("/guard"):
        student = get_student()
        thr = req.threshold if req.threshold is not None else student.threshold
        t0 = time.perf_counter()
        p = float(
            student.predict_proba([f"QUESTION: {req.question}\n[SEP]\nANSWER: {req.answer}"])[0]
        )
        dt = (time.perf_counter() - t0) * 1000

    feats = structural_features(req.answer)
    signals = {
        "length_chars": float(len(req.answer)),
        "type_token_ratio": float(feats[2]),
        "hedge_density": float(feats[3]),
        "filler_phrase_count": float(feats[4]),
        "numeric_density": float(feats[5]),
    }
    decision = "allow" if p >= thr else "block"
    if decision == "block":
        drivers = []
        if signals["hedge_density"] > 0.02:
            drivers.append("high hedge-word density")
        if signals["filler_phrase_count"] >= 2:
            drivers.append("content-free filler phrases")
        if signals["numeric_density"] < 0.005:
            drivers.append("few committed numeric claims")
        reason = f"p(acceptable)={p:.3f} below threshold {thr:.2f}" + (
            f"; contributing signals: {', '.join(drivers)}" if drivers else ""
        )
    else:
        reason = f"p(acceptable)={p:.3f} at or above threshold {thr:.2f}"

    response.headers["X-JudgeGuard-Latency-Ms"] = f"{dt:.3f}"
    response.headers["X-JudgeGuard-Budget-Ms"] = f"{budget:.0f}"
    response.headers["X-JudgeGuard-Path"] = "inline-guardrail"
    return GuardResponse(
        decision=decision,
        probability_good=p,
        threshold=thr,
        latency_ms=dt,
        budget_ms=budget,
        within_budget=dt <= budget,
        reason=reason,
        signals=signals,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Async path. Full LLM judge plus that judge's measured reliability."""
    reg = load_registry()
    judge = req.judge or reg.panel[0]
    if judge not in reg.aliases:
        raise HTTPException(status_code=400, detail=f"unknown judge {judge!r}; try {reg.panel}")
    spec = reg.by_alias(judge)
    item = Item(id="live", domain="live", context=req.context, question=req.question, reference="")
    prompt = score_prompt(req.config, item, req.answer)
    with RECORDER.time("/evaluate"):
        c = complete(
            judge,
            prompt,
            temperature=0.0,
            max_tokens=spec.token_budget(320 if req.config == "cot" else 160),
            system=SYSTEM,
            meta={"task": "score", "config": req.config, "uid": "live"},
        )
    score = parse_score(c.text)
    return EvaluateResponse(
        judge=judge,
        config=req.config,
        score=score,
        parsed_ok=score is not None,
        reasoning=c.text.strip(),
        latency_ms=c.latency_ms,
        cost_usd=spec.cost_usd(c.prompt_tokens, c.completion_tokens),
        model_served=c.model_served,
        simulated=c.simulated,
        judge_reliability=_reliability_for(judge),
    )


@app.get("/report")
def report(experiment: str | None = None) -> JSONResponse:
    """The bias battery, served from committed results."""
    if experiment:
        try:
            return JSONResponse(json_safe(load(experiment)))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not _report_cache:
        for p in sorted(RESULTS_DIR.glob("*.json")):
            if p.stem == "degraded_set":
                continue  # large; fetch explicitly via ?experiment=
            try:
                _report_cache[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return JSONResponse(
        json_safe(
            {
                "provider_mode": mode_banner(),
                "experiments": sorted(_report_cache),
                "results": _report_cache,
            }
        )
    )


@app.get("/summary")
def summary() -> dict[str, Any]:
    """The three findings, computed from results rather than typed by hand."""
    out: dict[str, Any] = {"provider_mode": mode_banner(), "findings": []}
    try:
        tr = load("08_trajectory_blindness")
        rows = [
            (j, tr["per_judge"][j]["by_degradation"]["wrong_argument"])
            for j in tr["judges"]
            if "wrong_argument" in tr["per_judge"][j]["by_degradation"]
        ]
        worst = max(rows, key=lambda r: r[1]["miss_rate"])
        acc = worst[1]["detection_accuracy"]
        out["findings"].append(
            {
                "id": "argument_blindness",
                "headline": (
                    f"{worst[0]} failed to detect malformed tool arguments in "
                    f"{100 * worst[1]['miss_rate']:.1f}% of agent trajectories "
                    f"(detection accuracy {100 * acc['value']:.1f}%, 95% CI "
                    f"[{100 * acc['lo']:.1f}, {100 * acc['hi']:.1f}])"
                ),
                "why": "A failure class invisible to output-only evaluation by construction.",
            }
        )
    except FileNotFoundError:
        pass
    try:
        pos = load("02_position_bias")
        w = max(pos["judges"], key=lambda j: pos["per_judge"][j]["inconsistency_rate"]["value"])
        a = pos["per_judge"][w]
        out["findings"].append(
            {
                "id": "position_bias",
                "headline": (
                    f"{w} changed its verdict on {100 * a['inconsistency_rate']['value']:.1f}% of "
                    f"comparisons when the two options were swapped; fixed-order evaluation "
                    f"reported {100 * a['accuracy_reference_first']['value']:.1f}% where "
                    f"swap-and-average gives {100 * a['accuracy_swap_and_average']['value']:.1f}%"
                ),
                "why": "The number a naive harness reports is inflated by the bias it did not control.",
            }
        )
    except FileNotFoundError:
        pass
    try:
        d = load("09_distill")
        g = d["accuracy_against_ground_truth"]["student_minus_teacher_oracle_cut"]
        e = d["economics"]
        out["findings"].append(
            {
                "id": "distillation",
                "headline": (
                    f"The distilled student was {d['accuracy_against_ground_truth']['verdict']} "
                    f"(Δ={100 * g['diff']:+.1f} pp, 95% CI [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}]) "
                    f"while running {e['latency_speedup']:.0f}x faster at zero marginal cost"
                ),
                "why": "An accuracy gap inside the interval is not an accuracy gap.",
            }
        )
    except FileNotFoundError:
        pass
    return out


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return RECORDER.prometheus()


@app.get("/latency")
def latency() -> dict[str, Any]:
    return RECORDER.snapshot()


@app.get("/figures/{name}", include_in_schema=False)
def figure(name: str) -> Response:
    safe = Path(name).name
    p = RESULTS_DIR / "figures" / safe
    if not p.exists() or p.suffix != ".png":
        raise HTTPException(status_code=404, detail=f"no figure {safe!r}")
    return Response(p.read_bytes(), media_type="image/png")


@app.get("/examples")
def examples() -> dict[str, Any]:
    """Prefilled demo payloads so the UI is one click, not one paragraph of typing."""
    from judgeguard.data.load import load_items
    from judgeguard.degrade.text import hedging, numeric_swap, omission, verbosity

    # Drawn from the same 500-item corpus the battery uses, so the demo shows the
    # guardrail on in-distribution text rather than on an item whose shape it has
    # never seen. (A 3-item corpus quietly changes the generator's domain balance.)
    item = load_items(500)[7]
    return {
        "question": item.question,
        "context": item.context,
        "candidates": [
            {
                "label": "reference (known good)",
                "expect": "allow",
                "note": "The correct answer. A guardrail that blocks this is worse than useless.",
                "text": item.reference,
            },
            {
                "label": "hedging (vague, uncheckable)",
                "expect": "block",
                "note": "Committed values replaced by vagueness. Caught by absolute surface signature.",
                "text": hedging(item, 0.6, seed=1).text,
            },
            {
                "label": "omission (a required fact removed)",
                "expect": "block",
                "note": "Caught via coverage of the source's content words.",
                "text": omission(item, 0.5, seed=1).text,
            },
            {
                "label": "numeric swap — the known blind spot",
                "expect": "allow (wrongly)",
                "note": (
                    "One number changed. The guardrail scores this identically to the "
                    "reference, because the source states which quantities were reported "
                    "but not their values. No reference-free string model can catch it. "
                    "This is what the cascade escalates to the LLM judge."
                ),
                "text": numeric_swap(item, 0.5, seed=1).text,
            },
            {
                "label": "padded reference (no error, just 3x longer)",
                "expect": "block",
                "note": (
                    "Mirror image of the judges' bias: they scored padding UP by as much as "
                    "1.5 points, the student blocks it outright. Neither is neutral about length."
                ),
                "text": verbosity(item, 0.9, seed=1).text,
            },
        ],
    }


try:  # pragma: no cover - optional dependency
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
except Exception:
    pass
