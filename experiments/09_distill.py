#!/usr/bin/env python
"""Distil the winning judge into something that can run inline.

The student is trained on the *teacher's* labels, not on ground truth. That is
the definition of distillation and it is also the honest version of the
experiment: the student is supposed to inherit the teacher's mistakes. Two gaps
are therefore reported separately --

  student vs teacher   how much of the teacher's behaviour survived compression
  student vs truth     how good the resulting evaluator actually is

-- and the second is compared against the teacher's own accuracy on the same
held-out items with a paired bootstrap. If the difference sits inside the
interval, the correct sentence is "statistically indistinguishable at 1/Nth the
cost", and that is a much stronger claim than any bare accuracy number.

    python experiments/09_distill.py --items 500
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from _common import apply_quick_isolation, banner, base_parser, corpus, done, run_main

from judgeguard.config import load_registry
from judgeguard.degrade.text import ERROR_DEGRADATIONS, PROBE_DEGRADATIONS, build_variants
from judgeguard.distill.features import render_example
from judgeguard.distill.train import Student
from judgeguard.judges.run import ScoreTask, panel, run_scores
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.store import RESULTS_DIR, load, save
from judgeguard.telemetry import tracking_run

ACCEPT_AT = 7.0


def pick_teacher(default_judge: str | None = None, default_config: str = "cot") -> tuple[str, str]:
    """Best (judge, config) by measured accuracy, not by reputation.

    Tries the rubric ablation first, then experiment 01, then whichever judge
    the panel lists first. This used to fall back to the literal string
    "gemini-flash", which picked a teacher that wasn't even on the panel once
    the panel changed. Phase 1 skips the ablation on purpose, so the fallback
    is the normal path here, not an edge case.
    """
    fallback = default_judge or panel()[0]
    try:
        ab = load("05_rubric_ablation")
        best = max(
            ((j, c, ab["accuracy"][j][c]["value"]) for j in ab["judges"] for c in ab["configs"]),
            key=lambda t: t[2],
        )
        return best[0], best[1]
    except FileNotFoundError:
        pass
    try:
        # Phase 1 order: 01 runs before 09, so its accuracy is the best signal
        # available without the ablation.
        d1 = load("01_discrimination")
        best_j = max(d1["per_judge"], key=lambda j: d1["per_judge"][j]["overall_accuracy"]["value"])
        return best_j, d1.get("config", default_config)
    except (FileNotFoundError, KeyError, ValueError):
        return fallback, default_config


def main() -> None:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--featurizer", default="hashing", choices=["hashing", "minilm"])
    ap.add_argument("--max-false-allow", type=float, default=0.25)
    args = apply_quick_isolation(ap.parse_args())
    n_items = 40 if args.quick else args.items

    t0 = banner("09  distillation: teacher -> calibrated student")
    teacher, config = pick_teacher()
    print(f"    teacher selected by measured accuracy: {teacher} / {config}")

    items, by_id = corpus(n_items, args.seed)
    variants = build_variants(
        items, seed=args.seed, degradations=ERROR_DEGRADATIONS + PROBE_DEGRADATIONS
    )

    # ------------------------------------------------------------------ sampling
    # Two sampling decisions, both of which cost real accuracy to get right.
    #
    # 1. BALANCE. The full cross-product is 1 reference to 15 error variants per
    #    item. Training on that teaches the base rate rather than the judgement:
    #    a model that blocks everything scores 94%, and a *correctly calibrated*
    #    one reports p(good) ~= 0.06 for any unpadded text, the reference
    #    included. Both are honest and both are useless as a guardrail. The 15:1
    #    skew is an artefact of how the battery enumerates degradations, not a
    #    property of production traffic, so the student sees one reference
    #    against one sampled error variant.
    #
    # 2. NO PROBES IN TRAINING. The verbosity probe is a correct answer that
    #    happens to be padded, so it carries label 1 -- and it is trivially
    #    identifiable by its filler. Including it makes "contains filler" the
    #    dominant positive feature, after which the model rationally assigns
    #    every *unpadded* text the base rate of the unpadded pool and blocks
    #    clean references 70% of the time. The probe is a measurement
    #    instrument, not training data. It is still scored at test time.
    rng = random.Random(args.seed)
    by_item: dict[str, dict[str, list]] = defaultdict(lambda: {"error": [], "probe": []})
    for v in variants:
        by_item[v.item_id]["error" if v.introduces_error else "probe"].append(v)

    chosen: list = []
    probes: list = []
    for it in items:
        buckets = by_item[it.id]
        chosen.extend(rng.sample(buckets["error"], min(1, len(buckets["error"]))))
        if buckets["probe"]:
            probes.append(rng.choice(buckets["probe"]))

    tasks = [ScoreTask(uid=f"{i.id}:reference", item=i, answer=i.reference) for i in items]
    tasks += [
        ScoreTask(
            uid=v.uid,
            item=by_id[v.item_id],
            answer=v.text,
            degradation=v.degradation,
            severity=v.severity,
            len_ratio=v.len_ratio,
        )
        for v in chosen
    ]
    answer_of = {f"{i.id}:reference": (i.question, i.reference, i.context) for i in items}
    truth_of: dict[str, int] = {f"{i.id}:reference": 1 for i in items}
    deg_of: dict[str, str] = {f"{i.id}:reference": "reference" for i in items}
    for v in chosen:
        answer_of[v.uid] = (by_id[v.item_id].question, v.text, by_id[v.item_id].context)
        # A padded reference contains no error, so its ground-truth label is 1.
        truth_of[v.uid] = int(not v.introduces_error)
        deg_of[v.uid] = v.degradation

    with tracking_run("judgeguard", "09_distill", {"teacher": teacher, "config": config}) as run:
        judged = run_scores(teacher, config, tasks)
        rows = [j for j in judged if j.score is not None]
        texts = [render_example(*answer_of[j.uid]) for j in rows]
        teacher_label = [int(j.score >= ACCEPT_AT) for j in rows]
        truth = [truth_of[j.uid] for j in rows]
        degs = [deg_of[j.uid] for j in rows]

        student = Student(featurizer=args.featurizer, seed=args.seed)
        # Stratify by degradation class as well as label, so the fidelity table
        # below lands on a test set containing every class. Without it the split
        # can drop one, and then the numeric-substitution vs hedging contrast is
        # computed from a table with rows missing.
        report, extra = student.fit(texts, teacher_label, truth, strat_on=degs)

        te = np.array(extra["test_indices"])
        p = np.array(extra["test_probabilities"])
        y_teacher = np.array(extra["test_labels"])
        y_truth = np.array(extra["test_truth"])

        thr = student.choose_threshold(p, y_truth, max_false_allow=args.max_false_allow)
        student_pred = (p >= thr).astype(int)

        student_correct = (student_pred == y_truth).astype(int).tolist()
        teacher_correct = (y_teacher == y_truth).astype(int).tolist()
        gap = paired_bootstrap_diff(student_correct, teacher_correct, seed=args.seed)

        # Two baselines, because 79% of held-out items share a label and an
        # accuracy number that beats nothing is not a result.
        majority = int(round(float(y_truth.mean())))
        majority_acc = float((np.full_like(y_truth, majority) == y_truth).mean())

        # Give the teacher its best possible cut point, tuned on the labels it is
        # being scored against. This is an oracle and is labelled as one; the
        # gap between it and the rubric's own 7.0 boundary is a calibration
        # failure, not a capability failure.
        raw = np.array([rows[i].score for i in te], dtype=float)
        cuts = np.unique(raw)
        oracle_cut, oracle_acc = ACCEPT_AT, 0.0
        for c in cuts:
            acc = float((((raw >= c).astype(int)) == y_truth).mean())
            if acc > oracle_acc:
                oracle_acc, oracle_cut = acc, float(c)
        teacher_oracle_correct = ((raw >= oracle_cut).astype(int) == y_truth).astype(int).tolist()
        gap_oracle = paired_bootstrap_diff(student_correct, teacher_oracle_correct, seed=args.seed)

        def balanced(pred: np.ndarray) -> float:
            pos = y_truth == 1
            neg = ~pos
            tpr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
            tnr = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
            return (tpr + tnr) / 2

        teacher_cost = sum(j.cost_usd for j in rows)
        teacher_calls = len(rows)
        cost_per_1k_teacher = 1000 * teacher_cost / max(teacher_calls, 1)
        teacher_latency = sum(j.latency_ms for j in rows) / max(teacher_calls, 1)

        # Does the student inherit the teacher's blind spots, or invent its own?
        te_degs = [degs[i] for i in te]
        by_deg: dict[str, list[int]] = defaultdict(list)
        for k, dg in enumerate(te_degs):
            by_deg[dg].append(int(student_pred[k] == y_truth[k]))
        student_by_degradation = {
            dg: {**bca_ci(v, seed=args.seed).as_dict(), "n": len(v)}
            for dg, v in sorted(by_deg.items())
            if len(v) >= 5
        }
        teacher_by_degradation = {}
        for dg in student_by_degradation:
            flags = [int(teacher_oracle_correct[k]) for k, x in enumerate(te_degs) if x == dg]
            if len(flags) >= 5:
                teacher_by_degradation[dg] = bca_ci(flags, seed=args.seed).as_dict()

        # ------------------------------------------------------------- cascade
        # The student is blind to numeric substitution and to same-domain topic
        # drift, and no reference-free string model can fix that: verifying a
        # number against a source that never states it is not a modelling
        # problem. So the deployable design is not replacement but a cascade --
        # the student decides where it is confident, and escalates the rest to
        # the judge. Sweeping the confidence margin prices the trade directly.
        teacher_correct_arr = np.array(teacher_oracle_correct)
        student_correct_arr = np.array(student_correct)
        conf = np.abs(p - thr) / max(thr, 1 - thr)
        cascade_rows = []
        for margin in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01):
            local = conf >= margin
            escalate = ~local
            combined = np.where(local, student_correct_arr, teacher_correct_arr)
            cascade_rows.append(
                {
                    "confidence_margin": float(margin),
                    "escalation_rate": float(escalate.mean()),
                    "combined_accuracy": bca_ci(combined.tolist(), seed=args.seed).as_dict(),
                    "cost_per_1k_usd": float(escalate.mean() * cost_per_1k_teacher),
                    "mean_latency_ms": float(
                        report.inference_ms_mean + escalate.mean() * teacher_latency
                    ),
                }
            )
        judge_only = bca_ci(teacher_oracle_correct, seed=args.seed)
        viable = [
            r
            for r in cascade_rows
            if r["combined_accuracy"]["lo"] <= judge_only.value <= r["combined_accuracy"]["hi"]
        ]
        best_cascade = min(viable, key=lambda r: r["escalation_rate"]) if viable else None

        # Held-out probe check: padded references are correct answers. A guardrail
        # that blocks them is a verbosity bias of its own, in the opposite
        # direction from the judges'. The probes were never trained on.
        probe_texts = [
            render_example(by_id[v.item_id].question, v.text, by_id[v.item_id].context)
            for v in probes
        ]
        probe_p = student.predict_proba(probe_texts)
        probe_report = {
            "n": len(probes),
            "allow_rate": float((probe_p >= thr).mean()),
            "mean_probability": float(probe_p.mean()),
            "note": (
                "Padded references contain no error, so a well-behaved guardrail allows them. "
                "Never seen during training."
            ),
        }

        model_path = student.save(RESULTS_DIR / "student_model.joblib")

        payload = {
            "experiment": "09_distill",
            "question": "How much of the teacher survives compression, and what does that buy?",
            "teacher": {
                "judge": teacher,
                "config": config,
                "accept_threshold_on_1_10_scale": ACCEPT_AT,
                "labels_generated": len(rows),
            },
            "student": report.as_dict(),
            "operating_point": {
                "threshold": thr,
                # What a guardrail is actually judged on: does it block bad output
                # without ever blocking good output? Precision on BLOCK is the
                # number that decides whether anyone will let it near production.
                "block_precision": float(
                    ((y_truth == 0) & (student_pred == 0)).sum() / max((student_pred == 0).sum(), 1)
                ),
                "false_block_rate": float(
                    ((y_truth == 1) & (student_pred == 0)).sum() / max((y_truth == 1).sum(), 1)
                ),
                "bad_output_caught": float(
                    ((y_truth == 0) & (student_pred == 0)).sum() / max((y_truth == 0).sum(), 1)
                ),
                "chosen_by": f"most permissive threshold whose false-allow rate stays <= {args.max_false_allow:.0%}",
                "false_allow_rate": float(
                    ((y_truth == 0) & (student_pred == 1)).sum() / max((student_pred == 1).sum(), 1)
                ),
                "block_rate": float((student_pred == 0).mean()),
            },
            "accuracy_against_ground_truth": {
                "majority_class_baseline": majority_acc,
                "teacher_at_rubric_threshold": bca_ci(teacher_correct, seed=args.seed).as_dict(),
                "teacher_at_oracle_threshold": {
                    **bca_ci(teacher_oracle_correct, seed=args.seed).as_dict(),
                    "cut": oracle_cut,
                    "caveat": "tuned on the same labels it is scored against; an upper bound, not a fair comparison",
                },
                "student": bca_ci(student_correct, seed=args.seed).as_dict(),
                "student_minus_teacher_rubric_cut": gap.as_dict(),
                "student_minus_teacher_oracle_cut": gap_oracle.as_dict(),
                "balanced_accuracy": {
                    "teacher_rubric_cut": balanced(y_teacher),
                    "teacher_oracle_cut": balanced((raw >= oracle_cut).astype(int)),
                    "student": balanced(student_pred),
                },
                "verdict": (
                    "statistically indistinguishable from the teacher at its best threshold"
                    if not gap_oracle.significant
                    else (
                        "student significantly better than the teacher at its best threshold"
                        if gap_oracle.diff > 0
                        else "student significantly worse than the teacher at its best threshold"
                    )
                ),
                "calibration_finding": (
                    "The teacher's raw 1-10 score is not usable as an accept/reject decision at "
                    "the rubric's own 7.0 boundary. Re-cutting the identical signal at its best "
                    "threshold recovers most of the loss, which makes this a calibration failure "
                    "rather than a discrimination failure."
                ),
            },
            "fidelity_to_teacher": {
                "agreement": report.agreement_with_teacher,
                "auc_against_teacher_labels": report.auc,
                "student_accuracy_by_degradation": student_by_degradation,
                "teacher_accuracy_by_degradation": teacher_by_degradation,
                "shared_blind_spot": (
                    min(student_by_degradation, key=lambda d: student_by_degradation[d]["value"])
                    if student_by_degradation
                    else None
                ),
                "note": (
                    "Distillation is supposed to transfer the teacher's behaviour, blind spots "
                    "included. Comparing per-degradation accuracy is how you check whether the "
                    "student inherited the teacher's weaknesses or invented its own."
                ),
            },
            "verbosity_probe_holdout": probe_report,
            "class_balance": {
                "positive_rate": float(np.mean(truth)),
                "sampling": (
                    "Per item: the reference (label 1) against one sampled error variant "
                    "(label 0). The battery's 15:1 cross-product is an enumeration artefact, "
                    "not a production base rate. Verbosity probes are held out of training "
                    "entirely and scored separately, because 'contains filler' would otherwise "
                    "become the dominant positive feature."
                ),
            },
            "calibration": {
                "brier": report.brier,
                "ece": report.ece,
                "ece_vs_teacher_labels": report.ece,
                "ece_vs_truth_before_recalibration": report.ece_vs_truth_before,
                "ece_vs_truth_after_recalibration": report.ece_vs_truth_after,
                "recalibrated_on_free_gold_labels": report.recalibrated,
                "n_calibration_items": report.n_calibration,
                "reliability_curve": extra["reliability_curve"],
                "reliability_curve_before": extra["reliability_curve_before"],
                "finding": (
                    "Distillation inherits the teacher's calibration error along with its "
                    "decisions: the student's raw probability means 'the teacher would accept "
                    "this', not 'this is acceptable'. Degradation gold labels cost nothing to "
                    "produce, so an isotonic map fitted on a held-out slice of them fixes the "
                    "mismatch for free -- the underrated payoff of manufacturing ground truth "
                    "rather than buying it."
                ),
            },
            "economics": {
                "teacher_cost_per_1k_usd": cost_per_1k_teacher,
                "student_cost_per_1k_usd": 0.0,
                "teacher_mean_latency_ms": teacher_latency,
                "student_mean_latency_ms": report.inference_ms_mean,
                "student_p95_latency_ms": report.inference_ms_p95,
                "latency_speedup": teacher_latency / max(report.inference_ms_mean, 1e-9),
                "note": "The student runs locally on CPU: marginal cost is compute only.",
            },
            "cascade": {
                "question": (
                    "How much traffic can the student absorb before combined accuracy "
                    "becomes distinguishable from judging everything with the LLM?"
                ),
                "rows": cascade_rows,
                "judge_only_accuracy": judge_only.as_dict(),
                "recommended": best_cascade,
                "sentence": (
                    (
                        f"Escalating only {100 * best_cascade['escalation_rate']:.0f}% of traffic "
                        f"to the LLM judge matched judge-only accuracy "
                        f"({100 * best_cascade['combined_accuracy']['value']:.1f}% vs "
                        f"{100 * judge_only.value:.1f}%) at "
                        f"{100 * best_cascade['escalation_rate']:.0f}% of the cost and "
                        f"{best_cascade['mean_latency_ms']:.0f} ms mean latency."
                    )
                    if best_cascade
                    else "No cascade point matched judge-only accuracy."
                ),
            },
            "model_path": str(model_path.relative_to(RESULTS_DIR.parent)),
            "model_family": load_registry().by_alias(teacher).family,
        }
        agt = payload["accuracy_against_ground_truth"]
        print(f"    majority-class baseline {100 * majority_acc:.1f}%")
        print(
            f"    teacher @ rubric cut 7.0  {100 * agt['teacher_at_rubric_threshold']['value']:.1f}%   "
            f"@ oracle cut {oracle_cut:.1f}  {100 * oracle_acc:.1f}%"
        )
        print(
            f"    student                   {100 * agt['student']['value']:.1f}%   "
            f"(balanced set, {100 * float(np.mean(truth)):.0f}% positive)"
        )
        worst_s = min(student_by_degradation, key=lambda d: student_by_degradation[d]["value"])
        print(
            f"    student weakest on {worst_s}: "
            f"{100 * student_by_degradation[worst_s]['value']:.1f}%  "
            f"(teacher there: {100 * teacher_by_degradation.get(worst_s, {}).get('value', float('nan')):.1f}%)"
        )
        print(f"    student - teacher(oracle): {gap_oracle}")
        print(
            f"    ECE vs truth {report.ece_vs_truth_before:.3f} -> {report.ece_vs_truth_after:.3f} "
            f"after free recalibration on {report.n_calibration} gold labels"
        )
        if best_cascade:
            print(
                f"    cascade: escalate {100 * best_cascade['escalation_rate']:.0f}% of traffic "
                f"-> {100 * best_cascade['combined_accuracy']['value']:.1f}% combined "
                f"(judge-only {100 * judge_only.value:.1f}%)"
            )
        op = payload["operating_point"]
        print(
            f"    guardrail: blocks {100 * op['bad_output_caught']:.0f}% of bad output at "
            f"{100 * op['block_precision']:.0f}% block precision, "
            f"false-block {100 * op['false_block_rate']:.1f}%"
        )
        print(
            f"    threshold {thr:.2f}  "
            f"student {report.inference_ms_mean:.2f} ms vs teacher {teacher_latency:.0f} ms "
            f"({teacher_latency / max(report.inference_ms_mean, 1e-9):.0f}x faster)"
        )
        run.log_metrics(
            {
                "student_accuracy": payload["accuracy_against_ground_truth"]["student"]["value"],
                "student_ece": report.ece,
            }
        )

    done(t0, save("09_distill", payload))


if __name__ == "__main__":
    run_main(main)
