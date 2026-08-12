#!/usr/bin/env python
"""Does prompt structure actually buy discrimination?

Three pointwise configurations on identical items: a vague one-liner, an explicit
rubric with score bands, and the same rubric with reason-before-scoring. Paired
throughout -- every configuration sees every item -- so McNemar and the paired
bootstrap both apply, and Holm-Bonferroni controls the family-wise error rate
across the whole grid of comparisons.

Reporting an uncorrected "p < 0.05" after two dozen tests would be a bug, not a
result.

    python experiments/05_rubric_ablation.py --items 500
"""

from __future__ import annotations

from itertools import combinations

from _common import apply_quick_isolation, banner, base_parser, corpus, done, resolve_judges

from judgeguard.degrade.text import ERROR_DEGRADATIONS, build_variants
from judgeguard.judges.prompts import POINTWISE_CONFIGS
from judgeguard.judges.run import ScoreTask, run_scores
from judgeguard.stats.intervals import bca_ci, paired_bootstrap_diff
from judgeguard.stats.tests import cliffs_delta, holm_bonferroni, mcnemar_test
from judgeguard.store import save
from judgeguard.telemetry import tracking_run


def main() -> None:
    ap = base_parser(__doc__ or "")
    args = apply_quick_isolation(ap.parse_args())
    n_items = 40 if args.quick else args.items
    judges = resolve_judges(args.judges)
    configs = list(POINTWISE_CONFIGS)

    t0 = banner("05  rubric ablation: vague vs rubric vs chain-of-thought")
    items, by_id = corpus(n_items, args.seed)
    variants = build_variants(items, seed=args.seed, degradations=ERROR_DEGRADATIONS)

    correct: dict[tuple[str, str], list[int]] = {}
    tokens: dict[tuple[str, str], int] = {}
    accuracy: dict[str, dict[str, dict]] = {}

    with tracking_run("judgeguard", "05_rubric_ablation", {"items": n_items}) as run:
        for judge in judges:
            accuracy[judge] = {}
            for config in configs:
                ref_j = run_scores(
                    judge,
                    config,
                    [ScoreTask(uid=f"{i.id}:reference", item=i, answer=i.reference) for i in items],
                )
                ref = {j.uid.split(":")[0]: j.score for j in ref_j}
                var_j = run_scores(
                    judge,
                    config,
                    [
                        ScoreTask(
                            uid=v.uid,
                            item=by_id[v.item_id],
                            answer=v.text,
                            degradation=v.degradation,
                            severity=v.severity,
                            len_ratio=v.len_ratio,
                        )
                        for v in variants
                    ],
                )
                flags = []
                for j in var_j:
                    r = ref.get(j.uid.split(":")[0])
                    flags.append(int(r is not None and j.score is not None and j.score < r))
                correct[(judge, config)] = flags
                tokens[(judge, config)] = sum(
                    j.prompt_tokens + j.completion_tokens for j in ref_j + var_j
                )
                est = bca_ci(flags, seed=args.seed)
                accuracy[judge][config] = {
                    **est.as_dict(),
                    "total_tokens": tokens[(judge, config)],
                    "cost_usd": sum(j.cost_usd for j in ref_j + var_j),
                    "mean_latency_ms": sum(j.latency_ms for j in var_j) / max(len(var_j), 1),
                }
                run.log_metrics({f"{judge}_{config}_accuracy": est.value})
            row = "  ".join(f"{c}={100 * accuracy[judge][c]['value']:.1f}%" for c in configs)
            print(f"    {judge:<14} {row}")

        comparisons = []
        for judge in judges:
            for a, b in combinations(configs, 2):
                ca, cb = correct[(judge, a)], correct[(judge, b)]
                diff = paired_bootstrap_diff(cb, ca, seed=args.seed)  # b minus a
                mc = mcnemar_test(cb, ca)
                extra_tokens = tokens[(judge, b)] - tokens[(judge, a)]
                comparisons.append(
                    {
                        "judge": judge,
                        "baseline": a,
                        "candidate": b,
                        "delta_accuracy": diff.as_dict(),
                        "mcnemar": mc.as_dict(),
                        "cliffs_delta": cliffs_delta(cb, ca),
                        "extra_tokens": extra_tokens,
                        "accuracy_points_per_1k_extra_tokens": (
                            100 * diff.diff / (extra_tokens / 1000)
                            if extra_tokens
                            else float("inf")
                        ),
                    }
                )
        holm = holm_bonferroni([c["mcnemar"]["p_value"] for c in comparisons])
        for c, adj, rej in zip(comparisons, holm["adjusted"], holm["reject"], strict=True):
            c["p_adjusted_holm"] = adj
            c["significant_after_correction"] = bool(rej)

    payload = {
        "experiment": "05_rubric_ablation",
        "question": "Does adding rubric structure or chain-of-thought improve discrimination, "
        "and is the improvement worth the tokens?",
        "judges": judges,
        "configs": configs,
        "n_items": len(items),
        "accuracy": accuracy,
        "comparisons": comparisons,
        "multiplicity": {k: v for k, v in holm.items() if k != "adjusted"},
    }
    sig = sum(1 for c in comparisons if c["significant_after_correction"])
    print(f"    {sig}/{len(comparisons)} pairwise config differences survive Holm-Bonferroni")
    done(t0, save("05_rubric_ablation", payload))


if __name__ == "__main__":
    main()
