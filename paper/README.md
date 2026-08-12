# Workshop paper draft

`main.tex` + `refs.bib`. Four pages plus references, targeting a workshop track on LLM
evaluation.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

or, with `latexmk`:

```bash
latexmk -pdf main.tex
```

No local LaTeX? Upload both files to Overleaf.

## Before you submit — three hard prerequisites

1. **Re-run the battery in live mode.** Every number in the draft currently comes from the
   deterministic simulator and is marked in red via `\simnote`. Reporting simulated numbers as
   measurements would be misconduct. One environment variable fixes it:
   `JUDGEGUARD_PROVIDER_MODE=live`, then `make all`, then update the two tables and the inline
   figures.
2. **Run the human-validation study** described in §5 — ~200 gold pairs, 3 annotators,
   report inter-annotator agreement and the correlation between human preference and the
   degradation label. Reviewers will ask whether the construct is valid, and they will be right
   to. This is the highest-value 200 dollars in the project.
3. **Swap in the venue style file** and check the page limit. The draft uses a plain
   `article` preamble as a placeholder.

## Where the novelty is

Not the bias taxonomy — position, verbosity and self-preference bias are all documented. The
contribution is:

- **fact-level auditable perturbation** — the gold label can be defended rather than asserted,
  because the reference is composed from an explicit fact list and every edit is enumerable;
- **the transfer to agent trajectories** — degrading the *process* rather than the output, with a
  full statistical treatment, exposing failure classes output-only evaluation cannot reach;
- **the calibration corollary** — distillation inherits the teacher's calibration error, and
  because manufactured gold labels are free, repairing it is free too.

A rigorously measured negative result is a real contribution. Argument blindness is one.

## Suggested venues

| venue | track | why |
|---|---|---|
| NeurIPS workshops | evaluation / statistical foundations of LLM eval | the interval-everywhere treatment fits the room |
| ICLR workshops | building trust in LLMs | guardrail + calibration angle |
| ACL Eval4NLP | evaluation & comparison of NLP systems | the most direct topical match |
| Agent-focused workshops | agent evaluation | §3 trajectory results are the differentiator |
