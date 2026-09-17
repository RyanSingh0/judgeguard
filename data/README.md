# Numeric source-support pilot

`pilot.jsonl` contains 100 evaluation pairs from 28 SQuAD v2 development articles.
Each has a human-annotated numeric reference and a programmatically altered answer.
The candidate is absent from the passage; reference offsets are checked. These
checks do not substitute for independent human review. No training uses this file.

Source: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), Rajpurkar, Jia, Liang
and collaborators, with passages from Wikipedia. Dataset and this adaptation:
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This differs from the repository's MIT code license. Per-record metadata describes
the numeric replacement and original article title. Source and dataset checksums,
seed, selection size and attribution are in `pilot.manifest.json`.

Rebuild from the official SQuAD `dev-v2.0.json` download:

```powershell
uv run python scripts/prepare_pilot.py --source path/to/dev-v2.0.json --pairs 100
```

This is deliberately a narrow pilot. Positive cases are short numeric answers,
negative cases are constructed substitutions. It does not measure real-world error
prevalence, long-form factuality, retrieval quality, or agent task completion.
Do not use the test split to select prompts, thresholds, or train a student.
