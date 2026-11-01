# Run notes: baseline-glimmer-base-medium

Medium-strength repeat of the base Glimmer baseline on the Vericoding test
split (extra run next to the ticketed baseline in issue #77).

- Command: same harness invocation as the baseline with
  `--out runs/baseline-glimmer-base-medium`, concurrency 32,
  `PCC_GLIMMER_REASONING_STRENGTH=medium`.
- Result: 39 of 569 eligible tasks verified (pass@k 0.069, 95% CI 0.051-0.092).
  2 of 571 tasks skipped (unbuildable prompts DA0152, DA0818).
  Token cost $0.00; GPU time is billed on the deployment (see the baseline
  notes file for spend and deployment status).
- Published from `runs/baseline-glimmer-base-medium` with
  `python -m pcc.evaluation.report publish`. `meta.json`, `summary.json` and
  `summary.md` are the verbatim publish output; this file only records setup
  the harness does not store (`reasoning_strength`, concurrency).
- History mapping: the `docs/results/history.md` format (date, commit, model,
  dataset, n, rate, cost) has no run-name column, so the two rows for this
  commit differ only in rate. This folder
  (`...-baseline-glimmer-base-medium`) is the `medium` run at 0.069;
  `...-baseline-glimmer-base` is the ticketed `low` baseline at 0.076.
