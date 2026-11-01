# Run notes: baseline-glimmer-base-medium

Medium-strength repeat of the base Glimmer baseline on the Vericoding test
split (extra run next to the ticketed baseline in issue #77; not required by
the ticket).

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
- History mapping: `docs/results/history.md` carries one row per published run
  with columns (date, commit, model, dataset, n, rate, cost) and no run-name
  column, so the two rows added here share every field except rate. The second
  of the two rows (rate 0.069) is this extra medium repeat
  (`...-baseline-glimmer-base-medium`, strength `medium`); the first row
  (rate 0.076) is the ticketed `low` baseline in
  `...-baseline-glimmer-base`. Use this mapping, not `history.md` alone, to
  attribute a row to a reasoning strength. This run is supplementary context
  for the reasoning-strength choice; the baseline to compare against is the
  0.076 row.
- Deployment and spend: see the baseline notes file. The deployment was
  shared with the ticketed baseline and is left up for the queued SFT 1
  sampling (#171); the final stop belongs to the final active evaluation,
  not to this ticket.
