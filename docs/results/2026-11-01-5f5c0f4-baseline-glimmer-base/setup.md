# Run notes: baseline-glimmer-base

Ticketed baseline of base Glimmer on the Vericoding test split (issue #77).

- Command: `python -m pcc.evaluation vericoding --split test --endpoint glimmer
  --model glimmer --attempts 4 --max-cost-usd 1 --out runs/baseline-glimmer-base`
- Sampling: concurrency 32, `PCC_GLIMMER_REASONING_STRENGTH=low`.
  Low was chosen because the template default (`high`) returns reasoning only
  and is cut at the 4096-token limit (0 of 569 verified, not published).
- Result: 43 of 569 eligible tasks verified (pass@k 0.076, 95% CI 0.057-0.100).
  2 of 571 tasks skipped (unbuildable prompts DA0152, DA0818).
  Token cost $0.00; GPU time is billed on the deployment (see below).
- Published from `runs/baseline-glimmer-base` with
  `python -m pcc.evaluation.report publish`. `meta.json`, `summary.json` and
  `summary.md` are the verbatim publish output; this file only records setup
  the harness does not store (`reasoning_strength`, concurrency).
- History mapping: `docs/results/history.md` carries one row per published run
  with columns (date, commit, model, dataset, n, rate, cost) and no run-name
  column, so the two rows added here share every field except rate. The first
  of the two rows (rate 0.076) is the ticketed baseline in this folder
  (`...-baseline-glimmer-base`, strength `low`); the second row (rate 0.069)
  is the extra medium-strength repeat in
  `...-baseline-glimmer-base-medium`. Use this mapping, not `history.md`
  alone, to attribute a row to a reasoning strength.

## Deployment and spend

- Done criterion "deployment is stopped": NOT met by this pull request. The
  run was served by the shared app `pcc-glimmer`
  (`ap-TIrDtVN5Vc8Qd0A4zekEqJ`), used by both `glimmer` and `glimmer-sft`.
  The app is intentionally left up for the queued SFT 1 sampling (#171); the
  final stop and its receipt belong to the final active evaluation, not to
  this ticket, because stopping here would break #171.
- Cost mitigation observed (not a stop): at 2026-11-01 00:11 CET the app had
  no request for 15 minutes and scaled to zero. GPU cost accrues while
  containers run; any spend up to the final stop is recorded where the stop
  is recorded.
- Spend for this issue: about $23 of Modal GPU credit, over the 3 H100-hour
  (~$12) cap from the "evaluation and demo serving" bucket. About $13 went to
  the lost runs (base-URL 404 run, 408-crash run that saved nothing,
  high-strength run at 0 of 569) and about $10 to the low and medium runs;
  about $2 of the deployment cost belongs to the #57 eval. Token cost is
  $0.00 (the Glimmer endpoint bills $0 per token). The overrun and its cause
  (lost runs that found two bugs every later eval needs) are accepted per the
  issue #77 comment of 2026-11-01; together with #57 the runs stay well under
  the shared $60 evaluation budget.
