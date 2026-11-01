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
- History mapping: the `docs/results/history.md` format (date, commit, model,
  dataset, n, rate, cost) has no run-name column, so the two rows for this
  commit differ only in rate. This folder (`...-baseline-glimmer-base`) is the
  `low` run at 0.076; `...-baseline-glimmer-base-medium` is the medium repeat
  at 0.069.

## Deployment and spend

- Served by the shared app `pcc-glimmer`
  (`ap-TIrDtVN5Vc8Qd0A4zekEqJ`), used by both `glimmer` and `glimmer-sft`.
  Kept up across this baseline and the queued SFT 1 sampling (#171), then
  stopped after the final active evaluation; this ticket alone cannot show a
  `modal app stop` because stopping earlier would break #171.
- At 2026-11-01 00:11 CET the app had no request for 15 minutes and scaled to
  zero.
- Spend: about $23 of Modal GPU credit for this issue (about $13 on the lost
  runs: 404, 408 crash, high-strength run; about $10 on the low and medium
  runs), over the 3 H100-hour (~$12) cap. The overrun is accepted per the
  issue comment; together with #57 the runs stay well under the shared $60
  evaluation budget. About $2 of the deployment cost belongs to the #57 eval.
