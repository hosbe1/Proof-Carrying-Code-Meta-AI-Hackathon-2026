# Run notes: eval-glimmer-sft1

SFT 1 evaluation of Glimmer on the Vericoding test split (first half of the
comparison in issue #57).

- Command (full reproduction, env included):
  `PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1" PCC_GLIMMER_REASONING_STRENGTH=low
  python -m pcc.evaluation vericoding --split test --endpoint glimmer
  --model glimmer-sft --attempts 4 --max-cost-usd 1 --concurrency 32
  --out runs/eval-glimmer-sft1`.
  `meta.json` records the same attempts (4), split hash `014017d4...`,
  commit `5f5c0f4`, endpoint `glimmer` and model `glimmer-sft`; the harness
  does not store the adapter path, the serving mapping, the reasoning
  strength or the concurrency, so this file does.
- Model and adapter: `glimmer-sft` served from the adapter at
  `pcc-glimmer:/adapters/sft1` via
  `PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1"`. The ticket text
  names the mapping `glimmer-sft1=adapters/sft1`; the served name used by
  the finished controller run reused here is `glimmer-sft`, so this folder
  keeps `glimmer-sft` for consistency with `meta.json` and the baseline
  naming (`glimmer`).
- Sampling: reasoning strength `low`, concurrency 32, like-for-like with
  the base Glimmer baseline (`...-baseline-glimmer-base`, strength `low`).
- Result: 158 of 569 eligible tasks verified (pass@k 0.278, 95% CI
  0.242-0.316; pass@1 0.202). 2 of 571 tasks skipped (unbuildable prompts
  DA0152, DA0818). Token cost $0.00; GPU time is billed on the deployment
  (see below).
- Published from `runs/eval-glimmer-sft1` with
  `python -m pcc.evaluation.report publish runs/eval-glimmer-sft1`.
  `meta.json`, `summary.json` and `summary.md` are the verbatim publish
  output; this file only records setup the harness does not store
  (adapter path, serving mapping, `reasoning_strength`, concurrency).
- History mapping: the row `| 2026-11-01 | 5f5c0f4 | glimmer-sft |
  vericoding/test | 569 | 0.278 | 0.00 |` in `docs/results/history.md`
  is this run. The cost column is the token cost ($0.00 for the Glimmer
  endpoint); it does not include GPU time.
- Source: the finished controller run, copied file by file into
  `runs/eval-glimmer-sft1`; no model was run and no deployment was touched
  in this pass.

## Deployment and spend

- Done criterion "deployment is stopped": NOT met by this pull request. No
  deployment was started, changed or stopped here; the final stop and its
  receipt belong to the final active evaluation (SFT 2 / GRPO), not to
  this SFT 1 publish.
- Spend for this run: the controller run sampled for about 28 minutes on
  the shared deployment (about $2 of GPU time, noted in the issue). The
  $0.00 in `history.md` and `summary.json` is token cost only. Spend up to
  the final stop is recorded where the stop is recorded; together with the
  baseline the runs stay within the shared evaluation budget.
