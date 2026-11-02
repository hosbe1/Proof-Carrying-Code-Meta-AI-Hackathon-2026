# Run notes: eval-glimmer-sft2

SFT 2 evaluation of Glimmer on the Vericoding test split (issue #57),
like-for-like with the SFT 1 and base Glimmer runs.

- Command (full reproduction, env included):
  `PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1,glimmer-sft2=/adapters/sft2"
  PCC_GLIMMER_QUANTIZATION=fp8` for the deployment, then
  `PCC_GLIMMER_REASONING_STRENGTH=low python -m pcc.evaluation vericoding
  --split test --endpoint glimmer --model glimmer-sft2 --attempts 4
  --max-cost-usd 1 --concurrency 32 --out runs/eval-glimmer-sft2`.
  The harness does not store the adapter path, the serving mapping, the
  reasoning strength or the concurrency, so this file does.
- Model and adapter: `glimmer-sft2`, served from `pcc-glimmer:/adapters/sft2`
  (training row `sft2-modal` in `docs/results/training.md`).
- Sampling: reasoning strength `low`, concurrency 32, 4 attempts, 4096
  output tokens per attempt; same as `...-eval-glimmer-sft1` and
  `...-baseline-glimmer-base`.
- Result: 255 of 569 eligible tasks verified (pass@k 0.448, 95% CI
  0.408-0.489). 2 of 571 tasks skipped (unbuildable prompts DA0152,
  DA0818). Token cost $0.00; GPU time is billed on the deployment.
- Comparison on the same 569 tasks: base Glimmer 0.076, Spark 0.097,
  SFT 1 0.278, SFT 2 0.448.
- Wall time: 2026-11-02 14:55:28 to 15:28:44 UTC (33 min).
- Published from `runs/eval-glimmer-sft2` with
  `python -m pcc.evaluation.report publish runs/eval-glimmer-sft2`.
  `meta.json`, `summary.json` and `summary.md` are the verbatim publish
  output.
- Spend: about $3 of Modal H200 time for the 33 min of sampling (Modal
  credits; not a billing total). The deployment scales to zero after 15
  idle minutes.
