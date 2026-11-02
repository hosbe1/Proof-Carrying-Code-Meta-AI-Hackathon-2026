# Run notes: proof-only-glimmer-base

Proof-only evaluation of base Glimmer on the step-5 evaluation set (issue
#85): the same 33 proof-only tasks as `...-proof-only-spark`, built from
Spark's verified Vericoding test-split answers.

- Tasks: `proof-only-test/tasks.jsonl` (33 lines, tasks-file SHA-256
  `4f7f368e...`, full value in `meta.json` as `split_sha256`); build details
  and skips are recorded in
  `docs/results/2026-11-02-93a247f-proof-only-spark/setup.md`.
- Command (run by the controller at commit `93a247f`):
  `PCC_GLIMMER_REASONING_STRENGTH=low python -m pcc.evaluation vericoding
  --tasks-file runs/proof-only-test/tasks.jsonl --endpoint glimmer --model glimmer
  --attempts 4 --max-cost-usd 5 --out runs/proof-only-glimmer-base`.
  The deployment served with `PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1,
  glimmer-sft2=/adapters/sft2"` at FP8 precision; this run used the base model
  with no adapter.
- Model and endpoint: `glimmer`, endpoint `glimmer`, standard mode.
- Sampling: 4 attempts per task, reasoning strength `low`.
- Result: 28 of 33 tasks verified (pass@k 0.848, 95% CI 0.691-0.933; pass@1 0.614).
  Complete for its task set; no cost cap stop (`stopped_at_cost_cap: false`,
  `stopped_early: false`, status `done`).
- Wall time: started 2026-11-02T15:49:58Z, 7971.0 s of sampling across the run.
- Published from `runs/proof-only-glimmer-base` with
  `python -m pcc.evaluation.report publish runs/proof-only-glimmer-base`.
  `meta.json`, `summary.json` and `summary.md` are the publish output, except
  that the harness-recorded absolute tasks-file path was relabeled to the
  portable `proof-only-test/tasks.jsonl` (`split`) and
  `vericoding/proof-only-test` (`dataset`) before publishing, so no machine
  path enters the repository; the original path and the tasks-file SHA-256
  are recorded here and in `meta.json`.
- Spend: token cost $0.00 (the Glimmer endpoint bills $0 per token); about
  1 USD of Modal GPU credits for the two Glimmer proof-only runs combined
  (not a billing total). The deployment scales to zero after 15 idle
  minutes; the controller confirms the stop in the issue.
- Paired comparison over the same 33 task IDs (command in
  `...-proof-only-spark/setup.md`): base Glimmer alone misses
  `proof-only/vericoding/DD0070`, `...DD0083`, `...DD0271`,
  `...DD0761` and `...DS0052`; it solves `...DD0648`, which Spark misses.
