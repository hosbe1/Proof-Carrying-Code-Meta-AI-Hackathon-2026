# Run notes: proof-only-glimmer-sft2

Proof-only evaluation of SFT 2 Glimmer on the step-5 evaluation set (issue
#85): the same 33 proof-only tasks as `...-proof-only-spark`, built from
Spark's verified Vericoding test-split answers.

- Tasks: `proof-only-test/tasks.jsonl` (33 lines, tasks-file SHA-256
  `4f7f368e...`, full value in `meta.json` as `split_sha256`); build details
  and skips are recorded in
  `docs/results/2026-11-02-93a247f-proof-only-spark/setup.md`.
- Command (run by the controller at commit `93a247f`):
  `PCC_GLIMMER_REASONING_STRENGTH=low python -m pcc.evaluation vericoding
  --tasks-file runs/proof-only-test/tasks.jsonl --endpoint glimmer --model glimmer-sft2
  --attempts 4 --max-cost-usd 5 --out runs/proof-only-glimmer-sft2`.
  The deployment served with `PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1,
  glimmer-sft2=/adapters/sft2"` at FP8 precision; this run used the `glimmer-sft2`
  adapter from `pcc-glimmer:/adapters/sft2`.
- Model and endpoint: `glimmer-sft2`, endpoint `glimmer`, standard mode.
- Sampling: 4 attempts per task, reasoning strength `low`.
- Result: 21 of 33 tasks verified (pass@k 0.636, 95% CI 0.466-0.778; pass@1 0.394).
  Complete for its task set; no cost cap stop (`stopped_at_cost_cap: false`,
  `stopped_early: false`, status `done`).
- Wall time: started 2026-11-02T15:49:58Z, 7685.0 s of sampling across the run.
  The harness-recorded `started_utc` is identical for all three proof-only
  runs and was kept verbatim, not staggered. `total_seconds` in `meta.json`
  is the sum of per-task `seconds` from `results.jsonl`, not elapsed wall
  time, so the shared start second does not contradict the per-run totals;
  launch order and overlap are not recorded in this branch.
- Published from `runs/proof-only-glimmer-sft2` with
  `python -m pcc.evaluation.report publish runs/proof-only-glimmer-sft2`.
  `summary.json` and `summary.md` are the publish output unchanged; `meta.json`
  is `run.json` with exactly two fields redacted to portable values (`split`
  to `proof-only-test/tasks.jsonl`, `dataset` to `vericoding/proof-only-test`)
  so no machine path enters the repository. Every other field, including
  `split_sha256`, metrics, `started_utc`, `stopped_early` and status, is kept
  verbatim. The original absolute tasks-file path is not retained in this
  branch (controller-side); the tasks-file SHA-256 in `meta.json` pins the
  exact bytes.
- Spend: token cost $0.00 (the Glimmer endpoint bills $0 per token); about
  1 USD of Modal GPU credits for the two Glimmer proof-only runs combined
  (not a billing total). The deployment scales to zero after 15 idle
  minutes; the controller confirms the stop in the issue.
- Paired comparison over the same 33 task IDs (command in
  `...-proof-only-spark/setup.md`): of the 28 tasks base Glimmer solves, SFT 2
  also solves 19 and misses 9 (`...DD0148`, `...DD0736`, `...DS0004`,
  `...DV0032`, `...DV0061`, `...DV0062`, `...DV0068`, `...DV0079`,
  `...DV0108`); it newly solves 2 (`...DD0070`, `...DS0052`), for a net of
  21. SFT 2 is below base Glimmer here, likely because SFT 2 was trained only
  on the full-task format.
