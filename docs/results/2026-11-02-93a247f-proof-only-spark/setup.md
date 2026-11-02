# Run notes: proof-only-spark

Proof-only evaluation of Spark on the step-5 evaluation set (issue #85):
33 proof-only tasks built from Spark's own verified Vericoding test-split
answers.

- Tasks: `proof-only-test/tasks.jsonl` (33 lines), built with
  `python -m pcc.evaluation.proof_only make --from <results.jsonl of the Spark
  Vericoding test run of #58> --format tasks --out runs/proof-only-test/tasks.jsonl`.
  The source run is the Spark Vericoding test run published as
  `docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low` (commit
  `ea0891a`, 55 verified answers on `vericoding/test`); all 33 `source_task`
  IDs are in its `solved_task_ids` list, and 55 minus the 20 answers without
  ghost code and 2 already-verifying programs gives the 33 written tasks.
  The build skipped 20 answers without ghost code and 2 already-verifying
  programs. The tasks-file SHA-256 (`4f7f368e...`, full value in `meta.json`
  as `split_sha256`) pins the exact bytes; all 33 task IDs are listed in
  `summary.json`. The tasks file itself is not committed in this branch
  (controller-side); what can be checked here is that the 33 task IDs are
  identical in all three published `summary.json` files and that each
  `proof-only/...` suffix matches a `vericoding/...` entry of the source
  run's `solved_task_ids` list.
- Command (run by the controller at commit `93a247f`):
  `python -m pcc.evaluation vericoding --tasks-file runs/proof-only-test/tasks.jsonl
  --model muse-spark-1.3-contributor --attempts 4 --max-cost-usd 5 --out runs/proof-only-spark`.
- Model and endpoint: `muse-spark-1.3-contributor`, endpoint `meta`, contributor mode on.
- Sampling: 4 attempts per task.
- Result: 30 of 33 tasks verified (pass@k 0.909, 95% CI 0.764-0.969; pass@1 0.803).
  Complete for its task set; no cost cap stop (`stopped_at_cost_cap: false`,
  `stopped_early: false`, status `done`).
- Wall time: started 2026-11-02T15:49:58Z, 4847.8 s of sampling across the run.
  The harness-recorded `started_utc` is identical for all three proof-only
  runs and was kept verbatim, not staggered. `total_seconds` in `meta.json`
  is the sum of per-task `seconds` from `results.jsonl`, not elapsed wall
  time, so the shared start second does not contradict the per-run totals;
  launch order and overlap are not recorded in this branch.
- Published from `runs/proof-only-spark` with
  `python -m pcc.evaluation.report publish runs/proof-only-spark`.
  `summary.json` and `summary.md` are the publish output unchanged; `meta.json`
  is `run.json` with exactly two fields redacted to portable values (`split`
  to `proof-only-test/tasks.jsonl`, `dataset` to `vericoding/proof-only-test`)
  so no machine path enters the repository. Every other field, including
  `split_sha256`, metrics, `started_utc`, `stopped_early` and status, is kept
  verbatim. The original absolute tasks-file path is not retained in this
  branch (controller-side); the tasks-file SHA-256 in `meta.json` pins the
  exact bytes.
- Spend: about 0.05 USD of Meta credits (not a billing total).
- Paired comparison with the base Glimmer and SFT 2 runs over the same 33
  task IDs:
  `python -m pcc.evaluation.report compare
  docs/results/2026-11-02-93a247f-proof-only-spark
  docs/results/2026-11-02-93a247f-proof-only-glimmer-base
  docs/results/2026-11-02-93a247f-proof-only-glimmer-sft2`.
  Spark alone misses `proof-only/vericoding/DD0083`,
  `proof-only/vericoding/DD0271` and `proof-only/vericoding/DD0648`;
  `DD0648` is solved by both Glimmer runs.
