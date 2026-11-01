# Training runs

One row per rejection-sampling round, plus one row for the HumanEval
collection run. Raw sampling output (`runs/collection/*`, `runs/round0/*`)
stays local and is not committed; only the aggregate rows below are recorded.
Spend figures are honest: an `estimate` label means an upper bound derived
from saved artifacts, not an account billing total.

| round | date (UTC) | commit | model | contributor | dataset | eligible | tasks run | tasks solved | attempts cached | SFT rows | input tokens | output tokens | wall time | spend (USD) | stop_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-11-01 | e473c81fa7f85058f4fc48c7498febb4e4a912a0 | muse-spark-1.3-contributor | yes | vericoding/train (2247 rows, 2237 eligible) | 2237 | 323 | 170 | 2589 | 301 | not persisted (prompt bytes bounded by 4050372 UTF-8 bytes) | not persisted (4096 per attempt cap) | about 8 h sampling | 2.53 estimate, not a billing total | timeout (8 h) |
| collection | 2026-11-01 | fa641442027a85fbe99d8d17452faf4f6d4ac169 | muse-spark-1.3-contributor | yes | humanevalfix/train (100 tasks, fixed+buggy; Python/32 excluded) | 101 tasks (100 evaluated, 1 excluded) | 200 | 0 | 0 (pipeline run, no attempt cache) | 0 | 50952 | 241543 | about 41 min pipeline | 0.0534 | done |

## Round 0 notes (partial checkpoint accepted as the round-0 result)

- Coverage: 323 of 2237 eligible Vericoding train tasks have all 8 attempts
  cached; the next task has 5 cached attempts (323 x 8 + 5 = 2589 cached
  responses). This is partial coverage, not full coverage of 2237 tasks.
- Sampling: rejection sampling with Dafny verification, 8 attempts per task,
  temperature 0.8, at most 4096 output tokens, `--concurrency 2 --workers 2`.
- Excluded: 10 of 2247 train rows lack `vc-spec`, so `build_prompt` rejects
  them and they never entered sampling: DA0060, DA0074, DA0288, DA0363,
  DA0438, DA0454, DA0525, DA0538, DA0695, DA0716.
- Rescore: after the formatting-only frozen-prefix correction, the 323
  complete tasks were rescored offline over cached responses with 0 new
  model requests at $0 cost: 170/323 tasks solved, 301 verified SFT rows.
- Spend: the interrupted sampling process did not persist API usage, so input
  and output token counts and exact spend are unknown. The 2.53 USD figure is
  a conservative token-cost upper bound for the saved successful calls only
  (input bounded by 4050372 prompt bytes at published contributor rates);
  it excludes unrecorded requests and retries and is not a billing total.
  Check the usage dashboard before any further paid sampling.
- Local artifacts (not committed): `runs/round0/round0.jsonl.attempts.jsonl`
  (SHA-256 `2ca0caf950298863e3174a2ec95389c526f12e2443eefc9c5724a331b6fcaf11`),
  `runs/round0/rescored.jsonl` (SHA-256
  `c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1`),
  `runs/round0/rescored-summary.json` (SHA-256
  `652bcd3af15b8630d23541cdfc855dfeff323fb0e546188e312aac09bff88197`).
- Downstream: SFT 1 uses `runs/round0/rescored.jsonl` (301 rows) as training
  data. No further paid sampling is planned from this round; round 1 adds
  data instead. No training job was started from this round.

## Collection notes (HumanEval pipeline collection, 2026-11-01)

- Coverage: 200 function versions evaluated (100 fixed + 100 buggy from 100
  HumanevalFix train tasks) with the full pipeline in contributor mode; 0
  proofs collected, so there is no `proofs.jsonl` and 0 SFT rows. The run
  finished with status `done`, not stopped early, within the $60 cap.
- States: 187 `not_checkable`, 7 `open`, 6 `counterexample`. All counts below
  are counted from the local `runs/collection/scan.jsonl`.
- `not_checkable` reasons: `no types` 140; list-typed 30 (`type list` 16,
  `type List[float]` 6, `type List[str]` 4, `type List[Any]` 2,
  `type List[int]` 2); calls into the project 6; `spec does not kill
  mutants` 5; `spec malformed` 4; `eval/exec` not supported 2.
- `open` (7): `original times out` in all 7 rows.
- `counterexample` (6): 4 on buggy versions (bugs found), 2 on fixed
  versions (false alarms).
- Spend: 174 of 200 rows made no model call (filtered before any model
  step); 50952 input and 241543 output tokens across 26 rows, $0.0534 at
  contributor rates.
- Excluded: `humanevalfix/Python/32` (fixed version fails its own tests), so
  it has no rows in the scan.
- Local artifacts (not committed): `runs/collection/run.json`,
  `runs/collection/scan.jsonl`.
- Downstream: SFT 1 and SFT 2 train on round 0 only. Type inference for
  untyped functions is follow-up work, tracked separately.
