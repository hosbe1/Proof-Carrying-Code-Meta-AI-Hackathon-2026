# Training runs

One row per rejection-sampling round. Raw sampling output (`runs/round0/*`)
stays local and is not committed; only the aggregate row below is recorded.
Spend figures are honest: an `estimate` label means an upper bound derived
from saved artifacts, not an account billing total.

| round | date (UTC) | commit | model | contributor | dataset | eligible | tasks run | tasks solved | attempts cached | SFT rows | input tokens | output tokens | wall time | spend (USD) | stop_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-11-01 | e473c81fa7f85058f4fc48c7498febb4e4a912a0 | muse-spark-1.3-contributor | yes | vericoding/train (2247 rows, 2237 eligible) | 2237 | 323 | 170 | 2589 | 301 | not persisted (prompt bytes bounded by 4050372 UTF-8 bytes) | not persisted (4096 per attempt cap) | about 8 h sampling | 2.53 estimate, not a billing total | timeout (8 h) |

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
