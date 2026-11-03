# Training runs

One row per rejection-sampling round, plus one row for the HumanEval
collection run; SFT runs are recorded in the SFT runs table below.
Raw sampling output (`runs/collection/*`, `runs/round0/*`)
stays local and is not committed; only the aggregate rows below are recorded.
Spend figures are honest: an `estimate` label means an upper bound derived
from saved artifacts, not an account billing total.

| round | date (UTC) | commit | model | contributor | dataset | eligible | tasks run | tasks solved | attempts cached | SFT rows | input tokens | output tokens | wall time | spend (USD) | stop_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-11-01 | e473c81fa7f85058f4fc48c7498febb4e4a912a0 | muse-spark-1.3-contributor | yes | vericoding/train (2247 rows, 2237 eligible) | 2237 | 323 | 170 | 2589 | 301 | not persisted (prompt bytes bounded by 4050372 UTF-8 bytes) | not persisted (4096 per attempt cap) | about 8 h sampling | 2.53 estimate, not a billing total | timeout (8 h) |
| 1 | 2026-11-02 | 404d01f6f4399cafd61ba082138c1162009818be | glimmer-sft (SFT 1 adapter on Modal, FP8, reasoning low) | no | vericoding/train (2247 rows, 2237 eligible) | 2237 | 2067 (170 solved in round 0 skipped) | 836 | 16536 | 1450 | 6955812 (last try) | 7225089 (last try) | 4.3 h last try (sampling 2.2 h, Dafny check 2.1 h) | about 16 (Modal credits, 08:00-14:00 UTC window, includes about 30 min of the #84 Glimmer run; not a billing total) | done |
| 2 | 2026-11-02 | 93a247fb73ff265d9b14793591ee83055d46a5df | glimmer-sft2 (SFT 2 adapter on Modal, FP8, reasoning low) | no | vericoding/train (2247 rows, 2237 eligible) | 2237 | 1231 (1006 solved in rounds 0 and 1 skipped) | 143 | 9848 | 5853992 (summary) | 6299896 (summary) | about 2.0 h sampling (7326 s) | unknown, Modal credits within the ticket's $30 cap; no billing total recorded locally (not a billing total) | done |
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

## Round 1 notes (SFT 1 sampling, 2026-11-02)

- Coverage: all 2067 eligible Vericoding train tasks not solved in round 0
  ran with 8 attempts each (16536 cached responses); the 170 tasks solved in
  round 0 were skipped with `--skip-solved runs/round0/rescored.jsonl`.
  Result: 836 of 2067 tasks solved, 1450 verified SFT rows.
- Sampling: `python -m training.rejection --endpoint glimmer --model
  glimmer-sft --tasks train --attempts 8 --concurrency 16 --workers 12
  --resume`, `PCC_GLIMMER_REASONING_STRENGTH=low` (the setting of the SFT 1
  test eval). Sampling and Dafny checks run in two phases; the run resumed
  once from the attempt cache after a restart (3436 attempts were cached
  then), so the token counts in the table cover the last try only.
- Excluded: the same 10 rows without `vc-spec` as round 0.
- Test split: never sampled; the run read `--tasks train` only.
- Spend: the deployment bills GPU time, not tokens (`cost_usd` 0 in the
  summary). Modal billing reports $15.97 for the `pcc-glimmer` app between
  08:00 and 14:00 UTC, which also covers the first half hour of the #84
  base Glimmer spec run. Paid from Modal credits; the month's billed cost
  is $0.00. The deployment scales to zero when idle.
- Local artifacts (not committed): `runs/round1/round1.jsonl` (SHA-256
  `bc3f3e6e4439cfbf669397517b7914845c193df09b6c73305d782c6e746ef6d2`),
  `runs/round1/round1.jsonl.attempts.jsonl` (SHA-256
  `9825ce87485f4f044c211a7ccb4e87e52d5f7dc3255a1739d42db273f3a14ac3`),
  `round1.jsonl.summary.json`, `round1.jsonl.tasks.jsonl`,
  `round1.jsonl.skipped.json`.
- Downstream: SFT 2 (#80) trains on round 0 (301 rows) plus round 1 (1450
  rows).

## Round 2 notes (SFT 2 sampling, 2026-11-02)

- Coverage: 1231 of 2237 eligible Vericoding train tasks ran with 8 attempts
  each (9848 cached responses in `round2.jsonl.attempts.jsonl`, exactly 8 per
  task); the 1006 tasks solved in rounds 0 (170) and 1 (836) were skipped with
  `--skip-solved` on the round 0 and round 1 rows. Result: 143 of 1231 tasks
  solved, 239 verified SFT rows (`round2.jsonl`).
- Sampling: `python -m training.rejection --endpoint glimmer --model
  glimmer-sft2 --tasks train --attempts 8 --concurrency 16 --workers 12
  --skip-solved <round 0 + round 1 rows>` (per #179),
  `PCC_GLIMMER_REASONING_STRENGTH=low` (the setting of the SFT 3 test eval).
  Sampling and Dafny checks run in two phases like round 1.
- Excluded: the same 10 rows without `vc-spec` as rounds 0 and 1.
- Test split: never sampled; the run read `--tasks train` only. Leakage check
  recorded: the 1231 sampled task IDs in `round2.jsonl.attempts.jsonl` have
  zero intersection with the 569 Vericoding test-split IDs (checked locally
  when publishing #179).
- Tokens and wall time: `round2.jsonl.summary.json` reports 5853992 input and
  6299896 output tokens, 7326 s wall time (about 2.0 h sampling),
  `stop_reason` done and `cost_usd` 0 (GPU time bills separately). The summary
  reports 9264 attempts while the attempts file holds 9848 lines (1231 x 8);
  the difference is not reconciled locally and both figures are recorded here.
- Spend: sampling ran on the shared `pcc-glimmer` Modal deployment, paid from
  Modal credits within the ticket's $30 cap. No billing total is recorded
  locally; confirming it remains open and this is not a billing total.
- Local artifacts (not committed): `runs/round2/round2.jsonl` (SHA-256
  `24e0ab7fe120c6bb5882888e5b890ee12b54e2e2231284c52ec96df66e8b05f5`
  239 verified rows), `round2.jsonl.attempts.jsonl` (9848 lines),
  `round2.jsonl.summary.json`, `round2.jsonl.tasks.jsonl` (1231 lines),
  `round2.jsonl.skipped.json`, `solved-r0r1.jsonl` (1751 lines: the 301
  round-0 plus 1450 round-1 SFT rows skipped),
  `full-r012.jsonl` (SHA-256
  `6ae2938e1201156c6635b5d244302143eeb713ec6030ccd4dd269f58eda73cb3`,
  1990 lines: the SFT 3 full-task input),
  `proof-only-r0r1.jsonl` (848 lines) and `proof-only-r2.jsonl` (181 lines),
  `proof-only-train-tasks.jsonl` (SHA-256
  `9842e8b9ea823bbfb50d67233c6770f1a6ab6f8366a8b521cc565b12d7965a02`,
  1029 train proof-only tasks built from the round 0-2 answers),
  `sft4-proof-only.jsonl` (SHA-256
  `c4534f136303d9d8ab7b75479a167120314cf0b61b03ddb99054b527365b32f5`,
  922 SFT 3-verified rows on those tasks).
- Downstream: SFT 3 trains on `full-r012.jsonl` plus the 1029 proof-only
  lines; SFT 4 adds `sft4-proof-only.jsonl`.

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
- Downstream: SFT 1 trains on round 0 only, SFT 2 on rounds 0 and 1. Type inference for
  untyped functions is follow-up work, tracked separately.

## SFT runs

| run | date (UTC) | commit | base model | job IDs | config | training lines | input SHA-256 | spend (USD) | adapter location |
|---|---|---|---|---|---|---|---|---|---|
| sft1 | 2026-11-01 | 9af9b2759d52647843d6c7632d5b78d5aa5dedc3 | accounts/fireworks/models/muse-glimmer-30b | session ts-274df4913ef44e39a9127857726365ff, run run-4b87798e63b84123ab56a5630637db13, checkpoint sft1-final-6b3d4ef4 | LoRA r16, 2 epochs, batch 8, 76 steps, Adam lr 1e-4 (beta1 0.9, beta2 0.95, eps 1e-8, wd 0), cross_entropy, seed 0 | 301 | c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1 | unknown, billing check pending (expected 0 while LoRA SFT is free until Nov 30; not a billing total) | accounts/mick/models/pcc-glimmer-sft1 (Fireworks; on record but not used — SFT 1 is served from the Modal adapter below) |
| sft1-modal | 2026-11-01 | 9af9b2759d52647843d6c7632d5b78d5aa5dedc3 | accounts/fireworks/models/muse-glimmer-30b | none (Modal run: 34 s weight load, 394 s training) | LoRA r16 alpha 32, targets q/k/v/o + gate/up/down of the text tower, 2 epochs, batch 8, 76 steps, AdamW lr 1e-4 (betas 0.9/0.95, wd 0), bf16, gradient checkpointing, loss on the assistant turn only, rows above 16384 dropped (none); 95.8M trainable params on 1 x H200 | 301 | c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1 | about 1 (Modal H200 credits, outside the $0 cap; not a billing total) | pcc-glimmer:adapters/sft1 (`adapter_config.json`, `adapter_model.safetensors`); served as `glimmer-sft` |
| sft2-modal | 2026-11-02 | ea911e5a6d196719565e22d382dca7794c86d94d | meta-models/Muse-Glimmer-30B (base Glimmer, not SFT 1) | none (Modal app pcc-sft2, run ap-lZBFBgCwb01epRufhUBjJX) | same recipe as sft1-modal: LoRA r16 alpha 32, text-tower q/k/v/o + gate/up/down, 2 epochs, batch 8, 438 steps, AdamW lr 1e-4 (betas 0.9/0.95, wd 0), bf16, loss on the assistant turn only, rows above 16384 dropped, shuffle seed 0; 1 x H200 | 1751 (round 0 rescored 301 + round 1 1450) | 294b72a20f4bf7a0053deaaf64c10f55fcc30c5bf48fed7b119504dbeb23fde5 | about 3 (Modal H200 credits, 35 min training; not a billing total) | pcc-glimmer:adapters/sft2 (`adapter_config.json`, `adapter_model.safetensors`); served as `glimmer-sft2` |
| sft3-modal | 2026-11-02 | 93a247fb73ff265d9b14793591ee83055d46a5df | meta-models/Muse-Glimmer-30B (base Glimmer, not SFT 2) | controller record in #179 (no local training log committed) | same recipe as sft2-modal: LoRA r16 alpha 32, text-tower q/k/v/o + gate/up/down, 2 epochs, batch 8, 756 steps, AdamW lr 1e-4 (betas 0.9/0.95, wd 0), bf16, loss on the assistant turn only, rows above 16384 dropped, shuffle seed 0; 1 x H200 | 3019 (rounds 0 + 1 + 2 verified proofs, 1990, plus 1029 train proof-only lines) | 8b2c5a797358416b8e9f66133aabad671de89558839c9fe61791f07813cc703f | Modal H200 credits within the ticket's $30 cap; billing total pending controller record (not a billing total) | pcc-glimmer:/adapters/sft3 (path by the sft1/sft2 naming convention; served as `glimmer-sft3`, proven by the eval runs below) |
| sft4-modal | 2026-11-03 | 2a2a777c2855b5a5bd0271b9422814766f9b5118 | meta-models/Muse-Glimmer-30B (base Glimmer, not SFT 3) | controller record in #179 (no local training log committed) | same recipe as sft3-modal: LoRA r16 alpha 32, text-tower q/k/v/o + gate/up/down, 2 epochs, batch 8, 986 steps, AdamW lr 1e-4 (betas 0.9/0.95, wd 0), bf16, loss on the assistant turn only, rows above 16384 dropped, shuffle seed 0; 1 x H200 | 3941 (SFT 3 data, 3019, plus 922 proof-only rows self-generated by SFT 3 on 1029 train proof-only tasks, 922/1029 verified at attempts 8 low) | 17fcef71cf92332d (SHA-256 prefix; full hash pending controller record) | Modal H200 credits within the ticket's $30 cap; billing total pending controller record (not a billing total) | pcc-glimmer:/adapters/sft4 (`adapter_model.safetensors`); served as `glimmer-sft4` |


## SFT 2 notes (Modal, 2026-11-02)

- Data: round 0 rescored (`runs/round0/rescored.jsonl`, 301 rows) followed
  by round 1 (`runs/round1/round1.jsonl`, 1450 rows), concatenated:
  1751 rows, SHA-256
  `294b72a20f4bf7a0053deaaf64c10f55fcc30c5bf48fed7b119504dbeb23fde5`.
  The HumanEval collection (#53) produced 0 SFT rows, so it adds nothing.
  Proof-only lines (#83) were not added: SFT 2 keeps the task format of
  SFT 1, so the SFT 1 to SFT 2 gain measures more data, not a new task
  type, and the proof-only evaluation (#85) stays a held-out format.
- Base: base Glimmer, not SFT 1, as the plan asks.
- Platform: Modal instead of Fireworks, for the reason in the SFT 1 retrain
  notes (no adapter download from Fireworks). The run used the controller
  script of the SFT 1 retrain with only the output path changed to
  `adapters/sft2`; `deploy/modal_sft.py` pins the SFT 1 input hash and was
  not used.
- Loss: from about 0.2 at the start to 0.01-0.03 at step 438; 2,780,606
  trained tokens; 2,099 s training on 1 x H200.
- Adapter config: PEFT wrote `target_modules` as the regex used for
  training; it was replaced with the module-name list of SFT 1 so vLLM
  loads it (same fix as SFT 1).
- Serving: `PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1,glimmer-sft2=/adapters/sft2"`,
  FP8; a warm-up request to `glimmer-sft2` returned HTTP 200.
- Result: Vericoding test pass@4 0.448 (SFT 1 0.278, base Glimmer 0.076),
  see `docs/results/2026-11-02-ea911e5-eval-glimmer-sft2/`.

## SFT 4 notes (Modal, 2026-11-03)

- Data: the SFT 3 data (3019 rows) plus `sft4-proof-only.jsonl`: 922
  proof-only rows that SFT 3 itself generated on the 1029 train proof-only
  tasks (`proof-only-train-tasks.jsonl`, built with `proof_only make
  --format tasks` from the round 0-2 train answers). The generating run
  (`runs/sft4-sample`, 2026-11-03, attempts 8, reasoning low) verified
  922/1029 (proof rate 0.896). Total: 3941 rows, SHA-256 prefix
  `17fcef71cf92332d` (full hash pending controller record; the local
  `sft4-proof-only.jsonl` hashes to
  `c4534f136303d9d8ab7b75479a167120314cf0b61b03ddb9a510b30fd6463c530`).
  The HumanEval collection (#53) produced 0 SFT rows, so it adds nothing.
- Base: base Glimmer, not SFT 3, as the plan asks.
- Platform: Modal, same recipe as SFT 3; 986 steps, 6.58M trained tokens,
  4895 s training on 1 x H200 (controller record in #179).
- Adapter config: same LoRA r16 setup as SFT 2/3 (PEFT target-module fix as
  in SFT 1/2 applies where needed; controller record).
- Serving: `PCC_GLIMMER_ADAPTERS` gains `glimmer-sft4=/adapters/sft4`; the
  eval runs below prove serving.
- Gate (decided before training): SFT 4 replaces SFT 3 only if test pass@4 >=
  253/569. Result 273/569 (0.480) vs SFT 3 259/569 (0.455): the gate passed
  and SFT 4 is the main model.
- Test split: never sampled for training. Leakage check recorded: the 1029
  train proof-only tasks all derive from round 0-2 train answers
  (`source_task` prefix `sft/` throughout), and the 33-task and 327-task
  proof-only eval sets are test-derived and used for evaluation only.
  Content check (controller record in #179): the 3941 SFT 4 rows vs the 571
  test-split specs (vc-preamble + vc-spec, whitespace-normalised) show 8 rows
  containing 2 test specs (DD0114, DD0926), both of which also appear inside
  train tasks upstream (e.g. DD0114 inside train DD0116) -- dataset
  near-duplicates, not leakage; effect at most 2/569.
- Provenance: step, token, time, and adapter figures above are as reported by
  the run controller in the issue record of 2026-11-03; no local training log
  is committed. Verification needs volume and console access; the repository
  alone cannot prove adapter presence, loss, token, or time figures.
- Downstream: SFT 4 is the proof stage 1 model for evaluation, demo, and bug
  hunt until a later round replaces it.

## SFT 3 notes (Modal, 2026-11-02)

- Data: `full-r012.jsonl` (1990 verified full-task rows: round 0 rescored 301
  plus round 1 1450 plus round 2 239) followed by 1029 proof-only SFT lines
  built from the round 0-2 train answers (`proof_only make --format sft` on
  the train split): 3019 rows, SHA-256
  `8b2c5a797358416b8e9f66133aabad671de89558839c9fe61791f07813cc703f`
  (combined training-file hash as reported by the controller; the local
  components hash to `6ae2938e...` for `full-r012.jsonl` and `9842e8b9...`
  for `proof-only-train-tasks.jsonl`). The HumanEval collection (#53)
  produced 0 SFT rows, so it adds nothing. Proof-only lines are new in SFT 3:
  SFT 2 kept the full-task format only, and the SFT 1 to SFT 3 gain now mixes
  more data with the new task type.
- Base: base Glimmer, not SFT 2, as the plan asks.
- Platform: Modal, same recipe as SFT 2; 756 steps (derived: ceil(3019/8)
  batches x 2 epochs, same batching as SFT 1/2) on 1 x H200. Trained tokens,
  loss curve, and wall time are not recorded locally (controller record in
  #179).
- Serving: `PCC_GLIMMER_ADAPTERS` gains `glimmer-sft3=/adapters/sft3`; the
  eval runs below prove serving.
- Result: Vericoding test pass@4 0.455 (SFT 2 0.448, SFT 1 0.278), see
  `docs/results/2026-11-02-93a247f-eval-glimmer-sft3/`; proof-only v2 (327
  tasks): 283/327 at reasoning low, 285/327 at reasoning medium. SFT 2 to
  SFT 3 is +4 net on the same 569 test IDs (32 newly solved, 28 missed).
- Test split: never sampled for training (see the round 2 leakage check).
- Provenance: recipe, row, hash, and adapter figures above are as reported by
  the run controller in the issue record of 2026-11-03; no local training log
  is committed. Verification needs volume and console access; the repository
  alone cannot prove adapter presence, loss, token, or time figures.

## SFT 1 notes (2026-11-01)

- Method: dataset upload is disabled for our account (`:upload` and
  `:getUploadEndpoint` return `Method Not Allowed`; external URLs must be
  `gs://`, `s3://` or Azure Blob), so SFT 1 ran through the Fireworks
  serverless Training API with a local loop: the rows are sent with each
  `forward_backward`, no dataset object. The plan records this serverless
  path as part of the free LoRA SFT window until Nov 30.
- Data: `runs/round0/rescored.jsonl` from #70, 301 rows, SHA-256
  `c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1`;
  318,604 tokens per epoch, at most 3,122 tokens per row, loss on the
  assistant turn only. The collection (#53) finished with 0 proofs, so it
  contributes nothing.
- Config: LoRA rank 16, 2 epochs, batch 8, 76 steps, Adam lr 1e-4 (beta1
  0.9, beta2 0.95, eps 1e-8, weight decay 0), loss `cross_entropy`,
  shuffle seed 0. Tokenizer `meta-models/Muse-Glimmer-30B`.
- Loss per response token: 0.19 (step 1) to 0.05-0.08 (epoch 1) to
  0.02-0.035 (epoch 2). Trained tokens: 637,208; wall time 423 s.
- Promoted model: `accounts/mick/models/pcc-glimmer-sft1`
  (kind `HF_PEFT_ADDON`, r=16), state READY.
- Commit: `9af9b2759d52647843d6c7632d5b78d5aa5dedc3` is the `origin/main`
  HEAD at run-record time (HumanEval collection record). The SFT run itself
  started no repository commit; the table records the code base the training
  data came from.
- Provenance: session, run, checkpoint, loss, token, and state figures above
  are as reported by the run controller in the issue record of 2026-11-01.
  Raw sampling output stays local per the header above, so no local training
  log is committed; checking those IDs needs access to the training console.
- Spend (unverified): unknown, billing check pending. The plan records LoRA SFT on this
  model as free until Nov 30 through the UI or the Serverless Training API,
  so the expected total is 0; this is not a billing total until confirmed
  on the billing page. Follow-up: confirm on the billing page and update
  this row (see issue #78); the repository alone cannot prove spend. A billed
  amount would exceed the ticket's $0 cap and must be reported in #78.
- Open: the PEFT download. Fireworks refuses it with an API key ("model
  downloads require signing in with `firectl signin` or the Fireworks
  console"). Once downloaded, the adapter goes to
  `pcc-glimmer:adapters/sft1` and the Modal deployment serves it with
  `PCC_GLIMMER_ADAPTERS`.
- Downstream: per the fallback in this issue, round 1 (#79) and the SFT 1
  evaluation run on a Fireworks on-demand deployment of the promoted
  model until the volume path above exists.

## SFT 1 retrain notes (Modal, 2026-11-01)

- Reason: the Fireworks adapter download stays unavailable without a
  support-granted exception, and sampling the Fireworks checkpoint through
  the training API timed out (2 x 600 s read timeout on a 68-token
  prompt). SFT 1 was therefore retrained on Modal with the same data and
  recipe; this adapter is the one served and evaluated. The Fireworks
  model stays on record but is not used.
- Data: `runs/round0/rescored.jsonl`, 301 rows, SHA-256
  `c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1`
  (same as the Fireworks run). The script refuses any other input.
- Recipe: LoRA r=16, alpha 32, attention q/k/v/o plus MLP gate/up/down
  of the text tower; 2 epochs, batch 8 (76 steps), AdamW lr 1e-4, betas
  0.9/0.95, no weight decay; loss on the assistant turn only; rows above
  16,384 dropped (none); seed 0 (script pin for deterministic reruns only;
  the original retrain log recorded no seed).
  Training script: `deploy/modal_sft.py`, which has not been executed
  end to end in CI (hermetic tests pin its constants only).
- Hardware: 1 x H200, bf16, gradient checkpointing; 95.8M trainable
  params.
- Tokens: 637,208 (identical to the Fireworks run).
- Loss: 0.190 (step 1) to 0.020 (step 76).
- Time: 34 s weight load, 394 s training.
- Adapter: `pcc-glimmer:/adapters/sft1` (`adapter_config.json`,
  `adapter_model.safetensors`).
- Cost (unverified, requires owner sign-off): Modal credits, about $1 of H200
  time; not a billing total. This is outside the ticket's original $0
  Fireworks-free cap and is recorded here as controller-reported for sign-off;
  it must not be treated as within the cap. Confirming the billing total remains
  open.
- Commit: `9af9b2759d52647843d6c7632d5b78d5aa5dedc3` is the `origin/main`
  HEAD at run-record time. The retrain itself started no repository
  commit; the table records the code base the training data came from.
- Serving: the Glimmer deployment now runs FP8 with
  `PCC_GLIMMER_ADAPTERS=glimmer-sft=/adapters/sft1`. A warm-up request to
  model `glimmer-sft` returned HTTP 200 after a 285 s cold start, and the
  base model `glimmer` answered from the warm container.
- Provenance: loss, token, time, and serving figures above are as
  reported by the run controller in the issue record of 2026-11-01; no
  local training log is committed. Verification needs volume and console
  access; the repository alone cannot prove adapter presence, serving,
  loss, token, or time figures.
- Downstream: supersedes the on-demand fallback above. Round 1 (#79) and
  the SFT 1 evaluation use `glimmer-sft` on Modal.
