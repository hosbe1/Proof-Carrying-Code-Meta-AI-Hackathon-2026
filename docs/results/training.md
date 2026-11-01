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

## SFT runs

| run | date (UTC) | commit | base model | job IDs | config | training lines | input SHA-256 | spend (USD) | adapter location |
|---|---|---|---|---|---|---|---|---|---|
| sft1 | 2026-11-01 | 9af9b2759d52647843d6c7632d5b78d5aa5dedc3 | accounts/fireworks/models/muse-glimmer-30b | session ts-274df4913ef44e39a9127857726365ff, run run-4b87798e63b84123ab56a5630637db13, checkpoint sft1-final-6b3d4ef4 | LoRA r16, 2 epochs, batch 8, 76 steps, Adam lr 1e-4 (beta1 0.9, beta2 0.95, eps 1e-8, wd 0), cross_entropy, seed 0 | 301 | c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1 | unknown, billing check pending (expected 0 while LoRA SFT is free until Nov 30; not a billing total) | accounts/mick/models/pcc-glimmer-sft1 (Fireworks; on record but not used — SFT 1 is served from the Modal adapter below) |
| sft1-modal | 2026-11-01 | 9af9b2759d52647843d6c7632d5b78d5aa5dedc3 | accounts/fireworks/models/muse-glimmer-30b | none (Modal run: 34 s weight load, 394 s training) | LoRA r16 alpha 32, targets q/k/v/o + gate/up/down of the text tower, 2 epochs, batch 8, 76 steps, AdamW lr 1e-4 (betas 0.9/0.95, wd 0), bf16, gradient checkpointing, loss on the assistant turn only, rows above 16384 dropped (none); 95.8M trainable params on 1 x H200 | 301 | c3d508bfbff7f1726b838e32e2753caa0279b3873a6688c0ec47a71f81dea9f1 | about 1 (Modal H200 credits, outside the $0 cap; not a billing total) | pcc-glimmer:adapters/sft1 (`adapter_config.json`, `adapter_model.safetensors`); served as `glimmer-sft` |

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
- Spend: unknown, billing check pending. The plan records LoRA SFT on this
  model as free until Nov 30 through the UI or the Serverless Training API,
  so the expected total is 0; this is not a billing total until confirmed
  on the billing page. Follow-up: confirm on the billing page and update
  this row (see issue #78); the repository alone cannot prove spend.
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
  16,384 dropped (none); seed 0 (script pin for deterministic reruns).
  Training script: `deploy/modal_sft.py`.
- Hardware: 1 x H200, bf16, gradient checkpointing; 95.8M trainable
  params.
- Tokens: 637,208 (identical to the Fireworks run).
- Loss: 0.190 (step 1) to 0.020 (step 76).
- Time: 34 s weight load, 394 s training.
- Adapter: `pcc-glimmer:/adapters/sft1` (`adapter_config.json`,
  `adapter_model.safetensors`).
- Cost: Modal credits, about $1 of H200 time; not a billing total. This is
  outside the ticket's original $0 Fireworks-free cap and is recorded here
  as controller-reported for sign-off; confirming the billing total remains
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
  local training log is committed.
- Downstream: supersedes the on-demand fallback above. Round 1 (#79) and
  the SFT 1 evaluation use `glimmer-sft` on Modal.
