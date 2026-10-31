# Training

This folder holds the training side of the project (docs/plan.md, Training): the shared
prompt, the graded proof reward, the SFT data export and the GRPO loop. It lives outside
`src/pcc`: it imports `pcc` (the verifier as the reward, the datasets as the train/test
source of truth) but the package never imports it.

| File | Does |
| --- | --- |
| `prompts.py` | `SYSTEM_MESSAGE` and `build_prompt`, shared verbatim by every data line and every sampling tool |
| `reward.py` | The graded proof reward: 1 verified, 0 not verified, −1 cheat or changed spec |
| `rejection.py` | `python -m training.rejection`: sample proofs per task, keep the Dafny-verified ones |
| `export_sft.py` | `python -m training.export_sft`: verified solutions become Fireworks SFT data |
| `grpo.py` | The GRPO loop itself, against a small `Backend` protocol with no SDK and no network, so it runs offline in tests |
| `configs/glimmer_fireworks.toml` | The GRPO run configuration |

## Rejection sampling

```bash
python -m training.rejection --endpoint meta|glimmer --model <name> --tasks train \
  --attempts 8 --out sft.jsonl [--skip-solved old.jsonl] [--max-cost-usd 40] [--resume]
```

`run(tasks, client, *, model, attempts, out, ...)` in `rejection.py` drives one round and
returns a `RoundSummary`. Every attempt is one `complete` call with the shared system
message and `build_prompt(task)` as the user message (`step="proof"`, free text, no
schema). Sampling runs the attempts of a task concurrently with `concurrency` tasks at
once; every raw attempt is appended to `<out>.attempts.jsonl` under a lock. After
sampling, `reward_fn` (default: the graded reward bound to `Config().dafny.resource_limit`)
scores every attempt of the finished tasks in a `ProcessPoolExecutor` with `--workers`
processes. Only a reward of exactly 1.0 counts as verified; 0 and −1 do not.

`<out>` holds one line per verified, deduplicated answer in the SFT chat format from the
section below. Identical answers of one task are written once, at most `--max-per-task`
per task, and `<out>` is rewritten from the attempts file at the end of each run.
`--endpoint meta` routes through `build_client` (round 0: the contributor model, with
`--max-cost-usd` required); `--endpoint glimmer` posts to `PCC_GLIMMER_BASE_URL` with
`PCC_GLIMMER_API_KEY` (round 1: the SFT 1 model on the tasks still unsolved, passed as
`--skip-solved`).

After verification, one line per finished task goes to `<out>.tasks.jsonl` with
`prompt_sha256` (lowercase hex SHA-256 of the exact UTF-8 bytes of `build_prompt(task)`),
`attempts` and `solved`. An existing `<out>` requires `--resume`, which appends to
`<out>.attempts.jsonl` (never truncates it), skips the finished tasks, and tops up tasks
that have attempts but no finished-task record. `--skip-solved` skips the tasks whose
exact user prompt occurs in that file; the union of both skips runs nothing twice.
With `--max-cost-usd` the client is wrapped so only the dollar cap constrains the run:
the call that crosses the cap completes, the next call stops sampling with
`stop_reason` `"cost cap"` (exit 0, resumable); an endpoint failure stops with
`"endpoint: <message>"` (exit 2). The summary is printed and written to
`<out>.summary.json` with `tasks_solved`, `tasks_run`, `attempts`, `input_tokens`,
`output_tokens`, `cost_usd`, `wall_seconds` and `stop_reason`; it describes the current
invocation while the output files keep the resumed cumulative data.

## SFT data export

```bash
python -m training.export_sft runs/<name>... --out sft.jsonl [--include <jsonl>...]
```

Each sampling run directory holds `samples.jsonl`: one JSON object per line with the keys
`task_id` (for example `vericoding/DA0002`), `answer` (the complete Dafny program) and
`verified` (whether Dafny verified it with a clean anti-cheat scan). A run directory may
additionally hold `proofs.jsonl` with proof-only lines: one JSON object per line with the
keys `task_id`, `messages` (the list of `{role, content}` prompt turns of the last `proof`
call, starting with the shared system message) and `answer`, which becomes the assistant
turn. The export writes one line per verified, deduplicated solution in the Fireworks chat
format:

```json
{"messages": [{"role": "system", ...}, {"role": "user", "<prompt>"}, {"role": "assistant", "<answer>"}]}
```

For `samples.jsonl` lines the user turn is rebuilt with `build_prompt` from
`training/prompts.py`: for a Vericoding row it is the Dafny instructions plus the task
description, preamble, helpers, spec and postamble (never `vc-code`, which is the
reference answer); for a HumanEvalFix row it is the Python fix instructions plus the
`prompt` stub. The sampling harness and the round 0 tool must import that same
`build_prompt` instead of copying it, so their user messages compare byte-identical with
the test-split prompt guard. For `proofs.jsonl` lines it
is the stored prompt, with `answer` appended as the assistant turn. `--include` merges
files that already have the chat format, such as the round 0 output. A `proofs.jsonl` or
`--include` line whose first message is not the shared `SYSTEM_MESSAGE`, or which holds
any other system message differing from it, raises `ValueError`. All inputs are
deduplicated together by `(user, answer)`: the first occurrence wins. A run that yields no
candidate line warns and contributes nothing, so an empty run still produces an (empty)
output file.

Only the train split (`datasets.split(name, "train")`) is used. A `task_id` from any
test split, a `task_id` excluded from train through another dataset's test `source_task`,
or a user message equal to `build_prompt` of a test-split task, raises `ValueError` and
fails the export: test data must never reach training, not even through another dataset's
task ID.

## Uploading `sft.jsonl` to Fireworks

1. Open the Fireworks dashboard and create a dataset from the exported `sft.jsonl` file
   (Datasets page, chat format: one `{messages: [...]}` object per line). Note the dataset
   ID for the training job.
2. Launch a supervised fine-tuning job on the base Glimmer model with that dataset,
   following docs/plan.md (Training): LoRA SFT, rank 16, over the exported proofs. The
   plan notes SFT is free until November 30, either through the dashboard or through the
   Serverless Training API when dataset uploads are disabled for the account; confirm the
   current account status in the training issue before launching.
3. When the job finishes, download the adapter in PEFT format. Record the job ID and the
   spend in the training issue, as every training run does; serving and evaluation follow
   the training issues.

## The GRPO loop

`run(config, backend, *, reward_fn=None)` in `grpo.py` drives the whole run.
`reward_fn(answer, task_spec)` grades one attempt; when it is `None`, `run` binds
`training.reward.reward` to the resource limit of `pcc.config`
(`Config().dafny.resource_limit`).

Every training task is the `vericoding` train split (`pcc.evaluation.split`) turned into
a prompt with the shared `build_prompt` from `training/prompts.py` — the same function
the SFT export uses to rebuild user turns, so training prompts compare byte-identical
with the exported lines. The sampled prompt string is `SYSTEM_MESSAGE`, a blank line,
then `build_prompt(task)`. The graded reward still scores the answer against the
immutable task prefix (the `vc-preamble`, `vc-helpers` and `vc-spec` payload parts
through the opening method brace), which must occur unchanged in the answer. The
evaluation harness reuses the same `build_prompt` when it lands, so training sees the
prompt that the evaluation uses.

## The pre-pass

The run first draws `prepass_tasks` (64) training tasks with the fixed `seed`, samples each one
`samples_per_task` (8) times with `Backend.sample`, and keeps the tasks the starting model solves
1 to 7 times — the tasks whose attempt rewards can still differ. Its sampling cost counts against
`max_spend_usd`. The kept tasks are reused in every step.

## A step

Each step draws `tasks_per_step` (16) tasks from the kept set, samples `samples_per_task`
attempts per task, and scores them with `reward_fn`. The advantage of each attempt is
`(reward − group mean) / (group std + 1e-6)` with the population standard deviation, so the
advantages of a kept group sum to 0. Groups whose rewards are all equal teach nothing and are
dropped; the rest go to `Backend.train` as `(prompt, completion, advantage)` tuples. When every
group is dropped, the run stops with `stop_reason` `"all groups dropped"` and calls no `train`.

## The stop rule and the spend

Each step's cost is estimated from the token counts the backend reports — all prompt tokens at
the prefill rate, sampled tokens at the sample rate, `TrainStats.tokens` at the train rate — with
the rates in the config (1.96, 4.88 and 5.86 per 1M tokens; one million of each costs $12.70).
A step starts only if the spend so far plus the previous step's cost stays within `max_spend_usd`
(50). Before the first step, the estimate is the pre-pass cost per sample times
`tasks_per_step * samples_per_task` (128 with the shipped values). A step that would exceed the
cap is not started. When the rule stops the run, `save()` is called once and `stop_reason` is
`"spend cap"`, except when the pre-pass alone exceeded the cap: then no step ran, `save()` is
skipped and only the reason is reported.

A `ToolError` from the backend passes through; when at least one step finished, `save()` runs
first so the trained progress is checkpointed.

## The GRPO config

`configs/glimmer_fireworks.toml` pins the run: the base model, the four rates, the $50 cap, the
pre-pass and step sizes, the seed, the LoRA rank and learning rate, and `base_adapter` — the
SFT 2 adapter the run starts from, still empty until the SFT export lands. The Fireworks backend
(a later change) reads `model` and `base_adapter` to point sampling and training at the right
adapter; the loop itself never sees them.
