# Training

This folder holds the training side of the project (docs/plan.md, Training): the shared
prompt, the graded proof reward, and the SFT data export. It lives outside `src/pcc`: it
imports `pcc` (the verifier as the reward, the datasets as the train/test source of truth)
but the package never imports it.

| File | Does |
| --- | --- |
| `prompts.py` | `SYSTEM_MESSAGE`, sent verbatim by every data line and every sampling tool |
| `reward.py` | The graded proof reward: 1 verified, 0 not verified, −1 cheat or changed spec |
| `export_sft.py` | `python -m training.export_sft`: verified solutions become Fireworks SFT data |

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
`training/export_sft.py` (instructions plus the task description, preamble, helpers, spec
and postamble; never `vc-code`, which is the reference answer). The sampling harness and
the round 0 tool must import that same `build_prompt` instead of copying it, so their user
messages compare byte-identical with the test-split prompt guard. For `proofs.jsonl` lines it
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
   following docs/plan.md (Training): LoRA SFT, rank 16, over the exported proofs. SFT is
   free until November 30, either through the dashboard or through the Serverless Training
   API when dataset uploads are disabled for the account.
3. When the job finishes, download the adapter in PEFT format and serve it with the
   configured `--lora-modules`. Record the job ID and the spend in the training issue, as
   every training run does.
