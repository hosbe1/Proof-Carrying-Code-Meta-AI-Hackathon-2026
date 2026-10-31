# Training: the GRPO loop

This folder holds the training side of the project (docs/plan.md, Training), outside the `pcc`
package: it imports `pcc` (the verifier as the reward) but the package never imports it.

- `prompts.py`: the shared system message every training prompt and the evaluation carry.
- `reward.py`: the graded Dafny reward of the proof step (1 verified, 0 not verified, −1 for a
  changed spec or an anti-cheat violation; a broken tool raises `ToolError`).
- `grpo.py`: the GRPO loop itself, against a small `Backend` protocol with no SDK and no network,
  so it runs offline in tests.
- `configs/glimmer_fireworks.toml`: the run configuration.

## The loop

`run(config, backend, *, reward_fn=None)` drives the whole run. `reward_fn(answer, task_spec)`
grades one attempt; when it is `None`, `run` binds `training.reward.reward` to the resource limit
of `pcc.config` (`Config().dafny.resource_limit`).

Every training task is the `vericoding` train split (`pcc.evaluation.split`) turned into a prompt
with the shared system message plus the immutable task prefix (the `vc-preamble`, `vc-helpers`
and `vc-spec` payload parts through the opening method brace), so training sees the prompt that
the evaluation uses. The prompt builder still stands in for the evaluation prompt helper until
that change lands; swap `_render_prompt`/`_task_spec` in `grpo.py` for it then.

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

## The config

`configs/glimmer_fireworks.toml` pins the run: the base model, the four rates, the $50 cap, the
pre-pass and step sizes, the seed, the LoRA rank and learning rate, and `base_adapter` — the
SFT 2 adapter the run starts from, still empty until the SFT export lands. The Fireworks backend
(a later change) reads `model` and `base_adapter` to point sampling and training at the right
adapter; the loop itself never sees them.
