# Run notes: baseline-glimmer-base

Ticketed baseline of base Glimmer on the Vericoding test split (issue #77).

- Command (full reproduction, env included):
  `PCC_GLIMMER_REASONING_STRENGTH=low python -m pcc.evaluation vericoding
  --split test --endpoint glimmer --model glimmer --attempts 4
  --max-cost-usd 1 --concurrency 32 --out runs/baseline-glimmer-base`.
  `meta.json` records the same attempts, split hash `014017d4...`,
  commit `5f5c0f4`, endpoint `glimmer` and model `glimmer`; the harness does
  not store concurrency or reasoning strength, so this file does.
- Code: commit `5f5c0f4` including the Glimmer client fixes required for any
  verified answer (HTTP 408 retry, `chat_template_kwargs` reasoning strength,
  leaked-channel stripping in `src/pcc/models`); earlier commits cannot
  reproduce this baseline. The fixes stay in this pull request because
  reverting them would invalidate the published numbers; a follow-up code
  ticket can own them.
- Sampling: concurrency 32, `PCC_GLIMMER_REASONING_STRENGTH=low`.
  Low was chosen per the coordination decision recorded in the issue #77
  comments of 2026-11-01: the template default (`high`) returns reasoning only
  and is cut at the 4096-token limit (0 of 569 verified, not published), and
  `low` is also the setting of the SFT 1 eval, so the comparison is
  like-for-like.
- Result: 43 of 569 eligible tasks verified (pass@k 0.076, 95% CI 0.057-0.100).
  2 of 571 tasks skipped (unbuildable prompts DA0152, DA0818).
  Token cost $0.00; GPU time is billed on the deployment (see below).
- Published from `runs/baseline-glimmer-base` with
  `python -m pcc.evaluation.report publish`. `meta.json`, `summary.json` and
  `summary.md` are the verbatim publish output; this file only records setup
  the harness does not store (`reasoning_strength`, concurrency).
- History mapping: `docs/results/history.md` carries one row per published run
  with columns (date, commit, model, dataset, n, rate, cost) and no run-name
  column, so the two rows added here share every field except rate. The first
  of the two rows (rate 0.076) is the ticketed baseline in this folder
  (`...-baseline-glimmer-base`, strength `low`); the second row (rate 0.069)
  is the extra medium-strength repeat in
  `...-baseline-glimmer-base-medium`. Use this mapping, not `history.md`
  alone, to attribute a row to a reasoning strength.

## Deployment and spend

- Warm-up: no separate warm-up receipt is stored in this branch. The
  deployment (`pcc-glimmer`) was created under #76 and was already serving
  when the baseline started: on 2026-11-01 18:46 UTC the server showed 16
  active requests and 2.53M generated tokens (issue #77 coordination
  comment). The completed sampling itself — verified completions returned
  by the endpoint for both runs — evidences that the endpoint answered
  requests.
- Done criterion "deployment is stopped": NOT met by this pull request. The
  run was served by the shared app `pcc-glimmer`
  (`ap-TIrDtVN5Vc8Qd0A4zekEqJ`), used by both `glimmer` and `glimmer-sft`.
  The app is intentionally left up for the queued SFT 1 sampling (#171); the
  final stop and its receipt belong to the final active evaluation, not to
  this ticket, because stopping here would break #171. Merging this pull
  request with `Closes #77` therefore still needs an explicit human-owner
  decision; otherwise the stop criterion stays open.
- Stop handoff for the finisher of the final active evaluation: run
  `modal app stop pcc-glimmer` after the last sampling finishes, then record
  the command output (or `modal app list` showing no running containers) and
  the final Modal spend in that evaluation's issue. Cost mitigation observed
  here (not a stop): at 2026-11-01 00:11 CET the app had no request for 15
  minutes and scaled to zero. GPU cost accrues while containers run; any spend
  up to the final stop is recorded where the stop is recorded.
- Spend for this issue: about $23 of Modal GPU credit, over the 3 H100-hour
  (~$12) cap from the "evaluation and demo serving" bucket. About $13 went to
  the lost runs (base-URL 404 run, 408-crash run that saved nothing,
  high-strength run at 0 of 569) and about $10 to the low and medium runs;
  about $2 of the deployment cost belongs to the #57 eval. Token cost is
  $0.00 (the Glimmer endpoint bills $0 per token). The overrun and its cause
  (lost runs that found two bugs every later eval needs) are noted as accepted
  in the issue #77 comment of 2026-11-01; together with #57 the runs stay well
  under the shared $60 evaluation budget. Human-owner acceptance of the
  overrun is still required before merge; this pull request does not claim cap
  compliance.
- Review acceptance: the PO stand-in assessment in the PR discussion of
  2026-11-01 accepts the stop transfer, the spend overrun, the
  `src/pcc/models` fixes, the `low` strength and the two-row `history.md`
  mapping (empty container list at 00:44 CET, cost stopped at scale-to-zero;
  overrun against the shared $60 budget; fixes required for any verified
  answer; `low` like-for-like with the SFT 1 eval). This is a stand-in
  assessment only. Whether to merge with the stop transferred and the cap
  exceeded is a human-owner decision.
