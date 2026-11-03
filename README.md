# Proof-Carrying Code

Turn Python functions into formally verified Dafny code, together with its proof.

Built for the [Meta Global AI Developer Hackathon 2026](https://dev.meta.ai/events/global-hackathon).

## Built during the hackathon

Everything in this repository was created after the official hackathon kickoff on October 26, 2026, 12:00 PM ET.
No code, data or assets were prepared before the event.

## What it does

Every Python function gets one of two things: a proof certificate, a Dafny file with spec and proof that anyone can
re-check with `dafny verify`, or a counterexample, a concrete input on which the code breaks its spec, delivered as a
ready-to-run Python test. The pipeline collects context, writes the intent in plain text and the spec as an executable
Python predicate without ever seeing the function body, translates the function to Dafny with a rule-based transpiler
(or a model fallback where the transpiler cannot reach), proves it with escalating prover stages, searches for a
counterexample only after a failed proof, and reports one result per function (`proven`, `counterexample`, `open` or
`not_checkable`) with its certificate and assumptions. Certificates live with the code under `proofs/` and stay
re-checkable without this package.

One example pull request comment:

```markdown
<!-- pcc-check -->
## Proof-carrying code

12 proven, 1 counterexample, 3 open.

### ❌ `strings/palindrome.py::is_palindrome_recursive` — counterexample (transpiler)

Intent: return whether the string reads the same forward and backward (docstring).

```python
def test_palindrome_counterexample():
    assert is_palindrome_recursive("abcdba") is True
```

`pre` accepted 100% of the generated inputs. Assumptions: none.

<details><summary>Dafny spec</summary>

```dafny
predicate spec(s: string, result: bool) {
  result == (s == Reverse(s))
}
```

</details>
```

## Quick start

Add the action to a repository that holds the Python code to check. Store a Meta Model API key as an Actions secret
named `META_API_KEY`; the action sets `PCC_META_ALLOWED=1` for its own run and never bakes a key in.

```yaml
name: proof-carrying-code
on:
  pull_request:
jobs:
  verify:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
      - uses: hosbe1/Proof-Carrying-Code-Meta-AI-Hackathon-2026@main
        with:
          meta-api-key: ${{ secrets.META_API_KEY }}
```

`fetch-depth: 0` provides the Git history needed to compare against the pull request's base branch.

On each pull request the action checks the changed functions plus affected callers with `pcc-check run`, uploads the
`proofs/` certificates as a workflow artifact, and updates one comment on the pull request. The check turns red only on
a real finding: a `counterexample`, or a stale certificate whose function is no longer proven.

## How it works

The pipeline runs eight steps per function; the full description, with budgets, checks and the supported Python subset,
is in [docs/plan.md](docs/plan.md):

1. Collect context: a fixed collector plus an agent that searches further when the intent is unclear.
2. Write the intent in plain text, citing a source for every claim, without seeing the function body.
3. Write the spec as an executable Python predicate, checked against the original with Hypothesis and against mutants.
4. Translate to Dafny: a rule-based transpiler for what it covers, a model fallback for the rest, both checked by
   differential testing against the original.
5. Prove with escalating prover stages until `dafny verify` passes; only ghost code may be added.
6. Search for a counterexample only after a failed proof; it is always a really executed input.
7. Second opinion only when something is in doubt, with the body in view, as a hint rather than a verdict.
8. Report a certificate under `proofs/` and a comment on the pull request.

Code layout, layers and the doors to outside systems (models, Dafny, untrusted Python) are described in
[docs/architecture.md](docs/architecture.md), and the contributor rules in [AGENTS.md](AGENTS.md).

## Which Meta model does what

At runtime the pipeline calls only Meta models. Every endpoint is OpenAI-compatible and configured by environment
variables: the Meta Model API by `META_API_KEY`, a Glimmer deployment by `PCC_GLIMMER_BASE_URL` and
`PCC_GLIMMER_API_KEY`. A stage whose endpoint is not configured is skipped.

| Step | Model |
|---|---|
| Intent, spec, AI fallback translation, counterexample search, second opinion | Muse Spark (`muse-spark-1.3`) |
| Proof, stage 1 (up to 8 attempts) | Trained Muse Glimmer, served on our own deployment |
| Proof, stage 2 (up to 3 attempts) | Muse Spark (`muse-spark-1.3`) |
| Proof, stage 3 (1 attempt, functions of at most 40 source lines) | Muse Wonder (`muse-wonder-1.0`) |

The spec writer defaults to Spark; the model comparison may promote a better model to the default. Runs over public
data (evaluation, training data, bug hunt) use Spark in contributor mode (`muse-spark-1.3-contributor`), which runs
only in the Meta environment. Licenses and terms of every model sit in [THIRD_PARTY.md](THIRD_PARTY.md).

## Results

The bug hunt report (issue #58) will be published at `docs/results/bug-hunt.md`: projects checked, result states per
function, confirmed bugs with links to the reported issues. That file does not exist yet, so this section names its
path without linking it. The model comparison (base Glimmer, SFT 1, SFT 2 against Spark and Wonder, with intervals
and cost per solved task) is published the same way once the evaluations have run.

### Vericoding: Spark against Wonder

Paired comparison on the same fixed 150-task subset of the Vericoding test split (see
[subset.txt](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/subset.txt)).
Both runs are complete for their defined task set; neither stopped at a cost cap
(`summary.json` reports `stopped_at_cost_cap: false`, `meta.json` reports `stopped_early: false`).
Wilson intervals are 95% intervals over solved tasks (pass@k); the fractional Spark pass@1
estimates carry no Wilson interval. No significance claim is made. A joint `report compare`
over both published folders rejects them because they cover different task IDs (569 vs 150),
so each published run is compared on its own (the exact commands are recorded in the pull
request).

| run | model | mode | n | pass@4 | pass@1 | cost (USD) | cost per solved task (USD) | results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common 150, Spark | muse-spark-1.3-contributor | contributor | 150 | 15/150 = 10.00% (95% CI 6.15-15.84%) | est. 47/600 = 7.83% | 0.341203 | 0.02275 | [summary](docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low/summary.md) |
| common 150, Wonder | muse-wonder-1.0 | standard | 150 | n/a (1 attempt) | 13/150 = 8.67% (95% CI 5.13-14.26%) | 6.646832 | 0.51129 | [summary](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/summary.md) |

The subset rows above share the exact 150 task IDs of
`docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/subset.txt`.
Spark solved 15 of them at pass@4
(`vericoding/DD0083`, `vericoding/DD0085`, `vericoding/DD0087`, `vericoding/DD0151`,
`vericoding/DD0271`, `vericoding/DD0654`, `vericoding/DD0680`, `vericoding/DD0727`,
`vericoding/DS0015`, `vericoding/DT0019`, `vericoding/DV0056`, `vericoding/DV0069`,
`vericoding/DV0084`, `vericoding/DV0148`, `vericoding/DV0173`;
intersection of the published Spark `solved_task_ids` with the 150-task subset).
The Spark subset pass@1 estimate (47/600 = 7.83%) and subset cost ($0.341203, hence
$0.341203 / 15 = $0.02275 per solved task) are the canonical values recorded for this
comparison from the raw per-task attempt records of the published Spark run
(commit `ea0891a6348996e34fad44dfd643ab0d4a18a1be`, 2026-10-31, 4 attempts per task;
see `docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low/meta.json`):
pass@1 is successful attempts over 150 x 4 = 600 attempts on exactly the subset IDs
above, and the subset cost is the summed per-task cost over those same IDs.
Those raw per-task records are not part of the published result folder, which records
only run totals in `summary.json`, so the two figures cannot be recomputed from the
published files alone. The 15 solved task IDs above
remain verifiable as the intersection of the published Spark `solved_task_ids` with the
150-task subset, as are the Wilson intervals and the divisions (47/600,
$0.341203 / 15, $6.646832 / 13).

Full published runs (different task sets, not directly comparable):

| run | model | mode | n | pass@4 | pass@1 | cost (USD) | cost per solved task (USD) | results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spark, eligible set | muse-spark-1.3-contributor | contributor | 569 | 55/569 = 9.67% (95% CI 7.50-12.37%) | est. 179/2276 = 7.86% | 1.2644484 | 0.02299 | [summary](docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low/summary.md) |
| Wonder, fixed 150 subset | muse-wonder-1.0 | standard | 150 | n/a (1 attempt) | 13/150 = 8.67% (95% CI 5.13-14.26%) | 6.646832 | 0.51129 | [summary](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/summary.md) |

The Spark full run covers 569 of 571 eligible tasks (skipped `vericoding/DA0152`,
`vericoding/DA0818`, unbuildable prompts). The HumanEvalFix spec-writer comparison is
published separately at [spec-writer-comparison](docs/results/2026-11-02-spec-writer-comparison.md);
the live palindrome certificate is tracked separately and is not claimed here.

### Vericoding: base Glimmer against SFT 1, SFT 2, SFT 3 and SFT 4

Paired comparison on the identical 569 eligible task IDs of the Vericoding test split
(same commit `5f5c0f4`, same split hash `014017d4...`, same skipped unbuildable prompts
`vericoding/DA0152`, `vericoding/DA0818`). Both runs are complete for their defined task
set; neither stopped at a cost cap (`summary.json` reports `stopped_at_cost_cap: false`,
`meta.json` reports `stopped_early: false`, status `done`). Each task ran 4 attempts at
temperature 0.8 with at most 4096 output tokens, reasoning strength `low`, served from the
same deployment at FP8 precision, checked with Dafny 4.11.0 at resource limit 1000000.
The comparator is the `low`-strength baseline; the extra `medium`-strength repeat in
`docs/results/2026-11-01-5f5c0f4-baseline-glimmer-base-medium` is not part of this
comparison. Wilson intervals are 95% intervals over solved tasks (pass@k). Token cost is
$0.00 for both runs (the Glimmer endpoint bills $0 per token); GPU cost is labelled
separately per run below and comes from the run notes, not from `summary.json`.

| run | model | adapter | n | pass@4 | pass@1 | token cost (USD) | GPU cost (USD) | results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline, low strength | glimmer | none | 569 | 43/569 = 7.56% (95% CI 0.057-0.100) | 0.054 | 0.00 | about $23, including about $13 of lost runs | [summary](docs/results/2026-11-01-5f5c0f4-baseline-glimmer-base/summary.md) |
| SFT 1, low strength | glimmer-sft | `pcc-glimmer:/adapters/sft1` | 569 | 158/569 = 27.77% (95% CI 0.242-0.316) | 0.202 | 0.00 | about $2 of shared-deployment GPU time | [summary](docs/results/2026-11-01-5f5c0f4-eval-glimmer-sft1/summary.md) |
| SFT 2, low strength | glimmer-sft2 | `pcc-glimmer:/adapters/sft2` | 569 | 255/569 = 44.82% (95% CI 0.408-0.489) | 0.363 | 0.00 | about $3 of shared-deployment GPU time | [summary](docs/results/2026-11-02-ea911e5-eval-glimmer-sft2/summary.md) |
| SFT 3, low strength | glimmer-sft3 | `pcc-glimmer:/adapters/sft3` | 569 | 259/569 = 45.52% (95% CI 0.415-0.496) | 0.370 | 0.00 | billing total pending | [summary](docs/results/2026-11-02-93a247f-eval-glimmer-sft3/summary.md) |
| SFT 4, low strength | glimmer-sft4 | `pcc-glimmer:/adapters/sft4` | 569 | 273/569 = 47.98% (95% CI 0.439-0.521) | 0.407 | 0.00 | billing total pending | [summary](docs/results/2026-11-03-2a2a777-eval-glimmer-sft4/summary.md) |

Of the 43 tasks the baseline solves, SFT 1 also solves 35 and misses 8; it newly solves
123 tasks, for a net gain of 115 tasks on the same IDs (see the `solved_task_ids` lists in
both `summary.json` files). SFT 1 was trained on 301 Vericoding train-split rows only
(hash `c3d508bf...`, see [training.md](docs/results/training.md)); the evaluated IDs are
held-out test tasks.

SFT 2 ran at commit `ea911e5` on the same split hash, task IDs and settings. It was
trained from base Glimmer on 1751 rows: the 301 above plus 1450 verified proofs that SFT 1
found on the train split in rejection-sampling round 1. Of the 158 tasks SFT 1 solves,
SFT 2 also solves 145 and misses 13; it newly solves 110, for a net gain of 97 tasks.
The model improves on proofs it produced itself, checked only by Dafny, with no human labels.

SFT 3 ran at commit `93a247f` on the same split hash, task IDs and settings. It was trained from base
Glimmer on 3019 rows: the 1990 round 0-2 proofs plus 1029 proof-only lines built from the round 0-2
train answers (see [training.md](docs/results/training.md)). Of the 255 tasks SFT 2 solves, SFT 3
also solves 227 and misses 28; it newly solves 32, for a net gain of 4 tasks. SFT 4 ran at commit
`2a2a777` on the same split hash and task IDs. It was trained from base Glimmer on 3941 rows: the
SFT 3 data plus 922 proof-only rows that SFT 3 itself generated on 1029 train proof-only tasks.
Of the 259 tasks SFT 3 solves, SFT 4 also solves 247 and misses 12; it newly solves 26, for a net
gain of 14 tasks. The pre-training gate (SFT 4 replaces SFT 3 only at pass@4 >= 253/569) passed at
273/569, so SFT 4 is the main model.

Eight attempts (pass@8) on the same 569 task IDs and split hash give a second paired table:

| run | model | n | pass@8 | pass@1 | token cost (USD) | results |
| --- | --- | --- | --- | --- | --- | --- |
| baseline, low strength | glimmer | 569 | 45/569 = 7.91% (95% CI 0.060-0.104) | 0.053 | 0.00 | [summary](docs/results/2026-11-03-2a2a777-t8-base/summary.md) |
| Spark, contributor | muse-spark-1.3-contributor | 569 | 60/569 = 10.54% (95% CI 0.083-0.133) | 0.081 | 2.53 | [summary](docs/results/2026-11-03-2a2a777-t8-spark/summary.md) |
| SFT 1, low strength | glimmer-sft | 569 | 184/569 = 32.34% (95% CI 0.286-0.363) | 0.210 | 0.00 | [summary](docs/results/2026-11-03-2a2a777-t8-sft1/summary.md) |
| SFT 2, low strength | glimmer-sft2 | 569 | 274/569 = 48.15% (95% CI 0.441-0.523) | 0.358 | 0.00 | [summary](docs/results/2026-11-03-2a2a777-t8-sft2/summary.md) |
| SFT 3, low strength | glimmer-sft3 | 569 | 268/569 = 47.10% (95% CI 0.430-0.512) | 0.359 | 0.00 | [summary](docs/results/2026-11-03-2a2a777-t8-sft3/summary.md) |
| SFT 4, low strength | glimmer-sft4 | 569 | 282/569 = 49.56% (95% CI 0.455-0.537) | 0.404 | 0.00 | [summary](docs/results/2026-11-03-fe93ada-t8-sft4/summary.md) |

The six runs cover the identical task IDs; only the harness commit differs for the last row
(`fe93ada` instead of `2a2a777`, a verifier-only change), and the split hash is the same
throughout. The Spark run costs $2.53 in contributor mode; every Glimmer run costs $0.00 in
tokens, with shared-deployment GPU billing totals pending.

All runs shared the `pcc-glimmer` deployment, which stays at zero
scale and belongs to the remaining evaluation work, not to this ticket. No significance
claim is made beyond the paired counts and intervals above.

### Proof-only tasks: Spark against base Glimmer, SFT 2, SFT 3 and SFT 4

Paired comparison on the same 33 proof-only tasks: the step-5 evaluation set on
the Vericoding test split, built from the verified Spark answers of that split.
The task build (`proof_only make` from the Spark Vericoding test run of #58)
skipped 20 answers without ghost code and 2 already-verifying programs. All three
runs are complete for their task set at commit `93a247f`; none stopped at a cost
cap (`summary.json` reports `stopped_at_cost_cap: false`, `meta.json` reports
`stopped_early: false`, status `done`). Each task ran 4 attempts with
`--max-cost-usd 5`; the Glimmer runs used reasoning strength `low`, served from the
same deployment at FP8 precision with
`PCC_GLIMMER_ADAPTERS="glimmer-sft=/adapters/sft1,glimmer-sft2=/adapters/sft2"`,
checked with Dafny 4.11.0 at resource limit 1000000. Wilson intervals are 95%
intervals over solved tasks (pass@k); the fractional pass@1 estimates carry no
Wilson interval. Token cost is $0.00 for both Glimmer runs (the Glimmer endpoint
bills $0 per token); GPU cost is labelled separately below and comes from the run
notes, not from `summary.json`. Published `meta.json` files redact only the
harness-recorded absolute tasks-file path to the portable
`proof-only-test/tasks.jsonl` and `vericoding/proof-only-test`; every other
field is verbatim and `split_sha256` pins the task bytes.

| run | model | adapter | mode | n | pass@4 | pass@1 | token cost (USD) | GPU cost (USD) | results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spark | muse-spark-1.3-contributor | none | contributor | 33 | 30/33 = 90.91% (95% CI 0.764-0.969) | est. 0.803 | 0.05 | n/a | [summary](docs/results/2026-11-02-93a247f-proof-only-spark/summary.md) |
| base Glimmer, low strength | glimmer | none | standard | 33 | 28/33 = 84.85% (95% CI 0.691-0.933) | est. 0.614 | 0.00 | about 1 USD for both Glimmer runs | [summary](docs/results/2026-11-02-93a247f-proof-only-glimmer-base/summary.md) |
| SFT 2, low strength | glimmer-sft2 | `pcc-glimmer:/adapters/sft2` | standard | 33 | 21/33 = 63.64% (95% CI 0.466-0.778) | est. 0.394 | 0.00 | about 1 USD for both Glimmer runs | [summary](docs/results/2026-11-02-93a247f-proof-only-glimmer-sft2/summary.md) |
| SFT 3, low strength | glimmer-sft3 | `pcc-glimmer:/adapters/sft3` | standard | 33 | 24/33 = 72.73% (95% CI 0.558-0.849) | est. 0.576 | 0.00 | billing total pending | [summary](docs/results/2026-11-02-93a247f-proof-only-glimmer-sft3/summary.md) |
| SFT 4, low strength | glimmer-sft4 | `pcc-glimmer:/adapters/sft4` | standard | 33 | 24/33 = 72.73% (95% CI 0.558-0.849) | est. 0.689 | 0.00 | billing total pending | [summary](docs/results/2026-11-03-2a2a777-proof-only-glimmer-sft4/summary.md) |

Spark alone misses `proof-only/vericoding/DD0083`, `proof-only/vericoding/DD0271`
and `proof-only/vericoding/DD0648`; base Glimmer alone misses
`proof-only/vericoding/DD0070`, `proof-only/vericoding/DD0083`,
`proof-only/vericoding/DD0271`, `proof-only/vericoding/DD0761` and
`proof-only/vericoding/DS0052`. Of the 28 tasks base Glimmer solves, SFT 2 also
solves 19 and misses 9; it newly solves `proof-only/vericoding/DD0070` and
`proof-only/vericoding/DS0052`, for a net of 21.

SFT 2 is below base Glimmer on proof-only (21 vs 28), likely because SFT 2 was
trained only on the full-task format. The tasks come from Spark's own verified answers, which favours Spark. No significance claim is made on n = 33.

SFT 3 and SFT 4 each solve 24 of 33 (pass@1 0.576 and 0.689). Of the 28 tasks base Glimmer
solves, SFT 3 also solves 22 and misses 6; it newly solves `proof-only/vericoding/DD0070` and
`proof-only/vericoding/DS0052`. SFT 4 matches that count with the same 22, but misses
`proof-only/vericoding/DD0701` instead of `proof-only/vericoding/DD0648`. Against SFT 2, SFT 3
newly solves `proof-only/vericoding/DD0736`, `proof-only/vericoding/DV0062`,
`proof-only/vericoding/DV0068` and `proof-only/vericoding/DV0108` while missing
`proof-only/vericoding/DD0648`; SFT 4 differs from SFT 3 only on `proof-only/vericoding/DD0648`
(solved) versus `proof-only/vericoding/DD0701` (missed). Like every other model, both miss
`proof-only/vericoding/DD0083` and `proof-only/vericoding/DD0271`.

### Proof-only v2: 327 tasks built from test-split answers

Paired comparison on 327 proof-only tasks built from verified model answers on the Vericoding test
split (`runs/proof-only-v2/tasks.jsonl`, SHA-256 `873c8249...`, pinned in each `meta.json` as
`split_sha256`). Each task ran 4 attempts with `--max-cost-usd 5`; every Glimmer run below except
the last used reasoning strength `low`, served from the same deployment at FP8 precision, checked
with Dafny 4.11.0 at resource limit 1000000. Wilson intervals are 95% intervals over solved tasks
(pass@k). Token cost is $0.00 for every Glimmer run; the Spark run costs $0.67 in contributor mode.

| run | model | adapter | mode | n | pass@4 | pass@1 | token cost (USD) | GPU cost (USD) | results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base Glimmer, low strength | glimmer | none | standard | 327 | 235/327 = 71.87% (95% CI 0.668-0.765) | est. 0.530 | 0.00 | billing total pending | [summary](docs/results/2026-11-02-93a247f-po2-base-low/summary.md) |
| Spark | muse-spark-1.3-contributor | none | contributor | 327 | 257/327 = 78.59% (95% CI 0.738-0.827) | est. 0.593 | 0.67 | n/a | [summary](docs/results/2026-11-02-93a247f-po2-spark/summary.md) |
| SFT 1, low strength | glimmer-sft | `pcc-glimmer:/adapters/sft1` | standard | 327 | 268/327 = 81.96% (95% CI 0.774-0.857) | est. 0.588 | 0.00 | billing total pending | [summary](docs/results/2026-11-03-93a247f-po2-sft1-low/summary.md) |
| SFT 2, low strength | glimmer-sft2 | `pcc-glimmer:/adapters/sft2` | standard | 327 | 198/327 = 60.55% (95% CI 0.552-0.657) | est. 0.365 | 0.00 | billing total pending | [summary](docs/results/2026-11-03-93a247f-po2-sft2-low/summary.md) |
| SFT 3, low strength | glimmer-sft3 | `pcc-glimmer:/adapters/sft3` | standard | 327 | 283/327 = 86.54% (95% CI 0.824-0.898) | est. 0.745 | 0.00 | billing total pending | [summary](docs/results/2026-11-02-93a247f-po2-sft3-low/summary.md) |
| SFT 4, low strength | glimmer-sft4 | `pcc-glimmer:/adapters/sft4` | standard | 327 | 282/327 = 86.24% (95% CI 0.821-0.896) | est. 0.758 | 0.00 | billing total pending | [summary](docs/results/2026-11-03-2a2a777-po2-sft4-low/summary.md) |
| SFT 3, medium strength | glimmer-sft3 | `pcc-glimmer:/adapters/sft3` | standard | 327 | 285/327 = 87.16% (95% CI 0.831-0.904) | est. 0.739 | 0.00 | billing total pending | [summary](docs/results/2026-11-02-93a247f-po2-sft3-med/summary.md) |

Task-source caveat: the 327 tasks are built from model answers on the test split, so most come from
SFT answers. `runs/proof-only-v2/sources.json` maps each task to the models whose answers it came
from (base 27, base-medium 26, SFT 1 68, SFT 2 130, SFT 3 111, Spark 33), so every model meets many
tasks built from its own answers. The comparison therefore also reports, per model, the rate on
tasks not sourced from the model itself and the rate on the 53 tasks sourced only from base or
Spark answers: base 211/300 = 70.33% and 45/53 = 84.91%; Spark 227/294 = 77.21% and 45/53 =
84.91%; SFT 1 207/259 = 79.92% and 41/53 = 77.36%; SFT 2 120/197 = 60.91% and 37/53 = 69.81%;
SFT 3 low 182/216 = 84.26% and 41/53 = 77.36%; SFT 4 low 282/327 = 86.24% (no task is sourced
from SFT 4, so the non-self rate equals the overall rate) and 41/53 = 77.36%; SFT 3 medium
185/216 = 85.60% and 42/53 = 79.25%. No significance claim is made beyond the paired counts and
intervals above.

## Development

```bash
uv sync
uv run pytest
```

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and [Dafny](https://dafny.org) 4.11. See
[docs/plan.md](docs/plan.md) for the plan, [docs/architecture.md](docs/architecture.md) for the code layout and
[AGENTS.md](AGENTS.md) for the rules every contributor, human or agent, follows. All work is tracked as GitHub issues
and follows the process in AGENTS.md.

The Vericoding evaluation harness (`python -m pcc.evaluation vericoding ...`) samples independent proofs per task;
`--repair N` sends up to N Dafny-feedback repair turns per failing attempt in the same conversation and records
`repair_turns_used` per task (`0` when the first turn verifies).

## Licensing

[AGPL-3.0](LICENSE).

Third-party datasets, demo sources and models are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Team

A hackathon team; see the
[contributors graph](https://github.com/hosbe1/Proof-Carrying-Code-Meta-AI-Hackathon-2026/graphs/contributors) for who
built what.
