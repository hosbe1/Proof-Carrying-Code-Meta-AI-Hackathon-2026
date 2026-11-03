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

### Vericoding: base Glimmer against SFT 1 and SFT 2

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

All runs shared the `pcc-glimmer` deployment, which stays at zero
scale and belongs to the remaining evaluation work, not to this ticket. No significance
claim is made beyond the paired counts and intervals above.

### Proof-only tasks: Spark against base Glimmer and SFT 2

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

Spark alone misses `proof-only/vericoding/DD0083`, `proof-only/vericoding/DD0271`
and `proof-only/vericoding/DD0648`; base Glimmer alone misses
`proof-only/vericoding/DD0070`, `proof-only/vericoding/DD0083`,
`proof-only/vericoding/DD0271`, `proof-only/vericoding/DD0761` and
`proof-only/vericoding/DS0052`. Of the 28 tasks base Glimmer solves, SFT 2 also
solves 19 and misses 9; it newly solves `proof-only/vericoding/DD0070` and
`proof-only/vericoding/DS0052`, for a net of 21.

SFT 2 is below base Glimmer on proof-only (21 vs 28), likely because SFT 2 was
trained only on the full-task format. The tasks come from Spark's own verified answers, which favours Spark. No significance claim is made on n = 33.

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
