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
      pull-requests: write
    steps:
      - uses: actions/checkout@v7
      - uses: hosbe1/Proof-Carrying-Code-Meta-AI-Hackathon-2026@main
        env:
          META_API_KEY: ${{ secrets.META_API_KEY }}
```

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

### Vericoding code-and-proof comparison (test split)

Paired comparison on the same fixed 150-task subset
([subset.txt](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/subset.txt) in the Wonder
run folder). Both runs are complete for their defined task sets; neither stopped at its cost cap.
A joint `report compare` over both published folders rejects them because they cover different task
IDs (569 vs 150), so each published run is compared on its own (the exact commands are recorded in
the pull request); the paired subset row is computed from the published `summary.json`/`meta.json`
files and, for the Spark 150-task subset, from that run's raw per-task results.

| run | model | mode | n | attempts | pass@4 | pass@1 (est.) | 95% Wilson interval | cost (USD) | cost per solved task (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spark, 150-task subset ([published run](docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low/summary.md)) | muse-spark-1.3-contributor | contributor | 150 | 4 | 15/150 (10.0%) | 47/600 (7.8%) | 6.15–15.84% (pass@4) | 0.341203 | 0.02275 |
| Wonder ([vericoding-wonder-150-valid-low](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/summary.md)) | muse-wonder-1.0 | standard | 150 | 1 | — (single attempt) | 13/150 (8.7%) | 5.13–14.26% (pass@1) | 6.646832 | 0.51129 |

Full published runs (different task sets, so not task-by-task comparable):

- Spark, full eligible set: 55/569 pass@4 (9.7%, 95% Wilson interval 7.50–12.37%), pass@1
  estimate 179/2276 (7.9%), cost $1.2644484, contributor mode, complete (cost cap not reached) —
  [published run](docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low/summary.md).
- Wonder, fixed 150-task subset: 13/150 pass@1, which equals pass@4 with one attempt per task
  (8.7%, 95% Wilson interval 5.13–14.26%), cost $6.646832, standard mode, complete (cost cap not
  reached) —
  [published run](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/summary.md).

Wilson intervals cover solved tasks (pass@k); no interval is given for the fractional pass@1
estimates. With 15 vs 13 solved tasks on the shared subset the intervals overlap widely, so no
significance claim is made.

### Vericoding: Spark against Wonder

Paired comparison on the same fixed 150-task subset of the Vericoding test split (see
[subset.txt](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/subset.txt)).
Both runs are complete for their defined task set; neither stopped at a cost cap
(`summary.json` reports `stopped_at_cost_cap: false`, `meta.json` reports `stopped_early: false`).
Wilson intervals are 95% intervals over solved tasks (pass@k); the fractional Spark pass@1
estimates carry no Wilson interval. No significance claim is made.

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
The Spark subset pass@1 estimate (47/600) and subset cost ($0.341203) come from the raw
per-task attempt records of the Spark run; the full-run `summary.json` only records totals.

Full published runs (different task sets, not directly comparable):

| run | model | mode | n | pass@4 | pass@1 | cost (USD) | cost per solved task (USD) | results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spark, eligible set | muse-spark-1.3-contributor | contributor | 569 | 55/569 = 9.67% (95% CI 7.50-12.37%) | est. 179/2276 = 7.86% | 1.2644484 | 0.02299 | [summary](docs/results/2026-10-31-ea0891a-vericoding-spark-eligible-low/summary.md) |
| Wonder, fixed 150 subset | muse-wonder-1.0 | standard | 150 | n/a (1 attempt) | 13/150 = 8.67% (95% CI 5.13-14.26%) | 6.646832 | 0.51129 | [summary](docs/results/2026-10-31-60cdd58-vericoding-wonder-150-valid-low/summary.md) |

The Spark full run covers 569 of 571 eligible tasks (skipped `vericoding/DA0152`,
`vericoding/DA0818`, unbuildable prompts). HumanEvalFix spec-writer, proof-only and
Glimmer rows will be added separately once those evaluations have run; the live
palindrome certificate is tracked separately and is not claimed here.

## Development

```bash
uv sync
uv run pytest
```

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and [Dafny](https://dafny.org) 4.11. See
[docs/plan.md](docs/plan.md) for the plan, [docs/architecture.md](docs/architecture.md) for the code layout and
[AGENTS.md](AGENTS.md) for the rules every contributor, human or agent, follows. All work is tracked as GitHub issues
and follows the process in AGENTS.md.

## Licensing

[AGPL-3.0](LICENSE).

Third-party datasets, demo sources and models are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Team

A hackathon team; see the
[contributors graph](https://github.com/hosbe1/Proof-Carrying-Code-Meta-AI-Hackathon-2026/graphs/contributors) for who
built what.
