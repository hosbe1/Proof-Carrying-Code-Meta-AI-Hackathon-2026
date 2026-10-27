# Architecture

How the code in `src/pcc` is organised. What the pipeline does and why is in [plan.md](plan.md); this file says where
each part lives and which rules keep the parts apart.

## Layers

Every module belongs to one layer and may import only from layers below it. Modules in the same layer never import
each other. CI enforces this with import-linter (`[tool.importlinter]` in `pyproject.toml`).

```text
entry points    cli, evaluation
pipeline        pipeline
steps           project, context, intent, spec, contracts, transpile, translate_ai,
                equivalence, prove, counterexample, second_opinion, certificate, report
infrastructure  models, verifier, sandbox
pylib           pylib
config          config
core            core
```

`pylib` has a layer of its own, below `infrastructure`: `verifier` verifies code together with `PyLib.dfy`, so it
imports `pylib`, and that import has to point down.

Why this shape:

- **Steps talk through data, not through each other.** A step takes core types and returns core types. Only
  `pipeline` knows the order of the steps, so a step can be rewritten, or built in parallel by another agent, without
  touching its neighbours.
- **One door to each outside system.** Language models only through `pcc.models.ModelClient`, the Dafny CLI only
  through `pcc.verifier`, untrusted Python only through `pcc.sandbox`. Tests swap the door (`FakeClient`), and limits,
  timeouts and cost accounting live in one place.
- **`core` changes rarely.** All shared types are there. A change to `core` touches every step: additive changes (new
  optional fields, new types) can come with a step ticket and are named in its pull request; changing or removing an
  existing field needs a ticket of its own.

## Modules

| Module | Plan step | Does |
| --- | --- | --- |
| `core` | all | Data types, errors, JSON round-trip |
| `config` | all | Models, prover stages, budgets |
| `models` | all | `ModelClient` protocol, `FakeClient`, real clients |
| `verifier` | 4, 5 | `dafny verify`, compiling back to Python, error parsing, anti-cheat scan |
| `sandbox` | 3, 4, 6 | Running untrusted Python with time and memory limits |
| `pylib` | 4 | `PyLib.dfy`: Python semantics as proven Dafny building blocks |
| `project` | 4 | Finding functions, call graph, bottom-up order, affected callers |
| `context` | 1 | Fixed evidence collector plus search agent |
| `intent` | 2 | Intent with sources, contradictions ranked by `SOURCE_RANK` |
| `spec` | 3 | Python predicate, Hypothesis and mutant checks |
| `contracts` | 4 | Contracts for library calls, recorded as assumptions |
| `transpile` | 4 | Rule-based translation, code mode and spec mode |
| `translate_ai` | 4 | Model translation for what the transpiler does not support |
| `equivalence` | 4 | Differential test and `crosshair diffbehavior` against the original |
| `prove` | 5 | Proof agent with escalating prover stages |
| `counterexample` | 6 | Search for an input that breaks the spec |
| `second_opinion` | 7 | Second spec with the body in view, as a hint |
| `certificate` | output | Certificates in `proofs/`, staleness check |
| `report` | output | Pull request comment |
| `pipeline` | 1–7 | Runs the steps, turns failures into result states |
| `cli` | output | `pcc-check`, used by the GitHub Action |
| `evaluation` | eval | Runs on fixed dataset splits, metrics |

Training code (SFT and GRPO) lives in `training/` outside the package: it imports `pcc` (for the verifier as reward)
but the package never imports it. The compute provider is decided in [plan.md](plan.md), not here.

## Data flow for one function

```text
FunctionRef
  -> context.collect            -> Context
  -> intent.derive              -> Intent
  -> spec.write, spec.check     -> Spec (frozen from here on)
  -> transpile_* or translate_ai -> Translation
  -> equivalence.check          -> EquivalenceReport   (mismatch: retranslate, at most max_retranslations)
  -> prove.prove                -> ProofOutcome
  -> counterexample.search      -> Counterexample | None   (only if not proven)
  -> second_opinion.review      -> SecondOpinion           (only on a reported problem)
  -> Result -> certificate, report
```

Expected failures (`UnsupportedConstruct`, `NotCheckable`, a failed proof) become result states in `pipeline`.
Only real bugs and broken tools (`ToolError`) escape as exceptions.

The intent step (`intent.derive`) and the spec step (`spec.write`) are blind to the function body. Their prompts carry
the name, signature, type hints and, for the intent, the context, but never the body, so a bug in the code cannot leak
into what the function should do. Although `FunctionRef.source` holds the body, these steps do not put it into a
prompt. Each of the two steps has a test that inspects `FakeClient.requests` and asserts that `function.source` appears
in no prompt.

## Tests

- `tests/` mirrors `src/pcc/`: tests for `pcc.spec` go to `tests/spec/`.
- Default run: no network, no keys. Model calls use `FakeClient`.
- `@pytest.mark.dafny`: needs the Dafny CLI. CI installs it and sets `PCC_REQUIRE_DAFNY=1`, so a missing CLI fails the
  run there; locally the test is skipped without it.
- `@pytest.mark.live`: calls real models; never runs in CI, run with `uv run pytest -m live`.
