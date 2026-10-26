# Plan

## Goal

Muse Spark and Muse Glimmer turn Python code into formally verified Dafny code, together with its proof. Glimmer is
trained with the proof checker as its reward.

The original Python code stays as it is. Every function gets one of two things:

- a **proof certificate**: a Dafny file with spec and proof that anyone can re-check with `dafny verify`, or
- a **counterexample**: a concrete input on which the code breaks its spec, delivered as a ready-to-run Python test.

The certificate travels with the code, which is what "proof-carrying code" means.

## Result states

Every function ends in exactly one of four states:

| State | Meaning |
|---|---|
| `proven` | The code satisfies the spec for all inputs, possibly under listed assumptions. |
| `counterexample` | An executed input shows that the code breaks the spec. |
| `open` | No proof was found, and no counterexample either. |
| `not_checkable` | The intent, the spec or the translation did not pass its checks. The reason is given. |

Verification sits behind one interface, `verify(code) -> state`. Dafny is the only backend in this project.

## Pipeline

```
Python function
  │
  ├─ 1. Collect context     fixed collector + agent that searches further when unclear
  ├─ 2. Write intent        plain text with cited sources, conflicts flagged
  ├─ 3. Write spec          executable Python predicate, checked with Hypothesis and mutants
  ├─ 4. Translate           Python → Dafny for code and spec, differential check
  ├─ 5. Prove               trained Glimmer → Spark Open or Spark → Wonder
  ├─ 6. Search counterexample   only after a failed proof
  ├─ 7. Second opinion      only when step 5 or 6 reports a problem
  └─ 8. Report              certificate in proofs/, comment on the pull request
```

### 1. Collect context

The intent of a function lives more in its surroundings than in its body. A fixed collector gathers a base package for
every function:

- **Required:** name, signature, type hints, docstring, comments, existing tests.
- **When available:** call sites, commit history of the function, pull request description, linked issues, README and
  project docs, conventions from similar functions.

An agent with tools (read files, search code, query git history) may search further when the base package leaves the
intent unclear.

### 2. Write the intent in plain text

The model writes what the function **should** do, in plain language, and cites a source for every claim, for example
"inclusive from 100 € (commit a3f9)". The function body is **not** shown in this step, so a bug in the code cannot leak
into the intent.

When sources contradict each other, a fixed ranking decides:
tests → issues and pull request → commits → docstring → comments → name.
Every contradiction is reported in the result. If the contradictions are so strong that no sensible spec can be
written, the function becomes `not_checkable` with the reason.

### 3. Write the spec

The spec is written as an executable Python predicate, for example:

```python
def spec_maximum(xs, result):
    return result in xs and all(x <= result for x in xs)
```

Before anything is proven, the spec is tested:

- **Against the original code:** Hypothesis calls the real function on many random inputs and checks the predicate.
  If the original already breaks it, either the code or the spec is wrong, which steps 6 and 7 sort out.
- **Against broken variants (mutants):** a good spec must reject them. Mutants come from fixed rules
  (`<` → `<=`, `+` → `-`, constants) and from the model ("insert bugs a developer would really make").
  Variants that behave exactly like the original on many inputs are filtered out first.

A spec that passes these checks is frozen. Later steps may not change it.

### 4. Translate to Dafny

The model translates both the function and the spec predicate to Dafny. There is no hand-written transpiler.

Each translation is checked differentially: the Dafny version is compiled back to Python
(`dafny build --target:py`) and compared with the original on 10,000 random inputs plus edge cases proposed by the
model (empty collections, negative numbers, boundaries). Equal means equal results **and** equal error behavior: if the
original raises on an input, the translation must raise too or carry a matching precondition.

On a mismatch the model retranslates with the counterexample as a hint, up to three times. After that the function is
`not_checkable`. The certificate states openly that equivalence was tested on many inputs, not proven for all.

**Calls inside the project** are handled bottom-up. Callees are proven first; callers then use only the callee's spec,
as Dafny does anyway. Independent functions can be proven in parallel. If a callee is not proven, or functions call
each other recursively, the callee's spec is used as an assumption.

**Parts that cannot be proven** (libraries, files, network) are modeled as contracts. Library type stubs give the base,
the model adds the behavior. Every contract is listed as an assumption in the certificate, so a result reads
"proven under 2 assumptions" with the list.

### 5. Prove

The proof agent adds loop invariants, lemmas and hints until `dafny verify` passes. After each failed attempt it gets
Dafny's error message back.

| Stage | Model | Attempts |
|---|---|---|
| 1 | trained Glimmer | up to 8 |
| 2 | Spark Open or Spark, whichever measures better | up to 3, with all previous failures |
| 3 | Wonder | 1, only for small or marked functions |

These numbers are starting values. They are tuned after the first evaluation run and are configurable in the action.

**Anti-cheat:** the checker rejects any proof that changes the frozen spec or uses `assume`, `{:axiom}` or
`{:verify false}`. Contracts from step 4 are allowed only in a separate, generated assumptions module that is listed in
the certificate.

### 6. Search for a counterexample

Only after the proof failed:

1. Hypothesis, focused on the `ensures` clause Dafny could not prove.
2. Suspicious inputs proposed by the model, each executed for real.

If the real Python code breaks the spec, the state is `counterexample`, with the input as a ready-to-run test.
Otherwise the state is `open`. A `counterexample` is always backed by an executed input, never by a failed proof alone.

A counterexample can also mean the spec is wrong rather than the code. That is why the result shows the plain-text
intent with its sources right next to the counterexample.

### 7. Second opinion

Only when step 5 or 6 reports a problem, or the sources contradict each other, the model writes a second spec **with**
the function body in view and compares it with the blind one. It is a hint, not a verdict: a model that reads the code
can copy its bug. Whether this lowers false alarms is measured on HumanEvalFix.

## Supported Python

The goal is a result for every function, and as many proofs as possible. Support grows in tiers:

1. Pure functions with numbers, strings, lists, dicts, tuples
2. Classes and methods that only change their own state
3. Functions using libraries, files or network, through contracts
4. Everything else, such as `eval`, metaprogramming or threads: `not_checkable` with the reason

The share of functions per tier is reported as a metric.

## Product: GitHub Action

The pipeline runs as a GitHub Action in the user's own CI. Nothing runs on our servers, so nothing of ours can fail
during the demo or the judging.

- **Scope:** on each pull request, the changed functions plus every caller whose proof may be affected. A manually
  triggered run covers the whole repository once, in batches.
- **API key:** each user stores their own Meta Model API key as a GitHub secret. No key is built into the action.
- **Trained Glimmer:** hosted as a LoRA adapter on Fireworks if that is possible per token, otherwise on Modal with
  vLLM. If no Glimmer endpoint is configured or reachable, the action proves with Spark through the Meta Model API.

### Certificate

Certificates live in the repository under `proofs/`: per function the Dafny file, the spec in Python and Dafny, the
assumptions and the input coverage of the differential check. Anyone can run `dafny verify proofs/`.
On every pull request the action checks whether each certificate still matches its code and flags stale ones.

### Pull request comment

A summary on top, for example "12 proven, 1 counterexample, 3 open". Counterexamples always come first. Per function:

1. Result state
2. Intent in plain text with sources
3. For `counterexample`: the input as a ready-to-copy Python test
4. Assumptions: contracts, unproven callees, contradictions between sources
5. Collapsed: the Dafny spec and a link to the certificate

## Models

At runtime the pipeline calls **only Meta models**.

| Step | Model |
|---|---|
| Intent, spec, translation | Muse Spark |
| Proof, stage 1 | trained Muse Glimmer |
| Proof, stage 2 | Muse Spark Open (hosted) or Muse Spark |
| Proof, stage 3 | Muse Wonder |

All four models run on the same evaluation tasks. The comparison table decides the roles in stage 2.
Only Glimmer is trained. At 816B parameters, Spark Open is too large to train within our means.

## Training

Only the proof step is trained, because there Dafny gives an unambiguous yes or no.

1. **Warm start with SFT:** collect successful proofs and train Glimmer on them (expert iteration). GRPO only learns
   when some attempts succeed and others fail, so this step comes first. Glimmer SFT is free on Fireworks until 30 Nov.
2. **GRPO with LoRA:** Glimmer learns directly from Dafny's verdict, with a graded reward. LoRA instead of full
   fine-tuning: for reinforcement learning it reaches nearly the same quality at a fraction of the GPU cost.

**Release:** the trained LoRA adapter is published on Hugging Face, with its evaluation results and the license of
the base model.

**Training data:** Vericoding, DafnyBench and translated Python functions, the exact input the proof agent gets in the
product. Evaluation tasks never enter training.

**Compute:** no own money is spent. GPU credits the team receives (Modal, Baseten, Fireworks) are pooled for training.
Without such credits the project stops after the free SFT step, and GRPO is shipped ready to run (code, reward
function, config) without a result.

## Evaluation

| Dataset | Measures |
|---|---|
| Vericoding | the proof step alone (human-written specs) |
| HumanEvalFix | the whole pipeline (each function correct and buggy) |
| BugsInPy | real bugs in real projects with commit history, next after the two above |
| DafnyBench | the proof step, as an extension of Vericoding |
| MBPP-DFY | translation and proof of simple functions, later candidate |

Fixed test splits never enter training, for example about 40 of the 164 HumanEvalFix functions.

**Metrics:** proof rate, bugs found, false alarms (a `counterexample` on correct code), share per Python tier, cost and
time per function. False alarms must stay close to zero. Target values are set after the first run.

**Comparison:** Glimmer before training, after SFT and after GRPO, against Spark, Spark Open and Wonder on the same
tasks.

## Demo

A known real bug from BugsInPy: the pull request with the bug turns red with a counterexample, the pull request with
the fix turns green with a certificate that the viewer can re-check. If the project finds a real, previously unknown
bug or contradiction along the way, that becomes the highlight.

## Licensing

The code is licensed under AGPL-3.0. `THIRD_PARTY.md` lists every dataset and model with its license and link.
If Muse Spark Open is used, the README carries "Built with Muse Spark Open". Licenses of DafnyBench, MBPP-DFY and
BugsInPy are checked before any of their data enters the repository.

## Decided later

- Repository structure
- Target values for the metrics, after the first evaluation run
- Attempt counts for the proof stages, after the first evaluation run
- Training compute, once it is known which GPU credits the team receives
- Tasks as GitHub issues, split by dependencies so that agents can work in parallel

## Outlook

- More verifier backends behind `verify(code)`, for example Lean
