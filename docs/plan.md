# Plan

## Goal

Muse Spark and Muse Glimmer turn Python code into formally verified Dafny code, together with its proof. Glimmer is
trained with the proof checker as its reward.

The original Python code stays as it is. Every function gets one of two things:

- a **proof certificate**: a Dafny file with spec and proof that anyone can re-check with `dafny verify`, or
- a **counterexample**: a concrete input on which the code breaks its spec, delivered as a ready-to-run Python test.

The certificate travels with the code, which is what "proof-carrying code" means.

The project is the verification of Python code with Meta models. What matters most is that the verification approach is
right and that we make measurable progress with the Meta models: on the evaluation and by training Glimmer. The bug hunt
in open-source projects is an add-on. When time or money runs short, the priority is the verification pipeline first,
then progress with the Meta models (evaluation, model comparison, training), then the bug hunt.

## Result states

Every function ends in exactly one of four states:

| State | Meaning |
|---|---|
| `proven` | The code satisfies the spec for all inputs, possibly under listed assumptions. |
| `counterexample` | An executed input shows that the code breaks the spec. |
| `open` | No proof was found, and no counterexample either. |
| `not_checkable` | The intent, the spec or the translation did not pass its checks. The reason is given. |

Verification sits behind one interface, `verify(code)`, which says whether the code verified and lists Dafny's errors;
the pipeline turns that into a state. Dafny is the only backend in this project.

## Pipeline

```
Python function
  │
  ├─ 1. Collect context     fixed collector + agent that searches further when unclear
  ├─ 2. Write intent        plain text with cited sources, conflicts flagged
  ├─ 3. Write spec          executable Python predicate, checked with Hypothesis and mutants
  ├─ 4. Translate           rule-based transpiler, AI only for the rest, then checked
  ├─ 5. Prove               trained Glimmer → Spark → Wonder
  ├─ 6. Search counterexample   only after a failed proof
  ├─ 7. Second opinion      only when step 5 or 6 reports a problem
  └─ 8. Report              certificate in proofs/, comment on the pull request
```

**Which functions:** every function the project defines is found and ends in one of the four states: top-level
functions, methods, nested functions, generators, `async` functions and lambdas bound to a name. Bugs hide everywhere,
so each one gets a result, always with the reason:

- What the transpiler cannot translate goes to the AI path; it becomes `not_checkable` only after that path failed its
  checks too.
- These become `not_checkable` right away, because no checked translation is possible yet: generators, `async`
  functions, functions using `eval`, `exec` or threads (their translation cannot be compiled back to Python for the
  equivalence checks), instance and class methods ("methods are tier 3"), and functions without type hints ("no types",
  because no test inputs can be generated). Inferring missing types with pyright is a stretch goal.
- Nested functions that capture outer variables are checked only as part of the enclosing function, other nested
  functions also on their own; anonymous lambdas only as part of their function.
- A function's name in results and certificates is its Python `__qualname__` with every `<locals>.` removed
  (`outer.inner`); a lambda bound to a name takes that name.

**Budget:** every function has a limit on cost, time and model calls, starting at $1.00, 900 seconds and 60 calls, and
every run has a cost limit, starting at $20. Reaching a limit ends the function as `open` with the reason "budget".
Spend limits of the model API do not stop billing, so the pipeline enforces its own. Costs are computed from the token
counts the API returns and the prices in the Models section. The time limit is checked before every model call and
between steps, so Dafny, Hypothesis and crosshair count too. A function that ends `open` for budget does not turn the
check red.

### 1. Collect context

The intent of a function lives more in its surroundings than in its body. A fixed collector gathers a base package for
every function:

- **Required:** name, signature, type hints, docstring, comments, existing tests.
- **When available:** call sites, commit history of the function, pull request description, linked issues, README and
  project docs, conventions from similar functions.

Pull request descriptions and linked issues come from the GitHub API with the `GITHUB_TOKEN` of the workflow run.
Without a token, for example in evaluation runs, these sources are skipped.

An agent with tools (read files, search code, query git history) may search further when the base package leaves the
intent unclear.

### 2. Write the intent in plain text

The model writes what the function **should** do, in plain language, and cites a source for every claim, for example
"inclusive from 100 € (commit a3f9)". The function body is **not** shown in this step, so a bug in the code cannot leak
into the intent.

When sources contradict each other, a fixed ranking decides:
tests → issues and pull request → commits → docstring → comments → call sites → docs → conventions → signature → name.
Every contradiction is reported in the result. If the contradictions are so strong that no sensible spec can be
written, the function becomes `not_checkable` with the reason.

### 3. Write the spec

The spec is written as an executable Python predicate named `spec`, with an optional precondition `pre`, for example:

```python
def spec(xs, result):
    return result in xs and all(x <= result for x in xs)


def pre(xs):
    return len(xs) > 0
```

Like the intent, the spec is written without seeing the function body. It uses only constructs the transpiler supports
in spec mode, so that it can be translated by rules. Every clause of `pre` cites a claim of the intent, because a too
strong `pre` hides bugs: `pre(s): return len(s) <= 2` would let the buggy palindrome pass. The result reports which
share of the generated inputs `pre` accepts, and the pull request comment shows `pre` next to the spec with a warning
when that share is below 20 %.

Before anything is proven, the spec is tested:

- **Against the original code:** Hypothesis calls the real function on many random inputs and checks the predicate.
  If the original already breaks it and the spec passes the mutant check, the spec is frozen and the failing input is a
  counterexample: translation and proof are skipped, and step 7 runs. An exception on an input that `pre` allows is
  such a failure too. If `pre` holds for none of the generated inputs, the function is `not_checkable` ("precondition
  never holds"), because a proof under a false precondition proves nothing.
- **Against broken variants (mutants):** a good spec must reject them. Mutants come from fixed rules
  (`<` → `<=`, `+` → `-`, constants) and from the model ("insert bugs a developer would really make"). The model mutants
  are written in a call of their own, which sees the body; the spec is never written in that call. Variants that behave
  exactly like the original on many inputs are filtered out first. The spec must reject at least 90 % of the remaining
  mutants; otherwise the model rewrites it up to twice, and after that the function is `not_checkable`. The rewrite hint
  holds only inputs with the original's and the mutant's result, never mutant code, so the spec step stays blind to the
  body. If the original already breaks the spec, some mutants fix that bug by chance and rightly survive. A surviving
  mutant that satisfies the spec on every input where the original fails is therefore a possible fix: it does not count
  in the 90 %, and the pull request comment shows the first one as "Possible fix". A vacuous spec cannot hide behind
  this rule, because the original never breaks it.

A spec that passes these checks is frozen. Later steps may not change it.

### 4. Translate to Dafny

If the translation is wrong, the proof is about a function that does not exist. The translation is therefore the part
everyone has to trust, and it is kept as small and as well checked as possible.

**What the transpiler can do, the transpiler does. Only the rest goes to the model.** Both paths exist from the start,
so the pipeline always runs end to end. Every new transpiler rule takes work away from the model.

**Rule-based transpiler.** It parses with Python's `ast` module and targets one fixed Python version (3.12). Each rule
is short and documented. Constructs outside its supported subset are handed to the model, never guessed. Missing type
hints are inferred with pyright where possible.

It has two modes:

- **Code mode** produces executable Dafny. `all(x <= r for x in xs)` becomes a loop or a recursive function.
- **Spec mode** produces logical Dafny. The same expression becomes `forall i :: 0 <= i < |xs| ==> xs[i] <= r`, which
  the prover can use directly. Translating specs like code would make proofs much harder.

**Python library in Dafny.** Python's behavior is rebuilt in Dafny as a library of building blocks, for example
`py_floordiv`, `py_mod`, `py_list_append` or `py_str_split`. The transpiler maps Python almost one to one onto these
blocks. This matters because the two languages differ even on basics: `7 // -2` is `-4` in Python, while `7 / -2` is
`-3` in Dafny, and `7 % -2` is `-1` versus `1`.

What each block guarantees, stated precisely:

- **Proven in Dafny against its own spec.** This shows that the block and its spec agree, not that they match Python.
- **Tested against CPython** on a large number of inputs, including edge cases such as negative indices, slices past
  the end, `x or default` returning a value instead of a bool, `max` returning the first of equal elements, stable
  `sorted`, and `zip` truncating. This is the evidence that the block matches Python.

A new transpiler rule only lands together with the blocks it needs, with their proofs and tests.

**Python errors become preconditions.** `xs[i]` can raise `IndexError`, `a // 0` raises `ZeroDivisionError`, `max([])`
raises `ValueError`. Each block requires that its error case does not happen, so a proof also shows that the function
never crashes on inputs its spec allows. Functions that raise on purpose (`raise`) are handled in a later tier. The only
`requires` clause of a translated method is `pre`; a translation that adds its own would shrink the proven domain
without anyone noticing, so the checks reject it.

**Mutation and aliasing.** Dafny sequences are immutable values, Python lists are shared objects: after
`ys = xs; ys.append(1)`, `xs` has changed too. The transpiler therefore translates

- lists that are never changed as Dafny sequences,
- local lists that are changed but never shared as reassigned sequences (`xs := xs + [v]`),
- shared lists and changed parameters not at all: they go to the AI fallback until a later tier covers them.

**AI fallback.** The model translates whatever function the transpiler does not support yet. These translations can
still reach `proven`, but the pull request comment and the certificate say clearly: translated by AI, tested, not
guaranteed. The spec is never translated by AI: if the transpiler cannot translate it, the spec step rewrites it with
the unsupported construct as a hint (this counts as one of its two rewrites), and after that the function is
`not_checkable`.

**Every translation is checked**, whichever path produced it, and the spec translation too:

- **Differential testing:** the Dafny version is compiled back to Python (`dafny build --target:py`) and compared with
  the original on 10,000 random inputs plus edge cases proposed by the model (empty collections, negative numbers,
  boundaries). Only inputs on which `pre` holds count. Equal means: both return equal values, or both raise;
  back-compiled Dafny does not check preconditions at run time, so the exception types are not compared.
- **Symbolic search (stretch):** `crosshair diffbehavior` runs the original and the back-compiled version with symbolic
  inputs on real CPython and searches for an input on which they differ. A first test with integers and lists of
  integers found every planted translation bug, including one that 10,000 random inputs missed, with no false alarms on
  correct translations. Strings, dicts and other types still have to be tested. "No differences found" means none were
  found, not that none exist.
- **Spec translation:** the transpiled `spec` and `pre` are compiled back to Python as well and evaluated next to the
  Python ones on the inputs from step 3, with the results of the original and of every mutant. They must agree on every
  input; otherwise the function is `not_checkable` ("spec translation differs"). Without this check a wrong spec
  translation, for example one that always holds, would be proven while the code is wrong.

Back-compiled Dafny uses its own runtime types, for example `_dafny.Seq` instead of `list`. A small converter turns
Python values into Dafny values and back for both checks. It is part of what has to be trusted, so it gets its own
tests.

A mismatch on the transpiler path is a transpiler bug. It is fixed in the transpiler, and the function falls back to
the AI path in the meantime. On the AI path the model retranslates with the counterexample as a hint, up to three
times. After that the function is `not_checkable`.

The certificate states openly what the proof rests on: the translation path, the checks it passed, and that
equivalence was checked, not proven for all inputs.

**Calls inside the project** are handled bottom-up, as a stretch goal after the demo. Until then a function that calls
another function of the project is `not_checkable` ("calls into the project"); a function that calls only itself is
checked normally, since Dafny proves its termination within one method. With the stretch goal, callees are proven first
and callers use only the callee's spec, as Dafny does anyway; independent functions are proven in parallel. An unproven
callee's spec becomes a listed assumption. A callee with a counterexample makes its callers `open` ("callee <name> has a
counterexample"), because an assumption known to be false proves nothing. Functions that call each other in a cycle are
`open` ("mutual recursion") until they are proven together in one Dafny module with a termination argument: assuming
each other's spec would prove anything.

**Parts that cannot be proven** (libraries, files, network) are modeled as contracts. Library type stubs give the base,
the model adds the behavior. Every contract is listed as an assumption in the certificate, so a result reads
"proven under 2 assumptions" with the list.

### 5. Prove

The proof agent adds loop invariants, lemmas and hints until `dafny verify` passes. After each failed attempt it gets
Dafny's error message back.

| Stage | Model | Attempts |
|---|---|---|
| 1 | trained Glimmer | up to 8 |
| 2 | Spark | up to 3, with all previous failures |
| 3 | Wonder | 1, only for functions of at most 40 source lines |

These numbers are starting values. They are tuned after the first evaluation run and are configurable in the action.
Stage 1 is skipped while no Glimmer endpoint is configured. Dafny always runs with a fixed resource limit and never
with `--allow-warnings`.

**The proof may only add ghost code:** loop invariants, `decreases` clauses, `assert` statements, `calc` blocks, lemmas
and calls to those lemmas. The executable code and the method header stay as translated, except for added `decreases`
clauses, which recursion needs. The checker removes all ghost code and compares the rest with the translation; any
difference rejects the attempt, so the prover cannot prove a different function.

**Anti-cheat:** the checker rejects any proof that changes the frozen spec or uses `assume`, `expect`, `{:axiom}`,
`@Axiom`, body-less lemmas, `{:verify false}`, `{:only}`, `{:extern}`, `decreases *` or `{:termination false}`,
`@VerifyOnly`, or attributes that override the resource limit (`{:rlimit}`, `{:resource_limit}`, `{:timeLimit}`,
`{:timeLimitMultiplier}`, and their `@` forms). Attributes are checked against an allowlist (`trigger`, `induction`,
`fuel`/`@Fuel`, `opaque`, `split_here`, `isolate_assertions`/`@IsolateAssertions`), with or without arguments; any other
attribute is rejected, because some, like `{:selective_checking}`, make Dafny accept `ensures false` with exit code 0.
The full list, with a test file for each form, is in `pcc.verifier.anti_cheat`. Contracts from step 4 are allowed only
in a separate, generated assumptions module that is listed in the certificate. Dafny's exit code alone is not enough: in
Dafny 4.11 `{:axiom}` and `@Axiom` verify with exit code 0, so the checker scans the code itself. Only exit code 0
together with a clean scan counts as verified.

### 6. Search for a counterexample

Only after the proof failed:

1. Hypothesis, focused on the `ensures` clause Dafny could not prove.
2. Suspicious inputs proposed by the model, each executed for real.

If the real Python code breaks the spec, the state is `counterexample`, with the input as a ready-to-run test.
Otherwise the state is `open`. A `counterexample` is always backed by an executed input, never by a failed proof alone.

A counterexample can also mean the spec is wrong rather than the code. That is why the result shows the plain-text
intent with its sources right next to the counterexample.

### 7. Second opinion

Only when there is something to doubt (a counterexample from step 3 or 6, a function that ends `open`, or sources that
contradict each other), the model writes a second spec **with** the function body in view and compares it with the blind
one. It is a hint, not a verdict: a model that reads the code can copy its bug. Whether this lowers false alarms is
measured on HumanEvalFix.

## Supported Python

The goal is a result for every function, and as many proofs as possible. The transpiler grows tier by tier; whatever
it does not cover yet goes to the AI fallback:

1. Pure functions with type hints: integers, booleans, strings, lists, tuples, loops, recursion, `all`, `any`,
   comprehensions. This also covers almost all spec predicates. Not in this tier:
   - `float` and true division `/`, which always returns a `float` in Python, even for `4 / 2`
   - arithmetic on booleans, since `True + 1 == 2` in Python
   - string methods that depend on Unicode tables, such as `upper()`, because `"ß".upper()` is `"SS"`. At first only
     concatenation, comparison, `len`, indexing, slicing (`s[1:-1]`) and reversing (`s[::-1]`), with negative indices
     and slices past the end as in Python. Lone surrogate characters are valid in Python strings but not in Dafny, so
     the string blocks must handle or reject them.
2. Dicts, sets, functions that raise on purpose (`raise`), as a result type
3. Classes and methods that only change their own state
4. Functions using libraries, files or network, through contracts

Everything else, such as `eval`, metaprogramming or threads, stays with the AI fallback or becomes `not_checkable` with
the reason. Rules and PyLib blocks needed by the demo functions come first.

Reported metrics: the share of functions per tier, and the share translated by the transpiler instead of the model.

## Product: GitHub Action

The pipeline runs as a GitHub Action in the user's own CI. Nothing runs on our servers, so nothing of ours can fail
during the demo or the judging.

- **Form:** a composite action that installs uv, Dafny and the `pcc` package and runs `pcc-check run`.
- **Scope:** on each pull request, the changed functions (diff hunks mapped to functions) plus every caller whose proof
  may be affected, found through the call graph. A manually triggered run covers the whole repository once, in
  batches.
- **Exit code:** `pcc-check` exits 1 when a function ends as `counterexample` or a certificate is stale and its function
  is no longer proven or no longer exists, 2 when a tool breaks, else 0, so the check turns red only on a real finding;
  `open`, including `open` for budget, stays green. Pull requests from forks get no secrets: the action skips them with
  a notice. The comment needs the `pull-requests: write` permission.
- **API key:** each user stores their own Meta Model API key as a GitHub secret. No key is built into the action. The
  action uses the standard mode of the Meta Model API, in which content is not used to train Meta models; contributor
  mode would hand the user's code to Meta. The action sets `PCC_META_ALLOWED=1`, because the user's CI is the
  environment the user chose for Meta models.
- **Dependencies:** the action runs the checked code with the Python of `pcc` itself. The project's third-party
  packages are not installed there, so a function that imports one is `not_checkable` with the reason. Projects with
  only the standard library, such as the demo, are not affected.
- **Glimmer stage:** optional. If no Glimmer endpoint is configured or reachable, the action proves with Spark through
  the Meta Model API. Glimmer is not offered on the Meta Model API, and no provider serves a trained LoRA adapter for
  free, so the default action runs without it.

### Certificate

The action writes certificates to `proofs/` and uploads that folder as a workflow artifact, linked in the comment. The
developer downloads it and commits it; proofs come from models, so a second run would not produce the same files.

In the repository, certificates live under `proofs/`: per function `proofs/<path>/<qualname>.dfy` with a `.json` next to
it holding the spec in Python and Dafny, the assumptions, the translation path of function and spec (transpiler or AI)
and the checks the translation passed. `proofs/` also holds a copy of `PyLib.dfy`, whose hash every certificate records,
and a `dfyconfig.toml` with the resource limit, so anyone can re-check everything without installing `pcc`:
`dafny verify proofs/dfyconfig.toml` re-runs the proofs, and `dafny audit` lists every assumption, because `{:axiom}`
passes `dafny verify` silently (whether `dafny audit` covers all forbidden forms in 4.11 is checked when certificates
are built). The generated assumptions modules are trusted and listed.

Every Dafny module name is derived from the file path and qualname, so all certificates verify together as one project.
On every pull request the action checks whether each certificate still matches its code and flags stale ones. A
certificate is stale when the function changed, or a callee's spec, a contract or PyLib changed; the certificate records
a hash of each of them. If the run proves the function again, the new certificate replaces the stale one; a stale
certificate is a finding only when the function is no longer proven or no longer exists.

When the hash of PyLib changes, every certificate is checked, not only those of the changed functions. Every certificate
also lists two standing assumptions: inputs match the type hints, and strings contain no lone surrogates.

### Pull request comment

One comment per pull request, updated on every run. A summary on top, for example "12 proven, 1 counterexample,
3 open". Counterexamples always come first. Per function:

1. Result state, whether the translation came from the transpiler or from AI, and `pre` with the share of inputs it
   accepts
2. Intent in plain text with sources
3. For `counterexample`: the input as a ready-to-copy Python test
4. Assumptions: contracts, unproven callees, contradictions between sources
5. Collapsed: the Dafny spec, and a link to the artifact with the certificates

### Local results viewer

`pcc-check view <path> [--port 8765]` is the way to browse the pipeline results (bug hunt and HumanEvalFix evaluation)
locally. The path is a `scan.jsonl` file or a run folder in `docs/results/`. It starts a server on `127.0.0.1` only and
opens the browser. One page shows a table of all functions (project, path, name, state,
reason, cost) with filters by state and project and a search box. A detail view per function shows the intent, the spec,
the Dafny code, the proof result, a counterexample as pytest, the second opinion and the cost. The viewer only reads
files and starts no runs. There is no hosted service: it would show results from running untrusted code, it would need
keys, and it would cost money to run.

## Models

At runtime the pipeline calls **only Meta models**.

| Step | Model | Where it runs |
|---|---|---|
| Intent, spec, AI fallback translation | Muse Spark (`muse-spark-1.3`) | Meta Model API |
| Proof, stage 1 | trained Muse Glimmer | own deployment on Modal (vLLM, SGLang as fallback, scale to zero); fallback a Fireworks on-demand deployment |
| Proof, stage 2 | Muse Spark (`muse-spark-1.3`) | Meta Model API |
| Proof, stage 3 | Muse Wonder (`muse-wonder-1.0`) | Meta Model API |

Every endpoint is OpenAI-compatible and configured by a base URL and an API key from the environment: the Meta Model
API by `META_API_KEY`, a Glimmer deployment by `PCC_GLIMMER_BASE_URL` and `PCC_GLIMMER_API_KEY`. A stage whose
endpoint is not configured is skipped. Prices per million input/output tokens: `muse-spark-1.3` $1.25/$4.25,
`muse-spark-1.3-contributor` $0.10/$0.20, `muse-wonder-1.0` $4/$20.

An optional `PCC_META_REASONING_EFFORT` (`minimal`, `low`, `medium`, `high` or `xhigh`) is sent only to the
Meta Model API; when unset the request field is omitted. A response with null content and `finish_reason`
`length` returns an empty answer with the reported usage, so the budget wrapper counts the billed attempt
instead of failing the run. Evaluation runs keep the fixed 4096-token limit.

Endpoints billed per GPU hour (a Glimmer deployment) count $0 per token in the function budget; their GPU time is
tracked per deployment and capped in Compute.

The model comparison covers Glimmer (base and trained), Spark and Wonder on the same evaluation tasks. The spec writer
is Spark by default (`ModelsConfig.spec`). The second opinion is Spark too (`ModelsConfig.second_opinion`); it differs
from the spec writer in what it sees, the function body, not in the model. The
comparison may change the spec writer: the better model becomes the default.

**Why Glimmer is trained, not Spark:** Spark is closed; Spark Open (816B) is too large to train or serve within our
means. Glimmer is Apache 2.0, a dense 30B model that fits one GPU, so the trained adapter can be served for evaluation,
the demo and the product, and published.

Our evaluation and training data are public, so Spark runs in contributor mode (`muse-spark-1.3-contributor`) for the
evaluation, the training data and the bug hunt, where Meta may train on the content. Wonder has no contributor mode and
is used sparingly. Contributor mode runs only in the Meta environment, where `PCC_META_ALLOWED=1` is set: the client
refuses to call Meta models without it, and tests that call real models never run in CI.

## Training

Only the proof step is trained, because there Dafny gives an unambiguous yes or no. The main method is rejection
sampling (expert iteration): the model proposes proofs, Dafny keeps the verified ones, and Glimmer is trained on them.
The training data are proofs: verified answers to Vericoding tasks, and proof-only tasks built from them, which match
the step the product runs. An answer may not change the method header or the spec. The check compares the header as a
whole, so an added `requires`, `modifies`, `reads` or `decreases` counts as a change. Two prompt formats exist, and
both stay as they are in the training data: the Vericoding format (spec given, the model writes code and proof) and the
pipeline format (the proof step's prompt in the product).

The collection is the set of verified proofs that the pipeline finds for HumanEval training functions, run with Spark in
contributor mode.

1. **Round 0:** Spark in contributor mode (`muse-spark-1.3-contributor`, Meta Model API, run in the Meta environment)
   makes 8 attempts per Vericoding training task. Dafny checks them on a team machine (it is not a model), and the
   verified proofs are kept.
2. **SFT 1:** LoRA SFT (rank 16) of base Glimmer on the round 0 proofs and the proofs collected so far. Glimmer SFT with
   LoRA is free on Fireworks until Nov 30, through the UI or the Serverless Training API. If dataset uploads stay
   disabled for our account (reported by other participants), SFT uses the Serverless Training API, which is also
   free. The adapter is downloaded in PEFT format (`firectl download model`), which vLLM loads with `--lora-modules`.
3. **Round 1:** the SFT 1 model, served with vLLM on Modal, samples again on the tasks still unsolved, and the verified
   proofs are added.
4. **SFT 2:** LoRA SFT (rank 16) of base Glimmer, not of SFT 1, on all verified proofs of rounds 0 and 1 and of the
   collection, plus the proof-only tasks where they are ready.
5. **GRPO (stretch only):** starts from SFT 2 on Fireworks with the tasks that are left, capped at $50, after a bounded
   pre-pass (see below).

**Proof-only tasks:** the verified answers are turned into tasks for the proof step alone. The executable code stays as
it is, the ghost code is removed, and the model is asked to add the annotations. That is the product's step 5. These
tasks go into SFT 2 and form the step-5 evaluation set (see Evaluation).

**Why only the proof step is trained, and not the spec:** technical depth: Dafny's verdict is a yes or no reward. Demo:
the proof rate of the same Glimmer model before and after training is the measurement. Impact: an open 30B model proves
in our own infrastructure. Meta tooling: each Muse model has one clear role. The spec is not trained, because it has no
unambiguous reward, so HumanEvalFix is its evaluation set. Trained Glimmer is proof stage 1 everywhere: in the
evaluation, the demo and the bug hunt.

**Why rejection sampling:** training is free, so the money buys data. The cost and speed of round 1 on Modal are
measured in its first hour. At the $50 GRPO cap only about 9 to 12 steps fit, which would not give a measurable gain, so
GRPO is a stretch goal.

**GRPO (stretch):** Glimmer learns directly from Dafny's verdict, with a graded reward:

| Outcome | Reward |
|---|---|
| verified and the anti-cheat scan is clean | 1 |
| not verified | 0 |
| anti-cheat violation, changed spec or changed method header (`assume`, `requires false`, …) | −1 |

Groups in which every attempt gets the same reward are dropped, because they teach nothing. Partial credit for partly
verified proofs comes only if too few groups differ, because it is easy to game with trivial lemmas. LoRA instead of
full fine-tuning: for reinforcement learning it matches full fine-tuning at a fraction of the GPU cost.

GRPO runs through the Fireworks Serverless Training API with a custom loop: Fireworks samples and trains, our code
computes the reward with Dafny. Glimmer rates per 1M tokens: prefill $1.96, cached prefill $0.39, sample $4.88, train
$5.86; spend cap $50.

GRPO is a small proof of concept. The pre-pass samples 64 Vericoding training tasks (fixed seed) 8 times each with SFT
2; its cost counts toward the $50. Assuming about 2k prompt and 2k sample tokens per attempt, a step of 16 tasks × 8
samples costs about $4.75 (the pre-pass about $7.00, prefill and sample only). After the pre-pass, $50 buys about 9
such steps; with 1.5k prompt and sample tokens per attempt, about 12. The pre-pass keeps the tasks that SFT 2 solves in
1 to 7 of 8 attempts, so the groups in a step can differ in reward.

**Training data:** 500 to 2,000 verified proofs from rounds 0 and 1 on the Vericoding training tasks, the collection,
and the proof-only tasks. The collection is the exact input the proof agent gets in the product, because pipeline output
uses the `py_*` blocks, which Vericoding never does, and the library's specs are part of the proof agent's prompt.
Evaluation tasks never enter training, nor does any dataset without a license that allows it (see Licensing). MBPP is
used only after the submission. Round 0 and the collection use Spark in contributor mode, which costs 8 % of standard
mode for input tokens and 5 % for output tokens (Wonder has no contributor mode). Together they stop at a spend cap of
$100; without contributor mode the target drops to about 500. Tasks that appear in more than one dataset are
deduplicated by their source task, so no test task reaches training through another dataset. DafnyBench is not used
before the submission, since no issue collects it. HumanEval training functions exclude the HumanEvalFix test tasks,
which are the same 164 problems.

**Compute:** No own money is spent. Local machines have no suitable GPU.

| Credits | Amount | Used for | Cap |
|---|---|---|---|
| Meta Model API | $150 per person, three people ($450) | Bug hunt (full pipeline, Spark in contributor mode, Wonder in proof stage 3), round 0 and training data collection (Spark, contributor mode), model comparison (Wonder), pipeline development, evaluation and demo, reserve | bug hunt $100, training data $100 (round 0 and collection), development, evaluation, comparison and demo $200, reserve $50 |
| Modal | $100, plus $30 free per month as reserve | Round 1 sampling with vLLM on an H100, evaluation of base, SFT 1 and SFT 2, demo serving | round 1 sampling $40, evaluation and demo serving $60 |
| Fireworks | $100 | SFT 1 and SFT 2 (free until Nov 30, if #3 confirms it), GRPO stretch; reserve for an on-demand deployment if a trained adapter cannot be served on Modal, or for sampling if Modal fails | GRPO $50, reserve $50 |

Modal's caps add up to the $100 partner credits. The free $30 per month, confirmed for November too, is the reserve. The
three Meta keys sit in one Meta environment on one team member's machine.

Hard limits per run, set with `--max-cost-usd` (Meta) or `--max-seconds` (Modal GPU time):

- Round 0 sampling $40, the collection $60. Together they are the training data bucket of $100.
- Bug hunt $100 in total, shared out over the chosen projects.
- The comparison, development and demo bucket of $200: Wonder spec comparison on the 80 HumanEvalFix test functions $30;
  Wonder proof comparison, one attempt on a fixed 150-task subset of the Vericoding test split $15; eight evaluation
  runs in contributor mode at $10 each $80; development $40; demo $10. $25 stays unassigned as margin.
- Reserve $50 on Meta.
- Modal: round 1 sampling $40, about 10 H100 hours; evaluation and demo serving $60, about 15 H100 hours.
- Fireworks: GRPO stretch $50, the `max_spend_usd` of the training config.

- Glimmer is served only while a job runs and scales to zero; nothing runs permanently.
- SFT is free on Fireworks until Nov 30, if #3 confirms it, through the UI or the Serverless Training API.
- Every job records its spend in its issue, and Meta spend is recorded per key.
- The open points of the Fireworks, Modal and Meta access are checked in #3, and the answers are recorded
  in the issue tracker, not in this plan.

**Release:** the trained LoRA adapter is published on Hugging Face, with its evaluation results and the license of
the base model, if the training provider allows downloading it. Otherwise the training data, config and evaluation
results are published so the adapter can be reproduced.

## Evaluation

| Dataset | Measures |
|---|---|
| Vericoding (Dafny part) | code and proof from a given spec (human-written specs); the proof step alone on proof-only tasks |
| HumanEvalFix | the whole pipeline (each function correct and buggy), and the spec writer alone (steps 1 to 3) |
| BugsInPy | real bugs in real projects with commit history, stretch goal |
| DafnyBench | not used before the submission, not evaluated |
| MBPP-DFY | translation and proof of simple functions, later candidate, used locally only |

Fixed test splits never enter training: 40 of the 164 HumanEvalFix functions and 20 % of the Vericoding Dafny tasks,
drawn with a fixed seed. The repository stores only the task IDs of each split and a script that downloads the
datasets and checks their hashes, never the datasets themselves. Vericoding and HumanEvalFix come first; the others only
if time remains.

**Protocol:** every run uses the same settings: temperature 0.8 and at most 4096 output tokens. Attempts are independent
samples. pass@1 is the mean over the attempts. pass@k, with k the number of attempts, is the share of tasks with at
least one verified attempt, with a 95 % Wilson interval. Comparisons are paired: all models get the same tasks, and the
summary lists the tasks that one model solves and another does not. Every run has a hard cap on cost and on time. It
records the git SHA, the UTC date, the endpoint, the model name it sent, the hash of the split file, and whether it ran
in contributor mode. Every results table states that mode. A run is published as a folder
`docs/results/<date>-<commit>-<name>/` with `summary.md`, `summary.json` and `meta.json`. `docs/results/history.md` has
one row per run: date, commit, model, dataset, n, rate and cost.

**Metrics:** proof rate, bugs found, false alarms (a `counterexample` on correct code), false proofs (a `proven` on
buggy code, listed by function), share per Python tier, share translated by the transpiler, the share that each proof
stage contributes (proven by Glimmer, then Spark, then Wonder), second-opinion agreement on counterexamples, the share
of functions whose `pre` accepts less than 20 % of inputs, and cost and time per function. False alarms and false proofs
must stay close to zero. Target values are set after the first run.

**Comparison:** base Glimmer, SFT 1, SFT 2 and, if it ran, GRPO on the same tasks, against Spark and Wonder:

- Vericoding test split: base Glimmer first, as the baseline, then SFT 1 and SFT 2, each with four
  attempts. Spark with four attempts. Wonder with one attempt on a fixed 150-task subset, compared by pass@1.
- Spec writer: Spark, Wonder and base Glimmer on the 40 HumanEvalFix test functions, each in its fixed and its buggy
  version (80 functions). They are compared on correct passes, buggy functions caught and false alarms. The better model
  becomes the default spec writer.
- Glimmer on the fixed HumanEvalFix versions: base against SFT 2, the whole pipeline with proof stage 1 only.
- Proof-only tasks: the step-5 evaluation set, on the test split, built from the verified Spark answers of that split.
  The Vericoding harness runs it.

All runs with Spark use contributor mode, because the evaluation data is public.

## Demo

Real, historical bugs from [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) (MIT), each fixed in a
merged pull request with an issue that states the intended behavior. No bug in BugsInPy fits the first tier of
supported Python.

- **Guaranteed demo:** `is_palindrome_recursive` in `strings/palindrome.py`
  ([issue #12941](https://github.com/TheAlgorithms/Python/issues/12941),
  [fix #12946](https://github.com/TheAlgorithms/Python/pull/12946)): `"abcdba"` was reported as a palindrome. Its
  proof is simple, so the green pull request is certain. It needs `len`, string indexing, slicing and recursion.
- **Highlight, if its proof succeeds in time:** `find_pattern` in `strings/z_function.py`
  ([issue #15303](https://github.com/TheAlgorithms/Python/issues/15303),
  [fix #15302](https://github.com/TheAlgorithms/Python/pull/15302)): the Z-algorithm count included matches inside the
  pattern itself, so `find_pattern("aa", "aaa")` returned 3 instead of 2.

In the demo repository the intent comes from the name, the docstring and its doctests. The original issue is recreated
there too, for when the GitHub sources of step 1 are built. The repository carries TheAlgorithms' MIT notice.

The demo lives in its own public repository that uses our action: the pull request with the bug turns red with a
counterexample, the pull request with the fix turns green with a certificate that the viewer can re-check. Both demo
functions are translated by the transpiler, not by AI. The palindrome pull requests are the demo. A confirmed bug from
the bug hunt is a bonus for the video, not the highlight.

## Bug hunt

The bug hunt runs the full pipeline (steps 1 to 8, with the Dafny translation and proof) on every checkable function of
the chosen projects. It shows the verification process on public open-source projects, with its numbers. A confirmed
real bug is a bonus for the video.

**Calibrate:** before unknown projects, the scan runs on TheAlgorithms/Python, limited to `strings/`, at the commits
before historical single-function fixes, starting with the palindrome fix. It measures whether the scan finds them and
how many false alarms it gives. `find_pattern` counts only if support for calls into the project is merged, since
otherwise it ends `not_checkable`.

1. **Choose projects:** 5 to 10 public Python projects with many pure functions, type hints and docstrings or doctests,
   mostly on `int`, `str` and lists (algorithm collections, parsing and utility libraries). `pcc-check survey` counts
   how many functions of a repository are checkable before any model is called. The full pipeline costs more per
   function than steps 1 to 3, so the number of projects is chosen to stay within the cap in step 2.
2. **Scan:** `pcc-check scan <repo>` runs the full pipeline on every checkable function: `precheck`, then steps 1 to 8.
   Each function ends `proven` (with its certificate, written to `--proofs-dir`), `counterexample`, `open` or
   `not_checkable`, always with the reason. Spark runs in contributor mode (`--contributor`), since the code is public.
   Wonder runs in proof stage 3. Glimmer is proof stage 1 whenever its endpoint is up. Each run
   has a hard cap on cost (`--max-cost-usd`); the bug hunt is capped at $100 of Meta credits in total.
3. **Triage:** automatic filters come first; the second opinion (Spark) then runs on what is left. The spec must accept
   every doctest example of the function, otherwise the counterexample is set to `open` ("spec contradicts docstring").
   A counterexample that only raises an exception is listed apart from wrong results, and wrong results are triaged
   first, since libraries often raise on purpose. Then a person checks it against the documentation and, where unclear,
   the project's tests and issues. A blind spec can misread a vague docstring, so nothing is reported without this
   check.
4. **Report:** a team member opens an issue in the project with the generated pytest and, if one exists, the possible
   fix, following the project's contribution guidelines. Only bugs a person has confirmed are reported.
5. **Results:** `docs/results/bug-hunt.md` lists projects, functions checked, the states (`proven` with certificates,
   `counterexample`, `open`, `not_checkable`), confirmed bugs and links to the reported issues. A confirmed bug is a
   bonus for the video; the palindrome pull requests remain the demo of red and green.

Scanning sends public code only.

## Submission

Due Wed Nov 4, 9:00 PM ET, through the Cerebral Valley submission page: the public repository, a working 3-minute demo
video, and every team member added. The demo in the video is the palindrome pull requests, one red and one green. The
video opens with one line on what was built during the event. It shows which Meta model does what in the pipeline, the
trained Glimmer and the model comparison, because "use of Meta AI tooling" is one of four equally weighted criteria,
next to impact, technical depth and demo quality. It has one line on how false proofs are prevented: the anti-cheat
scan, the ghost-only check, the header check and the spec-translation check. The bug hunt results
(`docs/results/bug-hunt.md`) go with the repository if they exist; a confirmed bug from them is a bonus for the video.

The README has a results section with one chart: base Glimmer, SFT 1 and SFT 2 against Spark and Wonder, with intervals
and the cost per solved task. Before the submission, every certificate is re-checked with `dafny verify` and
`dafny audit`, and a person reviews every proof of buggy code.

## Licensing

The code is licensed under AGPL-3.0. `THIRD_PARTY.md` lists every dataset and model with its license and link. The
hackathon disqualifies projects that use code or data without the rights to it, so:

| Dataset | License | Use |
|---|---|---|
| HumanEvalFix | MIT | evaluation and training, with notice |
| HumanEval | MIT | training (collection input), with notice |
| MBPP | CC-BY-4.0 | training, only after the submission, with attribution |
| Vericoding | MIT, tasks from sources with their own licenses | evaluation and training, without the unlicensed VerifiedCogen part |
| DafnyBench | Apache-2.0 | not used before the submission; with notice if used later |
| BugsInPy | none | referenced by bug ID and commit only, fetched from the original projects at run time, never trained on |
| MBPP-DFY | GPL-3.0 | local evaluation only, never committed |

## Cut order

If the demo is at risk, these go first, in this order: GRPO, the bug hunt, round 1 of rejection sampling, the search
agent of step 1, calls inside the project, the `find_pattern` highlight, crosshair (differential testing stays), the
second opinion. The red pull request needs steps 1 to 3 and, for its counterexample, the second opinion, so it is built
first; the green one needs translation and proof.

## Order of work

There is no day-by-day schedule. The only fixed date is the submission deadline (see Submission). Tickets are picked up
in the order of the critical paths, as soon as their blockers are closed:

1. Config and interfaces, which block almost every ticket.
2. The training path: the verifier, sandbox and model parts it needs; the Vericoding harness with the header check; the
   reward; the rejection sampling tool and the SFT export; the evaluation report. Then the Glimmer deployment with its
   smoke test and throughput, the baseline of base Glimmer on the Vericoding test split, and round 0.
3. Pipeline wave 1 (intent, spec, transpiler, equivalence and certificate parts), in parallel with the training path
   whenever a runner is free.
4. Pipeline wave 2 (proof, mutants, counterexample), then the offline end-to-end test on the palindrome, CLI and action,
   the demo repository and the live red and green pull requests, and the pipeline baseline on the HumanEvalFix test
   split.
5. SFT 1, round 1, SFT 2 and the evaluations after training, the spec writer comparison and the proof-only tasks,
   then the comparison table.
6. The bug hunt, and in parallel the video, which shows the bug hunt results so far.
7. The submission.

About 45 agent tickets lie on the main path. They run on two or three runners in parallel worktrees, in the order of
the critical paths, the training path first. Large tickets on the green-pull-request path are split into chained parts
where the parts are separable. The offline end-to-end test has the most blockers of any ticket, so a person or a strong
model takes it.

## Decided later

- Target values for the metrics, after the first evaluation run
- Attempt counts and budgets, after the first evaluation run

The repository structure is decided and described in [architecture.md](architecture.md). Tasks are GitHub issues.
The project is developed with agents that run in parallel, so the work is split for that: a large task is split into
smaller ones when they succeed more reliably, the number of tasks is not reduced for convenience, and every two tasks
that touch the same file are ordered by a dependency, so parallel runners never collide (see [AGENTS.md](../AGENTS.md)).

## Outlook

- Grow the transpiler beyond the four tiers
- `float` support through Dafny's `fp64` type, once it ships in a release and compiles to Python. It exists on Dafny's
  development branch but not in release 4.11, is verification-only so far, and Dafny's `==` on it differs from
  Python's (`NaN`, signed zero), so `==` would map to `fp64.Equal`.
- Prove the transpiler itself. This first needs a formal semantics of the supported Python subset and of Dafny.
  Research semantics of older Python versions exist (λπ, K-Python) and could serve as a starting point.
- More verifier backends behind `verify(code)`, for example Lean
- Install the checked project's dependencies into the sandbox, so functions that use third-party packages can run
