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
  ├─ 5. Prove               trained Glimmer → Spark Open or Spark → Wonder
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
- **Symbolic search:** `crosshair diffbehavior` runs the original and the back-compiled version with symbolic inputs on
  real CPython and searches for an input on which they differ. A first test with integers and lists of integers found
  every planted translation bug, including one that 10,000 random inputs missed, with no false alarms on correct
  translations. Strings, dicts and other types still have to be tested. "No differences found" means none were found,
  not that none exist.
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
| 2 | Spark Open or Spark, whichever measures better | up to 3, with all previous failures |
| 3 | Wonder | 1, only for functions of at most 40 source lines |

These numbers are starting values. They are tuned after the first evaluation run and are configurable in the action.
Stage 1 is skipped while no Glimmer endpoint is configured; stage 2 uses Spark Open only where a partner key is
available. Dafny always runs with a fixed resource limit and never with `--allow-warnings`.

**The proof may only add ghost code:** loop invariants, `decreases` clauses, `assert` statements, `calc` blocks, lemmas
and calls to those lemmas. The executable code and the method header stay as translated, except for added `decreases`
clauses, which recursion needs. The checker removes all ghost code and compares the rest with the translation; any
difference rejects the attempt, so the prover cannot prove a different function.

**Anti-cheat:** the checker rejects any proof that changes the frozen spec or uses `assume`, `expect`, `{:axiom}`,
`@Axiom`, body-less lemmas, `{:verify false}`, `{:only}`, `{:extern}`, `decreases *` or `{:termination false}`,
`@VerifyOnly`, or attributes that override the resource limit (`{:rlimit}`, `{:resource_limit}`, `{:timeLimit}`,
`{:timeLimitMultiplier}`). The full list, with a test file for each form, is in `pcc.verifier.anti_cheat`. Contracts
from step 4 are allowed only in a separate, generated assumptions module that is listed in the certificate. Dafny's exit
code alone is not enough: in Dafny 4.11 `{:axiom}` and `@Axiom` verify with exit code 0, so the checker scans the code
itself. Only exit code 0 together with a clean scan counts as verified.

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

## Models

At runtime the pipeline calls **only Meta models**.

| Step | Model | Where it runs |
|---|---|---|
| Intent, spec, AI fallback translation | Muse Spark (`muse-spark-1.3`) | Meta Model API |
| Proof, stage 1 | trained Muse Glimmer | own deployment, optional (see Training) |
| Proof, stage 2 | Muse Spark Open or Muse Spark | Spark Open through a partner (Baseten, Fireworks) if a key is available, else Meta Model API |
| Proof, stage 3 | Muse Wonder (`muse-wonder-1.0`) | Meta Model API |

Every endpoint is OpenAI-compatible and configured by a base URL and an API key from the environment: the Meta Model
API by `META_API_KEY`, a Glimmer deployment by `PCC_GLIMMER_BASE_URL` and `PCC_GLIMMER_API_KEY`, a Spark Open partner
by `PCC_SPARK_OPEN_BASE_URL` and `PCC_SPARK_OPEN_API_KEY`. A stage whose endpoint is not configured is skipped. Prices
per million input/output tokens: `muse-spark-1.3` $1.25/$4.25, `muse-spark-1.3-contributor` $0.10/$0.20,
`muse-wonder-1.0` $4/$20.

All four models run on the same evaluation tasks. The comparison table decides the roles in stage 2.
Only Glimmer is trained. At 816B parameters, Spark Open is too large to train within our means.

Our own evaluation runs on public data and may use the cheaper contributor mode (`muse-spark-1.3-contributor`), in
which Meta may train on the content. It runs only in a dedicated environment on each machine: the client refuses to
call Meta models unless `PCC_META_ALLOWED=1` is set there, and tests that call real models never run in CI.

## Training

Only the proof step is trained, because there Dafny gives an unambiguous yes or no.

1. **Warm start with SFT:** collect successful proofs and train Glimmer on them (expert iteration). GRPO only learns
   when some attempts succeed and others fail, so this step comes first. Glimmer SFT with LoRA is free on Fireworks
   until 30 Nov, so this step needs no credits.
2. **GRPO with LoRA:** Glimmer learns directly from Dafny's verdict, with a graded reward:

   | Outcome | Reward |
   |---|---|
   | verified and the anti-cheat scan is clean | 1 |
   | not verified | 0 |
   | anti-cheat violation (`assume`, changed spec, …) | −1 |

   Groups in which every attempt gets the same reward are dropped, because they teach nothing. Partial credit for partly
   verified proofs comes only if too few groups differ, because it is easy to game with trivial lemmas. LoRA instead of
   full fine-tuning: for reinforcement learning it matches full fine-tuning at a fraction of the GPU cost.

**Training data:** 500 to 2,000 verified proofs, collected during our own evaluation runs with Spark and Wonder on the
training splits of Vericoding and DafnyBench, plus transpiler output of MBPP and HumanEval training functions, the exact
input the proof agent gets in the product. Transpiler output uses the `py_*` blocks, which Vericoding and DafnyBench
never do, and the library's specs are part of the proof agent's prompt. Evaluation tasks never enter training, nor does
any dataset without a license that allows it (see Licensing). Collection uses Spark in contributor mode, which costs 8 %
of standard mode for input tokens and 5 % for output tokens (Wonder has no contributor mode). It stops at a spend cap of
$100; without contributor mode the target drops to about 500. Tasks that appear in more than one dataset are
deduplicated by their source task, so no test task reaches training through another dataset. DafnyBench is used for
training only, never evaluated. HumanEval training functions exclude the HumanEvalFix test tasks, which are the same 164
problems.

**Compute:** no own money is spent. The team has the Meta Model API credits ($150 per person), which pay for Spark and
Wonder only. Partner credits (Baseten, Modal, Fireworks) are applied for but not confirmed. Local machines have no
suitable GPU, so a 30B model would be too slow for evaluation. So:

- **Glimmer evaluation** (before and after training) runs on a GPU rented by the hour from free monthly credits (Modal)
  or partner credits, for a few hours at a time, never permanently.
- **SFT** runs on Fireworks for free.
- **GRPO** runs only if GPU credits arrive. Otherwise it is shipped ready to run: code, reward function and config,
  tested end to end on a small model, without a result for Glimmer.
- **Oct 28:** a person checks that Fireworks SFT is free for Glimmer and whether the trained adapter can be downloaded,
  and records the result here.

**Release:** the trained LoRA adapter is published on Hugging Face, with its evaluation results and the license of
the base model, if the training provider allows downloading it. Otherwise the training data, config and evaluation
results are published so the adapter can be reproduced.

## Evaluation

| Dataset | Measures |
|---|---|
| Vericoding (Dafny part) | the proof step alone (human-written specs) |
| HumanEvalFix | the whole pipeline (each function correct and buggy) |
| BugsInPy | real bugs in real projects with commit history, stretch goal |
| DafnyBench | training data only (see Training), not evaluated |
| MBPP-DFY | translation and proof of simple functions, later candidate, used locally only |

Fixed test splits never enter training: 40 of the 164 HumanEvalFix functions and 20 % of the Vericoding Dafny tasks,
drawn with a fixed seed. The repository stores only the task IDs of each split and a script that downloads the
datasets and checks their hashes, never the datasets themselves. Vericoding and HumanEvalFix come first; the others only
if time remains.

**Metrics:** proof rate, bugs found, false alarms (a `counterexample` on correct code), share per Python tier, share
translated by the transpiler, cost and time per function. False alarms must stay close to zero. Target values are set
after the first run.

**Comparison:** Glimmer before training, after SFT and after GRPO, against Spark, Spark Open and Wonder on the same
tasks.

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
functions are translated by the transpiler, not by AI. If the project finds a real, previously unknown bug or
contradiction along the way, that becomes the highlight.

## Submission

Due Wed Nov 4, 9:00 PM ET, through the Cerebral Valley submission page: the public repository, a working 3-minute demo
video, and every team member added. The video opens with one line on what was built during the event. It shows which
Meta model does what in the pipeline, the trained Glimmer and the model comparison, because "use of Meta AI tooling"
is one of four equally weighted criteria, next to impact, technical depth and demo quality.

## Licensing

The code is licensed under AGPL-3.0. `THIRD_PARTY.md` lists every dataset and model with its license and link.
If Muse Spark Open is used, the README carries "Built with Muse Spark Open". The hackathon disqualifies projects that
use code or data without the rights to it, so:

| Dataset | License | Use |
|---|---|---|
| HumanEvalFix | MIT | evaluation and training, with notice |
| HumanEval | MIT | training (transpiler input), with notice |
| MBPP | CC-BY-4.0 | training (transpiler input), with attribution |
| Vericoding | MIT, tasks from sources with their own licenses | evaluation and training, without the unlicensed VerifiedCogen part |
| DafnyBench | Apache-2.0 | training, with notice |
| BugsInPy | none | referenced by bug ID and commit only, fetched from the original projects at run time, never trained on |
| MBPP-DFY | GPL-3.0 | local evaluation only, never committed |

## Cut order

If the demo is at risk, these go first, in this order: GRPO, the search agent of step 1, calls inside the project, the
`find_pattern` highlight, crosshair (differential testing stays), the second opinion. The red pull request needs only
steps 1 to 3, so it is built first; the green one needs translation and proof.

## Decided later

- Target values for the metrics, after the first evaluation run
- Attempt counts and budgets, after the first evaluation run
- GRPO compute, once it is known which GPU credits the team receives

The repository structure is decided and described in [architecture.md](architecture.md). Tasks are GitHub issues,
split by dependencies so that agents can work in parallel (see [AGENTS.md](../AGENTS.md)).

## Outlook

- Grow the transpiler beyond the four tiers
- `float` support through Dafny's `fp64` type, once it ships in a release and compiles to Python. It exists on Dafny's
  development branch but not in release 4.11, is verification-only so far, and Dafny's `==` on it differs from
  Python's (`NaN`, signed zero), so `==` would map to `fp64.Equal`.
- Prove the transpiler itself. This first needs a formal semantics of the supported Python subset and of Dafny.
  Research semantics of older Python versions exist (λπ, K-Python) and could serve as a starting point.
- More verifier backends behind `verify(code)`, for example Lean
- Install the checked project's dependencies into the sandbox, so functions that use third-party packages can run
