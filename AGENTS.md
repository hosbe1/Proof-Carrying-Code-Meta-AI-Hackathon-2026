# Working in this repository

Read [docs/plan.md](docs/plan.md) for what we build and [docs/architecture.md](docs/architecture.md) for where it
lives. This file is the short list of rules.

## Commands

```bash
uv sync                      # install
uv run pytest                # tests (no network, no keys)
uv run ruff check . && uv run ruff format .
uv run mypy                  # strict
uv run lint-imports          # layer rules
dafny verify --resource-limit 1000000 --verification-time-limit 0 src/pcc/pylib/PyLib.dfy  # Config().dafny.resource_limit
```

All five checks must pass before a pull request; CI runs the same ones (the Dafny check runs inside `pytest`).

Proofs must pass with the Dafny 4.11.0 release binary and its Z3, as CI uses. Verification uses a fixed
`--resource-limit` and no time limit, so the result does not depend on the machine. Change the limit in
`src/pcc/config.py` only.

## How work flows

- Every change starts from a GitHub issue. Pick up only issues labelled `ready` whose "Blocked by" issues are all closed.
- One issue, one branch `agent/<issue number>-<slug>`, one pull request whose description says `Closes #<n>`.
- Merge (squash) only with green CI and a finished review. Never merge a red pull request.
- Issues labelled `run` are runs, not code: deployments, training runs and live evaluations. An agent picks them up like
  any `ready` issue, on a machine that holds the credentials the issue names (Meta models only in the Meta environment,
  see below). A run stays within the cost cap of its issue, stops every deployment it started, records run IDs and spend
  in the issue, and brings results into the repository through a pull request.
- Issues labelled `human` need a person (accounts, credits, demo video, submission, confirming a bug before it is
  reported upstream); assign them to yourself.
- Meta models are called only where `PCC_META_ALLOWED=1` is set: on team machines only in the dedicated Meta
  environment, and in the GitHub Action, which sets it. Tests that call them are marked `live` and never run in CI.

## Rules

1. **Stay in your module.** A ticket names the module it changes. Additive changes to `src/pcc/core` (new optional
   fields, new types) are allowed in a step ticket and must be named in the pull request. Changing or removing an
   existing core field needs a ticket of its own.
2. **Respect the layers.** Import only from lower layers, never from a sibling step (see architecture.md). If you
   need something from a sibling, it belongs in `core` or in the infrastructure layer: say so in the ticket.
3. **Outside systems through their door.** Models via `ModelClient` (passed in, never built inside a step), Dafny via
   `pcc.verifier`, untrusted Python via `pcc.sandbox`. The intent and spec steps never put the function body into a
   prompt; their tests check this (see architecture.md).
4. **No cheating proofs.** Never add `assume`, `{:axiom}` or `{:verify false}`, never weaken a frozen spec to make a
   proof pass. Library contracts are the only allowed assumptions, and each one is listed in the result. Only exit code
   0 counts as verified, so never pass `--allow-warnings`. The full list of forbidden forms is in the docstring of
   `pcc.verifier.anti_cheat`.
5. **PyLib blocks** carry an `ensures` spec of Python's behaviour, verify in Dafny, and get a test against CPython in
   `tests/pylib/`. A transpiler rule lands together with the blocks it needs.
6. **Tests first-class.** Every public function gets tests in the mirrored folder. Model calls use `FakeClient`;
   real calls are marked `@pytest.mark.live`.
7. **Types:** core types are frozen dataclasses; keep `mypy --strict` clean without `# type: ignore` unless the
   reason is in a comment.
8. **Public repository.** Everything is in English. No secrets, keys, personal data or code from other projects.
   Commit only what the ticket needs. On a merge conflict in `uv.lock`, take the version from `main` and run
   `uv lock`; never hand-edit `uv.lock`.
9. **Package layout.** Implementation goes into submodules of a step package (`pcc/spec/mutants.py`); `__init__.py`
   only re-exports. Move a scaffold stub into a submodule when its step is built.

## Definition of done

- The acceptance criteria of the ticket are met and covered by tests.
- All checks above pass locally.
- Docstrings of changed public functions match their behaviour; `docs/` updated if a decision changed.
- The pull request links its issue (`Closes #N`) and says what was not done.
