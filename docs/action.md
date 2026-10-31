# GitHub Action

`pcc-check` runs the proof-carrying-code pipeline on a pull request as a composite action defined in
[`action.yml`](../action.yml). On each run it checks the changed functions plus affected callers, uploads
the `proofs/` certificates as a workflow artifact, and updates one sticky comment on the pull request.
The check turns red only on a real finding: a `counterexample`, or a stale certificate whose function is
no longer proven or no longer exists. `open`, including `open` for budget, stays green.

## Example workflow

```yaml
name: proof-carrying-code

on:
  pull_request:

permissions:
  pull-requests: write
  contents: read

jobs:
  verify:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
      - uses: hosbe1/Proof-Carrying-Code-Meta-AI-Hackathon-2026@main
        with:
          meta-api-key: ${{ secrets.META_API_KEY }}
```

`fetch-depth: 0` is required so `pcc-check run` can diff against the base branch. The
`pull-requests: write` permission lets the action update its comment; `contents: read` checks out the
code. Store a Meta Model API key as an Actions secret named `META_API_KEY`; the action sets
`PCC_META_ALLOWED=1` for its own run and never bakes a key in. Pull requests from forks are skipped
with a notice, because they have no secrets.

## Inputs

| Input | Required | Default | Meaning |
|---|---|---|---|
| `meta-api-key` | yes | — | Meta Model API key. An empty key fails the run step with a clear error. |
| `paths` | no | `""` | Files or directories to check. Empty checks the functions changed against `base`. |
| `base` | no | `""` | Git ref to diff against. Empty diffs against the pull request base branch. |
| `proofs-dir` | no | `proofs` | Directory the certificates are written to and uploaded from. |

## How a run flows

1. The `fork` step skips fork pull requests with a notice; the other setup steps carry on only for
   non-forks.
2. `uv` and Python 3.12, the Dafny 4.11.0 release binary, and `pcc-check` installed from the action's
   own checkout (this repository at the action's ref).
3. `pcc-check run` with `META_API_KEY`, `PCC_META_ALLOWED=1`, `GH_TOKEN` and `GITHUB_TOKEN`. The step
   captures the exit code into an output and does not fail, so the comment and the artifact upload
   still run after a finding.
4. The `proofs/` directory is uploaded as the `proofs` workflow artifact.
5. The comment step finds the pull request comment starting with `<!-- pcc-check -->` and updates it,
   or creates it. The body is the report `pcc-check run` prints (from the marker on), plus a footer
   linking the workflow run. No functions changed renders as "No functions checked.".
6. The last step exits with the captured code, so the check turns red on findings.

Certificates are never committed by the action: download the `proofs` artifact from the workflow run
linked in the comment and commit it to the branch, so the next run re-checks the certificates instead
of re-proving.

## Manual verification in the demo repository

The three examples from the ticket are checked by hand in the demo repository (issue #59):

1. A pull request with one proven function gets exactly one comment; pushing again updates the same
   comment instead of creating a second one.
2. A pull request from a fork shows the skip notice and gets no comment.
3. A pull request with a counterexample turns red, and the comment lists the counterexample first.

Edge cases: a pull request with no changed functions comments "No functions checked."; running the
action without `meta-api-key` fails the run step with the missing-key error. Dispatch
`.github/workflows/action-selftest.yml` by hand for a first smoke test: it runs the local action and
today stops at the missing-key error until `pcc-check run` (issue #40) is implemented and a key is
configured.
