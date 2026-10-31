"""Acceptance tests for the GitHub Action (issue #41).

The action is a composite action in ``action.yml`` at the repository root, exercised through
``.github/workflows/action-selftest.yml`` and documented in ``docs/action.md``. These tests read the
checked-in files as text (no YAML library, no network) and assert the interface the ticket requires:
inputs, fork skip, pinned setup steps, the ``pcc-check run`` invocation with its environment, the
artifact upload, the sticky comment and the final exit code.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTION = REPO_ROOT / "action.yml"
SELFTEST = REPO_ROOT / ".github" / "workflows" / "action-selftest.yml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOC = REPO_ROOT / "docs" / "action.md"

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SKIP_GUARD = "steps.fork.outputs.skip != 'true'"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_action_file_exists() -> None:
    assert ACTION.is_file(), "action.yml must exist at the repository root"


def test_action_is_composite() -> None:
    assert "using: composite" in _read(ACTION), "the action must be a composite action"


def test_meta_api_key_is_required() -> None:
    body = _read(ACTION)
    assert "meta-api-key" in body, "the action must take a meta-api-key input"
    section = body.split("meta-api-key", 1)[1].split("paths:", 1)[0]
    assert "required: true" in section, "meta-api-key must be required"


def test_optional_inputs_have_documented_defaults() -> None:
    body = _read(ACTION)
    for name in ("paths", "base", "proofs-dir"):
        assert name in body, f"the action must take a {name} input"
    assert 'default: "proofs"' in body or "default: proofs" in body, "proofs-dir must default to proofs"
    assert "GITHUB_BASE_REF" in body, "an empty base must fall back to the pull request base"


def test_empty_base_prefers_origin_prefixed_pr_base() -> None:
    body = _read(ACTION)
    assert "GITHUB_BASE_REF" in body, "an empty base must fall back to the pull request base"
    assert "origin/$base" in body or "origin/${GITHUB_BASE_REF" in body, (
        "the pull request base must be resolved to origin/<branch> so git diff finds it"
    )


def test_last_step_runs_even_after_a_failure() -> None:
    body = _read(ACTION)
    section = body.split("Fail on findings", 1)[1]
    assert "always()" in section, "the last step must run even after a failed comment step"


def test_fork_step_sets_skip_and_prints_notice() -> None:
    body = _read(ACTION)
    assert "id: fork" in body, "the first step must be the fork check with id fork"
    assert "skip=true" in body, "the fork step must set the skip output for fork pull requests"
    assert "::notice::" in body, "the fork step must print a notice when it skips"


def test_setup_uv_is_pinned_by_sha() -> None:
    matches = [line for line in _read(ACTION).splitlines() if "astral-sh/setup-uv@" in line]
    assert matches, "the action must install uv with astral-sh/setup-uv"
    assert any(_SHA_RE.search(line) for line in matches), "astral-sh/setup-uv must be pinned by SHA"


def test_dafny_release_binary() -> None:
    assert "4.11.0" in _read(ACTION), "the action must install the Dafny 4.11.0 release binary"


def test_pcc_is_installed_from_the_action_ref() -> None:
    body = _read(ACTION)
    assert "uv tool install" in body, "the action must install this repository with uv tool install"


def test_check_step_environment_and_exit_code() -> None:
    body = _read(ACTION)
    assert "pcc-check run" in body, "the action must run pcc-check run"
    for variable in ("META_API_KEY", "PCC_META_ALLOWED", "GH_TOKEN", "GITHUB_TOKEN"):
        assert variable in body, f"the check step must set {variable}"
    assert '"1"' in body or ": 1" in body or "PCC_META_ALLOWED: '1'" in body, "PCC_META_ALLOWED must be 1"
    assert "exit_code=" in body, "the check step must capture the exit code into an output"
    assert "::error::" in body, "a missing meta-api-key must fail with a clear message"


def test_check_step_writes_report_file() -> None:
    body = _read(ACTION)
    assert "--report" in body, "the check step must ask pcc-check run to write its markdown report to a file"
    assert "pcc-report.md" in body, "the report file must be a stable name both steps share"


def test_comment_step_reads_report_file() -> None:
    body = _read(ACTION)
    comment = body.split("Comment results", 1)[1]
    assert "pcc-report.md" in comment, "the comment step must read the report file, not the run stdout"


def test_artifact_upload_is_pinned_by_sha() -> None:
    lines = [line for line in _read(ACTION).splitlines() if "actions/upload-artifact@" in line]
    assert lines, "the action must upload the proofs with actions/upload-artifact"
    assert any(_SHA_RE.search(line) for line in lines), "actions/upload-artifact must be pinned by SHA"
    assert "proofs" in _read(ACTION), "the upload must cover the proofs directory"


def test_comment_step_updates_one_sticky_comment() -> None:
    body = _read(ACTION)
    assert "always()" in body, "the comment step must run even after a failed check"
    assert _SKIP_GUARD in body, "the comment step must still skip fork pull requests"
    assert "<!-- pcc-check -->" in body, "the comment step must find the marker of the sticky comment"
    assert "gh api" in body, "the comment step must update or create the comment through gh api"


def test_comment_links_artifact_and_commit_step() -> None:
    body = _read(ACTION)
    lowered = body.lower()
    assert "artifact" in lowered, "the comment must link the proofs artifact"
    assert "commit" in lowered, "the comment must tell the developer to commit the proofs"


def test_last_step_exits_with_the_captured_code() -> None:
    assert "steps.check.outputs.exit_code" in _read(ACTION), "the last step must exit with the captured code"


def test_non_fork_steps_carry_the_skip_guard() -> None:
    count = _read(ACTION).count(_SKIP_GUARD)
    assert count >= 6, f"every later step needs the fork skip guard, found {count} guards"


def test_selftest_runs_the_local_action() -> None:
    assert SELFTEST.is_file(), "action-selftest.yml must exist so actionlint checks a realistic invocation"
    assert "uses: ./" in _read(SELFTEST), "the self-test must run the action with uses: ./"


def test_selftest_checks_out_full_history() -> None:
    assert "fetch-depth: 0" in _read(SELFTEST), "the self-test must check out with fetch-depth: 0"


def test_ci_runs_actionlint_on_the_selftest() -> None:
    body = _read(CI)
    assert "actionlint" in body, "CI must run actionlint"
    assert "action-selftest" in body, "CI must run actionlint on the self-test workflow"
    assert ".github/workflows/action-selftest.yml" in body, (
        "CI must lint the exact self-test workflow path, not a similarly named file"
    )
    assert SELFTEST.is_file(), "the linted self-test workflow must exist"
    assert "1.7.7" in body, "CI must pin the actionlint version for reproducible runs"
    assert "set -e" in body, "the actionlint step must propagate a finding as a step failure"
    assert "continue-on-error" not in body, "the actionlint step must fail the job on findings"


def test_docs_cover_the_example_workflow() -> None:
    assert DOC.is_file(), "docs/action.md must document the action"
    body = _read(DOC)
    assert "fetch-depth: 0" in body, "the example workflow must check out with fetch-depth: 0"
    assert "pull-requests: write" in body, "the example workflow must request the comment permission"
    assert "contents: read" in body, "the example workflow must request contents: read"
    assert "META_API_KEY" in body, "the docs must explain the model API key secret"


def test_docs_cover_artifact_and_manual_verification() -> None:
    body = _read(DOC)
    lowered = body.lower()
    assert "artifact" in lowered, "the docs must explain the proofs artifact"
    assert "commit" in lowered, "the docs must tell the developer to commit the proofs"
    assert "No functions checked." in body, "the docs must document the empty result comment"
    assert "fork" in lowered, "the docs must document the fork behaviour"
