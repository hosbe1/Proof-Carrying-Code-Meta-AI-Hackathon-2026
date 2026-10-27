"""Shared fixtures. Tests marked ``live`` call real models and are skipped by default; ``dafny`` tests need the CLI.

With ``PCC_REQUIRE_DAFNY=1`` (set in CI) the session fails at start when the dafny CLI is missing or reports a version
other than ``Config().dafny.version``. Without it, the ``dafny`` tests are skipped when the CLI is missing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from pcc.config import Config


def pytest_sessionstart(session: pytest.Session) -> None:
    """Runs once per session, before collection."""
    if os.environ.get("PCC_REQUIRE_DAFNY") == "1":
        _require_dafny_version()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shutil.which("dafny"):
        return
    skip = pytest.mark.skip(reason="dafny CLI not on PATH")
    for item in items:
        if "dafny" in item.keywords:
            item.add_marker(skip)


def _require_dafny_version() -> None:
    expected = Config().dafny.version
    if shutil.which("dafny") is None:
        raise pytest.UsageError(f"PCC_REQUIRE_DAFNY=1 but the dafny CLI is not on PATH; install Dafny {expected}")
    try:
        proc = subprocess.run(["dafny", "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise pytest.UsageError(f"PCC_REQUIRE_DAFNY=1 but `dafny --version` failed: {error}") from error
    reported = _release_of(proc.stdout) if proc.returncode == 0 else None
    if reported != expected:
        shown = (proc.stdout or proc.stderr).strip() or "no output"
        raise pytest.UsageError(
            f"PCC_REQUIRE_DAFNY=1: the dafny on PATH reports {shown!r}, but Config().dafny.version is {expected!r}. "
            f"Install Dafny {expected}; the proofs depend on the Z3 it bundles."
        )


def _release_of(output: str) -> str | None:
    """The release number in ``dafny --version`` output, without a build suffix such as ``+abc123``."""
    match = re.search(r"(\d+(?:\.\d+)+)(?:\+\S*)?", output)
    return match.group(1) if match else None
