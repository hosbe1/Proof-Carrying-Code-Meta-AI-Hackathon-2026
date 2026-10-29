"""Step 6, build: a ready-to-run pytest from a failing input. Filled in by a later ticket."""

from __future__ import annotations

from pathlib import Path

from pcc.core import Counterexample, FunctionRef, Spec


def build(function: FunctionRef, spec: Spec, repo_root: Path, arguments: str) -> Counterexample:
    """Run ``arguments`` against the real code from ``repo_root`` and package the observed breakage as a test."""
    raise NotImplementedError


__all__ = ["build"]
