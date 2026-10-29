"""Which functions changed on a pull request. Filled in by a later ticket."""

from __future__ import annotations

from pathlib import Path

from pcc.core import FunctionRef


def changed(repo_root: Path, base: str) -> tuple[FunctionRef, ...]:
    """Every function defined in the diff hunks of ``repo_root`` against ``base``."""
    raise NotImplementedError


__all__ = ["changed"]
