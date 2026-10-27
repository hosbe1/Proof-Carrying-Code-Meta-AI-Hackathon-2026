"""Runs the steps for one function and for a set of functions. The only module that imports every step."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pcc.config import Config
from pcc.core import FunctionRef, Result
from pcc.models import ModelClient


def run(function: FunctionRef, repo_root: Path, *, client: ModelClient, config: Config) -> Result:
    """Steps 1 to 7 for one function; every expected failure becomes a result state, never an exception."""
    raise NotImplementedError


def run_many(
    functions: Sequence[FunctionRef], repo_root: Path, *, client: ModelClient, config: Config
) -> tuple[Result, ...]:
    """Run in ``pcc.project.call_order``, reusing proven callee specs; functions of one level run in parallel."""
    raise NotImplementedError


__all__ = ["run", "run_many"]
