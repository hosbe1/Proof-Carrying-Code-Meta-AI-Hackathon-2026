"""Runs the steps for one function and for a set of functions. The only module that imports every step."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from pcc.config import Config
from pcc.core import FunctionRef, Result
from pcc.models import ModelClient


def run(
    function: FunctionRef,
    repo_root: Path,
    *,
    client: ModelClient,
    config: Config,
    callees: Mapping[FunctionRef, Result] = MappingProxyType({}),
) -> Result:
    """Steps 1 to 7 for one function; every expected failure becomes a result state, never an exception.

    ``callees`` holds the already proven results of callees in the project, so the run uses their specs."""
    raise NotImplementedError


def run_many(
    functions: Sequence[FunctionRef], repo_root: Path, *, client: ModelClient, config: Config
) -> tuple[Result, ...]:
    """Run in ``pcc.project.call_order``, reusing proven callee specs; functions of one level run in parallel."""
    raise NotImplementedError


__all__ = ["run", "run_many"]
