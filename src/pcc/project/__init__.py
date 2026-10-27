"""Finding the functions to check and the order to check them in (docs/plan.md, step 4, calls inside the project)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pcc.core import FunctionRef


def discover(repo_root: Path, paths: Sequence[str]) -> tuple[FunctionRef, ...]:
    """Return every function and method defined in ``paths`` (relative to ``repo_root``)."""
    raise NotImplementedError


def affected(repo_root: Path, changed: Sequence[FunctionRef]) -> tuple[FunctionRef, ...]:
    """Return ``changed`` plus every caller whose proof may depend on them."""
    raise NotImplementedError


def call_order(functions: Sequence[FunctionRef]) -> tuple[tuple[FunctionRef, ...], ...]:
    """Group ``functions`` into levels: callees before callers; functions in one level can run in parallel.

    Mutually recursive functions share a level and use each other's spec as an assumption.
    """
    raise NotImplementedError


__all__ = ["affected", "call_order", "discover"]
