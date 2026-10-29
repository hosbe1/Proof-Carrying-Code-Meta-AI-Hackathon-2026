"""The call graph between functions of the project. Filled in by a later ticket."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pcc.core import FunctionRef


def call_graph(repo_root: Path, functions: Sequence[FunctionRef]) -> Mapping[FunctionRef, frozenset[FunctionRef]]:
    """The callees of every function in ``functions``, restricted to the project itself."""
    raise NotImplementedError


__all__ = ["call_graph"]
