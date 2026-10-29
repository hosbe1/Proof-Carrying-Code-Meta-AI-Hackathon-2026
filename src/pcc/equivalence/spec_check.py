"""Step 4, spec check: does the transpiled spec agree with the Python spec? Filled in by a later ticket."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pcc.core import CheckRecord, FunctionRef, Spec


def check_spec(
    spec: Spec,
    dafny_spec: str,
    function: FunctionRef,
    repo_root: Path,
    *,
    cases: Sequence[tuple[str, str]],
) -> CheckRecord:
    """Evaluate the transpiled ``dafny_spec`` (compiled back to Python) next to the Python ``spec`` on ``cases``;
    they must agree on every input. The original code runs from ``repo_root``, so its imports resolve as they do in
    the project."""
    raise NotImplementedError


__all__ = ["check_spec"]
