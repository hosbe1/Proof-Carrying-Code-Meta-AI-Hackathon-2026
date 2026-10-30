"""Step 1: collect the evidence about a function's intent (docs/plan.md, step 1)."""

from __future__ import annotations

from pathlib import Path

from pcc.context.search import search_further
from pcc.core import Context, FunctionRef


def collect(function: FunctionRef, repo_root: Path) -> Context:
    """Fixed collector: name, signature, type hints, docstring, comments, tests, call sites, commit history,
    pull request description, linked issues, README and docs. Deterministic, no model calls."""
    raise NotImplementedError


__all__ = ["collect", "search_further"]
