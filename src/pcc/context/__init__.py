"""Step 1: collect the evidence about a function's intent (docs/plan.md, step 1)."""

from __future__ import annotations

from pathlib import Path

from pcc.core import Context, FunctionRef
from pcc.models import ModelClient


def collect(function: FunctionRef, repo_root: Path) -> Context:
    """Fixed collector: name, signature, type hints, docstring, comments, tests, call sites, commit history,
    pull request description, linked issues, README and docs. Deterministic, no model calls."""
    raise NotImplementedError


def search_further(context: Context, repo_root: Path, *, client: ModelClient, model: str) -> Context:
    """Agent with read-only tools that adds evidence when the base package leaves the intent unclear."""
    raise NotImplementedError


__all__ = ["collect", "search_further"]
