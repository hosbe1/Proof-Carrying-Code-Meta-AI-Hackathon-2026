"""Follow-up search agent for step 1 (docs/plan.md); built by its own ticket."""

from __future__ import annotations

from pathlib import Path

from pcc.core import Context
from pcc.models import ModelClient


def search_further(context: Context, repo_root: Path, *, client: ModelClient, model: str) -> Context:
    """Agent with read-only tools that adds evidence when the base package leaves the intent unclear."""
    raise NotImplementedError


__all__ = ["search_further"]
