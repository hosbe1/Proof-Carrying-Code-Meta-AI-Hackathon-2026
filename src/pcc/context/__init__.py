"""Step 1: collect the evidence about a function's intent (docs/plan.md, step 1)."""

from __future__ import annotations

from pcc.context.collect import collect
from pcc.context.github import github_evidence
from pcc.context.search import search_further

__all__ = ["collect", "github_evidence", "search_further"]
