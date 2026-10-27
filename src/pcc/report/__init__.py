"""The pull request comment (docs/plan.md, section Pull request comment)."""

from __future__ import annotations

from collections.abc import Sequence

from pcc.core import Result


def render(results: Sequence[Result]) -> str:
    """Markdown: summary line on top, counterexamples first, then per function state, translation path, intent with
    sources, counterexample as a ready-to-copy test, assumptions, and the Dafny spec collapsed."""
    raise NotImplementedError


__all__ = ["render"]
