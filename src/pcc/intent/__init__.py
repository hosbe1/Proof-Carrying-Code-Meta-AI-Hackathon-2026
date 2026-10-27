"""Step 2: the intent in plain language, with a source for every claim (docs/plan.md, step 2).

The function body is never shown to the model in this step."""

from __future__ import annotations

from pcc.core import Context, Intent
from pcc.models import ModelClient


def derive(context: Context, *, client: ModelClient, model: str) -> Intent:
    """Write the intent from ``context`` without the function body; resolve contradictions by ``SOURCE_RANK`` and
    record every one of them. Raise ``NotCheckable`` when no sensible intent can be written."""
    raise NotImplementedError


__all__ = ["derive"]
