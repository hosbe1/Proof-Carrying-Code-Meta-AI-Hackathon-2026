"""Step 7: a second spec written with the function body in view; a hint, never a verdict."""

from __future__ import annotations

from pcc.core import Context, SecondOpinion, Spec
from pcc.models import ModelClient


def review(context: Context, spec: Spec, *, client: ModelClient, model: str) -> SecondOpinion:
    """Only called when step 5 or 6 reported a problem or the sources contradict each other."""
    raise NotImplementedError


__all__ = ["review"]
