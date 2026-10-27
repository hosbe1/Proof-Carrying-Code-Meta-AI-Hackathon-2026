"""Contracts for code that cannot be proven: libraries, files, network (docs/plan.md, step 4).

Type stubs give the base, the model adds behaviour; every contract becomes a listed assumption."""

from __future__ import annotations

from pcc.core import Assumption
from pcc.models import ModelClient


def for_call(qualified_name: str, *, client: ModelClient, model: str) -> Assumption:
    """Return the contract for a call to ``qualified_name`` (e.g. ``json.loads``) as an assumption with Dafny text."""
    raise NotImplementedError


__all__ = ["for_call"]
