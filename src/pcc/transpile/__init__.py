"""Step 4: the rule-based transpiler (docs/plan.md, step 4). What it supports, it translates; anything else
raises ``UnsupportedConstruct`` and the pipeline falls back to AI translation."""

from __future__ import annotations

from pcc.core import FunctionRef, Spec


def transpile_function(function: FunctionRef) -> str:
    """Code mode: executable Dafny built from ``pcc.pylib`` blocks."""
    raise NotImplementedError


def transpile_spec(spec: Spec) -> str:
    """Spec mode: logical Dafny (``forall``/``exists`` instead of loops) as ``requires``/``ensures`` clauses."""
    raise NotImplementedError


__all__ = ["transpile_function", "transpile_spec"]
