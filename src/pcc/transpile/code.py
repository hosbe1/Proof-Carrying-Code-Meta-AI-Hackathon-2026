"""Step 4: the rule-based transpiler, code mode and spec mode. Filled in by a later ticket."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pcc.core import FunctionRef, Spec


def transpile_function(function: FunctionRef, *, callees: Mapping[str, str] = MappingProxyType({})) -> str:
    """Code mode: executable Dafny built from ``pcc.pylib`` blocks.

    ``callees`` maps the qualname of each callee in the project to its Dafny module name, so callers use only the
    callee's spec.
    """
    raise NotImplementedError


def transpile_spec(spec: Spec) -> str:
    """Spec mode: logical Dafny (``forall``/``exists`` instead of loops) as ``requires``/``ensures`` clauses."""
    raise NotImplementedError


__all__ = ["transpile_function", "transpile_spec"]
