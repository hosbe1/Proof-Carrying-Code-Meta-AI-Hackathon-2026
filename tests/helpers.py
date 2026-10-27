"""Factories shared by the tests."""

from __future__ import annotations

from pcc.core import FunctionRef, source_hash

DEFAULT_SOURCE = "def total(self): ..."


def function_ref(
    source: str = DEFAULT_SOURCE,
    *,
    path: str = "src/cart.py",
    qualname: str = "Cart.total",
    lineno: int = 12,
) -> FunctionRef:
    """A ``FunctionRef`` whose ``source_hash`` is computed from ``source``, as the reader does."""
    return FunctionRef(path, qualname, lineno, source, source_hash(source))
