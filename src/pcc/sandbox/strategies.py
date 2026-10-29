"""Hypothesis strategies for the function's inputs. Filled in by a later ticket."""

from __future__ import annotations

from pcc.core import FunctionRef


def strategies_source(function: FunctionRef) -> str:
    """Python source that builds the Hypothesis strategies for the inputs of ``function``."""
    raise NotImplementedError


__all__ = ["strategies_source"]
