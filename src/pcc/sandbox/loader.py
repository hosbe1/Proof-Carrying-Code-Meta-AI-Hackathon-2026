"""Loading the function under test in the sandbox. Filled in by a later ticket."""

from __future__ import annotations

from pcc.core import FunctionRef


def target_loader(function: FunctionRef) -> str:
    """Python source that imports ``function`` from the repository under test."""
    raise NotImplementedError


__all__ = ["target_loader"]
