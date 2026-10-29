"""Dafny module and method names, and assembling the verified file. Filled in by a later ticket."""

from __future__ import annotations

from collections.abc import Sequence

from pcc.core import FunctionRef


def method_name(function: FunctionRef) -> str:
    """The Dafny method name for ``function``."""
    raise NotImplementedError


def module_name(function: FunctionRef) -> str:
    """The Dafny module name derived from the file path and qualname of ``function``."""
    raise NotImplementedError


def assemble(
    function: FunctionRef,
    dafny_spec: str,
    dafny_function: str,
    *,
    lemmas: str = "",
    imports: Sequence[str] = (),
) -> str:
    """The complete Dafny source that ``dafny verify`` checks for ``function``."""
    raise NotImplementedError


__all__ = ["assemble", "method_name", "module_name"]
