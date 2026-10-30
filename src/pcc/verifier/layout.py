"""Dafny module and method names, and assembling the verified file.

Every Dafny file the pipeline produces has the same shape: one module named after the function, opening ``PyLib``,
with the translated spec, the proof lemmas and the translated function. The module name is unique per function, so all
certificates verify together as one project. ``imports`` holds the callee module names the caller uses; each one
becomes an ``import <name>`` line after ``import opened PyLib``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from pcc.core import FunctionRef


def _sanitize(text: str) -> str:
    """``text`` with every character outside ``[A-Za-z0-9]`` replaced by ``_``."""
    return re.sub(r"[^A-Za-z0-9]", "_", text)


def method_name(function: FunctionRef) -> str:
    """The Dafny method name for ``function``: its qualname with ``.`` replaced by ``_``.

    Names that would start with ``_`` get the prefix ``pcc_``, because a leading underscore needs escaping for the
    transpiler, the AI translation and the equivalence wrapper that all use this name.
    """
    name = function.qualname.replace(".", "_")
    if name.startswith("_"):
        return f"pcc_{name}"
    return name


def module_name(function: FunctionRef) -> str:
    """The Dafny module name derived from the file path and qualname of ``function``.

    The shape is ``"PCC_" + s(path without .py) + "__" + s(qualname) + "_" + h``: ``s`` replaces every character
    outside ``[A-Za-z0-9]`` with ``_``, and ``h`` is the first 8 hex digits of
    ``sha256(f"{path}:{qualname}")``. The hash keeps names unique when two paths sanitize to the same stem, for
    example ``a/b.py`` and ``a_b.py``.
    """
    stem = function.path.removesuffix(".py")
    digest = hashlib.sha256(f"{function.path}:{function.qualname}".encode()).hexdigest()[:8]
    return f"PCC_{_sanitize(stem)}__{_sanitize(function.qualname)}_{digest}"


def _indent(block: str) -> str:
    """``block`` with every non-blank line indented by two spaces."""
    return "\n".join(f"  {line}" if line.strip() else "" for line in block.split("\n"))


def assemble(
    function: FunctionRef,
    dafny_spec: str,
    dafny_function: str,
    *,
    lemmas: str = "",
    imports: Sequence[str] = (),
) -> str:
    """The complete Dafny source that ``dafny verify`` checks for ``function``.

    The file holds one module named by :func:`module_name` that opens ``PyLib``, then one ``import <name>`` line
    per entry of ``imports`` (deduplicated, order-preserving), then the translated spec, the proof lemmas (skipped
    when empty, so no blank lemma section remains) and the translated function, each indented by two spaces. There is
    no ``include`` line: ``PyLib.dfy`` is passed to Dafny as a second file.
    """
    parts = [f"module {module_name(function)} {{", "  import opened PyLib"]
    seen: set[str] = set()
    for name in imports:
        if name not in seen:
            seen.add(name)
            parts.append(f"  import {name}")
    for block in (dafny_spec, lemmas, dafny_function):
        if block:
            parts.append(_indent(block))
    parts.append("}")
    return "\n".join(parts) + "\n"


__all__ = ["assemble", "method_name", "module_name"]
