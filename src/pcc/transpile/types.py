"""Step 4, code mode: the Python to Dafny type mapping.

Supported annotations and their Dafny type:

- ``int`` to ``int``, ``bool`` to ``bool``, ``str`` to ``string``;
- ``list[T]`` and ``typing.List[T]`` to ``seq<T>``, only when the list is never mutated
  (``pcc.transpile.code`` rejects mutated lists with ``"mutated list"``);
- ``tuple`` and ``typing.Tuple`` are not in this subset and raise ``"type tuple"``;
- anything else raises ``"type <T>"`` with the annotation as written, and a missing
  annotation raises ``"no types"``.
"""

from __future__ import annotations

import ast

from pcc.core import UnsupportedConstruct


def _text(node: ast.expr) -> str:
    """``node`` as written, for ``"type <T>"`` errors."""
    try:
        return ast.unparse(node).strip()
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return type(node).__name__


def _element_type(node: ast.expr, *, lineno: int) -> str:
    """The Dafny element type of a ``list[T]`` annotation."""
    return dafny_type(node, lineno=lineno)


def dafny_type(node: ast.expr | None, *, lineno: int) -> str:
    """The Dafny type of the annotation ``node``.

    ``node`` is ``None`` when the parameter or return has no hint, which raises
    ``UnsupportedConstruct("no types")``. ``lineno`` is relative to the function
    source (line 1 is its first line), as in ``pcc.transpile.code``.
    """
    if node is None:
        raise UnsupportedConstruct("no types", lineno=lineno)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            parsed = ast.parse(node.value, mode="eval").body
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno) from None
        try:
            return dafny_type(parsed, lineno=node.lineno)
        except UnsupportedConstruct as error:
            raise UnsupportedConstruct(error.construct, lineno=node.lineno) from None
    if isinstance(node, ast.Name):
        if node.id == "int":
            return "int"
        if node.id == "bool":
            return "bool"
        if node.id == "str":
            return "string"
        if node.id in ("tuple", "Tuple"):
            raise UnsupportedConstruct("type tuple", lineno=node.lineno)
        if node.id in ("list", "List"):
            raise UnsupportedConstruct(f"type {node.id}", lineno=node.lineno)
        raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)
    if isinstance(node, ast.Subscript):
        value = node.value
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if isinstance(value, ast.Name) and value.id in ("list", "List"):
            if len(args) != 1:
                raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)
            return f"seq<{_element_type(args[0], lineno=lineno)}>"
        if isinstance(value, ast.Attribute) and value.attr == "List":
            if isinstance(value.value, ast.Name) and value.value.id == "typing" and len(args) == 1:
                return f"seq<{_element_type(args[0], lineno=lineno)}>"
            raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)
        if isinstance(value, ast.Name) and value.id in ("tuple", "Tuple"):
            raise UnsupportedConstruct("type tuple", lineno=node.lineno)
        if isinstance(value, ast.Attribute) and value.attr == "Tuple":
            raise UnsupportedConstruct("type tuple", lineno=node.lineno)
        raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)
    if isinstance(node, ast.Attribute):
        if node.attr in ("List", "Tuple"):
            base = node.value
            if isinstance(base, ast.Name) and base.id == "typing":
                if node.attr == "Tuple":
                    raise UnsupportedConstruct("type tuple", lineno=node.lineno)
                raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)
        raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)
    if isinstance(node, ast.Constant) and node.value is None:
        raise UnsupportedConstruct("type None", lineno=node.lineno)
    raise UnsupportedConstruct(f"type {_text(node)}", lineno=node.lineno)


__all__ = ["dafny_type"]
