"""Step 4: Python type hints to Dafny types, and the Dafny keyword rename map.

Spec mode and code mode both take the parameter and result types from the function's hints; a hint outside the
supported subset raises ``UnsupportedConstruct`` and the pipeline falls back to AI translation. Parameters that are
Dafny keywords are renamed to ``<name>_`` so the emitted code resolves; the ``result`` parameter of a spec predicate
keeps its name.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pcc.core import UnsupportedConstruct

#: Dafny 4.11 reserved words. A parameter with one of these names is renamed to ``<name>_``.
DAFNY_KEYWORDS = frozenset(
    {
        "abstract",
        "array",
        "as",
        "assert",
        "assume",
        "await",
        "bool",
        "break",
        "by",
        "calc",
        "case",
        "char",
        "class",
        "codatatype",
        "const",
        "constructor",
        "copredicate",
        "datatype",
        "decreases",
        "default",
        "do",
        "else",
        "ensures",
        "exists",
        "export",
        "extends",
        "false",
        "for",
        "forall",
        "fresh",
        "function",
        "ghost",
        "if",
        "imap",
        "implies",
        "import",
        "in",
        "include",
        "int",
        "invariant",
        "is",
        "iset",
        "iterator",
        "label",
        "lemma",
        "map",
        "match",
        "method",
        "modifies",
        "modify",
        "module",
        "multiset",
        "nat",
        "new",
        "newtype",
        "null",
        "object",
        "old",
        "opaque",
        "predicate",
        "print",
        "protected",
        "provides",
        "reads",
        "real",
        "refines",
        "requires",
        "return",
        "returns",
        "reveal",
        "reveals",
        "sequence",
        "set",
        "static",
        "string",
        "then",
        "this",
        "trait",
        "true",
        "try",
        "twostate",
        "type",
        "var",
        "where",
        "while",
        "yield",
        "yields",
    }
)


def rename(name: str) -> str:
    """``name`` with ``_`` appended when it is a Dafny keyword, unchanged otherwise."""
    return f"{name}_" if name in DAFNY_KEYWORDS else name


def _split_generic(hint: str) -> tuple[str, str] | None:
    """``("list", "int")`` for ``"list[int]"``; ``None`` when ``hint`` is not one generic application."""
    if not hint.endswith("]"):
        return None
    depth = 0
    for index, char in enumerate(hint):
        if char == "[":
            if depth == 0 and index > 0:
                head = hint[:index].strip()
                inner = hint[index + 1 : -1]
                if head and "[" not in head and "," not in inner:
                    return head, inner
            depth += 1
        elif char == "]":
            depth -= 1
    return None


def dafny_type(hint: str) -> str:
    """The Dafny type for the Python hint ``hint``, with surrounding whitespace ignored.

    ``int`` maps to ``int``, ``bool`` to ``bool``, ``str`` to ``string`` and ``list[T]`` (or ``typing.List[T]``) to
    ``seq[T]``, recursively. Anything else raises ``UnsupportedConstruct("type <hint>")``.
    """
    normalized = hint.strip()
    if normalized == "int":
        return "int"
    if normalized == "bool":
        return "bool"
    if normalized == "str":
        return "string"
    generic = _split_generic(normalized)
    if generic is not None:
        head, inner = generic
        if head in ("list", "typing.List"):
            return f"seq<{dafny_type(inner)}>"
    raise UnsupportedConstruct(f"type {normalized}")


@dataclass(frozen=True, slots=True)
class Parameter:
    """One function parameter: its Python name, its Dafny name and its Dafny type."""

    python: str  # as written in the function signature
    dafny: str  # ``rename(python)``
    dafny_type: str  # ``dafny_type`` of the annotation


def _function_def(source: str) -> ast.FunctionDef:
    """The single function definition in ``source``."""
    (definition,) = ast.parse(source).body
    assert isinstance(definition, ast.FunctionDef)
    return definition


def function_parameters(source: str) -> tuple[Parameter, ...]:
    """The positional and keyword-only parameters of the function ``source``, with Dafny names and types.

    A parameter without an annotation raises ``UnsupportedConstruct("no types")``; ``*args`` and ``**kwargs``
    raise ``UnsupportedConstruct("parameter *<name>")``.
    """
    definition = _function_def(source)
    args = definition.args
    if args.vararg is not None:
        raise UnsupportedConstruct(f"parameter *{args.vararg.arg}")
    if args.kwarg is not None:
        raise UnsupportedConstruct(f"parameter **{args.kwarg.arg}")
    parameters: list[Parameter] = []
    for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if argument.annotation is None:
            raise UnsupportedConstruct("no types")
        parameters.append(Parameter(argument.arg, rename(argument.arg), dafny_type(ast.unparse(argument.annotation))))
    return tuple(parameters)


def function_return_type(source: str) -> str:
    """The Dafny type of the return annotation of the function ``source``.

    A missing return annotation raises ``UnsupportedConstruct("no types")``.
    """
    definition = _function_def(source)
    if definition.returns is None:
        raise UnsupportedConstruct("no types")
    return dafny_type(ast.unparse(definition.returns))


__all__ = ["DAFNY_KEYWORDS", "Parameter", "dafny_type", "function_parameters", "function_return_type", "rename"]
