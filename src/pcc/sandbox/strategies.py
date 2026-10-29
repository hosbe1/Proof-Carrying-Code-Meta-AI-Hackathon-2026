"""Hypothesis strategies for the inputs of the function under test.

The spec check, the counterexample search and the equivalence check all draw inputs from the type hints of the
function. They must not import each other, so the source they embed lives here. Each script executes the source and
draws argument tuples from ``args_strategy``.
"""

from __future__ import annotations

import ast

from pcc.core import FunctionRef, UnsupportedConstruct


def strategies_source(function: FunctionRef) -> str:
    """Python source defining ``args_strategy`` for the inputs of ``function``.

    The source imports ``hypothesis.strategies`` as ``st`` and defines ``args_strategy``, a strategy of argument
    tuples in parameter order (positional-only, positional-or-keyword, then keyword-only). Each hint is parsed with
    ``ast`` and mapped to a strategy: ``int`` to ``st.integers()``, ``bool`` to ``st.booleans()``, ``str`` to text
    without lone surrogates, ``float`` to finite floats, ``list[T]`` to bounded lists, ``tuple[A, B]`` to tuples, and
    ``Optional[T]`` or ``T | None`` to ``st.none() | <T>``. A parameter without a hint raises
    ``UnsupportedConstruct("no types")``; any other hint raises ``UnsupportedConstruct("type <T>")``.
    """
    tree = ast.parse(function.source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        raise UnsupportedConstruct("no types")
    parameters = [*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs]
    extras = [extra for extra in (definition.args.vararg, definition.args.kwarg) if extra is not None]
    expressions = [_strategy_expression(parameter.annotation) for parameter in parameters]
    for extra in extras:
        if extra.annotation is None:
            raise UnsupportedConstruct("no types")
        raise UnsupportedConstruct(f"type {ast.unparse(extra.annotation)}")
    return f"import hypothesis.strategies as st\n\nargs_strategy = st.tuples({', '.join(expressions)})\n"


def _strategy_expression(hint: ast.expr | None) -> str:
    """The Hypothesis strategy source for one parameter hint; ``None`` means the parameter has no hint."""
    if hint is None:
        raise UnsupportedConstruct("no types")
    if isinstance(hint, ast.Name):
        if hint.id == "int":
            return "st.integers()"
        if hint.id == "bool":
            return "st.booleans()"
        if hint.id == "str":
            return 'st.text(alphabet=st.characters(exclude_categories=("Cs",)), max_size=20)'
        if hint.id == "float":
            return "st.floats(allow_nan=False, allow_infinity=False)"
    elif isinstance(hint, ast.Subscript):
        items = _subscript_items(hint)
        if _is_plain_or_typing(hint.value, "list", "List") and len(items) == 1:
            return f"st.lists({_strategy_expression(items[0])}, max_size=20)"
        if _is_plain_or_typing(hint.value, "tuple", "Tuple"):
            inner = ", ".join(_strategy_expression(item) for item in items)
            return f"st.tuples({inner})"
        if _is_plain_or_typing(hint.value, "Optional", "Optional") and len(items) == 1:
            return f"st.none() | {_strategy_expression(items[0])}"
    elif isinstance(hint, ast.BinOp) and isinstance(hint.op, ast.BitOr):
        if _is_none(hint.left):
            return f"st.none() | {_strategy_expression(hint.right)}"
        if _is_none(hint.right):
            return f"st.none() | {_strategy_expression(hint.left)}"
    raise UnsupportedConstruct(f"type {ast.unparse(hint)}")


def _subscript_items(hint: ast.Subscript) -> list[ast.expr]:
    """The type arguments of a subscript hint; ``tuple[A, B]`` gives ``[A, B]``, ``list[T]`` gives ``[T]``."""
    if isinstance(hint.slice, ast.Tuple):
        return list(hint.slice.elts)
    return [hint.slice]


def _is_plain_or_typing(node: ast.expr, builtin: str, typing_name: str) -> bool:
    """Whether ``node`` is the bare name ``builtin`` or the attribute ``typing.<typing_name>``."""
    if isinstance(node, ast.Name):
        return node.id == builtin
    return _is_typing_attribute(node, typing_name)


def _is_typing_attribute(node: ast.expr, name: str) -> bool:
    """Whether ``node`` is the attribute ``typing.<name>``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
    )


def _is_none(node: ast.expr) -> bool:
    """Whether ``node`` is the ``None`` constant of a ``T | None`` union."""
    return isinstance(node, ast.Constant) and node.value is None


__all__ = ["strategies_source"]
