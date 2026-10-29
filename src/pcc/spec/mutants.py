"""Step 3, rule mutants: cheap deterministic variants of the function source.

Split out of the mutant kill check so the rules can be tested alone (docs/plan.md, step 3). Each mutant changes
exactly one node of the parsed function and is returned as a complete ``def``. Model-written mutants live in a later
ticket.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Mutation:
    """One single-node edit: where it is, and how to apply it to a copied tree."""

    key: tuple[int, int, int, int]  # source position for ordering
    order: int  # insertion order, for ties at the same position
    path: tuple[tuple[str, int | None], ...]  # steps from the module root to the target node
    mutate: Callable[[ast.AST], ast.AST | None]  # edits the target in place, or returns its replacement


_COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

_BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
}

_BOOLOP_SWAPS: dict[type[ast.boolop], type[ast.boolop]] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


def _swap_compare(index: int, new_op: type[ast.cmpop]) -> Callable[[ast.AST], ast.AST | None]:
    def mutate(node: ast.AST) -> ast.AST | None:
        assert isinstance(node, ast.Compare)
        node.ops[index] = new_op()
        return None

    return mutate


def _swap_binop(new_op: type[ast.operator]) -> Callable[[ast.AST], ast.AST | None]:
    def mutate(node: ast.AST) -> ast.AST | None:
        assert isinstance(node, ast.BinOp)
        node.op = new_op()
        return None

    return mutate


def _swap_boolop(new_op: type[ast.boolop]) -> Callable[[ast.AST], ast.AST | None]:
    def mutate(node: ast.AST) -> ast.AST | None:
        assert isinstance(node, ast.BoolOp)
        node.op = new_op()
        return None

    return mutate


def _set_constant(value: object) -> Callable[[ast.AST], ast.AST | None]:
    def mutate(node: ast.AST) -> ast.AST | None:
        assert isinstance(node, ast.Constant)
        node.value = value
        return None

    return mutate


def _remove_not(node: ast.AST) -> ast.AST | None:
    assert isinstance(node, ast.UnaryOp)
    return node.operand


def _position_of(node: ast.AST) -> tuple[int, int, int, int]:
    """The source position of ``node``; inner nodes sort before outer ones at the same start."""

    def number(name: str) -> int:
        value = getattr(node, name, 0)
        return value if isinstance(value, int) else 0

    return (number("lineno"), number("col_offset"), number("end_lineno"), number("end_col_offset"))


def _collect(root: ast.AST) -> list[_Mutation]:
    """Every candidate single-node edit in the tree, before ordering and limiting."""
    mutations: list[_Mutation] = []

    def add(node: ast.AST, path: tuple[tuple[str, int | None], ...], mutate: Callable[[ast.AST], ast.AST | None]) -> None:
        mutations.append(_Mutation(_position_of(node), len(mutations), path, mutate))

    def visit(node: ast.AST, path: tuple[tuple[str, int | None], ...]) -> None:
        if isinstance(node, ast.Compare):
            for index, op in enumerate(node.ops):
                new_op = _COMPARE_SWAPS.get(type(op))
                if new_op is not None:
                    add(node, path, _swap_compare(index, new_op))
        elif isinstance(node, ast.BinOp):
            new_binop = _BINOP_SWAPS.get(type(node.op))
            if new_binop is not None:
                add(node, path, _swap_binop(new_binop))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                add(node, path, _set_constant(not node.value))
            elif isinstance(node.value, int):
                add(node, path, _set_constant(node.value + 1))
                add(node, path, _set_constant(node.value - 1))
        elif isinstance(node, ast.BoolOp):
            new_boolop = _BOOLOP_SWAPS.get(type(node.op))
            if new_boolop is not None:
                add(node, path, _swap_boolop(new_boolop))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            add(node, path, _remove_not)
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, path + ((field, None),))
            elif isinstance(value, list):
                for item_index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        visit(item, path + ((field, item_index),))

    visit(root, ())
    return mutations


def _navigate(root: ast.AST, path: tuple[tuple[str, int | None], ...]) -> ast.AST:
    node = root
    for field, index in path:
        value = getattr(node, field)
        if index is None:
            assert isinstance(value, ast.AST)
            node = value
        else:
            assert isinstance(value, list)
            item = value[index]
            assert isinstance(item, ast.AST)
            node = item
    return node


def _apply(root: ast.AST, mutation: _Mutation) -> None:
    """Apply ``mutation`` to ``root``, a copy of the tree the mutation was collected from."""
    target = _navigate(root, mutation.path)
    replacement = mutation.mutate(target)
    if replacement is None:
        return
    parent = _navigate(root, mutation.path[:-1])
    field, index = mutation.path[-1]
    if index is None:
        setattr(parent, field, replacement)
    else:
        holder = getattr(parent, field)
        assert isinstance(holder, list)
        holder[index] = replacement


def rule_mutants(source: str, limit: int) -> tuple[str, ...]:
    """Return up to ``limit`` mutants of the function ``source``, in source order.

    Each mutant changes exactly one node (``<`` <-> ``<=``, ``>`` <-> ``>=``, ``==`` <-> ``!=``; ``+`` <-> ``-``,
    ``*`` -> ``+``; int constant ``k`` -> ``k+1`` then ``k-1``; ``True`` <-> ``False``; ``and`` <-> ``or``;
    ``not x`` -> ``x``) and is returned as a complete ``def``. Mutants at the same node keep the rule order listed
    here. ``limit`` is ``config.spec.max_rule_mutants``; ``limit=0`` gives ``()``, as does a function without any
    candidate node.
    """
    if limit <= 0:
        return ()
    tree = ast.parse(source)
    mutations = _collect(tree)
    mutations.sort(key=lambda mutation: (mutation.key, mutation.order))
    mutants: list[str] = []
    for mutation in mutations[:limit]:
        mutant = copy.deepcopy(tree)
        _apply(mutant, mutation)
        ast.fix_missing_locations(mutant)
        mutants.append(ast.unparse(mutant))
    return tuple(mutants)


__all__ = ["rule_mutants"]
