"""Step 3, writing: the spec as an executable Python predicate, blind to the body."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from pcc.core import Claim, Intent, NotCheckable, Spec
from pcc.models import Message, ModelClient
from pcc.spec.prompts import SCHEMA, SYSTEM_PROMPT, build_user_prompt

_STEP = "spec"

_LIST_HINT = re.compile(r"\blist\b", re.IGNORECASE)


class _Invalid(Exception):
    """A model answer that fails spec validation; the writer retries it once."""

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem


def write(intent: Intent, *, client: ModelClient, model: str, hint: str = "") -> Spec:
    """Turn the intent into ``spec(<args>, result) -> bool`` and an optional ``pre(<args>) -> bool``.

    The function body is never shown to the model. The prompt holds only the function's name, signature, type hints
    and the intent, even though ``intent.function.source`` contains the body. ``hint`` holds only inputs with the
    original's and a mutant's result, never mutant code. On a validation error there is one retry with the error; if
    it fails again, the spec is ``NotCheckable("spec malformed")``.
    """
    signature, params, annotations = _signature(intent.function.source)
    messages = (Message("system", SYSTEM_PROMPT), Message("user", build_user_prompt(intent, signature, hint)))
    first = client.complete(model, messages, step=_STEP, schema=SCHEMA).text
    try:
        parsed = _parse(first, intent, params, annotations)
    except _Invalid as exc:
        retry = (*messages, Message("user", _retry_text(exc.problem, first)))
        second = client.complete(model, retry, step=_STEP, schema=SCHEMA).text
        try:
            parsed = _parse(second, intent, params, annotations)
        except _Invalid:
            raise NotCheckable("spec malformed") from None
    predicate, precondition, sources = parsed
    return Spec(intent.function, predicate, precondition, (), sources)


def _retry_text(problem: str, previous: str) -> str:
    """Ask for a corrected answer, naming what was wrong and echoing the previous output."""
    return (
        f"Invalid spec: {problem}.\n"
        f"Previous answer:\n{previous}\n"
        "Answer again with corrected JSON using only the signature, the claims and the allowed constructs."
    )


def _signature(source: str) -> tuple[str, list[str], dict[str, str | None]]:
    """Split the function ``source`` into its display signature, parameter names and annotations.

    Only the name, the parameter names with their hints and the return hint are rendered; the body never leaves
    this function. Anything but one plain ``def`` with positional parameters and no defaults is
    ``NotCheckable("spec malformed")``, so call semantics are never silently changed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise NotCheckable("spec malformed") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise NotCheckable("spec malformed")
    func = tree.body[0]
    if func.type_params:
        raise NotCheckable("spec malformed")
    args = func.args
    if args.posonlyargs or args.vararg is not None or args.kwonlyargs or args.kwarg is not None:
        raise NotCheckable("spec malformed")
    if args.defaults or args.kw_defaults:
        raise NotCheckable("spec malformed")
    params = [arg.arg for arg in args.args]
    annotations = {arg.arg: (ast.unparse(arg.annotation) if arg.annotation is not None else None) for arg in args.args}
    rendered = ", ".join(
        f"{arg.arg}: {annotations[arg.arg]}" if annotations[arg.arg] is not None else arg.arg for arg in args.args
    )
    display = f"{func.name}({rendered})"
    if func.returns is not None:
        display += f" -> {ast.unparse(func.returns)}"
    return display, params, annotations


def _parse(
    text: str, intent: Intent, params: list[str], annotations: dict[str, str | None]
) -> tuple[str, str | None, tuple[Claim, ...]]:
    """Validate one model answer; return the predicate, the precondition and the cited claims."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Invalid(f"answer is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise _Invalid(f"answer must be a JSON object, got {type(data).__name__}")
    if set(data) != {"predicate", "precondition", "precondition_sources"}:
        raise _Invalid("answer must hold exactly the keys predicate, precondition and precondition_sources")
    predicate = data["predicate"]
    if not isinstance(predicate, str):
        raise _Invalid("predicate must be a string")
    precondition = data["precondition"]
    if not isinstance(precondition, str) and precondition is not None:
        raise _Invalid("precondition must be a string or null")
    sources = data["precondition_sources"]
    if not isinstance(sources, list):
        raise _Invalid("precondition_sources must be a list")
    _check_function(predicate, "spec", [*params, "result"], annotations)
    if precondition is not None:
        _check_function(precondition, "pre", params, annotations)
    return predicate, precondition, _resolve_sources(sources, intent, precondition is not None)


def _resolve_sources(sources: list[Any], intent: Intent, has_precondition: bool) -> tuple[Claim, ...]:
    """Map 1-based claim indices to claims in source order; anything else is a validation error."""
    if not has_precondition:
        if sources:
            raise _Invalid("precondition_sources must be empty without a precondition")
        return ()
    if not sources:
        raise _Invalid("a precondition needs at least one claim index")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in sources):
        raise _Invalid("precondition_sources must hold integers")
    if len(set(sources)) != len(sources):
        raise _Invalid("precondition_sources must not repeat an index")
    if any(index < 1 or index > len(intent.claims) for index in sources):
        raise _Invalid(f"precondition_sources must hold 1-based indices into C1 to C{len(intent.claims)}")
    return tuple(intent.claims[index - 1] for index in sorted(sources))


def _check_function(source: str, name: str, expected: list[str], annotations: dict[str, str | None]) -> None:
    """Check that ``source`` is exactly one ``def <name>`` with the expected parameters and one ``return``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise _Invalid(f"{name} does not parse: {exc}") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise _Invalid(f"{name} must be exactly one `def {name}`")
    func = tree.body[0]
    if func.name != name:
        raise _Invalid(f"expected `def {name}`, got `def {func.name}`")
    if func.decorator_list or func.type_params:
        raise _Invalid(f"`def {name}` takes no decorators or type parameters")
    func_args = func.args
    if (
        func_args.posonlyargs
        or func_args.vararg is not None
        or func_args.kwonlyargs
        or func_args.kwarg is not None
        or func_args.defaults
        or func_args.kw_defaults
    ):
        raise _Invalid(f"`def {name}` takes plain parameters only")
    if [arg.arg for arg in func_args.args] != expected:
        joined = ", ".join(expected)
        raise _Invalid(f"{name} parameters must be ({joined})")
    body = list(func.body)
    if body and _is_docstring(body[0]):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise _Invalid(f"`def {name}` holds an optional docstring plus exactly one `return` of an expression")
    _check_expr(body[0].value, set(expected), annotations)


def _is_docstring(statement: ast.stmt) -> bool:
    """Whether ``statement`` is a leading docstring, which a predicate may carry."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _check_expr(node: ast.expr, scope: set[str], annotations: dict[str, str | None]) -> None:
    """Check that an expression uses only ``SPEC_CONSTRUCTS``; anything else is a validation error."""
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, bool | int | str):
            return
        raise _Invalid(f"unsupported constant of type {type(node.value).__name__}")
    if isinstance(node, ast.Name):
        if node.id not in scope:
            raise _Invalid(f"unknown name `{node.id}`")
        return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And | ast.Or):
            raise _Invalid(f"unsupported construct `{type(node.op).__name__}`")
        for value in node.values:
            _check_expr(value, scope, annotations)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not | ast.UAdd | ast.USub):
            raise _Invalid(f"unsupported construct `{type(node.op).__name__}`")
        _check_expr(node.operand, scope, annotations)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.FloorDiv | ast.Mod):
            raise _Invalid(f"unsupported construct `{type(node.op).__name__}`")
        _check_expr(node.left, scope, annotations)
        _check_expr(node.right, scope, annotations)
        return
    if isinstance(node, ast.Compare):
        _check_compare(node, scope, annotations)
        return
    if isinstance(node, ast.IfExp):
        _check_expr(node.test, scope, annotations)
        _check_expr(node.body, scope, annotations)
        _check_expr(node.orelse, scope, annotations)
        return
    if isinstance(node, ast.Call):
        _check_call(node, scope, annotations)
        return
    if isinstance(node, ast.Subscript):
        _check_expr(node.value, scope, annotations)
        _check_slice(node.slice, scope, annotations)
        return
    if isinstance(node, ast.List | ast.Tuple):
        for element in node.elts:
            _check_expr(element, scope, annotations)
        return
    if isinstance(node, ast.Slice):
        for bound in (node.lower, node.upper, node.step):
            if bound is not None:
                _check_expr(bound, scope, annotations)
        return
    raise _Invalid(f"unsupported construct `{type(node).__name__}`")


def _check_compare(node: ast.Compare, scope: set[str], annotations: dict[str, str | None]) -> None:
    """Check comparison operators, enforcing that `in` needs a list on the right."""
    for op, comparator in zip(node.ops, node.comparators, strict=True):
        if isinstance(op, ast.In | ast.NotIn) and not _is_list_expr(comparator, annotations):
            raise _Invalid("`in` needs a list on the right; `in` on `str` is not supported")
        if not isinstance(op, ast.Eq | ast.NotEq | ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.In | ast.NotIn):
            raise _Invalid(f"unsupported construct `{type(op).__name__}`")
    _check_expr(node.left, scope, annotations)
    for comparator in node.comparators:
        _check_expr(comparator, scope, annotations)


def _is_list_expr(node: ast.expr, annotations: dict[str, str | None]) -> bool:
    """Whether ``node`` is statically a list: a list display or a name with a list hint."""
    if isinstance(node, ast.List):
        return True
    return isinstance(node, ast.Name) and _is_list_annotation(annotations.get(node.id))


def _is_list_annotation(annotation: str | None) -> bool:
    """Whether a signature hint names a list; unknown types do not count, so they are rejected."""
    return annotation is not None and _LIST_HINT.search(annotation) is not None


def _check_call(node: ast.Call, scope: set[str], annotations: dict[str, str | None]) -> None:
    """Check that a call is ``len``, ``range``, ``all`` or ``any`` with fitting arguments."""
    func = node.func
    name = func.id if isinstance(func, ast.Name) else ""
    if name not in ("len", "range", "all", "any"):
        raise _Invalid("only `len`, `range`, `all` and `any` calls are supported")
    if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
        raise _Invalid("calls take plain positional arguments only")
    if name in ("all", "any"):
        if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
            raise _Invalid(f"`{name}` takes one generator expression")
        _check_genexp(node.args[0], scope, annotations)
    elif name == "len":
        if len(node.args) != 1:
            raise _Invalid("`len` takes one argument")
        _check_expr(node.args[0], scope, annotations)
    elif not 1 <= len(node.args) <= 3:
        raise _Invalid("`range` takes one to three arguments")
    else:
        for arg in node.args:
            _check_expr(arg, scope, annotations)


def _check_genexp(node: ast.GeneratorExp, scope: set[str], annotations: dict[str, str | None]) -> None:
    """Check a generator expression of ``all``/``any`` over ``range(...)`` or a sequence."""
    if len(node.generators) != 1:
        raise _Invalid("generator expressions take exactly one `for`")
    generator = node.generators[0]
    if generator.is_async:
        raise _Invalid("async generator expressions are not supported")
    if not isinstance(generator.target, ast.Name):
        raise _Invalid("the loop variable must be a plain name")
    _check_iterable(generator.iter, scope, annotations)
    extended = scope | {generator.target.id}
    for condition in generator.ifs:
        _check_expr(condition, extended, annotations)
    _check_expr(node.elt, extended, annotations)


def _check_iterable(node: ast.expr, scope: set[str], annotations: dict[str, str | None]) -> None:
    """Check what a generator expression iterates over: ``range(...)`` or a sequence."""
    _check_expr(node, scope, annotations)
    if isinstance(node, ast.Call | ast.List | ast.Tuple | ast.Name | ast.Subscript):
        return
    raise _Invalid("generator expressions iterate over `range(...)` or a sequence")


def _check_slice(node: ast.expr, scope: set[str], annotations: dict[str, str | None]) -> None:
    """Check an index or slice; multi-dimensional indexing is outside the subset."""
    if isinstance(node, ast.Tuple):
        raise _Invalid("unsupported construct `Tuple`")
    _check_expr(node, scope, annotations)


__all__ = ["write"]
