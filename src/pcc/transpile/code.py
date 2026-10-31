"""Step 4, code mode: the rule-based transpiler for executable Dafny (docs/plan.md, step 4).

What the transpiler supports, it translates exactly; anything else raises
``UnsupportedConstruct`` and the pipeline falls back to AI translation.

Supported subset (this ticket; spec mode lives in ``pcc.transpile.spec``):

- Types: ``int`` to ``int``, ``bool`` to ``bool``, ``str`` to ``string``, ``list[T]``
  and ``typing.List[T]`` to ``seq<T>`` (only when never shared; see below). ``tuple``
  and ``typing.Tuple`` raise ``"type tuple"``; anything else raises ``"type <T>"``;
  missing hints raise ``"no types"``. The mapping lives in ``pcc.transpile.types``.
- Statements: ``if``/``elif``/``else``, ``return e``, local assignment ``x = e``
  (``var x := e;`` the first time, ``x := e;`` after), ``for i in range(a[, b])``
  (a ``while`` loop with a counter; a step raises ``"range step"``),
  ``for x in xs`` over a ``seq`` (an index loop), and ``while`` (a plain loop, or
  ``while true`` with hoisted calls when the condition calls the function itself or
  a project callee). A docstring as the first statement is skipped.
  Tuple assignment ``a, b = e1, e2`` becomes ``var a, b := e1, e2;`` the first
  time and ``a, b := e1, e2;`` after. ``x += e`` on ints becomes ``x := x + e;``.
  ``xs[i] = e`` on a local list becomes ``xs := py_set_index(xs, i, e);`` (and
  ``xs[i] += e`` reads the element back first); on a parameter it raises
  ``"shared list"``.
- A list that is aliased (``ys = xs``) or passed to a project callee that is not
  certified read-only raises ``"shared list"``. Passing a list to the function
  itself or to a callee in ``read_only_callees`` is allowed (a value in Dafny).
- Parameter and local names that are Dafny keywords or reserved (``seq``, ``set``,
  ``map``, ``string``, ``type``, ``method``, ``function``, ``var``, ``int``, ``nat``,
  ``real``, ``object``, ``array``, ``iset``, ``imap``, ``multiset``, ``char``,
  ``bool``, ``result``) are renamed to ``<name>_`` everywhere.
- Expressions: int literals, ``True``/``False``, string literals without surrogates;
  ``+``, ``-``, ``*`` on int; ``//`` to ``py_floordiv``, ``%`` to ``py_mod``; unary
  minus ``-x`` on int; comparisons on int; ``==`` and ``!=`` on ``str`` and ``list``;
  ``and``, ``or`` and ``not`` on bool operands only; ``len(x)`` to ``py_len(x)``;
  ``x[i]`` to ``py_index(x, i)`` (on ``str`` this yields a Dafny ``char``; ``==`` and
  ``!=`` between two chars are allowed, char against string raises
  ``"char vs str"``); ``x[a:b]`` to ``py_slice(x, a, b)`` with an omitted ``a`` to
  ``0`` and an omitted ``b`` to ``py_len(x)``; ``x[::-1]`` to ``py_reversed(x)``;
  ``+`` on ``str`` and ``list`` to ``+``; ``a if c else b`` to ``if c then a else b``;
  ``[e for i in range(n)]`` to ``seq(n, i => e)``; ``min(a, b)`` on ints to
  ``py_min(a, b)``; ``sum(<bool> for v in xs)`` to ``py_count(xs, v => <bool>)``.
- Whole-variable rebinding of a list (``xs = expression``) is supported when the
  inferred type stays compatible. Slice stores, ``AugAssign`` on whole lists
  (including ``+=``) and calls to mutating methods (``append``, ``extend``,
  ``insert``, ``pop``, ``remove``, ``clear``, ``sort``, ``reverse``) raise
  ``"mutated list"``.
- Recursive calls to the function itself are hoisted into fresh locals
  ``var pcc_t1 := f(...);``, ``pcc_t2``, ... in evaluation order before the
  statement that uses them. Under ``return``, a call in the right operand of
  ``and``/``or`` or in a branch of ``a if c else b`` is expanded into an ``if``
  statement preserving short-circuit order; elsewhere it raises
  ``"call in short-circuit"``. Calls to project functions use
  ``callees[qualname]`` as ``<Module>.<name>(...)``; they stay inline except under
  ``return`` short-circuit expansion, where they bind a fresh ``pcc_tN`` local.

Not supported: ``<`` (and friends) on strings, anything on ``float``, ``/``,
arithmetic on bools, string methods, ``raise``, ``try``,
``with``, ``yield``, ``await``, ``global`` and ``nonlocal``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pcc.core import FunctionRef, UnsupportedConstruct
from pcc.transpile.types import dafny_type
from pcc.verifier.layout import method_name

#: Python names that would collide with a Dafny keyword or reserved word. They are
#: renamed to ``<name>_`` everywhere (parameters and locals). ``result`` is reserved
#: for the return value.
RESERVED = frozenset(
    {
        "seq",
        "set",
        "map",
        "string",
        "type",
        "method",
        "function",
        "var",
        "int",
        "nat",
        "real",
        "object",
        "array",
        "iset",
        "imap",
        "multiset",
        "char",
        "bool",
        "result",
    }
)

#: Method names that mutate a list in place. A call to one of them on a list-typed
#: value raises ``"mutated list"``; mutation support itself lives in a later ticket.
MUTATING_METHODS = frozenset({"append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse"})

_INT = "int"
_BOOL = "bool"
_STR = "string"
_CHAR = "char"


def _is_seq(tag: str) -> bool:
    """Whether ``tag`` is a ``seq<...>`` type."""
    return tag.startswith("seq<") and tag.endswith(">")


def _seq_elem(tag: str) -> str:
    """The element type of the ``seq<...>`` type ``tag``."""
    return tag[len("seq<") : -1]


@dataclass(frozen=True, slots=True)
class _Typed:
    """A translated expression with its Dafny type tag, or ``None`` when unknown."""

    code: str
    tag: str | None


class _Ctx:
    """Translation state: renames plus the live inferred type environment.

    ``params`` holds the Python names of the parameters (never mutable when they
    hold a list). ``func_name`` is the Python name of the function itself.
    ``callees`` maps each project callee qualname to its Dafny module name,
    ``callee_types`` maps a callee qualname to its Dafny return type tag, and
    ``read_only`` holds the qualnames certified not to mutate or escape a list.
    """

    __slots__ = ("callee_types", "callees", "func_name", "params", "read_only", "renames", "types")

    def __init__(
        self,
        renames: Mapping[str, str],
        types: dict[str, str],
        *,
        params: frozenset[str] = frozenset(),
        func_name: str = "",
        callees: Mapping[str, str] = MappingProxyType({}),
        callee_types: Mapping[str, str] = MappingProxyType({}),
        read_only: frozenset[str] = frozenset(),
    ) -> None:
        self.renames = renames
        self.types = types
        self.params = params
        self.func_name = func_name
        self.callees = callees
        self.callee_types = callee_types
        self.read_only = read_only

    def child(self) -> _Ctx:
        """A context sharing everything but the type environment, which is copied."""
        return _Ctx(
            self.renames,
            dict(self.types),
            params=self.params,
            func_name=self.func_name,
            callees=self.callees,
            callee_types=self.callee_types,
            read_only=self.read_only,
        )

    def name(self, python_name: str) -> str:
        """The Dafny name of the Python variable ``python_name``."""
        return self.renames.get(python_name, python_name)

    def tag_of(self, python_name: str) -> str | None:
        """The Dafny type tag of ``python_name``, or ``None`` when unknown."""
        return self.types.get(python_name)


class _HoistState:
    """Hoisting state: fresh ``pcc_tN`` and ``pcc_c`` names plus the recursive callee."""

    __slots__ = ("callee", "cond_counter", "counter", "func_name", "return_tag", "used")

    def __init__(self, func_name: str, callee: str, return_tag: str, used: set[str]) -> None:
        self.func_name = func_name
        self.callee = callee
        self.return_tag = return_tag
        self.counter = 1
        self.cond_counter = 1
        self.used = set(used)

    def fresh(self) -> str:
        """A fresh ``pcc_tN`` name that collides with no user name."""
        while f"pcc_t{self.counter}" in self.used:
            self.counter += 1
        name = f"pcc_t{self.counter}"
        self.used.add(name)
        self.counter += 1
        return name

    def fresh_cond(self) -> str:
        """A fresh ``pcc_c`` name for a hoisted ``while`` condition call."""
        while True:
            name = "pcc_c" if self.cond_counter == 1 else f"pcc_c{self.cond_counter}"
            self.cond_counter += 1
            if name not in self.used:
                self.used.add(name)
                return name


def _is_self_call(node: ast.Call, func_name: str, renames: Mapping[str, str]) -> bool:
    """Whether ``node`` calls the function itself (shadowing excluded)."""
    func = node.func
    return isinstance(func, ast.Name) and func.id == func_name and func.id not in renames


def _contains_self_call(node: ast.AST, func_name: str, renames: Mapping[str, str]) -> bool:
    """Whether ``node`` contains a call to the function itself."""
    if func_name in renames:
        return False
    return any(isinstance(child, ast.Call) and _is_self_call(child, func_name, renames) for child in ast.walk(node))


def _has_sc_self_call(node: ast.expr, func_name: str, renames: Mapping[str, str]) -> bool:
    """Whether a self-call sits in short-circuit position (right of ``and``/``or`` or branch of ``if``)."""

    def visit(expr: ast.AST, in_sc: bool) -> bool:
        if isinstance(expr, ast.Call) and _is_self_call(expr, func_name, renames):
            return in_sc
        if isinstance(expr, ast.BoolOp):
            return any(visit(value, in_sc or index > 0) for index, value in enumerate(expr.values))
        if isinstance(expr, ast.IfExp):
            if visit(expr.test, in_sc):
                return True
            return visit(expr.body, True) or visit(expr.orelse, True)
        return any(visit(child, in_sc) for child in ast.iter_child_nodes(expr))

    if func_name in renames:
        return False
    return visit(node, False)


def _has_call_inside_boolop(node: ast.AST, func_name: str, renames: Mapping[str, str]) -> bool:
    """Whether a self-call sits anywhere inside ``and``/``or`` or a conditional."""

    def visit(expr: ast.AST, inside: bool) -> bool:
        if isinstance(expr, ast.Call) and _is_self_call(expr, func_name, renames):
            return inside
        if isinstance(expr, (ast.BoolOp, ast.IfExp)):
            return any(visit(child, True) for child in ast.iter_child_nodes(expr))
        return any(visit(child, inside) for child in ast.iter_child_nodes(expr))

    if func_name in renames:
        return False
    return visit(node, False)


def _project_key(name: str, ctx: _Ctx) -> str | None:
    """The single callee qualname that ``name`` refers to, or ``None``.

    ``None`` means no callee matches (an unknown call) or several do (ambiguous);
    both raise ``"call <name>"``. A name bound to a local value is never a callee.
    """
    if name in ctx.types:
        return None
    matches = [key for key in ctx.callees if key == name or key.split(".")[-1] == name]
    if len(matches) != 1:
        return None
    return matches[0]


def _is_project_call(node: ast.Call, ctx: _Ctx) -> bool:
    """Whether ``node`` calls a project callee (not the function itself)."""
    func = node.func
    if not isinstance(func, ast.Name):
        return False
    name = func.id
    if name in ("len", "min", "sum", "range"):
        return False
    if name == ctx.func_name and name not in ctx.renames:
        return False
    return _project_key(name, ctx) is not None


def _contains_proj_call(node: ast.AST, ctx: _Ctx) -> bool:
    """Whether ``node`` contains a call to a project callee."""
    return any(isinstance(child, ast.Call) and _is_project_call(child, ctx) for child in ast.walk(node))


def _has_sc_proj_call(node: ast.expr, ctx: _Ctx) -> bool:
    """Whether a project call sits in short-circuit position (mirrors the self-call check)."""

    def visit(expr: ast.AST, in_sc: bool) -> bool:
        if isinstance(expr, ast.Call) and _is_project_call(expr, ctx):
            return in_sc
        if isinstance(expr, ast.BoolOp):
            return any(visit(value, in_sc or index > 0) for index, value in enumerate(expr.values))
        if isinstance(expr, ast.IfExp):
            if visit(expr.test, in_sc):
                return True
            return visit(expr.body, True) or visit(expr.orelse, True)
        return any(visit(child, in_sc) for child in ast.iter_child_nodes(expr))

    return visit(node, False)


def _has_proj_inside_boolop(node: ast.AST, ctx: _Ctx) -> bool:
    """Whether a project call sits anywhere inside ``and``/``or`` or a conditional."""

    def visit(expr: ast.AST, inside: bool) -> bool:
        if isinstance(expr, ast.Call) and _is_project_call(expr, ctx):
            return inside
        if isinstance(expr, (ast.BoolOp, ast.IfExp)):
            return any(visit(child, True) for child in ast.iter_child_nodes(expr))
        return any(visit(child, inside) for child in ast.iter_child_nodes(expr))

    return visit(node, False)


def _contains_any_call(node: ast.AST, func_name: str, ctx: _Ctx) -> bool:
    """Whether ``node`` calls the function itself or a project callee."""
    return _contains_self_call(node, func_name, ctx.renames) or _contains_proj_call(node, ctx)


def _has_sc_any_call(node: ast.expr, func_name: str, ctx: _Ctx) -> bool:
    """Whether a self or project call sits in short-circuit position."""
    return _has_sc_self_call(node, func_name, ctx.renames) or _has_sc_proj_call(node, ctx)


def _has_any_inside_boolop(node: ast.AST, func_name: str, ctx: _Ctx) -> bool:
    """Whether a self or project call sits inside ``and``/``or`` or a conditional."""
    return _has_call_inside_boolop(node, func_name, ctx.renames) or _has_proj_inside_boolop(node, ctx)


def _fail(node: ast.stmt | ast.expr, construct: str) -> UnsupportedConstruct:
    """The error for ``node``; its lineno is relative to the function source."""
    return UnsupportedConstruct(construct, lineno=node.lineno)


def _need(tag: str | None, node: ast.stmt | ast.expr) -> str:
    """``tag``, or raise when the type of ``node`` cannot be established."""
    if tag is None:
        raise _fail(node, "unknown type")
    return tag


def _escape_string(value: str, node: ast.expr) -> str:
    """``value`` as a Dafny string literal; lone surrogates are rejected."""
    parts = ['"']
    for char in value:
        point = ord(char)
        if 0xD800 <= point <= 0xDFFF:
            raise _fail(node, "lone surrogate")
        if char == '"':
            parts.append('\\"')
        elif char == "\\":
            parts.append("\\\\")
        elif char == "\n":
            parts.append("\\n")
        elif char == "\r":
            parts.append("\\r")
        elif char == "\t":
            parts.append("\\t")
        elif 0x20 <= point <= 0x7E or point > 0xFFFF:
            parts.append(char)
        else:
            parts.append(f"\\u{point:04X}")
    parts.append('"')
    return "".join(parts)


def _maybe_paren(code: str, node: ast.expr) -> str:
    """``code`` in parentheses when ``node`` binds looser than its context."""
    if isinstance(node, (ast.BoolOp, ast.Compare, ast.IfExp)):
        return f"({code})"
    return code


def _operand(code: str, node: ast.expr) -> str:
    """``code`` in parentheses when ``node`` is not atomic for an arithmetic context."""
    if isinstance(node, (ast.BoolOp, ast.Compare, ast.IfExp, ast.BinOp)):
        return f"({code})"
    return code


def translate_expr(
    node: ast.expr,
    *,
    renames: Mapping[str, str],
    types: Mapping[str, str] | None = None,
) -> str:
    """Translate the expression ``node`` to Dafny.

    ``renames`` maps Python variable names to their Dafny names (reserved words
    become ``<name>_``); ``transpile_function`` builds it for the whole function.
    ``types`` optionally maps Python variable names to their Dafny type tags
    (``"int"``, ``"bool"``, ``"string"``, ``"char"``, ``"seq<...>"``); the spec
    mode ticket reuses this function for spec expressions (including ``result``)
    by passing its own environment, without editing this module. Where a required
    type cannot be established, this raises ``UnsupportedConstruct`` instead of
    guessing.
    """
    return _expr(node, _Ctx(renames, dict(types) if types is not None else {})).code


def _expr(node: ast.expr, ctx: _Ctx) -> _Typed:
    """Translate ``node`` to Dafny code with its type tag."""
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.Name):
        return _Typed(ctx.name(node.id), ctx.tag_of(node.id))
    if isinstance(node, ast.BinOp):
        return _binop(node, ctx)
    if isinstance(node, ast.UnaryOp):
        return _unaryop(node, ctx)
    if isinstance(node, ast.BoolOp):
        return _boolop(node, ctx)
    if isinstance(node, ast.Compare):
        return _compare(node, ctx)
    if isinstance(node, ast.Call):
        return _call(node, ctx)
    if isinstance(node, ast.Subscript):
        return _subscript(node, ctx)
    if isinstance(node, ast.IfExp):
        return _ifexp(node, ctx)
    if isinstance(node, ast.List):
        return _list_display(node, ctx)
    if isinstance(node, ast.Attribute):
        raise _fail(node, "attribute access")
    if isinstance(node, ast.Tuple):
        raise _fail(node, "tuple literal")
    if isinstance(node, ast.Dict):
        raise _fail(node, "dict literal")
    if isinstance(node, ast.Set):
        raise _fail(node, "set literal")
    if isinstance(node, ast.ListComp):
        return _comprehension(node, ctx, _expr)
    if isinstance(node, (ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        raise _fail(node, "comprehension")
    if isinstance(node, ast.Lambda):
        raise _fail(node, "lambda")
    if isinstance(node, ast.NamedExpr):
        raise _fail(node, "walrus")
    if isinstance(node, ast.JoinedStr):
        raise _fail(node, "f-string")
    if isinstance(node, ast.Starred):
        raise _fail(node, "starred")
    if isinstance(node, ast.Await):
        raise _fail(node, "await")
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
        raise _fail(node, "yield")
    if isinstance(node, ast.FormattedValue):
        raise _fail(node, "f-string")
    raise _fail(node, f"expression {type(node).__name__}")


def _constant(node: ast.Constant) -> _Typed:
    """Translate a literal."""
    value = node.value
    if isinstance(value, bool):
        return _Typed("true" if value else "false", _BOOL)
    if isinstance(value, int):
        return _Typed(str(value), _INT)
    if isinstance(value, str):
        return _Typed(_escape_string(value, node), _STR)
    if isinstance(value, float):
        raise _fail(node, "float literal")
    if isinstance(value, complex):
        raise _fail(node, "complex literal")
    if isinstance(value, bytes):
        raise _fail(node, "bytes literal")
    if value is None:
        raise _fail(node, "none literal")
    if value is Ellipsis:
        raise _fail(node, "ellipsis")
    raise _fail(node, f"literal {type(value).__name__}")


def _binop(node: ast.BinOp, ctx: _Ctx) -> _Typed:
    """Translate ``+``/``-``/``*``/``//``/``%`` and reject the other operators."""
    op = node.op
    if isinstance(op, ast.Div):
        raise _fail(node, "operator /")
    if isinstance(op, ast.Pow):
        raise _fail(node, "operator **")
    if isinstance(op, ast.BitAnd):
        raise _fail(node, "operator &")
    if isinstance(op, ast.BitOr):
        raise _fail(node, "operator |")
    if isinstance(op, ast.BitXor):
        raise _fail(node, "operator ^")
    if isinstance(op, ast.LShift):
        raise _fail(node, "operator <<")
    if isinstance(op, ast.RShift):
        raise _fail(node, "operator >>")
    if isinstance(op, ast.MatMult):
        raise _fail(node, "operator @")
    left = _expr(node.left, ctx)
    right = _expr(node.right, ctx)
    left_tag = _need(left.tag, node.left)
    right_tag = _need(right.tag, node.right)
    if left_tag == _BOOL or right_tag == _BOOL:
        raise _fail(node, "bool arithmetic")
    symbol = "+" if isinstance(op, ast.Add) else ("-" if isinstance(op, ast.Sub) else "*")
    code = f"{_operand(left.code, node.left)} {symbol} {_operand(right.code, node.right)}"
    if isinstance(op, ast.Add):
        if left_tag == _INT and right_tag == _INT:
            return _Typed(code, _INT)
        if left_tag == _STR and right_tag == _STR:
            return _Typed(code, _STR)
        if _is_seq(left_tag) and left_tag == right_tag:
            return _Typed(code, left_tag)
        raise _fail(node, "operand types")
    if isinstance(op, (ast.Sub, ast.Mult)):
        if left_tag == _INT and right_tag == _INT:
            return _Typed(code, _INT)
        raise _fail(node, "operand types")
    if isinstance(op, ast.FloorDiv):
        if left_tag == _INT and right_tag == _INT:
            return _Typed(f"py_floordiv({left.code}, {right.code})", _INT)
        raise _fail(node, "operand types")
    if left_tag == _INT and right_tag == _INT:
        return _Typed(f"py_mod({left.code}, {right.code})", _INT)
    raise _fail(node, "operand types")


def _unaryop(node: ast.UnaryOp, ctx: _Ctx) -> _Typed:
    """Translate unary minus and ``not``."""
    op = node.op
    if isinstance(op, ast.UAdd):
        raise _fail(node, "unary +")
    if isinstance(op, ast.Invert):
        raise _fail(node, "unary ~")
    operand = _expr(node.operand, ctx)
    tag = _need(operand.tag, node.operand)
    if isinstance(op, ast.USub):
        if tag == _BOOL:
            raise _fail(node, "bool arithmetic")
        if tag != _INT:
            raise _fail(node, "operand types")
        return _Typed(f"-{_operand(operand.code, node.operand)}", _INT)
    if tag != _BOOL:
        raise _fail(node, "non-bool logic")
    return _Typed(f"!{_maybe_paren(operand.code, node.operand)}", _BOOL)


def _boolop(node: ast.BoolOp, ctx: _Ctx) -> _Typed:
    """Translate ``and``/``or`` on bool operands only."""
    parts = [_expr(value, ctx) for value in node.values]
    for value, part in zip(node.values, parts, strict=True):
        if _need(part.tag, value) != _BOOL:
            raise _fail(node, "non-bool logic")
    joiner = " && " if isinstance(node.op, ast.And) else " || "
    codes = [_maybe_paren(part.code, value) for part, value in zip(parts, node.values, strict=True)]
    return _Typed(joiner.join(codes), _BOOL)


def _compare_op_name(op: ast.cmpop) -> str:
    """The source symbol of a comparison operator."""
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    if isinstance(op, ast.In):
        return "in"
    if isinstance(op, ast.NotIn):
        return "not in"
    if isinstance(op, ast.Is):
        return "is"
    return "is not"


def _compare(node: ast.Compare, ctx: _Ctx) -> _Typed:
    """Translate one comparison; chained comparisons are rejected."""
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise _fail(node, "chained comparison")
    op = node.ops[0]
    symbol = _compare_op_name(op)
    if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
        raise _fail(node, f"operator {symbol}")
    left = _expr(node.left, ctx)
    right = _expr(node.comparators[0], ctx)
    left_tag = _need(left.tag, node.left)
    right_tag = _need(right.tag, node.comparators[0])
    code = f"{_maybe_paren(left.code, node.left)} {symbol} {_maybe_paren(right.code, node.comparators[0])}"
    if isinstance(op, (ast.Eq, ast.NotEq)):
        if left_tag == right_tag:
            return _Typed(code, _BOOL)
        if {left_tag, right_tag} == {_CHAR, _STR}:
            raise _fail(node, "char vs str")
        raise _fail(node, "operand types")
    if left_tag == _INT and right_tag == _INT:
        return _Typed(code, _BOOL)
    if left_tag == _STR and right_tag == _STR:
        raise _fail(node, f"str comparison {symbol}")
    if {left_tag, right_tag} == {_CHAR, _STR}:
        raise _fail(node, "char vs str")
    if left_tag == _BOOL or right_tag == _BOOL:
        raise _fail(node, "bool comparison")
    if left_tag == _CHAR and right_tag == _CHAR:
        raise _fail(node, "char comparison")
    if _is_seq(left_tag) and left_tag == right_tag:
        raise _fail(node, "list comparison")
    raise _fail(node, "operand types")


def _min_call(node: ast.Call, ctx: _Ctx, sub: Callable[[ast.expr, _Ctx], _Typed]) -> _Typed:
    """``min(a, b)`` on exactly two ints to ``py_min(a, b)``."""
    if node.keywords or len(node.args) != 2:
        raise _fail(node, "min arguments")
    args = [sub(arg, ctx) for arg in node.args]
    for arg, original in zip(args, node.args, strict=True):
        if _need(arg.tag, original) != _INT:
            raise _fail(node, "min arguments")
    return _Typed(f"py_min({args[0].code}, {args[1].code})", _INT)


def _range_single_bound(call: ast.Call) -> ast.expr | None:
    """The single ``range(n)`` bound in a comprehension, or ``None``."""
    if call.keywords or len(call.args) != 1:
        return None
    return call.args[0]


def _comprehension(node: ast.ListComp, ctx: _Ctx, sub: Callable[[ast.expr, _Ctx], _Typed]) -> _Typed:
    """``[e for i in range(n)]`` to ``seq(n, i => e)`` with ``sub`` for the parts."""
    if len(node.generators) != 1:
        raise _fail(node, "comprehension")
    gen = node.generators[0]
    if gen.is_async or gen.ifs or not isinstance(gen.target, ast.Name):
        raise _fail(node, "comprehension")
    if not isinstance(gen.iter, ast.Call):
        raise _fail(node, "comprehension")
    iter_func = gen.iter.func
    if (
        not isinstance(iter_func, ast.Name)
        or iter_func.id != "range"
        or "range" in ctx.types
        or (bound_node := _range_single_bound(gen.iter)) is None
    ):
        raise _fail(node, "comprehension")
    bound = sub(bound_node, ctx)
    if _need(bound.tag, bound_node) != _INT:
        raise _fail(node, "comprehension")
    child = ctx.child()
    child.types[gen.target.id] = _INT
    elt = sub(node.elt, child)
    elt_tag = _need(elt.tag, node.elt)
    return _Typed(f"seq({bound.code}, {child.name(gen.target.id)} => {elt.code})", f"seq<{elt_tag}>")


def _sum_call(node: ast.Call, ctx: _Ctx, sub: Callable[[ast.expr, _Ctx], _Typed]) -> _Typed:
    """``sum(<bool> for v in xs)`` to ``py_count(xs, v => <bool>)``."""
    if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
        raise _fail(node, "sum generator")
    gen = node.args[0]
    if len(gen.generators) != 1:
        raise _fail(node, "sum generator")
    comp = gen.generators[0]
    if comp.is_async or comp.ifs or not isinstance(comp.target, ast.Name):
        raise _fail(node, "sum generator")
    seq_value = sub(comp.iter, ctx)
    seq_tag = _need(seq_value.tag, comp.iter)
    if not _is_seq(seq_tag):
        raise _fail(node, "sum generator")
    child = ctx.child()
    child.types[comp.target.id] = _seq_elem(seq_tag)
    elt = sub(gen.elt, child)
    if _need(elt.tag, gen.elt) != _BOOL:
        raise _fail(node, "sum generator")
    return _Typed(f"py_count({seq_value.code}, {child.name(comp.target.id)} => {elt.code})", _INT)


def _project_call(
    node: ast.Call, ctx: _Ctx, key: str, sub: Callable[[ast.expr, _Ctx], _Typed]
) -> tuple[str, str | None]:
    """The inline ``<Module>.<name>(args)`` code and return tag of a project call.

    A list-typed variable passed to a callee outside ``read_only_callees`` raises
    ``"shared list"``; anything but a plain argument list raises ``"call <name>"``.
    """
    name = node.func.id if isinstance(node.func, ast.Name) else "<call>"
    if node.keywords:
        raise _fail(node, f"call {name}")
    module = ctx.callees[key]
    codes: list[str] = []
    for arg in node.args:
        value = sub(arg, ctx)
        _need(value.tag, arg)
        if isinstance(arg, ast.Name) and _is_seq(value.tag or "") and key not in ctx.read_only:
            raise _fail(node, "shared list")
        codes.append(value.code)
    return f"{module}.{name}({', '.join(codes)})", _callee_tag(ctx, key)


def _callee_tag(ctx: _Ctx, key: str) -> str | None:
    """The Dafny return tag for the callee ``key``, or ``None`` when unknown.

    Looks up ``callee_types`` by qualname, falling back to the short name, so a
    caller that only knows ``g`` still finds the type recorded under ``pkg.g``.
    """
    tag = ctx.callee_types.get(key)
    if tag is None:
        short = key.split(".")[-1]
        for other, other_tag in ctx.callee_types.items():
            if other.split(".")[-1] == short:
                return other_tag
    return tag


def _call(node: ast.Call, ctx: _Ctx) -> _Typed:
    """Translate ``len``/``min``/``sum`` and project calls; anything else is rejected."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "len" and "len" not in ctx.types:
        if node.keywords or len(node.args) != 1:
            raise _fail(node, "len arguments")
        arg = _expr(node.args[0], ctx)
        tag = _need(arg.tag, node.args[0])
        if tag != _STR and not _is_seq(tag):
            raise _fail(node, "len operand")
        return _Typed(f"py_len({arg.code})", _INT)
    if isinstance(func, ast.Name) and func.id == "min" and "min" not in ctx.types:
        return _min_call(node, ctx, _expr)
    if isinstance(func, ast.Name) and func.id == "sum" and "sum" not in ctx.types:
        return _sum_call(node, ctx, _expr)
    if isinstance(func, ast.Name):
        name = func.id
        if _is_self_call(node, ctx.func_name, ctx.renames):
            raise _fail(node, f"call {name}")
        key = _project_key(name, ctx)
        if key is not None:
            code, call_tag = _project_call(node, ctx, key, _expr)
            return _Typed(code, call_tag)
        raise _fail(node, f"call {name}")
    if isinstance(func, ast.Attribute):
        value = _expr(func.value, ctx)
        if value.tag is None:
            raise _fail(node, "unknown type")
        if _is_seq(value.tag) and func.attr in MUTATING_METHODS:
            raise _fail(node, "mutated list")
        raise _fail(node, f"call {func.attr}")
    raise _fail(node, "call expression")


def _bound(node: ast.expr | None, ctx: _Ctx, default: str, whole: ast.expr) -> str:
    """Translate a slice bound; a missing bound becomes ``default``."""
    if node is None:
        return default
    bound = _expr(node, ctx)
    if _need(bound.tag, node) != _INT:
        raise _fail(whole, "slice bounds")
    return bound.code


def _is_negative_one(node: ast.expr) -> bool:
    """Whether ``node`` is the literal ``-1``."""
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
        and node.operand.value == 1
    )


def _subscript(node: ast.Subscript, ctx: _Ctx) -> _Typed:
    """Translate ``x[i]``, ``x[a:b]`` and ``x[::-1]``."""
    value = _expr(node.value, ctx)
    tag = _need(value.tag, node.value)
    if tag != _STR and not _is_seq(tag):
        raise _fail(node, "index type")
    elem = _CHAR if tag == _STR else _seq_elem(tag)
    subscript = node.slice
    if isinstance(subscript, ast.Slice):
        if subscript.step is not None:
            if subscript.lower is None and subscript.upper is None and _is_negative_one(subscript.step):
                return _Typed(f"py_reversed({value.code})", tag)
            raise _fail(node, "slice step")
        lower = _bound(subscript.lower, ctx, "0", node)
        upper = _bound(subscript.upper, ctx, f"py_len({value.code})", node)
        return _Typed(f"py_slice({value.code}, {lower}, {upper})", tag)
    index = _expr(subscript, ctx)
    if _need(index.tag, subscript) != _INT:
        raise _fail(node, "index type")
    return _Typed(f"py_index({value.code}, {index.code})", elem)


def _ifexp(node: ast.IfExp, ctx: _Ctx) -> _Typed:
    """Translate ``a if c else b``."""
    cond = _expr(node.test, ctx)
    if _need(cond.tag, node.test) != _BOOL:
        raise _fail(node, "non-bool condition")
    when_true = _expr(node.body, ctx)
    when_false = _expr(node.orelse, ctx)
    true_tag = _need(when_true.tag, node.body)
    false_tag = _need(when_false.tag, node.orelse)
    if true_tag != false_tag:
        if {true_tag, false_tag} == {_CHAR, _STR}:
            raise _fail(node, "char vs str")
        raise _fail(node, "operand types")
    body = when_true.code if not isinstance(node.body, ast.IfExp) else f"({when_true.code})"
    orelse = when_false.code if not isinstance(node.orelse, ast.IfExp) else f"({when_false.code})"
    cond_code = cond.code if not isinstance(node.test, ast.IfExp) else f"({cond.code})"
    return _Typed(f"if {cond_code} then {body} else {orelse}", true_tag)


def _list_display(node: ast.List, ctx: _Ctx) -> _Typed:
    """Translate a list display of equally typed elements."""
    if not node.elts:
        raise _fail(node, "empty list")
    elts = [_expr(elt, ctx) for elt in node.elts]
    tags = [_need(elt.tag, original) for elt, original in zip(elts, node.elts, strict=True)]
    if any(tag != tags[0] for tag in tags[1:]):
        raise _fail(node, "operand types")
    return _Typed(f"[{', '.join(elt.code for elt in elts)}]", f"seq<{tags[0]}>")


def _hoist_expr(node: ast.expr, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """Translate ``node``, hoisting self-calls into ``pre`` in evaluation order."""
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.Name):
        return _Typed(ctx.name(node.id), ctx.tag_of(node.id))
    if isinstance(node, ast.BinOp):
        return _hoist_binop(node, ctx, st, pre)
    if isinstance(node, ast.UnaryOp):
        return _hoist_unaryop(node, ctx, st, pre)
    if isinstance(node, ast.BoolOp):
        return _hoist_boolop(node, ctx, st, pre)
    if isinstance(node, ast.Compare):
        return _hoist_compare(node, ctx, st, pre)
    if isinstance(node, ast.Call):
        return _hoist_call(node, ctx, st, pre)
    if isinstance(node, ast.Subscript):
        return _hoist_subscript(node, ctx, st, pre)
    if isinstance(node, ast.IfExp):
        return _hoist_ifexp(node, ctx, st, pre)
    if isinstance(node, ast.List):
        return _hoist_list_display(node, ctx, st, pre)
    if isinstance(node, ast.ListComp):
        return _comprehension(node, ctx, lambda e, c: _hoist_expr(e, c, st, pre))
    return _expr(node, ctx)


def _hoist_binop(node: ast.BinOp, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """``+``/``-``/``*``/``//``/``%`` with hoisted operands."""
    op = node.op
    if isinstance(op, ast.Div):
        raise _fail(node, "operator /")
    if isinstance(op, ast.Pow):
        raise _fail(node, "operator **")
    if isinstance(op, ast.BitAnd):
        raise _fail(node, "operator &")
    if isinstance(op, ast.BitOr):
        raise _fail(node, "operator |")
    if isinstance(op, ast.BitXor):
        raise _fail(node, "operator ^")
    if isinstance(op, ast.LShift):
        raise _fail(node, "operator <<")
    if isinstance(op, ast.RShift):
        raise _fail(node, "operator >>")
    if isinstance(op, ast.MatMult):
        raise _fail(node, "operator @")
    left = _hoist_expr(node.left, ctx, st, pre)
    right = _hoist_expr(node.right, ctx, st, pre)
    left_tag = _need(left.tag, node.left)
    right_tag = _need(right.tag, node.right)
    if left_tag == _BOOL or right_tag == _BOOL:
        raise _fail(node, "bool arithmetic")
    symbol = "+" if isinstance(op, ast.Add) else ("-" if isinstance(op, ast.Sub) else "*")
    code = f"{_operand(left.code, node.left)} {symbol} {_operand(right.code, node.right)}"
    if isinstance(op, ast.Add):
        if left_tag == _INT and right_tag == _INT:
            return _Typed(code, _INT)
        if left_tag == _STR and right_tag == _STR:
            return _Typed(code, _STR)
        if _is_seq(left_tag) and left_tag == right_tag:
            return _Typed(code, left_tag)
        raise _fail(node, "operand types")
    if isinstance(op, (ast.Sub, ast.Mult)):
        if left_tag == _INT and right_tag == _INT:
            return _Typed(code, _INT)
        raise _fail(node, "operand types")
    if isinstance(op, ast.FloorDiv):
        if left_tag == _INT and right_tag == _INT:
            return _Typed(f"py_floordiv({left.code}, {right.code})", _INT)
        raise _fail(node, "operand types")
    if left_tag == _INT and right_tag == _INT:
        return _Typed(f"py_mod({left.code}, {right.code})", _INT)
    raise _fail(node, "operand types")


def _hoist_unaryop(node: ast.UnaryOp, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """Unary minus and ``not`` with a hoisted operand."""
    op = node.op
    if isinstance(op, ast.UAdd):
        raise _fail(node, "unary +")
    if isinstance(op, ast.Invert):
        raise _fail(node, "unary ~")
    operand = _hoist_expr(node.operand, ctx, st, pre)
    tag = _need(operand.tag, node.operand)
    if isinstance(op, ast.USub):
        if tag == _BOOL:
            raise _fail(node, "bool arithmetic")
        if tag != _INT:
            raise _fail(node, "operand types")
        return _Typed(f"-{_operand(operand.code, node.operand)}", _INT)
    if tag != _BOOL:
        raise _fail(node, "non-bool logic")
    return _Typed(f"!{_maybe_paren(operand.code, node.operand)}", _BOOL)


def _hoist_boolop(node: ast.BoolOp, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """``and``/``or`` on bool operands with hoisted values."""
    parts = [_hoist_expr(value, ctx, st, pre) for value in node.values]
    for value, part in zip(node.values, parts, strict=True):
        if _need(part.tag, value) != _BOOL:
            raise _fail(node, "non-bool logic")
    joiner = " && " if isinstance(node.op, ast.And) else " || "
    codes = [_maybe_paren(part.code, value) for part, value in zip(parts, node.values, strict=True)]
    return _Typed(joiner.join(codes), _BOOL)


def _hoist_compare(node: ast.Compare, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """One comparison with hoisted sides."""
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise _fail(node, "chained comparison")
    op = node.ops[0]
    symbol = _compare_op_name(op)
    if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
        raise _fail(node, f"operator {symbol}")
    left = _hoist_expr(node.left, ctx, st, pre)
    right = _hoist_expr(node.comparators[0], ctx, st, pre)
    left_tag = _need(left.tag, node.left)
    right_tag = _need(right.tag, node.comparators[0])
    code = f"{_maybe_paren(left.code, node.left)} {symbol} {_maybe_paren(right.code, node.comparators[0])}"
    if isinstance(op, (ast.Eq, ast.NotEq)):
        if left_tag == right_tag:
            return _Typed(code, _BOOL)
        if {left_tag, right_tag} == {_CHAR, _STR}:
            raise _fail(node, "char vs str")
        raise _fail(node, "operand types")
    if left_tag == _INT and right_tag == _INT:
        return _Typed(code, _BOOL)
    if left_tag == _STR and right_tag == _STR:
        raise _fail(node, f"str comparison {symbol}")
    if {left_tag, right_tag} == {_CHAR, _STR}:
        raise _fail(node, "char vs str")
    if left_tag == _BOOL or right_tag == _BOOL:
        raise _fail(node, "bool comparison")
    if left_tag == _CHAR and right_tag == _CHAR:
        raise _fail(node, "char comparison")
    if _is_seq(left_tag) and left_tag == right_tag:
        raise _fail(node, "list comparison")
    raise _fail(node, "operand types")


def _hoist_call(node: ast.Call, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """``len``/``min``/``sum`` inline; self and project calls hoisted to ``var pcc_tN``."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == st.func_name and func.id not in ctx.renames:
        if node.keywords:
            raise _fail(node, f"call {func.id}")
        arg_codes: list[str] = []
        for arg in node.args:
            hoisted = _hoist_expr(arg, ctx, st, pre)
            _need(hoisted.tag, arg)
            arg_codes.append(hoisted.code)
        temp = st.fresh()
        pre.append(f"var {temp} := {st.callee}({', '.join(arg_codes)});")
        return _Typed(temp, st.return_tag)
    if isinstance(func, ast.Name) and func.id == "len" and "len" not in ctx.types:
        if node.keywords or len(node.args) != 1:
            raise _fail(node, "len arguments")
        hoisted_arg = _hoist_expr(node.args[0], ctx, st, pre)
        tag = _need(hoisted_arg.tag, node.args[0])
        if tag != _STR and not _is_seq(tag):
            raise _fail(node, "len operand")
        return _Typed(f"py_len({hoisted_arg.code})", _INT)
    if isinstance(func, ast.Name) and func.id == "min" and "min" not in ctx.types:
        return _min_call(node, ctx, lambda e, c: _hoist_expr(e, c, st, pre))
    if isinstance(func, ast.Name) and func.id == "sum" and "sum" not in ctx.types:
        return _sum_call(node, ctx, lambda e, c: _hoist_expr(e, c, st, pre))
    if isinstance(func, ast.Name):
        name = func.id
        key = _project_key(name, ctx)
        if key is not None:
            if node.keywords:
                raise _fail(node, f"call {name}")
            arg_codes = []
            for arg in node.args:
                hoisted = _hoist_expr(arg, ctx, st, pre)
                _need(hoisted.tag, arg)
                if isinstance(arg, ast.Name) and _is_seq(hoisted.tag or "") and key not in ctx.read_only:
                    raise _fail(node, "shared list")
                arg_codes.append(hoisted.code)
            module = ctx.callees[key]
            temp = st.fresh()
            pre.append(f"var {temp} := {module}.{name}({', '.join(arg_codes)});")
            return _Typed(temp, _callee_tag(ctx, key))
        raise _fail(node, f"call {name}")
    if isinstance(func, ast.Attribute):
        value = _hoist_expr(func.value, ctx, st, pre)
        for arg in node.args:
            _hoist_expr(arg, ctx, st, pre)
        if value.tag is None:
            raise _fail(node, "unknown type")
        if _is_seq(value.tag) and func.attr in MUTATING_METHODS:
            raise _fail(node, "mutated list")
        raise _fail(node, f"call {func.attr}")
    raise _fail(node, "call expression")


def _hoist_bound(
    node: ast.expr | None, ctx: _Ctx, st: _HoistState, pre: list[str], default: str, whole: ast.expr
) -> str:
    """A slice bound with hoisted calls; a missing bound becomes ``default``."""
    if node is None:
        return default
    bound = _hoist_expr(node, ctx, st, pre)
    if _need(bound.tag, node) != _INT:
        raise _fail(whole, "slice bounds")
    return bound.code


def _hoist_subscript(node: ast.Subscript, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """``x[i]``, ``x[a:b]`` and ``x[::-1]`` with hoisted subexpressions."""
    value = _hoist_expr(node.value, ctx, st, pre)
    tag = _need(value.tag, node.value)
    if tag != _STR and not _is_seq(tag):
        raise _fail(node, "index type")
    elem = _CHAR if tag == _STR else _seq_elem(tag)
    subscript = node.slice
    if isinstance(subscript, ast.Slice):
        if subscript.step is not None:
            if subscript.lower is None and subscript.upper is None and _is_negative_one(subscript.step):
                return _Typed(f"py_reversed({value.code})", tag)
            raise _fail(node, "slice step")
        lower = _hoist_bound(subscript.lower, ctx, st, pre, "0", node)
        upper = _hoist_bound(subscript.upper, ctx, st, pre, f"py_len({value.code})", node)
        return _Typed(f"py_slice({value.code}, {lower}, {upper})", tag)
    index = _hoist_expr(subscript, ctx, st, pre)
    if _need(index.tag, subscript) != _INT:
        raise _fail(node, "index type")
    return _Typed(f"py_index({value.code}, {index.code})", elem)


def _hoist_ifexp(node: ast.IfExp, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """``a if c else b`` with hoisted subexpressions."""
    cond = _hoist_expr(node.test, ctx, st, pre)
    if _need(cond.tag, node.test) != _BOOL:
        raise _fail(node, "non-bool condition")
    when_true = _hoist_expr(node.body, ctx, st, pre)
    when_false = _hoist_expr(node.orelse, ctx, st, pre)
    true_tag = _need(when_true.tag, node.body)
    false_tag = _need(when_false.tag, node.orelse)
    if true_tag != false_tag:
        if {true_tag, false_tag} == {_CHAR, _STR}:
            raise _fail(node, "char vs str")
        raise _fail(node, "operand types")
    body = when_true.code if not isinstance(node.body, ast.IfExp) else f"({when_true.code})"
    orelse = when_false.code if not isinstance(node.orelse, ast.IfExp) else f"({when_false.code})"
    cond_code = cond.code if not isinstance(node.test, ast.IfExp) else f"({cond.code})"
    return _Typed(f"if {cond_code} then {body} else {orelse}", true_tag)


def _hoist_list_display(node: ast.List, ctx: _Ctx, st: _HoistState, pre: list[str]) -> _Typed:
    """A list display with hoisted elements."""
    if not node.elts:
        raise _fail(node, "empty list")
    elts = [_hoist_expr(elt, ctx, st, pre) for elt in node.elts]
    tags = [_need(elt.tag, original) for elt, original in zip(elts, node.elts, strict=True)]
    if any(tag != tags[0] for tag in tags[1:]):
        raise _fail(node, "operand types")
    return _Typed(f"[{', '.join(elt.code for elt in elts)}]", f"seq<{tags[0]}>")


def _collect_assigned(stmts: list[ast.stmt]) -> set[str]:
    """Every name assigned anywhere in ``stmts``, for the rename map.

    Plain, annotated and augmented assignments, tuple targets and walrus targets
    count, including inside ``if``, ``for`` and ``while`` bodies.
    """
    assigned: set[str] = set()

    def targets(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            assigned.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                targets(elt)
        elif isinstance(target, ast.Starred):
            targets(target.value)

    def _subscript_target(target: ast.expr) -> None:
        if isinstance(target, ast.Subscript):
            _subscript_target(target.value)
        else:
            targets(target)

    def visit_block(block: list[ast.stmt]) -> None:
        for stmt in block:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    _subscript_target(target)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                _subscript_target(stmt.target)
            elif isinstance(stmt, ast.If):
                visit_block(stmt.body)
                visit_block(stmt.orelse)
            elif isinstance(stmt, ast.For):
                targets(stmt.target)
                visit_block(stmt.body)
                visit_block(stmt.orelse)
            elif isinstance(stmt, ast.While):
                visit_block(stmt.body)
                visit_block(stmt.orelse)
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.NamedExpr):
                    targets(child.target)

    visit_block(stmts)
    return assigned


def _statements(
    stmts: list[ast.stmt],
    ctx: _Ctx,
    declared: set[str],
    return_tag: str,
    st: _HoistState,
    *,
    depth: int,
    first: bool,
) -> tuple[list[str], bool]:
    """Translate ``stmts``; return the lines and whether every path returns.

    ``declared`` holds the Python names already declared in Dafny (parameters plus
    earlier ``var`` bindings); it is updated in place, as is the type environment.
    ``st`` carries the recursion hoisting state.
    """
    lines: list[str] = []
    returns = False
    for index, stmt in enumerate(stmts):
        if first and index == 0 and _is_docstring(stmt):
            continue
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Return):
            lines.extend(_return_lines(stmt, ctx, declared, st, return_tag, depth=depth))
            returns = True
            break
        if isinstance(stmt, ast.Expr):
            _expr(stmt.value, ctx)
            raise _fail(stmt, "expression statement")
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            lines.extend(_indent(_assign(stmt, ctx, declared, st), depth))
            continue
        if isinstance(stmt, ast.AugAssign):
            lines.extend(_indent(_augassign(stmt, ctx, declared, st), depth))
            continue
        if isinstance(stmt, ast.If):
            block, done = _branch(stmt, ctx, declared, return_tag, st, depth=depth)
            lines.extend(block)
            if done:
                returns = True
                break
            continue
        if isinstance(stmt, ast.For):
            lines.extend(_for_stmt(stmt, ctx, declared, return_tag, st, depth=depth))
            continue
        if isinstance(stmt, ast.While):
            lines.extend(_while_stmt(stmt, ctx, declared, return_tag, st, depth=depth))
            continue
        if isinstance(stmt, (ast.Break, ast.Continue)):
            raise _fail(stmt, type(stmt).__name__.lower())
        raise _statement_error(stmt)
    return lines, returns


def _is_range_call(node: ast.expr, ctx: _Ctx) -> bool:
    """Whether ``node`` is a call to the ``range`` builtin (not a shadowed name)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and "range" not in ctx.types
    )


def _merge_body_env(ctx: _Ctx, body_ctx: _Ctx) -> None:
    """Merge a loop body environment back: new names stay body-local.

    Assignments to names declared before the loop keep their (checked, unchanged)
    types; names first bound inside the body are dropped, since the body may run
    zero times and Dafny scopes them to the loop.
    """
    for name in list(body_ctx.types):
        if name not in ctx.types:
            del body_ctx.types[name]
    ctx.types.update(body_ctx.types)


def _body_lines(
    body: list[ast.stmt],
    ctx: _Ctx,
    declared: set[str],
    return_tag: str,
    st: _HoistState,
    *,
    depth: int,
) -> list[str]:
    """Translate a loop body at ``depth`` without leaking its new names."""
    body_ctx = ctx.child()
    body_declared = set(declared)
    body_lines, _ = _statements(body, body_ctx, body_declared, return_tag, st, depth=depth, first=False)
    _merge_body_env(ctx, body_ctx)
    return body_lines


def _for_stmt(
    stmt: ast.For, ctx: _Ctx, declared: set[str], return_tag: str, st: _HoistState, *, depth: int
) -> list[str]:
    """Translate ``for i in range(a[, b])`` and ``for x in xs`` to ``while`` loops."""
    if stmt.orelse:
        raise _fail(stmt, "loop else")
    if stmt.type_comment is not None:
        raise _fail(stmt, "for loop")
    if not isinstance(stmt.target, ast.Name):
        raise _fail(stmt, "for target")
    var = stmt.target.id
    dvar = ctx.name(var)
    target_iter = stmt.iter
    if isinstance(target_iter, ast.Call) and _is_range_call(target_iter, ctx):
        if target_iter.keywords:
            raise _fail(stmt, "range arguments")
        return _for_range(stmt, var, dvar, target_iter, ctx, declared, return_tag, st, depth=depth)
    pre: list[str] = []
    if _has_any_inside_boolop(stmt.iter, st.func_name, ctx):
        raise _fail(stmt, "call in short-circuit")
    seq_value = _hoist_expr(stmt.iter, ctx, st, pre)
    seq_tag = _need(seq_value.tag, stmt.iter)
    if seq_tag is None:
        raise _fail(stmt, "unknown type")
    if not _is_seq(seq_tag):
        raise _fail(stmt, "for iterable")
    elem = _seq_elem(seq_tag)
    if var in ctx.types and ctx.types[var] != elem:
        raise _fail(stmt, "assignment type")
    ctx.types[var] = elem
    snapshot = st.fresh()
    index = st.fresh()
    if var in declared:
        bind = f"{dvar} := py_index({snapshot}, {index});"
    else:
        bind = f"var {dvar} := py_index({snapshot}, {index});"
    pad = "  " * (depth + 1)
    lines = _indent([*pre, f"var {snapshot} := {seq_value.code};", f"var {index} := 0;"], depth)
    lines.append(f"{pad}while {index} < py_len({snapshot})")
    lines.append(f"{pad}{{")
    lines.extend(_indent([bind], depth + 1))
    lines.extend(_body_lines(stmt.body, ctx, declared, return_tag, st, depth=depth + 1))
    lines.extend(_indent([f"{index} := {index} + 1;"], depth + 1))
    lines.append(f"{pad}}}")
    return lines


def _for_range(
    stmt: ast.For,
    var: str,
    dvar: str,
    call: ast.Call,
    ctx: _Ctx,
    declared: set[str],
    return_tag: str,
    st: _HoistState,
    *,
    depth: int,
) -> list[str]:
    """Translate ``for i in range(a[, b])`` to a counter ``while`` loop."""
    args = call.args
    if len(args) == 1:
        lower_node: ast.expr | None = None
        upper_node = args[0]
    elif len(args) == 2:
        lower_node = args[0]
        upper_node = args[1]
    elif len(args) == 3:
        raise _fail(stmt, "range step")
    else:
        raise _fail(stmt, "range arguments")
    if _has_any_inside_boolop(call, st.func_name, ctx):
        raise _fail(stmt, "call in short-circuit")
    pre: list[str] = []
    lower = "0"
    if lower_node is not None:
        lower_value = _hoist_expr(lower_node, ctx, st, pre)
        if _need(lower_value.tag, lower_node) != _INT:
            raise _fail(stmt, "range arguments")
        lower = lower_value.code
    upper_value = _hoist_expr(upper_node, ctx, st, pre)
    if _need(upper_value.tag, upper_node) != _INT:
        raise _fail(stmt, "range arguments")
    if var in ctx.types and ctx.types[var] != _INT:
        raise _fail(stmt, "assignment type")
    ctx.types[var] = _INT
    bound = st.fresh()
    head = [*pre]
    if var not in declared:
        head.append(f"var {dvar} := {lower};")
        declared.add(var)
    else:
        head.append(f"{dvar} := {lower};")
    head.append(f"var {bound} := {upper_value.code};")
    pad = "  " * (depth + 1)
    lines = _indent(head, depth)
    lines.append(f"{pad}while {dvar} < {bound}")
    lines.append(f"{pad}{{")
    lines.extend(_body_lines(stmt.body, ctx, declared, return_tag, st, depth=depth + 1))
    lines.extend(_indent([f"{dvar} := {dvar} + 1;"], depth + 1))
    lines.append(f"{pad}}}")
    return lines


def _while_operand(node: ast.expr, ctx: _Ctx, st: _HoistState, pre: list[str]) -> str:
    """Translate one ``while`` condition operand, hoisting calls for one evaluation.

    A call that is the whole operand binds ``var pcc_c := ...``; calls nested in a
    larger expression bind ``pcc_tN`` locals through ``_hoist_expr``. A project
    call without a known type defaults to ``bool`` here, since the position
    requires it; anything else must already be ``bool``.
    """
    if isinstance(node, ast.Call) and (_is_self_call(node, ctx.func_name, ctx.renames) or _is_project_call(node, ctx)):
        if _is_self_call(node, ctx.func_name, ctx.renames):
            if node.keywords:
                raise _fail(node, f"call {ctx.func_name}")
            arg_codes = []
            for arg in node.args:
                hoisted = _hoist_expr(arg, ctx, st, pre)
                _need(hoisted.tag, arg)
                arg_codes.append(hoisted.code)
            code = f"{st.callee}({', '.join(arg_codes)})"
            tag: str | None = st.return_tag
        else:
            assert isinstance(node.func, ast.Name)
            key = _project_key(node.func.id, ctx)
            assert key is not None
            code, tag = _project_call(node, ctx, key, lambda e, c: _hoist_expr(e, c, st, pre))
            tag = tag if tag is not None else _BOOL
        if tag != _BOOL:
            raise _fail(node, "non-bool condition")
        temp = st.fresh_cond()
        pre.append(f"var {temp} := {code};")
        return temp
    if _has_any_inside_boolop(node, st.func_name, ctx):
        raise _fail(node, "while condition")
    value = _hoist_expr(node, ctx, st, pre)
    if _need(value.tag, node) != _BOOL:
        raise _fail(node, "non-bool condition")
    return value.code


def _while_and_guards(operands: list[ast.expr], ctx: _Ctx, st: _HoistState, *, depth: int) -> list[str]:
    """Guard lines for an ``and`` chain: each operand checked in order."""
    lines: list[str] = []
    for operand in operands:
        pre: list[str] = []
        code = _while_operand(operand, ctx, st, pre)
        lines.extend(_indent(pre, depth))
        lines.extend(_indent([f"if !{code} {{ break; }}"], depth))
    return lines


def _while_or_guards(operands: list[ast.expr], ctx: _Ctx, st: _HoistState, *, depth: int) -> list[str]:
    """Guard lines for an ``or`` chain, nesting so Python order is kept."""
    pre: list[str] = []
    code = _while_operand(operands[0], ctx, st, pre)
    lines = _indent(pre, depth)
    if len(operands) == 1:
        lines.extend(_indent([f"if !{code} {{ break; }}"], depth))
        return lines
    indented = "  " * (depth + 1)
    lines.extend(_indent([f"if !{code} {{"], depth))
    lines.extend(_while_or_guards(operands[1:], ctx, st, depth=depth + 1))
    lines.append(f"{indented}}}")
    return lines


def _while_stmt(
    stmt: ast.While, ctx: _Ctx, declared: set[str], return_tag: str, st: _HoistState, *, depth: int
) -> list[str]:
    """Translate ``while``: plain when call-free, else ``while true`` with hoists."""
    if stmt.orelse:
        raise _fail(stmt, "loop else")
    test = stmt.test
    pad = "  " * (depth + 1)
    if not _contains_any_call(test, st.func_name, ctx):
        cond = _expr(test, ctx)
        if _need(cond.tag, test) != _BOOL:
            raise _fail(stmt, "non-bool condition")
        lines = [f"{pad}while {cond.code}", f"{pad}{{"]
        lines.extend(_body_lines(stmt.body, ctx, declared, return_tag, st, depth=depth + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(test, ast.BoolOp):
        operands = list(test.values)
        if any(isinstance(value, (ast.BoolOp, ast.IfExp)) for value in operands):
            raise _fail(stmt, "while condition")
        is_and = isinstance(test.op, ast.And)
    elif isinstance(test, ast.IfExp):
        raise _fail(stmt, "while condition")
    else:
        operands = [test]
        is_and = True
    lines = [f"{pad}while true", f"{pad}{{"]
    if is_and:
        lines.extend(_while_and_guards(operands, ctx, st, depth=depth + 1))
    else:
        lines.extend(_while_or_guards(operands, ctx, st, depth=depth + 1))
    lines.extend(_body_lines(stmt.body, ctx, declared, return_tag, st, depth=depth + 1))
    lines.append(f"{pad}}}")
    return lines


def _is_docstring(stmt: ast.stmt) -> bool:
    """Whether ``stmt`` is a docstring expression statement."""
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _return(stmt: ast.Return, ctx: _Ctx, return_tag: str) -> str:
    """Translate ``return e`` with a check against the declared return type."""
    if stmt.value is None:
        raise _fail(stmt, "empty return")
    value = _expr(stmt.value, ctx)
    if _need(value.tag, stmt.value) != return_tag:
        raise _fail(stmt, "return type")
    return f"return {value.code};"


def _return_lines(
    stmt: ast.Return,
    ctx: _Ctx,
    declared: set[str],
    st: _HoistState,
    return_tag: str,
    *,
    depth: int,
) -> list[str]:
    """``return`` with calls hoisted, expanding short-circuit under ``return``."""
    _ = declared
    if stmt.value is None:
        raise _fail(stmt, "empty return")
    expr = stmt.value
    if _contains_self_call(expr, st.func_name, ctx.renames):
        if _has_sc_self_call(expr, st.func_name, ctx.renames):
            if isinstance(expr, (ast.BoolOp, ast.IfExp)):
                return _return_sc_lines(expr, stmt, ctx, st, return_tag, depth=depth)
            raise _fail(stmt, "call in short-circuit")
        pre: list[str] = []
        value = _hoist_expr(expr, ctx, st, pre)
        if _need(value.tag, expr) != return_tag:
            raise _fail(stmt, "return type")
        return _indent([*pre, f"return {value.code};"], depth)
    if _contains_proj_call(expr, ctx):
        if _has_sc_proj_call(expr, ctx):
            if isinstance(expr, (ast.BoolOp, ast.IfExp)):
                return _return_sc_lines(expr, stmt, ctx, st, return_tag, depth=depth)
            raise _fail(stmt, "call in short-circuit")
        pre_proj: list[str] = []
        value = _hoist_expr(expr, ctx, st, pre_proj)
        if _need(value.tag, expr) != return_tag:
            raise _fail(stmt, "return type")
        return _indent([*pre_proj, f"return {value.code};"], depth)
    value = _expr(expr, ctx)
    if _need(value.tag, expr) != return_tag:
        raise _fail(stmt, "return type")
    return _indent([f"return {value.code};"], depth)


def _return_expr_lines(
    expr: ast.expr,
    stmt: ast.Return | ast.If,
    ctx: _Ctx,
    st: _HoistState,
    return_tag: str,
    *,
    depth: int,
) -> list[str]:
    """Lines for ``return <expr>`` at statement ``stmt`` (already indented)."""
    if _contains_self_call(expr, st.func_name, ctx.renames):
        if _has_sc_self_call(expr, st.func_name, ctx.renames):
            if isinstance(expr, (ast.BoolOp, ast.IfExp)):
                return _return_sc_lines(expr, stmt, ctx, st, return_tag, depth=depth)
            raise _fail(stmt, "call in short-circuit")
        pre: list[str] = []
        value = _hoist_expr(expr, ctx, st, pre)
        if _need(value.tag, expr) != return_tag:
            raise _fail(stmt, "return type")
        return _indent([*pre, f"return {value.code};"], depth)
    if _contains_proj_call(expr, ctx):
        if _has_sc_proj_call(expr, ctx):
            if isinstance(expr, (ast.BoolOp, ast.IfExp)):
                return _return_sc_lines(expr, stmt, ctx, st, return_tag, depth=depth)
            raise _fail(stmt, "call in short-circuit")
        pre = []
        value = _hoist_expr(expr, ctx, st, pre)
        if _need(value.tag, expr) != return_tag:
            raise _fail(stmt, "return type")
        return _indent([*pre, f"return {value.code};"], depth)
    value = _expr(expr, ctx)
    if _need(value.tag, expr) != return_tag:
        raise _fail(stmt, "return type")
    return _indent([f"return {value.code};"], depth)


def _return_sc_lines(
    expr: ast.BoolOp | ast.IfExp,
    stmt: ast.Return | ast.If,
    ctx: _Ctx,
    st: _HoistState,
    return_tag: str,
    *,
    depth: int,
) -> list[str]:
    """Expand ``return`` of ``and``/``or``/conditional preserving short-circuit order."""
    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And):
            return _emit_and_list(list(expr.values), stmt, ctx, st, return_tag, depth=depth)
        return _emit_or_list(list(expr.values), stmt, ctx, st, return_tag, depth=depth)
    return _emit_ifexp(expr, stmt, ctx, st, return_tag, depth=depth)


def _emit_and_list(
    values: list[ast.expr],
    stmt: ast.Return | ast.If,
    ctx: _Ctx,
    st: _HoistState,
    return_tag: str,
    *,
    depth: int,
) -> list[str]:
    """``return v0 and v1 ...`` keeping Python's left-to-right short-circuit."""
    if return_tag != _BOOL:
        raise _fail(stmt, "return type")
    if len(values) == 1:
        return _return_expr_lines(values[0], stmt, ctx, st, return_tag, depth=depth)
    first = values[0]
    rest = values[1:]
    if _has_sc_any_call(first, st.func_name, ctx):
        # A nested ``and``/``or``/conditional inside the first operand is not
        # directly under ``return``, so it is rejected instead of expanded.
        raise _fail(stmt, "call in short-circuit")
    pre: list[str] = []
    hoisted = _hoist_expr(first, ctx, st, pre)
    if _need(hoisted.tag, first) != _BOOL:
        raise _fail(stmt, "non-bool logic")
    indented = "  " * (depth + 1)
    lines = _indent(pre, depth)
    lines.append(f"{indented}if {hoisted.code} {{")
    lines.extend(_emit_and_list(rest, stmt, ctx, st, return_tag, depth=depth + 1))
    lines.append(f"{indented}}} else {{")
    lines.extend(_indent(["return false;"], depth + 1))
    lines.append(f"{indented}}}")
    return lines


def _emit_or_list(
    values: list[ast.expr],
    stmt: ast.Return | ast.If,
    ctx: _Ctx,
    st: _HoistState,
    return_tag: str,
    *,
    depth: int,
) -> list[str]:
    """``return v0 or v1 ...`` keeping Python's left-to-right short-circuit."""
    if return_tag != _BOOL:
        raise _fail(stmt, "return type")
    if len(values) == 1:
        return _return_expr_lines(values[0], stmt, ctx, st, return_tag, depth=depth)
    first = values[0]
    rest = values[1:]
    if _has_sc_any_call(first, st.func_name, ctx):
        # A nested ``and``/``or``/conditional inside the first operand is not
        # directly under ``return``, so it is rejected instead of expanded.
        raise _fail(stmt, "call in short-circuit")
    pre: list[str] = []
    hoisted = _hoist_expr(first, ctx, st, pre)
    if _need(hoisted.tag, first) != _BOOL:
        raise _fail(stmt, "non-bool logic")
    indented = "  " * (depth + 1)
    lines = _indent(pre, depth)
    lines.append(f"{indented}if {hoisted.code} {{")
    lines.extend(_indent(["return true;"], depth + 1))
    lines.append(f"{indented}}} else {{")
    lines.extend(_emit_or_list(rest, stmt, ctx, st, return_tag, depth=depth + 1))
    lines.append(f"{indented}}}")
    return lines


def _emit_ifexp(
    node: ast.IfExp,
    stmt: ast.Return | ast.If,
    ctx: _Ctx,
    st: _HoistState,
    return_tag: str,
    *,
    depth: int,
) -> list[str]:
    """``return a if c else b`` with short-circuit branches expanded."""
    if _has_sc_any_call(node.test, st.func_name, ctx):
        # A short-circuit call inside the condition is not directly under
        # ``return``, so it is rejected instead of expanded.
        raise _fail(stmt, "call in short-circuit")
    pre: list[str] = []
    cond = _hoist_expr(node.test, ctx, st, pre)
    if _need(cond.tag, node.test) != _BOOL:
        raise _fail(stmt, "non-bool condition")
    indented = "  " * (depth + 1)
    lines = _indent(pre, depth)
    lines.append(f"{indented}if {cond.code} {{")
    lines.extend(_return_expr_lines(node.body, stmt, ctx, st, return_tag, depth=depth + 1))
    lines.append(f"{indented}}} else {{")
    lines.extend(_return_expr_lines(node.orelse, stmt, ctx, st, return_tag, depth=depth + 1))
    lines.append(f"{indented}}}")
    return lines


def _assign_target(
    target: ast.expr,
    value_node: ast.expr,
    value: _Typed,
    stmt: ast.stmt,
    ctx: _Ctx,
) -> tuple[str, str]:
    """Check an assignment to ``target``; return the bound name and its tag.

    Subscript stores on lists, aliasing of a list-typed value and stores through
    other non-names raise, so element mutation never becomes an immutable sequence.
    """
    tag = _need(value.tag, value_node)
    if isinstance(target, ast.Name):
        name = target.id
        if isinstance(value_node, ast.Name) and _is_seq(ctx.types.get(value_node.id) or "") and _is_seq(tag):
            raise _fail(stmt, "shared list")
        if name in ctx.types and ctx.types[name] != tag:
            raise _fail(stmt, "assignment type")
        return name, tag
    if isinstance(target, ast.Subscript):
        base = _expr(target.value, ctx)
        base_tag = _need(base.tag, target.value)
        if _is_seq(base_tag):
            raise _fail(stmt, "mutated list")
        if base_tag == _STR:
            raise _fail(stmt, "mutated string")
        raise _fail(stmt, "index assignment")
    if isinstance(target, ast.Attribute):
        raise _fail(stmt, "attribute assignment")
    raise _fail(stmt, "destructuring")


def _tuple_assign(
    stmt: ast.Assign | ast.AnnAssign,
    target: ast.Tuple | ast.List,
    value_node: ast.expr,
    declared_type: str | None,
    ctx: _Ctx,
    declared: set[str],
    st: _HoistState,
) -> list[str]:
    """Translate ``a, b = e1, e2`` as a parallel Dafny assignment.

    Both right-hand sides are evaluated before either left-hand side changes.
    Mixing a fresh name with an already declared one raises ``"tuple assignment"``.
    """
    if declared_type is not None:
        raise _fail(stmt, "tuple assignment")
    if any(not isinstance(elt, ast.Name) for elt in target.elts):
        raise _fail(stmt, "tuple assignment")
    names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
    if not isinstance(value_node, ast.Tuple) or len(value_node.elts) != len(names):
        raise _fail(stmt, "tuple assignment")
    if _has_any_inside_boolop(value_node, st.func_name, ctx):
        raise _fail(stmt, "call in short-circuit")
    pre: list[str] = []
    codes: list[str] = []
    tags: list[str] = []
    use_hoist = _contains_any_call(value_node, st.func_name, ctx)
    for elt in value_node.elts:
        value = _hoist_expr(elt, ctx, st, pre) if use_hoist else _expr(elt, ctx)
        tag = _need(value.tag, elt)
        if isinstance(elt, ast.Name) and _is_seq(ctx.types.get(elt.id) or "") and _is_seq(tag):
            raise _fail(stmt, "shared list")
        codes.append(value.code)
        tags.append(tag)
    for name, tag in zip(names, tags, strict=True):
        if name in ctx.types and ctx.types[name] != tag:
            raise _fail(stmt, "assignment type")
    if any(tag != tags[0] for tag in tags[1:]):
        raise _fail(stmt, "assignment type")
    fresh = [name for name in names if name not in declared]
    if fresh and len(fresh) != len(names):
        raise _fail(stmt, "tuple assignment")
    for name, tag in zip(names, tags, strict=True):
        ctx.types[name] = tag
    joined_names = ", ".join(ctx.name(name) for name in names)
    line = f"{joined_names} := {', '.join(codes)};"
    if fresh:
        declared.update(names)
        return [*pre, f"var {line}"]
    return [*pre, line]


def _store_subscript(
    stmt: ast.Assign | ast.AnnAssign,
    target: ast.Subscript,
    value_node: ast.expr,
    declared_type: str | None,
    ctx: _Ctx,
    declared: set[str],
    st: _HoistState,
) -> list[str]:
    """Translate ``xs[i] = e`` on a local list to ``xs := py_set_index(xs, i, e)``.

    Storing through a parameter raises ``"shared list"``; a slice store raises
    ``"mutated list"``, as before.
    """
    _ = declared
    if isinstance(target.slice, ast.Slice):
        base = _expr(target.value, ctx)
        base_tag = _need(base.tag, target.value)
        if _is_seq(base_tag):
            raise _fail(stmt, "mutated list")
        if base_tag == _STR:
            raise _fail(stmt, "mutated string")
        raise _fail(stmt, "index assignment")
    if not isinstance(target.value, ast.Name):
        base = _expr(target.value, ctx)
        base_tag = _need(base.tag, target.value)
        if _is_seq(base_tag):
            raise _fail(stmt, "mutated list")
        if base_tag == _STR:
            raise _fail(stmt, "mutated string")
        raise _fail(stmt, "index assignment")
    name = target.value.id
    base_tag = _need(ctx.tag_of(name), target.value)
    if not _is_seq(base_tag):
        if base_tag == _STR:
            raise _fail(stmt, "mutated string")
        raise _fail(stmt, "index assignment")
    if name in ctx.params:
        raise _fail(stmt, "shared list")
    elem = _seq_elem(base_tag)
    if _has_any_inside_boolop(target.slice, st.func_name, ctx) or _has_any_inside_boolop(value_node, st.func_name, ctx):
        raise _fail(stmt, "call in short-circuit")
    pre: list[str] = []
    use_hoist = _contains_any_call(target.slice, st.func_name, ctx) or _contains_any_call(value_node, st.func_name, ctx)
    if use_hoist:
        index = _hoist_expr(target.slice, ctx, st, pre)
        value = _hoist_expr(value_node, ctx, st, pre)
    else:
        index = _expr(target.slice, ctx)
        value = _expr(value_node, ctx)
    if _need(index.tag, target.slice) != _INT:
        raise _fail(stmt, "index type")
    tag = _need(value.tag, value_node)
    if tag != elem:
        raise _fail(stmt, "assignment type")
    if declared_type is not None and declared_type != elem:
        raise _fail(stmt, "assignment type")
    dname = ctx.name(name)
    return [*pre, f"{dname} := py_set_index({dname}, {index.code}, {value.code});"]


def _bind_scalar(
    name: str,
    tag: str,
    code: str,
    ctx: _Ctx,
    declared: set[str],
    pre: list[str],
) -> list[str]:
    """Emit ``var x := e;`` or ``x := e;`` for a scalar and record its type."""
    ctx.types[name] = tag
    if name in declared:
        return [*pre, f"{ctx.name(name)} := {code};"]
    declared.add(name)
    return [*pre, f"var {ctx.name(name)} := {code};"]


def _assign(stmt: ast.Assign | ast.AnnAssign, ctx: _Ctx, declared: set[str], st: _HoistState) -> list[str]:
    """Translate ``x = e`` and ``x: T = e``, hoisting self and project calls in ``e``."""
    target: ast.expr
    value_node: ast.expr | None
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            raise _fail(stmt, "multiple assignment")
        target = stmt.targets[0]
        value_node = stmt.value
        declared_type: str | None = None
    else:
        target = stmt.target
        value_node = stmt.value
        if value_node is None:
            raise _fail(stmt, "bare annotation")
        declared_type = dafny_type(stmt.annotation, lineno=stmt.lineno)
    if isinstance(target, ast.Starred):
        raise _fail(stmt, "starred")
    if isinstance(target, (ast.Tuple, ast.List)):
        assert value_node is not None
        return _tuple_assign(stmt, target, value_node, declared_type, ctx, declared, st)
    if isinstance(target, ast.Subscript):
        assert value_node is not None
        return _store_subscript(stmt, target, value_node, declared_type, ctx, declared, st)
    assert value_node is not None
    if _contains_self_call(value_node, st.func_name, ctx.renames):
        if _has_call_inside_boolop(value_node, st.func_name, ctx.renames):
            raise _fail(stmt, "call in short-circuit")
        pre: list[str] = []
        value = _hoist_expr(value_node, ctx, st, pre)
        name, tag = _assign_target(target, value_node, value, stmt, ctx)
        if declared_type is not None and declared_type != tag:
            raise _fail(stmt, "assignment type")
        return _bind_scalar(name, tag, value.code, ctx, declared, pre)
    if _contains_proj_call(value_node, ctx):
        if _has_proj_inside_boolop(value_node, ctx):
            raise _fail(stmt, "call in short-circuit")
        if (
            isinstance(value_node, ast.Call)
            and _is_project_call(value_node, ctx)
            and not any(_contains_any_call(arg, st.func_name, ctx) for arg in value_node.args)
        ):
            # A single `x = M.g(args)` with call-free args is a Dafny call
            # statement, so it stays inline; anything composite must hoist.
            value = _expr(value_node, ctx)
            name, tag = _assign_target(target, value_node, value, stmt, ctx)
            if declared_type is not None and declared_type != tag:
                raise _fail(stmt, "assignment type")
            return _bind_scalar(name, tag, value.code, ctx, declared, [])
        pre_proj: list[str] = []
        value = _hoist_expr(value_node, ctx, st, pre_proj)
        name, tag = _assign_target(target, value_node, value, stmt, ctx)
        if declared_type is not None and declared_type != tag:
            raise _fail(stmt, "assignment type")
        return _bind_scalar(name, tag, value.code, ctx, declared, pre_proj)
    value = _expr(value_node, ctx)
    name, tag = _assign_target(target, value_node, value, stmt, ctx)
    if declared_type is not None and declared_type != tag:
        raise _fail(stmt, "assignment type")
    return _bind_scalar(name, tag, value.code, ctx, declared, [])


def _augassign(stmt: ast.AugAssign, ctx: _Ctx, declared: set[str], st: _HoistState) -> list[str]:
    """Translate ``x += e`` on ints to ``x := x + e;`` and ``xs[i] += e`` to a store.

    Anything else raises ``"augmented assignment"``; whole-list ``+=`` and slice
    targets keep raising ``"mutated list"``, as before.
    """
    _ = declared
    if not isinstance(stmt.op, ast.Add):
        raise _fail(stmt, "augmented assignment")
    target = stmt.target
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Slice):
        base = _expr(target.value, ctx)
        if base.tag is not None and _is_seq(base.tag):
            raise _fail(stmt, "mutated list")
        raise _fail(stmt, "augmented assignment")
    if isinstance(target, ast.Name):
        name = target.id
        if ctx.tag_of(name) != _INT:
            if _is_seq(ctx.types.get(name) or ""):
                raise _fail(stmt, "mutated list")
            raise _fail(stmt, "augmented assignment")
        if _has_any_inside_boolop(stmt.value, st.func_name, ctx):
            raise _fail(stmt, "call in short-circuit")
        pre: list[str] = []
        if _contains_any_call(stmt.value, st.func_name, ctx):
            value = _hoist_expr(stmt.value, ctx, st, pre)
        else:
            value = _expr(stmt.value, ctx)
        if _need(value.tag, stmt.value) != _INT:
            raise _fail(stmt, "augmented assignment")
        dname = ctx.name(name)
        return [*pre, f"{dname} := {dname} + {value.code};"]
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        name = target.value.id
        base_tag = _need(ctx.tag_of(name), target.value)
        if not _is_seq(base_tag):
            raise _fail(stmt, "augmented assignment")
        if name in ctx.params:
            raise _fail(stmt, "shared list")
        if _seq_elem(base_tag) != _INT:
            raise _fail(stmt, "augmented assignment")
        if _has_any_inside_boolop(target.slice, st.func_name, ctx) or _has_any_inside_boolop(
            stmt.value, st.func_name, ctx
        ):
            raise _fail(stmt, "call in short-circuit")
        pre = []
        use_hoist = _contains_any_call(target.slice, st.func_name, ctx) or _contains_any_call(
            stmt.value, st.func_name, ctx
        )
        if use_hoist:
            index = _hoist_expr(target.slice, ctx, st, pre)
            value = _hoist_expr(stmt.value, ctx, st, pre)
        else:
            index = _expr(target.slice, ctx)
            value = _expr(stmt.value, ctx)
        if _need(index.tag, target.slice) != _INT:
            raise _fail(stmt, "index type")
        if _need(value.tag, stmt.value) != _INT:
            raise _fail(stmt, "augmented assignment")
        dname = ctx.name(name)
        read = f"py_index({dname}, {index.code})"
        return [*pre, f"{dname} := py_set_index({dname}, {index.code}, {read} + {value.code});"]
    if isinstance(target, ast.Subscript):
        base = _expr(target.value, ctx)
        if base.tag is not None and _is_seq(base.tag):
            raise _fail(stmt, "mutated list")
    raise _fail(stmt, "augmented assignment")


def _branch(
    stmt: ast.If, ctx: _Ctx, declared: set[str], return_tag: str, st: _HoistState, *, depth: int
) -> tuple[list[str], bool]:
    """Translate an ``if``/``elif``/``else`` statement, hoisting calls in the condition."""
    indented = "  " * (depth + 1)
    pre_cond: list[str] = []
    if _contains_self_call(stmt.test, st.func_name, ctx.renames):
        if _has_call_inside_boolop(stmt.test, st.func_name, ctx.renames):
            raise _fail(stmt, "call in short-circuit")
        cond_hoisted = _hoist_expr(stmt.test, ctx, st, pre_cond)
        if _need(cond_hoisted.tag, stmt.test) != _BOOL:
            raise _fail(stmt, "non-bool condition")
        cond = cond_hoisted
    elif _contains_proj_call(stmt.test, ctx):
        if _has_proj_inside_boolop(stmt.test, ctx):
            raise _fail(stmt, "call in short-circuit")
        cond = _hoist_expr(stmt.test, ctx, st, pre_cond)
        if _need(cond.tag, stmt.test) != _BOOL:
            raise _fail(stmt, "non-bool condition")
    else:
        cond = _expr(stmt.test, ctx)
        if _need(cond.tag, stmt.test) != _BOOL:
            raise _fail(stmt, "non-bool condition")
    then_ctx = ctx.child()
    then_declared = set(declared)
    then_lines, then_returns = _statements(
        stmt.body, then_ctx, then_declared, return_tag, st, depth=depth + 1, first=False
    )
    lines = _indent(pre_cond, depth)
    lines.append(f"{indented}if {cond.code} {{")
    lines.extend(then_lines)
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
        else_ctx = ctx.child()
        else_declared = set(declared)
        nested, nested_returns = _branch(stmt.orelse[0], else_ctx, else_declared, return_tag, st, depth=depth)
        nested_first = nested[0].strip()
        lines.append(f"{indented}}} else {nested_first}")
        lines.extend(nested[1:])
        _merge(ctx.types, declared, then_ctx.types, then_declared, else_ctx.types, else_declared)
        return lines, then_returns and nested_returns
    else_returns = False
    else_lines: list[str] = []
    if stmt.orelse:
        else_ctx = ctx.child()
        else_declared = set(declared)
        else_lines, else_returns = _statements(
            stmt.orelse, else_ctx, else_declared, return_tag, st, depth=depth + 1, first=False
        )
        _merge(ctx.types, declared, then_ctx.types, then_declared, else_ctx.types, else_declared)
        lines.append(f"{indented}}} else {{")
        lines.extend(else_lines)
        lines.append(f"{indented}}}")
        return lines, then_returns and else_returns
    for name in list(then_ctx.types):
        if name not in ctx.types:
            del then_ctx.types[name]
    ctx.types.update(then_ctx.types)
    lines.append(f"{indented}}}")
    return lines, False


def _merge(
    types: dict[str, str],
    declared: set[str],
    then_types: dict[str, str],
    then_declared: set[str],
    else_types: dict[str, str],
    else_declared: set[str],
) -> None:
    """Merge the branch environments back into ``types``/``declared`` in place."""
    merged = {name: tag for name, tag in then_types.items() if else_types.get(name) == tag}
    for name in [name for name in types if name not in merged]:
        del types[name]
    types.update(merged)
    declared.update(then_declared & else_declared)


def _statement_error(stmt: ast.stmt) -> UnsupportedConstruct:
    """The ``UnsupportedConstruct`` for a statement outside the subset."""
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return _fail(stmt, "loop")
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _fail(stmt, "with")
    if isinstance(stmt, (ast.Try, ast.TryStar)):
        return _fail(stmt, "try")
    if isinstance(stmt, ast.Raise):
        return _fail(stmt, "raise")
    if isinstance(stmt, ast.FunctionDef):
        return _fail(stmt, "nested function")
    if isinstance(stmt, ast.AsyncFunctionDef):
        return _fail(stmt, "async function")
    if isinstance(stmt, ast.ClassDef):
        return _fail(stmt, "class definition")
    if isinstance(stmt, ast.Global):
        return _fail(stmt, "global")
    if isinstance(stmt, ast.Nonlocal):
        return _fail(stmt, "nonlocal")
    if isinstance(stmt, ast.Assert):
        return _fail(stmt, "assert")
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return _fail(stmt, "import")
    if isinstance(stmt, ast.Delete):
        return _fail(stmt, "del")
    if isinstance(stmt, ast.Match):
        return _fail(stmt, "match")
    if isinstance(stmt, ast.Expr):
        return _fail(stmt, "expression statement")
    return _fail(stmt, f"statement {type(stmt).__name__}")


def _indent(lines: list[str], depth: int) -> list[str]:
    """``lines`` indented for nesting ``depth`` inside the method body."""
    prefix = "  " * (depth + 1)
    return [f"{prefix}{line}" if line else "" for line in lines]


def _parameters(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Parameter declarations, renamed argument names and their Dafny types."""
    if func.type_params:
        raise _fail(func, "generic function")
    args = func.args
    if args.vararg is not None or args.kwarg is not None:
        raise _fail(func, "variadic")
    params: list[str] = []
    names: list[str] = []
    types: dict[str, str] = {}
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        tag = dafny_type(arg.annotation, lineno=arg.lineno)
        renamed = arg.arg + "_" if arg.arg in RESERVED else arg.arg
        params.append(f"{renamed}: {tag}")
        names.append(renamed)
        types[arg.arg] = tag
    return params, names, types


def transpile_function(
    function: FunctionRef,
    *,
    callees: Mapping[str, str] = MappingProxyType({}),
    read_only_callees: frozenset[str] = frozenset(),
    callee_types: Mapping[str, str] = MappingProxyType({}),
) -> str:
    """Code mode: executable Dafny built from ``pcc.pylib`` blocks.

    Returns ``method <name>(<params>) returns (result: <T>)`` with exactly one
    ``requires pre(<params>)`` clause and one ``ensures spec(<params>, result)``
    clause. ``<name>`` is ``verifier.layout.method_name`` of ``function``.
    ``callees`` maps the qualname of each callee in the project to its Dafny
    module name, so callers use only the callee's spec; a call to a project
    function becomes ``<Module>.<name>(...)``. ``read_only_callees`` holds the
    callee qualnames certified not to mutate or escape a list parameter, so a
    list passed to them stays a value; a list passed to any other project callee
    raises ``UnsupportedConstruct("shared list")``. ``callee_types`` maps a
    callee qualname to its Dafny return type tag; without it the call has an
    unknown type (a ``bool`` default applies in ``while`` conditions only).

    ``lineno`` of every error is relative to ``function.source``: line 1 is the
    first line of that source (the first decorator, if any).
    """
    try:
        tree = ast.parse(function.source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise UnsupportedConstruct("syntax", lineno=1) from None
    func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for top in tree.body:
        if isinstance(top, ast.AsyncFunctionDef):
            raise UnsupportedConstruct("async function", lineno=top.lineno)
        if isinstance(top, ast.FunctionDef):
            func = top
            break
    if func is None:
        raise UnsupportedConstruct("no function", lineno=1)
    params, names, types = _parameters(func)
    return_tag = dafny_type(func.returns, lineno=func.lineno)
    renames = {name: (name + "_" if name in RESERVED else name) for name in types}
    for extra in _collect_assigned(func.body):
        if extra not in renames:
            renames[extra] = extra + "_" if extra in RESERVED else extra
    declared = set(types)
    ctx = _Ctx(
        renames,
        types,
        params=frozenset(types),
        func_name=func.name,
        callees=callees,
        callee_types=callee_types,
        read_only=read_only_callees,
    )
    st = _HoistState(func.name, method_name(function), return_tag, set(renames.values()))
    body, returns = _statements(func.body, ctx, declared, return_tag, st, depth=0, first=True)
    if not returns:
        last = func.body[-1].lineno if func.body else func.lineno
        raise UnsupportedConstruct("missing return", lineno=last)
    spec_args = ", ".join([*names, "result"])
    header = [
        f"method {method_name(function)}({', '.join(params)}) returns (result: {return_tag})",
        f"  requires pre({', '.join(names)})",
        f"  ensures spec({spec_args})",
        "{",
    ]
    return "\n".join([*header, *body, "}"])


__all__ = ["MUTATING_METHODS", "RESERVED", "translate_expr", "transpile_function"]
