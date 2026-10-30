"""Step 4, code mode: the rule-based transpiler for executable Dafny (docs/plan.md, step 4).

What the transpiler supports, it translates exactly; anything else raises
``UnsupportedConstruct`` and the pipeline falls back to AI translation.

Supported subset (this ticket; recursion, spec mode, project calls, loops and mutated
lists live in later tickets):

- Types: ``int`` to ``int``, ``bool`` to ``bool``, ``str`` to ``string``, ``list[T]``
  and ``typing.List[T]`` to ``seq<T>`` (only when never mutated; see below). ``tuple``
  and ``typing.Tuple`` raise ``"type tuple"``; anything else raises ``"type <T>"``;
  missing hints raise ``"no types"``. The mapping lives in ``pcc.transpile.types``.
- Statements: ``if``/``elif``/``else``, ``return e``, local assignment ``x = e``
  (``var x := e;`` the first time, ``x := e;`` after). A docstring as the first
  statement is skipped.
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
  ``+`` on ``str`` and ``list`` to ``+``; ``a if c else b`` to ``if c then a else b``.
- Whole-variable rebinding of a list (``xs = expression``) is supported when the
  inferred type stays compatible. Subscript and slice stores, ``AugAssign`` on lists
  (including ``+=``), calls to mutating methods (``append``, ``extend``, ``insert``,
  ``pop``, ``remove``, ``clear``, ``sort``, ``reverse``) and aliasing a list-typed
  value into another name raise ``"mutated list"``.

Not supported: ``<`` (and friends) on strings, anything on ``float``, ``/``,
arithmetic on bools, string methods, loops, calls, ``raise``, ``try``, ``with``,
``yield``, ``await``, ``global`` and ``nonlocal``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pcc.core import FunctionRef, Spec, UnsupportedConstruct
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
    """Translation state: renames plus the live inferred type environment."""

    __slots__ = ("renames", "types")

    def __init__(self, renames: Mapping[str, str], types: dict[str, str]) -> None:
        self.renames = renames
        self.types = types

    def child(self) -> _Ctx:
        """A context sharing the renames with a copied type environment."""
        return _Ctx(self.renames, dict(self.types))

    def name(self, python_name: str) -> str:
        """The Dafny name of the Python variable ``python_name``."""
        return self.renames.get(python_name, python_name)

    def tag_of(self, python_name: str) -> str | None:
        """The Dafny type tag of ``python_name``, or ``None`` when unknown."""
        return self.types.get(python_name)


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
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
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


def _call(node: ast.Call, ctx: _Ctx) -> _Typed:
    """Translate ``len(x)``; every other call is rejected."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "len" and "len" not in ctx.types:
        if node.keywords or len(node.args) != 1:
            raise _fail(node, "len arguments")
        arg = _expr(node.args[0], ctx)
        tag = _need(arg.tag, node.args[0])
        if tag != _STR and not _is_seq(tag):
            raise _fail(node, "len operand")
        return _Typed(f"py_len({arg.code})", _INT)
    if isinstance(func, ast.Name):
        raise _fail(node, f"call {func.id}")
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


def _collect_assigned(stmts: list[ast.stmt]) -> set[str]:
    """Every name assigned anywhere in ``stmts``, for the rename map.

    Only plain and annotated assignments and walrus targets count; loops and other
    rejected statements never reach translation, so they contribute no renames.
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

    def visit_block(block: list[ast.stmt]) -> None:
        for stmt in block:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    targets(target)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                targets(stmt.target)
            elif isinstance(stmt, ast.If):
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
    *,
    depth: int,
    first: bool,
) -> tuple[list[str], bool]:
    """Translate ``stmts``; return the lines and whether every path returns.

    ``declared`` holds the Python names already declared in Dafny (parameters plus
    earlier ``var`` bindings); it is updated in place, as is the type environment.
    """
    lines: list[str] = []
    returns = False
    for index, stmt in enumerate(stmts):
        if first and index == 0 and _is_docstring(stmt):
            continue
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Return):
            lines.extend(_indent([_return(stmt, ctx, return_tag)], depth))
            returns = True
            break
        if isinstance(stmt, ast.Expr):
            _expr(stmt.value, ctx)
            raise _fail(stmt, "expression statement")
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            lines.extend(_indent(_assign(stmt, ctx, declared), depth))
            continue
        if isinstance(stmt, ast.AugAssign):
            raise _augassign(stmt, ctx)
        if isinstance(stmt, ast.If):
            block, done = _branch(stmt, ctx, declared, return_tag, depth=depth)
            lines.extend(block)
            if done:
                returns = True
                break
            continue
        raise _statement_error(stmt)
    return lines, returns


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
            raise _fail(stmt, "mutated list")
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


def _assign(stmt: ast.Assign | ast.AnnAssign, ctx: _Ctx, declared: set[str]) -> list[str]:
    """Translate ``x = e`` and ``x: T = e``."""
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
    if isinstance(target, (ast.Tuple, ast.List)):
        raise _fail(stmt, "destructuring")
    if isinstance(target, ast.Starred):
        raise _fail(stmt, "starred")
    value = _expr(value_node, ctx)
    name, tag = _assign_target(target, value_node, value, stmt, ctx)
    if declared_type is not None and declared_type != tag:
        raise _fail(stmt, "assignment type")
    assert isinstance(target, ast.Name)
    ctx.types[name] = tag
    if name in declared:
        return [f"{ctx.name(name)} := {value.code};"]
    declared.add(name)
    return [f"var {ctx.name(name)} := {value.code};"]


def _augassign(stmt: ast.AugAssign, ctx: _Ctx) -> UnsupportedConstruct:
    """``AugAssign`` on a list is element mutation; anything else is unsupported."""
    target = stmt.target
    if isinstance(target, ast.Name) and _is_seq(ctx.types.get(target.id) or ""):
        return _fail(stmt, "mutated list")
    if isinstance(target, ast.Subscript):
        base = _expr(target.value, ctx)
        if base.tag is not None and _is_seq(base.tag):
            return _fail(stmt, "mutated list")
    return _fail(stmt, "augmented assignment")


def _branch(stmt: ast.If, ctx: _Ctx, declared: set[str], return_tag: str, *, depth: int) -> tuple[list[str], bool]:
    """Translate an ``if``/``elif``/``else`` statement."""
    cond = _expr(stmt.test, ctx)
    if _need(cond.tag, stmt.test) != _BOOL:
        raise _fail(stmt, "non-bool condition")
    then_ctx = ctx.child()
    then_declared = set(declared)
    then_lines, then_returns = _statements(stmt.body, then_ctx, then_declared, return_tag, depth=depth + 1, first=False)
    indented = "  " * (depth + 1)
    lines = [f"{indented}if {cond.code} {{"]
    lines.extend(then_lines)
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
        else_ctx = ctx.child()
        else_declared = set(declared)
        nested, nested_returns = _branch(stmt.orelse[0], else_ctx, else_declared, return_tag, depth=depth)
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
            stmt.orelse, else_ctx, else_declared, return_tag, depth=depth + 1, first=False
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


def transpile_function(function: FunctionRef, *, callees: Mapping[str, str] = MappingProxyType({})) -> str:
    """Code mode: executable Dafny built from ``pcc.pylib`` blocks.

    Returns ``method <name>(<params>) returns (result: <T>)`` with exactly one
    ``requires pre(<params>)`` clause and one ``ensures spec(<params>, result)``
    clause. ``<name>`` is ``verifier.layout.method_name`` of ``function``.
    ``callees`` maps the qualname of each callee in the project to its Dafny
    module name, so callers use only the callee's spec; project calls are owned
    by a later ticket, so any call raises ``UnsupportedConstruct`` here.

    ``lineno`` of every error is relative to ``function.source``: line 1 is the
    first line of that source (the first decorator, if any).
    """
    _ = callees  # Reserved for project calls in a later ticket; any call raises here.
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
    ctx = _Ctx(renames, types)
    body, returns = _statements(func.body, ctx, declared, return_tag, depth=0, first=True)
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


def transpile_spec(spec: Spec) -> str:
    """Spec mode: logical Dafny (``forall``/``exists`` instead of loops) as ``requires``/``ensures`` clauses."""
    raise NotImplementedError


__all__ = ["MUTATING_METHODS", "RESERVED", "translate_expr", "transpile_function", "transpile_spec"]
