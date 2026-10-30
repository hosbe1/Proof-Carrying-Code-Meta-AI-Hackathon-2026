"""Step 4, spec mode: executable Python predicates to logical Dafny (docs/plan.md, step 4).

``transpile_spec`` turns the ``spec`` and ``pre`` functions of a ``Spec`` into ``predicate spec`` and ``predicate
pre``. Quantifiers replace loops: ``all``/``any`` over ``range`` become ``forall``/``exists`` over the bounds, and
over a list they become a ``forall``/``exists`` over indices with the loop variable replaced by the indexed element.
The output is the frozen spec the prover uses directly. Anything outside ``SPEC_CONSTRUCTS`` raises
``UnsupportedConstruct``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pcc.core import Spec, UnsupportedConstruct
from pcc.transpile.code import RESERVED
from pcc.transpile.types import dafny_type

#: The constructs spec mode supports. The translator below implements exactly these; anything else raises
#: ``UnsupportedConstruct``. Expression rules: comparisons, ``and``/``or``/``not``, ``+``/``-``/``*``,
#: ``//``/``%`` (through ``py_floordiv``/``py_mod``), ``a if c else b``, ``len``, indexing, ``[::-1]`` (through
#: ``py_reversed``), ``in`` on lists, and calls to ``all``/``any``/``range`` only as quantifiers.
SPEC_CONSTRUCTS = frozenset(
    {
        "FunctionDef",
        "Return",
        "docstring",
        "BoolOp",
        "and",
        "or",
        "BinOp",
        "+",
        "-",
        "*",
        "//",
        "%",
        "UnaryOp",
        "not",
        "unary -",
        "Compare",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "in",
        "not in",
        "IfExp",
        "Call",
        "all",
        "any",
        "len",
        "range",
        "Name",
        "Constant",
        "int",
        "bool",
        "str",
        "Subscript",
        "index",
        "reverse",
        "GeneratorExp",
    }
)

# Precedence classes for parenthesization (higher binds tighter). ``if``-expressions are loosest; anything nested
# inside a tighter context gets parentheses.
_PREC_IF = 0
_PREC_OR = 1
_PREC_AND = 2
_PREC_COMPARE = 3
_PREC_ADD = 4
_PREC_MUL = 5
_PREC_UNARY = 6
_PREC_ATOM = 7

Scope = dict[str, str]  # Python name -> Dafny text (renamed names and substituted loop variables)

_CMP_SYMBOLS: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}

_BINOP_SYMBOLS: dict[type[ast.operator], str] = {
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.MatMult: "@",
}

_PURE_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _precedence(node: ast.AST) -> int:
    """The precedence class of the expression ``node`` for parenthesization."""
    if isinstance(node, ast.IfExp):
        return _PREC_IF
    if _is_quantifier(node):
        return _PREC_IF
    if isinstance(node, ast.BoolOp):
        return _PREC_OR if isinstance(node.op, ast.Or) else _PREC_AND
    if isinstance(node, ast.Compare):
        return _PREC_COMPARE
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _PREC_ADD
        return _PREC_MUL
    if isinstance(node, ast.UnaryOp):
        return _PREC_UNARY
    return _PREC_ATOM


def _is_quantifier(node: ast.AST) -> bool:
    """Whether ``node`` is an ``all``/``any`` call, emitted as a ``forall``/``exists``."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("all", "any")


def _is_atomic(node: ast.AST) -> bool:
    """Whether ``node`` never needs parentheses as an operand or call argument."""
    return isinstance(node, (ast.Name, ast.Constant, ast.Call, ast.Subscript, ast.UnaryOp))


def _unsupported(node: ast.AST) -> UnsupportedConstruct:
    """The catch-all for nodes outside ``SPEC_CONSTRUCTS``."""
    return UnsupportedConstruct(f"expression {type(node).__name__.lower()}")


def _emit_constant(node: ast.Constant) -> str:
    """Dafny text for an ``int``, ``bool`` or ``str`` literal."""
    if isinstance(node.value, bool):
        return "true" if node.value else "false"
    if isinstance(node.value, int):
        return str(node.value)
    if isinstance(node.value, str):
        return _emit_string(node.value)
    raise UnsupportedConstruct(f"constant {node.value!r}")


def _emit_string(value: str) -> str:
    """``value`` as a Dafny double-quoted string literal."""
    parts = ['"']
    for char in value:
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
        elif char.isprintable():
            parts.append(char)
        else:
            parts.append(f"\\u{ord(char):04x}")
    parts.append('"')
    return "".join(parts)


def _full_type(node: ast.AST, types: dict[str, str]) -> str | None:
    """The Dafny type of ``node`` (``"string"``, ``"seq<...>"``, ...) or ``None`` when unknown."""
    if isinstance(node, ast.Name):
        return types.get(node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, str):
            return "string"
        return None
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return _full_type(node.value, types)
        inner = _inner_type(_full_type(node.value, types))
        return inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _full_type(node.left, types)
        right = _full_type(node.right, types)
        if left == "string" or right == "string":
            return "string"
        if left == "int" and right == "int":
            return "int"
        return None
    if isinstance(node, ast.IfExp):
        body = _full_type(node.body, types)
        orelse = _full_type(node.orelse, types)
        return body if body is not None and body == orelse else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "len":
            return "int"
        if node.func.id in ("all", "any"):
            return "bool"
        return None
    if isinstance(node, (ast.BoolOp, ast.Compare)):
        return "bool"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "bool"
    return None


def _inner_type(sequence: str | None) -> str | None:
    """The element type of the Dafny sequence type ``sequence``, or ``None``."""
    if sequence is not None and sequence.startswith("seq<") and sequence.endswith(">"):
        return sequence[4:-1]
    return None


def _kind(node: ast.AST, types: dict[str, str]) -> str:
    """``"string"`` or ``"seq"`` for the Dafny type of ``node``; ``"other"`` for anything else."""
    full = _full_type(node, types)
    if full == "string":
        return "string"
    if full is not None and full.startswith("seq<"):
        return "seq"
    return "other"


def _emit(node: ast.AST, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for the spec expression ``node``."""
    if isinstance(node, ast.Constant):
        return _emit_constant(node)
    if isinstance(node, ast.Name):
        if node.id not in scope:
            raise UnsupportedConstruct(f"name {node.id}")
        return scope[node.id]
    if isinstance(node, ast.BoolOp):
        joiner = " || " if isinstance(node.op, ast.Or) else " && "
        return joiner.join(_bool_operand(value, scope, types) for value in node.values)
    if isinstance(node, ast.BinOp):
        return _emit_binop(node, scope, types)
    if isinstance(node, ast.UnaryOp):
        return _emit_unary(node, scope, types)
    if isinstance(node, ast.Compare):
        return _emit_compare(node, scope, types)
    if isinstance(node, ast.IfExp):
        return _emit_if(node, scope, types)
    if isinstance(node, ast.Call):
        return _emit_call(node, scope, types)
    if isinstance(node, ast.Subscript):
        return _emit_subscript(node, scope, types)
    if isinstance(node, ast.Attribute):
        raise UnsupportedConstruct(f"attribute {node.attr}")
    if isinstance(node, ast.GeneratorExp):
        raise UnsupportedConstruct("generator")
    raise _unsupported(node)


def _bool_operand(node: ast.AST, scope: Scope, types: dict[str, str]) -> str:
    """One operand of ``and``/``or``: parentheses at or below ``and``/``or`` precedence."""
    text = _emit(node, scope, types)
    if _precedence(node) <= _PREC_AND:
        return f"({text})"
    return text


def _comparison_operand(node: ast.AST, scope: Scope, types: dict[str, str]) -> str:
    """One side of a comparison or bound: parentheses at or below comparison precedence."""
    text = _emit(node, scope, types)
    if _precedence(node) <= _PREC_COMPARE:
        return f"({text})"
    return text


def _call_argument(node: ast.AST, scope: Scope, types: dict[str, str]) -> str:
    """One call argument: parentheses for ``if``-expressions and quantifiers."""
    text = _emit(node, scope, types)
    if isinstance(node, ast.GeneratorExp) or _precedence(node) <= _PREC_IF:
        return f"({text})"
    return text


def _emit_binop(node: ast.BinOp, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for ``+``/``-``/``*`` and for ``//``/``%`` through the ``PyLib`` blocks."""
    if isinstance(node.op, ast.FloorDiv):
        return f"py_floordiv({_call_argument(node.left, scope, types)}, {_call_argument(node.right, scope, types)})"
    if isinstance(node.op, ast.Mod):
        return f"py_mod({_call_argument(node.left, scope, types)}, {_call_argument(node.right, scope, types)})"
    if isinstance(node.op, ast.Add):
        symbol, level = "+", _PREC_ADD
    elif isinstance(node.op, ast.Sub):
        symbol, level = "-", _PREC_ADD
    elif isinstance(node.op, ast.Mult):
        symbol, level = "*", _PREC_MUL
    elif isinstance(node.op, ast.Div):
        raise UnsupportedConstruct("operator /")
    elif isinstance(node.op, ast.Pow):
        raise UnsupportedConstruct("operator **")
    else:
        symbol = _BINOP_SYMBOLS.get(type(node.op), type(node.op).__name__)
        raise UnsupportedConstruct(f"operator {symbol}")
    left = _emit(node.left, scope, types)
    if _precedence(node.left) < level:
        left = f"({left})"
    right = _emit(node.right, scope, types)
    if _precedence(node.right) <= level:
        right = f"({right})"
    return f"{left} {symbol} {right}"


def _emit_unary(node: ast.UnaryOp, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for ``not x`` and ``-x``."""
    if isinstance(node.op, ast.Not):
        symbol = "!"
    elif isinstance(node.op, ast.USub):
        symbol = "-"
    elif isinstance(node.op, ast.UAdd):
        return _emit(node.operand, scope, types)
    else:
        raise UnsupportedConstruct("operator ~")
    text = _emit(node.operand, scope, types)
    if not _is_atomic(node.operand):
        text = f"({text})"
    return f"{symbol}{text}"


def _emit_compare(node: ast.Compare, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for comparisons and ``in``/``not in`` on lists."""
    if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
        return _emit_membership(node, scope, types)
    if any(not isinstance(op, _PURE_COMPARE) for op in node.ops):
        bad = next(op for op in node.ops if not isinstance(op, _PURE_COMPARE))
        symbol = _CMP_SYMBOLS.get(type(bad), type(bad).__name__)
        raise UnsupportedConstruct(f"operator {symbol}")
    parts = [_comparison_operand(node.left, scope, types)]
    for op, comparator in zip(node.ops, node.comparators, strict=True):
        assert isinstance(op, _PURE_COMPARE)
        parts.append(_CMP_SYMBOLS[type(op)])
        parts.append(_comparison_operand(comparator, scope, types))
    return " ".join(parts)


def _emit_membership(node: ast.Compare, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for ``x in xs``; ``in`` on a string is substring search and unsupported."""
    (op,) = node.ops
    (container,) = node.comparators
    if _kind(container, types) == "string":
        raise UnsupportedConstruct("substring in")
    if _kind(container, types) != "seq":
        raise UnsupportedConstruct(f"in {ast.unparse(node)}")
    left = _comparison_operand(node.left, scope, types)
    right = _comparison_operand(container, scope, types)
    if isinstance(op, ast.In):
        return f"{left} in {right}"
    return f"{left} !in {right}"


def _branch_text(node: ast.AST, scope: Scope, types: dict[str, str]) -> str:
    """One part of an ``if``-expression: parentheses unless it is one atomic expression."""
    text = _emit(node, scope, types)
    if not _is_atomic(node) or _is_quantifier(node):
        return f"({text})"
    return text


def _emit_if(node: ast.IfExp, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for ``a if c else b``."""
    return (
        f"if {_branch_text(node.test, scope, types)}"
        f" then {_branch_text(node.body, scope, types)}"
        f" else {_branch_text(node.orelse, scope, types)}"
    )


def _emit_call(node: ast.Call, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for ``len`` and for ``all``/``any`` quantifiers; any other call is unsupported."""
    if not isinstance(node.func, ast.Name):
        raise _unsupported(node.func)
    name = node.func.id
    if name in scope:
        raise UnsupportedConstruct(f"call {name}")
    if any(isinstance(arg, ast.Starred) for arg in node.args) or node.keywords:
        raise UnsupportedConstruct(f"call {name}")
    if name == "len":
        if len(node.args) != 1:
            raise UnsupportedConstruct("call len")
        return f"py_len({_call_argument(node.args[0], scope, types)})"
    if name in ("all", "any"):
        if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
            raise UnsupportedConstruct(f"call {name}")
        return _emit_quantifier(node.args[0], scope, types, forall=name == "all")
    if name == "range":
        raise UnsupportedConstruct("call range")
    raise UnsupportedConstruct(f"call {name}")


def _is_range(node: ast.AST, scope: Scope) -> ast.Call | None:
    """``node`` when it is a ``range`` call with one or two arguments, else ``None``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and node.func.id not in scope
        and not node.keywords
        and not any(isinstance(arg, ast.Starred) for arg in node.args)
        and len(node.args) in (1, 2)
    ):
        return node
    return None


def _fresh_index(taken: set[str]) -> str:
    """A Dafny index variable not in ``taken``: ``i``, ``j``, ``k``, then ``i_0`` and on."""
    for candidate in ("i", "j", "k"):
        if candidate not in taken:
            return candidate
    index = 0
    while f"i_{index}" in taken:
        index += 1
    return f"i_{index}"


def _emit_quantifier(node: ast.GeneratorExp, scope: Scope, types: dict[str, str], *, forall: bool) -> str:
    """Dafny text for ``all``/``any`` over ``range`` bounds and over list indices."""
    keyword = "forall" if forall else "exists"
    link = "==>" if forall else "&&"
    for generator in node.generators:
        if generator.ifs:
            raise UnsupportedConstruct("comprehension if")
        if not isinstance(generator.target, ast.Name):
            raise UnsupportedConstruct("comprehension target")
    # Dafny names already spoken for: the parameters, ``result`` and every bound or index variable so far.
    # Index variables come from here (never shadowing a parameter), and a generator variable that collides is
    # suffixed, so the emitted binder can never capture a name the body resolves to an outer scope.
    taken = {_rename(name) for name in types}
    bound: list[str] = []
    variables: list[str] = []
    for generator in node.generators:
        assert isinstance(generator.target, ast.Name)
        variable = generator.target.id
        renamed = _rename(variable)
        while renamed in taken:
            renamed += "_"
        taken.add(renamed)
        range_call = _is_range(generator.iter, scope)
        if range_call is not None:
            if len(range_call.args) == 2:
                (lower, upper) = range_call.args
            else:
                (upper,) = range_call.args
                lower = ast.Constant(0)
            # Bounds evaluate in the enclosing scope, before the variable is bound.
            bound.append(
                f"{_comparison_operand(lower, scope, types)} <= {renamed} < {_comparison_operand(upper, scope, types)}"
            )
            scope = {**scope, variable: renamed}
            variables.append(renamed)
        else:
            element = _emit(generator.iter, scope, types)
            if _precedence(generator.iter) < _PREC_ATOM:
                element = f"({element})"
            index = _fresh_index(taken)
            taken.add(index)
            scope = {**scope, variable: f"{element}[{index}]"}
            bound.append(f"0 <= {index} < |{element}|")
            variables.append(index)
    body = _emit(node.elt, scope, types)
    if isinstance(node.elt, ast.IfExp) or _is_quantifier(node.elt):
        body = f"({body})"
    return f"{keyword} {', '.join(variables)} :: {' && '.join(bound)} {link} {body}"


def _emit_subscript(node: ast.Subscript, scope: Scope, types: dict[str, str]) -> str:
    """Dafny text for indexing through ``py_index`` and for ``[::-1]`` through ``py_reversed``."""
    value = _call_argument(node.value, scope, types)
    if isinstance(node.slice, ast.Slice):
        if (
            node.slice.lower is None
            and node.slice.upper is None
            and isinstance(node.slice.step, ast.UnaryOp)
            and isinstance(node.slice.step.op, ast.USub)
            and isinstance(node.slice.step.operand, ast.Constant)
            and node.slice.step.operand.value == 1
        ):
            return f"py_reversed({value})"
        raise UnsupportedConstruct(f"slice {ast.unparse(node.slice)}")
    if isinstance(node.slice, ast.Tuple):
        raise UnsupportedConstruct(f"slice {ast.unparse(node.slice)}")
    index = _emit(node.slice, scope, types)
    if _precedence(node.slice) <= _PREC_COMPARE:
        index = f"({index})"
    return f"py_index({value}, {index})"


def _single_return(source: str, kind: str) -> ast.expr:
    """The returned expression of the ``kind`` (``"spec"`` or ``"pre"``) function in ``source``."""
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise UnsupportedConstruct(f"{kind} shape") from error
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise UnsupportedConstruct(f"{kind} shape") from None
    definition = module.body[0]
    if definition.name != kind or definition.decorator_list:
        raise UnsupportedConstruct(f"{kind} shape") from None
    body = definition.body
    if (
        len(body) == 2
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise UnsupportedConstruct(f"{kind} shape") from None
    return body[0].value


#: The code-mode rename map without the ``result`` entry: spec parameters colliding
#: with a Dafny keyword are renamed to ``<name>_``, while ``result`` keeps its name.
_SPEC_RESERVED = RESERVED - {"result"}


def _rename(name: str) -> str:
    """``name`` with ``_`` appended when it collides with a Dafny keyword, unchanged otherwise."""
    return f"{name}_" if name in _SPEC_RESERVED else name


@dataclass(frozen=True, slots=True)
class _Parameter:
    """One function parameter: its Python name, its Dafny name and its Dafny type."""

    python: str  # as written in the function signature
    dafny: str  # ``_rename(python)``
    dafny_type: str  # ``dafny_type`` of the annotation


def _function_parameters(source: str) -> tuple[_Parameter, ...]:
    """The positional and keyword-only parameters of the function ``source``, with Dafny names and types.

    A parameter without an annotation raises ``UnsupportedConstruct("no types")``; ``*args`` and ``**kwargs``
    raise ``UnsupportedConstruct("parameter *<name>")``.
    """
    (definition,) = ast.parse(source).body
    assert isinstance(definition, ast.FunctionDef)
    args = definition.args
    if args.vararg is not None:
        raise UnsupportedConstruct(f"parameter *{args.vararg.arg}")
    if args.kwarg is not None:
        raise UnsupportedConstruct(f"parameter **{args.kwarg.arg}")
    parameters: list[_Parameter] = []
    for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        tag = dafny_type(argument.annotation, lineno=argument.lineno)
        parameters.append(_Parameter(argument.arg, _rename(argument.arg), tag))
    return tuple(parameters)


def _function_return_type(source: str) -> str:
    """The Dafny type of the return annotation of the function ``source``.

    A missing return annotation raises ``UnsupportedConstruct("no types")``.
    """
    (definition,) = ast.parse(source).body
    assert isinstance(definition, ast.FunctionDef)
    return dafny_type(definition.returns, lineno=definition.lineno)


def transpile_spec(spec: Spec) -> str:
    """Spec mode: logical Dafny (``forall``/``exists`` instead of loops) as ``requires``/``ensures`` clauses.

    Returns ``predicate pre(<params>) { ... }`` (``true`` when there is no precondition) followed by
    ``predicate spec(<params>, result: <T>) requires pre(<params>) { ... }``. Parameter types come from the
    function's hints through ``pcc.transpile.types``; parameters colliding with a Dafny keyword are renamed to
    ``<name>_`` (the code-mode rename map without the ``result`` entry), while ``result`` keeps its name.
    Anything outside ``SPEC_CONSTRUCTS`` raises ``UnsupportedConstruct``.
    """
    parameters = _function_parameters(spec.function.source)
    result_type = _function_return_type(spec.function.source)
    names = ", ".join(f"{parameter.dafny}: {parameter.dafny_type}" for parameter in parameters)
    arguments = ", ".join(parameter.dafny for parameter in parameters)
    pre_scope = {parameter.python: parameter.dafny for parameter in parameters}
    pre_types = {parameter.python: parameter.dafny_type for parameter in parameters}
    spec_scope = {**pre_scope, "result": "result"}
    spec_types = {**pre_types, "result": result_type}
    if spec.precondition is None:
        pre_text = "true"
    else:
        pre_text = _emit(_single_return(spec.precondition, "pre"), pre_scope, pre_types)
    spec_text = _emit(_single_return(spec.predicate, "spec"), spec_scope, spec_types)
    head = f"predicate spec({f'{names}, ' if names else ''}result: {result_type})"
    return f"predicate pre({names}) {{ {pre_text} }}\n{head} requires pre({arguments}) {{ {spec_text} }}\n"


__all__ = ["SPEC_CONSTRUCTS", "transpile_spec"]
