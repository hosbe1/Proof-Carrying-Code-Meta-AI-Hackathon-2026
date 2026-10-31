"""Step 4, wrapper: the ``translated`` function the differential check runs in the sandbox.

The wrapper has the original's exact parameters and type hints (the symbolic search in #75 needs them), converts each
argument with its own hint (``to_dafny(arg, hint)``), calls the back-compiled Dafny method and converts the result
back with ``to_python``. The converter source is embedded into the generated module (read with ``inspect``), never
imported, so the sandbox script is self-contained besides the compiled Dafny module on ``sys.path``.

The symbolic search (#75) reuses this wrapper through :func:`symbolic_module_source`, which adds one importable
module defining ``original`` and ``translated`` with the same signature, both guarded by the compiled Dafny ``pre``.
"""

from __future__ import annotations

import ast
import inspect

import pcc.equivalence.convert as convert_module
from pcc.core import FunctionRef, UnsupportedConstruct
from pcc.sandbox import target_loader
from pcc.verifier import method_name, python_name


def _hint_source(hint: ast.expr) -> str:
    """The hint as evaluatable source; string annotations are unpacked to the inner expression."""
    if isinstance(hint, ast.Constant) and isinstance(hint.value, str):
        try:
            inner = ast.parse(hint.value, mode="eval").body
        except SyntaxError:
            raise UnsupportedConstruct(f"type {hint.value!r}") from None
        return ast.unparse(inner)
    return ast.unparse(hint)


def _definition(function: FunctionRef) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The first function definition in ``function.source``."""
    tree = ast.parse(function.source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        raise UnsupportedConstruct("no types")
    return definition


def _signature_parts(definition: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, str, list[str]]:
    """The ``def`` line pieces shared by the differential and the symbolic wrappers.

    Returns the parameter list with the original hints and defaults (positional-only, positional-or-keyword and
    keyword-only), the return hint, and the parameter names in order. ``*args``/``**kwargs`` and missing hints
    raise ``UnsupportedConstruct``, exactly as the differential wrapper reports them.
    """
    args = definition.args
    if args.vararg is not None or args.kwarg is not None:
        raise UnsupportedConstruct("unsupported *args or **kwargs")
    if definition.returns is None:
        raise UnsupportedConstruct("no types")

    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    for param in params:
        if param.annotation is None:
            raise UnsupportedConstruct("no types")
        _hint_source(param.annotation)  # validate string annotations the way the wrapper does
    return_hint = _hint_source(definition.returns)

    pos_params = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(pos_params) - len(args.defaults)) + list(args.defaults)
    parts: list[str] = []
    for param, default in zip(pos_params, defaults, strict=True):
        assert param.annotation is not None  # checked above
        piece = f"{param.arg}: {_hint_source(param.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        parts.append(piece)
    if args.posonlyargs:
        parts.insert(len(args.posonlyargs), "/")
    if args.kwonlyargs:
        parts.append("*")
        for param, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            assert param.annotation is not None  # checked above
            piece = f"{param.arg}: {_hint_source(param.annotation)}"
            if default is not None:
                piece += f" = {ast.unparse(default)}"
            parts.append(piece)
    return ", ".join(parts), return_hint, [param.arg for param in params]


def wrapper_source(function: FunctionRef, module_name: str) -> str:
    """Python source defining ``translated`` for ``function`` against Dafny module ``module_name``.

    ``module_name`` is the Dafny module name from ``verifier.layout.module_name`` (unescaped); both it and the
    method name from ``verifier.layout.method_name`` are escaped with ``verifier.python_name`` for the Python
    backend. The caller puts the compiled ``m-py`` directory on ``sys.path`` before executing this source.

    The signature copies positional-only, positional-or-keyword and keyword-only parameters with their hints and
    defaults, plus the return annotation, and imports ``typing`` so ``typing.List`` hints evaluate. ``*args`` and
    ``**kwargs`` raise ``UnsupportedConstruct``, as does a missing hint.
    """
    definition = _definition(function)
    if isinstance(definition, ast.AsyncFunctionDef):
        raise UnsupportedConstruct("async function")
    signature, return_hint, _ = _signature_parts(definition)

    escaped_module = python_name(module_name)
    escaped_method = python_name(method_name(function))
    converter_source = inspect.getsource(convert_module).rstrip("\n")
    params = [*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs]
    hint_sources = [_hint_source(param.annotation) for param in params if param.annotation is not None]
    call_args = ", ".join(f"_dafny_{param.arg}" for param in params)
    conversions = "\n".join(
        f"    _dafny_{param.arg} = to_dafny({param.arg}, {hint})"
        for param, hint in zip(params, hint_sources, strict=True)
    )
    lines = [
        "import typing",
        converter_source,
        f"import {escaped_module} as _dafny_module",
        f"def translated({signature}) -> {return_hint}:",
        conversions,
        f"    _dafny_result = _dafny_module.default__.{escaped_method}({call_args})",
        f"    return to_python(_dafny_result, {return_hint})",
        "",
    ]
    return "\n".join(lines)


def symbolic_module_source(function: FunctionRef, module_name: str) -> str:
    """Python source of the importable module the symbolic search compares entry points of.

    The module defines ``original`` and ``translated`` with the original's exact parameters, type hints, defaults
    and return annotation. Both evaluate the compiled Dafny ``pre`` on the converted arguments and return ``None``
    when it is false (a missing ``pre`` accepts every input), so inputs outside the precondition never differ.
    The compiled module is imported the same way the differential wrapper imports it, and ``pre`` is read as
    ``default__.<python_name("pre")>``; ``translated`` then calls the differential wrapper (renamed to
    ``_translated_impl``) and ``original`` calls the original Python function.
    """
    definition = _definition(function)
    if isinstance(definition, ast.AsyncFunctionDef):
        raise UnsupportedConstruct("async function")
    signature, return_hint, names = _signature_parts(definition)
    args = definition.args
    hint_sources = [
        _hint_source(param.annotation)
        for param in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if param.annotation is not None
    ]

    renamed = wrapper_source(function, module_name).replace("def translated(", "def _translated_impl(", 1)
    assert "def _translated_impl(" in renamed

    positional = [param.arg for param in [*args.posonlyargs, *args.args]]
    keywords = [param.arg for param in args.kwonlyargs]
    python_call = ", ".join([*positional, *[f"{name}={name}" for name in keywords]])
    dafny_call = ", ".join(f"_dafny_{name}" for name in names)
    conversions = "\n".join(
        f"    _dafny_{name} = to_dafny({name}, {hint})" for name, hint in zip(names, hint_sources, strict=True)
    )
    guarded = "\n".join(f"    {line}" if line.strip() else line for line in conversions.split("\n"))
    guard = (
        "    _pre = getattr(_dafny_module.default__, " + f"{python_name('pre')!r}, None)\n"
        "    if _pre is not None:\n"
        "        try:\n"
        f"            _holds = bool(_pre({dafny_call}) is True)\n"
        "        except Exception:\n"
        "            return None\n"
        "        if not _holds:\n"
        "            return None"
    )
    lines = [
        renamed.rstrip("\n"),
        target_loader(function).rstrip("\n"),
        f"def original({signature}) -> {return_hint}:",
        "    try:",
        guarded,
        "    except Exception:",
        "        return None",
        guard,
        f"    return target({python_call})",
        f"def translated({signature}) -> {return_hint}:",
        "    try:",
        guarded,
        "    except Exception:",
        "        return None",
        guard,
        f"    return _translated_impl({python_call})",
        "",
    ]
    return "\n".join(lines)


__all__ = ["symbolic_module_source", "wrapper_source"]
