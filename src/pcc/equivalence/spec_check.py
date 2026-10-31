"""Step 4, spec check: does the transpiled spec agree with the Python spec?

The transpiled ``pre`` and ``spec`` predicates are compiled back to Python and evaluated next to the Python
``pre``/``spec`` on the cases the pipeline passes in. A wrong spec translation (for example one that always
holds) would otherwise be proven while the code is wrong.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pcc.core import CheckKind, CheckRecord, FunctionRef, NotCheckable, Spec, ToolError
from pcc.sandbox import RunOutcome, run_python
from pcc.verifier import assemble, build_python, module_name, python_name

#: Sandbox script template. Placeholders (``@TOKEN@``) are filled in by :func:`check_spec` with JSON values, so no
#: quoting is needed. The script prints one line per problem and ``OK <n>`` when every case agrees:
#:
#: - ``ERROR tool <message>``: the compiled Dafny module cannot be imported; the driver raises ``ToolError``.
#: - ``ERROR check <message>``: the Python spec or the annotations cannot be evaluated; ``NotCheckable``.
#: - ``MISMATCH <index> pre|spec <python> <dafny>``: the first differing case; the driver reports it and stops.
#: - ``CASE-ERROR <index> <message>``: one case cannot be evaluated; the driver reports it like a mismatch.
#:
#: Every predicate call is guarded: an exception counts as ``False``, because a predicate that raises does not
#: hold. Compiled Dafny code does not enforce ``requires`` at run time, so the script only calls the Dafny
#: ``spec`` where the Python and Dafny ``pre`` agree that it holds. The compiled module is imported by its
#: derived name (``python_name(module_name(function))``), the same way the differential check imports it.
_SCRIPT_TEMPLATE = """\
import ast as _ast
import importlib as _importlib
import importlib.util as _importlib_util
import sys as _sys
import typing as _typing

_sys.path.insert(0, @MODULE_DIR@)

_CONVERT_SPEC = _importlib_util.spec_from_file_location("pcc_check_spec_convert", @CONVERT_PATH@)
if _CONVERT_SPEC is None or _CONVERT_SPEC.loader is None:
    print("ERROR tool cannot load the value converter")
    raise SystemExit(1)
_CONVERT = _importlib_util.module_from_spec(_CONVERT_SPEC)
_sys.modules["pcc_check_spec_convert"] = _CONVERT
_CONVERT_SPEC.loader.exec_module(_CONVERT)
_to_dafny = _CONVERT.to_dafny

# The compiled predicates live in the module derived from the function, escaped the way the Dafny Python
# backend escapes it (``_`` becomes ``__``); every check runs in a fresh sandbox process, so a rebuilt module
# with the same name is never shadowed by an earlier import.
try:
    _DAFNY_MOD = _importlib.import_module(@MODULE_NAME@)
    _DAFNY = _DAFNY_MOD.default__
    _dafny_spec_fn = getattr(_DAFNY, @SPEC_NAME@)
except Exception as _error:
    print(f"ERROR tool cannot import the compiled spec module: {_error}")
    raise SystemExit(1)
_DAFNY_PRE = getattr(_DAFNY, @PRE_NAME@, None)

_PY_NAMESPACE: dict = {}
try:
    exec(@PREDICATE@, _PY_NAMESPACE)
    _py_spec = _PY_NAMESPACE["spec"]
except Exception as _error:
    print(f"ERROR check cannot evaluate the Python spec: {_error}")
    raise SystemExit(1)

_PRECONDITION_SOURCE = @PRECONDITION@
if _PRECONDITION_SOURCE is None:
    _py_pre = None
else:
    _PRE_NAMESPACE: dict = {}
    try:
        exec(_PRECONDITION_SOURCE, _PRE_NAMESPACE)
        _py_pre = _PRE_NAMESPACE["pre"]
    except Exception as _error:
        print(f"ERROR check cannot evaluate the Python precondition: {_error}")
        raise SystemExit(1)

_TYPE_ALIASES = {
    "typing": _typing,
    "List": _typing.List,
    "Dict": _typing.Dict,
    "Tuple": _typing.Tuple,
    "Optional": _typing.Optional,
    "Set": _typing.Set,
    "FrozenSet": _typing.FrozenSet,
    "Any": _typing.Any,
}


def _resolve(source, aliases):
    try:
        _tree = _ast.parse(source)
    except SyntaxError as _error:
        return None, f"cannot parse the function source: {_error}"
    _func = None
    for _node in _ast.walk(_tree):
        if isinstance(_node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            _func = _node
            break
    if _func is None:
        return None, "no function definition found"
    if _func.args.vararg is not None or _func.args.kwarg is not None or _func.args.kwonlyargs:
        return None, "only plain positional parameters are supported"
    _names = [_a.arg for _a in (*_func.args.posonlyargs, *_func.args.args)]
    _segments = [_ast.get_source_segment(source, _a.annotation) for _a in (*_func.args.posonlyargs, *_func.args.args)]
    _types = []
    for _name, _segment in zip(_names, _segments, strict=True):
        if not _segment:
            return None, f"parameter {_name!r} has no type annotation"
        try:
            _evaluated = eval(_segment, dict(aliases))
            if isinstance(_evaluated, str):
                _evaluated = eval(_evaluated, dict(aliases))
        except Exception as _error:
            return None, f"cannot resolve the annotation of parameter {_name!r}: {_error}"
        _types.append(_evaluated)
    _defaults = {}
    _with_defaults = _names[len(_names) - len(_func.args.defaults):] if _func.args.defaults else []
    for _name, _node in zip(_with_defaults, _func.args.defaults, strict=True):
        _segment = _ast.get_source_segment(source, _node)
        try:
            _defaults[_name] = eval(_segment, dict(aliases))
        except Exception as _error:
            return None, f"cannot resolve the default of parameter {_name!r}: {_error}"
    if _func.returns is None:
        return None, "the function has no return annotation"
    _return_segment = _ast.get_source_segment(source, _func.returns)
    try:
        _return_type = eval(_return_segment, dict(aliases))
        if isinstance(_return_type, str):
            _return_type = eval(_return_type, dict(aliases))
    except Exception as _error:
        return None, f"cannot resolve the return annotation: {_error}"
    return (_names, _types, _defaults, _return_type), ""


_RESOLVED, _RESOLVE_ERROR = _resolve(@FUNC_SOURCE@, _TYPE_ALIASES)
if _RESOLVED is None:
    print(f"ERROR check {_RESOLVE_ERROR}")
    raise SystemExit(1)
_PARAM_NAMES, _PARAM_TYPES, _DEFAULTS, _RETURN_TYPE = _RESOLVED

_CAPTURED: dict = {}


def _cap(*args, **kwargs):
    _CAPTURED["args"] = args
    _CAPTURED["kwargs"] = kwargs


for _index, (_args_source, _result_source) in enumerate(@CASES@):
    try:
        eval(f"_cap({_args_source})", {"_cap": _cap})
        _argv = list(_CAPTURED["args"])
        _kwargs = dict(_CAPTURED["kwargs"])
        _expected = eval(_result_source, {})
        _ordered = list(_argv)
        for _name in _PARAM_NAMES[len(_ordered):]:
            if _name in _kwargs:
                _ordered.append(_kwargs.pop(_name))
            elif _name in _DEFAULTS:
                _ordered.append(_DEFAULTS[_name])
            else:
                raise TypeError(f"missing argument {_name!r}")
        if _kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(_kwargs)}")
        if len(_ordered) != len(_PARAM_TYPES):
            raise TypeError("too many positional arguments")
        _dafny_args = [_to_dafny(_v, _t) for _v, _t in zip(_ordered, _PARAM_TYPES, strict=True)]
        _dafny_result = _to_dafny(_expected, _RETURN_TYPE)
    except Exception as _error:
        print(f"CASE-ERROR {_index} {_error}")
        break
    try:
        _python_pre = True if _py_pre is None else bool(_py_pre(*_ordered))
    except Exception:
        _python_pre = False
    try:
        _dafny_pre = True if _DAFNY_PRE is None else bool(_DAFNY_PRE(*_dafny_args))
    except Exception:
        _dafny_pre = False
    if _python_pre != _dafny_pre:
        print(f"MISMATCH {_index} pre {_python_pre} {_dafny_pre}")
        break
    if not _python_pre:
        continue
    try:
        _python_spec = bool(_py_spec(*_ordered, _expected))
    except Exception:
        _python_spec = False
    try:
        _dafny_spec = bool(_dafny_spec_fn(*_dafny_args, _dafny_result))
    except Exception:
        _dafny_spec = False
    if _python_spec != _dafny_spec:
        print(f"MISMATCH {_index} spec {_python_spec} {_dafny_spec}")
        break
else:
    print(f"OK {len(@CASES@)}")
"""


def check_spec(
    spec: Spec,
    dafny_spec: str,
    function: FunctionRef,
    repo_root: Path,
    *,
    cases: Sequence[tuple[str, str]],
) -> CheckRecord:
    """Evaluate the transpiled ``dafny_spec`` (compiled back to Python) next to the Python ``spec`` on ``cases``.

    ``cases`` holds ``(args, result)`` pairs: ``args`` is the Python source of the call arguments and ``result``
    is the ``repr`` of the expected result. The module holds only the predicates (``assemble`` with an empty
    function) and is compiled with ``build_python``; a ``NotCheckable`` from there passes through, as does a
    ``ToolError`` from anywhere. The original code runs from ``repo_root``, so its imports resolve as they do
    in the project.

    On every case the Python ``pre`` runs against the Dafny ``pre``; where ``pre`` holds, the Python ``spec``
    runs against the Dafny ``spec``, with values converted by the ``equivalence.convert`` converter. A case
    where a predicate raises counts as ``False``. Any disagreement gives ``passed=False`` with the first
    differing case in the detail. With no cases the check fails with detail ``"spec: no cases"``.
    """
    if not cases:
        return CheckRecord(CheckKind.DIFFERENTIAL, False, "spec: no cases")
    dafny_code = assemble(function, dafny_spec, "")
    with tempfile.TemporaryDirectory(prefix="pcc-spec-check-") as tmp:
        module_dir = build_python(dafny_code, Path(tmp))
        script = _script(spec, function, module_dir, cases)
        outcome = run_python(script, cwd=repo_root)
    return _interpret(outcome, cases)


def _script(
    spec: Spec,
    function: FunctionRef,
    module_dir: Path,
    cases: Sequence[tuple[str, str]],
) -> str:
    """The sandbox script that evaluates ``cases``, with every value embedded as JSON."""
    replacements = {
        "@MODULE_DIR@": json.dumps(str(module_dir)),
        "@MODULE_NAME@": json.dumps(python_name(module_name(function))),
        "@SPEC_NAME@": json.dumps(python_name("spec")),
        "@PRE_NAME@": json.dumps(python_name("pre")),
        "@CONVERT_PATH@": json.dumps(str(Path(__file__).with_name("convert.py"))),
        "@FUNC_SOURCE@": json.dumps(function.source),
        "@PREDICATE@": json.dumps(spec.predicate),
        # JSON has no missing value: a missing precondition is the Python ``None``, not ``null``.
        "@PRECONDITION@": "None" if spec.precondition is None else json.dumps(spec.precondition),
        "@CASES@": json.dumps([list(case) for case in cases]),
    }
    script = _SCRIPT_TEMPLATE
    for token, value in replacements.items():
        script = script.replace(token, value)
    return script


def _interpret(outcome: RunOutcome, cases: Sequence[tuple[str, str]]) -> CheckRecord:
    """Turn the sandbox transcript into a ``DIFFERENTIAL`` record; a broken harness is a ``ToolError``."""
    for line in outcome.stdout.splitlines():
        if line.startswith("ERROR "):
            parts = line.split(" ", 3)
            if len(parts) == 4:
                _, kind, _, message = parts
                if kind == "tool":
                    raise ToolError(f"spec: {message}")
                raise NotCheckable(f"spec: {message}")
        if line.startswith("MISMATCH "):
            parts = line.split(" ", 4)
            if len(parts) == 5:
                _, index_text, which, python_text, dafny_text = parts
                return _mismatch(cases, index_text, f"{which} differs (python={python_text}, dafny={dafny_text})")
        if line.startswith("CASE-ERROR "):
            parts = line.split(" ", 3)
            if len(parts) == 4:
                _, index_text, _, message = parts
                return _mismatch(cases, index_text, f"cannot evaluate ({message})")
        if line.startswith("OK "):
            return CheckRecord(
                CheckKind.DIFFERENTIAL,
                True,
                f"spec: {len(cases)} cases agree",
                inputs_tried=len(cases),
                inputs_total=len(cases),
            )
    detail = outcome.stderr.strip().splitlines()
    tail = detail[-1] if detail else f"exit {outcome.returncode}"
    raise ToolError(f"spec: sandbox evaluation failed ({tail})")


def _mismatch(cases: Sequence[tuple[str, str]], index_text: str, what: str) -> CheckRecord:
    """A failed record naming the first differing case; an unreadable index fails on the first case."""
    try:
        index = int(index_text)
        args, result = cases[index]
    except (ValueError, IndexError):
        index = 0
        args, result = cases[0]
    return CheckRecord(
        CheckKind.DIFFERENTIAL,
        False,
        f"spec: case {args} -> {result}: {what}",
        inputs_tried=index + 1,
        inputs_total=len(cases),
    )


__all__ = ["check_spec"]
