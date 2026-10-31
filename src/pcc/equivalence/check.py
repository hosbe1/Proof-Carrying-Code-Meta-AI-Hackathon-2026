"""Step 4, differential check: the back-compiled Dafny translation equals the original.

The translation is compiled back to Python (``dafny build --target:py``) and compared with the original on
``inputs`` random Hypothesis inputs plus up to 20 model-proposed edge cases, all run in the sandbox. Only inputs on
which the compiled Dafny ``pre`` holds count (a missing ``pre`` counts every input). The first mismatch becomes the
``mismatch`` counterexample.

When ``symbolic_seconds > 0``, the differential test is followed by the stretch symbolic search (#75): one importable
sandbox module defines ``original`` and ``translated`` with the original's exact typed signature, both guarded by the
compiled Dafny ``pre``, and ``diffbehavior`` compares them with ``--per_condition_timeout <symbolic_seconds>``. The
sandbox wall timeout is ``symbolic_seconds + 30``. The outcome is one more SYMBOLIC record; a symbolic input only
becomes ``mismatch`` when the differential test found none. ``symbolic_seconds == 0`` writes no SYMBOLIC record.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pcc.core import (
    CheckKind,
    CheckRecord,
    Counterexample,
    EquivalenceReport,
    FunctionRef,
    ToolError,
    Translation,
    UnsupportedConstruct,
)
from pcc.equivalence.wrapper import symbolic_module_source, wrapper_source
from pcc.models import Message, ModelClient
from pcc.sandbox import RunOutcome, run_python, strategies_source, target_loader
from pcc.verifier import assemble, build_python, module_name, python_name

#: The model step name for edge-case inputs, for cost reporting.
STEP = "edge_cases"

#: The JSON shape the model answer must follow: ``{"edge_cases": [...]}`` of call-argument sources.
SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"edge_cases": {"type": "array", "items": {"type": "string"}}},
    "required": ["edge_cases"],
    "additionalProperties": False,
}

#: Cap on model-proposed edge cases actually tried.
_MAX_EDGE_CASES = 20

_DETAIL_NEVER = "precondition never holds"
_DETAIL_TIMEOUT = "timeout"

#: The importable sandbox module the symbolic search compares entry points of.
_CROSSHAIR_MODULE = "pcc_symbolic"
#: The wall timeout on top of ``symbolic_seconds`` bounding the whole symbolic sandbox run.
_SYMBOLIC_GRACE_SECONDS = 30.0
#: Outcome of a crashed symbolic run with blank stderr.
_DETAIL_CROSSHAIR_FAILED = "crosshair failed"
#: Outcome of a symbolic run that found no difference.
_DETAIL_NO_DIFFERENCES = "no differences found"
#: The success line crosshair diffbehavior prints on exit 0 (capital N, with a period).
_CROSSHAIR_NO_DIFFERENCES = "No differences found."

#: Prefix marking the collected-inputs line on stdout, so prints from the target cannot be mistaken for it.
_MARKER_INPUTS = "PCC_EQUIV_INPUTS "
#: Prefix marking progress lines of the evaluation run.
_MARKER_PROGRESS = "PCC_EQUIV_PROGRESS "
#: Prefix marking the final evaluation result line on stdout.
_MARKER_DONE = "PCC_EQUIV_DONE "

#: Sandbox timeout for the single Hypothesis collection run (no translation involved).
_COLLECT_TIMEOUT_SECONDS = 120.0
#: Sandbox timeout for the evaluation run over all inputs.
_EVAL_TIMEOUT_SECONDS = 300.0
#: Sandbox timeout for pinpointing one hanging input.
_PINPOINT_TIMEOUT_SECONDS = 20.0
#: Progress heartbeat every this many evaluated inputs, to stay far below the 1 MB output cap.
_PROGRESS_EVERY = 100


class _ScriptTimeout(Exception):
    """A sandbox script produced no result marker but timed out."""


def _param_names(function: FunctionRef) -> list[str]:
    """Parameter names in order (positional-only, positional-or-keyword, then keyword-only)."""
    tree = ast.parse(function.source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        raise UnsupportedConstruct("no types")
    return [arg.arg for arg in [*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs]]


def _hint_sources(function: FunctionRef) -> tuple[list[str], str]:
    """Evaluatable hint sources in parameter order plus the return hint source.

    String annotations are unpacked to the inner expression. A missing hint raises
    ``UnsupportedConstruct("no types")``; ``*args``/``**kwargs`` raise ``UnsupportedConstruct``.
    """
    tree = ast.parse(function.source)
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        raise UnsupportedConstruct("no types")
    if isinstance(definition, ast.AsyncFunctionDef):
        raise UnsupportedConstruct("async function")
    if definition.args.vararg is not None or definition.args.kwarg is not None:
        raise UnsupportedConstruct("unsupported *args or **kwargs")
    if definition.returns is None:
        raise UnsupportedConstruct("no types")
    params = [*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs]
    for param in params:
        if param.annotation is None:
            raise UnsupportedConstruct("no types")

    def _source(hint: ast.expr | None) -> str:
        assert hint is not None
        if isinstance(hint, ast.Constant) and isinstance(hint.value, str):
            try:
                inner = ast.parse(hint.value, mode="eval").body
            except SyntaxError:
                raise UnsupportedConstruct(f"type {hint.value!r}") from None
            return ast.unparse(inner)
        return ast.unparse(hint)

    return [_source(param.annotation) for param in params], _source(definition.returns)


def _require_convertible_hints(function: FunctionRef) -> None:
    """Raise ``UnsupportedConstruct`` when a hint is outside the converter language (tier 1).

    ``strategies_source`` supports more (``float``, ``Optional``) than ``convert`` does; without this check legally
    generated inputs would fail conversion and look like translation bugs. ``TypeError`` from a trial conversion is
    therefore never a mismatch.
    """
    import typing

    hints, return_hint = _hint_sources(function)
    namespace: dict[str, object] = {
        "typing": typing,
        "list": list,
        "tuple": tuple,
        "int": int,
        "bool": bool,
        "str": str,
    }
    for source in [*hints, return_hint]:
        try:
            py_type = eval(source, {"__builtins__": {}}, namespace)  # hint sources from the local AST
        except Exception:
            raise UnsupportedConstruct(f"type {source}") from None
        origin = typing.get_origin(py_type)
        args = typing.get_args(py_type)
        if py_type in (int, bool, str):
            continue
        if origin is list and len(args) == 1:
            _require_nested(args[0])
            continue
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                _require_nested(args[0])
                continue
            if args == ((),):
                continue
            if args and all(argument is not ... for argument in args):
                for argument in args:
                    _require_nested(argument)
                continue
        raise UnsupportedConstruct(f"type {source}")


def _require_nested(py_type: object) -> None:
    """Raise ``TypeError`` unless ``py_type`` is a nested tier-1 element type."""
    import typing

    if py_type in (int, bool, str):
        return
    origin = typing.get_origin(py_type)
    args = typing.get_args(py_type)
    if origin is list and len(args) == 1:
        _require_nested(args[0])
        return
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            _require_nested(args[0])
            return
        if args == ((),):
            return
        if args and all(argument is not ... for argument in args):
            for argument in args:
                _require_nested(argument)
            return
    raise TypeError(f"unsupported nested type: {py_type!r}")


def _collect_snippet(max_examples: int) -> str:
    """The Hypothesis run collecting argument call sources (no target calls, so no hang risk)."""
    return (
        "import hypothesis\n"
        "from hypothesis import given, settings\n"
        "def _safe_repr(_value):\n"
        "    try:\n"
        "        return repr(_value)\n"
        "    except Exception:\n"
        "        return '<unrepresentable>'\n"
        "_collected = []\n"
        "@given(args_strategy)\n"
        f"@settings(max_examples={max_examples}, database=None, derandomize=True, deadline=None)\n"
        "def _collect(_args):\n"
        "    _collected.append(', '.join(_safe_repr(_v) for _v in _args))\n"
        "_collect()\n"
    )


def _inputs_script(function: FunctionRef, max_examples: int) -> str:
    """A sandbox script printing one ``PCC_EQUIV_INPUTS`` line with generated call sources."""
    chunks = [strategies_source(function), _collect_snippet(max_examples)]
    chunks.append(f"import json\nprint({_MARKER_INPUTS!r} + json.dumps({{'inputs': _collected}}))\n")
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _marker_payload(stdout: str, marker: str) -> dict[str, object] | None:
    """The last marker-line payload in ``stdout``, or ``None`` when there is none."""
    payload: dict[str, object] | None = None
    for line in stdout.splitlines():
        if not line.startswith(marker):
            continue
        try:
            data = json.loads(line[len(marker) :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            payload = data
    return payload


def _progress_index(stdout: str) -> int | None:
    """The last evaluated index from progress lines, or ``None`` when there are none."""
    index: int | None = None
    for line in stdout.splitlines():
        if not line.startswith(_MARKER_PROGRESS):
            continue
        try:
            data = json.loads(line[len(_MARKER_PROGRESS) :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("idx"), int):
            index = data["idx"]
    return index


def _run_script(script: str, repo_root: Path, marker: str, *, timeout_seconds: float) -> dict[str, object]:
    """Run ``script`` in the sandbox and return its marker payload.

    A script whose marker parses is used as is, even on timeout. A run with no marker raises ``_ScriptTimeout``
    on timeout and ``ToolError`` otherwise.
    """
    outcome = run_python(script, cwd=repo_root, timeout_seconds=timeout_seconds)
    payload = _marker_payload(outcome.stdout, marker)
    if payload is not None:
        return payload
    if outcome.timed_out:
        raise _ScriptTimeout(f"sandbox run timed out after printing no result (exit {outcome.returncode})")
    raise ToolError(
        "equivalence sandbox run produced no result "
        f"(exit {outcome.returncode}, timed out {outcome.timed_out}): {outcome.stderr.strip()}"
    )


def _model_edge_cases(function: FunctionRef, *, client: ModelClient, model: str) -> tuple[str, ...]:
    """Ask the model once for up to 20 edge-case inputs; invalid entries are dropped.

    Each entry is a call-argument source (as in ``target(<entry>)``). An empty or malformed answer contributes
    zero edge cases. A ``ToolError`` from the client passes through.
    """
    names = _param_names(function)
    lines = [
        f"Function {function.qualname} ({function.path}:{function.lineno}):",
        function.source.rstrip("\n"),
        "",
        "Propose up to 20 edge-case inputs for this function: empty collections, negative numbers, boundaries.",
        "Each edge case must be the call arguments only, as in `target(<arguments>)`, for example `'abc'` or `0, -2`.",
        'Reply as JSON with one key "edge_cases": a list of strings, each one call-argument source.',
    ]
    messages: tuple[Message, ...] = (
        Message("system", "You propose edge-case inputs for differential testing. Reply with JSON only."),
        Message("user", "\n".join(lines)),
    )
    completion = client.complete(model, messages, step=STEP, schema=SCHEMA)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, dict):
        return ()
    raw = data.get("edge_cases")
    if not isinstance(raw, list):
        return ()
    valid: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip() or entry in seen:
            continue
        if _valid_call_source(entry, len(names), set(names)):
            valid.append(entry)
            seen.add(entry)
        if len(valid) >= _MAX_EDGE_CASES:
            break
    return tuple(valid)


def _valid_call_source(source: str, arity: int, names: set[str]) -> bool:
    """Whether ``source`` parses as ``target(<source>)`` with matching arity and known keywords."""
    try:
        tree = ast.parse(f"target({source})", mode="eval")
    except SyntaxError:
        return False
    call = tree.body
    if not isinstance(call, ast.Call):
        return False
    for argument in call.args:
        if isinstance(argument, ast.Starred):
            return False
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in names:
            return False
    return len(call.args) <= arity


def _eval_script(
    function: FunctionRef,
    dafny_module: str,
    wrapper: str,
    inputs: list[str],
    pre_name: str,
) -> str:
    """A sandbox script evaluating ``inputs`` against the original and the translation.

    Prints a progress heartbeat every ``_PROGRESS_EVERY`` inputs and one final ``PCC_EQUIV_DONE`` line with the
    pre-holding count and the first mismatch. Per-input lines are deliberately not printed, so a 10,000-input run
    stays far below the 1 MB output cap; a timeout is attributed by pinpointing the hanging input afterwards.
    """
    hints, _ = _hint_sources(function)
    names = _param_names(function)
    assign = "\n".join(f"_HINTS.append({_hint})" for _hint in hints)
    chunks = [
        "import ast",
        "import inspect",
        "import json",
        target_loader(function),
        "import typing",
        "_HINTS = []",
        assign,
        f"_NAMES = {names!r}",
        wrapper,
        "def _safe_repr(_value):",
        "    try:",
        "        return repr(_value)",
        "    except Exception:",
        "        return '<unrepresentable>'",
        "def _parse_src(_src):",
        "    try:",
        "        _tree = ast.parse('target(' + _src + ')', mode='eval')",
        "    except SyntaxError:",
        "        return None",
        "    _call = _tree.body",
        "    if not isinstance(_call, ast.Call):",
        "        return None",
        "    _pos = []",
        "    for _node in _call.args:",
        "        if isinstance(_node, ast.Starred):",
        "            return None",
        "        try:",
        "            _pos.append(ast.literal_eval(_node))",
        "        except Exception:",
        "            return None",
        "    _kw = {}",
        "    for _keyword in _call.keywords:",
        "        if _keyword.arg is None or _keyword.arg not in _NAMES:",
        "            return None",
        "        try:",
        "            _kw[_keyword.arg] = ast.literal_eval(_keyword.value)",
        "        except Exception:",
        "            return None",
        "    return (_pos, _kw)",
        "def _ordered(_pos, _kw):",
        "    try:",
        "        _bound = inspect.signature(target).bind(*_pos, **_kw)",
        "    except TypeError:",
        "        return None",
        "    _bound.apply_defaults()",
        "    try:",
        "        return tuple(_bound.arguments[_name] for _name in _NAMES)",
        "    except KeyError:",
        "        return None",
        "_pre = getattr(_dafny_module.default__, " + f"{pre_name!r}, None)",
        "def _call_bound(_fn, _pos, _kw):",
        "    _sig = inspect.signature(_fn)",
        "    _binding = {}",
        "    _positional = [n for n, p in _sig.parameters.items() if p.kind in"
        " (inspect.Parameter.POSITIONAL_ONLY,"
        " inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]",
        "    for _name, _value in zip(_positional, _pos, strict=False):",
        "        _binding[_name] = _value",
        "    _binding.update({k: v for k, v in _kw.items() if k in _sig.parameters})",
        "    _posonly = [n for n, p in _sig.parameters.items() if p.kind is inspect.Parameter.POSITIONAL_ONLY]",
        "    _args = tuple(_binding[n] for n in _posonly)",
        "    _kwargs = {k: v for k, v in _binding.items() if k not in _posonly and k in _sig.parameters}",
        "    return _fn(*_args, **_kwargs)",
        f"_INPUTS = {inputs!r}",
        "_tried = 0",
        "_first = None",
        "for _idx, _src in enumerate(_INPUTS):",
        "    _parsed = _parse_src(_src)",
        "    if _parsed is None:",
        "        continue",
        "    _pos, _kw = _parsed",
        "    _ordered_args = _ordered(_pos, _kw)",
        "    if _ordered_args is None:",
        "        continue",
        "    try:",
        "        _dafny_args = tuple(to_dafny(_v, _h) for _v, _h in zip(_ordered_args, _HINTS, strict=True))",
        "    except TypeError:",
        "        continue",
        "    except Exception:",
        "        continue",
        "    try:",
        "        _holds = True if _pre is None else bool(_pre(*_dafny_args) is True)",
        "    except Exception:",
        "        continue",
        "    if not _holds:",
        "        continue",
        "    _tried += 1",
        "    try:",
        "        _orig_value = _call_bound(target, _pos, _kw)",
        "    except Exception as _exc:",
        "        _orig = ['raises', type(_exc).__name__]",
        "    else:",
        "        _orig = ['return', _safe_repr(_orig_value)]",
        "    try:",
        "        _trans_value = _call_bound(translated, _pos, _kw)",
        "    except Exception as _exc:",
        "        _trans = ['raises', type(_exc).__name__]",
        "    else:",
        "        _trans = ['return', _safe_repr(_trans_value)]",
        "    if _orig[0] == 'return' and _trans[0] == 'return':",
        "        _equal = _orig[1] == _trans[1]",
        "    elif _orig[0] == 'raises' and _trans[0] == 'raises':",
        "        _equal = True",
        "    else:",
        "        _equal = False",
        "    if not _equal and _first is None:",
        "        _first = {'arguments': _src, 'orig': _orig, 'trans': _trans}",
        f"    if _idx % {_PROGRESS_EVERY} == 0:",
        f"        print({_MARKER_PROGRESS!r} + json.dumps({{'idx': _idx}}), flush=True)",
        "print(",
        f"    {_MARKER_DONE!r} + json.dumps({{'tried': _tried, 'total': len(_INPUTS), 'mismatch': _first}}),",
        "    flush=True,",
        ")",
        "",
    ]
    _ = dafny_module  # the wrapper already imports the compiled module; kept for signature clarity
    return "\n".join(chunks)


def _single_script(
    function: FunctionRef,
    dafny_module: str,
    wrapper: str,
    source: str,
    pre_name: str,
) -> str:
    """A sandbox script evaluating one call-argument source ``source``."""
    return _eval_script(function, dafny_module, wrapper, [source], pre_name)


def _original_probe_script(function: FunctionRef, source: str) -> str:
    """A sandbox script running ``source`` against the original only."""
    chunks = [
        "import ast",
        "import inspect",
        "import json",
        target_loader(function),
        "def _safe_repr(_value):",
        "    try:",
        "        return repr(_value)",
        "    except Exception:",
        "        return '<unrepresentable>'",
        f"_SRC = {source!r}",
        "_tree = ast.parse('target(' + _SRC + ')', mode='eval')",
        "_call = _tree.body",
        "_pos = [ast.literal_eval(n) for n in _call.args]",
        "_kw = {k.arg: ast.literal_eval(k.value) for k in _call.keywords}",
        "try:",
        "    _value = target(*_pos, **_kw)",
        "except Exception as _exc:",
        "    print('PCC_EQUIV_ORIG ' + json.dumps({'kind': 'raises', 'payload': type(_exc).__name__}), flush=True)",
        "else:",
        "    print('PCC_EQUIV_ORIG ' + json.dumps({'kind': 'return', 'payload': _safe_repr(_value)}), flush=True)",
        "",
    ]
    return "\n".join(chunks)


def _describe(kind: str, payload: str) -> str:
    """The human-readable outcome part: ``returns <repr>``, ``raises <Exc>`` or ``timeout``."""
    if kind == "raises":
        return f"raises {payload}"
    if kind == "timeout":
        return _DETAIL_TIMEOUT
    return f"returns {payload}"


def _test_name(qualname: str) -> str:
    """The pytest function name for ``qualname``."""
    return "test_" + "".join(char if char.isalnum() else "_" for char in qualname) + "_differs"


def _test_source(
    function: FunctionRef,
    arguments: str,
    orig_kind: str,
    orig_payload: str,
    trans_kind: str,
    trans_payload: str,
) -> str:
    """A ready-to-run pytest module replaying ``arguments`` against the original.

    The translation lives in a build-specific temporary directory that is gone after the check, so it cannot be
    imported here. The test replays the original (bound through ``target_loader``) and records both outcomes: the
    expected original outcome is asserted, the actual translation outcome is quoted in a comment. When the original
    itself times out, replaying it would hang too, so the test only records the hanging input.
    """
    chunks = [target_loader(function), "import pytest"]
    expected_comment = (
        f"# original {_describe(orig_kind, orig_payload)}, translation {_describe(trans_kind, trans_payload)}"
    )
    if orig_kind == "timeout":
        chunks.append(
            f"{expected_comment}\n"
            f"def {_test_name(function.qualname)}():\n"
            f"    # both sides time out on ({arguments}); replaying would hang.\n"
            "    pass\n"
        )
    elif orig_kind == "raises":
        chunks.append(
            f"{expected_comment}\n"
            f"def {_test_name(function.qualname)}():\n"
            f"    with pytest.raises({orig_payload}):\n"
            f"        target({arguments})\n"
        )
    else:
        chunks.append(
            f"{expected_comment}\n"
            f"def {_test_name(function.qualname)}():\n"
            f"    result = target({arguments})\n"
            f"    assert repr(result) == {orig_payload!r}\n"
        )
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _mismatch_counterexample(
    function: FunctionRef,
    arguments: str,
    orig: list[str],
    trans: list[str],
) -> Counterexample:
    """A ``Counterexample`` for one evaluated input: expected is the original, actual the translation."""
    return Counterexample(
        arguments=arguments,
        expected=_describe(str(orig[0]), str(orig[1])),
        actual=_describe(str(trans[0]), str(trans[1])),
        test=_test_source(function, arguments, str(orig[0]), str(orig[1]), str(trans[0]), str(trans[1])),
    )


def _symbolic_argv(symbolic_seconds: float) -> list[str]:
    """The prescribed ``diffbehavior`` invocation: per-condition budget, then the two entry points."""
    return [
        "crosshair",
        "diffbehavior",
        "--per_condition_timeout",
        str(symbolic_seconds),
        f"{_CROSSHAIR_MODULE}.original",
        f"{_CROSSHAIR_MODULE}.translated",
    ]


def _symbolic_script(module_source: str, built: Path, symbolic_seconds: float) -> str:
    """A sandbox script writing ``module_source`` next to the build and running the prescribed call in-process.

    The module lands as ``pcc_symbolic.py`` in the compiled ``m-py`` directory (already on ``sys.path``, cleaned up
    with the build directory), then ``sys.argv`` is set and ``crosshair.main.main()`` is called. No executable runs.
    """
    chunks = [
        "import pathlib",
        "import sys",
        f"sys.path.insert(0, {str(built)!r})",
        f"pathlib.Path({str(built)!r}, {_CROSSHAIR_MODULE + '.py'!r}).write_text({module_source!r}, encoding='utf-8')",
        f"sys.argv = {_symbolic_argv(symbolic_seconds)!r}",
        "import crosshair.main",
        "crosshair.main.main()",
        "",
    ]
    return "\n".join(chunks)


def _given_arguments(stdout: str) -> str | None:
    """The call-argument source of the first ``Given: (...)`` line, or ``None`` when there is none."""
    for line in stdout.splitlines():
        if not line.startswith("Given:"):
            continue
        text = line[len("Given:") :].strip()
        if text.endswith(","):
            text = text[:-1].strip()
        if len(text) >= 2 and text.startswith("(") and text.endswith(")"):
            return text[1:-1]
        return None
    return None


def _parse_describe(text: str) -> tuple[str, str] | None:
    """The ``(kind, payload)`` of a ``diffbehavior`` result description, or ``None`` when unreadable."""
    core = text.split(", after execution ", 1)[0].strip()
    if core.startswith("returns "):
        return ("return", core[len("returns ") :].strip())
    if core.startswith("raises "):
        rest = core[len("raises ") :].split()
        return ("raises", rest[0]) if rest else None
    return None


def _symbolic_difference(stdout: str) -> tuple[str, tuple[str, str] | None, tuple[str, str] | None] | None:
    """The differing input of an exit-1 run: arguments plus the two parsed outcomes (``None`` when unreadable)."""
    arguments = _given_arguments(stdout)
    if arguments is None:
        return None
    original: tuple[str, str] | None = None
    translated: tuple[str, str] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        for entry, slot in (
            (f"{_CROSSHAIR_MODULE}.original", "original"),
            (f"{_CROSSHAIR_MODULE}.translated", "translated"),
        ):
            if not stripped.startswith(entry + " :"):
                continue
            describe = stripped.split(" : ", 1)[1] if " : " in stripped else ""
            parsed = _parse_describe(describe)
            if slot == "original":
                original = parsed
            else:
                translated = parsed
    return (arguments, original, translated)


def _replay_test_source(function: FunctionRef, arguments: str) -> str:
    """A ready-to-run pytest module replaying ``arguments`` against the original without asserting an outcome."""
    chunks = [
        target_loader(function),
        f"def {_test_name(function.qualname)}():\n    target({arguments})\n",
    ]
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _symbolic_counterexample(
    function: FunctionRef,
    arguments: str,
    original: tuple[str, str] | None,
    translated: tuple[str, str] | None,
) -> Counterexample:
    """A ``Counterexample`` for a symbolic input: parsed outcomes when readable, a replay test otherwise."""
    if original is not None and translated is not None:
        return _mismatch_counterexample(function, arguments, [original[0], original[1]], [translated[0], translated[1]])
    expected = _describe(original[0], original[1]) if original is not None else "unknown"
    actual = _describe(translated[0], translated[1]) if translated is not None else "unknown"
    return Counterexample(
        arguments=arguments,
        expected=expected,
        actual=actual,
        test=_replay_test_source(function, arguments),
    )


def _symbolic_record(
    symbolic_seconds: float, outcome: RunOutcome
) -> tuple[CheckRecord, tuple[str, tuple[str, str] | None, tuple[str, str] | None] | None]:
    """The SYMBOLIC record for a sandbox ``outcome``, plus the differing input when one was found.

    Exit 0 with ``No differences found.`` passes; the wall timeout passes with the within-budget detail because no
    difference was found. Exit 1 with a ``Given: (...)`` line fails on that input (with the parsed outcomes, which
    may be ``None`` when unreadable); exit 2 or malformed output fails with the last nonblank stderr line
    (``crosshair failed`` when blank). Only a found input yields a difference; a crash never does. A ``ToolError``
    from the sandbox run escapes from the caller.
    """
    if outcome.timed_out:
        return CheckRecord(CheckKind.SYMBOLIC, True, f"no differences found within {symbolic_seconds:g} s"), None
    if outcome.returncode == 0 and _CROSSHAIR_NO_DIFFERENCES in outcome.stdout:
        return CheckRecord(CheckKind.SYMBOLIC, True, _DETAIL_NO_DIFFERENCES), None
    if outcome.returncode == 1:
        difference = _symbolic_difference(outcome.stdout)
        if difference is not None:
            arguments, original, translated = difference
            if original is not None and translated is not None:
                detail = (
                    f"{arguments} -> original {_describe(original[0], original[1])}, "
                    f"translation {_describe(translated[0], translated[1])}"
                )
            else:
                detail = arguments
            return CheckRecord(CheckKind.SYMBOLIC, False, detail), difference
    tail = [line for line in outcome.stderr.splitlines() if line.strip()]
    return CheckRecord(CheckKind.SYMBOLIC, False, tail[-1].strip() if tail else _DETAIL_CROSSHAIR_FAILED), None


def check(
    function: FunctionRef,
    translation: Translation,
    repo_root: Path,
    *,
    inputs: int,
    symbolic_seconds: float,
    client: ModelClient,
    model: str,
) -> EquivalenceReport:
    """Compile the translation back to Python and compare it with the original in the sandbox.

    ``inputs`` random Hypothesis inputs plus up to 20 model-proposed edge cases (one ``edge_cases`` call) run
    against both implementations; only inputs on which the compiled Dafny ``pre`` holds count. The result carries
    one DIFFERENTIAL record, plus one SYMBOLIC record when ``symbolic_seconds > 0``: the stretch symbolic search
    compares the original with the back-compiled translation on symbolic inputs. The symbolic input only becomes
    ``mismatch`` when the differential test found none; a crash never sets one. ``symbolic_seconds == 0`` keeps the
    differential-only result. A ``ToolError`` from the sandbox escapes.
    """
    _require_convertible_hints(function)
    dafny_code = assemble(function, translation.dafny_spec, translation.dafny_function)
    dafny_mod = module_name(function)
    pre_name = python_name("pre")
    with tempfile.TemporaryDirectory(prefix="pcc-equiv-") as tmp:
        out_dir = Path(tmp) / "out"
        # Only a missing or wrong-version Dafny raises ToolError; a translation that does not compile raises
        # NotCheckable from build_python and escapes (the pipeline turns it into not_checkable).
        built = build_python(dafny_code, out_dir)
        wrapper = wrapper_source(function, dafny_mod)
        try:
            payload = _run_script(
                _inputs_script(function, inputs),
                repo_root,
                _MARKER_INPUTS,
                timeout_seconds=_COLLECT_TIMEOUT_SECONDS,
            )
        except _ScriptTimeout as error:
            raise ToolError(f"equivalence input collection timed out: {error}") from error
        raw_inputs = payload.get("inputs")
        if not isinstance(raw_inputs, list) or not all(isinstance(entry, str) for entry in raw_inputs):
            raise ToolError(f"equivalence script returned a malformed inputs list: {payload!r}")
        hyp_inputs: list[str] = [entry for entry in raw_inputs if isinstance(entry, str)]
        seen = set(hyp_inputs)
        edge_cases = [entry for entry in _model_edge_cases(function, client=client, model=model) if entry not in seen]
        combined: list[str] = [*hyp_inputs, *edge_cases]
        script = _eval_script(function, dafny_mod, wrapper, combined, pre_name)
        full_script = f"import sys\nsys.path.insert(0, {str(built)!r})\n" + script
        outcome = run_python(full_script, cwd=repo_root, timeout_seconds=_EVAL_TIMEOUT_SECONDS)
        done = _marker_payload(outcome.stdout, _MARKER_DONE)
        if done is None:
            if outcome.timed_out:
                diff_record, diff_mismatch = _timeout_report(
                    function, wrapper, built, combined, repo_root, pre_name, outcome.stdout
                )
            else:
                raise ToolError(
                    "equivalence sandbox run produced no result "
                    f"(exit {outcome.returncode}, timed out {outcome.timed_out}): {outcome.stderr.strip()}"
                )
        else:
            tried = done.get("tried")
            total = done.get("total")
            mismatch = done.get("mismatch")
            if not isinstance(tried, int) or not isinstance(total, int):
                raise ToolError(f"equivalence script returned a malformed result: {done!r}")
            if tried == 0:
                diff_record = CheckRecord(CheckKind.DIFFERENTIAL, False, _DETAIL_NEVER, 0, total)
                diff_mismatch = None
            elif mismatch is None:
                diff_record = CheckRecord(CheckKind.DIFFERENTIAL, True, "", tried, total)
                diff_mismatch = None
            else:
                if not isinstance(mismatch, dict):
                    raise ToolError(f"equivalence script returned a malformed mismatch: {done!r}")
                arguments = mismatch.get("arguments")
                orig = mismatch.get("orig")
                trans = mismatch.get("trans")
                if not isinstance(arguments, str) or not isinstance(orig, list) or not isinstance(trans, list):
                    raise ToolError(f"equivalence script returned a malformed mismatch: {done!r}")
                detail = (
                    f"{arguments} -> original {_describe(str(orig[0]), str(orig[1]))}, "
                    f"translation {_describe(str(trans[0]), str(trans[1]))}"
                )
                diff_record = CheckRecord(CheckKind.DIFFERENTIAL, False, detail, tried, total)
                diff_mismatch = _mismatch_counterexample(
                    function, arguments, [str(orig[0]), str(orig[1])], [str(trans[0]), str(trans[1])]
                )
        if symbolic_seconds <= 0:
            return EquivalenceReport(records=(diff_record,), mismatch=diff_mismatch)
        symbolic_record, difference = _run_symbolic(function, wrapper, built, dafny_mod, symbolic_seconds, repo_root)
        final_mismatch = diff_mismatch
        if difference is not None and final_mismatch is None:
            symbolic_arguments, symbolic_orig, symbolic_trans = difference
            final_mismatch = _symbolic_counterexample(function, symbolic_arguments, symbolic_orig, symbolic_trans)
        return EquivalenceReport(records=(diff_record, symbolic_record), mismatch=final_mismatch)


def _timeout_report(
    function: FunctionRef,
    wrapper: str,
    built: Path,
    combined: list[str],
    repo_root: Path,
    pre_name: str,
    stdout: str,
) -> tuple[CheckRecord, Counterexample | None]:
    """Pinpoint the hanging input after an evaluation timeout and report it as a timeout mismatch."""

    last = _progress_index(stdout)
    if last is None:
        candidates = combined[:_PROGRESS_EVERY] if combined else []
    else:
        candidates = combined[last : last + _PROGRESS_EVERY]
    hanging = _pinpoint_hanging(function, wrapper, built, candidates, repo_root, pre_name)
    if hanging is None:
        hanging = candidates[0] if candidates else (combined[0] if combined else "")
    orig_kind, orig_payload = _probe_original(function, hanging, repo_root)
    if orig_kind == "timeout":
        # Both sides hang on the same input: there is no differing outcome to report as a
        # counterexample, so the check fails without a mismatch instead of one with identical
        # expected and actual outcomes.
        record = CheckRecord(CheckKind.DIFFERENTIAL, False, f"{hanging} -> both sides timeout", None, len(combined))
        return record, None
    record = CheckRecord(CheckKind.DIFFERENTIAL, False, f"{hanging} -> {_DETAIL_TIMEOUT}", None, len(combined))
    # For the timeout case the translation never answered; the original outcome (when known) is expected.
    counterexample = Counterexample(
        arguments=hanging,
        expected=_describe(orig_kind, orig_payload),
        actual=_DETAIL_TIMEOUT,
        test=_test_source(function, hanging, orig_kind, orig_payload, "timeout", _DETAIL_TIMEOUT),
    )
    return record, counterexample


def _run_symbolic(
    function: FunctionRef,
    wrapper: str,
    built: Path,
    dafny_mod: str,
    symbolic_seconds: float,
    repo_root: Path,
) -> tuple[CheckRecord, tuple[str, tuple[str, str] | None, tuple[str, str] | None] | None]:
    """Run the stretch symbolic search and interpret its outcome (a ``ToolError`` escapes).

    The ``pcc_symbolic`` module (both sides guarded by the compiled Dafny ``pre``) is written next to the build
    and compared with the prescribed ``diffbehavior`` call, bounded by ``symbolic_seconds + 30`` wall seconds.
    ``wrapper`` is only named for signature clarity: the symbolic module embeds the differential wrapper itself.
    """
    _ = wrapper  # embedded in the symbolic module; kept for signature clarity
    module_source = symbolic_module_source(function, dafny_mod)
    script = _symbolic_script(module_source, built, symbolic_seconds)
    outcome = run_python(script, cwd=repo_root, timeout_seconds=symbolic_seconds + _SYMBOLIC_GRACE_SECONDS)
    return _symbolic_record(symbolic_seconds, outcome)


def _pinpoint_hanging(
    function: FunctionRef,
    wrapper: str,
    built: Path,
    candidates: list[str],
    repo_root: Path,
    pre_name: str,
) -> str | None:
    """Return the first candidate whose single-input evaluation times out, or ``None`` when all complete."""
    from pcc.sandbox import run_python

    dafny_mod = module_name(function)
    for source in candidates:
        single = f"import sys\nsys.path.insert(0, {str(built)!r})\n" + _single_script(
            function, dafny_mod, wrapper, source, pre_name
        )
        outcome = run_python(single, cwd=repo_root, timeout_seconds=_PINPOINT_TIMEOUT_SECONDS)
        if _marker_payload(outcome.stdout, _MARKER_DONE) is None and outcome.timed_out:
            return source
    return None


def _probe_original(function: FunctionRef, source: str, repo_root: Path) -> tuple[str, str]:
    """Run ``source`` against the original only; timeouts become ``("timeout", "timeout")``."""
    from pcc.sandbox import run_python

    outcome = run_python(
        _original_probe_script(function, source), cwd=repo_root, timeout_seconds=_PINPOINT_TIMEOUT_SECONDS
    )
    payload = _marker_payload(outcome.stdout, "PCC_EQUIV_ORIG ")
    if payload is not None:
        kind = payload.get("kind")
        detail = payload.get("payload")
        if kind in ("return", "raises") and isinstance(detail, str):
            return (str(kind), detail)
    if outcome.timed_out:
        return ("timeout", _DETAIL_TIMEOUT)
    return ("timeout", _DETAIL_TIMEOUT)


__all__ = ["SCHEMA", "STEP", "check"]
