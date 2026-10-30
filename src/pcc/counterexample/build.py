"""Step 6, build: a ready-to-run pytest from a failing input."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from pcc.core import Counterexample, FunctionRef, Spec, ToolError
from pcc.sandbox import run_python
from pcc.sandbox.loader import target_loader

#: Prefix marking the probe's result line on stdout, so prints from the target cannot be mistaken for it.
_MARKER = "PCC_BUILD "

#: Helper emitted into the generated test when ``arguments`` hold a keyword argument.
_SPLIT_HELPER = "def _split(*args, **kwargs):\n    return args, kwargs"


def _one_line(source: str) -> str:
    """Collapse every run of whitespace in ``source`` to a single space."""
    return " ".join(source.split())


def _test_name(qualname: str) -> str:
    """The pytest function name for ``qualname``, with every non-alphanumeric replaced by ``_``."""
    return "test_" + "".join(char if char.isalnum() else "_" for char in qualname) + "_breaks_spec"


def _takes_kwargs(arguments: str) -> bool:
    """Whether ``arguments`` pass a keyword argument, so ``spec(args, result)`` would not stay valid Python."""
    try:
        tree = ast.parse(f"target({arguments})", mode="eval")
    except SyntaxError:
        return False
    call = tree.body
    return isinstance(call, ast.Call) and bool(call.keywords)


def _probe_script(function: FunctionRef, spec: Spec, arguments: str) -> str:
    """A sandbox script that prints one ``PCC_BUILD {"actual": <repr>}`` line for ``target(arguments)``.

    The script binds ``target`` the same way the generated test does, then defines ``spec`` (and ``pre`` when the
    spec has a precondition), so a broken spec source fails the probe instead of the test. The ``target`` call
    sits in a ``try``/``except`` that captures ``repr`` of the returned value or of the raised exception; only
    that ``repr`` crosses back, as JSON after the marker. Every helper name starts with an underscore, so none
    of them can clash with the target module's globals.
    """
    chunks = [target_loader(function), spec.predicate]
    if spec.precondition is not None:
        chunks.append(spec.precondition)
    chunks.append(
        "import json\n"
        "try:\n"
        f"    _call_result = target({arguments})\n"
        "    _actual_repr = repr(_call_result)\n"
        "except Exception as _call_error:\n"
        "    _actual_repr = repr(_call_error)\n"
        f"print({_MARKER!r} + json.dumps({{'actual': _actual_repr}}))\n"
    )
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _parse_actual(stdout: str) -> str | None:
    """The ``actual`` payload of the last marker line in ``stdout``, or ``None`` when there is none."""
    actual: str | None = None
    for line in stdout.splitlines():
        if not line.startswith(_MARKER):
            continue
        try:
            payload = json.loads(line[len(_MARKER) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            value = payload.get("actual")
            if isinstance(value, str):
                actual = value
    return actual


def _test_source(function: FunctionRef, spec: Spec, arguments: str) -> str:
    """A ready-to-run pytest module replaying ``arguments`` and asserting the spec.

    The module binds ``target`` through ``target_loader``, then carries the spec predicate source (and the
    precondition source when there is one) and a single test named after the qualname. Without keyword
    arguments the test calls ``result = target(arguments)`` and asserts ``assert spec(arguments, result)``;
    with a keyword argument ``assert spec(arguments, result)`` would not stay valid Python, so the test binds
    through ``_split`` and asserts ``assert spec(*_args, result, **_kwargs)`` instead. With no arguments at
    all the test calls ``target()`` and asserts ``spec(result)``.
    """
    chunks = [target_loader(function), spec.predicate]
    if spec.precondition is not None:
        chunks.append(spec.precondition)
    if _takes_kwargs(arguments):
        chunks.append(_SPLIT_HELPER)
        chunks.append(
            f"def {_test_name(function.qualname)}():\n"
            f"    _args, _kwargs = _split({arguments})\n"
            "    result = target(*_args, **_kwargs)\n"
            "    assert spec(*_args, result, **_kwargs)\n"
        )
    elif arguments.strip():
        chunks.append(
            f"def {_test_name(function.qualname)}():\n"
            f"    result = target({arguments})\n"
            f"    assert spec({arguments}, result)\n"
        )
    else:
        chunks.append(f"def {_test_name(function.qualname)}():\n    result = target()\n    assert spec(result)\n")
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def build(function: FunctionRef, spec: Spec, repo_root: Path, arguments: str) -> Counterexample:
    """Run ``arguments`` against the real code from ``repo_root`` and package the observed breakage as a test.

    The arguments run once more in the sandbox (``cwd=repo_root``), so a broken setup raises ``ToolError`` the
    same way ``run_python`` does. The returned counterexample echoes ``arguments`` unchanged, carries the spec
    predicate collapsed to one line as ``expected``, ``repr`` of the returned value or of the raised exception
    as ``actual``, and a ready-to-run pytest module as ``test``. When the sandbox process itself produces no
    result line (an unrelated failure, never a ``target`` exception), this raises ``ToolError`` with stderr.
    """
    outcome = run_python(_probe_script(function, spec, arguments), cwd=repo_root)
    actual = _parse_actual(outcome.stdout)
    if actual is None:
        raise ToolError(
            "counterexample probe produced no result "
            f"(exit {outcome.returncode}, timed out {outcome.timed_out}): {outcome.stderr.strip()}"
        )
    return Counterexample(
        arguments=arguments,
        expected=_one_line(spec.predicate),
        actual=actual,
        test=_test_source(function, spec, arguments),
    )


__all__ = ["build"]
