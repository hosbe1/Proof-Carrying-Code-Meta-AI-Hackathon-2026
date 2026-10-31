"""Step 3, checks: the spec against the original code and against mutants.

Hypothesis probes the original function in the sandbox: an input on which the spec fails (or the function raises
while ``pre`` holds) is a finding. Rule-based mutants (``pcc.spec.mutants``) and model-written variants must then be
rejected by the spec. Mutants that behave exactly like the original on the input set are dropped first; a surviving
mutant that satisfies the spec on every input where the original fails is recorded as a possible fix and kept out of
the kill ratio, so a real bug cannot hide behind its own fix.
"""

from __future__ import annotations

import ast
import difflib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from pcc.config import SpecConfig
from pcc.core import CheckKind, CheckRecord, FunctionRef, Spec, ToolError
from pcc.models import Message, ModelClient
from pcc.sandbox import run_python, strategies_source, target_loader
from pcc.spec.mutants import rule_mutants

#: The model step name, used for cost reporting.
STEP = "mutants"

#: The JSON shape the model answer must follow: ``{"mutants": [...]}`` of complete variant sources.
SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"mutants": {"type": "array", "items": {"type": "string"}}},
    "required": ["mutants"],
    "additionalProperties": False,
}

_DETAIL_NEVER = "precondition never holds"
_DETAIL_NO_MUTANTS = "no mutants"
_DETAIL_TIMEOUT = "timeout"

#: Prefix marking the collected-inputs line on stdout, so prints from the target cannot be mistaken for it.
_MARKER_INPUTS = "PCC_CHECK_INPUTS "
#: Prefix marking the Hypothesis-find result line on stdout.
_MARKER_FIND = "PCC_CHECK_FIND "
#: Prefix marking the mutant-evaluation result line on stdout.
_MARKER_MUTANT = "PCC_CHECK_MUTANT "


@dataclass(frozen=True, slots=True)
class _Evaluated:
    """One input with its precondition, outcome and spec verdict, all evaluated in the sandbox."""

    arguments: str  # the call arguments as ``", ".join(repr(a) for a in args)``
    pre_holds: bool
    outcome: tuple[str, str] | None  # ("return", repr) | ("raises", ExcType); None when pre does not hold
    spec: Literal["true", "false", "error", "na"]  # spec verdict; "na" when pre fails, on raise, or spec raised

    def breaks_spec(self) -> bool:
        """Whether the original breaks the spec on this input: it raises, or the spec is false or raises."""
        if not self.pre_holds or self.outcome is None:
            return False
        if self.outcome[0] == "raises":
            return True
        return self.spec in ("false", "error")

    def display(self) -> str:
        """The outcome as shown in mutant-kill details: the ``repr`` or ``"raises <ExcType>"``."""
        assert self.outcome is not None
        kind, payload = self.outcome
        if kind == "raises":
            return f"raises {payload}"
        return payload


@dataclass(frozen=True, slots=True)
class _MutantResult:
    """What one mutant did on the input set (and on the shrunk failing input, when there is one)."""

    source: str
    load_error: bool  # the variant failed to parse or bind in the sandbox; it is dropped
    timed_out: bool  # the sandbox run produced no marker but timed out; counts as killed
    set_results: tuple[_Evaluated, ...]  # aligned with the input set; empty on timeout
    extra_results: tuple[_Evaluated, ...]  # the shrunk failing input, if any; empty on timeout


class _ScriptTimeout(Exception):
    """A sandbox script produced no result marker but timed out; the code under test hangs."""


def _binding_snippet(callee: str, has_pre: bool) -> str:
    """Sandbox helpers binding ``callee``: positional calls by parameter name, like the generated test does.

    ``spec`` (and ``pre`` when there is one) are already defined above this snippet; ``callee`` is ``target`` for
    the original or ``_mutant`` for a variant. Calls go through ``inspect`` so ``result`` may sit anywhere in the
    spec signature and keyword-only parameters work. ``_evaluate`` never raises: a ``pre`` (or ``spec``) that
    raises is reported as not holding (or as ``"error"``), and a ``target`` that raises is reported as a break.
    """
    pre_line = (
        "    return _call_bound(pre, _pre_sig, dict(zip(_func_names(), _args)))" if has_pre else "    return True"
    )
    return "\n".join(
        [
            "import inspect",
            f"_callee_sig = inspect.signature({callee})",
            "_spec_sig = inspect.signature(spec)",
            *(
                ["_pre_sig = inspect.signature(pre)"]
                if has_pre
                else ["_pre_sig = None  # no precondition; _pre_at always holds"]
            ),
            "def _call_bound(_fn, _sig, _binding):",
            "    _posonly = [n for n, p in _sig.parameters.items() if p.kind is inspect.Parameter.POSITIONAL_ONLY]",
            "    _pos = tuple(_binding[n] for n in _posonly)",
            "    _kw = {n: v for n, v in _binding.items() if n not in _posonly and n in _sig.parameters}",
            "    return _fn(*_pos, **_kw)",
            "def _func_names():",
            "    return [n for n, p in _callee_sig.parameters.items() if p.kind in"
            " (inspect.Parameter.POSITIONAL_ONLY,"
            " inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]",
            "def _call_at(_args):",
            f"    return _call_bound({callee}, _callee_sig, dict(zip(_func_names(), _args)))",
            "def _spec_at(_args, _result):",
            "    _binding = dict(zip(_func_names(), _args))",
            "    _binding['result'] = _result",
            "    return _call_bound(spec, _spec_sig, _binding)",
            "def _pre_at(_args):",
            pre_line,
            "def _safe_repr(_value):",
            "    try:",
            "        return repr(_value)",
            "    except Exception:",
            "        return '<unrepresentable>'",
            "def _evaluate(_args):",
            "    try:",
            "        _holds = bool(_pre_at(_args))",
            "    except Exception:",
            "        return [False, None, None, 'na']",
            "    if not _holds:",
            "        return [False, None, None, 'na']",
            "    try:",
            "        _result = _call_at(_args)",
            "    except Exception as _exc:",
            "        return [True, 'raises', type(_exc).__name__, 'na']",
            "    try:",
            "        _ok = bool(_spec_at(_args, _result))",
            "    except Exception:",
            "        return [True, 'return', _safe_repr(_result), 'error']",
            "    return [True, 'return', _safe_repr(_result), 'true' if _ok else 'false']",
        ]
    )


def _collect_snippet(max_examples: int) -> str:
    """The Hypothesis run collecting every generated input with its evaluation (``_collected``).

    There is no filtering here: ``pre`` is evaluated per input in ``_evaluate``, so the counts of generated
    inputs (``inputs_total``) and pre-holding inputs (``inputs_tried``) come from one deterministic run
    (``derandomize=True``), shared by the original and every mutant.

    The derandomized seed is a hash of ``_collect``'s own source, so this snippet must be byte-identical
    (same indentation, same body) in every script that regenerates the set; only ``max_examples`` may differ,
    and a shorter run is a prefix of a longer one.
    """
    return (
        "import hypothesis\n"
        "from hypothesis import given, settings\n"
        "_collected = []\n"
        "@given(args_strategy)\n"
        f"@settings(max_examples={max_examples}, database=None, derandomize=True, deadline=None)\n"
        "def _collect(_args):\n"
        "    _collected.append([', '.join(_safe_repr(_v) for _v in _args)] + _evaluate(_args))\n"
        "_collect()\n"
    )


def _inputs_script(function: FunctionRef, spec: Spec, max_examples: int) -> str:
    """A sandbox script printing one ``PCC_CHECK_INPUTS`` line with every generated input evaluated."""
    chunks = [strategies_source(function), target_loader(function), spec.predicate]
    if spec.precondition is not None:
        chunks.append(spec.precondition)
    chunks.append(_binding_snippet("target", spec.precondition is not None))
    chunks.append(_collect_snippet(max_examples))
    chunks.append(f"import json\nprint({_MARKER_INPUTS!r} + json.dumps({{'inputs': _collected}}))\n")
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _find_script(function: FunctionRef, spec: Spec, max_examples: int) -> str:
    """A sandbox script printing one ``PCC_CHECK_FIND`` line: the shrunk spec break, if there is one.

    ``hypothesis.find`` returns the minimal input on which ``pre`` holds and the original breaks the spec (it
    raises, or the spec is false or raises). ``NoSuchExample`` (nothing found) prints a clean result, so the
    caller knows the original holds; ``Unsatisfiable`` and ``FailedHealthCheck`` (``pre`` never holds) print
    the same clean result and the caller reports "precondition never holds" from the collected counts.
    """
    chunks = [strategies_source(function), target_loader(function), spec.predicate]
    if spec.precondition is not None:
        chunks.append(spec.precondition)
    chunks.append(_binding_snippet("target", spec.precondition is not None))
    chunks.append(
        "import hypothesis\n"
        "import json\n"
        "from hypothesis.errors import FailedHealthCheck, NoSuchExample, Unsatisfiable\n"
        "def _breaks(_args):\n"
        "    _entry = _evaluate(_args)\n"
        "    if not _entry[0]:\n"
        "        return False\n"
        "    if _entry[1] == 'raises':\n"
        "        return True\n"
        "    return _entry[3] in ('false', 'error')\n"
        "try:\n"
        "    _winner = hypothesis.find(\n"
        "        args_strategy, _breaks, settings=hypothesis.settings(\n"
        f"            max_examples={max_examples}, database=None, derandomize=True, deadline=None))\n"
        "except (NoSuchExample, Unsatisfiable, FailedHealthCheck):\n"
        f"    print({_MARKER_FIND!r} + json.dumps({{'status': 'clean'}}))\n"
        "else:\n"
        "    _found = _evaluate(_winner)\n"
        "    if _found[1] == 'raises':\n"
        "        _kind, _exc = 'raises', _found[2]\n"
        "    elif _found[3] == 'error':\n"
        "        _kind, _exc = 'spec_error', None\n"
        "    else:\n"
        "        _kind, _exc = 'spec_false', None\n"
        f"    print({_MARKER_FIND!r} + json.dumps({{'status': 'break',"
        " 'arguments': ', '.join(_safe_repr(_v) for _v in _winner), 'kind': _kind, 'exc': _exc}))\n"
    )
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _mutant_script(
    function: FunctionRef,
    spec: Spec,
    mutant_source: str,
    max_examples: int,
    take: int,
    extras: list[str],
) -> str:
    """A sandbox script printing one ``PCC_CHECK_MUTANT`` line with the variant evaluated on the input set.

    The variant is ``exec``'d in a copy of the module globals under the function's name, so recursive calls hit
    the mutant; a variant that fails to bind prints ``load_error`` and is dropped. The input set is regenerated
    with the same deterministic Hypothesis run as the original, so both see the same inputs, plus ``extras``
    (the shrunk failing input, parsed with ``ast.literal_eval``).
    """
    simple_name = [part for part in function.qualname.split(".") if part != "<locals>"][-1]
    chunks = [strategies_source(function), target_loader(function)]
    chunks.append(
        "_namespace = dict(vars(_module))\n"
        "_load_error = None\n"
        "try:\n"
        f"    exec({mutant_source!r}, _namespace)\n"
        f"    _mutant = _namespace[{simple_name!r}]\n"
        "except Exception as _exc:\n"
        "    _load_error = type(_exc).__name__\n"
        "if _load_error is not None:\n"
        "    import json\n"
        f"    print({_MARKER_MUTANT!r} + json.dumps({{'status': 'load_error', 'error': _load_error}}))\n"
        "    raise SystemExit(0)\n"
    )
    chunks.append(spec.predicate)
    if spec.precondition is not None:
        chunks.append(spec.precondition)
    chunks.append(_binding_snippet("_mutant", spec.precondition is not None))
    chunks.append(_collect_snippet(max_examples))
    chunks.append(f"_selected = [_e for _e in _collected if _e[1]][:{take}]\n")
    chunks.append(
        "import ast\n"
        "import json\n"
        "_extras = []\n"
        f"for _src in {extras!r}:\n"
        "    _parsed = () if not _src.strip() else ast.literal_eval('(' + _src + ',)')\n"
        "    _extras.append([_src] + _evaluate(_parsed))\n"
        f"print({_MARKER_MUTANT!r} + json.dumps({{'status': 'ok', 'inputs': _selected, 'extras': _extras}}))\n"
    )
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


def _run_script(script: str, repo_root: Path, marker: str) -> dict[str, object]:
    """Run ``script`` in the sandbox and return its marker payload.

    A script whose marker parses is used as is, even on timeout (the result was printed before the hang). A run
    with no marker raises ``_ScriptTimeout`` on timeout and ``ToolError`` otherwise, so a broken setup is loud
    instead of silently moving on.
    """
    outcome = run_python(script, cwd=repo_root)
    payload = _marker_payload(outcome.stdout, marker)
    if payload is not None:
        return payload
    if outcome.timed_out:
        raise _ScriptTimeout(f"sandbox run timed out after printing no result (exit {outcome.returncode})")
    raise ToolError(
        "spec check sandbox run produced no result "
        f"(exit {outcome.returncode}, timed out {outcome.timed_out}): {outcome.stderr.strip()}"
    )


def _payload_list(payload: dict[str, object], key: str) -> list[object]:
    """The ``key`` entry of a marker payload, which must be a list."""
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ToolError(f"spec check script returned no {key!r} list: {payload!r}")
    return raw


def _parse_entry(raw: object) -> _Evaluated:
    """An evaluated input from a marker payload; malformed entries raise ``ToolError``."""
    if not isinstance(raw, list) or len(raw) != 5:
        raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
    arguments, pre_holds, kind, result, spec = raw
    if not isinstance(arguments, str) or not isinstance(pre_holds, bool):
        raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
    verdict = spec if isinstance(spec, str) else ""
    if verdict not in ("true", "false", "error", "na"):
        raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
    spec_verdict = cast("Literal['true', 'false', 'error', 'na']", verdict)
    outcome: tuple[str, str] | None = None
    if pre_holds:
        if kind not in ("return", "raises") or not isinstance(result, str):
            raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
        if kind == "return" and verdict == "na":
            raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
        if kind == "raises" and verdict != "na":
            raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
        outcome = (kind, result)
    elif kind is not None or result is not None or verdict != "na":
        raise ToolError(f"spec check script returned a malformed entry: {raw!r}")
    return _Evaluated(arguments, pre_holds, outcome, spec_verdict)


def _parse_find(payload: dict[str, object]) -> tuple[str, str, str | None] | None:
    """The shrunk break ``(arguments, kind, exc)`` of a find payload, or ``None`` when clean."""
    status = payload.get("status")
    if status == "clean":
        return None
    if status != "break":
        raise ToolError(f"spec check script returned a malformed find result: {payload!r}")
    arguments = payload.get("arguments")
    kind = payload.get("kind")
    exc = payload.get("exc")
    if not isinstance(arguments, str) or kind not in ("spec_false", "raises", "spec_error"):
        raise ToolError(f"spec check script returned a malformed find result: {payload!r}")
    if not (exc is None or isinstance(exc, str)):
        raise ToolError(f"spec check script returned a malformed find result: {payload!r}")
    return (arguments, kind, exc)


def _model_mutants(function: FunctionRef, *, client: ModelClient, model: str, count: int) -> tuple[str, ...]:
    """Ask the model once for up to ``count`` behavior-changing variants of the function.

    The prompt sees the function body but never the spec. An empty or malformed answer contributes zero model
    mutants; a variant that fails to parse or does not define the function is dropped. A ``ToolError`` from
    the client passes through.
    """
    lines = [
        f"Function {function.qualname} ({function.path}:{function.lineno}):",
        function.source.rstrip("\n"),
        "",
        f"Propose up to {count} behavior-changing variants of this function: bugs a developer would really make.",
        "Each variant must be the complete replacement source of the function.",
        'Reply as JSON with one key "mutants": a list of strings, each the complete source of one variant.',
    ]
    messages: tuple[Message, ...] = (
        Message("system", "You write behavior-changing variants of a Python function. Reply with JSON only."),
        Message("user", "\n".join(lines)),
    )
    completion = client.complete(model, messages, step=STEP, schema=SCHEMA)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, dict):
        return ()
    raw = data.get("mutants")
    if not isinstance(raw, list):
        return ()
    simple_name = [part for part in function.qualname.split(".") if part != "<locals>"][-1]
    variants: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        try:
            tree = ast.parse(entry)
        except SyntaxError:
            continue
        defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if simple_name not in defined:
            continue
        variants.append(entry)
        if len(variants) >= count:
            break
    return tuple(variants)


def _satisfies(entry: _Evaluated | None) -> bool:
    """Whether ``entry`` is a returned value the spec accepts: a possible fix must do this on every failing input."""
    return entry is not None and entry.outcome is not None and entry.outcome[0] == "return" and entry.spec == "true"


def _result_repr(entry: _Evaluated) -> str:
    """The result part of a ``spec_cases`` pair: the ``repr``, or ``"raises <ExcType>"``."""
    assert entry.outcome is not None
    return entry.display()


def _mutant_result(
    function: FunctionRef,
    spec: Spec,
    repo_root: Path,
    mutant_source: str,
    max_examples: int,
    take: int,
    extras: list[str],
) -> _MutantResult:
    """Evaluate one mutant in the sandbox; a timeout counts as killed, a crash raises ``ToolError``."""
    try:
        payload = _run_script(
            _mutant_script(function, spec, mutant_source, max_examples, take, extras),
            repo_root,
            _MARKER_MUTANT,
        )
    except _ScriptTimeout:
        return _MutantResult(mutant_source, load_error=False, timed_out=True, set_results=(), extra_results=())
    if payload.get("status") == "load_error":
        return _MutantResult(mutant_source, load_error=True, timed_out=False, set_results=(), extra_results=())
    if payload.get("status") != "ok":
        raise ToolError(f"spec check script returned a malformed mutant result: {payload!r}")
    set_results = tuple(_parse_entry(raw) for raw in _payload_list(payload, "inputs"))
    extra_results = tuple(_parse_entry(raw) for raw in _payload_list(payload, "extras"))
    return _MutantResult(mutant_source, False, False, set_results, extra_results)


def check(
    spec: Spec, function: FunctionRef, repo_root: Path, *, client: ModelClient, model: str, config: SpecConfig
) -> Spec:
    """Test the spec against the original code (Hypothesis) and against mutants (fixed rules plus realistic bugs from
    the model, equivalent mutants filtered out). The original code runs from ``repo_root``, so its imports resolve as
    they do in the project. Return the spec with the check records added; after this the spec is frozen."""
    cap = max(100, 20 * config.mutant_inputs)
    total = max(config.hypothesis_examples, cap)
    try:
        payload = _run_script(_inputs_script(function, spec, total), repo_root, _MARKER_INPUTS)
        collected = [_parse_entry(raw) for raw in _payload_list(payload, "inputs")]
    except _ScriptTimeout:
        hyp_record = CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, _DETAIL_TIMEOUT, 0, 0)
        mutants_record = CheckRecord(CheckKind.MUTANTS, False, _DETAIL_TIMEOUT)
        return replace(spec, checks=(*spec.checks, hyp_record, mutants_record))
    hyp_slice = collected[: config.hypothesis_examples]
    hyp_tried = sum(1 for entry in hyp_slice if entry.pre_holds)
    if not any(entry.pre_holds for entry in collected[:cap]):
        hyp_record = CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, _DETAIL_NEVER, 0, len(hyp_slice))
        mutants_record = CheckRecord(CheckKind.MUTANTS, False, _DETAIL_NEVER)
        return replace(spec, checks=(*spec.checks, hyp_record, mutants_record))

    selected = [entry for entry in collected[:cap] if entry.pre_holds][: config.mutant_inputs]
    selected_args = {entry.arguments for entry in selected}
    try:
        find_payload = _run_script(_find_script(function, spec, config.hypothesis_examples), repo_root, _MARKER_FIND)
        found = _parse_find(find_payload)
    except _ScriptTimeout:
        hyp_record = CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, _DETAIL_TIMEOUT, hyp_tried, len(hyp_slice))
        found = None
        hyp_failed = True
    else:
        hyp_failed = found is not None
        if found is None:
            hyp_record = CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, True, "", hyp_tried, len(hyp_slice))
        elif found[1] == "raises" and found[2] is not None:
            hyp_record = CheckRecord(
                CheckKind.HYPOTHESIS_ORIGINAL, False, f"{found[0]} raises {found[2]}", hyp_tried, len(hyp_slice)
            )
        else:
            hyp_record = CheckRecord(CheckKind.HYPOTHESIS_ORIGINAL, False, found[0], hyp_tried, len(hyp_slice))
    shrunk_args = found[0] if found is not None and found[0] not in selected_args else None
    extras = [shrunk_args] if shrunk_args is not None else []
    failing = [entry.arguments for entry in selected if entry.breaks_spec()]
    if shrunk_args is not None:
        failing.append(shrunk_args)

    model_sources = (
        _model_mutants(function, client=client, model=model, count=config.model_mutants)
        if config.model_mutants > 0
        else ()
    )
    sources = (*rule_mutants(function.source, config.max_rule_mutants), *model_sources)
    results = [
        _mutant_result(function, spec, repo_root, source, cap, config.mutant_inputs, extras) for source in sources
    ]

    kept: list[int] = []
    killed: set[int] = set()
    for index, result in enumerate(results):
        if result.load_error:
            continue
        if result.timed_out:
            kept.append(index)
            killed.add(index)
            continue
        if len(result.set_results) != len(selected):
            raise ToolError(
                f"spec check mutant run evaluated {len(result.set_results)} inputs, expected {len(selected)}"
            )
        if all(
            mutant.outcome == original.outcome for mutant, original in zip(result.set_results, selected, strict=True)
        ):
            continue
        kept.append(index)
        for mutant, original in zip(result.set_results, selected, strict=True):
            assert mutant.outcome is not None
            assert original.outcome is not None
            if mutant.outcome[0] == "return" and mutant.spec == "false":
                killed.add(index)
                break
            if mutant.outcome[0] == "raises" and original.outcome[0] == "return":
                killed.add(index)
                break

    fixes: list[int] = []
    if hyp_failed and failing:
        for index in kept:
            if index in killed:
                continue
            result = results[index]
            if result.timed_out:
                continue
            by_position = {entry.arguments: entry for entry in (*result.set_results, *result.extra_results)}
            if all(_satisfies(by_position.get(arguments)) for arguments in failing):
                fixes.append(index)
    fix_set = set(fixes)
    remaining = len(kept) - len(fix_set)
    if remaining == 0:
        mutants_record = CheckRecord(CheckKind.MUTANTS, True, _DETAIL_NO_MUTANTS)
    else:
        slain = len(killed - fix_set)
        if slain / remaining >= config.min_kill_ratio:
            mutants_record = CheckRecord(CheckKind.MUTANTS, True, "")
        else:
            cases: list[str] = []
            for index in kept:
                if index in killed or index in fix_set:
                    continue
                result = results[index]
                assert not result.timed_out
                for mutant, original in zip(result.set_results, selected, strict=True):
                    if mutant.outcome != original.outcome:
                        assert mutant.outcome is not None
                        assert original.outcome is not None
                        cases.append(f"{mutant.arguments} -> original {original.display()}, mutant {mutant.display()}")
                        break
                if len(cases) >= 5:
                    break
            mutants_record = CheckRecord(CheckKind.MUTANTS, False, "; ".join(cases))

    fix_records = tuple(
        CheckRecord(
            CheckKind.CANDIDATE_FIX,
            True,
            "\n".join(
                difflib.unified_diff(
                    function.source.splitlines(),
                    results[index].source.splitlines(),
                    fromfile="original",
                    tofile="mutant",
                )
            ),
        )
        for index in fixes
    )
    return replace(spec, checks=(*spec.checks, hyp_record, mutants_record, *fix_records))


def spec_cases(
    spec: Spec, function: FunctionRef, repo_root: Path, *, config: SpecConfig
) -> tuple[tuple[str, str], ...]:
    """The inputs the spec translation check replays: ``(arguments, expected)`` pairs from the spec checks."""
    cap = max(100, 20 * config.mutant_inputs)
    try:
        payload = _run_script(_inputs_script(function, spec, cap), repo_root, _MARKER_INPUTS)
        collected = [_parse_entry(raw) for raw in _payload_list(payload, "inputs")]
    except _ScriptTimeout:
        return ()
    selected = [entry for entry in collected[:cap] if entry.pre_holds][: config.mutant_inputs]
    if not selected:
        return ()
    pairs = [(entry.arguments, _result_repr(entry)) for entry in selected]
    for mutant_source in rule_mutants(function.source, config.max_rule_mutants):
        result = _mutant_result(function, spec, repo_root, mutant_source, cap, config.mutant_inputs, [])
        if result.timed_out:
            continue
        if result.load_error or len(result.set_results) != len(selected):
            continue
        if all(
            mutant.outcome == original.outcome for mutant, original in zip(result.set_results, selected, strict=True)
        ):
            continue
        pairs.extend((entry.arguments, _result_repr(entry)) for entry in result.set_results)
    return tuple(pairs)


__all__ = ["check", "spec_cases"]
