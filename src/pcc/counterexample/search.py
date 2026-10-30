"""Step 6, search: an input on which the real code breaks the spec."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from pcc.config import CounterexampleConfig
from pcc.core import Counterexample, FunctionRef, Spec
from pcc.counterexample.build import build
from pcc.models import Message, ModelClient
from pcc.sandbox import run_python, strategies_source, target_loader

#: The model step name, used for cost reporting.
STEP = "counterexample"

#: The JSON shape the model answer must follow: ``{"inputs": [...]}`` of call-argument sources.
SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"inputs": {"type": "array", "items": {"type": "string"}}},
    "required": ["inputs"],
    "additionalProperties": False,
}

#: Prefix marking the search result line on stdout, so prints from the target cannot be mistaken for it.
_MARKER = "PCC_SEARCH "


def _hypothesis_script(function: FunctionRef, spec: Spec, max_examples: int) -> str:
    """A sandbox script printing one ``PCC_SEARCH {"arguments": ...}`` line for the first spec break.

    The script draws argument tuples from ``args_strategy``, binds ``target`` the same way the generated test
    does, then defines ``spec`` (and ``pre`` when the spec has a precondition). ``_breaks`` holds only when the
    precondition holds (or there is none) and the spec is violated: a ``target`` that raises on a pre-holding
    input counts as a break, while a ``spec`` (or ``pre``) that raises counts as no break (or as not holding).
    ``hypothesis.find`` prints the winner with ``repr(args[0])`` for a single parameter and ``repr(args)``
    otherwise; a ``NoSuchExample`` (nothing found) prints nothing, so the caller moves on to the model phase.
    """
    chunks = [strategies_source(function), target_loader(function), spec.predicate]
    if spec.precondition is not None:
        chunks.append(spec.precondition)
        breaks = (
            "def _breaks(_args):\n"
            "    try:\n"
            "        _holds = pre(*_args)\n"
            "    except Exception:\n"
            "        return False\n"
            "    if not _holds:\n"
            "        return False\n"
            "    try:\n"
            "        _result = target(*_args)\n"
            "    except Exception:\n"
            "        return True\n"
            "    try:\n"
            "        return not spec(*_args, _result)\n"
            "    except Exception:\n"
            "        return False\n"
        )
    else:
        breaks = (
            "def _breaks(_args):\n"
            "    try:\n"
            "        _result = target(*_args)\n"
            "    except Exception:\n"
            "        return True\n"
            "    try:\n"
            "        return not spec(*_args, _result)\n"
            "    except Exception:\n"
            "        return False\n"
        )
    chunks.append(
        breaks + "import hypothesis\n"
        "import json\n"
        "from hypothesis.errors import NoSuchExample\n"
        "try:\n"
        "    _winner = hypothesis.find(\n"
        "        args_strategy, _breaks, settings=hypothesis.settings(\n"
        f"            max_examples={max_examples}, database=None, deadline=None))\n"
        "except NoSuchExample:\n"
        "    pass\n"
        "else:\n"
        "    if len(_winner) == 0:\n"
        "        _winner_arguments = ''\n"
        "    elif len(_winner) == 1:\n"
        "        _winner_arguments = repr(_winner[0])\n"
        "    else:\n"
        "        _winner_arguments = repr(_winner)\n"
        f"    print({_MARKER!r} + json.dumps({{'arguments': _winner_arguments}}))\n"
    )
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _parse_winner(stdout: str) -> str | None:
    """The ``arguments`` payload of the last marker line in ``stdout``, or ``None`` when there is none."""
    winner: str | None = None
    for line in stdout.splitlines():
        if not line.startswith(_MARKER):
            continue
        try:
            payload = json.loads(line[len(_MARKER) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            value = payload.get("arguments")
            if isinstance(value, str):
                winner = value
    return winner


def _probe_script(function: FunctionRef, spec: Spec, candidate: str) -> str:
    """A sandbox script printing one marker line saying whether ``candidate`` breaks the spec.

    The script binds ``target`` the same way the generated test does, then defines ``spec`` (and ``pre`` when
    the spec has a precondition). Arguments bind through ``_split`` (as in the generated test), so keyword
    arguments stay valid Python. The marker payload carries ``pre_holds`` and ``breaks``: a ``target`` that
    raises on a pre-holding input breaks, while a ``spec`` (or ``pre``) that raises means no break (or a
    pre that does not hold).
    """
    chunks = [target_loader(function), spec.predicate]
    if spec.precondition is not None:
        chunks.append(spec.precondition)
        pre_check = (
            "try:\n"
            f"    _pos, _kw = _split({candidate})\n"
            "    _pre_holds = bool(pre(*_pos, **_kw))\n"
            "except Exception:\n"
            "    _pre_holds = False\n"
        )
    else:
        pre_check = f"_pos, _kw = _split({candidate})\n_pre_holds = True\n"
    chunks.append(
        "import json\n"
        "def _split(*args, **kwargs):\n"
        "    return args, kwargs\n"
        + pre_check
        + "if not _pre_holds:\n"
        + f"    print({_MARKER!r} + json.dumps({{'pre_holds': False, 'breaks': False}}))\n"
        + "else:\n"
        + "    try:\n"
        + "        _probe_result = target(*_pos, **_kw)\n"
        + "    except Exception:\n"
        + f"        print({_MARKER!r} + json.dumps({{'pre_holds': True, 'breaks': True}}))\n"
        + "    else:\n"
        + "        try:\n"
        + "            _probe_ok = spec(*_pos, _probe_result, **_kw)\n"
        + "        except Exception:\n"
        + f"            print({_MARKER!r} + json.dumps({{'pre_holds': True, 'breaks': False}}))\n"
        + "        else:\n"
        + f"            print({_MARKER!r} + json.dumps({{'pre_holds': True, 'breaks': bool(not _probe_ok)}}))\n"
    )
    return "\n".join(chunk.rstrip("\n") for chunk in chunks) + "\n"


def _parse_probe(stdout: str) -> tuple[bool, bool]:
    """The ``(pre_holds, breaks)`` payload of the last marker line in ``stdout`` (``(False, False)`` by default)."""
    pre_holds = False
    breaks = False
    for line in stdout.splitlines():
        if not line.startswith(_MARKER):
            continue
        try:
            payload = json.loads(line[len(_MARKER) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            pre_holds = payload.get("pre_holds") is True
            breaks = payload.get("breaks") is True
    return pre_holds, breaks


def _model_candidates(
    function: FunctionRef,
    spec: Spec,
    failed_clause: str | None,
    client: ModelClient,
    model: str,
    config: CounterexampleConfig,
) -> list[str]:
    """Ask the model once for suspicious call-argument tuples and return them, truncated to ``model_inputs``.

    The prompt carries the function body, the spec predicate source, the precondition source when there is one,
    and ``failed_clause`` when it is not ``None``. A malformed answer (not JSON, missing or non-list
    ``inputs``, non-string entries) raises ``ValueError``; a ``ToolError`` from the client passes through.
    """
    lines = [
        f"Function {function.qualname} ({function.path}:{function.lineno}):",
        function.source.rstrip("\n"),
        "",
        "Spec the function must satisfy:",
        spec.predicate.rstrip("\n"),
    ]
    if spec.precondition is not None:
        lines += ["", "Precondition (only inputs satisfying it count):", spec.precondition.rstrip("\n")]
    if failed_clause is not None:
        lines += ["", f"The proof failed on this clause: {failed_clause}"]
    lines += [
        "",
        f"Propose up to {config.model_inputs} suspicious call-argument tuples as Python source, aimed at the "
        "failing clause when one is given.",
        'Reply as JSON with one key "inputs": a list of strings, each the argument list of one call, e.g. '
        '\'"abca"\' or "3, 4".',
    ]
    messages: tuple[Message, ...] = (
        Message(
            "system",
            "You propose inputs that break the spec. Study the body and the spec, then reply with JSON only.",
        ),
        Message("user", "\n".join(lines)),
    )
    completion = client.complete(model, messages, step=STEP, schema=SCHEMA)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError as error:
        raise ValueError(f"counterexample model answer is not JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"counterexample model answer must be a JSON object, got {type(data).__name__}")
    raw_inputs = data.get("inputs")
    if not isinstance(raw_inputs, list):
        raise ValueError(f"counterexample model answer needs an 'inputs' list, got {raw_inputs!r}")
    candidates: list[str] = []
    for entry in raw_inputs:
        if not isinstance(entry, str):
            raise ValueError(f"counterexample model answer 'inputs' must be strings, got {entry!r}")
        candidates.append(entry)
    return candidates[: config.model_inputs]


def search(
    function: FunctionRef,
    spec: Spec,
    repo_root: Path,
    *,
    failed_clause: str | None,
    client: ModelClient,
    model: str,
    config: CounterexampleConfig,
) -> Counterexample | None:
    """Find an executed input on which the real code breaks the spec, or return ``None`` when there is none.

    Hypothesis runs first in a single sandbox script (``cwd=repo_root``) with ``config.hypothesis_examples``
    examples drawn from ``pcc.sandbox.strategies_source``. When it finds a break, that winner is built into a
    counterexample immediately and the model is never called (first break wins). Otherwise the model is asked
    exactly once for up to ``config.model_inputs`` suspicious argument tuples; each is validated with
    ``ast.parse`` (unparseable proposals are skipped) and replayed in its own sandbox run, and the first one
    where ``pre`` holds and ``spec`` is false wins. A ``ToolError`` from the sandbox or the client passes
    through.
    """
    outcome = run_python(_hypothesis_script(function, spec, config.hypothesis_examples), cwd=repo_root)
    winner = _parse_winner(outcome.stdout)
    if winner is not None:
        return build(function, spec, repo_root, winner)
    for candidate in _model_candidates(function, spec, failed_clause, client, model, config):
        try:
            ast.parse(f"target({candidate})", mode="eval")
        except SyntaxError:
            continue
        pre_holds, breaks = _parse_probe(run_python(_probe_script(function, spec, candidate), cwd=repo_root).stdout)
        if pre_holds and breaks:
            return build(function, spec, repo_root, candidate)
    return None


__all__ = ["SCHEMA", "STEP", "search"]
