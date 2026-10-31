"""Step 5: the proof agent with escalating prover stages (docs/plan.md, step 5)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pcc.config import ProverStage
from pcc.core import Assumption, DafnyError, ProofOutcome, Translation
from pcc.models import Message, ModelClient
from pcc.prove.prompts import SCHEMA, SYSTEM_PROMPT, build_user_prompt
from pcc.pylib import signatures
from pcc.verifier import anti_cheat, assemble, same_executable, verify

STEP = "proof"

_METHOD_RE = re.compile(r"(?<![A-Za-z0-9_'])method(?![A-Za-z0-9_'])")
_ASSUMPTIONS_MODULE_RE = re.compile(r"(?<![A-Za-z0-9_'])module\s+([A-Za-z_][A-Za-z0-9_']*)")
_DECREASES_RE = re.compile(r"(?<![A-Za-z0-9_'])decreases(?![A-Za-z0-9_'])")
_LEMMA_RE = re.compile(r"(?m)^\s*lemma\s+([A-Za-z_][A-Za-z0-9_']*)")
_METHOD_DECL_RE = re.compile(r"(?m)^\s*method\s+[A-Za-z_]")
_HEADER_STOP_WORDS = frozenset({"requires", "ensures", "decreases", "modifies", "reads"})


class _AttemptFailed(Exception):
    """One proof attempt failed with this error; the run continues with the next attempt."""


def _parse_answer(text: str) -> tuple[str, str]:
    """Split a model answer into ``(lemmas, method)``; anything off-schema fails the attempt."""
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _AttemptFailed(f"invalid proof answer: not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _AttemptFailed(f"invalid proof answer: expected a JSON object, got {type(data).__name__}")
    lemmas = data.get("lemmas")
    method = data.get("method")
    if not isinstance(lemmas, str) or not isinstance(method, str):
        raise _AttemptFailed("invalid proof answer: 'lemmas' and 'method' must be strings")
    return lemmas, method


def _lemma_names(lemmas: str) -> tuple[str, ...]:
    """The names declared with ``lemma`` in ``lemmas``, in order without duplicates."""
    return tuple(dict.fromkeys(_LEMMA_RE.findall(lemmas)))


def _assumptions_name(assumptions_module: str) -> str:
    """The module declared by the ``assumptions_module`` source, or ``""`` when there is none."""
    match = _ASSUMPTIONS_MODULE_RE.search(assumptions_module)
    return match.group(1) if match is not None else ""


def _reject_method_in_lemmas(lemmas: str) -> None:
    """Fail the attempt when ``lemmas`` declares a method; lemmas hold only lemmas."""
    if _METHOD_DECL_RE.search(lemmas) is not None:
        raise _AttemptFailed("method declaration in lemmas")


def _header_end(text: str, start: int) -> int | None:
    """Index of the ``{`` that opens the method body, or ``None`` when there is none."""
    depth_paren = 0
    depth_brack = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        elif char == "{" and depth_paren == 0 and depth_brack == 0:
            return index
    return None


def _clause_end(header: str, start: int) -> int:
    """Index where the ``decreases`` clause starting at ``start`` ends."""
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    index = start
    size = len(header)
    while index < size:
        char = header[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            if depth_paren > 0:
                depth_paren -= 1
            else:
                return index
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            if depth_brack > 0:
                depth_brack -= 1
            else:
                return index
        elif char == "{":
            if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
                return index
            depth_brace += 1
        elif char == "}":
            if depth_brace > 0:
                depth_brace -= 1
            elif depth_paren == 0 and depth_brack == 0:
                return index
        elif char == ";" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            return index
        elif char.isalpha() and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            end = index
            while end < size and (header[end].isalnum() or header[end] in "_'"):
                end += 1
            if header[index:end] in _HEADER_STOP_WORDS:
                return index
            index = end
            continue
        index += 1
    return size


def _strip_decreases(header: str) -> str:
    """``header`` with every ``decreases`` clause blanked out, the only allowed header change."""
    parts: list[str] = []
    pos = 0
    for match in _DECREASES_RE.finditer(header):
        parts.append(header[pos : match.start()])
        pos = _clause_end(header, match.end())
    parts.append(header[pos:])
    return "".join(parts)


def _normalize(text: str) -> str:
    """``text`` with every whitespace run collapsed to one space."""
    return " ".join(text.split())


def _reject_unless_header_kept(original: str, candidate: str) -> None:
    """Fail the attempt unless the method header only gains ``decreases`` clauses.

    The signature, ``requires`` and ``ensures`` must stay exactly as translated; any other addition (an extra
    ``requires``, ``modifies`` or ``reads``) fails with "executable code changed".
    """
    original_match = _METHOD_RE.search(original)
    candidate_match = _METHOD_RE.search(candidate)
    if original_match is None or candidate_match is None:
        raise _AttemptFailed("executable code changed")
    original_end = _header_end(original, original_match.end())
    candidate_end = _header_end(candidate, candidate_match.end())
    if original_end is None or candidate_end is None:
        raise _AttemptFailed("executable code changed")
    original_header = original[original_match.start() : original_end]
    candidate_header = _strip_decreases(candidate[candidate_match.start() : candidate_end])
    if _normalize(candidate_header) != _normalize(original_header):
        raise _AttemptFailed("executable code changed")


def _cheat_form(violation: str) -> str:
    """The violation of ``anti_cheat`` without its ``"line <n>: "`` prefix."""
    _, _, form = violation.partition(": ")
    return form or violation


def _format_dafny_errors(errors: tuple[DafnyError, ...]) -> str:
    """One error line for the next prompt, always carrying the ``dafny`` marker."""
    if not errors:
        return "dafny: verification failed"
    return "dafny: " + "; ".join(f"line {error.line}: {error.message}" for error in errors)


def prove(
    translation: Translation,
    assumptions: Sequence[Assumption],
    *,
    client: ModelClient,
    stages: Sequence[ProverStage],
    resource_limit: int,
    function_lines: int,
    extra_modules: Sequence[str] = (),
    assumptions_module: str = "",
    imports: Sequence[str] = (),
) -> ProofOutcome:
    """Add invariants, lemmas and hints until ``pcc.verifier.verify`` passes, feeding Dafny's errors back after each
    attempt. Every attempt goes through ``pcc.verifier.anti_cheat``; the spec stays frozen.

    Each stage tries up to ``stage.attempts`` times with ``stage.model``; a stage whose ``max_source_lines`` is below
    ``function_lines`` is skipped. The prompt holds the frozen spec, the current method, the PyLib signatures and the
    failures so far (only the last one in the first stage, all of them from the second stage on). An answer that is
    not a ``{lemmas, method}`` object, changes the header or the executable code, declares a method in ``lemmas`` or
    trips the anti-cheat scan fails that attempt without stopping the run. The first verified attempt is returned;
    otherwise the last outcome, with ``stage`` set to the last stage tried. ``assumptions`` lists what the proof
    rests on (unproven callees, contracts). ``extra_modules`` holds complete, already-checked callee modules, and
    ``imports`` the Dafny module names the caller uses: each entry becomes an ``import`` line in the checked file, so
    Dafny links the caller against the callee specs. ``assumptions_module`` holds the generated assumptions module
    source (``""`` when every callee is proven): it is appended to the checked file, its module is imported, and its
    name is passed to ``verify`` and ``anti_cheat``, so ``{:axiom}`` is allowed only there. ``ToolError`` from the
    model client or the verifier passes through.
    """
    _ = assumptions
    assume_name = _assumptions_name(assumptions_module)
    if assumptions_module and not assume_name:
        raise ValueError("assumptions_module holds no module")
    import_names = (*imports, assume_name) if assume_name else tuple(imports)
    pylib_signatures = signatures()
    failures: list[str] = []
    attempts = 0
    last_stage = ""
    last_full = ""
    last_error = "no prover stage available"
    last_errors: tuple[DafnyError, ...] = ()
    for stage_index, stage in enumerate(stages):
        if stage.max_source_lines is not None and stage.max_source_lines < function_lines:
            continue
        last_stage = stage.name
        for _ in range(stage.attempts):
            attempts += 1
            shown = failures if stage_index > 0 else failures[-1:]
            messages = (
                Message("system", SYSTEM_PROMPT),
                Message("user", build_user_prompt(translation, pylib=pylib_signatures, failures=shown)),
            )
            completion = client.complete(stage.model, messages, step=STEP, schema=SCHEMA)
            attempt_errors: tuple[DafnyError, ...] = ()
            try:
                lemmas, method = _parse_answer(completion.text)
                full = assemble(
                    translation.function, translation.dafny_spec, method, lemmas=lemmas, imports=import_names
                )
                if assumptions_module:
                    full += assumptions_module if full.endswith("\n") else "\n" + assumptions_module
                last_full = full
                _reject_unless_header_kept(translation.dafny_function, method)
                lemma_names = _lemma_names(lemmas)
                if not same_executable(translation.dafny_function, method, lemma_names=lemma_names):
                    raise _AttemptFailed("executable code changed")
                _reject_method_in_lemmas(lemmas)
                violations = anti_cheat(full, translation.dafny_spec, assumptions_module=assume_name)
                if violations:
                    raise _AttemptFailed(f"anti-cheat: {_cheat_form(violations[0])}")
                outcome = verify(
                    full,
                    assumptions_module=assume_name,
                    resource_limit=resource_limit,
                    extra_modules=extra_modules,
                )
                if outcome.verified:
                    return ProofOutcome(True, full, attempts, stage.name, "", outcome.errors)
                attempt_errors = outcome.errors
                raise _AttemptFailed(_format_dafny_errors(outcome.errors))
            except _AttemptFailed as failed:
                last_error = str(failed)
                last_errors = attempt_errors
                failures.append(last_error)
    return ProofOutcome(False, last_full, attempts, last_stage, last_error, last_errors)


__all__ = ["STEP", "prove"]
