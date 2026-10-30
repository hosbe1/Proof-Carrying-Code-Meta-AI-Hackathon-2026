"""Step 2, deriving: turn context evidence into an intent with cited sources."""

from __future__ import annotations

import json
from typing import Any

from pcc.core import Claim, Context, Contradiction, Evidence, Intent, NotCheckable, SourceKind, ToolError, source_rank
from pcc.intent.prompts import SCHEMA, SYSTEM_PROMPT, build_user_prompt
from pcc.models import Message, ModelClient

_STEP = "intent"


def derive(context: Context, *, client: ModelClient, model: str) -> Intent:
    """Write the intent from ``context`` without the function body; resolve contradictions by ``SOURCE_RANK`` and
    record every one of them. Raise ``NotCheckable`` when no sensible intent can be written."""
    messages = (Message("system", SYSTEM_PROMPT), Message("user", build_user_prompt(context)))
    data = _complete(client, model, messages)
    _raise_if_not_checkable(data)
    problems = _invalid_indices(data, len(context.evidence))
    if problems:
        retry = Message("user", _retry_text(problems, len(context.evidence)))
        data = _complete(client, model, (*messages, retry))
        _raise_if_not_checkable(data)
        if _invalid_indices(data, len(context.evidence)):
            raise NotCheckable("intent without sources")
    return _build_intent(context, data)


def _complete(client: ModelClient, model: str, messages: tuple[Message, ...]) -> dict[str, Any]:
    """One model call for step ``intent``; return the answer as a JSON object.

    A ``ToolError`` from the client passes through. An answer that is not a JSON object breaks the call contract,
    so it raises ``ToolError`` too.
    """
    completion = client.complete(model, messages, step=_STEP, schema=SCHEMA)
    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"intent answer is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ToolError(f"intent answer must be a JSON object, got {type(data).__name__}")
    return data


def _raise_if_not_checkable(data: dict[str, Any]) -> None:
    """Raise ``NotCheckable`` with the model's reason when the answer says no sensible intent can be written."""
    checkable = data.get("checkable", True)
    if checkable is False:
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            raise ToolError(f"intent reason must be a string, got {type(reason).__name__}")
        raise NotCheckable(reason or "intent not checkable")
    if not isinstance(checkable, bool):
        raise ToolError(f"intent checkable must be a bool, got {type(checkable).__name__}")


def _entries(data: dict[str, Any], key: str) -> list[Any]:
    """Return the ``claims`` or ``contradictions`` list of an answer, or raise ``ToolError`` for a broken shape."""
    entries = data.get(key, [])
    if not isinstance(entries, list):
        raise ToolError(f"intent {key} must be a list, got {type(entries).__name__}")
    return entries


def _indices(entry: Any, label: str) -> tuple[list[int], list[str]]:
    """Split one claim/contradiction entry into valid indices and problem descriptions.

    Raises ``ToolError`` when the entry itself breaks the schema; unknown indices are reported, not raised, so the
    caller can retry them once.
    """
    if not isinstance(entry, dict):
        raise ToolError(f"intent {label} must be an object, got {type(entry).__name__}")
    sources = entry.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return [], [f"{label} has no sources"]
    valid: list[int] = []
    problems: list[str] = []
    for source in sources:
        if isinstance(source, bool) or not isinstance(source, int):
            problems.append(f"{label} cites invalid evidence E{source}")
        else:
            valid.append(source)
    return valid, problems


def _invalid_indices(data: dict[str, Any], count: int) -> list[str]:
    """Describe every claim/contradiction source index outside ``E1 ... En``; an empty list means all are valid."""
    problems: list[str] = []
    for position, entry in enumerate(_entries(data, "claims"), start=1):
        valid, entry_problems = _indices(entry, f"claim {position}")
        problems.extend(entry_problems)
        problems.extend(
            f"claim {position} cites unknown evidence E{index}" for index in valid if not 1 <= index <= count
        )
    for position, entry in enumerate(_entries(data, "contradictions"), start=1):
        valid, entry_problems = _indices(entry, f"contradiction {position}")
        problems.extend(entry_problems)
        problems.extend(
            f"contradiction {position} cites unknown evidence E{index}" for index in valid if not 1 <= index <= count
        )
    return problems


def _retry_text(problems: list[str], count: int) -> str:
    """Ask the model to answer again using only valid evidence numbers, naming what was wrong."""
    detail = "; ".join(problems)
    if count == 0:
        return (
            f"Invalid source indices: {detail}. "
            "There is no evidence; answer again with checkable set to false and a reason."
        )
    return (
        f"Invalid source indices: {detail}. Evidence is numbered E1 to E{count}; answer again using only those numbers."
    )


def _resolved_by(sources: tuple[Evidence, ...]) -> SourceKind | None:
    """The kind with the lowest ``source_rank``; ``None`` when several sources share that tier."""
    ranks = [source_rank(evidence.kind) for evidence in sources]
    best = min(ranks)
    winners = [evidence.kind for evidence, rank in zip(sources, ranks, strict=True) if rank == best]
    return winners[0] if len(winners) == 1 else None


def _build_intent(context: Context, data: dict[str, Any]) -> Intent:
    """Turn a validated answer into an ``Intent``; indices are valid here, so lookups cannot fail."""
    evidence = context.evidence
    claims = tuple(
        Claim(text=_text(entry, f"claim {position}"), sources=tuple(evidence[index - 1] for index in _valid(entry)))
        for position, entry in enumerate(_entries(data, "claims"), start=1)
    )
    contradictions = tuple(
        Contradiction(
            description=_text(entry, f"contradiction {position}", field="description"),
            sources=tuple(evidence[index - 1] for index in _valid(entry)),
            resolved_by=_resolved_by(tuple(evidence[index - 1] for index in _valid(entry))),
        )
        for position, entry in enumerate(_entries(data, "contradictions"), start=1)
    )
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        raise ToolError(f"intent summary must be a string, got {type(summary).__name__}")
    return Intent(context.function, summary, claims, contradictions)


def _valid(entry: Any) -> list[int]:
    """Indices of an entry that passed ``_invalid_indices``; every source is a usable int here."""
    sources: list[Any] = entry["sources"]
    return [source for source in sources if isinstance(source, int) and not isinstance(source, bool)]


def _text(entry: Any, label: str, field: str = "text") -> str:
    """Return the ``text`` (or ``description``) of an entry, or raise ``ToolError`` for a broken shape."""
    if not isinstance(entry, dict) or not isinstance(entry.get(field), str):
        raise ToolError(f"intent {label} must carry {field} as a string")
    text: str = entry[field]
    return text


__all__ = ["derive"]
