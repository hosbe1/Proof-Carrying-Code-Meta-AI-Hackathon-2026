"""Step 7: a second spec written with the function body in view, as a hint."""

from __future__ import annotations

import ast
import json

from pcc.core import Context, SecondOpinion, Spec
from pcc.models import Message, ModelClient

STEP = "second_opinion"
NO_NOTE = "no note"
DROPPED_SUFFIX = "(dropped alternative: not a `def spec` predicate)"

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "agrees": {"type": "boolean"},
        "alternative_predicate": {"type": ["string", "null"]},
        "note": {"type": "string"},
    },
    "required": ["agrees", "alternative_predicate", "note"],
    "additionalProperties": False,
}


def defines_spec(source: str) -> bool:
    """Whether ``source`` defines a function named ``spec``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "spec" for node in ast.walk(tree))


def _evidence_lines(context: Context) -> str:
    if not context.evidence:
        return "(no evidence)"
    return "\n".join(f"- [{item.kind.value}] {item.ref}: {item.excerpt}" for item in context.evidence)


def review(context: Context, spec: Spec, *, client: ModelClient, model: str) -> SecondOpinion:
    """Write a second spec with the body in view and compare it with the blind one.

    Makes one model call with step ``"second_opinion"`` whose prompt holds the function source, the context
    evidence and the blind spec. The answer follows ``SCHEMA``: ``agrees`` says whether the second spec matches
    the blind one, ``alternative_predicate`` holds the second spec when they disagree, and ``note`` explains.

    An empty note becomes ``"no note"``. An alternative that does not define ``def spec`` is dropped to ``None``
    and the note says so. An alternative is ignored when ``agrees`` is true. ``ToolError`` from the model client
    passes through; the alternative is a hint and never feeds any decision.
    """
    function = context.function
    precondition = f"\n{spec.precondition}\n" if spec.precondition is not None else ""
    messages: tuple[Message, ...] = (
        Message(
            "system",
            "You are a second-opinion reviewer. Compare the blind spec against the function body. "
            "Agree when they match; otherwise propose an alternative predicate named spec. "
            "Reply with JSON only.",
        ),
        Message(
            "user",
            f"Function {function.qualname} ({function.path}:{function.lineno}):\n"
            f"{function.source}\n"
            f"\nContext evidence:\n{_evidence_lines(context)}\n"
            f"\nBlind spec:\n{spec.predicate}\n{precondition}"
            "Reply as JSON with keys agrees, alternative_predicate and note.",
        ),
    )
    completion = client.complete(model, messages, step=STEP, schema=SCHEMA)
    data = json.loads(completion.text)
    if not isinstance(data, dict):
        raise ValueError(f"second opinion answer must be a JSON object, got {type(data).__name__}")
    agrees = data.get("agrees")
    if not isinstance(agrees, bool):
        raise ValueError(f"second opinion answer needs a boolean 'agrees', got {agrees!r}")
    raw_note = data.get("note", "")
    note = raw_note if isinstance(raw_note, str) and raw_note.strip() else NO_NOTE
    if agrees:
        return SecondOpinion(agrees=True, alternative_predicate=None, note=note)
    raw_alternative = data.get("alternative_predicate")
    if raw_alternative is None:
        return SecondOpinion(agrees=False, alternative_predicate=None, note=note)
    if isinstance(raw_alternative, str) and defines_spec(raw_alternative):
        return SecondOpinion(agrees=False, alternative_predicate=raw_alternative, note=note)
    return SecondOpinion(agrees=False, alternative_predicate=None, note=f"{note} {DROPPED_SUFFIX}")


__all__ = ["DROPPED_SUFFIX", "NO_NOTE", "SCHEMA", "STEP", "defines_spec", "review"]
