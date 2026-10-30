"""Step 1 search agent: a JSON action loop with blind read-only tools (issue #32)."""

from __future__ import annotations

import json
from pathlib import Path

from pcc.context.tools import git_log, grep, read_file
from pcc.core import Context, Evidence, SourceKind
from pcc.models import Message, ModelClient

STEP: str = "context_search"

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read_file", "grep", "git_log", "done"]},
        "argument": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "ref": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["kind", "ref", "excerpt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["action", "argument", "evidence"],
    "additionalProperties": False,
}

_TOOLS = {"read_file": read_file, "grep": grep, "git_log": git_log}


def _base_evidence_lines(context: Context) -> str:
    if not context.evidence:
        return "(no evidence)"
    return "\n".join(f"- [{item.kind.value}] {item.ref}: {item.excerpt}" for item in context.evidence)


def _initial_messages(context: Context) -> list[Message]:
    function = context.function
    system = Message(
        "system",
        "You are a context search agent. Gather evidence about what the function should do. "
        "Use read_file to read a file, grep to search code, git_log for commit history, "
        "or done when no further search is needed. Reply with JSON only.",
    )
    user = Message(
        "user",
        f"Function {function.qualname} ({function.path}:{function.lineno}):\n"
        f"\nBase evidence:\n{_base_evidence_lines(context)}\n"
        "\nReply as JSON with keys action, argument and evidence.",
    )
    return [system, user]


def _collectable(raw: object, source: str) -> Evidence | None:
    """Return ``raw`` as ``Evidence`` or ``None`` when it must be dropped."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    ref = raw.get("ref")
    excerpt = raw.get("excerpt")
    if not isinstance(kind, str) or not isinstance(ref, str) or not isinstance(excerpt, str):
        return None
    try:
        parsed = SourceKind(kind)
    except ValueError:
        return None
    if source:
        stripped_source = source.strip()
        stripped_excerpt = excerpt.strip()
        if source in excerpt or source in stripped_excerpt:
            return None
        if stripped_source and (stripped_source in excerpt or stripped_source in stripped_excerpt):
            return None
    return Evidence(kind=parsed, ref=ref, excerpt=excerpt)


def search_further(
    context: Context, repo_root: Path, *, client: ModelClient, model: str, max_turns: int = 8
) -> Context:
    """Run a JSON action loop that appends evidence without ever seeing the function body.

    Makes at most ``max_turns`` model calls with step ``"context_search"``. The prompt describes
    the function by qualname, path and lineno plus its base evidence, never the body. Each answer
    follows ``SCHEMA``: ``read_file``/``grep``/``git_log`` run the same-named tool from
    ``pcc.context.tools`` and feed the (already masked) result back to the model, ``done`` ends the
    loop. Evidence items with an unknown kind, non-string fields, or an excerpt holding the
    function source are dropped. Without ``done``, the evidence collected so far is returned. Base
    evidence is kept first; collected items are appended after. ``ToolError`` passes through.
    """
    messages = _initial_messages(context)
    collected: list[Evidence] = []
    for _ in range(max_turns):
        completion = client.complete(model, tuple(messages), step=STEP, schema=SCHEMA)
        try:
            data = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"context search answer must be JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"context search answer must be a JSON object, got {type(data).__name__}")
        action = data.get("action")
        if action not in ("read_file", "grep", "git_log", "done"):
            raise ValueError(f"context search answer has an unknown action: {action!r}")
        raw_evidence = data.get("evidence", [])
        if isinstance(raw_evidence, list):
            for raw in raw_evidence:
                item = _collectable(raw, context.function.source)
                if item is not None:
                    collected.append(item)
        if action == "done":
            break
        argument = data.get("argument", "")
        if not isinstance(argument, str):
            raise ValueError(f"context search answer needs a string 'argument', got {argument!r}")
        result = _TOOLS[action](repo_root, argument, context.function)
        reply = f"Result of {action} {argument!r}:\n{result}\nReply as JSON."
        messages.append(Message("assistant", completion.text))
        messages.append(Message("user", reply))
    return Context(function=context.function, evidence=tuple(context.evidence) + tuple(collected))


__all__ = ["SCHEMA", "STEP", "search_further"]
