"""Prompt text for step 3 (spec). The function body never enters a prompt."""

from __future__ import annotations

from pcc.core import Intent

SYSTEM_PROMPT = (
    "You write a Python specification for a function whose body you never see.\n"
    "Work only from the signature and the intent claims below.\n"
    'Answer with JSON only: {"predicate": str, "precondition": str | null, "precondition_sources": [int]}.\n'
    "`predicate` holds exactly one `def spec(<args>, result)` returning a bool expression; "
    "`precondition` holds exactly one `def pre(<args>)` or null when no precondition is needed."
)

SPEC_CONSTRUCTS = (
    "- comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) and `==` on sequences\n"
    "- `and`, `or`, `not`\n"
    "- `+`, `-`, `*`, `//`, `%` on int\n"
    "- `len`, indexing, slicing (`[a:b]`, `[::-1]`)\n"
    "- `all` and `any` over generator expressions on `range(...)` or on a sequence\n"
    "- `in` on lists only (never on `str`)\n"
    "- conditional expressions (`x if c else y`)"
)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string"},
        "precondition": {"type": ["string", "null"]},
        "precondition_sources": {"type": "array", "items": {"type": "integer", "minimum": 1}},
    },
    "required": ["predicate", "precondition", "precondition_sources"],
    "additionalProperties": False,
}


def build_user_prompt(intent: Intent, signature: str, hint: str = "") -> str:
    """Render the blind spec prompt: name, signature with hints, intent claims and allowed constructs.

    Only the signature (parsed locally from the source) and the intent enter the prompt;
    ``intent.function.source`` never does, so a bug in the body cannot leak into the spec.
    """
    lines = [
        f"Function {intent.function.qualname} ({intent.function.path}:{intent.function.lineno}).",
        f"Signature: {signature}",
        f"Intent: {intent.summary}",
        "Claims:",
    ]
    for index, claim in enumerate(intent.claims, start=1):
        lines.append(f"C{index}: {claim.text}")
    lines.append("Allowed constructs:")
    lines.append(SPEC_CONSTRUCTS)
    if hint:
        lines.append(f"Hint: {hint}")
    lines.append("Keep `pre` as weak as the intent allows; every clause needs a claim that demands it.")
    lines.append(
        'Answer with the spec as JSON: {"predicate": "<def spec ...>", '
        '"precondition": "<def pre ...>" or null, '
        '"precondition_sources": [<1-based claim numbers>]}, with [] when there is no `pre`.'
    )
    return "\n".join(lines)


__all__ = ["SCHEMA", "SPEC_CONSTRUCTS", "SYSTEM_PROMPT", "build_user_prompt"]
