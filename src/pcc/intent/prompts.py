"""Prompt text for step 2 (intent). The function body never enters a prompt."""

from __future__ import annotations

from pcc.core import Context

SYSTEM_PROMPT = (
    "You describe what a Python function should do, in plain language.\n"
    "You never see the function body; work only from the numbered evidence below.\n"
    "Every claim must cite at least one evidence item by its number (E1, E2, ...).\n"
    "Flag every contradiction between evidence items.\n"
    'Answer with JSON only: {"summary": str, "claims": [{"text": str, "sources": [int]}], '
    '"contradictions": [{"description": str, "sources": [int]}], "checkable": bool, "reason": str}.\n'
    "Set checkable to false with a reason when the contradictions make any sensible intent impossible."
)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                },
                "required": ["text", "sources"],
            },
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                },
                "required": ["description", "sources"],
            },
        },
        "checkable": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["summary", "claims", "contradictions", "checkable", "reason"],
}


def build_user_prompt(context: Context) -> str:
    """Render the evidence of ``context`` as numbered items ``E1 ... En``.

    Only the kind, ref and excerpt of each evidence item enter the prompt; ``context.function.source`` never does, so
    a bug in the body cannot leak into the intent.
    """
    lines = [f"Function {context.function.qualname} ({context.function.path}:{context.function.lineno}).", "Evidence:"]
    if not context.evidence:
        lines.append("(no evidence)")
    for index, evidence in enumerate(context.evidence, start=1):
        lines.append(f"E{index} [{evidence.kind.value}] {evidence.ref}: {evidence.excerpt}")
    return "\n".join(lines)


__all__ = ["SCHEMA", "SYSTEM_PROMPT", "build_user_prompt"]
