"""Prompt text for step 5 (prove). The spec and the executable code stay frozen."""

from __future__ import annotations

from collections.abc import Sequence

from pcc.core import Translation

SYSTEM_PROMPT = (
    "You strengthen a Dafny method with ghost code only: loop invariants, decreases clauses, "
    "assert statements, calc blocks, lemmas and calls to those lemmas.\n"
    "Keep the method signature, its requires and its ensures exactly as they are; only decreases "
    "clauses may be added between the signature and the body. Never change the executable code.\n"
    "Never use assume, expect, {:axiom} or any other form the checker rejects.\n"
    'Answer with JSON only: {"lemmas": str, "method": str}.'
)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "lemmas": {"type": "string"},
        "method": {"type": "string"},
    },
    "required": ["lemmas", "method"],
    "additionalProperties": False,
}


def build_user_prompt(translation: Translation, *, pylib: str, failures: Sequence[str]) -> str:
    """Render the frozen spec, the current method, the PyLib signatures and the failures to fix.

    ``failures`` holds the earlier errors of this function the model must fix; it is empty on the first attempt.
    """
    function = translation.function
    lines = [
        f"Function {function.qualname} ({function.path}:{function.lineno}).",
        "Frozen spec (do not change):",
        translation.dafny_spec,
        "Current method (keep the signature, requires and ensures; "
        "only add decreases clauses, loop invariants, asserts, calc blocks, lemmas and lemma calls):",
        translation.dafny_function,
        "PyLib signatures (the proven building blocks the method may use):",
        pylib,
        "Previous failures (fix them with ghost code only):",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("(none yet)")
    lines.append(
        "Reply as JSON with keys lemmas and method: lemmas holds helper lemmas (no method declarations), "
        "method holds the complete proven method."
    )
    return "\n".join(lines)


__all__ = ["SCHEMA", "SYSTEM_PROMPT", "build_user_prompt"]
