"""Prompt text for step 4 (AI fallback). Only the function is translated; the spec is kept."""

from __future__ import annotations

from pcc.core import Counterexample, FunctionRef, Spec

SYSTEM_PROMPT = (
    "You translate a Python function to Dafny.\n"
    "The Dafny spec is already translated and frozen; keep it unchanged and translate only the function.\n"
    "Use only the PyLib building blocks below for Python semantics.\n"
    "Reply with JSON only."
)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"dafny_function": {"type": "string"}},
    "required": ["dafny_function"],
    "additionalProperties": False,
}


def build_user_prompt(
    function: FunctionRef,
    spec: Spec,
    *,
    method_name: str,
    dafny_spec: str,
    pylib_signatures: str,
    mismatch: Counterexample | None = None,
    feedback: str = "",
) -> str:
    """Render the translation prompt for ``function``.

    The prompt holds the function source, the Python spec (predicate and precondition), the Dafny
    spec to keep unchanged, the required method shape (exactly ``requires pre(<params>)`` and
    ``ensures spec(<params>, result)`` on the method named ``method_name``) and the PyLib
    signatures. ``mismatch`` is the counterexample of the previous attempt, if any; ``feedback``
    holds the errors of the previous attempt, if any.
    """
    lines = [
        f"Translate the Python function {function.qualname} ({function.path}:{function.lineno}) to Dafny.",
        "",
        "Python function:",
        function.source,
        "",
        "Python spec (predicate spec, with an optional precondition pre):",
        spec.predicate,
    ]
    if spec.precondition is not None:
        lines.append(spec.precondition)
    lines.extend(
        [
            "",
            "Dafny spec (translated by the transpiler; keep it unchanged and do not repeat it):",
            dafny_spec,
            "",
            f"Required shape: a single method named {method_name} with exactly this header:",
            "  requires pre(<params>)",
            "  ensures spec(<params>, result)",
            "where <params> are the method parameters in order. No other requires or ensures clause is allowed.",
            "",
            "Use only these PyLib blocks for Python semantics:",
            pylib_signatures,
        ]
    )
    if mismatch is not None:
        lines.extend(
            [
                "",
                "The previous translation failed this equivalence check; fix it:",
                f"arguments: {mismatch.arguments}",
                f"expected: {mismatch.expected}",
                f"actual: {mismatch.actual}",
                "failing test:",
                mismatch.test,
            ]
        )
    if feedback:
        lines.extend(["", "The previous answer failed with these errors; fix them:", feedback])
    lines.append('Reply as JSON with key dafny_function: {"dafny_function": "<the Dafny method>"}.')
    return "\n".join(lines)


__all__ = ["SCHEMA", "SYSTEM_PROMPT", "build_user_prompt"]
