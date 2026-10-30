"""Step 4, fallback: translation by a model for whatever the transpiler does not support yet."""

from __future__ import annotations

import json
import re

from pcc.core import Counterexample, DafnyError, FunctionRef, NotCheckable, Spec, Translation, TranslationPath
from pcc.models import Message, ModelClient
from pcc.pylib import signatures
from pcc.translate_ai.prompts import SCHEMA, SYSTEM_PROMPT, build_user_prompt
from pcc.verifier import assemble, method_name, resolve

STEP = "translate_ai"


def _clauses(dafny_function: str, keyword: str) -> list[str]:
    """The stripped lines of ``dafny_function`` that are a ``keyword`` clause."""
    found = []
    for raw in dafny_function.splitlines():
        stripped = raw.strip()
        if stripped == keyword or stripped.startswith(f"{keyword} "):
            found.append(stripped)
    return found


def _same_params(first: str, second: str) -> bool:
    """Whether two parameter lists hold the same names, ignoring whitespace."""
    return "".join(first.split()) == "".join(second.split())


def _check_header(dafny_function: str) -> str | None:
    """None when the method header is exactly the required one, else the retry feedback.

    The method must carry exactly one ``requires pre(<params>)`` clause and one
    ``ensures spec(<params>, result)`` clause with the same parameters. Any other
    ``requires`` or ``ensures`` is a resolve failure ("header changed").
    """
    requires = _clauses(dafny_function, "requires")
    ensures = _clauses(dafny_function, "ensures")
    if len(requires) != 1 or len(ensures) != 1:
        return (
            "header changed: the method must have exactly one requires clause and one ensures clause, "
            f"got {len(requires)} requires and {len(ensures)} ensures"
        )
    requires_match = re.fullmatch(r"requires\s+pre\s*\((.*)\)", requires[0], re.DOTALL)
    ensures_match = re.fullmatch(r"ensures\s+spec\s*\((.*),\s*result\s*\)", ensures[0], re.DOTALL)
    if requires_match is None or ensures_match is None:
        return (
            "header changed: the method header must be exactly requires pre(<params>) "
            "and ensures spec(<params>, result)"
        )
    if not _same_params(requires_match.group(1), ensures_match.group(1)):
        return "header changed: the requires and ensures clauses must take the same parameters"
    return None


def _check_spec_kept(dafny_function: str) -> str | None:
    """None when the answer leaves the given spec alone, else the retry feedback ("spec changed")."""
    for raw in dafny_function.splitlines():
        stripped = raw.strip()
        if stripped.startswith("predicate ") or re.match(r"function\s+(pre|spec)\s*\(", stripped):
            return "spec changed: keep the given dafny_spec unchanged and translate only the function"
    return None


def _check_answer(text: str) -> tuple[str | None, str]:
    """Split a model answer into the Dafny function or the retry feedback.

    Returns ``(dafny_function, "")`` when the answer is usable, else ``(None, feedback)``.
    Answers that are not JSON, or that miss the ``dafny_function`` string, count as failed attempts.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"answer was not valid JSON: {error}"
    if not isinstance(data, dict):
        return None, f"answer must be a JSON object, got {type(data).__name__}"
    dafny_function = data.get("dafny_function")
    if not isinstance(dafny_function, str) or not dafny_function.strip():
        return None, "answer needs a non-empty string field 'dafny_function'"
    spec_error = _check_spec_kept(dafny_function)
    if spec_error is not None:
        return None, spec_error
    header_error = _check_header(dafny_function)
    if header_error is not None:
        return None, header_error
    return dafny_function, ""


def _format_errors(errors: tuple[DafnyError, ...]) -> str:
    """The resolve errors as retry feedback, one per line."""
    lines = []
    for error in errors:
        line = f"line {error.line}, column {error.column}: {error.message}"
        if error.related:
            line += f" ({error.related})"
        lines.append(f"- {line}")
    return "\n".join(lines)


def translate(
    function: FunctionRef,
    spec: Spec,
    *,
    client: ModelClient,
    model: str,
    dafny_spec: str,
    mismatch: Counterexample | None = None,
    resolve_retries: int = 2,
) -> Translation:
    """Translate the function; the spec is always translated by the transpiler and passed in as ``dafny_spec``.

    Makes one model call with step ``"translate_ai"`` whose prompt holds the function source, the Python
    spec, the PyLib signatures, the required method shape and the given ``dafny_spec`` to keep unchanged.
    ``mismatch`` is the counterexample of the previous attempt, if any, and its text goes into the prompt.

    The answer follows ``SCHEMA``: only ``dafny_function`` is translated. An answer that redefines the
    spec is rejected as a resolve failure ("spec changed"); a method whose header is not exactly
    ``requires pre(<params>)`` and ``ensures spec(<params>, result)`` is rejected as a resolve failure
    ("header changed"). Otherwise the assembled file is checked with ``verifier.resolve``. Every failure
    is retried with the errors in the prompt, up to ``resolve_retries`` times; when every attempt fails,
    raises ``NotCheckable("AI translation does not resolve")``. ``ToolError`` from the model client or
    from ``resolve`` passes through.
    """
    name = method_name(function)
    feedback = ""
    for _ in range(resolve_retries + 1):
        prompt = build_user_prompt(
            function,
            spec,
            method_name=name,
            dafny_spec=dafny_spec,
            pylib_signatures=signatures(),
            mismatch=mismatch,
            feedback=feedback,
        )
        messages = (Message("system", SYSTEM_PROMPT), Message("user", prompt))
        completion = client.complete(model, messages, step=STEP, schema=SCHEMA)
        dafny_function, feedback = _check_answer(completion.text)
        if dafny_function is None:
            continue
        errors = resolve(assemble(function, dafny_spec, dafny_function))
        if errors:
            feedback = _format_errors(errors)
            continue
        return Translation(
            function=function,
            function_path=TranslationPath.AI,
            spec_path=TranslationPath.TRANSPILER,
            dafny_function=dafny_function,
            dafny_spec=dafny_spec,
        )
    raise NotCheckable("AI translation does not resolve")


__all__ = ["STEP", "translate"]
