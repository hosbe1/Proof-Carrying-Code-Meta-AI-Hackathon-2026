"""The shared prompt pieces of the training path (docs/plan.md, Training).

``SYSTEM_MESSAGE`` is the system turn sent verbatim by every data line and every
sampling tool. ``build_prompt`` is the reference user prompt for one task: the sampling
harness (#46) and the round 0 tool (#69) must import it instead of copying it, so their
user messages compare byte-identical with the test-split prompt guard of the SFT export,
which relies on exact equality.
"""

from __future__ import annotations

from pcc.evaluation.datasets import Task

SYSTEM_MESSAGE = (
    "You are a Dafny 4.11 verification assistant. Return only a complete Dafny program. "
    "Preserve the supplied specification and method declaration exactly. "
    "If executable code is supplied, preserve it and add only ghost proof annotations. "
    "Do not use assume, axioms, verification-disabling attributes, or resource-limit overrides. "
    "The program must verify without --allow-warnings."
)

_PROMPT_INSTRUCTIONS = "Complete the Dafny method below so that it verifies. Return only the complete Dafny program."

_FIX_INSTRUCTIONS = (
    "Fix the buggy Python function below so that it passes its tests. Return only the complete fixed Python function."
)


def build_prompt(task: Task) -> str:
    """The reference user prompt for ``task``: instructions plus task text, never the answer.

    For a Vericoding row this joins the description, preamble, helpers, spec and postamble,
    skipping blank parts; ``vc-code`` holds the reference answer, so it stays out. For a
    HumanEvalFix row it is the Python fix instructions plus the ``prompt`` stub. Anything
    else raises ``ValueError`` instead of guessing a prompt.
    """
    payload = task.payload
    spec = payload.get("vc-spec")
    if isinstance(spec, str) and spec.strip():
        blocks = [
            payload.get("vc-description"),
            payload.get("vc-preamble"),
            payload.get("vc-helpers"),
            spec,
            payload.get("vc-postamble"),
        ]
        body = "\n\n".join(block for block in blocks if isinstance(block, str) and block.strip())
        if body:
            return f"{_PROMPT_INSTRUCTIONS}\n\n{body}"
        return _PROMPT_INSTRUCTIONS
    stub = payload.get("prompt")
    if isinstance(stub, str) and stub.strip():
        return f"{_FIX_INSTRUCTIONS}\n\n{stub}"
    raise ValueError(f"cannot build a training prompt for task {task.id!r}: unsupported payload")


__all__ = ["SYSTEM_MESSAGE", "build_prompt"]
