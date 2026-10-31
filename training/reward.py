"""The graded reward of the proof step for training (docs/plan.md, Training).

``reward`` grades one candidate Dafny program with Dafny's own verdict: verified with a clean
anti-cheat scan earns 1, any ordinary nonverification earns 0, and a changed or absent task
prefix or any anti-cheat violation earns -1. A broken Dafny tool is not a grade: ``ToolError``
escapes as an exception.
"""

from __future__ import annotations

from pcc.verifier import anti_cheat, verify


def reward(answer: str, task_spec: str, *, resource_limit: int) -> float:
    """Grade the candidate Dafny program ``answer`` against the immutable prefix ``task_spec``.

    ``answer`` is the complete candidate Dafny source passed to ``verifier.verify``.
    ``task_spec`` is the exact immutable Dafny task prefix (the supplied specification and
    method declaration through the opening method brace); it must occur unchanged in
    ``answer``, which may only add method body and ghost proof content after it.

    A changed or absent ``task_spec``, or any ``verifier.anti_cheat`` violation, gives -1.0.
    Otherwise ``verifier.verify`` decides: verified gives 1.0, every ordinary nonverification
    (a failed proof, a resolve error, a timeout, unparseable diagnostics) gives 0.0.

    Raises ``ToolError`` when the Dafny CLI is missing or reports the wrong version.
    """
    if not task_spec or task_spec not in answer:
        return -1.0
    if anti_cheat(answer, task_spec):
        return -1.0
    outcome = verify(answer, resource_limit=resource_limit)
    return 1.0 if outcome.verified else 0.0


__all__ = ["reward"]
