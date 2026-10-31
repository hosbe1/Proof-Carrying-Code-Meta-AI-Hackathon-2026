"""Tests for the shared training prompts (issue #49).

``build_prompt`` is the reference Vericoding/HumanEvalFix user prompt: the sampling
harness and the round 0 tool must import it instead of copying it, so their user
messages compare byte-identical with the test-split prompt guard of the SFT export.
"""

from __future__ import annotations

from typing import Any

import pytest
import training.export_sft as export_sft
from training.prompts import SYSTEM_MESSAGE, build_prompt

from pcc.evaluation import prompts as canonical_prompts
from pcc.evaluation.datasets import Task


def _vericoding_task(
    row_id: str,
    *,
    description: str = "Add two numbers.",
    preamble: str = "",
    helpers: str = "",
    spec: str = "method add(a: int, b: int) returns (result: int)",
    code: str = "{ result := a + b; }",
    postamble: str = "",
) -> Task:
    """One hand-written Vericoding-style task; ``vc-code`` is the reference answer."""
    payload: dict[str, Any] = {
        "id": row_id,
        "language": "dafny",
        "source": "apps",
        "source-id": f"apps_{row_id}",
        "vc-description": description,
        "vc-preamble": preamble,
        "vc-helpers": helpers,
        "vc-spec": spec,
        "vc-code": code,
        "vc-postamble": postamble,
    }
    return Task(
        id=f"vericoding/{row_id}",
        source_task=f"vericoding/apps/apps_{row_id}",
        payload=payload,
    )


def _humanevalfix_task(stub: str = "def add(a, b):\n    return a - b\n") -> Task:
    """One hand-written HumanEvalFix-style task with a buggy-function stub."""
    payload: dict[str, Any] = {
        "task_id": "Python/0",
        "prompt": stub,
        "declaration": "def add(a, b):",
        "canonical_solution": "    return a + b",
        "buggy_solution": "    return a - b",
        "bug_type": "wrong operator",
        "test": "def check(add):\n    assert add(1, 2) == 3\ncheck(add)",
        "entry_point": "add",
    }
    return Task(
        id="humanevalfix/Python/0",
        source_task="humaneval/HumanEval/0",
        payload=payload,
    )


def test_vericoding_prompt_holds_spec_but_not_reference_answer() -> None:
    prompt = build_prompt(_vericoding_task("DA0002", spec="method solve(n: int) returns (result: int)"))
    assert "method solve(n: int) returns (result: int)" in prompt
    assert "{ result := a + b; }" not in prompt


def test_vericoding_prompt_skips_empty_blocks() -> None:
    prompt = build_prompt(_vericoding_task("DA0004", description="Wave over spectators."))
    assert "\n\n\n" not in prompt
    assert prompt.strip() != ""


def test_humanevalfix_prompt_uses_python_instructions() -> None:
    stub = "def add(a, b):\n    return a - b\n"
    prompt = build_prompt(_humanevalfix_task(stub))
    assert stub in prompt
    assert "Python" in prompt
    assert "Dafny" not in prompt


def test_build_prompt_rejects_unknown_payload_shape() -> None:
    task = Task(id="x/Y", source_task="x", payload={"id": "Y"})
    with pytest.raises(ValueError, match="prompt"):
        build_prompt(task)


def test_export_uses_the_shared_prompt_builder() -> None:
    """The SFT export must reuse the shared builder so guards stay byte-identical."""
    assert export_sft.build_prompt is build_prompt


def test_canonical_prompts_live_in_pcc_evaluation() -> None:
    """``training/prompts.py`` re-exports the canonical ``pcc.evaluation.prompts`` objects."""
    assert build_prompt is canonical_prompts.build_prompt
    assert SYSTEM_MESSAGE == canonical_prompts.SYSTEM_MESSAGE
