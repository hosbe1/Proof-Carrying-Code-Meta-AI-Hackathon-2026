"""Tests for the canonical evaluation prompts (issue #46).

``SYSTEM_MESSAGE`` and ``build_prompt`` live in ``pcc.evaluation.prompts`` so the package never
imports ``training``; ``training/prompts.py`` re-exports them unchanged. These tests lock the
behaviour on this side and the byte-identical re-export on the training side.
"""

from __future__ import annotations

from typing import Any

import training.prompts as training_prompts

from pcc.evaluation import prompts as eval_prompts
from pcc.evaluation.datasets import Task
from pcc.evaluation.prompts import SYSTEM_MESSAGE, build_prompt


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


def test_vericoding_prompt_holds_spec_but_not_reference_answer() -> None:
    prompt = build_prompt(_vericoding_task("DA0002", spec="method solve(n: int) returns (result: int)"))
    assert "method solve(n: int) returns (result: int)" in prompt
    assert "{ result := a + b; }" not in prompt


def test_vericoding_prompt_skips_empty_blocks() -> None:
    prompt = build_prompt(_vericoding_task("DA0004"))
    assert "\n\n\n" not in prompt
    assert prompt.strip() != ""


def test_training_module_re_exports_the_canonical_objects() -> None:
    """``training/prompts.py`` must re-export, not copy, the canonical prompt pieces."""
    assert training_prompts.build_prompt is build_prompt
    assert training_prompts.SYSTEM_MESSAGE == SYSTEM_MESSAGE
    assert training_prompts.SYSTEM_MESSAGE is eval_prompts.SYSTEM_MESSAGE
