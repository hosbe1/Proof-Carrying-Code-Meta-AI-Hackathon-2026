"""Skipped unbuildable Vericoding rows (issue #154).

Two of the 571 fixed test rows have an empty ``vc-spec`` and no usable prompt
(``vericoding/DA0152``, ``vericoding/DA0818``), so ``build_prompt`` raises
``ValueError`` for them. The harness must skip exactly those rows before any
sampling, for both ``--split test`` and a supplied ``tasks`` subset, and report
them in ``skipped.json``. All other errors are preserved.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from pcc.core import Usage
from pcc.evaluation import vericoding
from pcc.evaluation.datasets import Task
from pcc.evaluation.prompts import build_prompt as _prompts_build_prompt
from pcc.models.client import Completion, Message
from pcc.models.fake import Call

GOOD_MODEL = "muse-spark-1.3-contributor"


def empty_spec_task(row_id: str) -> Task:
    """A task with the real unbuildable shape: empty ``vc-spec``, no prompt stub."""
    payload: dict[str, Any] = {
        "id": row_id,
        "language": "dafny",
        "source": "apps",
        "source-id": f"apps_{row_id}",
        "vc-description": f"Task {row_id}.",
        "vc-preamble": "",
        "vc-helpers": "",
        "vc-spec": "",
        "vc-code": "",
        "vc-postamble": "",
    }
    return Task(
        id=f"vericoding/{row_id}",
        source_task=f"vericoding/apps/apps_{row_id}",
        payload=payload,
    )


def blank_spec_task(row_id: str) -> Task:
    """A whitespace-only ``vc-spec`` is unbuildable too."""
    task = empty_spec_task(row_id)
    payload = dict(task.payload)
    payload["vc-spec"] = "   \n  "
    return Task(id=task.id, source_task=task.source_task, payload=payload)


def _good_task() -> Task:
    return Task(
        id="vericoding/DH0000",
        source_task="vericoding/apps/apps_DH0000",
        payload={
            "id": "DH0000",
            "language": "dafny",
            "source": "apps",
            "source-id": "apps_DH0000",
            "vc-description": "Add two numbers.",
            "vc-preamble": "function Add(a: int, b: int): int\n{\n  a + b\n}",
            "vc-helpers": "",
            "vc-spec": "method add(a: int, b: int) returns (r: int)\n  ensures r == Add(a, b)",
            "vc-code": "{ r := Add(a, b); }",
            "vc-postamble": "",
        },
    )


def _good_answer(task: Task) -> str:
    return f"{vericoding.task_spec_of(task)}  r := Add(a, b);\n}}\n"


class _ScriptedClient:
    """A thread-safe scripted client: answers are keyed by a marker in the user prompt."""

    def __init__(self, answers: dict[str, list[tuple[str, float, float]]]) -> None:
        self._answers = {marker: list(queue) for marker, queue in answers.items()}
        self._lock = threading.Lock()
        self.calls: list[Call] = []

    def complete(
        self,
        model: str,
        messages: tuple[Message, ...],
        *,
        step: str,
        schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        with self._lock:
            self.calls.append(Call(model, messages, step, schema, temperature, max_tokens))
            user = next(message.content for message in messages if message.role == "user")
            for marker, queue in self._answers.items():
                if marker in user:
                    if not queue:
                        raise AssertionError(f"no scripted answer left for {marker!r}")
                    text, cost, seconds = queue.pop(0)
                    usage = Usage(
                        step=step, model=model, input_tokens=0, output_tokens=0, cost_usd=cost, seconds=seconds
                    )
                    return Completion(text, usage)
            raise AssertionError(f"no scripted answer for prompt: {user[:80]!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_empty_vc_spec_payload_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported payload"):
        _prompts_build_prompt(empty_spec_task("DA0152"))


def test_unbuildable_rows_skipped_before_sampling(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "skipped"
    good = _good_task()
    bad_first = empty_spec_task("DA0152")
    bad_second = blank_spec_task("DA0818")
    client = _ScriptedClient({"method add(": [(_good_answer(good), 0.2, 1.0)]})
    summary = vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(good, bad_first, bad_second),
        client=client,
        verify_fn=lambda answer, frozen: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert len(client.calls) == 1
    lines = _read_jsonl(out / "results.jsonl")
    assert [line["task_id"] for line in lines] == [good.id]
    skipped = json.loads((out / "skipped.json").read_text(encoding="utf-8"))
    assert skipped == {
        "total_tasks": 3,
        "eligible_tasks": 1,
        "skipped_count": 2,
        "skipped_ids": ["vericoding/DA0152", "vericoding/DA0818"],
    }
    assert summary["metrics"] is not None
    assert summary["metrics"]["total"] == 1
    assert summary["tasks"] == [good.id]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "vericoding/DA0152" in text
    assert "vericoding/DA0818" in text


def test_non_value_error_from_build_prompt_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "runs" / "other-error"

    def flaky(task: Task) -> str:
        if task.id == "vericoding/DA9999":
            raise RuntimeError("boom: unexpected prompt failure")
        return _prompts_build_prompt(task)

    monkeypatch.setattr("pcc.evaluation.vericoding.build_prompt", flaky)
    bad = Task(id="vericoding/DA9999", source_task="vericoding/apps/apps_DA9999", payload={"id": "DA9999"})
    with pytest.raises(RuntimeError, match="boom"):
        vericoding.run_harness(
            split="test",
            out_dir=out,
            model=GOOD_MODEL,
            attempts=1,
            max_cost_usd=10.0,
            tasks=(_good_task(), bad),
            client=_ScriptedClient({"method add(": [(_good_answer(_good_task()), 0.2, 1.0)]}),
            verify_fn=lambda answer, frozen: (True, 0.5),
            git_sha="0" * 40,
            started_utc="2026-10-31T00:00:00Z",
        )


def test_resume_redetects_skips_without_new_calls(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "resume-skip"
    good = _good_task()
    tasks = (good, empty_spec_task("DA0152"), blank_spec_task("DA0818"))
    client = _ScriptedClient({"method add(": [(_good_answer(good), 0.2, 1.0)]})
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=tasks,
        client=client,
        verify_fn=lambda answer, frozen: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    before = (out / "results.jsonl").read_bytes()
    assert len(client.calls) == 1
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=tasks,
        client=client,
        verify_fn=lambda answer, frozen: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert len(client.calls) == 1
    assert (out / "results.jsonl").read_bytes() == before
    skipped = json.loads((out / "skipped.json").read_text(encoding="utf-8"))
    assert skipped["skipped_ids"] == ["vericoding/DA0152", "vericoding/DA0818"]
    assert skipped["total_tasks"] == 3
    assert skipped["eligible_tasks"] == 1


def test_all_skipped_selection_writes_empty_results(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "all-skipped"
    tasks = (empty_spec_task("DA0152"), blank_spec_task("DA0818"))
    client = _ScriptedClient({})
    summary = vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=tasks,
        client=client,
        verify_fn=lambda answer, frozen: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert client.calls == []
    assert _read_jsonl(out / "results.jsonl") == []
    assert summary["status"] == "done"
    assert summary["metrics"] is not None
    assert summary["metrics"]["total"] == 0
    assert summary["tasks"] == []
    skipped = json.loads((out / "skipped.json").read_text(encoding="utf-8"))
    assert skipped == {
        "total_tasks": 2,
        "eligible_tasks": 0,
        "skipped_count": 2,
        "skipped_ids": ["vericoding/DA0152", "vericoding/DA0818"],
    }
