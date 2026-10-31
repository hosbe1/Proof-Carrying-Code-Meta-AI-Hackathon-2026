"""Tests for the Vericoding harness (issue #46).

Sampling is scripted: ``FakeClient`` and the thread-safe ``ScriptedClient`` answer without any
network, and ``verify_fn`` replaces Dafny except in the one ``dafny``-marked example that checks a
real proof and a real cheat against the Dafny CLI.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest
import training.prompts as training_prompts

from pcc.core import ToolError, Usage
from pcc.evaluation import __main__ as eval_main
from pcc.evaluation import vericoding
from pcc.evaluation.datasets import Task
from pcc.evaluation.prompts import SYSTEM_MESSAGE, build_prompt
from pcc.models import FakeClient
from pcc.models.client import Completion, Message
from pcc.models.fake import Call
from training import grpo as training_grpo

GOOD_MODEL = "muse-spark-1.3-contributor"
OTHER_MODEL = "muse-spark-1.3"


def _payload(
    row_id: str,
    *,
    description: str,
    preamble: str = "",
    helpers: str = "",
    spec: str,
    code: str,
    postamble: str = "",
) -> dict[str, Any]:
    return {
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


def add_task() -> Task:
    """A task whose reference body verifies: ``r := Add(a, b)``."""
    return Task(
        id="vericoding/DH0000",
        source_task="vericoding/apps/apps_DH0000",
        payload=_payload(
            "DH0000",
            description="Add two numbers.",
            preamble="function Add(a: int, b: int): int\n{\n  a + b\n}",
            spec="method add(a: int, b: int) returns (r: int)\n  ensures r == Add(a, b)",
            code="{ r := Add(a, b); }",
        ),
    )


def neg_task() -> Task:
    """A task with a different method marker so scripted answers stay with their task."""
    return Task(
        id="vericoding/DH0001",
        source_task="vericoding/apps/apps_DH0001",
        payload=_payload(
            "DH0001",
            description="Negate a number.",
            preamble="function Neg(x: int): int\n{\n  -x\n}",
            spec="method neg(x: int) returns (r: int)\n  ensures r == Neg(x)",
            code="{ r := Neg(x); }",
        ),
    )


def clamp_task() -> Task:
    return Task(
        id="vericoding/DH0002",
        source_task="vericoding/apps/apps_DH0002",
        payload=_payload(
            "DH0002",
            description="Clamp a number.",
            spec="method clamp(x: int, lo: int, hi: int) returns (r: int)\n  ensures lo <= r && r <= hi",
            code="{ r := x; }",
        ),
    )


def frozen_of(task: Task) -> str:
    return vericoding.task_spec_of(task)


def good_answer(task: Task) -> str:
    """A complete answer that keeps the frozen prefix: the reference body after it."""
    code = task.payload["vc-code"]
    assert isinstance(code, str) and code.startswith("{") and code.endswith("}")
    return f"{frozen_of(task)}  {code[1:-1].strip()}\n}}\n"


def cheat_answer(task: Task) -> str:
    """An answer that changes the spec: the frozen prefix no longer occurs in it."""
    spec = task.payload["vc-spec"]
    assert isinstance(spec, str)
    return spec.replace("ensures", "ensures r == r ||", 1) + "\n{\n  r := 0;\n}\n"


class ScriptedClient:
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


def _ok(answer: str, frozen: str) -> tuple[bool, float]:
    return True, 0.5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_task_spec_of_joins_prefix_blocks() -> None:
    assert frozen_of(add_task()) == (
        "function Add(a: int, b: int): int\n{\n  a + b\n}\n"
        "method add(a: int, b: int) returns (r: int)\n  ensures r == Add(a, b)\n{\n"
    )
    assert (
        frozen_of(clamp_task())
        == "method clamp(x: int, lo: int, hi: int) returns (r: int)\n  ensures lo <= r && r <= hi\n{\n"
    )


def test_task_spec_of_matches_grpo_prefix_contract() -> None:
    """The harness and the GRPO reward share the frozen prefix byte for byte."""
    for task in (add_task(), neg_task(), clamp_task()):
        assert frozen_of(task) == training_grpo._task_spec(task)


def test_task_spec_of_skips_blank_parts() -> None:
    task = neg_task()
    payload = dict(task.payload)
    payload["vc-helpers"] = "   \n  "
    blank_helpers = Task(id=task.id, source_task=task.source_task, payload=payload)
    assert frozen_of(blank_helpers) == frozen_of(task)
    assert "vc-code" not in frozen_of(task)
    description = task.payload["vc-description"]
    assert isinstance(description, str)
    assert description not in frozen_of(task)


def test_messages_send_system_message_and_shared_prompt(tmp_path: Path) -> None:
    task = add_task()
    fake = FakeClient([good_answer(task)])
    out = tmp_path / "runs" / "messages"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(task,),
        client=fake,
        verify_fn=_ok,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert fake.requests == [
        (
            GOOD_MODEL,
            (Message("system", SYSTEM_MESSAGE), Message("user", build_prompt(task))),
            "vericoding",
        )
    ]
    assert fake.calls[0].temperature == pytest.approx(0.8)
    assert fake.calls[0].max_tokens == 4096


def test_first_verified_answer_wins(tmp_path: Path) -> None:
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    good = good_answer(task)
    seen: list[str] = []

    def verify_second(answer: str, prefix: str) -> tuple[bool, float]:
        seen.append(answer)
        return answer == good, 0.5

    out = tmp_path / "runs" / "second"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=2,
        max_cost_usd=10.0,
        tasks=(task,),
        client=ScriptedClient({"method add(": [(bad, 0.25, 1.0), (good, 0.25, 1.0)]}),
        verify_fn=verify_second,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert seen == [bad, good]
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is True
    assert line["attempts"] == 2
    assert line["verified_attempts"] == 1
    assert line["answer"] == good
    assert line["cost_usd"] == pytest.approx(0.5)
    assert line["seconds"] == pytest.approx(3.0)


def test_failed_attempts_keep_last_answer(tmp_path: Path) -> None:
    task = add_task()
    frozen = frozen_of(task)
    out = tmp_path / "runs" / "failed"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=2,
        max_cost_usd=10.0,
        tasks=(task,),
        client=ScriptedClient({"method add(": [(f"{frozen}  r := 0;\n}}\n", 0.25, 1.0)] * 2}),
        verify_fn=lambda answer, prefix: (False, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is False
    assert line["verified_attempts"] == 0
    assert line["answer"] == f"{frozen}  r := 0;\n}}\n"


def test_run_json_holds_protocol_fields(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "fields"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=2,
        max_cost_usd=10.0,
        tasks=(add_task(),),
        client=ScriptedClient({"method add(": [(good_answer(add_task()), 0.2, 1.0)] * 2}),
        verify_fn=_ok,
        git_sha="a" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["harness"] == "vericoding"
    assert run["endpoint"] == "meta"
    assert run["model"] == GOOD_MODEL
    assert run["attempts"] == 2
    assert run["split"] == "test"
    expected_sha = hashlib.sha256(vericoding.SPLIT_FILE.read_bytes()).hexdigest()
    assert run["split_sha256"] == expected_sha
    assert run["contributor"] is True
    assert run["git_sha"] == "a" * 40
    assert run["started_utc"] == "2026-10-31T07:53:42Z"
    assert run["max_cost_usd"] == pytest.approx(10.0)


def test_contributor_false_for_standard_model(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "standard"
    vericoding.run_harness(
        split="train",
        out_dir=out,
        model=OTHER_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        endpoint="glimmer",
        tasks=(add_task(),),
        client=ScriptedClient({"method add(": [(good_answer(add_task()), 0.2, 1.0)]}),
        verify_fn=_ok,
        git_sha="b" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["contributor"] is False
    assert run["endpoint"] == "glimmer"
    assert run["split"] == "train"


@pytest.mark.dafny
def test_two_fixture_tasks_with_fake_client(tmp_path: Path) -> None:
    """Example 1: one correct answer verifies, one cheat does not (needs the Dafny CLI)."""
    first, second = add_task(), neg_task()
    fake = FakeClient([good_answer(first), cheat_answer(second)])
    out = tmp_path / "runs" / "example"
    summary = vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        concurrency=1,
        tasks=(first, second),
        client=fake,
        git_sha="c" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    lines = read_jsonl(out / "results.jsonl")
    assert [line["verified"] for line in lines] == [True, False]
    assert [line["task_id"] for line in lines] == ["vericoding/DH0000", "vericoding/DH0001"]
    assert summary["metrics"]["verified"] == 1
    assert summary["metrics"]["total"] == 2


def test_resume_skips_completed_tasks(tmp_path: Path) -> None:
    """Example 2: tasks already in results.jsonl are not sampled again."""
    out = tmp_path / "runs" / "resume"
    first, second = add_task(), neg_task()
    client = ScriptedClient(
        {
            "method add(": [(good_answer(first), 0.2, 1.0)],
            "method neg(": [(good_answer(second), 0.3, 2.0)],
        }
    )
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(first, second),
        client=client,
        verify_fn=_ok,
        git_sha="d" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    before = (out / "results.jsonl").read_bytes()
    before_run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert len(client.calls) == 2
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(first, second),
        client=client,
        verify_fn=_ok,
        git_sha="d" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    assert len(client.calls) == 2
    assert (out / "results.jsonl").read_bytes() == before
    after_run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    for key in ("started_utc", "git_sha", "split_sha256", "contributor"):
        assert after_run[key] == before_run[key]


def test_cost_cap_stops_and_reports(tmp_path: Path) -> None:
    """Example 3: the run stops at --max-cost-usd and summary.md says so."""
    out = tmp_path / "runs" / "capped"
    tasks = (add_task(), neg_task(), clamp_task())
    answers = {
        "method add(": [(good_answer(add_task()), 0.5, 1.0)],
        "method neg(": [(good_answer(neg_task()), 0.5, 1.0)],
        "method clamp(": [(good_answer(clamp_task()), 0.5, 1.0)],
    }
    summary = vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=1.0,
        concurrency=1,
        tasks=tasks,
        client=ScriptedClient(answers),
        verify_fn=_ok,
        git_sha="e" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    lines = read_jsonl(out / "results.jsonl")
    assert len(lines) == 2
    assert summary["stopped_early"] is True
    assert "stopped at cost cap" in (out / "summary.md").read_text(encoding="utf-8")


def test_cheat_never_reaches_verifier(tmp_path: Path) -> None:
    """An answer that changes the spec counts as a cheat without calling Dafny."""
    task = add_task()
    calls: list[str] = []

    def spy(answer: str, frozen: str) -> tuple[bool, float]:
        calls.append(answer)
        return True, 0.1

    out = tmp_path / "runs" / "cheat"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(task,),
        client=ScriptedClient({"method add(": [(cheat_answer(task), 0.2, 1.0)]}),
        verify_fn=spy,
        git_sha="f" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    assert calls == []
    assert read_jsonl(out / "results.jsonl")[0]["verified"] is False


def test_tool_error_keeps_partial_results(tmp_path: Path) -> None:
    """Errors: ToolError stops the run, the lines so far are kept."""
    out = tmp_path / "runs" / "aborted"
    first, second = add_task(), neg_task()

    def abort_on_neg(answer: str, frozen: str) -> tuple[bool, float]:
        if "Neg(" in frozen:
            raise ToolError("boom: verifier failed")
        return True, 0.5

    client = ScriptedClient(
        {
            "method add(": [(good_answer(first), 0.2, 1.0)],
            "method neg(": [(good_answer(second), 0.2, 1.0)],
        }
    )
    with pytest.raises(ToolError, match="boom"):
        vericoding.run_harness(
            split="test",
            out_dir=out,
            model=GOOD_MODEL,
            attempts=1,
            max_cost_usd=10.0,
            concurrency=1,
            tasks=(first, second),
            client=client,
            verify_fn=abort_on_neg,
            git_sha="0" * 40,
            started_utc="2026-10-31T07:53:42Z",
        )
    assert len(read_jsonl(out / "results.jsonl")) == 1
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["status"] == "error"


def test_concurrent_sampling_covers_all_tasks(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "concurrent"
    tasks = (add_task(), neg_task(), clamp_task())
    client = ScriptedClient(
        {
            "method add(": [(good_answer(add_task()), 0.1, 1.0)],
            "method neg(": [(good_answer(neg_task()), 0.1, 1.0)],
            "method clamp(": [(good_answer(clamp_task()), 0.1, 1.0)],
        }
    )
    summary = vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        concurrency=4,
        tasks=tasks,
        client=client,
        verify_fn=_ok,
        git_sha="0" * 40,
        started_utc="2026-10-31T07:53:42Z",
    )
    lines = read_jsonl(out / "results.jsonl")
    assert [line["task_id"] for line in lines] == ["vericoding/DH0000", "vericoding/DH0001", "vericoding/DH0002"]
    assert summary["metrics"]["verified"] == 3


def test_parser_defaults() -> None:
    args = eval_main.build_parser().parse_args(
        [
            "vericoding",
            "--split",
            "test",
            "--model",
            GOOD_MODEL,
            "--attempts",
            "2",
            "--max-cost-usd",
            "1",
            "--out",
            "runs/a",
        ]
    )
    assert args.endpoint == "meta"
    assert args.concurrency == 16
    assert args.workers is None


def test_started_utc_default_format(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "clock"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(add_task(),),
        client=ScriptedClient({"method add(": [(good_answer(add_task()), 0.2, 1.0)]}),
        verify_fn=_ok,
    )
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", run["started_utc"]) is not None
    assert re.fullmatch(r"[0-9a-f]{40}", run["git_sha"]) is not None


def test_gitignore_ignores_runs() -> None:
    root = Path(vericoding.__file__).resolve().parents[3]
    lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "runs/" in lines


def test_training_prompts_stay_identical() -> None:
    assert training_prompts.build_prompt is build_prompt
