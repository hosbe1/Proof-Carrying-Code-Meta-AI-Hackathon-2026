"""Tests for rejection sampling (issue #69).

Every example and edge case of the ticket runs offline: the glimmer path talks to a
localhost stub (as in the model client tests), the meta path uses a cost/usage-injecting
test double around the production ``FakeClient`` (which stays unchanged), and the reward
is a module-level fake so the process pool can pickle it. No network, no Dafny, no GPU.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import os
import threading
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from training.rejection import RoundSummary, main, prompt_sha256, run

from pcc.core import ToolError, Usage
from pcc.evaluation.datasets import Task
from pcc.evaluation.prompts import SYSTEM_MESSAGE, build_prompt
from pcc.evaluation.vericoding import task_spec_of
from pcc.models import FakeClient
from pcc.models.client import Completion, Message
from pcc.models.fake import Call
from training import rejection


def fake_ok_reward(answer: str, task_spec: str) -> float:
    """Module-level fake reward (picklable): 1.0 for answers starting with ``ok``."""
    assert task_spec != ""
    return 1.0 if answer.startswith("ok") else 0.0


def fake_prefix_reward(answer: str, task_spec: str) -> float:
    """A changed immutable prefix fails; formatting-only changes may be normalized first."""
    return 1.0 if answer.startswith(task_spec) else -1.0


def test_fenced_answer_with_extra_blank_line_reuses_exact_prefix(tmp_path: Path) -> None:
    original = _task("DA0001", TASK_ONE)
    task = dataclasses.replace(original, payload={**original.payload, "vc-preamble": "predicate Ready() { true }"})
    frozen = task_spec_of(task)
    answer = "```dafny\n\n" + frozen.replace("}\nmethod", "}\n\nmethod") + "    r := 0;\n}\n```"
    summary = run(
        (task,), FakeClient([answer]), model="m", attempts=1, out=tmp_path / "round.jsonl", reward_fn=fake_prefix_reward
    )

    assert summary.tasks_solved == 1
    saved = _read_jsonl(tmp_path / "round.jsonl")
    assert saved[0]["messages"][2]["content"] == frozen + "    r := 0;\n}\n"


def test_changed_prefix_still_fails_rejection(tmp_path: Path) -> None:
    task = _task("DA0001", TASK_ONE)
    frozen = task_spec_of(task)
    answer = frozen.replace("returns (r: int)", "returns (r: bool)") + "    r := 0;\n}\n"
    summary = run(
        (task,), FakeClient([answer]), model="m", attempts=1, out=tmp_path / "round.jsonl", reward_fn=fake_prefix_reward
    )

    assert summary.tasks_solved == 0


def fake_solved_reward(answer: str, task_spec: str) -> float:
    """Module-level fake reward: 1.0 for ``yes`` answers, 0.0 otherwise."""
    assert task_spec != ""
    return 1.0 if answer == "yes" else 0.0


def fake_mixed_reward(answer: str, task_spec: str) -> float:
    """Module-level fake reward: -1.0 for cheats, 0.0 for everything else."""
    assert task_spec != ""
    if "assume false" in answer:
        return -1.0
    return 0.0


def fake_boom_reward(answer: str, task_spec: str) -> float:
    """Module-level fake reward that fails like a broken Dafny tool."""
    raise ToolError("boom: reward failed")


def fake_value_reward(answer: str, task_spec: str) -> float:
    """Module-level fake reward (picklable) that fails with ``ValueError``."""
    raise ValueError("boom: broken scorer")


def fake_graded_reward(answer: str, task_spec: str, *, resource_limit: int) -> float:
    """Module-level fake for the ``reward`` binding: 1.0 for ``ok`` answers.

    CLI tests patch ``rejection.reward`` with this, the same seam the GRPO tests use.
    """
    assert task_spec != ""
    assert resource_limit > 0
    return 1.0 if answer.startswith("ok") else 0.0


def fake_probe_reward(answer: str, task_spec: str, *, resource_limit: int) -> float:
    """Module-level fake recording the bound resource limit into a probe file.

    The process pool runs in a child process, so the limit travels through a file named
    by ``PCC_REJECTION_TEST_PROBE`` instead of shared memory.
    """
    probe = os.environ.get("PCC_REJECTION_TEST_PROBE", "")
    if probe:
        with open(probe, "a", encoding="utf-8") as handle:
            handle.write(f"{resource_limit}\n")
    return 0.0


def _task(row_id: str, marker: str, *, source: str = "apps") -> Task:
    """One hand-written Vericoding-style task carrying ``marker`` in its prompt."""
    payload: dict[str, Any] = {
        "id": row_id,
        "language": "dafny",
        "source": source,
        "source-id": f"{source}_{row_id}",
        "vc-description": f"Prove task {marker}.",
        "vc-preamble": "",
        "vc-helpers": "",
        "vc-spec": f"method {marker}() returns (r: int) // {marker}",
        "vc-code": "{ r := 0; }",
        "vc-postamble": "",
    }
    return Task(
        id=f"vericoding/{row_id}",
        source_task=f"vericoding/{source}/{source}_{row_id}",
        payload=payload,
    )


TASK_ONE = "TASK-ONE"
TASK_TWO = "TASK-TWO"


class UsageClient:
    """A cost/usage-injecting wrapper around the production ``FakeClient``.

    The production client always reports zero tokens and zero cost, so tests that need
    usage wrap it here instead of changing it. Answers come from the inner client in
    order; every completion carries the configured usage.
    """

    def __init__(
        self,
        answers: list[str],
        *,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cost_usd: float = 0.0,
        seconds: float = 0.0,
    ) -> None:
        self._fake = FakeClient(list(answers))
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost_usd = cost_usd
        self._seconds = seconds

    @property
    def calls(self) -> list[Call]:
        """The recorded calls of the inner client."""
        return self._fake.calls

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
        completion = self._fake.complete(
            model, messages, step=step, schema=schema, temperature=temperature, max_tokens=max_tokens
        )
        return Completion(
            completion.text,
            Usage(
                step=step,
                model=model,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cost_usd=self._cost_usd,
                seconds=self._seconds,
            ),
        )


class MarkerClient:
    """A thread-safe usage-injecting client answering per prompt marker.

    ``answers`` maps a prompt marker to its queued answers; the answer for a call is the
    next queued answer of the marker contained in the user prompt.
    """

    def __init__(
        self,
        answers: Mapping[str, list[str]],
        *,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cost_usd: float = 0.0,
    ) -> None:
        self._queues = {marker: list(queue) for marker, queue in answers.items()}
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost_usd = cost_usd
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
            for marker, queue in self._queues.items():
                if marker in user:
                    if not queue:
                        raise AssertionError(f"no scripted answer left for {marker!r}")
                    return Completion(
                        queue.pop(0),
                        Usage(
                            step=step,
                            model=model,
                            input_tokens=self._input_tokens,
                            output_tokens=self._output_tokens,
                            cost_usd=self._cost_usd,
                        ),
                    )
            raise AssertionError(f"no scripted answer for prompt: {user[:80]!r}")


class FlakyClient(MarkerClient):
    """A marker client failing with ``ToolError`` from the given call number on."""

    def __init__(
        self,
        answers: Mapping[str, list[str]],
        *,
        fail_from: int,
        message: str = "boom: endpoint down",
    ) -> None:
        super().__init__(answers)
        self._fail_from = fail_from
        self._message = message
        self.total_calls = 0

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
            self.total_calls += 1
            if self.total_calls >= self._fail_from:
                raise ToolError(self._message)
        return super().complete(
            model, messages, step=step, schema=schema, temperature=temperature, max_tokens=max_tokens
        )


class StubState:
    """Scripted answers of the localhost stub, keyed by prompt marker."""

    def __init__(self, answers: Mapping[str, list[str]]) -> None:
        self.queues = {marker: list(queue) for marker, queue in answers.items()}
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()


def _ok_body(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _serve(state: StubState, *, status: int = 200, error: str = "") -> ThreadingHTTPServer:
    parent: dict[str, Any] = {"state": state, "status": status, "error": error}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            st: StubState = parent["state"]
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload: Any = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                payload = None
            with st.lock:
                st.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": payload,
                    }
                )
                text: str | None = None
                if isinstance(payload, dict):
                    for message in payload.get("messages", []):
                        if isinstance(message, dict) and message.get("role") == "user":
                            for marker, queue in st.queues.items():
                                content = message.get("content", "")
                                if isinstance(content, str) and marker in content and queue:
                                    text = queue.pop(0)
                if text is None:
                    if parent["status"] == 200:
                        answer: Any = {"error": {"message": "stub ran out of scripted responses"}}
                        code = 500
                    else:
                        answer = {"error": {"message": parent["error"] or "stub error"}}
                        code = parent["status"]
                else:
                    answer = _ok_body(text)
                    code = 200
            raw_body = json.dumps(answer).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw_body)))
            self.end_headers()
            self.wfile.write(raw_body)

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _glimmer_env(monkeypatch: pytest.MonkeyPatch, server: ThreadingHTTPServer) -> None:
    monkeypatch.setenv("PCC_GLIMMER_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("PCC_GLIMMER_API_KEY", "test-key")


def _stub_split(monkeypatch: pytest.MonkeyPatch, tasks: list[Task]) -> None:
    def _fake(name: str, which: str) -> tuple[Task, ...]:
        assert (name, which) == ("vericoding", "train")
        return tuple(tasks)

    monkeypatch.setattr(rejection, "split", _fake)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _user_of(line: dict[str, Any]) -> str:
    for message in line["messages"]:
        if message["role"] == "user":
            return str(message["content"])
    raise AssertionError("chat line holds no user message")


def test_prompt_sha256_is_lowercase_hex_of_utf8_prompt() -> None:
    task = _task("DA0001", TASK_ONE)
    expected = hashlib.sha256(build_prompt(task).encode("utf-8")).hexdigest()
    assert prompt_sha256(task) == expected
    assert expected == expected.lower()
    assert len(expected) == 64


def test_run_defaults_match_ticket() -> None:
    params = inspect.signature(run).parameters
    assert params["reward_fn"].default is None
    assert params["skip_solved"].default is None
    assert params["concurrency"].default == 64
    assert params["workers"].default is None
    assert params["temperature"].default == 0.8
    assert params["max_tokens"].default == 4096
    assert params["max_per_task"].default == 2
    assert params["max_cost_usd"].default is None
    assert params["resume"].default is False


def test_example1_two_tasks_glimmer_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Example 1: 8 attempts on two tasks, only task 1 verifies, capped at 2 lines."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    _stub_split(monkeypatch, tasks)
    monkeypatch.setattr(rejection, "reward", fake_graded_reward)
    state = StubState({TASK_ONE: [f"ok-{i}" for i in range(1, 9)], TASK_TWO: ["bad"] * 8})
    server = _serve(state)
    try:
        _glimmer_env(monkeypatch, server)
        out = tmp_path / "round0.jsonl"
        code = main(
            [
                "--endpoint",
                "glimmer",
                "--model",
                "glimmer",
                "--tasks",
                "train",
                "--attempts",
                "8",
                "--out",
                str(out),
            ]
        )
    finally:
        server.shutdown()
    assert code == 0
    lines = _read_jsonl(out)
    assert len(lines) == 2
    users = {_user_of(line) for line in lines}
    assert users == {build_prompt(tasks[0])}
    summary = json.loads((tmp_path / "round0.jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["tasks_solved"] == 1
    assert summary["tasks_run"] == 2
    assert summary["attempts"] == 16
    assert summary["input_tokens"] == 1600
    assert summary["output_tokens"] == 800
    assert summary["stop_reason"] == "done"
    assert len(_read_jsonl(Path(str(out) + ".attempts.jsonl"))) == 16
    task_lines = _read_jsonl(Path(str(out) + ".tasks.jsonl"))
    assert len(task_lines) == 2
    by_sha = {line["prompt_sha256"]: line for line in task_lines}
    assert by_sha[prompt_sha256(tasks[0])]["solved"] is True
    assert by_sha[prompt_sha256(tasks[0])]["attempts"] == 8
    assert by_sha[prompt_sha256(tasks[1])]["solved"] is False


def test_example2_skip_solved_round1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Example 2: the round 0 output as skip_solved leaves only task 2 to sample."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    _stub_split(monkeypatch, tasks)
    monkeypatch.setattr(rejection, "reward", fake_graded_reward)
    skipped = tmp_path / "round0.jsonl"
    skipped.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": build_prompt(tasks[0])},
                    {"role": "assistant", "content": "ok-old"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = StubState({TASK_TWO: [f"ok-r1-{i}" for i in range(1, 9)]})
    server = _serve(state)
    try:
        _glimmer_env(monkeypatch, server)
        out = tmp_path / "round1.jsonl"
        code = main(
            [
                "--endpoint",
                "glimmer",
                "--model",
                "glimmer",
                "--tasks",
                "train",
                "--attempts",
                "8",
                "--out",
                str(out),
                "--skip-solved",
                str(skipped),
            ]
        )
    finally:
        server.shutdown()
    assert code == 0
    assert len(state.requests) == 8
    assert all(TASK_ONE not in json.dumps(item["body"]) for item in state.requests)
    lines = _read_jsonl(out)
    assert len(lines) == 2
    assert {_user_of(line) for line in lines} == {build_prompt(tasks[1])}


def test_example3_negative_and_zero_rewards_not_written(tmp_path: Path) -> None:
    """Example 3: -1 and 0 rewards count as attempts but are never written."""
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: ["method m() { assume false; }", "bad"]})
    summary = run(
        (task,),
        client,
        model="m",
        attempts=2,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_mixed_reward,
        concurrency=1,
    )
    assert summary.tasks_solved == 0
    assert summary.tasks_run == 1
    assert summary.attempts == 2
    assert summary.stop_reason == "done"
    assert _read_jsonl(tmp_path / "out.jsonl") == []


def test_example4_stub_receives_temperature_and_max_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Example 4: the stub sees temperature 0.8 and max_tokens 4096 and no schema."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    monkeypatch.setattr(rejection, "reward", fake_graded_reward)
    state = StubState({TASK_ONE: ["ok-1"]})
    server = _serve(state)
    try:
        _glimmer_env(monkeypatch, server)
        code = main(
            [
                "--endpoint",
                "glimmer",
                "--model",
                "glimmer",
                "--tasks",
                "train",
                "--attempts",
                "1",
                "--out",
                str(tmp_path / "out.jsonl"),
            ]
        )
    finally:
        server.shutdown()
    assert code == 0
    assert len(state.requests) == 1
    body = state.requests[0]["body"]
    assert body["temperature"] == pytest.approx(0.8)
    assert body["max_tokens"] == 4096
    assert "response_format" not in body


def test_example5_cost_cap_stops_sampling(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Example 5: the call crossing the cap completes, the next raises, exit is 0."""
    tasks = [_task(f"DA000{i}", f"TASK-{i}") for i in range(3)]
    markers = {f"TASK-{i}": ["yes", "yes"] for i in range(3)}
    client = MarkerClient(markers, cost_usd=0.10)
    out = tmp_path / "out.jsonl"
    summary = run(
        tuple(tasks),
        client,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_solved_reward,
        concurrency=1,
        max_cost_usd=0.5,
    )
    assert summary.stop_reason == "cost cap"
    assert summary.attempts == 5
    assert summary.cost_usd == pytest.approx(0.50)
    assert summary.tasks_solved == 2
    assert summary.tasks_run == 2
    assert len(_read_jsonl(out)) == 2
    assert len(_read_jsonl(Path(str(out) + ".attempts.jsonl"))) == 5
    assert len(_read_jsonl(Path(str(out) + ".tasks.jsonl"))) == 2


def test_example5_cli_meta_exit_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Example 5 through the CLI: cost cap exits 0 with a FakeClient-based double."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    client = UsageClient(["yes"] * 8, cost_usd=0.10)
    monkeypatch.setattr(rejection, "build_client", lambda config, env: client)
    monkeypatch.setenv("PCC_META_ALLOWED", "1")
    out = tmp_path / "out.jsonl"
    code = main(
        [
            "--endpoint",
            "meta",
            "--model",
            "muse-spark-1.3-contributor",
            "--tasks",
            "train",
            "--attempts",
            "8",
            "--out",
            str(out),
            "--max-cost-usd",
            "0.5",
            "--concurrency",
            "1",
        ]
    )
    assert code == 0
    assert len(client.calls) == 5


def test_meta_without_allowed_stops_at_first_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without PCC_META_ALLOWED=1 the router message becomes the stop reason (exit 2)."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    monkeypatch.delenv("PCC_META_ALLOWED", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    out = tmp_path / "out.jsonl"
    code = main(
        [
            "--endpoint",
            "meta",
            "--model",
            "muse-spark-1.3-contributor",
            "--tasks",
            "train",
            "--attempts",
            "2",
            "--out",
            str(out),
            "--max-cost-usd",
            "1.0",
            "--concurrency",
            "1",
        ]
    )
    assert code == 2
    summary = json.loads((tmp_path / "out.jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"].startswith("endpoint: ")
    assert "PCC_META_ALLOWED" in summary["stop_reason"] or "META_API_KEY" in summary["stop_reason"]


def test_glimmer_sends_configured_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The glimmer path authenticates with the key from the environment."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    monkeypatch.setattr(rejection, "reward", fake_graded_reward)
    state = StubState({TASK_ONE: ["ok-1"]})
    server = _serve(state)
    try:
        _glimmer_env(monkeypatch, server)
        code = main(
            [
                "--endpoint",
                "glimmer",
                "--model",
                "glimmer",
                "--tasks",
                "train",
                "--attempts",
                "1",
                "--out",
                str(tmp_path / "out.jsonl"),
            ]
        )
    finally:
        server.shutdown()
    assert code == 0
    assert state.requests[0]["auth"] == "Bearer test-key"


def test_messages_carry_system_message_and_shared_prompt(tmp_path: Path) -> None:
    """Each attempt sends the shared system message and build_prompt with step proof."""
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: ["ok-1", "ok-2"]})
    run((task,), client, model="m", attempts=2, out=tmp_path / "out.jsonl", reward_fn=fake_ok_reward, concurrency=1)
    assert len(client.calls) == 2
    for call in client.calls:
        assert call.model == "m"
        assert call.messages == (Message("system", SYSTEM_MESSAGE), Message("user", build_prompt(task)))
        assert call.step == "proof"
        assert call.schema is None
        assert call.temperature == pytest.approx(0.8)
        assert call.max_tokens == 4096


def test_output_lines_have_sft_chat_shape(tmp_path: Path) -> None:
    """Every output line has the SFT chat shape: system, user prompt, assistant answer."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    client = MarkerClient({TASK_ONE: ["ok-a", "ok-b", "ok-c"], TASK_TWO: ["bad"] * 3})
    run(
        tuple(tasks),
        client,
        model="m",
        attempts=3,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
        max_per_task=2,
    )
    lines = _read_jsonl(tmp_path / "out.jsonl")
    assert len(lines) == 2
    for line in lines:
        messages = line["messages"]
        assert [message["role"] for message in messages] == ["system", "user", "assistant"]
        assert messages[0]["content"] == SYSTEM_MESSAGE
        assert messages[1]["content"] == build_prompt(tasks[0])
        assert isinstance(messages[2]["content"], str) and messages[2]["content"].strip() != ""
    assistants = [line["messages"][2]["content"] for line in lines]
    assert assistants == ["ok-a", "ok-b"]


def test_identical_answers_written_once(tmp_path: Path) -> None:
    """Two identical answers of one task give one line."""
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: ["ok-same"] * 4})
    run(
        (task,),
        client,
        model="m",
        attempts=4,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    lines = _read_jsonl(tmp_path / "out.jsonl")
    assert len(lines) == 1
    assert lines[0]["messages"][2]["content"] == "ok-same"


def test_empty_split_warns_and_writes_empty_file(tmp_path: Path) -> None:
    """An empty split gives an empty file and a warning."""
    out = tmp_path / "out.jsonl"
    with pytest.warns(UserWarning):
        summary = run(
            (),
            MarkerClient({}),
            model="m",
            attempts=8,
            out=out,
            reward_fn=fake_ok_reward,
        )
    assert summary.tasks_solved == 0
    assert summary.tasks_run == 0
    assert summary.attempts == 0
    assert summary.stop_reason == "done"
    assert out.read_text(encoding="utf-8") == ""


def test_tool_error_stops_sampling_and_records_finished(tmp_path: Path) -> None:
    """A ToolError keeps sampled attempts and records the tasks that finished sampling."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    client = FlakyClient({TASK_ONE: ["ok-1", "ok-2"], TASK_TWO: ["ok-3", "ok-4"]}, fail_from=3)
    summary = run(
        tuple(tasks),
        client,
        model="m",
        attempts=2,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    assert summary.stop_reason == "endpoint: boom: endpoint down"
    assert summary.tasks_run == 1
    assert summary.tasks_solved == 1
    out = tmp_path / "out.jsonl"
    assert len(_read_jsonl(out)) == 2
    assert len(_read_jsonl(Path(str(out) + ".attempts.jsonl"))) == 2
    assert len(_read_jsonl(Path(str(out) + ".tasks.jsonl"))) == 1


def test_tool_error_cli_exit_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The CLI exits 2 when the endpoint fails."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    client = FlakyClient({TASK_ONE: ["ok-1"]}, fail_from=1)
    monkeypatch.setattr(rejection, "build_client", lambda config, env: client)
    code = main(
        [
            "--endpoint",
            "meta",
            "--model",
            "muse-spark-1.3-contributor",
            "--tasks",
            "train",
            "--attempts",
            "1",
            "--out",
            str(tmp_path / "out.jsonl"),
            "--max-cost-usd",
            "1.0",
        ]
    )
    assert code == 2


def test_reward_tool_error_passes_through(tmp_path: Path) -> None:
    """A ToolError from the reward propagates instead of becoming a stop reason."""
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    with pytest.raises(ToolError, match="boom: reward failed"):
        run(
            (task,),
            client,
            model="m",
            attempts=1,
            out=tmp_path / "out.jsonl",
            reward_fn=fake_boom_reward,
            concurrency=1,
        )


def test_key_never_appears_in_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The API key never appears in the stop reason or the summary file."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    state = StubState({})
    server = _serve(state, status=500, error="upstream failure")
    marker = "must-not-leak-1"
    try:
        monkeypatch.setenv("PCC_GLIMMER_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("PCC_GLIMMER_API_KEY", marker)
        code = main(
            [
                "--endpoint",
                "glimmer",
                "--model",
                "glimmer",
                "--tasks",
                "train",
                "--attempts",
                "1",
                "--out",
                str(tmp_path / "out.jsonl"),
                "--concurrency",
                "1",
            ]
        )
    finally:
        server.shutdown()
    assert code == 2
    summary_path = tmp_path / "out.jsonl.summary.json"
    assert marker not in summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert marker not in summary["stop_reason"]


def test_resume_skips_finished_tasks_without_new_calls(tmp_path: Path) -> None:
    """On --resume, tasks recorded in tasks.jsonl are skipped with no new call."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    out = tmp_path / "out.jsonl"
    first = MarkerClient({TASK_ONE: ["ok-1", "ok-2"], TASK_TWO: ["bad", "bad"]})
    run(
        tuple(tasks),
        first,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    assert len(first.calls) == 4
    before = _read_jsonl(out)
    second = MarkerClient({TASK_ONE: ["ok-new"], TASK_TWO: ["ok-new"]})
    summary = run(
        tuple(tasks),
        second,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
        resume=True,
    )
    assert second.calls == []
    assert summary.tasks_run == 0
    assert summary.attempts == 0
    assert summary.stop_reason == "done"
    assert _read_jsonl(out) == before
    assert len(_read_jsonl(Path(str(out) + ".attempts.jsonl"))) == 4
    assert len(_read_jsonl(Path(str(out) + ".tasks.jsonl"))) == 2


def test_resume_and_skip_solved_union_counted_once(tmp_path: Path) -> None:
    """A task skipped by both --resume and --skip-solved is skipped once, with no call."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    out = tmp_path / "out.jsonl"
    first = MarkerClient({TASK_ONE: ["ok-1"], TASK_TWO: ["ok-2"]})
    run(
        tuple(tasks),
        first,
        model="m",
        attempts=1,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    skip_file = tmp_path / "skip.jsonl"
    skip_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": build_prompt(tasks[0])},
                    {"role": "assistant", "content": "ok-1"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second = MarkerClient({TASK_ONE: ["ok-new"], TASK_TWO: ["ok-new"]})
    summary = run(
        tuple(tasks),
        second,
        model="m",
        attempts=1,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
        resume=True,
        skip_solved=skip_file,
    )
    assert second.calls == []
    assert summary.tasks_run == 0
    assert summary.attempts == 0


def test_resume_retains_partial_attempts_and_tops_up(tmp_path: Path) -> None:
    """Attempts without a finished-task record are retained; only missing ones are sampled."""
    task = _task("DA0001", TASK_ONE)
    out = tmp_path / "out.jsonl"
    attempts_path = Path(str(out) + ".attempts.jsonl")
    attempts_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task.id,
                    "prompt_sha256": prompt_sha256(task),
                    "prompt": build_prompt(task),
                    "answer": f"ok-old-{i}",
                }
            )
            + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )
    client = MarkerClient({TASK_ONE: [f"ok-new-{i}" for i in range(5)]})
    summary = run(
        (task,),
        client,
        model="m",
        attempts=8,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
        resume=True,
    )
    assert len(client.calls) == 5
    assert summary.attempts == 5
    assert summary.tasks_run == 1
    assert summary.tasks_solved == 1
    assert len(_read_jsonl(attempts_path)) == 8
    task_lines = _read_jsonl(Path(str(out) + ".tasks.jsonl"))
    assert len(task_lines) == 1
    assert task_lines[0]["attempts"] == 8
    assert task_lines[0]["prompt_sha256"] == prompt_sha256(task)
    lines = _read_jsonl(out)
    assert len(lines) == 2
    assert lines[0]["messages"][2]["content"] == "ok-old-0"


def test_resume_appends_attempts_never_truncates(tmp_path: Path) -> None:
    """Resuming keeps earlier attempt lines and appends the new ones after them."""
    task = _task("DA0001", TASK_ONE)
    out = tmp_path / "out.jsonl"
    attempts_path = Path(str(out) + ".attempts.jsonl")
    first = MarkerClient({TASK_ONE: ["ok-1", "ok-2"]})
    run(
        (task,),
        first,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    assert len(_read_jsonl(attempts_path)) == 2
    second = MarkerClient({TASK_TWO: ["ok-3", "ok-4"]})
    other = _task("DA0002", TASK_TWO)
    summary = run(
        (task, other),
        second,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
        resume=True,
    )
    assert summary.tasks_run == 1
    stored = _read_jsonl(attempts_path)
    assert len(stored) == 4
    assert [line["answer"] for line in stored[:2]] == ["ok-1", "ok-2"]
    assert [line["answer"] for line in stored[2:]] == ["ok-3", "ok-4"]
    assert len(_read_jsonl(out)) == 4


def test_summary_json_uses_exact_fields(tmp_path: Path) -> None:
    """The summary file uses exactly the eight fields of the ticket."""
    task = _task("DA0001", TASK_ONE)
    client = UsageClient(["ok-1"], input_tokens=100, output_tokens=50, cost_usd=0.25)
    summary = run(
        (task,),
        client,
        model="m",
        attempts=1,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    assert dataclasses.is_dataclass(summary)
    payload = json.loads((tmp_path / "out.jsonl.summary.json").read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "tasks_solved",
        "tasks_run",
        "attempts",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wall_seconds",
        "stop_reason",
    }
    assert payload["tasks_solved"] == 1
    assert payload["tasks_run"] == 1
    assert payload["attempts"] == 1
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 50
    assert payload["cost_usd"] == pytest.approx(0.25)
    assert payload["wall_seconds"] >= 0.0
    assert payload["stop_reason"] == "done"


def test_summary_printed_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The round summary is printed as well as written."""
    task = _task("DA0001", TASK_ONE)
    run(
        (task,),
        MarkerClient({TASK_ONE: ["ok-1"]}),
        model="m",
        attempts=1,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    printed = capsys.readouterr().out
    assert "1 of 1 tasks solved" in printed


def test_skip_solved_matches_exact_user_prompt(tmp_path: Path) -> None:
    """skip_solved skips the task whose exact user prompt occurs in the file."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    skip_file = tmp_path / "skip.jsonl"
    skip_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": build_prompt(tasks[1])},
                    {"role": "assistant", "content": "ok-old"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    summary = run(
        tuple(tasks),
        client,
        model="m",
        attempts=1,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
        skip_solved=skip_file,
    )
    assert len(client.calls) == 1
    assert summary.tasks_run == 1
    assert summary.tasks_solved == 1
    lines = _read_jsonl(tmp_path / "out.jsonl")
    assert len(lines) == 1
    assert _user_of(lines[0]) == build_prompt(tasks[0])


def test_run_rejects_invalid_arguments(tmp_path: Path) -> None:
    """Invalid run arguments raise ValueError instead of running."""
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="attempts"):
        run((task,), client, model="m", attempts=0, out=out, reward_fn=fake_ok_reward)
    with pytest.raises(ValueError, match="concurrency"):
        run((task,), client, model="m", attempts=1, out=out, reward_fn=fake_ok_reward, concurrency=0)
    with pytest.raises(ValueError, match="workers"):
        run((task,), client, model="m", attempts=1, out=out, reward_fn=fake_ok_reward, workers=0)
    with pytest.raises(ValueError, match="max_per_task"):
        run((task,), client, model="m", attempts=1, out=out, reward_fn=fake_ok_reward, max_per_task=0)
    assert client.calls == []


def test_existing_out_without_resume_is_usage_error(tmp_path: Path) -> None:
    """An existing output file requires resume=True."""
    task = _task("DA0001", TASK_ONE)
    out = tmp_path / "out.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    with pytest.raises(ValueError, match="resume"):
        run((task,), client, model="m", attempts=1, out=out, reward_fn=fake_ok_reward)
    assert client.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--endpoint",
            "meta",
            "--model",
            "m",
            "--tasks",
            "train",
            "--attempts",
            "0",
            "--out",
            "o.jsonl",
            "--max-cost-usd",
            "1.0",
        ],
        [
            "--endpoint",
            "meta",
            "--model",
            "m",
            "--tasks",
            "train",
            "--attempts",
            "1",
            "--out",
            "o.jsonl",
            "--max-cost-usd",
            "1.0",
            "--concurrency",
            "0",
        ],
        [
            "--endpoint",
            "meta",
            "--model",
            "m",
            "--tasks",
            "train",
            "--attempts",
            "1",
            "--out",
            "o.jsonl",
            "--max-cost-usd",
            "1.0",
            "--workers",
            "0",
        ],
        [
            "--endpoint",
            "meta",
            "--model",
            "m",
            "--tasks",
            "train",
            "--attempts",
            "1",
            "--out",
            "o.jsonl",
            "--max-cost-usd",
            "1.0",
            "--max-per-task",
            "0",
        ],
        [
            "--endpoint",
            "meta",
            "--model",
            "m",
            "--tasks",
            "dev",
            "--attempts",
            "1",
            "--out",
            "o.jsonl",
            "--max-cost-usd",
            "1.0",
        ],
        ["--endpoint", "meta", "--model", "m", "--tasks", "train", "--attempts", "1", "--out", "o.jsonl"],
    ],
)
def test_cli_usage_errors_exit_two(argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each usage error of the ticket exits 2 without any model call."""
    _stub_split(monkeypatch, [_task("DA0001", TASK_ONE)])
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    monkeypatch.setattr(rejection, "build_client", lambda config, env: client)
    replaced = [value if value != "o.jsonl" else str(tmp_path / "o.jsonl") for value in argv]
    with pytest.raises(SystemExit) as exc:
        main(replaced)
    assert exc.value.code == 2
    assert client.calls == []


def test_cli_existing_out_without_resume_exit_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An existing --out without --resume is a usage error (exit 2)."""
    _stub_split(monkeypatch, [_task("DA0001", TASK_ONE)])
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    monkeypatch.setattr(rejection, "build_client", lambda config, env: client)
    out = tmp_path / "o.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--endpoint",
                "meta",
                "--model",
                "m",
                "--tasks",
                "train",
                "--attempts",
                "1",
                "--out",
                str(out),
                "--max-cost-usd",
                "1.0",
            ]
        )
    assert exc.value.code == 2
    assert client.calls == []


def test_help_lists_every_option(capsys: pytest.CaptureFixture[str]) -> None:
    """--help lists every option of the ticket."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in (
        "--endpoint",
        "--model",
        "--tasks",
        "--attempts",
        "--out",
        "--skip-solved",
        "--concurrency",
        "--workers",
        "--temperature",
        "--max-tokens",
        "--max-per-task",
        "--max-cost-usd",
        "--resume",
    ):
        assert option in text


def test_rejection_imports_no_training_frameworks() -> None:
    """The module imports no Fireworks, TRL, Modal or openai package."""
    source = Path(rejection.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert "fireworks" not in roots
    assert "trl" not in roots
    assert "modal" not in roots
    assert "openai" not in roots


def test_no_dynamic_vendor_import() -> None:
    """No dynamic import can smuggle a training framework in."""
    source = Path(rejection.__file__).read_text(encoding="utf-8")
    assert "__import__" not in source
    assert "importlib" not in source


def test_round_summary_is_frozen() -> None:
    """RoundSummary is a frozen dataclass with exactly the ticket fields."""
    fields = [field.name for field in dataclasses.fields(RoundSummary)]
    assert fields == [
        "tasks_solved",
        "tasks_run",
        "attempts",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wall_seconds",
        "stop_reason",
    ]
    summary = RoundSummary(
        tasks_solved=1,
        tasks_run=2,
        attempts=16,
        input_tokens=1600,
        output_tokens=800,
        cost_usd=0.5,
        wall_seconds=1.0,
        stop_reason="done",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.tasks_solved = 0  # type: ignore[misc]


def test_max_per_task_caps_verified_answers(tmp_path: Path) -> None:
    """At most max_per_task verified answers per task are written."""
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: [f"ok-{i}" for i in range(5)]})
    run(
        (task,),
        client,
        model="m",
        attempts=5,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=1,
        max_per_task=2,
    )
    lines = _read_jsonl(tmp_path / "out.jsonl")
    assert [line["messages"][2]["content"] for line in lines] == ["ok-0", "ok-1"]


def test_sampling_uses_concurrent_workers(tmp_path: Path) -> None:
    """Sampling runs attempts across tasks with the configured concurrency."""
    tasks = [_task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    client = MarkerClient({TASK_ONE: ["ok-1", "ok-2"], TASK_TWO: ["ok-3", "ok-4"]})
    summary = run(
        tuple(tasks),
        client,
        model="m",
        attempts=2,
        out=tmp_path / "out.jsonl",
        reward_fn=fake_ok_reward,
        concurrency=2,
        workers=1,
    )
    assert summary.tasks_solved == 2
    assert summary.attempts == 4


def test_default_reward_binds_config_resource_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without reward_fn, run binds the graded reward to the configured resource limit."""
    from pcc.config import Config

    probe = tmp_path / "limits.txt"
    monkeypatch.setenv("PCC_REJECTION_TEST_PROBE", str(probe))
    monkeypatch.setattr(rejection, "reward", fake_probe_reward)
    task = _task("DA0001", TASK_ONE)
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    run(
        (task,),
        client,
        model="m",
        attempts=1,
        out=tmp_path / "out.jsonl",
        concurrency=1,
        workers=1,
    )
    seen = [int(line) for line in probe.read_text(encoding="utf-8").splitlines()]
    assert seen != []
    assert all(limit == Config().dafny.resource_limit for limit in seen)


def test_out_rewritten_from_attempts_file(tmp_path: Path) -> None:
    """The output is rebuilt from the attempts file, not only from new attempts."""
    task = _task("DA0001", TASK_ONE)
    out = tmp_path / "out.jsonl"
    client = MarkerClient({TASK_ONE: ["ok-keep", "bad"]})
    run(
        (task,),
        client,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
    )
    stored = _read_jsonl(Path(str(out) + ".attempts.jsonl"))
    assert [line["answer"] for line in stored] == ["ok-keep", "bad"]
    lines = _read_jsonl(out)
    assert len(lines) == 1
    assert lines[0]["messages"][2]["content"] == "ok-keep"


def test_resume_keeps_cumulative_out_across_runs(tmp_path: Path) -> None:
    """The output file retains resumed cumulative data while the summary is per-run."""
    one = _task("DA0001", TASK_ONE)
    two = _task("DA0002", TASK_TWO)
    out = tmp_path / "out.jsonl"
    first = MarkerClient({TASK_ONE: ["ok-1"]})
    run((one,), first, model="m", attempts=1, out=out, reward_fn=fake_ok_reward, concurrency=1)
    second = MarkerClient({TASK_TWO: ["ok-2"]})
    summary = run(
        (one, two),
        second,
        model="m",
        attempts=1,
        out=out,
        reward_fn=fake_ok_reward,
        concurrency=1,
        resume=True,
    )
    assert summary.tasks_run == 1
    assert summary.attempts == 1
    lines = _read_jsonl(out)
    assert len(lines) == 2
    assert {_user_of(line) for line in lines} == {build_prompt(one), build_prompt(two)}


def test_cost_cap_counts_only_this_run(tmp_path: Path) -> None:
    """A resumed run counts only its own calls against the cap."""
    task = _task("DA0001", TASK_ONE)
    out = tmp_path / "out.jsonl"
    first = MarkerClient({TASK_ONE: ["yes", "yes"]}, cost_usd=0.10)
    first_summary = run(
        (task,),
        first,
        model="m",
        attempts=2,
        out=out,
        reward_fn=fake_solved_reward,
        concurrency=1,
        max_cost_usd=0.5,
    )
    assert first_summary.stop_reason == "done"
    assert first_summary.cost_usd == pytest.approx(0.20)
    other = _task("DA0002", TASK_TWO)
    second = MarkerClient({TASK_TWO: ["yes", "yes", "yes"]}, cost_usd=0.10)
    second_summary = run(
        (task, other),
        second,
        model="m",
        attempts=3,
        out=out,
        reward_fn=fake_solved_reward,
        concurrency=1,
        max_cost_usd=0.15,
        resume=True,
    )
    assert second_summary.stop_reason == "cost cap"
    assert second_summary.attempts == 2
    assert second_summary.cost_usd == pytest.approx(0.20)


def test_os_environ_not_leaked_into_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Endpoint error messages carry the router message without secret values."""
    tasks = [_task("DA0001", TASK_ONE)]
    _stub_split(monkeypatch, tasks)
    monkeypatch.delenv("PCC_META_ALLOWED", raising=False)
    monkeypatch.setenv("META_API_KEY", "meta-test-value")
    out = tmp_path / "out.jsonl"
    code = main(
        [
            "--endpoint",
            "meta",
            "--model",
            "muse-spark-1.3-contributor",
            "--tasks",
            "train",
            "--attempts",
            "1",
            "--out",
            str(out),
            "--max-cost-usd",
            "1.0",
            "--concurrency",
            "1",
        ]
    )
    assert code == 2
    text = (tmp_path / "out.jsonl.summary.json").read_text(encoding="utf-8")
    assert "meta-test-value" not in text
    assert os.environ["META_API_KEY"] == "meta-test-value"


def _unsupported_task(row_id: str, *, source: str = "apps") -> Task:
    """One Vericoding-style row whose prompt cannot be built.

    This mirrors the real train rows with an empty ``vc-spec`` (no ``prompt``
    stub either), so ``build_prompt`` raises ``ValueError`` for it.
    """
    payload: dict[str, Any] = {
        "id": row_id,
        "language": "dafny",
        "source": source,
        "source-id": f"{source}_{row_id}",
        "vc-description": f"Prove task {row_id}.",
        "vc-preamble": "",
        "vc-helpers": "",
        "vc-spec": "",
        "vc-code": "{ r := 0; }",
        "vc-postamble": "",
    }
    return Task(
        id=f"vericoding/{row_id}",
        source_task=f"vericoding/{source}/{source}_{row_id}",
        payload=payload,
    )


def _read_skipped(out: Path) -> dict[str, Any]:
    """The parsed ``<out>.skipped.json`` sibling report of a run."""
    payload: dict[str, Any] = json.loads(Path(str(out) + ".skipped.json").read_text(encoding="utf-8"))
    return payload


def test_unsupported_rows_skipped_with_report(tmp_path: Path) -> None:
    """Eligible tasks sample normally while empty-vc-spec rows are skipped and reported."""
    tasks = [_task("DA0001", TASK_ONE), _unsupported_task("DA0060"), _task("DA0002", TASK_TWO)]
    out = tmp_path / "out.jsonl"
    client = MarkerClient({TASK_ONE: ["ok-1"], TASK_TWO: ["ok-2"]})
    with pytest.warns(UserWarning, match="skipped 1"):
        summary = run(
            tuple(tasks),
            client,
            model="m",
            attempts=1,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
        )
    assert len(client.calls) == 2
    assert summary.tasks_run == 2
    assert summary.tasks_solved == 2
    assert summary.stop_reason == "done"
    skipped = _read_skipped(out)
    assert set(skipped.keys()) == {"total_tasks", "eligible_tasks", "skipped_count", "skipped_ids"}
    assert skipped["total_tasks"] == 3
    assert skipped["eligible_tasks"] == 2
    assert skipped["skipped_count"] == 1
    assert skipped["skipped_ids"] == ["vericoding/DA0060"]
    assert isinstance(skipped["total_tasks"], int)
    assert isinstance(skipped["eligible_tasks"], int)
    assert isinstance(skipped["skipped_count"], int)
    lines = _read_jsonl(out)
    assert len(lines) == 2
    assert {_user_of(line) for line in lines} == {build_prompt(tasks[0]), build_prompt(tasks[2])}
    task_lines = _read_jsonl(Path(str(out) + ".tasks.jsonl"))
    assert len(task_lines) == 2
    assert {line["prompt_sha256"] for line in task_lines} == {
        prompt_sha256(tasks[0]),
        prompt_sha256(tasks[2]),
    }
    assert len(_read_jsonl(Path(str(out) + ".attempts.jsonl"))) == 2
    summary_payload = json.loads((tmp_path / "out.jsonl.summary.json").read_text(encoding="utf-8"))
    assert set(summary_payload.keys()) == {
        "tasks_solved",
        "tasks_run",
        "attempts",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "wall_seconds",
        "stop_reason",
    }


def test_skipped_ids_sorted_and_exact_schema(tmp_path: Path) -> None:
    """The skip report holds exactly four keys with sorted task IDs."""
    tasks = [_unsupported_task("DA0099"), _task("DA0001", TASK_ONE), _unsupported_task("DA0060")]
    out = tmp_path / "out.jsonl"
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    with pytest.warns(UserWarning, match="skipped 2"):
        summary = run(
            tuple(tasks),
            client,
            model="m",
            attempts=1,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
        )
    assert summary.tasks_run == 1
    skipped = _read_skipped(out)
    assert set(skipped.keys()) == {"total_tasks", "eligible_tasks", "skipped_count", "skipped_ids"}
    assert skipped == {
        "total_tasks": 3,
        "eligible_tasks": 1,
        "skipped_count": 2,
        "skipped_ids": ["vericoding/DA0060", "vericoding/DA0099"],
    }


def test_all_rows_skipped_finishes_done(tmp_path: Path) -> None:
    """When every row is unsupported the run finishes done with no model call."""
    tasks = [_unsupported_task("DA0060"), _unsupported_task("DA0099")]
    out = tmp_path / "out.jsonl"
    client = MarkerClient({})
    with pytest.warns(UserWarning, match="skipped 2"):
        summary = run(
            tuple(tasks),
            client,
            model="m",
            attempts=2,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
        )
    assert client.calls == []
    assert summary.tasks_run == 0
    assert summary.tasks_solved == 0
    assert summary.attempts == 0
    assert summary.stop_reason == "done"
    assert out.read_text(encoding="utf-8") == ""
    assert _read_jsonl(Path(str(out) + ".tasks.jsonl")) == []
    skipped = _read_skipped(out)
    assert skipped == {
        "total_tasks": 2,
        "eligible_tasks": 0,
        "skipped_count": 2,
        "skipped_ids": ["vericoding/DA0060", "vericoding/DA0099"],
    }


def test_skip_redetected_on_resume(tmp_path: Path) -> None:
    """Skipped rows are re-detected on --resume and never recorded as finished."""
    tasks = [_task("DA0001", TASK_ONE), _unsupported_task("DA0060")]
    out = tmp_path / "out.jsonl"
    first = MarkerClient({TASK_ONE: ["ok-1"]})
    with pytest.warns(UserWarning, match="skipped 1"):
        run(
            tuple(tasks),
            first,
            model="m",
            attempts=1,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
        )
    assert len(_read_jsonl(Path(str(out) + ".tasks.jsonl"))) == 1
    second = MarkerClient({TASK_ONE: ["ok-new"]})
    with pytest.warns(UserWarning, match="skipped 1"):
        summary = run(
            tuple(tasks),
            second,
            model="m",
            attempts=1,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
            resume=True,
        )
    assert second.calls == []
    assert summary.tasks_run == 0
    assert summary.attempts == 0
    assert summary.stop_reason == "done"
    assert len(_read_jsonl(Path(str(out) + ".tasks.jsonl"))) == 1
    skipped = _read_skipped(out)
    assert skipped["skipped_ids"] == ["vericoding/DA0060"]
    assert skipped["total_tasks"] == 2
    assert skipped["eligible_tasks"] == 1


def test_skip_does_not_swallow_endpoint_error(tmp_path: Path) -> None:
    """An endpoint failure on eligible rows still stops the run despite skipped rows."""
    tasks = [_unsupported_task("DA0060"), _task("DA0001", TASK_ONE), _task("DA0002", TASK_TWO)]
    out = tmp_path / "out.jsonl"
    client = FlakyClient({TASK_ONE: ["ok-1", "ok-2"], TASK_TWO: ["ok-3", "ok-4"]}, fail_from=1)
    with pytest.warns(UserWarning, match="skipped 1"):
        summary = run(
            tuple(tasks),
            client,
            model="m",
            attempts=2,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
        )
    assert summary.stop_reason.startswith("endpoint: ")
    skipped = _read_skipped(out)
    assert skipped["skipped_ids"] == ["vericoding/DA0060"]
    assert skipped["total_tasks"] == 3
    assert skipped["eligible_tasks"] == 2


def test_skip_does_not_swallow_corrupt_resume(tmp_path: Path) -> None:
    """A corrupt resume file still raises ToolError when skipped rows are present."""
    tasks = [_unsupported_task("DA0060"), _task("DA0001", TASK_ONE)]
    out = tmp_path / "out.jsonl"
    tasks_path = Path(str(out) + ".tasks.jsonl")
    tasks_path.write_text("not json\n", encoding="utf-8")
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    with pytest.raises(ToolError, match="cannot resume from tasks"):
        run(
            tuple(tasks),
            client,
            model="m",
            attempts=1,
            out=out,
            reward_fn=fake_ok_reward,
            concurrency=1,
            resume=True,
        )
    assert client.calls == []


def test_reward_value_error_propagates_despite_skipped_rows(tmp_path: Path) -> None:
    """A ValueError from the reward is not mistaken for an unbuildable prompt."""
    tasks = [_unsupported_task("DA0060"), _task("DA0001", TASK_ONE)]
    out = tmp_path / "out.jsonl"
    client = MarkerClient({TASK_ONE: ["ok-1"]})
    with pytest.warns(UserWarning, match="skipped 1"), pytest.raises(ValueError, match="boom: broken scorer"):
        run(
            tuple(tasks),
            client,
            model="m",
            attempts=1,
            out=out,
            reward_fn=fake_value_reward,
            concurrency=1,
        )


def test_reward_signature_matches_grpo() -> None:
    """The fake rewards take (answer, task_spec) like the graded reward."""
    callback: Callable[[str, str], float] = fake_ok_reward
    assert callback("ok-x", "spec") == 1.0
    assert callback("bad", "spec") == 0.0
