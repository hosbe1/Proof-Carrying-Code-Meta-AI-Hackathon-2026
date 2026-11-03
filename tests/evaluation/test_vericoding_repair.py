"""Tests for --repair: Dafny-feedback repair turns in the Vericoding harness (issue #180).

Each failed attempt may send up to ``repair`` follow-up turns in the same conversation: the
previous answer as the assistant turn, then a user turn with the failure feedback and the
instruction to return the full corrected program. Sampling is scripted (``ScriptedClient``,
``FakeClient``); the fake ``verify_fn`` returns the ``(verified, seconds, feedback)`` triple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pcc.core import ToolError
from pcc.evaluation import __main__ as eval_main
from pcc.evaluation import vericoding
from pcc.evaluation.prompts import SYSTEM_MESSAGE, build_prompt
from pcc.evaluation.proof_only import build_proof_only_prompt
from pcc.models import FakeClient
from pcc.models.client import Message
from tests.evaluation.test_vericoding import (
    GOOD_MODEL,
    ScriptedClient,
    add_task,
    cheat_answer,
    frozen_of,
    good_answer,
    read_jsonl,
)

FAIL_FEEDBACK = "dafny: line 3: postcondition could not be proved"

SUM_PROGRAM = (
    "method Sum(n: int) returns (r: int)\n"
    "  requires n >= 0\n"
    "  ensures r >= 0\n"
    "{\n"
    "  r := 0;\n"
    "  var i := 0;\n"
    "  while i < n\n"
    "  {\n"
    "    r := r + i;\n"
    "    i := i + 1;\n"
    "  }\n"
    "}\n"
)

SUM_FIXED = (
    "method Sum(n: int) returns (r: int)\n"
    "  requires n >= 0\n"
    "  ensures r >= 0\n"
    "{\n"
    "  r := 0;\n"
    "  var i := 0;\n"
    "  while i < n\n"
    "    invariant 0 <= i <= n\n"
    "    invariant r >= 0\n"
    "  {\n"
    "    r := r + i;\n"
    "    i := i + 1;\n"
    "  }\n"
    "}\n"
)

SUM_BROKEN = SUM_PROGRAM.replace("r := r + i;", "r := r + 1;")


def _write_tasks_file(path: Path) -> Path:
    path.write_text(
        json.dumps({"id": "proof-only/sum", "source_task": "vericoding/DD0001", "program": SUM_PROGRAM}) + "\n",
        encoding="utf-8",
    )
    return path


def _repair_user_text(messages: tuple[Message, ...]) -> str:
    return messages[-1].content


def test_repair_zero_sends_no_repair_turns(tmp_path: Path) -> None:
    """--repair 0 (the default) keeps today's behaviour: no follow-up model call."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    good = good_answer(task)
    seen: list[str] = []

    def verify_second(answer: str, prefix: str) -> tuple[bool, float, str]:
        seen.append(answer)
        return (answer == good, 0.5, "" if answer == good else FAIL_FEEDBACK)

    out = tmp_path / "runs" / "repair-zero"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=2,
        max_cost_usd=10.0,
        tasks=(task,),
        client=FakeClient([bad, good]),
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
    assert line["repair_turns_used"] == 0
    assert line["repair_turns_total"] == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["repair"] == 0


def test_repair_zero_is_default(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "repair-default"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        tasks=(add_task(),),
        client=FakeClient([good_answer(add_task())]),
        verify_fn=lambda answer, prefix: (True, 0.5, ""),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["repair"] == 0
    assert read_jsonl(out / "results.jsonl")[0]["repair_turns_total"] == 0


def test_verified_on_second_turn(tmp_path: Path) -> None:
    """A failed first turn is repaired in the same conversation with the fake feedback."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    good = good_answer(task)
    calls: list[tuple[str, str]] = []

    def verify(answer: str, prefix: str) -> tuple[bool, float, str]:
        calls.append((answer, prefix))
        if answer == good:
            return True, 0.7, ""
        return False, 0.5, FAIL_FEEDBACK

    client = ScriptedClient({"method add(": [(bad, 0.25, 1.0), (good, 0.3, 2.0)]})
    out = tmp_path / "runs" / "repair-one"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=2,
        tasks=(task,),
        client=client,
        verify_fn=verify,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert calls == [(bad, frozen), (good, frozen)]
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is True
    assert line["answer"] == good
    assert line["repair_turns_used"] == 1
    assert line["repair_turns_total"] == 1
    assert line["cost_usd"] == pytest.approx(0.55)
    assert line["seconds"] == pytest.approx(1.0 + 2.0 + 0.5 + 0.7)
    assert len(client.calls) == 2
    first, second = client.calls
    assert first.messages == (Message("system", SYSTEM_MESSAGE), Message("user", build_prompt(task)))
    assert second.messages[:3] == (
        Message("system", SYSTEM_MESSAGE),
        Message("user", build_prompt(task)),
        Message("assistant", bad),
    )
    repair_text = _repair_user_text(second.messages)
    assert repair_text.startswith(FAIL_FEEDBACK)
    assert "Return the full corrected program." in repair_text


def test_never_verified_within_n(tmp_path: Path) -> None:
    """Every turn fails: each chain sends exactly N repair turns, then keeps the last answer."""
    task = add_task()
    frozen = frozen_of(task)
    texts = [f"{frozen}  r := {n};\n}}\n" for n in range(3)]
    client = ScriptedClient({"method add(": [(text, 0.2, 1.0) for text in texts]})
    out = tmp_path / "runs" / "repair-never"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=2,
        tasks=(task,),
        client=client,
        verify_fn=lambda answer, prefix: (False, 0.5, FAIL_FEEDBACK),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert len(client.calls) == 3
    for call in client.calls[1:]:
        assert FAIL_FEEDBACK in _repair_user_text(call.messages)
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is False
    assert line["verified_attempts"] == 0
    assert line["answer"] == texts[-1]
    assert line["repair_turns_used"] is None
    assert line["repair_turns_total"] == 2


def test_cheat_on_repair_turn_consumes_a_turn(tmp_path: Path) -> None:
    """A cheat on a repair turn fails the turn without reaching the verifier and uses one turn."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    cheat = cheat_answer(task)
    calls: list[str] = []

    def spy(answer: str, prefix: str) -> tuple[bool, float, str]:
        calls.append(answer)
        return False, 0.5, FAIL_FEEDBACK

    client = ScriptedClient({"method add(": [(bad, 0.2, 1.0), (cheat, 0.2, 1.0)]})
    out = tmp_path / "runs" / "repair-cheat"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=1,
        tasks=(task,),
        client=client,
        verify_fn=spy,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert calls == [bad]
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is False
    assert line["repair_turns_used"] is None
    assert line["repair_turns_total"] == 1


def test_anti_cheat_feedback_names_the_violation(tmp_path: Path) -> None:
    """An anti-cheat repair answer is fed back as 'anti-cheat: <form>; ...' on the next turn."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    assume = f"{good_answer(task)}  assume false;\n"
    third = f"{frozen}  r := 1;\n}}\n"
    client = ScriptedClient({"method add(": [(bad, 0.2, 1.0), (assume, 0.2, 1.0), (third, 0.2, 1.0)]})
    out = tmp_path / "runs" / "repair-anticheat"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=2,
        tasks=(task,),
        client=client,
        verify_fn=lambda answer, prefix: (False, 0.5, FAIL_FEEDBACK),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert len(client.calls) == 3
    repair_text = _repair_user_text(client.calls[2].messages)
    assert "anti-cheat: assume; remove it and prove without it" in repair_text
    assert "Return the full corrected program." in repair_text
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["repair_turns_total"] == 2
    assert line["repair_turns_used"] is None


def test_header_cheat_feedback_names_the_header(tmp_path: Path) -> None:
    """A header cheat on a repair turn is fed back naming the changed method."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    header_cheat = (
        good_answer(task)
        + "\nmethod add(a: int, b: int) returns (r: int)\n"
        + "  ensures r == Add(a, b)\n"
        + "  requires false\n"
        + "{\n  r := 0;\n}\n"
    )
    assert frozen in header_cheat
    client = ScriptedClient({"method add(": [(bad, 0.2, 1.0), (header_cheat, 0.2, 1.0), (bad, 0.2, 1.0)]})
    out = tmp_path / "runs" / "repair-header"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=2,
        tasks=(task,),
        client=client,
        verify_fn=lambda answer, prefix: (False, 0.5, FAIL_FEEDBACK),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    repair_text = _repair_user_text(client.calls[2].messages)
    assert "header changed: add" in repair_text


def test_proof_only_repair_verified_on_turn_two(tmp_path: Path) -> None:
    """Proof-only tasks repair with the program prompt and verify on the second turn."""
    tasks_file = _write_tasks_file(tmp_path / "tasks.jsonl")
    seen: list[str] = []

    def verify(answer: str, program: str) -> tuple[bool, float, str]:
        seen.append(answer)
        assert program == SUM_PROGRAM
        if answer == SUM_FIXED:
            return True, 0.4, ""
        return False, 0.5, FAIL_FEEDBACK

    client = ScriptedClient({"Sum(": [(SUM_PROGRAM, 0.2, 1.0), (SUM_FIXED, 0.3, 2.0)]})
    out = tmp_path / "runs" / "repair-proof-only"
    vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=2,
        tasks=None,
        client=client,
        verify_fn=verify,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert seen == [SUM_PROGRAM, SUM_FIXED]
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is True
    assert line["answer"] == SUM_FIXED
    assert line["repair_turns_used"] == 1
    assert line["repair_turns_total"] == 1
    first, second = client.calls
    assert first.messages[1].content == build_proof_only_prompt(SUM_PROGRAM)
    assert second.messages[2] == Message("assistant", SUM_PROGRAM)
    assert _repair_user_text(second.messages).startswith(FAIL_FEEDBACK)


def test_proof_only_code_change_feedback(tmp_path: Path) -> None:
    """A proof-only repair answer that changes executable code is fed back verbatim."""
    tasks_file = _write_tasks_file(tmp_path / "tasks.jsonl")
    client = ScriptedClient({"Sum(": [(SUM_PROGRAM, 0.2, 1.0), (SUM_BROKEN, 0.2, 1.0), (SUM_PROGRAM, 0.2, 1.0)]})
    out = tmp_path / "runs" / "repair-proof-only-cheat"
    vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=2,
        tasks=None,
        client=client,
        verify_fn=lambda answer, program: (False, 0.5, FAIL_FEEDBACK),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    repair_text = _repair_user_text(client.calls[2].messages)
    assert repair_text == (
        "executable code changed: keep every line of executable code and every method "
        "header exactly as given and add only proof annotations\n\n"
        "Return the full corrected program."
    )
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is False
    assert line["repair_turns_used"] is None
    assert line["repair_turns_total"] == 2


def test_chains_report_min_used_and_total(tmp_path: Path) -> None:
    """All chains run: verified_attempts counts verified chains, used is the fewest turns."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    good = good_answer(task)

    def verify(answer: str, prefix: str) -> tuple[bool, float, str]:
        if answer == good:
            return True, 0.5, ""
        return False, 0.5, FAIL_FEEDBACK

    client = ScriptedClient({"method add(": [(good, 0.2, 1.0), (bad, 0.2, 1.0), (good, 0.2, 1.0)]})
    out = tmp_path / "runs" / "repair-chains"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=2,
        max_cost_usd=10.0,
        repair=2,
        tasks=(task,),
        client=client,
        verify_fn=verify,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is True
    assert line["attempts"] == 2
    assert line["verified_attempts"] == 2
    assert line["answer"] == good
    assert line["repair_turns_used"] == 0
    assert line["repair_turns_total"] == 1


def test_unverified_chains_report_null_used(tmp_path: Path) -> None:
    task = add_task()
    frozen = frozen_of(task)
    texts = [f"{frozen}  r := {n};\n}}\n" for n in range(4)]
    client = ScriptedClient({"method add(": [(text, 0.2, 1.0) for text in texts]})
    out = tmp_path / "runs" / "repair-chains-null"
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=2,
        max_cost_usd=10.0,
        repair=1,
        tasks=(task,),
        client=client,
        verify_fn=lambda answer, prefix: (False, 0.5, FAIL_FEEDBACK),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is False
    assert line["verified_attempts"] == 0
    assert line["answer"] == texts[-1]
    assert line["repair_turns_used"] is None
    assert line["repair_turns_total"] == 2


def test_repair_turns_stop_at_cost_cap(tmp_path: Path) -> None:
    """A repair turn is not sent once the completed cost reaches --max-cost-usd."""
    task = add_task()
    frozen = frozen_of(task)
    bad = f"{frozen}  r := 0;\n}}\n"
    client = ScriptedClient({"method add(": [(bad, 0.6, 1.0)]})
    out = tmp_path / "runs" / "repair-capped"
    summary = vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=0.6,
        repair=2,
        tasks=(task,),
        client=client,
        verify_fn=lambda answer, prefix: (False, 0.5, FAIL_FEEDBACK),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert len(client.calls) == 1
    assert summary["stopped_early"] is True
    assert "stopped at cost cap" in (out / "summary.md").read_text(encoding="utf-8")
    line = read_jsonl(out / "results.jsonl")[0]
    assert line["verified"] is False
    assert line["repair_turns_used"] is None
    assert line["repair_turns_total"] == 0


def test_resume_refuses_different_repair(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "repair-resume"
    task = add_task()
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=1,
        tasks=(task,),
        client=FakeClient([good_answer(task)]),
        verify_fn=lambda answer, prefix: (True, 0.5, ""),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    with pytest.raises(ToolError, match="repair"):
        vericoding.run_harness(
            split="test",
            out_dir=out,
            model=GOOD_MODEL,
            attempts=1,
            max_cost_usd=10.0,
            repair=2,
            tasks=(task,),
            client=FakeClient([good_answer(task)]),
            verify_fn=lambda answer, prefix: (True, 0.5, ""),
            git_sha="0" * 40,
            started_utc="2026-10-31T00:00:00Z",
        )
    second = FakeClient([good_answer(task)])
    vericoding.run_harness(
        split="test",
        out_dir=out,
        model=GOOD_MODEL,
        attempts=1,
        max_cost_usd=10.0,
        repair=1,
        tasks=(task,),
        client=second,
        verify_fn=lambda answer, prefix: (True, 0.5, ""),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert second.requests == []


def test_repair_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="repair"):
        vericoding.run_harness(
            split="test",
            out_dir=Path("runs/never"),
            model=GOOD_MODEL,
            attempts=1,
            max_cost_usd=10.0,
            repair=-1,
            tasks=(add_task(),),
            client=FakeClient([]),
        )


def test_parser_repair_flag() -> None:
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
    assert args.repair == 0
    repaired = eval_main.build_parser().parse_args(
        [
            "vericoding",
            "--split",
            "test",
            "--model",
            GOOD_MODEL,
            "--attempts",
            "2",
            "--repair",
            "3",
            "--max-cost-usd",
            "1",
            "--out",
            "runs/a",
        ]
    )
    assert repaired.repair == 3


def test_main_passes_repair_to_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(eval_main, "run_vericoding", fake_run)
    assert (
        eval_main.main(
            [
                "vericoding",
                "--split",
                "test",
                "--model",
                GOOD_MODEL,
                "--attempts",
                "2",
                "--repair",
                "2",
                "--max-cost-usd",
                "1",
                "--out",
                "runs/a",
            ]
        )
        == 0
    )
    assert seen["repair"] == 2
