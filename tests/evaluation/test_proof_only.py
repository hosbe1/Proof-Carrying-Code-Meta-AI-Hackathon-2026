"""Tests for proof-only tasks (issue #83).

Verified answers become tasks for the proof step alone: ``strip_ghost`` removes
the annotations, ``make`` writes harness or SFT lines, and the Vericoding
harness runs a tasks file in place of a split. Sampling is scripted
(``FakeClient``); Dafny runs only in the ``dafny``-marked examples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pcc.core import ToolError
from pcc.evaluation import __main__ as eval_main
from pcc.evaluation import proof_only, vericoding
from pcc.evaluation.prompts import SYSTEM_MESSAGE
from pcc.models import FakeClient
from pcc.verifier.anti_cheat import _normalize
from pcc.verifier.ghost import declared_lemma_names, executable_part, strip_ghost

ANSWER = (
    "method Sum(n: int) returns (r: int)\n"
    "  requires n >= 0\n"
    "  ensures r >= 0\n"
    "{\n"
    "  r := 0;\n"
    "  var i := 0;\n"
    "  while i < n\n"
    "    invariant 0 <= i <= n\n"
    "    invariant 0 <= r\n"
    "    decreases n - i\n"
    "  {\n"
    "    r := r + i;\n"
    "    LemmaAddNonneg(r, i);\n"
    "    i := i + 1;\n"
    "  }\n"
    "}\n"
    "\n"
    "lemma LemmaAddNonneg(a: int, b: int)\n"
    "  requires a >= 0 && b >= 0\n"
    "  ensures a + b >= 0\n"
    "{\n"
    "}\n"
)

PLAIN = "method Double(x: int) returns (r: int)\n  ensures r == 2 * x\n{\n  r := x + x;\n}\n"

SPEC_LEMMA_ANSWER = (
    "method M(x: int) returns (r: int)\n"
    "  ensures LemmaProp(x, r)\n"
    "{\n"
    "  LemmaProp(x, 0);\n"
    "  r := 0;\n"
    "}\n"
    "\n"
    "lemma LemmaProp(x: int, r: int)\n"
    "  ensures r >= 0\n"
    "{\n"
    "}\n"
)


def _results_line(task_id: str, answer: str, *, verified: bool = True, **extra: Any) -> str:
    return json.dumps({"task_id": task_id, "verified": verified, "answer": answer, **extra})


def _sft_line(answer: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": "prove it"},
                {"role": "assistant", "content": answer},
            ]
        }
    )


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _program_of(task_line: dict[str, Any]) -> str:
    program = task_line["program"]
    assert isinstance(program, str)
    return program


def test_prompt_asks_for_annotations_and_keeps_code() -> None:
    program = "method M() {\n  var x := 0;\n}\n"
    prompt = proof_only.build_proof_only_prompt(program)
    lowered = prompt.lower()
    assert "invariant" in lowered
    assert "decreases" in lowered
    assert "assert" in lowered
    assert "lemma" in lowered
    assert "unchanged" in lowered
    assert program in prompt


def test_make_tasks_from_verified_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(tmp_path / "results.jsonl", [_results_line("vericoding/DH0000", ANSWER)])
    out = tmp_path / "tasks.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 0
    )
    lines = _read_lines(out)
    assert len(lines) == 1
    assert lines[0]["id"] == "proof-only/vericoding/DH0000"
    assert lines[0]["source_task"] == "vericoding/DH0000"
    program = _program_of(lines[0])
    assert "invariant" not in program
    assert "LemmaAddNonneg" not in program
    assert "requires n >= 0" in program
    assert "r := r + i;" in program
    names = declared_lemma_names(ANSWER)
    assert executable_part(program, lemma_names=names) == executable_part(ANSWER, lemma_names=names)
    printed = capsys.readouterr().out
    assert "skipped 0" in printed


def test_make_source_task_falls_back_to_task_id(tmp_path: Path) -> None:
    source = _write(tmp_path / "results.jsonl", [_results_line("abc/1", ANSWER)])
    out = tmp_path / "tasks.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 0
    )
    assert _read_lines(out)[0]["source_task"] == "abc/1"


def test_make_skips_answer_without_ghost(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[str] = []

    def reject(program: str) -> bool:
        seen.append(program)
        return False

    source = _write(tmp_path / "results.jsonl", [_results_line("abc/1", PLAIN)])
    out = tmp_path / "tasks.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=reject,
        )
        == 0
    )
    assert _read_lines(out) == []
    assert seen == []
    assert "without ghost code" in capsys.readouterr().out


def test_make_skips_already_verifying_program(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(tmp_path / "results.jsonl", [_results_line("abc/1", ANSWER)])
    out = tmp_path / "tasks.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: True,
        )
        == 0
    )
    assert _read_lines(out) == []
    assert "already verifying" in capsys.readouterr().out


def test_make_ignores_unverified_and_empty_answers(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "results.jsonl",
        [
            _results_line("abc/1", ANSWER, verified=False),
            _results_line("abc/2", "   \n  ", verified=True),
            _results_line("abc/3", ANSWER, verified=True),
        ],
    )
    out = tmp_path / "tasks.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 0
    )
    assert [line["id"] for line in _read_lines(out)] == ["proof-only/abc/3"]


def test_make_sft_format_from_results(tmp_path: Path) -> None:
    source = _write(tmp_path / "results.jsonl", [_results_line("abc/1", ANSWER)])
    out = tmp_path / "sft.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "sft", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 0
    )
    lines = _read_lines(out)
    assert len(lines) == 1
    messages = lines[0]["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_MESSAGE}
    assert messages[1]["role"] == "user"
    assert messages[2] == {"role": "assistant", "content": ANSWER}
    program = _program_of({"program": messages[1]["content"].split("\n\n", 1)[1]})
    names = declared_lemma_names(ANSWER)
    assert executable_part(program, lemma_names=names) == executable_part(ANSWER, lemma_names=names)


def test_make_tasks_from_sft_input(tmp_path: Path) -> None:
    source = _write(tmp_path / "sft.jsonl", [_sft_line(ANSWER), _sft_line(PLAIN)])
    out = tmp_path / "tasks.jsonl"
    seen: list[str] = []

    def reject(program: str) -> bool:
        seen.append(program)
        return False

    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=reject,
        )
        == 0
    )
    lines = _read_lines(out)
    assert len(lines) == 1
    digest = hashlib.sha256(ANSWER.encode("utf-8")).hexdigest()
    assert lines[0]["id"] == f"proof-only/sft/{digest}"
    assert lines[0]["source_task"] == f"sft/{digest}"
    assert len(seen) == 1
    assert _normalize(_program_of(lines[0])) != _normalize(ANSWER)


def test_make_sft_from_sft_input_example(tmp_path: Path) -> None:
    source = _write(tmp_path / "sft.jsonl", [_sft_line(ANSWER), _sft_line(ANSWER + "\n")])
    out = tmp_path / "sft.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "sft", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 0
    )
    lines = _read_lines(out)
    assert len(lines) == 2
    for line, answer in zip(lines, [ANSWER, ANSWER + "\n"], strict=True):
        assert line["messages"][0] == {"role": "system", "content": SYSTEM_MESSAGE}
        assert line["messages"][2] == {"role": "assistant", "content": answer}
        assert "invariant" in line["messages"][1]["content"].lower()


def test_make_warns_on_lemma_in_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(tmp_path / "results.jsonl", [_results_line("abc/1", SPEC_LEMMA_ANSWER)])
    out = tmp_path / "tasks.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 0
    )
    lines = _read_lines(out)
    assert len(lines) == 1
    assert "lemma LemmaProp" in _program_of(lines[0])
    captured = capsys.readouterr()
    assert "line 1" in captured.err
    assert "LemmaProp" in captured.err


@pytest.mark.parametrize(
    "line",
    [
        "{not json",
        "[1, 2]",
        json.dumps({"unexpected": "shape"}),
        json.dumps({"verified": True, "answer": ANSWER}),
        json.dumps({"task_id": "abc/1", "verified": True}),
        json.dumps({"messages": []}),
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        json.dumps({"messages": [{"role": "assistant", "content": "   "}]}),
    ],
)
def test_make_malformed_line_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str], line: str) -> None:
    source = _write(tmp_path / "in.jsonl", [line])
    out = tmp_path / "out.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 2
    )
    assert capsys.readouterr().err.startswith("error: line 1: malformed proof-only input:")


def test_make_error_names_later_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(
        tmp_path / "in.jsonl",
        [_results_line("abc/1", PLAIN), _results_line("abc/2", PLAIN), "{broken"],
    )
    out = tmp_path / "out.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 2
    )
    assert capsys.readouterr().err.startswith("error: line 3: malformed proof-only input:")


def test_make_duplicate_id_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(
        tmp_path / "in.jsonl",
        [_results_line("abc/1", ANSWER), _results_line("abc/1", ANSWER)],
    )
    out = tmp_path / "out.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 2
    )
    assert capsys.readouterr().err.startswith("error: line 2: malformed proof-only input:")


def test_make_duplicate_sft_answer_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(tmp_path / "in.jsonl", [_sft_line(ANSWER), _sft_line(ANSWER)])
    out = tmp_path / "out.jsonl"
    assert (
        proof_only.main(
            ["make", "--from", str(source), "--format", "tasks", "--out", str(out)],
            verify_fn=lambda program: False,
        )
        == 2
    )
    assert capsys.readouterr().err.startswith("error: line 2: malformed proof-only input:")


def test_parser_make_requires_format(tmp_path: Path) -> None:
    args = proof_only.build_parser().parse_args(
        ["make", "--from", str(tmp_path / "in.jsonl"), "--format", "tasks", "--out", str(tmp_path / "o.jsonl")]
    )
    assert args.format == "tasks"


@pytest.mark.dafny
def test_make_skips_program_that_already_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(tmp_path / "results.jsonl", [_results_line("abc/1", PLAIN)])
    out = tmp_path / "tasks.jsonl"
    assert proof_only.main(["make", "--from", str(source), "--format", "tasks", "--out", str(out)]) == 0
    assert _read_lines(out) == []
    assert "already verifying" in capsys.readouterr().out


def _tasks_file(tmp_path: Path, program: str) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps({"id": "proof-only/abc/1", "source_task": "abc/1", "program": program}), encoding="utf-8"
    )
    return path


def test_tasks_file_sends_proof_only_prompt(tmp_path: Path) -> None:
    program = strip_ghost(ANSWER, lemma_names=["LemmaAddNonneg"])
    assert _normalize(program) != _normalize(ANSWER)
    tasks_file = _tasks_file(tmp_path, program)
    fake = FakeClient([ANSWER])
    out = tmp_path / "runs" / "file-prompt"
    vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        tasks=None,
        client=fake,
        verify_fn=lambda answer, context: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert len(fake.requests) == 1
    _model, messages, _step = fake.requests[0]
    assert messages[0].content == SYSTEM_MESSAGE
    assert messages[1].content == proof_only.build_proof_only_prompt(program)


def test_tasks_file_run_json_holds_file_split(tmp_path: Path) -> None:
    program = strip_ghost(ANSWER, lemma_names=["LemmaAddNonneg"])
    tasks_file = _tasks_file(tmp_path, program)
    out = tmp_path / "runs" / "file-meta"
    summary = vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=FakeClient([ANSWER]),
        verify_fn=lambda answer, context: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["split"] == str(tasks_file.resolve())
    assert run["split_sha256"] == hashlib.sha256(tasks_file.read_bytes()).hexdigest()
    assert summary["metrics"]["verified"] == 1


def test_tasks_file_body_change_is_not_verified(tmp_path: Path) -> None:
    program = "method M() returns (r: int)\n{\n  var x := 0;\n  if x <= 2 {\n    x := x + 1;\n  }\n}\n"
    answer = program.replace("x <= 2", "x <= 1")
    calls: list[str] = []

    def spy(text: str, context: str) -> tuple[bool, float]:
        calls.append(text)
        return True, 0.5

    out = tmp_path / "runs" / "file-body"
    vericoding.run_harness(
        tasks_file=_tasks_file(tmp_path, program),
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=FakeClient([answer]),
        verify_fn=spy,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert calls == []
    assert _read_lines(out / "results.jsonl")[0]["verified"] is False


def test_tasks_file_header_change_is_not_verified(tmp_path: Path) -> None:
    program = "method M(x: int) {\n  var y := x;\n}\n"
    answer = "method M(x: int) requires x > 0 {\n  var y := x;\n}\n"
    calls: list[str] = []

    def spy(text: str, context: str) -> tuple[bool, float]:
        calls.append(text)
        return True, 0.5

    out = tmp_path / "runs" / "file-header"
    vericoding.run_harness(
        tasks_file=_tasks_file(tmp_path, program),
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=FakeClient([answer]),
        verify_fn=spy,
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert calls == []
    assert _read_lines(out / "results.jsonl")[0]["verified"] is False


def test_tasks_file_resume_skips_completed(tmp_path: Path) -> None:
    program = strip_ghost(ANSWER, lemma_names=["LemmaAddNonneg"])
    tasks_file = _tasks_file(tmp_path, program)
    out = tmp_path / "runs" / "file-resume"
    first = FakeClient([ANSWER])
    vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=first,
        verify_fn=lambda answer, context: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    before = (out / "results.jsonl").read_bytes()
    second = FakeClient([ANSWER])
    vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=second,
        verify_fn=lambda answer, context: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert second.requests == []
    assert (out / "results.jsonl").read_bytes() == before


def test_tasks_file_changed_content_or_path_is_an_error(tmp_path: Path) -> None:
    program = strip_ghost(ANSWER, lemma_names=["LemmaAddNonneg"])
    tasks_file = _tasks_file(tmp_path, program)
    out = tmp_path / "runs" / "file-changed"
    vericoding.run_harness(
        tasks_file=tasks_file,
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=FakeClient([ANSWER]),
        verify_fn=lambda answer, context: (True, 0.5),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    other = tmp_path / "other.jsonl"
    other.write_text(tasks_file.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ToolError):
        vericoding.run_harness(
            tasks_file=other,
            out_dir=out,
            model="muse-spark-1.3-contributor",
            attempts=1,
            max_cost_usd=10.0,
            client=FakeClient([ANSWER]),
            verify_fn=lambda answer, context: (True, 0.5),
            git_sha="0" * 40,
            started_utc="2026-10-31T00:00:00Z",
        )
    with tasks_file.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ToolError):
        vericoding.run_harness(
            tasks_file=tasks_file,
            out_dir=out,
            model="muse-spark-1.3-contributor",
            attempts=1,
            max_cost_usd=10.0,
            client=FakeClient([ANSWER]),
            verify_fn=lambda answer, context: (True, 0.5),
            git_sha="0" * 40,
            started_utc="2026-10-31T00:00:00Z",
        )


def test_tasks_file_and_split_are_exclusive(tmp_path: Path) -> None:
    tasks_file = _write(tmp_path / "t.jsonl", [json.dumps({"id": "a", "source_task": "b", "program": PLAIN})])
    with pytest.raises(ValueError, match="mutually exclusive"):
        vericoding.run_harness(
            split="test",
            tasks_file=tasks_file,
            out_dir=tmp_path / "runs" / "exclusive",
            model="muse-spark-1.3-contributor",
            attempts=1,
            max_cost_usd=10.0,
            client=FakeClient([PLAIN]),
        )
    with pytest.raises(SystemExit):
        eval_main.build_parser().parse_args(
            [
                "vericoding",
                "--split",
                "test",
                "--tasks-file",
                str(tasks_file),
                "--model",
                "muse-spark-1.3-contributor",
                "--attempts",
                "1",
                "--max-cost-usd",
                "1",
                "--out",
                "runs/a",
            ]
        )


def test_tasks_file_bad_line_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "a", "source_task": "b"}) + "\n", encoding="utf-8")
    with pytest.raises(ToolError):
        vericoding.run_harness(
            tasks_file=bad,
            out_dir=tmp_path / "runs" / "bad",
            model="muse-spark-1.3-contributor",
            attempts=1,
            max_cost_usd=10.0,
            client=FakeClient([PLAIN]),
        )


@pytest.mark.dafny
def test_tasks_file_invariant_back_verifies(tmp_path: Path) -> None:
    program = strip_ghost(ANSWER, lemma_names=["LemmaAddNonneg"])
    assert _normalize(program) != _normalize(ANSWER)
    out = tmp_path / "runs" / "file-dafny"
    summary = vericoding.run_harness(
        tasks_file=_tasks_file(tmp_path, program),
        out_dir=out,
        model="muse-spark-1.3-contributor",
        attempts=1,
        max_cost_usd=10.0,
        client=FakeClient([ANSWER]),
        git_sha="0" * 40,
        started_utc="2026-10-31T00:00:00Z",
    )
    assert summary["metrics"]["verified"] == 1
