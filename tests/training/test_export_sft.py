"""Tests for the SFT data export (issue #49).

The four examples from the ticket are ``test_overlapping_verified_task_is_deduplicated``,
``test_unverified_answer_is_not_exported``, ``test_test_split_task_id_raises`` and
``test_include_line_with_test_prompt_raises``. Everything runs offline on hand-written tasks;
the dataset cache is only touched through monkeypatched fixtures in the ``main`` tests.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from training.export_sft import (
    PROOFS_FILENAME,
    SAMPLES_FILENAME,
    TaskIndex,
    build_prompt,
    collect_lines,
    main,
)
from training.prompts import SYSTEM_MESSAGE

from pcc.evaluation import datasets
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
    source: str = "apps",
    source_id: str = "",
) -> Task:
    """One hand-written Vericoding-style task; ``vc-code`` is the reference answer."""
    payload: dict[str, Any] = {
        "id": row_id,
        "language": "dafny",
        "source": source,
        "source-id": source_id or f"{source}_{row_id}",
        "vc-description": description,
        "vc-preamble": preamble,
        "vc-helpers": helpers,
        "vc-spec": spec,
        "vc-code": code,
        "vc-postamble": postamble,
    }
    return Task(
        id=f"vericoding/{row_id}",
        source_task=f"vericoding/{source}/{source_id or source}_{row_id}",
        payload=payload,
    )


TRAIN_A = _vericoding_task("DA0002", spec="method solve(n: int) returns (result: int)")
TRAIN_B = _vericoding_task("DA0004", description="Wave over spectators.")
TEST_TASK = _vericoding_task("DA0003", description="Fill a barn over days.")


def _index(tasks: Sequence[Task], test_ids: Sequence[str]) -> TaskIndex:
    """A task index over hand-written tasks with the given IDs on the test split."""
    by_id = {task.id: task for task in tasks}
    return TaskIndex(
        tasks=by_id,
        test_ids=frozenset(test_ids),
        test_prompts=frozenset(build_prompt(by_id[task_id]) for task_id in test_ids),
    )


INDEX = _index((TRAIN_A, TRAIN_B, TEST_TASK), (TEST_TASK.id,))

ANSWER_A = "method solve(n: int) returns (result: int)\n{\n  result := n;\n}\n"


def _write_lines(path: Path, raw_lines: Sequence[str]) -> Path:
    """Write raw JSONL lines to ``path``, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(line + "\n" for line in raw_lines)
    path.write_text(text, encoding="utf-8")
    return path


def _sample(task_id: str, answer: str, *, verified: bool = True) -> str:
    return json.dumps({"task_id": task_id, "answer": answer, "verified": verified})


def _proof_line(task_id: str, user: str, answer: str) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user},
            ],
            "answer": answer,
        }
    )


def _chat_line(user: str, answer: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ]
        }
    )


def test_chat_line_has_system_user_assistant_format(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [_sample(TRAIN_A.id, ANSWER_A)])
    lines = collect_lines((run,), (), index=INDEX)
    assert lines == [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": build_prompt(TRAIN_A)},
                {"role": "assistant", "content": ANSWER_A},
            ]
        }
    ]


def test_overlapping_verified_task_is_deduplicated(tmp_path: Path) -> None:
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    _write_lines(first / SAMPLES_FILENAME, [_sample(TRAIN_A.id, ANSWER_A)])
    _write_lines(second / SAMPLES_FILENAME, [_sample(TRAIN_A.id, ANSWER_A)])
    lines = collect_lines((first, second), (), index=INDEX)
    assert len(lines) == 1


def test_same_task_with_different_answers_keeps_both(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    other = ANSWER_A.replace("result := n;", "result := n + 0;")
    _write_lines(run / SAMPLES_FILENAME, [_sample(TRAIN_A.id, ANSWER_A), _sample(TRAIN_A.id, other)])
    lines = collect_lines((run,), (), index=INDEX)
    assert [line["messages"][2]["content"] for line in lines] == [ANSWER_A, other]


def test_unverified_answer_is_not_exported(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(
        run / SAMPLES_FILENAME,
        [_sample(TRAIN_A.id, ANSWER_A, verified=False), _sample(TRAIN_B.id, ANSWER_A)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lines = collect_lines((run,), (), index=INDEX)
    assert len(lines) == 1
    assert lines[0]["messages"][1]["content"] == build_prompt(TRAIN_B)


def test_missing_verified_flag_is_not_exported(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [json.dumps({"task_id": TRAIN_A.id, "answer": ANSWER_A})])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lines = collect_lines((run,), (), index=INDEX)
    assert lines == []


def test_test_split_task_id_raises(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [_sample(TEST_TASK.id, ANSWER_A)])
    with pytest.raises(ValueError, match="test split"):
        collect_lines((run,), (), index=INDEX)


def test_test_split_task_id_raises_for_unverified_line(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [_sample(TEST_TASK.id, ANSWER_A, verified=False)])
    with pytest.raises(ValueError, match="test split"):
        collect_lines((run,), (), index=INDEX)


def test_test_split_task_id_raises_for_proofs_line(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / PROOFS_FILENAME, [_proof_line(TEST_TASK.id, "do it", ANSWER_A)])
    with pytest.raises(ValueError, match="test split"):
        collect_lines((run,), (), index=INDEX)


def test_unknown_task_id_raises(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [_sample("vericoding/NOPE", ANSWER_A)])
    with pytest.raises(ValueError, match="unknown task"):
        collect_lines((run,), (), index=INDEX)


def test_proofs_jsonl_lines_are_exported_with_answer_as_assistant(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    user = "prove this method with ghost code only"
    _write_lines(run / PROOFS_FILENAME, [_proof_line(TRAIN_A.id, user, ANSWER_A)])
    lines = collect_lines((run,), (), index=INDEX)
    assert lines == [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ANSWER_A},
            ]
        }
    ]


def test_samples_and_proofs_and_include_are_deduplicated_together(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    prompt = build_prompt(TRAIN_A)
    _write_lines(run / SAMPLES_FILENAME, [_sample(TRAIN_A.id, ANSWER_A)])
    _write_lines(run / PROOFS_FILENAME, [_proof_line(TRAIN_A.id, prompt, ANSWER_A)])
    include = tmp_path / "round0.jsonl"
    _write_lines(include, [_chat_line(prompt, ANSWER_A)])
    lines = collect_lines((run,), (include,), index=INDEX)
    assert len(lines) == 1


def test_include_line_with_test_prompt_raises(tmp_path: Path) -> None:
    include = tmp_path / "round0.jsonl"
    _write_lines(include, [_chat_line(build_prompt(TEST_TASK), ANSWER_A)])
    with pytest.raises(ValueError, match="test split"):
        collect_lines((), (include,), index=INDEX)


def test_include_line_without_user_message_raises(tmp_path: Path) -> None:
    include = tmp_path / "round0.jsonl"
    _write_lines(
        include,
        [
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "assistant", "content": ANSWER_A},
                    ]
                }
            )
        ],
    )
    with pytest.raises(ValueError, match="user"):
        collect_lines((), (include,), index=INDEX)


def test_include_line_not_ending_in_assistant_raises(tmp_path: Path) -> None:
    include = tmp_path / "round0.jsonl"
    _write_lines(
        include,
        [
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": build_prompt(TRAIN_A)},
                    ]
                }
            )
        ],
    )
    with pytest.raises(ValueError, match="assistant"):
        collect_lines((), (include,), index=INDEX)


def test_malformed_sample_line_raises(tmp_path: Path) -> None:
    for raw in (
        "[1, 2]",
        json.dumps({"answer": ANSWER_A}),
        json.dumps({"task_id": TRAIN_A.id}),
        json.dumps({"task_id": TRAIN_A.id, "answer": ANSWER_A, "verified": "yes"}),
        json.dumps({"task_id": TRAIN_A.id, "answer": "  ", "verified": True}),
    ):
        run = tmp_path / "run-a"
        _write_lines(run / SAMPLES_FILENAME, [raw])
        with pytest.raises(ValueError, match=r"samples\.jsonl"):
            collect_lines((run,), (), index=INDEX)


def test_broken_json_sample_line_raises(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, ["{not json"])
    with pytest.raises(ValueError, match=r"samples\.jsonl"):
        collect_lines((run,), (), index=INDEX)


def test_malformed_proofs_line_raises(tmp_path: Path) -> None:
    for raw in (
        json.dumps({"task_id": TRAIN_A.id, "answer": ANSWER_A}),
        json.dumps({"task_id": TRAIN_A.id, "messages": [], "answer": ANSWER_A}),
        json.dumps(
            {
                "task_id": TRAIN_A.id,
                "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                "answer": ANSWER_A,
            }
        ),
    ):
        run = tmp_path / "run-a"
        _write_lines(run / PROOFS_FILENAME, [raw])
        with pytest.raises(ValueError, match=r"proofs\.jsonl"):
            collect_lines((run,), (), index=INDEX)


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no-such-run"):
        collect_lines((tmp_path / "no-such-run",), (), index=INDEX)


def test_run_without_samples_or_proofs_warns(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    run.mkdir()
    with pytest.warns(UserWarning, match="no training lines"):
        lines = collect_lines((run,), (), index=INDEX)
    assert lines == []


def test_empty_run_warns(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [])
    with pytest.warns(UserWarning, match="no training lines"):
        lines = collect_lines((run,), (), index=INDEX)
    assert lines == []


def test_build_prompt_holds_spec_but_not_reference_answer() -> None:
    prompt = build_prompt(TRAIN_A)
    assert "method solve(n: int) returns (result: int)" in prompt
    assert "{ result := a + b; }" not in prompt


def test_build_prompt_skips_empty_blocks() -> None:
    prompt = build_prompt(TRAIN_B)
    assert "\n\n\n" not in prompt
    assert prompt.strip() != ""


def test_build_prompt_rejects_unknown_payload_shape() -> None:
    task = Task(id="x/Y", source_task="x", payload={"id": "Y"})
    with pytest.raises(ValueError, match="prompt"):
        build_prompt(task)


def _fixture_cache_and_splits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, dict[str, Task]]:
    """Serve a tiny two-dataset cache plus split files through a monkeypatched download."""
    vericoding_rows = [
        {
            "id": "DA0002",
            "language": "dafny",
            "source": "apps",
            "source-id": "apps_DA0002",
            "vc-description": "Solve it.",
            "vc-preamble": "",
            "vc-helpers": "",
            "vc-spec": "method solve(n: int) returns (result: int)",
            "vc-code": "{ result := n; }",
            "vc-postamble": "",
        },
        {
            "id": "DA0003",
            "language": "dafny",
            "source": "apps",
            "source-id": "apps_DA0003",
            "vc-description": "Held back.",
            "vc-preamble": "",
            "vc-helpers": "",
            "vc-spec": "method held(n: int) returns (result: int)",
            "vc-code": "{ result := n; }",
            "vc-postamble": "",
        },
    ]
    vericoding_bytes = "".join(json.dumps(row) + "\n" for row in vericoding_rows).encode("utf-8")
    humanevalfix_rows = [
        {
            "task_id": "Python/0",
            "prompt": "def f(x):\n    return x\n",
            "canonical_solution": "def f(x):\n    return x\n",
            "buggy_solution": "def f(x):\n    return 0\n",
            "bug_type": "logic",
            "test": "assert f(1) == 1",
        }
    ]
    humanevalfix_bytes = json.dumps(humanevalfix_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    specs = {
        "humanevalfix": datasets._DatasetSpec(
            name="humanevalfix",
            url="https://example.invalid/humanevalfix-rows",
            revision="fixture",
            sha256=hashlib.sha256(humanevalfix_bytes).hexdigest(),
            cache_filename="humanevalfix.json",
            kind="rows",
        ),
        "vericoding": datasets._DatasetSpec(
            name="vericoding",
            url="https://example.invalid/vericoding-train.jsonl",
            revision="fixture",
            sha256=hashlib.sha256(vericoding_bytes).hexdigest(),
            cache_filename="vericoding-train.jsonl",
            kind="jsonl",
        ),
    }

    def fake_http_get(url: str) -> bytes:
        if url == specs["vericoding"].url:
            return vericoding_bytes
        if url == specs["humanevalfix"].url or url.startswith(specs["humanevalfix"].url + "?"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            offset = int(query.get("offset", ["0"])[0])
            length = int(query.get("length", ["100"])[0])
            return json.dumps({"rows": [{"row": row} for row in humanevalfix_rows[offset : offset + length]]}).encode()
        raise AssertionError(f"unexpected download URL {url!r}")

    monkeypatch.setattr(datasets, "_DATASETS", specs)
    monkeypatch.setattr(datasets, "_http_get", fake_http_get)
    cache_dir = tmp_path / "cache"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "vericoding_test.txt").write_text("vericoding/DA0003\n", encoding="utf-8")
    (splits_dir / "humanevalfix_test.txt").write_text("humanevalfix/Python/0\n", encoding="utf-8")
    tasks = {task.id: task for task in datasets.split("vericoding", "train", cache_dir, splits_dir)}
    return cache_dir, splits_dir, tasks


def test_main_writes_deduplicated_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_dir, splits_dir, tasks = _fixture_cache_and_splits(monkeypatch, tmp_path)
    assert set(tasks) == {"vericoding/DA0002"}
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    answer = "method solve(n: int) returns (result: int)\n{\n  result := n;\n}\n"
    _write_lines(first / SAMPLES_FILENAME, [_sample("vericoding/DA0002", answer)])
    _write_lines(
        second / SAMPLES_FILENAME,
        [_sample("vericoding/DA0002", answer), _sample("vericoding/DA0002", answer, verified=False)],
    )
    out = tmp_path / "nested" / "sft.jsonl"
    assert main([str(first), str(second), "--out", str(out)], cache_dir=cache_dir, splits_dir=splits_dir) == 0
    stored = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stored == [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": build_prompt(tasks["vericoding/DA0002"])},
                {"role": "assistant", "content": answer},
            ]
        }
    ]


def test_main_rejects_test_task_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_dir, splits_dir, _tasks = _fixture_cache_and_splits(monkeypatch, tmp_path)
    run = tmp_path / "run-a"
    answer = "method held(n: int) returns (result: int)\n{\n  result := n;\n}\n"
    _write_lines(run / SAMPLES_FILENAME, [_sample("vericoding/DA0003", answer)])
    with pytest.raises(ValueError, match="test split"):
        main([str(run), "--out", str(tmp_path / "sft.jsonl")], cache_dir=cache_dir, splits_dir=splits_dir)


def test_main_empty_run_writes_empty_file_with_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_dir, splits_dir, _tasks = _fixture_cache_and_splits(monkeypatch, tmp_path)
    run = tmp_path / "run-a"
    _write_lines(run / SAMPLES_FILENAME, [])
    out = tmp_path / "sft.jsonl"
    with pytest.warns(UserWarning, match="no training lines"):
        assert main([str(run), "--out", str(out)], cache_dir=cache_dir, splits_dir=splits_dir) == 0
    assert out.read_text(encoding="utf-8") == ""


def test_main_without_args_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([], cache_dir=tmp_path, splits_dir=tmp_path)
    assert excinfo.value.code == 2


def test_index_tasks_are_frozen() -> None:
    assert isinstance(INDEX.tasks, dict)
    assert INDEX.test_ids == frozenset({TEST_TASK.id})
    assert INDEX.test_prompts == frozenset({build_prompt(TEST_TASK)})
