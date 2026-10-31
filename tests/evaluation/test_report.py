"""Tests for the evaluation report (issue #82): ``wilson``, ``publish`` and ``compare``.

All runs are hand-written fixtures in ``tmp_path``: no models, no network, no Dafny. The fixtures follow the
formats the harnesses write (``run.json`` plus ``results.jsonl`` for Vericoding, ``scan.jsonl`` for the
HumanEvalFix pipeline and ``spec_only.jsonl`` for the spec writer run).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pcc.evaluation.report import UsageError, compare, main, publish, wilson


def write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def vericoding_run(
    runs: Path,
    name: str,
    lines: list[dict[str, Any]],
    *,
    date: str = "2026-10-20",
    commit: str = "abc1234def5678",
    model: str = "muse-spark-1.3-contributor",
    contributor: bool = True,
    stopped_early: bool = False,
) -> Path:
    """A hand-written Vericoding run: ``run.json`` plus ``results.jsonl``."""
    run_dir = runs / name
    run_dir.mkdir(parents=True)
    task_ids = [line["task_id"] for line in lines]
    write_json(
        run_dir / "run.json",
        {
            "harness": "vericoding",
            "split": "test",
            "date": date,
            "commit": commit,
            "model": model,
            "dataset": "vericoding/test",
            "contributor": contributor,
            "max_cost_usd": 10.0,
            "stopped_early": stopped_early,
            "status": "done",
            "tasks": task_ids,
        },
    )
    write_jsonl(run_dir / "results.jsonl", lines)
    return run_dir


def vericoding_line(
    task_id: str, attempts: int, verified_attempts: int, cost: float = 0.4, seconds: float = 3.0
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "attempts": attempts,
        "verified_attempts": verified_attempts,
        "cost_usd": cost,
        "seconds": seconds,
    }


def scan_line(
    task_id: str,
    version: str,
    state: str,
    *,
    name: str = "add",
    stage: str | None = None,
    path: str | None = None,
    cost: float = 0.5,
    seconds: float = 2.0,
) -> dict[str, Any]:
    """A hand-written ``scan.jsonl`` line in the shape ``humanevalfix.scan_line`` writes."""
    result: dict[str, Any] = {}
    if stage is not None:
        result["proof"] = {"proven": True, "stage": stage}
    if path is not None:
        result["translation"] = {"function_path": path}
    return {
        "project": f"humanevalfix-{version}",
        "task_id": task_id,
        "version": version,
        "state": state,
        "reason": "",
        "name": name,
        "path": "solution.py",
        "cost_usd": cost,
        "seconds": seconds,
        "result": result,
    }


def spec_line(task_id: str, version: str, outcome: str, cost: float = 0.2, seconds: float = 1.0) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "version": version,
        "outcome": outcome,
        "reason": "",
        "cost_usd": cost,
        "seconds": seconds,
    }


def pipeline_run(runs: Path, name: str, lines: list[dict[str, Any]], **kwargs: Any) -> Path:
    """A hand-written HumanEvalFix pipeline run: ``run.json`` plus ``scan.jsonl``."""
    run_dir = runs / name
    run_dir.mkdir(parents=True)
    payload: dict[str, Any] = {
        "harness": "humanevalfix",
        "split": "test",
        "date": "2026-10-21",
        "commit": "def5678abc1234",
        "model": "muse-spark-1.3-contributor",
        "dataset": "humanevalfix/test",
        "contributor": True,
        "max_cost_usd": 10.0,
        "stages": None,
        "spec_model": None,
        "spec_only": False,
        "stopped_early": False,
        "status": "done",
    }
    payload.update(kwargs)
    write_json(run_dir / "run.json", payload)
    write_jsonl(run_dir / "scan.jsonl", lines)
    return run_dir


def spec_run(runs: Path, name: str, lines: list[dict[str, Any]], **kwargs: Any) -> Path:
    """A hand-written spec writer run: ``run.json`` plus ``spec_only.jsonl``."""
    run_dir = pipeline_run(runs, name, [], **kwargs)
    (run_dir / "scan.jsonl").unlink()
    payload = read_json(run_dir / "run.json")
    payload["spec_only"] = True
    write_json(run_dir / "run.json", payload)
    write_jsonl(run_dir / "spec_only.jsonl", lines)
    return run_dir


def test_wilson_known_value() -> None:
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.490, abs=0.001)
    assert high == pytest.approx(0.943, abs=0.001)


def test_wilson_empty_run() -> None:
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_bounds_cover_and_shrink() -> None:
    low, high = wilson(2, 3)
    assert 0.0 <= low <= 2 / 3 <= high <= 1.0
    narrow_low, narrow_high = wilson(200, 300)
    assert narrow_low > low
    assert narrow_high < high


def test_vericoding_example_metrics(tmp_path: Path) -> None:
    """Three tasks with 4 attempts and 4, 1 and 0 verified: pass@1 is 5/12, pass@k is 2/3."""
    runs = tmp_path / "runs"
    run_dir = vericoding_run(
        runs,
        "first",
        [
            vericoding_line("vericoding/t1", 4, 4),
            vericoding_line("vericoding/t2", 4, 1),
            vericoding_line("vericoding/t3", 4, 0),
        ],
    )
    dest = publish(run_dir, tmp_path / "docs" / "results")
    summary = read_json(dest / "summary.json")
    assert summary["kind"] == "vericoding"
    assert summary["n"] == 3
    assert summary["metrics"]["pass_at_1"] == pytest.approx(5 / 12)
    assert summary["metrics"]["pass_at_k"] == pytest.approx(2 / 3)
    assert summary["rate"] == pytest.approx(2 / 3)
    expected_low, expected_high = wilson(2, 3)
    assert summary["rate_lo"] == pytest.approx(expected_low)
    assert summary["rate_hi"] == pytest.approx(expected_high)
    assert summary["metrics"]["interval"] == pytest.approx([expected_low, expected_high])


def test_publish_twice_appends_history(tmp_path: Path) -> None:
    """Two publishes into an empty results dir: two folders, a header and two rows."""
    runs = tmp_path / "runs"
    results = tmp_path / "docs" / "results"
    first = vericoding_run(runs, "first", [vericoding_line("vericoding/t1", 4, 4)])
    second = vericoding_run(
        runs, "second", [vericoding_line("vericoding/t1", 4, 0)], date="2026-10-21", commit="def5678abc1234"
    )
    dest_first = publish(first, results)
    dest_second = publish(second, results)
    assert dest_first != dest_second
    assert dest_first.is_dir()
    assert dest_second.is_dir()
    assert (dest_first / "summary.md").is_file()
    assert (dest_first / "summary.json").is_file()
    assert read_json(dest_first / "meta.json") == read_json(first / "run.json")
    history = (results / "history.md").read_text(encoding="utf-8").splitlines()
    assert history[0] == "| date | commit | model | dataset | n | rate | cost |"
    assert len(history) == 3
    assert history[1].startswith("| 2026-10-20 | abc1234 |")
    assert history[2].startswith("| 2026-10-21 | def5678 |")


def test_publish_does_not_duplicate_history_header(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "docs" / "results"
    publish(vericoding_run(runs, "first", [vericoding_line("vericoding/t1", 4, 1)]), results)
    publish(vericoding_run(runs, "second", [vericoding_line("vericoding/t1", 4, 1)]), results)
    history = (results / "history.md").read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in history if line.startswith("| date |")) == 1
    assert len(history) == 3


def test_history_rate_is_pass_at_k(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "docs" / "results"
    publish(
        vericoding_run(
            runs,
            "first",
            [
                vericoding_line("vericoding/t1", 4, 4),
                vericoding_line("vericoding/t2", 4, 1),
                vericoding_line("vericoding/t3", 4, 0),
            ],
        ),
        results,
    )
    row = (results / "history.md").read_text(encoding="utf-8").splitlines()[1]
    assert "| 0.667 |" in row


def test_sha_and_date_come_from_run_json(tmp_path: Path) -> None:
    """The folder name uses run.json values, never the current checkout or the clock."""
    runs = tmp_path / "runs"
    run_dir = vericoding_run(
        runs, "custom", [vericoding_line("vericoding/t9", 2, 1)], date="2020-01-02", commit="0123456789abcdef"
    )
    dest = publish(run_dir, tmp_path / "results")
    assert dest.name == "2020-01-02-0123456-custom"
    summary = read_json(dest / "summary.json")
    assert summary["date"] == "2020-01-02"
    assert summary["commit"] == "0123456789abcdef"


def test_publish_without_run_json_is_usage_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "norun"
    run_dir.mkdir(parents=True)
    with pytest.raises(UsageError, match=r"run\.json"):
        publish(run_dir, tmp_path / "results")
    assert main(["publish", str(run_dir), "--results-dir", str(tmp_path / "results")]) == 2


def test_publish_without_date_or_commit_is_usage_error(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "nodate"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run.json", {"harness": "humanevalfix", "spec_only": False})
    write_jsonl(run_dir / "scan.jsonl", [])
    with pytest.raises(UsageError, match="date"):
        publish(run_dir, tmp_path / "results")


def test_publish_existing_folder_is_not_overwritten(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    run_dir = vericoding_run(runs, "first", [vericoding_line("vericoding/t1", 4, 1)])
    dest = publish(run_dir, results)
    sentinel = dest / "summary.md"
    assert sentinel.is_file()
    with pytest.raises(UsageError, match="already exists"):
        publish(run_dir, results)
    assert sentinel.is_file()
    history = (results / "history.md").read_text(encoding="utf-8").splitlines()
    assert len(history) == 2


def test_publish_empty_vericoding_run(tmp_path: Path) -> None:
    run_dir = vericoding_run(tmp_path / "runs", "empty", [])
    dest = publish(run_dir, tmp_path / "results")
    summary = read_json(dest / "summary.json")
    assert summary["n"] == 0
    assert summary["rate"] == 0.0
    assert (summary["rate_lo"], summary["rate_hi"]) == (0.0, 0.0)


def test_pipeline_metrics(tmp_path: Path) -> None:
    """Proof rate on fixed versions, bugs found, false alarms, false proofs and stage shares."""
    runs = tmp_path / "runs"
    run_dir = pipeline_run(
        runs,
        "pipe",
        [
            scan_line("humanevalfix/Python/0", "fixed", "proven", name="add", stage="glimmer", path="transpiler"),
            scan_line("humanevalfix/Python/0", "buggy", "counterexample", name="add"),
            scan_line("humanevalfix/Python/1", "fixed", "proven", name="neg", stage="spark", path="ai"),
            scan_line("humanevalfix/Python/1", "buggy", "proven", name="neg", stage="spark", path="ai"),
            scan_line("humanevalfix/Python/2", "fixed", "open", name="mid"),
            scan_line("humanevalfix/Python/2", "buggy", "not_checkable", name="mid"),
        ],
    )
    dest = publish(run_dir, tmp_path / "results")
    summary = read_json(dest / "summary.json")
    assert summary["kind"] == "pipeline"
    assert summary["n"] == 3
    metrics = summary["metrics"]
    assert metrics["proof_rate_fixed"] == pytest.approx(2 / 3)
    assert metrics["n_fixed"] == 3
    assert metrics["n_buggy"] == 3
    assert metrics["bugs_found"] == 1
    assert metrics["false_alarms"] == 0
    assert metrics["false_proofs"] == ["humanevalfix/Python/1"]
    assert summary["rate"] == pytest.approx(2 / 3)
    expected_low, expected_high = wilson(2, 3)
    assert (summary["rate_lo"], summary["rate_hi"]) == pytest.approx((expected_low, expected_high))
    assert metrics["state_counts"] == {"proven": 3, "counterexample": 1, "open": 1, "not_checkable": 1}
    assert metrics["state_shares"]["proven"] == pytest.approx(0.5)
    assert metrics["proof_stages"]["glimmer"] == {"count": 1, "share": pytest.approx(1 / 3)}
    assert metrics["proof_stages"]["spark"] == {"count": 2, "share": pytest.approx(2 / 3)}
    assert metrics["transpiler_share"] == pytest.approx(1 / 3)
    assert metrics["mean_cost_usd_per_function"] == pytest.approx(0.5)
    assert metrics["mean_seconds_per_function"] == pytest.approx(2.0)
    assert metrics["stopped_at_cost_cap"] is False


def test_pipeline_cost_cap_recorded(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = pipeline_run(
        runs,
        "capped",
        [scan_line("humanevalfix/Python/0", "fixed", "open")],
        stopped_early=True,
    )
    summary = read_json(publish(run_dir, tmp_path / "results") / "summary.json")
    assert summary["metrics"]["stopped_at_cost_cap"] is True
    body = (publish(run_dir, tmp_path / "results2") / "summary.md").read_text(encoding="utf-8")
    assert "Cost cap reached: yes" in body


def test_spec_writer_metrics(tmp_path: Path) -> None:
    """Correct passes, caught bugs, false alarms and not_checkable, each with its interval."""
    runs = tmp_path / "runs"
    run_dir = spec_run(
        runs,
        "spec",
        [
            spec_line("humanevalfix/Python/0", "fixed", "pass"),
            spec_line("humanevalfix/Python/0", "buggy", "caught"),
            spec_line("humanevalfix/Python/1", "fixed", "false_alarm"),
            spec_line("humanevalfix/Python/1", "buggy", "missed"),
            spec_line("humanevalfix/Python/2", "fixed", "not_checkable"),
            spec_line("humanevalfix/Python/2", "buggy", "not_checkable"),
        ],
    )
    dest = publish(run_dir, tmp_path / "results")
    summary = read_json(dest / "summary.json")
    assert summary["kind"] == "spec"
    metrics = summary["metrics"]
    assert metrics["correct_passes"] == 1
    assert metrics["n_fixed"] == 3
    assert metrics["buggy_caught"] == 1
    assert metrics["n_buggy"] == 3
    assert metrics["false_alarms"] == 1
    assert metrics["missed"] == 1
    assert metrics["not_checkable"] == 2
    assert metrics["correct_passes_interval"] == pytest.approx(list(wilson(1, 3)))
    assert metrics["buggy_caught_interval"] == pytest.approx(list(wilson(1, 3)))
    assert metrics["false_alarms_interval"] == pytest.approx(list(wilson(1, 3)))
    assert metrics["not_checkable_interval"] == pytest.approx(list(wilson(2, 6)))
    assert summary["rate"] == pytest.approx(2 / 6)
    assert summary["rate_name"] == "correct_rate"


def test_summary_md_names_contributor_mode(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    contributor = publish(
        vericoding_run(runs, "with-mode", [vericoding_line("vericoding/t1", 4, 1)], contributor=True),
        tmp_path / "results",
    )
    assert "contributor" in (contributor / "summary.md").read_text(encoding="utf-8")
    standard = publish(
        vericoding_run(
            runs,
            "std-mode",
            [vericoding_line("vericoding/t1", 4, 1)],
            model="muse-spark-1.3",
            contributor=False,
        ),
        tmp_path / "results2",
    )
    body = (standard / "summary.md").read_text(encoding="utf-8")
    assert "standard" in body
    assert "contributor mode" in body.lower()


def test_compare_lists_tasks_solved_by_one_side_only(tmp_path: Path) -> None:
    """Task A is solved only by the first run, task B only by the second: both are listed."""
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    first = publish(
        vericoding_run(
            runs,
            "first",
            [
                vericoding_line("vericoding/A", 4, 2),
                vericoding_line("vericoding/B", 4, 0),
            ],
        ),
        results,
    )
    second = publish(
        vericoding_run(
            runs,
            "second",
            [
                vericoding_line("vericoding/A", 4, 0),
                vericoding_line("vericoding/B", 4, 3),
            ],
        ),
        results,
    )
    table = compare([first, second])
    assert "vericoding/A" in table
    assert "vericoding/B" in table
    assert "pass@k" in table
    assert "contributor" in table
    assert "cost" in table.lower()


def test_compare_different_task_sets_exits_2(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    first = publish(vericoding_run(runs, "first", [vericoding_line("vericoding/A", 4, 1)]), results)
    second = publish(
        vericoding_run(
            runs,
            "second",
            [vericoding_line("vericoding/A", 4, 1), vericoding_line("vericoding/B", 4, 1)],
        ),
        results,
    )
    with pytest.raises(UsageError, match="vericoding/B"):
        compare([first, second])
    assert main(["compare", str(first), str(second)]) == 2


def test_compare_table_has_model_mode_n_rate_cost(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    lines_first = [vericoding_line("vericoding/A", 4, 1), vericoding_line("vericoding/B", 4, 0)]
    lines_second = [vericoding_line("vericoding/A", 4, 4), vericoding_line("vericoding/B", 4, 4)]
    first = publish(vericoding_run(runs, "first", lines_first), results)
    second = publish(vericoding_run(runs, "second", lines_second), results)
    table = compare([first, second])
    assert "| run | model | mode | dataset | n | rate | cost (USD) |" in table
    assert "muse-spark-1.3-contributor" in table
    assert "0.500" in table
    assert "1.000" in table


def test_compare_out_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    first = publish(vericoding_run(runs, "first", [vericoding_line("vericoding/A", 4, 1)]), results)
    second = publish(vericoding_run(runs, "second", [vericoding_line("vericoding/A", 4, 1)]), results)
    out = tmp_path / "comparison.md"
    assert main(["compare", str(first), str(second), "--out", str(out)]) == 0
    assert out.is_file()
    assert "vericoding/A" not in out.read_text(encoding="utf-8")
    capsys.readouterr()


def test_compare_identical_runs_have_no_differences(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    lines = [vericoding_line("vericoding/A", 4, 1)]
    first = publish(vericoding_run(runs, "first", lines), results)
    second = publish(vericoding_run(runs, "second", [vericoding_line("vericoding/A", 4, 4)]), results)
    assert "same tasks" in compare([first, second])


def test_publish_main_returns_zero(tmp_path: Path) -> None:
    run_dir = vericoding_run(tmp_path / "runs", "cli", [vericoding_line("vericoding/A", 4, 4)])
    assert main(["publish", str(run_dir), "--results-dir", str(tmp_path / "results")]) == 0


def test_compare_missing_summary_is_usage_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UsageError, match=r"summary\.json"):
        compare([empty])
    assert main(["compare", str(empty), str(empty)]) == 2


def test_publish_vericoding_states_skipped_ids_and_denominator(tmp_path: Path) -> None:
    """A run with skipped rows publishes the excluded IDs and the eligible denominator (issue #154)."""
    runs = tmp_path / "runs"
    run_dir = vericoding_run(runs, "skipped", [vericoding_line("vericoding/t1", 1, 1)])
    payload = read_json(run_dir / "run.json")
    payload["tasks"] = ["vericoding/t1"]
    payload["total_tasks"] = 3
    payload["eligible_tasks"] = 1
    payload["skipped_count"] = 2
    payload["skipped_ids"] = ["vericoding/DA0152", "vericoding/DA0818"]
    write_json(run_dir / "run.json", payload)
    dest = publish(run_dir, tmp_path / "results")
    body = (dest / "summary.md").read_text(encoding="utf-8")
    assert "vericoding/DA0152" in body
    assert "vericoding/DA0818" in body
    assert "1 of 3" in body
    summary = read_json(dest / "summary.json")
    assert summary["skipped_ids"] == ["vericoding/DA0152", "vericoding/DA0818"]
    assert summary["total_tasks"] == 3


def test_publish_vericoding_without_skips_has_empty_skip_report(tmp_path: Path) -> None:
    """Runs without skipped rows publish an empty skip report and no skip lines."""
    runs = tmp_path / "runs"
    run_dir = vericoding_run(runs, "noskips", [vericoding_line("vericoding/t1", 1, 1)])
    dest = publish(run_dir, tmp_path / "results")
    body = (dest / "summary.md").read_text(encoding="utf-8")
    assert "Skipped tasks" not in body
    summary = read_json(dest / "summary.json")
    assert summary["skipped_ids"] == []
    assert summary["total_tasks"] == 1
