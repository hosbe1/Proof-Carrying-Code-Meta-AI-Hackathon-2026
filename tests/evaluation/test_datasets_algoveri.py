"""AlgoVeri loader edge cases (issue #181).

The main loader shape is covered in ``test_datasets.py``; these tests pin the
rejection of malformed rows and unknown split IDs. They run offline on the
three-problem fixture with a monkeypatched download.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pcc.core.errors import ToolError
from pcc.evaluation import datasets
from pcc.evaluation.datasets import load, split

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ALGOVERI_BYTES = (FIXTURES / "algoveri-dafny.jsonl").read_bytes()


def _spec_for(raw: bytes) -> dict[str, datasets._DatasetSpec]:
    return {
        datasets.ALGOVERI: datasets._DatasetSpec(
            name="algoveri",
            url="https://example.invalid/algoveri-dafny.jsonl",
            revision="fixture",
            sha256=hashlib.sha256(raw).hexdigest(),
            cache_filename="algoveri-dafny.jsonl",
            kind="jsonl",
        )
    }


@pytest.fixture
def algoveri_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Serve the AlgoVeri fixture through a monkeypatched download into a bare cache."""
    monkeypatch.setattr(datasets, "_DATASETS", _spec_for(ALGOVERI_BYTES))
    monkeypatch.setattr(datasets, "_http_get", lambda url: ALGOVERI_BYTES)
    return tmp_path / "datasets"


def _with_extra_row(extra: dict[str, str]) -> bytes:
    rows = [json.loads(line) for line in ALGOVERI_BYTES.decode("utf-8").splitlines() if line.strip()]
    rows.append(extra)
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")


def _load_raw(raw: bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(datasets, "_DATASETS", _spec_for(raw))
    monkeypatch.setattr(datasets, "_http_get", lambda url: raw)
    with pytest.raises(ToolError):
        load(datasets.ALGOVERI, tmp_path / "cache")


def test_code_hole_kept_but_prompt_and_prefix_stay_clean(algoveri_env: Path) -> None:
    tasks = load(datasets.ALGOVERI, algoveri_env)
    by_id = {task.id: task for task in tasks}
    assert "absolute value here" in by_id["algoveri/abs_val"].payload["vc-code"]
    assert "method Main()" in by_id["algoveri/is_empty"].payload["vc-postamble"]
    assert by_id["algoveri/max_two"].payload["vc-postamble"] == ""


def test_train_is_empty_even_for_a_partial_test_file(algoveri_env: Path, tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "algoveri_test.txt").write_text("algoveri/abs_val\n", encoding="utf-8")
    assert [task.id for task in split(datasets.ALGOVERI, "test", algoveri_env, splits_dir)] == ["algoveri/abs_val"]
    assert split(datasets.ALGOVERI, "train", algoveri_env, splits_dir) == ()


def test_duplicate_problem_id_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = json.loads(ALGOVERI_BYTES.decode("utf-8").splitlines()[0])
    _load_raw(_with_extra_row(first), monkeypatch, tmp_path)


def test_bad_problem_id_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _load_raw(
        _with_extra_row({"problem_id": "not/an/id", "nl_desc": "Bad ID.", "spec": "x"}),
        monkeypatch,
        tmp_path,
    )


def test_row_missing_nl_desc_raises_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _load_raw(
        _with_extra_row({"problem_id": "no_desc", "spec": "x"}),
        monkeypatch,
        tmp_path,
    )


def test_split_rejects_unknown_algoveri_ids(algoveri_env: Path, tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "algoveri_test.txt").write_text("algoveri/abs_val\nalgoveri/no_such_task\n", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        split(datasets.ALGOVERI, "test", algoveri_env, splits_dir)
    assert "algoveri/no_such_task" in str(excinfo.value)
    assert "algoveri_test.txt" in str(excinfo.value)
