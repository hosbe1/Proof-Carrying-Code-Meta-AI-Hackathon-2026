"""Tests for the fixed evaluation datasets (issue #15).

Unit tests run offline: a tiny hand-written fixture stands in for each dataset and the download is
monkeypatched. The live test downloads both datasets for real and checks the counts.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from pcc.core.errors import ToolError
from pcc.evaluation import datasets
from pcc.evaluation.datasets import Task, fetch, load, main, make_splits, split

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HUMANEVALFIX_ROWS: list[dict[str, str]] = json.loads((FIXTURES / "humanevalfix.json").read_text(encoding="utf-8"))
VERICODING_BYTES = (FIXTURES / "vericoding-train.jsonl").read_bytes()

HUMANEVALFIX_IDS = [f"humanevalfix/Python/{n}" for n in range(10)]
VERICODING_IDS = [
    "vericoding/DA0002",
    "vericoding/DA0003",
    "vericoding/DA0004",
    "vericoding/DH0000",
    "vericoding/DH0001",
    "vericoding/DH0005",
    "vericoding/DH0007",
]


def _canonical(rows: object) -> bytes:
    """The cache bytes ``fetch`` stores for a rows-API dataset."""
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fixture_datasets() -> dict[str, datasets._DatasetSpec]:
    """A ``_DATASETS`` table with the fixtures served from local files, for monkeypatching."""
    humanevalfix_cache = _canonical(HUMANEVALFIX_ROWS)
    return {
        "humanevalfix": datasets._DatasetSpec(
            name="humanevalfix",
            url="https://example.invalid/humanevalfix-rows",
            revision="fixture",
            sha256=hashlib.sha256(humanevalfix_cache).hexdigest(),
            cache_filename="humanevalfix.json",
            kind="rows",
        ),
        "vericoding": datasets._DatasetSpec(
            name="vericoding",
            url="https://example.invalid/vericoding-train.jsonl",
            revision="fixture",
            sha256=hashlib.sha256(VERICODING_BYTES).hexdigest(),
            cache_filename="vericoding-train.jsonl",
            kind="jsonl",
        ),
    }


@pytest.fixture
def fixture_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Serve the fixture datasets through a monkeypatched download into a bare cache directory."""
    specs = _fixture_datasets()

    def fake_http_get(url: str) -> bytes:
        for spec in specs.values():
            if url == spec.url or url.startswith(spec.url + "?") or url.startswith(spec.url + "&"):
                if spec.kind == "jsonl":
                    return VERICODING_BYTES
                query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                offset = int(query.get("offset", ["0"])[0])
                length = int(query.get("length", ["100"])[0])
                page = HUMANEVALFIX_ROWS[offset : offset + length]
                return json.dumps({"rows": [{"row": row} for row in page]}).encode("utf-8")
        raise AssertionError(f"unexpected download URL {url!r}")

    monkeypatch.setattr(datasets, "_DATASETS", specs)
    monkeypatch.setattr(datasets, "_http_get", fake_http_get)
    cache_dir = tmp_path / "no-such-dir-yet" / "datasets"
    yield cache_dir


@pytest.fixture
def fixture_splits(tmp_path: Path) -> Path:
    """Split files over the fixture task IDs, with a cross-dataset source overlap for dedup tests."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "humanevalfix_test.txt").write_text(
        "humanevalfix/Python/5\nhumanevalfix/Python/8\n", encoding="utf-8"
    )
    (splits_dir / "vericoding_test.txt").write_text("vericoding/DA0003\n", encoding="utf-8")
    return splits_dir


def test_fetch_downloads_verifies_and_caches(fixture_env: Path) -> None:
    first = fetch("humanevalfix", fixture_env)
    assert first == fixture_env / "humanevalfix.json"
    assert first.is_file()
    assert json.loads(first.read_bytes()) == HUMANEVALFIX_ROWS
    # A missing cache directory is created; the second fetch needs no download.
    second = fetch("vericoding", fixture_env)
    assert second == fixture_env / "vericoding-train.jsonl"
    assert second.read_bytes() == VERICODING_BYTES


def test_fetch_reuses_a_valid_cache_without_downloading(fixture_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = fetch("humanevalfix", fixture_env)

    def no_network(url: str) -> bytes:
        raise AssertionError(f"must not download {url!r}")

    monkeypatch.setattr(datasets, "_http_get", no_network)
    assert fetch("humanevalfix", fixture_env) == path


def test_fetch_paginates_rows_api(fixture_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_fake = datasets._http_get
    seen_lengths: list[int] = []

    def paged(url: str) -> bytes:
        raw = real_fake(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "offset" in query:
            seen_lengths.append(int(query.get("length", ["0"])[0]))
        return raw

    monkeypatch.setattr(datasets, "_http_get", paged)
    monkeypatch.setattr(datasets, "_PAGE_LENGTH", 4)
    path = fetch("humanevalfix", fixture_env)
    assert [row["task_id"] for row in json.loads(path.read_bytes())] == [f"Python/{n}" for n in range(10)]
    assert seen_lengths == [4, 4, 4]


def test_fetch_network_failure_raises_tool_error(fixture_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(url: str) -> bytes:
        raise ToolError(f"could not download {url}: connection refused")

    monkeypatch.setattr(datasets, "_http_get", broken)
    with pytest.raises(ToolError):
        fetch("humanevalfix", fixture_env)


def test_fetch_hash_mismatch_raises_tool_error_naming_the_file(
    fixture_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "_http_get", lambda url: b'{"rows": []}')
    with pytest.raises(ToolError) as excinfo:
        fetch("humanevalfix", fixture_env)
    assert "humanevalfix.json" in str(excinfo.value)
    # The rows API offers no revision selector, so the drift error names the recorded revision too.
    assert "fixture" in str(excinfo.value)


def test_corrupt_cache_raises_tool_error_naming_the_file(fixture_env: Path) -> None:
    path = fetch("humanevalfix", fixture_env)
    path.write_bytes(b"corrupted")
    with pytest.raises(ToolError) as excinfo:
        load("humanevalfix", fixture_env)
    assert path.name in str(excinfo.value)
    with pytest.raises(ToolError) as excinfo:
        fetch("humanevalfix", fixture_env)
    assert path.name in str(excinfo.value)


def test_unknown_dataset_raises_value_error(fixture_env: Path) -> None:
    with pytest.raises(ValueError):
        fetch("bugs in py", fixture_env)
    with pytest.raises(ValueError):
        load("bugs in py", fixture_env)


def test_load_humanevalfix_task_shape(fixture_env: Path) -> None:
    tasks = load("humanevalfix", fixture_env)
    assert [task.id for task in tasks] == HUMANEVALFIX_IDS
    assert all(isinstance(task, Task) for task in tasks)
    by_id = {task.id: task for task in tasks}
    assert by_id["humanevalfix/Python/3"].source_task == "humaneval/HumanEval/3"
    assert by_id["humanevalfix/Python/0"].payload["entry_point"] == "add"
    assert by_id["humanevalfix/Python/0"].payload["task_id"] == "Python/0"


def test_load_vericoding_filters_and_source_tasks(fixture_env: Path) -> None:
    tasks = load("vericoding", fixture_env)
    assert [task.id for task in tasks] == VERICODING_IDS
    by_id = {task.id: task for task in tasks}
    assert by_id["vericoding/DH0000"].source_task == "humaneval/HumanEval/0"
    assert by_id["vericoding/DH0001"].source_task == "humaneval/HumanEval/1"
    assert by_id["vericoding/DA0002"].source_task == "vericoding/apps/apps_test_7"
    assert by_id["vericoding/DH0001"].payload["source-id"] == "humaneval_001_separate-paren-groups"
    assert "vericoding/DV0001" not in by_id
    assert "vericoding/LV0001" not in by_id
    assert "vericoding/VX0001" not in by_id


def test_split_test_and_train(fixture_env: Path, fixture_splits: Path) -> None:
    test_tasks = split("humanevalfix", "test", fixture_env, fixture_splits)
    assert [task.id for task in test_tasks] == ["humanevalfix/Python/5", "humanevalfix/Python/8"]
    train_tasks = split("humanevalfix", "train", fixture_env, fixture_splits)
    assert [task.id for task in train_tasks] == [
        i for i in HUMANEVALFIX_IDS if "Python/5" not in i and "Python/8" not in i
    ]


def test_split_train_excludes_test_source_tasks_of_other_datasets(fixture_env: Path, fixture_splits: Path) -> None:
    train_tasks = split("vericoding", "train", fixture_env, fixture_splits)
    assert [task.id for task in train_tasks] == [
        "vericoding/DA0002",
        "vericoding/DA0004",
        "vericoding/DH0000",
        "vericoding/DH0001",
        "vericoding/DH0007",
    ]
    test_source_tasks = {task.source_task for task in split("humanevalfix", "test", fixture_env, fixture_splits)} | {
        task.source_task for task in split("vericoding", "test", fixture_env, fixture_splits)
    }
    assert "humaneval/HumanEval/5" in test_source_tasks
    assert all(task.source_task not in test_source_tasks for task in train_tasks)


def test_split_rejects_unknown_split(fixture_env: Path, fixture_splits: Path) -> None:
    with pytest.raises(ValueError):
        split("humanevalfix", "valid", fixture_env, fixture_splits)  # type: ignore[arg-type]


def test_split_missing_split_file_raises_tool_error(fixture_env: Path, tmp_path: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        split("humanevalfix", "test", fixture_env, tmp_path)
    assert "humanevalfix_test.txt" in str(excinfo.value)


def test_make_splits_writes_sorted_files(fixture_env: Path, tmp_path: Path) -> None:
    splits_dir = tmp_path / "out"
    first, second = make_splits(fixture_env, splits_dir, humanevalfix_test_size=3, vericoding_test_fraction=0.25)
    assert first == splits_dir / "humanevalfix_test.txt"
    assert second == splits_dir / "vericoding_test.txt"
    for path, expected_count in ((first, 3), (second, 1)):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected_count
        assert lines == sorted(lines)
        assert path.read_text(encoding="utf-8").endswith("\n")
    # The draw is deterministic: running again gives the same files.
    before = (first.read_bytes(), second.read_bytes())
    make_splits(fixture_env, splits_dir, humanevalfix_test_size=3, vericoding_test_fraction=0.25)
    assert (first.read_bytes(), second.read_bytes()) == before


def test_main_rejects_unknown_command() -> None:
    assert main(["bogus"]) == 2
    assert main([]) == 2


def test_main_make_splits_calls_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Sequence[object]] = []

    def fake_make_splits(*args: object, **kwargs: object) -> tuple[Path, Path]:
        calls.append(args)
        first = tmp_path / "humanevalfix_test.txt"
        second = tmp_path / "vericoding_test.txt"
        first.write_text("", encoding="utf-8")
        second.write_text("", encoding="utf-8")
        return first, second

    monkeypatch.setattr(datasets, "make_splits", fake_make_splits)
    assert main(["make-splits"]) == 0
    assert len(calls) == 1


def test_committed_split_files_have_committed_shape() -> None:
    humanevalfix_ids = (datasets._SPLITS_DIR / "humanevalfix_test.txt").read_text(encoding="utf-8").splitlines()
    assert len(humanevalfix_ids) == 40
    assert humanevalfix_ids == sorted(humanevalfix_ids)
    assert len(set(humanevalfix_ids)) == 40
    assert all(i.startswith("humanevalfix/Python/") for i in humanevalfix_ids)
    vericoding_ids = (datasets._SPLITS_DIR / "vericoding_test.txt").read_text(encoding="utf-8").splitlines()
    assert len(vericoding_ids) == 571
    assert vericoding_ids == sorted(vericoding_ids)
    assert len(set(vericoding_ids)) == 571
    assert all(i.startswith("vericoding/") for i in vericoding_ids)


@pytest.mark.live
def test_live_downloads_both_datasets_and_checks_counts(tmp_path: Path) -> None:
    cache_dir = tmp_path / "datasets"
    humanevalfix_path = fetch("humanevalfix", cache_dir)
    vericoding_path = fetch("vericoding", cache_dir)
    assert humanevalfix_path.is_file()
    assert vericoding_path.is_file()
    humanevalfix_tasks = load("humanevalfix", cache_dir)
    vericoding_tasks = load("vericoding", cache_dir)
    assert len(humanevalfix_tasks) == 164
    assert len(vericoding_tasks) == 2857
    assert {task.source_task for task in humanevalfix_tasks} == {f"humaneval/HumanEval/{n}" for n in range(164)}
    assert len(split("humanevalfix", "test", cache_dir)) == 40
    assert len(split("vericoding", "test", cache_dir)) == 571
    vericoding_train = split("vericoding", "train", cache_dir)
    test_source_tasks = {task.source_task for task in split("humanevalfix", "test", cache_dir)} | {
        task.source_task for task in split("vericoding", "test", cache_dir)
    }
    assert all(task.source_task not in test_source_tasks for task in vericoding_train)
