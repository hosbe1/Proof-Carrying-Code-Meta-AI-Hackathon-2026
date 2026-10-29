"""Fixed evaluation datasets and their test splits (docs/plan.md, "Evaluation").

Datasets are never committed. :func:`fetch` downloads each dataset once at a pinned revision, checks its sha256
and stores it in a cache; :func:`load` reads tasks from the cache and :func:`split` serves the fixed test split
(whose task IDs live in ``src/pcc/evaluation/splits/``) or the matching train split. ``python -m
pcc.evaluation.datasets make-splits`` draws the split files with ``random.Random(2026)``.

Dataset findings (checked against the live sources while writing this module):

- ``humanevalfix`` is the Python part of ``bigcode/humanevalpack`` (the HumanEvalFix data of the OctoPack paper,
  MIT): 164 rows with fields such as ``task_id`` (``"Python/0"`` .. ``"Python/163"``), ``prompt``,
  ``canonical_solution``, ``buggy_solution``, ``bug_type`` and ``test``. The files in the repository are parquet,
  so this module pages through the JSON rows API of the datasets server instead of adding a parquet dependency.
- ``vericoding`` is ``beneficial-ai-foundation/vericoding`` (MIT), file ``train.jsonl`` (12,504 rows) with fields
  ``id``, ``language``, ``source``, ``source-id`` and the ``vc-*`` Dafny parts. Only rows with ``language ==
  "dafny"`` are kept, and rows with ``source == "verified_cogen"`` are excluded (unlicensed origin, see
  ``THIRD_PARTY.md``), leaving 2,857 tasks. Rows with ``source == "humaneval"`` carry ``source-id`` values such as
  ``humaneval_001_separate-paren-groups``; their numeric part is the HumanEval index, so they share the
  ``humaneval/HumanEval/<n>`` source-task form with HumanEvalFix for deduplication.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pcc.core.errors import ToolError

HUMANEVALFIX = "humanevalfix"
VERICODING = "vericoding"

DEFAULT_CACHE_DIR = Path.home() / ".cache/pcc/datasets"

_SPLITS_DIR = Path(__file__).resolve().parent / "splits"

_SPLIT_SEED = 2026
_HUMANEVALFIX_TEST_SIZE = 40
_VERICODING_TEST_FRACTION = 0.2
_PAGE_LENGTH = 100
_DOWNLOAD_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class Task:
    """One evaluation task: its ID in the dataset, the benchmark task it came from, and its raw payload."""

    id: str
    source_task: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _DatasetSpec:
    """How to fetch one dataset: URL, pinned revision, sha256 of the cached bytes and the cache file name."""

    name: str
    url: str
    revision: str
    sha256: str
    cache_filename: str
    kind: Literal["rows", "jsonl"]


_DATASETS: dict[str, _DatasetSpec] = {
    HUMANEVALFIX: _DatasetSpec(
        name=HUMANEVALFIX,
        url=("https://datasets-server.huggingface.co/rows?dataset=bigcode%2Fhumanevalpack&config=python&split=test"),
        revision="9a41762f73a8cb23bb5811b73d5aab164efcf378",
        sha256="e831c02d4b304c2adca8afb9dec7699cd34c59b47429a091d0b85f92fc9a536b",
        cache_filename="humanevalfix.json",
        kind="rows",
    ),
    VERICODING: _DatasetSpec(
        name=VERICODING,
        url=(
            "https://huggingface.co/datasets/beneficial-ai-foundation/vericoding/resolve/"
            "f14842db1b41fa92035940a520cc76b8620293a1/train.jsonl"
        ),
        revision="f14842db1b41fa92035940a520cc76b8620293a1",
        sha256="4f8400da4b2cf0edb4b479d3511f9979bac85fe19763c5c9c691c907a085b341",
        cache_filename="vericoding-train.jsonl",
        kind="jsonl",
    ),
}


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    """The default cache directory when ``cache_dir`` is None, else ``cache_dir`` itself."""
    return DEFAULT_CACHE_DIR if cache_dir is None else cache_dir


def _spec(name: str) -> _DatasetSpec:
    """The dataset spec for ``name``, or a ``ValueError`` naming the known datasets."""
    try:
        return _DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(_DATASETS))
        raise ValueError(f"unknown dataset {name!r}; expected one of: {known}") from None


def _http_get(url: str) -> bytes:
    """The raw bytes behind ``url``; a network failure becomes a ``ToolError``."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "pcc-evaluation/1"})
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            data: bytes = response.read()
            return data
    except (urllib.error.URLError, OSError) as exc:
        raise ToolError(f"could not download {url}: {exc}") from exc


def _check_sha256(data: bytes, expected: str, label: str) -> None:
    """Raise a ``ToolError`` naming ``label`` when ``data`` does not match the expected sha256."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ToolError(f"sha256 mismatch for {label}: expected {expected}, got {actual}")


def _canonical_rows(rows: list[dict[str, Any]]) -> bytes:
    """The cache bytes for rows-API datasets: canonical JSON, so the hash pins the content, not the paging."""
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fetch_rows_page(spec: _DatasetSpec, offset: int) -> list[dict[str, Any]]:
    """One page of row dicts from the rows API; a broken response becomes a ``ToolError``."""
    separator = "&" if "?" in spec.url else "?"
    raw = _http_get(f"{spec.url}{separator}offset={offset}&length={_PAGE_LENGTH}")
    try:
        page = json.loads(raw)
        return [entry["row"] for entry in page["rows"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise ToolError(f"could not parse rows page at offset {offset} from {spec.url}: {exc}") from exc


def _download(spec: _DatasetSpec) -> bytes:
    """The verified cache bytes for ``spec``: all rows-API pages assembled, or the raw jsonl file."""
    if spec.kind == "jsonl":
        raw = _http_get(spec.url)
    else:
        rows: list[dict[str, Any]] = []
        while True:
            page = _fetch_rows_page(spec, len(rows))
            rows.extend(page)
            if len(page) < _PAGE_LENGTH:
                break
        raw = _canonical_rows(rows)
    _check_sha256(raw, spec.sha256, f"{spec.url} (cache file {spec.cache_filename})")
    return raw


def fetch(name: str, cache_dir: Path | None = None) -> Path:
    """Download dataset ``name`` into ``cache_dir`` (created when missing) and return the cache file path.

    A cache file that already verifies against its pinned sha256 is reused without any download. A network
    failure or a hash mismatch raises a ``ToolError``.
    """
    spec = _spec(name)
    directory = _resolve_cache_dir(cache_dir)
    path = directory / spec.cache_filename
    if path.is_file():
        _check_sha256(path.read_bytes(), spec.sha256, str(path))
        return path
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_download(spec))
    return path


def _humaneval_source_task(number: int) -> str:
    """The shared source-task form for HumanEval-derived tasks, used for deduplication across datasets."""
    return f"humaneval/HumanEval/{number}"


def _parse_humanevalfix(data: bytes) -> tuple[Task, ...]:
    """Tasks from cached humanevalfix bytes; malformed rows become a ``ToolError``."""
    try:
        rows = json.loads(data)
    except ValueError as exc:
        raise ToolError(f"could not parse the humanevalfix cache: {exc}") from exc
    if not isinstance(rows, list):
        raise ToolError("could not parse the humanevalfix cache: expected a JSON array of rows")
    tasks: list[Task] = []
    for row in rows:
        try:
            task_id = row["task_id"]
            language, _, number = task_id.partition("/")
            source_task = _humaneval_source_task(int(number)) if language == "Python" else ""
            if not source_task or not number.isdigit():
                raise ValueError(task_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError(f"could not parse humanevalfix row {row!r}: {exc}") from exc
        tasks.append(Task(id=f"{HUMANEVALFIX}/{task_id}", source_task=source_task, payload=row))
    return tuple(sorted(tasks, key=lambda task: task.id))


_HUMANEVAL_SOURCE_ID = re.compile(r"^humaneval_(\d+)")


def _parse_vericoding(data: bytes) -> tuple[Task, ...]:
    """Dafny tasks from cached vericoding jsonl bytes, without the VerifiedCogen source."""
    tasks: list[Task] = []
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ToolError(f"could not parse the vericoding cache line {lineno}: {exc}") from exc
        try:
            language = row["language"]
            source = row["source"]
            source_id = row["source-id"]
            row_id = row["id"]
        except (KeyError, TypeError) as exc:
            raise ToolError(f"could not parse the vericoding cache line {lineno}: {exc}") from exc
        if language != "dafny" or source == "verified_cogen":
            continue
        if source == "humaneval":
            match = _HUMANEVAL_SOURCE_ID.match(source_id)
            if match is None:
                raise ToolError(f"could not parse humaneval source-id {source_id!r} in the vericoding cache")
            source_task = _humaneval_source_task(int(match.group(1)))
        else:
            source_task = f"{VERICODING}/{source}/{source_id}"
        tasks.append(Task(id=f"{VERICODING}/{row_id}", source_task=source_task, payload=row))
    return tuple(sorted(tasks, key=lambda task: task.id))


def load(name: str, cache_dir: Path | None = None) -> tuple[Task, ...]:
    """All tasks of dataset ``name`` from its cache, sorted by ID (fetching first when the cache is missing)."""
    spec = _spec(name)
    raw = fetch(name, cache_dir).read_bytes()
    if spec.kind == "jsonl":
        return _parse_vericoding(raw)
    return _parse_humanevalfix(raw)


def _read_test_ids(name: str, splits_dir: Path | None) -> frozenset[str]:
    """The task IDs of the committed test split file for ``name``."""
    path = (_SPLITS_DIR if splits_dir is None else splits_dir) / f"{name}_test.txt"
    if not path.is_file():
        raise ToolError(f"missing split file {path}")
    return frozenset(line for line in path.read_text(encoding="utf-8").splitlines() if line)


def _test_source_tasks(cache_dir: Path | None, splits_dir: Path | None) -> set[str]:
    """The source tasks of every test split of every dataset, for train-split deduplication."""
    blocked: set[str] = set()
    for name in _DATASETS:
        test_ids = _read_test_ids(name, splits_dir)
        blocked.update(task.source_task for task in load(name, cache_dir) if task.id in test_ids)
    return blocked


def split(
    name: str,
    which: Literal["test", "train"],
    cache_dir: Path | None = None,
    splits_dir: Path | None = None,
) -> tuple[Task, ...]:
    """The ``which`` split of dataset ``name``, sorted by ID.

    The test split holds exactly the task IDs of the committed ``<name>_test.txt`` file. Train holds every other
    task except those whose ``source_task`` appears in any test split of any dataset, so no benchmark task leaks
    from a test split into training through another dataset.
    """
    if which not in ("test", "train"):
        raise ValueError(f"unknown split {which!r}; expected 'test' or 'train'")
    tasks = load(name, cache_dir)
    test_ids = _read_test_ids(name, splits_dir)
    if which == "test":
        return tuple(task for task in tasks if task.id in test_ids)
    blocked = _test_source_tasks(cache_dir, splits_dir)
    return tuple(task for task in tasks if task.id not in test_ids and task.source_task not in blocked)


def make_splits(
    cache_dir: Path | None = None,
    splits_dir: Path | None = None,
    *,
    humanevalfix_test_size: int = _HUMANEVALFIX_TEST_SIZE,
    vericoding_test_fraction: float = _VERICODING_TEST_FRACTION,
) -> tuple[Path, Path]:
    """Draw both test splits with ``random.Random(2026)`` and write them sorted, one ID per line.

    Forty of the 164 HumanEvalFix tasks and twenty percent of the Vericoding tasks go to test; the rest,
    minus cross-dataset test source tasks, is train (see :func:`split`). Returns the two split file paths.
    """
    for name in _DATASETS:
        fetch(name, cache_dir)
    rng = random.Random(_SPLIT_SEED)
    humanevalfix_ids = sorted(task.id for task in load(HUMANEVALFIX, cache_dir))
    humanevalfix_test = sorted(rng.sample(humanevalfix_ids, humanevalfix_test_size))
    vericoding_ids = sorted(task.id for task in load(VERICODING, cache_dir))
    vericoding_test = sorted(rng.sample(vericoding_ids, int(len(vericoding_ids) * vericoding_test_fraction)))
    directory = _SPLITS_DIR if splits_dir is None else splits_dir
    directory.mkdir(parents=True, exist_ok=True)
    first = directory / f"{HUMANEVALFIX}_test.txt"
    second = directory / f"{VERICODING}_test.txt"
    for path, ids in ((first, humanevalfix_test), (second, vericoding_test)):
        path.write_text("".join(f"{task_id}\n" for task_id in ids), encoding="utf-8")
    return first, second


def main(argv: Sequence[str]) -> int:
    """Entry point for ``python -m pcc.evaluation.datasets``; returns the process exit code."""
    if list(argv) != ["make-splits"]:
        print("usage: python -m pcc.evaluation.datasets make-splits", file=sys.stderr)
        return 2
    make_splits()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
