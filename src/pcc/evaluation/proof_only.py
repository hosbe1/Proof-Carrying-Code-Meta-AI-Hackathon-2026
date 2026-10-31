"""Proof-only tasks for the proof step alone (docs/plan.md, Training, "Proof-only tasks").

Verified answers become tasks that keep the executable code and drop the proof:
``strip_ghost`` removes the annotations, ``make`` writes one line per answer that
still needs a proof. ``--format tasks`` writes harness lines
(``{"id", "source_task", "program"}``) for ``python -m pcc.evaluation vericoding
--tasks-file``; ``--format sft`` writes one line in the chat format of
``training/export_sft.py`` with the shared system message, the proof-only user
prompt and the original verified answer as the assistant message.

Input is a ``results.jsonl`` of the Vericoding harness (issue #46, rows with
``task_id``/``id``, ``verified`` and ``answer``) or an SFT file (issue #49 rows
with ``messages`` ending in an assistant turn); both shapes are detected per
line. Only verified, nonempty answers become candidates. A candidate whose
program already verifies needs no proof and is skipped, as is an answer without
ghost code; both skip counts are printed to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcc.config import Config
from pcc.evaluation.prompts import SYSTEM_MESSAGE
from pcc.verifier import verify
from pcc.verifier.anti_cheat import _normalize
from pcc.verifier.ghost import declared_lemma_names, spec_referenced_lemmas, strip_ghost


class _InputError(Exception):
    """A malformed proof-only input line: its 1-based line number and short reason."""

    def __init__(self, lineno: int, reason: str) -> None:
        super().__init__(lineno, reason)
        self.lineno = lineno
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One answer that may become a proof-only task."""

    id: str
    source_task: str
    answer: str


def build_proof_only_prompt(program: str) -> str:
    """Ask for ``program`` with proof annotations added and everything else unchanged.

    The model adds loop invariants, ``decreases`` clauses, asserts and helper
    lemmas; the executable code and every method header, including ``requires``,
    ``ensures`` and ``decreases`` clauses, stay exactly unchanged.
    """
    return (
        "Add the missing proof annotations to the Dafny program below so that it verifies: "
        "loop invariants, decreases clauses, asserts and helper lemmas. "
        "Keep the executable code and every method header, including requires, "
        "ensures and decreases clauses, exactly unchanged. "
        "Return only the complete Dafny program.\n\n" + program
    )


def _parse_result_row(record: dict[str, Any], lineno: int) -> _Candidate | None:
    """A ``results.jsonl`` row: only verified, nonempty answers become candidates."""
    raw_id = record.get("task_id", record.get("id"))
    if not isinstance(raw_id, str) or not raw_id:
        raise _InputError(lineno, "missing task id")
    if record.get("verified") is not True:
        return None
    answer = record.get("answer")
    if answer is None or not isinstance(answer, str):
        raise _InputError(lineno, "missing answer")
    if not answer.strip():
        return None
    raw_source = record.get("source_task")
    source_task = raw_source if isinstance(raw_source, str) and raw_source else raw_id
    return _Candidate(id=f"proof-only/{raw_id}", source_task=source_task, answer=answer)


def _parse_sft_row(record: dict[str, Any], lineno: int) -> _Candidate:
    """An SFT chat line: the assistant answer becomes the candidate."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _InputError(lineno, "missing messages")
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        raise _InputError(lineno, "missing assistant message")
    answer = last.get("content")
    if not isinstance(answer, str):
        raise _InputError(lineno, "missing assistant message")
    if not answer.strip():
        raise _InputError(lineno, "empty assistant message")
    digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    return _Candidate(id=f"proof-only/sft/{digest}", source_task=f"sft/{digest}", answer=answer)


def _parse_line(line: str, lineno: int) -> _Candidate | None:
    """One input line: a candidate, ``None`` for a skipped row, or an ``_InputError``."""
    try:
        record: Any = json.loads(line)
    except ValueError:
        raise _InputError(lineno, "invalid JSON") from None
    if not isinstance(record, dict):
        raise _InputError(lineno, "expected a JSON object")
    if "messages" in record:
        return _parse_sft_row(record, lineno)
    if any(key in record for key in ("task_id", "id", "verified", "answer")):
        return _parse_result_row(record, lineno)
    raise _InputError(lineno, "unrecognized shape")


def _default_verify(program: str) -> bool:
    """Whether ``program`` already verifies, under the harness resource limit."""
    return verify(program, resource_limit=Config().dafny.resource_limit).verified


def run_make(
    source: Path,
    *,
    out_format: str,
    out: Path,
    verify_fn: Callable[[str], bool] = _default_verify,
) -> tuple[int, int, int]:
    """Build proof-only lines from ``source`` into ``out``.

    Returns ``(written, skipped_without_ghost, skipped_verifying)``. Lemmas
    referenced by a kept header or specification stay in the program and are
    reported to stderr as ``warning: line <N>: ...``; malformed lines raise
    ``_InputError`` naming the line. The caller prints the skip counts.
    """
    if out_format not in ("tasks", "sft"):
        raise ValueError(f"unknown format {out_format!r}; expected 'tasks' or 'sft'")
    candidates: list[tuple[int, _Candidate]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        candidate = _parse_line(line, lineno)
        if candidate is None:
            continue
        if candidate.id in seen:
            raise _InputError(lineno, "duplicate task id")
        seen.add(candidate.id)
        candidates.append((lineno, candidate))
    written = 0
    skipped_without_ghost = 0
    skipped_verifying = 0
    lines: list[str] = []
    for lineno, candidate in candidates:
        names = declared_lemma_names(candidate.answer)
        for name in spec_referenced_lemmas(candidate.answer, names):
            print(
                f"warning: line {lineno}: lemma {name} is referenced in a specification; keeping its declaration",
                file=sys.stderr,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = strip_ghost(candidate.answer, lemma_names=names)
        if _normalize(program) == _normalize(candidate.answer):
            skipped_without_ghost += 1
            continue
        if verify_fn(program):
            skipped_verifying += 1
            continue
        if out_format == "tasks":
            lines.append(json.dumps({"id": candidate.id, "source_task": candidate.source_task, "program": program}))
        else:
            lines.append(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": build_proof_only_prompt(program)},
                            {"role": "assistant", "content": candidate.answer},
                        ]
                    }
                )
            )
        written += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return written, skipped_without_ghost, skipped_verifying


def build_parser() -> argparse.ArgumentParser:
    """The command line: ``make`` proof-only lines from verified answers."""
    parser = argparse.ArgumentParser(
        prog="python -m pcc.evaluation.proof_only",
        description="Build proof-only tasks from verified answers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="write proof-only lines from a results or SFT file")
    make.add_argument("--from", dest="source", required=True, type=Path, help="input results.jsonl or SFT file")
    make.add_argument("--format", required=True, choices=("tasks", "sft"), help="output line format")
    make.add_argument("--out", required=True, type=Path, help="output JSONL file")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    verify_fn: Callable[[str], bool] | None = None,
) -> int:
    """Run the proof-only command named in ``argv``; return the process exit code."""
    args = build_parser().parse_args(argv)
    check = verify_fn if verify_fn is not None else _default_verify
    try:
        written, skipped_without_ghost, skipped_verifying = run_make(
            args.source, out_format=args.format, out=args.out, verify_fn=check
        )
    except _InputError as exc:
        print(
            f"error: line {exc.lineno}: malformed proof-only input: {exc.reason}",
            file=sys.stderr,
        )
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"skipped {skipped_without_ghost} answers without ghost code")
    print(f"skipped {skipped_verifying} already verifying programs")
    print(f"wrote {written} proof-only tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
