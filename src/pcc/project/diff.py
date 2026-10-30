"""Which functions changed on a pull request (docs/plan.md, Product).

``changed`` maps the new-side hunks of ``git diff --unified=0 <base>...HEAD`` onto the
functions ``discover`` finds, so the action checks exactly the touched functions.
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from pathlib import Path

from pcc.core import FunctionRef, ToolError
from pcc.project.discover import discover

logger = logging.getLogger(__name__)

#: Bound on ``git diff``: a stuck git process is a broken tool, not an empty diff.
_GIT_TIMEOUT_SECONDS = 120.0

#: Cap on the git error text carried in a ``ToolError``.
_MAX_GIT_ERROR_CHARS = 500

#: ``@@ -old[,count] +new[,count] @@``; a missing count means one line.
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

#: Key of ``_function_ranges``: the ``(qualname, lineno)`` that identifies a ``FunctionRef``.
_RangeKey = tuple[str, int]


def changed(repo_root: Path, base: str) -> tuple[FunctionRef, ...]:
    """Every function whose range overlaps a new-side diff hunk against ``base``.

    It runs ``git diff --unified=0 <base>...HEAD -- '*.py'`` in ``repo_root`` and returns
    every function from ``discover`` over the changed files whose line range (first
    decorator line to the end) overlaps a hunk on the new side. Results are ordered by
    path, then by ``lineno``.

    A deleted file holds no current functions, so it contributes nothing; a deleted
    function does not appear either. A hunk outside every function (imports, constants,
    comments) reports nothing.

    Raises ``ToolError`` when git fails: ``repo_root`` is no repository, ``base`` names
    no ref, or the git process cannot be started or times out.
    """
    hunks = _new_side_hunks(repo_root, base)
    found: list[FunctionRef] = []
    for rel in sorted(hunks):
        full = (repo_root.resolve() / rel).resolve()
        if repo_root.resolve() not in (full, *full.parents):
            continue
        if not full.is_file():
            continue  # Deleted or renamed-away: no current functions to report.
        refs = discover(repo_root, [rel])
        ranges = _function_ranges(full)
        file_hunks = hunks[rel]
        for ref in refs:
            span = ranges.get((ref.qualname, ref.lineno))
            if span is None:  # Cannot happen: the same file parsed for both.
                continue
            start, end = span
            if any(start <= hunk_end and hunk_start <= end for hunk_start, hunk_end in file_hunks):
                found.append(ref)
    return tuple(found)


def _new_side_hunks(repo_root: Path, base: str) -> dict[str, list[tuple[int, int]]]:
    """Map every changed ``*.py`` file to its new-side hunk ranges (inclusive)."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--unified=0", f"{base}...HEAD", "--", "*.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            cwd=str(repo_root),
            check=False,
        )
    except OSError as error:
        raise ToolError(f"cannot run git in {repo_root}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise ToolError(f"git diff in {repo_root} timed out after {_GIT_TIMEOUT_SECONDS:g} s") from error
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"git diff exited with code {proc.returncode}"
        raise ToolError(f"git diff {base}...HEAD in {repo_root} failed: {detail[:_MAX_GIT_ERROR_CHARS]}")
    hunks: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("+++ "):
            current = _diff_path(line[len("+++ ") :])
        elif line.startswith("@@ ") and current is not None:
            match = _HUNK_RE.match(line)
            if match is None:
                continue
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            if count > 0:  # A zero count is a deletion: it covers no new-side line.
                hunks.setdefault(current, []).append((start, start + count - 1))
    return hunks


def _diff_path(marked: str) -> str | None:
    """The new-side path of a ``+++`` line, or ``None`` for a deleted file."""
    text = marked.strip()
    if text == "/dev/null":
        return None
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("b/"):
        text = text[len("b/") :]
    return text


def _function_ranges(path: Path) -> dict[_RangeKey, tuple[int, int]]:
    """Map ``(qualname, lineno)`` to ``(first decorator line, end line)`` in ``path``.

    The walk mirrors ``discover`` (classes and functions push a scope, bound lambdas do
    not), so every ``FunctionRef`` ``discover`` reports has an entry here.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("skipping %s: cannot read it (%s)", path, error)
        return {}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        logger.warning("skipping %s: syntax error (%s)", path, error)
        return {}
    collector = _RangeCollector()
    collector.visit(tree)
    return collector.ranges


class _RangeCollector(ast.NodeVisitor):
    """Collect ``(qualname, lineno) -> (start, end)`` in source order."""

    def __init__(self) -> None:
        self.ranges: dict[_RangeKey, tuple[int, int]] = {}
        self._scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self._visit_lambda_binding(node.targets[0].id, node, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._visit_lambda_binding(node.target.id, node, node.value)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        self._record(".".join([*self._scope, node.name]), node.lineno, start, node.end_lineno)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _visit_lambda_binding(self, name: str, stmt: ast.Assign | ast.AnnAssign, value: ast.expr | None) -> None:
        if isinstance(value, ast.Lambda):
            self._record(".".join([*self._scope, name]), stmt.lineno, stmt.lineno, stmt.end_lineno)

    def _record(self, qualname: str, lineno: int, start: int, end: int | None) -> None:
        if end is not None:  # Always set by ``ast.parse``; guarded for the type checker.
            self.ranges[(qualname, lineno)] = (start, end)


__all__ = ["changed"]
