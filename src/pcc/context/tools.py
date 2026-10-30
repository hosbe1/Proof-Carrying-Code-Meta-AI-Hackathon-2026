"""Read-only tools for the context search agent (issue #32).

The tools never return the function body: body lines of the function under check are replaced with
``OMITTED`` in every result, and paths are confined to ``repo_root``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pcc.core import FunctionRef
from pcc.core.errors import ToolError

OMITTED: str = "<function body omitted>"

__all__ = ["OMITTED", "git_log", "grep", "read_file"]

_MAX_READ_LINES = 200
_MAX_GREP_MATCHES = 50
_MAX_GIT_COMMITS = 20


def _require_repo_root(repo_root: Path) -> Path:
    """Return the resolved ``repo_root`` or raise ``ToolError`` when it is missing."""
    if not repo_root.exists() or not repo_root.is_dir():
        raise ToolError(f"repository root not found or not a directory: {repo_root}")
    return repo_root.resolve()


def _resolve_within(root: Path, argument: str) -> Path | None:
    """Resolve ``argument`` against ``root``; return ``None`` when it escapes the repository."""
    candidate = (root / argument).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _function_file(root: Path, function: FunctionRef) -> Path:
    """Resolved absolute path of the file that defines ``function``."""
    return (root / function.path).resolve()


def _body_lines(function: FunctionRef) -> set[int]:
    """1-based line numbers of the function body within its file (the ``def`` line excluded)."""
    count = len(function.source.splitlines())
    if count <= 1:
        return set()
    return set(range(function.lineno + 1, function.lineno + count))


def _hide_source(text: str, function: FunctionRef) -> str:
    """Replace any verbatim copy of the function source with ``OMITTED`` as a safety net."""
    source = function.source
    if source and source in text:
        text = text.replace(source, OMITTED)
    stripped = source.strip()
    if stripped and stripped != source and stripped in text:
        text = text.replace(stripped, OMITTED)
    return text


def read_file(repo_root: Path, argument: str, function: FunctionRef) -> str:
    """Return at most 200 lines of the file at ``argument`` relative to ``repo_root``.

    Body lines of ``function`` are replaced with ``OMITTED`` when its own file is read. A path
    outside ``repo_root`` or a missing/unreadable file returns an error string for the model.
    """
    root = _require_repo_root(repo_root)
    target = _resolve_within(root, argument)
    if target is None:
        return _hide_source(f"error: path {argument!r} is outside the repository", function)
    if not target.is_file():
        return _hide_source(f"error: file not found: {argument!r}", function)
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return _hide_source(f"error: cannot read file {argument!r}: {exc}", function)
    lines = text.splitlines()[:_MAX_READ_LINES]
    if target == _function_file(root, function):
        hidden = _body_lines(function)
        lines = [OMITTED if (index + 1) in hidden else line for index, line in enumerate(lines)]
    return _hide_source("\n".join(lines), function)


def grep(repo_root: Path, argument: str, function: FunctionRef) -> str:
    """Search ``argument`` as a regex over text files under ``repo_root``, skipping ``.git``.

    Returns at most 50 matches as ``relpath:lineno:content`` lines. Match lines on body lines of
    the function file carry ``OMITTED`` instead of the file content. An invalid regex returns an
    error string for the model.
    """
    root = _require_repo_root(repo_root)
    try:
        pattern = re.compile(argument)
    except re.error as exc:
        return _hide_source(f"error: invalid regex {argument!r}: {exc}", function)
    hidden = _body_lines(function)
    function_target = _function_file(root, function)
    matches: list[str] = []
    for candidate in sorted(root.rglob("*")):
        if len(matches) >= _MAX_GREP_MATCHES:
            break
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if ".git" in relative.parts:
            continue
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        relpath = relative.as_posix()
        is_function_file = candidate.resolve() == function_target
        for lineno, line in enumerate(text.splitlines(), start=1):
            if len(matches) >= _MAX_GREP_MATCHES:
                break
            if pattern.search(line) is None:
                continue
            content = OMITTED if (is_function_file and lineno in hidden) else line
            matches.append(f"{relpath}:{lineno}:{content}")
    return _hide_source("\n".join(matches), function)


def git_log(repo_root: Path, argument: str, function: FunctionRef) -> str:
    """Return at most 20 one-line commits for the path ``argument`` via the git CLI.

    A path outside ``repo_root`` or a missing path returns an error string for the model. A
    failing git subprocess raises ``ToolError``.
    """
    root = _require_repo_root(repo_root)
    command = ["git", "log", f"--max-count={_MAX_GIT_COMMITS}", "--oneline"]
    if argument:
        target = _resolve_within(root, argument)
        if target is None:
            return _hide_source(f"error: path {argument!r} is outside the repository", function)
        if not target.exists():
            return _hide_source(f"error: path not found: {argument!r}", function)
        command += ["--", target.relative_to(root).as_posix()]
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ToolError(f"git log failed: {exc}") from exc
    if completed.returncode != 0:
        raise ToolError(f"git log failed: {completed.stderr.strip() or completed.stdout.strip()}")
    lines = completed.stdout.splitlines()[:_MAX_GIT_COMMITS]
    return _hide_source("\n".join(lines), function)
