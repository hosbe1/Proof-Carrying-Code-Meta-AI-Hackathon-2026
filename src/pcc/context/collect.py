"""Fixed evidence collector for step 1 (docs/plan.md): name, signature, docstring, comments, tests, call
sites, commit history and docs.

Deterministic and without model calls. Sections that cannot be collected are skipped without an error: files
that are missing or unparseable contribute nothing, a repository without git gives no COMMIT evidence, and a
failed ``git log -L`` only logs a warning. The target module is only parsed, never imported.
"""

from __future__ import annotations

import ast
import fnmatch
import io
import logging
import os
import re
import subprocess
import tokenize
from pathlib import Path

from pcc.context.github import github_evidence
from pcc.core import Context, Evidence, FunctionRef, SourceKind

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 2000
MAX_COMMENT_EVIDENCE = 10
MAX_TEST_EVIDENCE = 5
MAX_TEST_LINES = 40
MAX_CALL_SITE_EVIDENCE = 5
CALL_SITE_CONTEXT_LINES = 2
MAX_COMMIT_EVIDENCE = 10
MAX_DOCS_EVIDENCE = 3
_GIT_TIMEOUT_SECONDS = 60.0

_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{7,40}")


def collect(function: FunctionRef, repo_root: Path) -> Context:
    """Collect the base evidence package for ``function`` in ticket order.

    Never raises for missing history or unparseable files; a failed ``git log -L`` only logs a warning.
    """
    name_ref = f"{function.path}:{function.lineno}"
    short = _short_name(function.qualname)
    evidence: list[Evidence] = [Evidence(SourceKind.NAME, name_ref, _cut(function.qualname))]
    signature = _signature(function.source)
    if signature is not None:
        evidence.append(Evidence(SourceKind.SIGNATURE, name_ref, _cut(signature)))
    docstring = _docstring(function.source)
    if docstring is not None:
        evidence.append(Evidence(SourceKind.DOCSTRING, name_ref, _cut(docstring)))
    evidence.extend(_comments(function, repo_root))
    sources = _python_sources(repo_root)
    evidence.extend(_tests(function, sources))
    evidence.extend(_call_sites(function, sources, short))
    evidence.extend(_commits(function, repo_root))
    evidence.extend(_docs(repo_root, short))
    evidence.extend(github_evidence([item for item in evidence if item.kind is SourceKind.COMMIT], os.environ))
    return Context(function=function, evidence=tuple(evidence))


def _short_name(qualname: str) -> str:
    """The identifier to match: the last component of the qualified name."""
    parts = [part for part in qualname.split(".") if part and part != "<locals>"]
    return parts[-1] if parts else qualname


def _cut(text: str) -> str:
    """Truncate excerpts longer than 2,000 characters with an ellipsis."""
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    return text[:MAX_EXCERPT_CHARS] + "…"


def _signature(source: str) -> str | None:
    """The ``def`` line(s) with hints and defaults, never including the body.

    The header ends where the first body statement starts (from ``ast``), so a trailing comment on the
    ``def`` line or a single-line definition cannot leak the body into the excerpt.
    """
    lines = source.splitlines()
    if not lines:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return lines[0]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                return lines[node.lineno - 1]
            first = node.lineno - 1
            last = node.body[0].lineno - 1
            if last <= first:
                head = lines[first][: node.body[0].col_offset].rstrip()
                return head or lines[first]
            end = first
            while end < last and not lines[end].rstrip().endswith(":"):
                end += 1
            return "\n".join(lines[first : end + 1] if end < last else lines[first:last])
    return lines[0]


def _docstring(source: str) -> str | None:
    """The docstring with its doctests, or None when there is none."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ast.get_docstring(node)
    return None


def _end_lineno(function: FunctionRef) -> int:
    """The last line of the function, derived from its source."""
    return function.lineno + max(len(function.source.splitlines()), 1) - 1


def _repo_lines(repo_root: Path, rel: str) -> list[str] | None:
    """The lines of a repository file, or None when it cannot be read."""
    try:
        return (repo_root / rel).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _comment_tokens(text: str) -> list[tuple[int, str]]:
    """(1-based row, comment text) for every non-empty ``#`` comment in ``text``."""
    found: list[tuple[int, str]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment = token.string[1:].strip()
                if comment:
                    found.append((token.start[0], comment))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


def _comments(function: FunctionRef, repo_root: Path) -> list[Evidence]:
    """The comment block directly above ``def``, then each comment line in the body."""
    found: list[Evidence] = []
    lines = _repo_lines(repo_root, function.path)
    if lines is None:
        tokens = [(row + function.lineno - 1, text) for row, text in _comment_tokens(function.source)]
        return [
            Evidence(SourceKind.COMMENT, f"{function.path}:{row}", _cut(text))
            for row, text in tokens[:MAX_COMMENT_EVIDENCE]
        ]
    block = _above_block(lines, function.lineno)
    if block:
        excerpt = "\n".join(text for _, text in block)
        found.append(Evidence(SourceKind.COMMENT, f"{function.path}:{block[0][0]}", _cut(excerpt)))
    end = min(_end_lineno(function), len(lines))
    for row, text in _comment_tokens("\n".join(lines)):
        if function.lineno <= row <= end:
            found.append(Evidence(SourceKind.COMMENT, f"{function.path}:{row}", _cut(text)))
        if len(found) >= MAX_COMMENT_EVIDENCE:
            break
    return found[:MAX_COMMENT_EVIDENCE]


def _above_block(lines: list[str], lineno: int) -> list[tuple[int, str]]:
    """(1-based row, comment text) of the comment block directly above the ``def`` line, in source order."""
    picked: list[tuple[int, str]] = []
    index = lineno - 2
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped.startswith("#"):
            break
        text = stripped[1:].strip()
        if text:
            picked.append((index + 1, text))
        index -= 1
    picked.reverse()
    return picked


def _python_sources(repo_root: Path) -> list[tuple[str, str]]:
    """(POSIX relative path, text) of every readable ``*.py`` file, in sorted order."""
    found: list[tuple[str, str]] = []
    if not repo_root.is_dir():
        return found
    for path in sorted(repo_root.rglob("*.py")):
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in rel.split("/")[:-1]):
            continue
        try:
            found.append((rel, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return found


def _is_test_file(rel: str) -> bool:
    """Whether ``rel`` holds tests: a ``test_*.py``/``*_test.py`` file or anything under ``tests/``."""
    parts = rel.split("/")
    name = parts[-1]
    return fnmatch.fnmatchcase(name, "test_*.py") or fnmatch.fnmatchcase(name, "*_test.py") or "tests" in parts[:-1]


def _test_functions(tree: ast.Module) -> list[tuple[str, int, int, ast.AST]]:
    """(display name, first line, last line, node) of test functions at module or class level."""
    found: list[tuple[str, int, int, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                found.append((node.name, node.lineno, node.end_lineno or node.lineno, node))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    found.append(
                        (
                            f"{node.name}::{child.name}",
                            child.lineno,
                            child.end_lineno or child.lineno,
                            child,
                        )
                    )
    return found


def _mentions(node: ast.AST, short: str) -> bool:
    """Whether ``short`` occurs as a whole identifier: a ``Name`` or an attribute, never a substring."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == short:
            return True
        if isinstance(child, ast.Attribute) and child.attr == short:
            return True
    return False


def _tests(function: FunctionRef, sources: list[tuple[str, str]]) -> list[Evidence]:
    """The source of each test function that uses the function's name as an identifier."""
    found: list[Evidence] = []
    for rel, text in sources:
        if not _is_test_file(rel):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for name, start, end, node in _test_functions(tree):
            if rel == function.path and start == function.lineno:
                continue
            if not _mentions(node, _short_name(function.qualname)):
                continue
            excerpt = "\n".join(lines[start - 1 : end][:MAX_TEST_LINES])
            found.append(Evidence(SourceKind.TEST, f"{rel}::{name}", _cut(excerpt)))
            if len(found) >= MAX_TEST_EVIDENCE:
                return found
    return found


def _call_sites(function: FunctionRef, sources: list[tuple[str, str]], short: str) -> list[Evidence]:
    """Each call outside the function: the calling line with two lines around it.

    The window never overlaps the function's own lines in the same file, and a candidate whose
    excerpt still contains the whole ``function.source`` is skipped, so the body cannot leak.
    """
    start = function.lineno
    end = _end_lineno(function)
    matches: list[tuple[str, int, Evidence]] = []
    for rel, text in sources:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls = node.func.id == short
            elif isinstance(node.func, ast.Attribute):
                calls = node.func.attr == short
            else:
                continue
            if not calls:
                continue
            if rel == function.path and start <= node.lineno <= end:
                continue
            first = max(node.lineno - CALL_SITE_CONTEXT_LINES, 1)
            last = min(node.lineno + CALL_SITE_CONTEXT_LINES, len(lines))
            if rel == function.path:
                if node.lineno > end:
                    first = max(first, end + 1)
                elif node.lineno < start:
                    last = min(last, start - 1)
            excerpt = "\n".join(lines[first - 1 : last])
            if function.source in excerpt:
                continue
            matches.append((rel, node.lineno, Evidence(SourceKind.CALL_SITE, f"{rel}:{node.lineno}", _cut(excerpt))))
    matches.sort(key=lambda match: (match[0], match[1]))
    return [evidence for _, _, evidence in matches[:MAX_CALL_SITE_EVIDENCE]]


def _inside_git_repo(repo_root: Path) -> bool:
    """Whether ``repo_root`` lies inside a git work tree."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _commits(function: FunctionRef, repo_root: Path) -> list[Evidence]:
    """The commit history of the function's lines; empty outside git or when ``git log -L`` fails."""
    if not _inside_git_repo(repo_root):
        return []
    start = function.lineno
    end = _end_lineno(function)
    try:
        proc = subprocess.run(
            ["git", "log", "-L", f"{start},{end}:{function.path}", "--format=%h%n%s%n%b", "--no-patch"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("git log -L for %s lines %d-%d failed: %s", function.path, start, end, error)
        return []
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"exit code {proc.returncode}"
        logger.warning("git log -L for %s lines %d-%d failed: %s", function.path, start, end, reason)
        return []
    valid = _commit_hashes(function, repo_root, start, end)
    return _parse_commits(proc.stdout, valid)[:MAX_COMMIT_EVIDENCE]


def _commit_hashes(function: FunctionRef, repo_root: Path, start: int, end: int) -> set[str] | None:
    """The short hashes of the commits touching the function's lines, or None when unknown."""
    try:
        proc = subprocess.run(
            ["git", "log", "-L", f"{start},{end}:{function.path}", "--format=%h", "--no-patch"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return {line.strip() for line in proc.stdout.splitlines() if _COMMIT_HASH_RE.fullmatch(line.strip())}


def _is_commit_boundary(line: str, valid: set[str] | None) -> bool:
    """Whether ``line`` starts a new commit block: a hash, restricted to known hashes when available."""
    if not _COMMIT_HASH_RE.fullmatch(line):
        return False
    return valid is None or line in valid


def _parse_commits(output: str, valid: set[str] | None = None) -> list[Evidence]:
    """Split ``git log --format=%h%n%s%n%b`` output into one evidence item per commit, newest first."""
    found: list[Evidence] = []
    lines = output.splitlines()
    index = 0
    while index < len(lines):
        if not _is_commit_boundary(lines[index], valid):
            index += 1
            continue
        short_hash = lines[index]
        subject = lines[index + 1] if index + 1 < len(lines) else ""
        body_lines: list[str] = []
        cursor = index + 2
        while cursor < len(lines) and not _is_commit_boundary(lines[cursor], valid):
            if lines[cursor].startswith("diff --git "):
                break
            body_lines.append(lines[cursor])
            cursor += 1
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        body = "\n".join(body_lines)
        excerpt = subject if not body else f"{subject}\n\n{body}"
        found.append(Evidence(SourceKind.COMMIT, f"commit {short_hash}", _cut(excerpt)))
        index = cursor
    return found


def _word_present(text: str, word: str) -> bool:
    """Whether ``word`` occurs as a whole word, so a common name like ``max`` never matches ``maximum``."""
    if not word:
        return False
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _docs(repo_root: Path, short: str) -> list[Evidence]:
    """Paragraphs of ``README*`` and ``docs/**/*.md`` that contain the function name as a whole word."""
    candidates = sorted(path for path in repo_root.glob("README*") if path.is_file())
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        candidates += sorted(docs_dir.rglob("*.md"))
    found: list[Evidence] = []
    for path in candidates:
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for paragraph in re.split(r"\n\s*\n", text):
            stripped = paragraph.strip()
            if stripped and _word_present(stripped, short):
                found.append(Evidence(SourceKind.DOCS, rel, _cut(stripped)))
                if len(found) >= MAX_DOCS_EVIDENCE:
                    return found
    return found
