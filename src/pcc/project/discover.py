"""Finding the functions a repository defines (docs/plan.md, step 4, calls inside the project).

``discover`` walks files and directories for ``*.py`` files and reports every function as a ``FunctionRef``: every
``def`` and ``async def`` at any depth, and every ``name = lambda ...`` or ``name: T = lambda ...`` assignment.
Nested functions that capture outer variables, and lambdas not bound to a name, are left out: they are checked only
as part of their enclosing function.
"""

from __future__ import annotations

import ast
import logging
import symtable
import textwrap
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path

from pcc.core import FunctionRef, source_hash

logger = logging.getLogger(__name__)

# Directory names never descended into while walking a directory for ``*.py`` files. Hidden directories are skipped
# by the leading-dot rule; ``.venv`` is listed as well because the ticket names it.
_SKIP_DIRS = frozenset({".venv", "venv", "node_modules", "__pycache__", "build", "dist", "proofs"})

# Map key for matching an ``ast`` function or lambda with its ``symtable`` entry. Positions are unique per file, so
# no enclosing scope is needed to tell entries apart.
_FreeKey = tuple[str, int]


def discover(repo_root: Path, paths: Sequence[str]) -> tuple[FunctionRef, ...]:
    """Return every function and method defined in ``paths`` (relative to ``repo_root``).

    Directories in ``paths`` are walked recursively for ``*.py`` files, skipping hidden directories, ``.venv``,
    ``venv``, ``node_modules``, ``__pycache__``, ``build``, ``dist`` and ``proofs``. A file passed directly is read
    whatever its name is.

    ``qualname`` is Python ``__qualname__`` with every ``<locals>.`` removed; a lambda takes the bound name.
    ``lineno`` is the line of ``def``, not of the first decorator. ``source`` holds the lines from the first
    decorator to ``end_lineno``, dedented, with LF line ends. Nested functions with free variables (found with
    ``symtable``) and lambdas not bound to a name are left out. When a qualname is defined twice in one file, the
    last definition wins and a warning is logged. Results are ordered by file path, then by ``lineno``.

    A file with a syntax error is skipped with a ``logging.warning``. A path outside ``repo_root`` raises
    ``ValueError``; a path that does not exist raises ``FileNotFoundError``.
    """
    root = repo_root.resolve()
    files: dict[Path, None] = {}
    for entry in paths:
        full = (root / entry).resolve()
        if full != root and root not in full.parents:
            raise ValueError(f"path {entry!r} is outside the repository root {str(root)!r}")
        if not full.exists():
            raise FileNotFoundError(f"path {entry!r} does not exist under {str(root)!r}")
        if full.is_dir():
            for found in _iter_py_files(full):
                files.setdefault(found, None)
        else:
            files.setdefault(full, None)
    refs: list[FunctionRef] = []
    for file in sorted(files):
        refs.extend(_discover_in_file(root, file))
    refs.sort(key=lambda ref: (ref.path, ref.lineno))
    return tuple(refs)


def _iter_py_files(directory: Path) -> Iterator[Path]:
    """Yield every ``*.py`` file under ``directory``, skipping ignored directories."""
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError as error:
        logger.warning("skipping %s: cannot list directory (%s)", directory, error)
        return
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                continue
            yield from _iter_py_files(entry)
        elif entry.is_file() and entry.suffix == ".py":
            yield entry


def _discover_in_file(root: Path, file: Path) -> list[FunctionRef]:
    """Return the functions defined in one file, ordered by ``lineno``."""
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("skipping %s: cannot read it (%s)", file, error)
        return []
    try:
        tree = ast.parse(text, filename=str(file))
    except SyntaxError as error:
        logger.warning("skipping %s: syntax error (%s)", file, error)
        return []
    # ``ast.parse`` already succeeded, so building the symbol table cannot fail on syntax.
    free = _free_variables(symtable.symtable(text, str(file), "exec"))
    # Text mode translates CRLF to LF on read, so splitting on "\n" matches ``ast`` line numbers.
    lines = text.split("\n")
    collector = _Collector(lines, free, file.relative_to(root).as_posix())
    collector.visit(tree)
    by_qualname: dict[str, FunctionRef] = {}
    for ref in collector.refs:
        if ref.qualname in by_qualname:
            logger.warning("duplicate definition of %r in %s: keeping the last one", ref.qualname, file)
        by_qualname[ref.qualname] = ref
    return sorted(by_qualname.values(), key=lambda ref: ref.lineno)


def _free_variables(table: symtable.SymbolTable) -> dict[_FreeKey, deque[bool]]:
    """Map ``(name, lineno)`` of every nested function and lambda to whether it has free variables.

    A symbol counts as free exactly when ``symtable`` reports it so: used in the block but bound in an enclosing
    function scope. Names read from the module or from builtins are global, not free, so they never exclude a
    function.
    """
    found: dict[_FreeKey, deque[bool]] = {}

    def visit(current: symtable.SymbolTable) -> None:
        if current.get_type() == "function":
            name = current.get_name()
            key = (name if name not in ("lambda", "<lambda>") else "lambda", current.get_lineno())
            has_free = any(symbol.is_free() for symbol in current.get_symbols())
            found.setdefault(key, deque()).append(has_free)
        for child in current.get_children():
            visit(child)

    visit(table)
    return found


class _Collector(ast.NodeVisitor):
    """Collect ``FunctionRef`` objects in source order, tracking the enclosing classes and functions."""

    def __init__(self, lines: list[str], free: dict[_FreeKey, deque[bool]], path: str) -> None:
        self._lines = lines
        self._free = free
        self._path = path
        self._scope: list[str] = []
        self.refs: list[FunctionRef] = []

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
        if not self._take_free(node.name, node.lineno):
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            self._record(".".join([*self._scope, node.name]), node.lineno, start, node.end_lineno)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _visit_lambda_binding(self, name: str, stmt: ast.Assign | ast.AnnAssign, value: ast.expr | None) -> None:
        if not isinstance(value, ast.Lambda):
            return
        if self._take_free("lambda", value.lineno):
            return
        self._record(".".join([*self._scope, name]), stmt.lineno, stmt.lineno, stmt.end_lineno)

    def _take_free(self, name: str, lineno: int) -> bool:
        """Whether the function or lambda at ``(name, lineno)`` captures outer variables."""
        queue = self._free.get((name, lineno))
        if not queue:
            return False
        return queue.popleft()

    def _record(self, qualname: str, lineno: int, start: int, end: int | None) -> None:
        if end is None:  # Always set by ``ast.parse``; guarded for the type checker.
            return
        source = textwrap.dedent("\n".join(self._lines[start - 1 : end]))
        self.refs.append(FunctionRef(self._path, qualname, lineno, source, source_hash(source)))


__all__ = ["discover"]
