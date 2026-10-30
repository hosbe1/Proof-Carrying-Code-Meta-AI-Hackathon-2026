"""The call graph between functions of the project (docs/plan.md, step 4, calls inside the project).

``call_graph`` parses each module with ``ast`` and resolves calls against the module's
imports. A call resolves to a project function when it is a bare name defined in the
same module, a name imported with ``from <project module> import name``, ``module.name``
for an imported project module, or ``self.name`` to a method of the same class. Other
calls (builtins, library calls, parameters, variables, dynamic calls, ``getattr``) are
ignored. Calls inside a nested function that ``discover`` skips (it captures an outer
variable) belong to the enclosing function, which is checked with them.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pcc.core import FunctionRef
from pcc.project.discover import discover

logger = logging.getLogger(__name__)

_SCOPE_CLASS = "class"
_SCOPE_FUNCTION = "function"


@dataclass
class _ModuleInfo:
    """One project file: its dotted name, imports and discovered functions."""

    path: str
    module: str  # dotted module name derived from ``path``
    qualnames: set[str]  # discovered function qualnames in this file
    top_level: set[str]  # discovered qualnames without a dot
    tree: ast.Module | None  # ``None`` when the file no longer parses
    import_alias: dict[str, str] = field(default_factory=dict)  # local name -> dotted module
    from_alias: dict[str, tuple[str, str]] = field(default_factory=dict)  # local name -> (module, name)


@dataclass
class _ProjectIndex:
    """Every discovered function plus the per-file import tables."""

    root: Path
    refs: dict[tuple[str, str], FunctionRef]  # (path, qualname) -> ref
    modules: dict[str, str]  # dotted module -> path
    files: dict[str, _ModuleInfo]  # path -> module info
    _edges_cache: dict[str, dict[str, frozenset[FunctionRef]]] = field(default_factory=dict)

    @classmethod
    def build(cls, repo_root: Path, functions: Sequence[FunctionRef]) -> _ProjectIndex:
        """Index ``functions`` by ``(path, qualname)`` and parse each file once."""
        root = repo_root.resolve()
        refs = {(ref.path, ref.qualname): ref for ref in functions}
        files: dict[str, _ModuleInfo] = {}
        modules: dict[str, str] = {}
        for ref in functions:
            if ref.path in files:
                continue
            tree = _parse_module(root / ref.path)
            qualnames = {other.qualname for other in functions if other.path == ref.path}
            info = _ModuleInfo(
                path=ref.path,
                module=_module_name(ref.path),
                qualnames=qualnames,
                top_level={name for name in qualnames if "." not in name},
                tree=tree,
            )
            files[ref.path] = info
            modules.setdefault(info.module, ref.path)
        index = cls(root=root, refs=refs, modules=modules, files=files)
        for info in files.values():
            if info.tree is not None:
                _read_imports(info)
        return index

    def lookup(self, path: str, qualname: str) -> FunctionRef | None:
        """The project function ``qualname`` in ``path``, or ``None``."""
        return self.refs.get((path, qualname))

    def edges_of(self, info: _ModuleInfo) -> Mapping[str, frozenset[FunctionRef]]:
        """Map every discovered qualname of ``info.path`` to its project callees."""
        cached = self._edges_cache.get(info.path)
        if cached is None:
            visitor = _CallVisitor(self, info)
            if info.tree is not None:
                visitor.visit(info.tree)
            cached = {name: frozenset(callees) for name, callees in visitor.edges.items()}
            self._edges_cache[info.path] = cached
        return cached


def call_graph(repo_root: Path, functions: Sequence[FunctionRef]) -> Mapping[FunctionRef, frozenset[FunctionRef]]:
    """The callees of every function in ``functions``, restricted to the project itself.

    The callee universe is every function ``discover`` finds under ``repo_root``; the
    result holds one entry per passed function, in order, with an empty frozenset for
    functions that call no project function. A passed function whose current file no
    longer defines it maps to an empty frozenset.
    """
    index = _ProjectIndex.build(repo_root, discover(repo_root, ["."]))
    graph: dict[FunctionRef, frozenset[FunctionRef]] = {}
    for ref in functions:
        info = index.files.get(ref.path)
        if info is None:
            graph[ref] = frozenset()
        else:
            graph[ref] = index.edges_of(info).get(ref.qualname, frozenset())
    return graph


def subset_edges(functions: Sequence[FunctionRef]) -> Mapping[FunctionRef, frozenset[FunctionRef]]:
    """Call edges among ``functions`` derived from their sources alone.

    Unlike ``call_graph`` this reads no files: each ``source`` snippet is parsed and
    calls resolve against the given functions. A bare name prefers an enclosing nested
    definition, then a same-file top-level function, then a top-level function of any
    other given file; ``self.name`` tries every enclosing class prefix of the qualname;
    a dotted ``a.b.c`` resolves when ``a.b`` (or a shorter prefix) is the module of a
    given function and the rest is its qualname. Ambiguous names may add edges
    ``call_graph`` would not, which only constrains the order further, never wrongly.
    A snippet that no longer parses contributes no edges.
    """
    index = {(ref.path, ref.qualname): ref for ref in functions}
    modules: dict[str, str] = {}
    for ref in functions:
        modules.setdefault(_module_name(ref.path), ref.path)
    top: dict[str, list[FunctionRef]] = {}
    for ref in functions:
        if "." not in ref.qualname:
            top.setdefault(ref.qualname, []).append(ref)
    return {ref: frozenset(_snippet_callees(ref, index, modules, top)) for ref in functions}


def _module_name(path: str) -> str:
    """The dotted module name of a POSIX repository-relative ``*.py`` path."""
    if path.endswith("/__init__.py"):
        return path[: -len("/__init__.py")].replace("/", ".")
    if path.endswith(".py"):
        return path[: -len(".py")].replace("/", ".")
    return path.replace("/", ".")


def _parse_module(path: Path) -> ast.Module | None:
    """Parse ``path``; warn and return ``None`` when it cannot be read or parsed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("skipping %s: cannot read it (%s)", path, error)
        return None
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError as error:
        logger.warning("skipping %s: syntax error (%s)", path, error)
        return None


def _package_of(info: _ModuleInfo) -> str:
    """The package ``from . import ...`` in ``info`` resolves against."""
    if info.path.endswith("/__init__.py") or info.path == "__init__.py":
        return info.module
    return info.module.rpartition(".")[0]


def _read_imports(info: _ModuleInfo) -> None:
    """Fill the alias tables of ``info`` from every import in the file."""
    assert info.tree is not None
    for node in ast.walk(info.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    info.import_alias[alias.asname] = alias.name
                else:
                    info.import_alias[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            module = _from_module(info, node)
            if module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                info.from_alias[alias.asname or alias.name] = (module, alias.name)


def _from_module(info: _ModuleInfo, node: ast.ImportFrom) -> str | None:
    """The absolute dotted module of a ``from`` import, or ``None`` when unresolvable."""
    if node.level == 0:
        return node.module or ""
    package = _package_of(info)
    if node.level > 1:
        parts = package.split(".") if package else []
        if len(parts) < node.level - 1:
            return None
        package = ".".join(parts[: len(parts) - (node.level - 1)])
    if node.module:
        return f"{package}.{node.module}" if package else node.module
    return package


def _dotted_name(node: ast.expr) -> list[str] | None:
    """The dotted path of ``a.b.c`` calls, or ``None`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        parts.reverse()
        return parts
    return None


class _CallVisitor(ast.NodeVisitor):
    """Collect ``qualname -> callees`` for one module, tracking scope and context.

    ``scope`` holds ``(kind, name)`` entries for the enclosing classes and functions;
    ``context`` is the nearest enclosing discovered function the calls belong to (a
    nested closure that ``discover`` skips reports to its enclosing function, module
    level code reports to no one).
    """

    def __init__(self, index: _ProjectIndex, info: _ModuleInfo) -> None:
        self._index = index
        self._info = info
        self.edges: dict[str, set[FunctionRef]] = {}
        self._scope: list[tuple[str, str]] = []
        self._context: str | None = None

    def visit_Import(self, node: ast.Import) -> None:
        pass  # Aliases come from ``_read_imports``; imports bind no calls by themselves.

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        pass  # Same as ``visit_Import``.

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._scope.append((_SCOPE_CLASS, node.name))
        for stmt in node.body:
            self.visit(stmt)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node.name, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node.name, node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Lambda):
            self._visit_lambda_body(node.targets[0].id, node.value)
        else:
            self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.value, ast.Lambda):
            self._visit_lambda_body(node.target.id, node.value)
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._resolve(node.func)
        if callee is not None and self._context is not None:
            self.edges.setdefault(self._context, set()).add(callee)
        self.generic_visit(node)

    def _visit_function(self, name: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([entry[1] for entry in self._scope] + [name])
        previous = self._context
        if qualname in self._info.qualnames:
            self._context = qualname
        self._scope.append((_SCOPE_FUNCTION, name))
        self.generic_visit(node)
        self._scope.pop()
        self._context = previous

    def _visit_lambda_body(self, name: str, node: ast.Lambda) -> None:
        # Like ``discover``, a bound lambda adds no scope level of its own.
        qualname = ".".join([entry[1] for entry in self._scope] + [name])
        previous = self._context
        if qualname in self._info.qualnames:
            self._context = qualname
        self.generic_visit(node)
        self._context = previous

    def _resolve(self, func: ast.expr) -> FunctionRef | None:
        if isinstance(func, ast.Name):
            return self._resolve_bare(func.id)
        parts = _dotted_name(func)
        if parts is None:
            return None
        return self._resolve_dotted(parts)

    def _resolve_bare(self, name: str) -> FunctionRef | None:
        for pos in range(len(self._scope) - 1, -1, -1):
            if self._scope[pos][0] != _SCOPE_FUNCTION:
                continue
            candidate = ".".join([entry[1] for entry in self._scope[: pos + 1]] + [name])
            found = self._index.lookup(self._info.path, candidate)
            if found is not None:
                return found
        if name in self._info.top_level:
            return self._index.lookup(self._info.path, name)
        target = self._info.from_alias.get(name)
        if target is not None:
            module, original = target
            path = self._index.modules.get(module)
            if path is not None:
                return self._index.lookup(path, original)
        return None

    def _resolve_dotted(self, parts: list[str]) -> FunctionRef | None:
        if parts[0] == "self":
            for pos in range(len(self._scope) - 1, -1, -1):
                if self._scope[pos][0] != _SCOPE_CLASS:
                    continue
                candidate = ".".join([entry[1] for entry in self._scope[: pos + 1]] + parts[1:])
                found = self._index.lookup(self._info.path, candidate)
                if found is not None:
                    return found
            return None
        module = self._info.import_alias.get(parts[0])
        if module is None and parts[0] in self._info.from_alias:
            parent, original = self._info.from_alias[parts[0]]
            if f"{parent}.{original}" in self._index.modules:
                module = f"{parent}.{original}"
        if module is None:
            return None
        expanded = module.split(".") + parts[1:]
        for end in range(len(expanded) - 1, 0, -1):
            module_name = ".".join(expanded[:end])
            path = self._index.modules.get(module_name)
            if path is None:
                continue
            found = self._index.lookup(path, ".".join(expanded[end:]))
            if found is not None:
                return found
        return None


def _snippet_callees(
    ref: FunctionRef,
    index: dict[tuple[str, str], FunctionRef],
    modules: dict[str, str],
    top: dict[str, list[FunctionRef]],
) -> set[FunctionRef]:
    """The callees of ``ref`` found by parsing ``ref.source`` alone (see ``subset_edges``)."""
    try:
        tree = ast.parse(ref.source)
    except SyntaxError:
        return set()
    visitor = _SnippetVisitor(ref, index, modules, top)
    visitor.visit(tree)
    return visitor.callees


class _SnippetVisitor(ast.NodeVisitor):
    """Collect the project callees of one function snippet (see ``subset_edges``)."""

    def __init__(
        self,
        ref: FunctionRef,
        index: dict[tuple[str, str], FunctionRef],
        modules: dict[str, str],
        top: dict[str, list[FunctionRef]],
    ) -> None:
        self._ref = ref
        self._index = index
        self._modules = modules
        self._top = top
        self.callees: set[FunctionRef] = set()
        self._scope: list[tuple[str, str]] = []
        self._prefix = ref.qualname.split(".")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append((_SCOPE_CLASS, node.name))
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_nested(node.name, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_nested(node.name, node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._resolve_bare(node.func.id)
        else:
            parts = _dotted_name(node.func)
            if parts is not None:
                self._resolve_dotted(parts)
        self.generic_visit(node)

    def _visit_nested(self, name: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope.append((_SCOPE_FUNCTION, name))
        self.generic_visit(node)
        self._scope.pop()

    def _owner(self, suffix: list[str]) -> str:
        return ".".join(self._prefix + suffix)

    def _resolve_bare(self, name: str) -> None:
        names = [entry[1] for entry in self._scope]
        for end in range(len(names), -1, -1):
            if end > 0 and self._scope[end - 1][0] != _SCOPE_FUNCTION:
                continue
            found = self._index.get((self._ref.path, self._owner([*names[:end], name])))
            if found is not None:
                self.callees.add(found)
                return
        if (self._ref.path, name) in self._index:
            self.callees.add(self._index[(self._ref.path, name)])
            return
        for callee in self._top.get(name, []):
            if callee.path != self._ref.path:
                self.callees.add(callee)

    def _resolve_dotted(self, parts: list[str]) -> None:
        if parts[0] == "self":
            for end in range(len(self._prefix) - 1, 0, -1):
                candidate = ".".join(self._prefix[:end] + parts[1:])
                if (self._ref.path, candidate) in self._index:
                    self.callees.add(self._index[(self._ref.path, candidate)])
            return
        for end in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:end])
            path = self._modules.get(module_name)
            if path is None:
                continue
            found = self._index.get((path, ".".join(parts[end:])))
            if found is not None:
                self.callees.add(found)
                return


__all__ = ["call_graph", "subset_edges"]
