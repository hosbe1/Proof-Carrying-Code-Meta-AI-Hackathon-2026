"""Tests for ``call_graph``, ``affected`` and ``call_order`` with temporary repositories.

``call_graph`` and ``affected`` read plain directories (no git needed); the diff-to-order
chain is covered through ``discover`` plus ``call_order`` on the same files.
"""

from __future__ import annotations

from pathlib import Path

from pcc.core import FunctionRef
from pcc.project import affected, call_graph, call_order
from pcc.project.discover import discover


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _by_qualname(repo: Path, rel: str) -> dict[str, FunctionRef]:
    return {ref.qualname: ref for ref in discover(repo, [rel])}


def _edges(repo: Path, rel: str) -> dict[str, frozenset[str]]:
    """Call graph of one file as ``qualname -> {callee qualname}``, ignoring paths."""
    refs = discover(repo, [rel])
    graph = call_graph(repo, refs)
    return {caller.qualname: frozenset(callee.qualname for callee in graph[caller]) for caller in refs}


def test_bare_call_in_same_module(tmp_path: Path) -> None:
    """A bare name defined in the same module resolves to that function."""
    _write(tmp_path, "mod.py", "def f():\n    return g()\n\n\ndef g():\n    return 1\n")
    assert _edges(tmp_path, "mod.py") == {"f": frozenset({"g"}), "g": frozenset()}


def test_from_import_resolves(tmp_path: Path) -> None:
    """``from m import g`` lets a bare ``g()`` resolve into the project module."""
    _write(tmp_path, "m.py", "def g():\n    return 1\n")
    _write(tmp_path, "a.py", "from m import g\n\n\ndef f():\n    return g()\n")
    refs = discover(tmp_path, ["a.py"])
    (callee,) = call_graph(tmp_path, refs)[refs[0]]
    assert (callee.path, callee.qualname) == ("m.py", "g")


def test_from_import_with_alias_resolves(tmp_path: Path) -> None:
    """``from m import g as h`` resolves the local alias ``h``."""
    _write(tmp_path, "m.py", "def g():\n    return 1\n")
    _write(tmp_path, "a.py", "from m import g as h\n\n\ndef f():\n    return h()\n")
    refs = discover(tmp_path, ["a.py"])
    (callee,) = call_graph(tmp_path, refs)[refs[0]]
    assert (callee.path, callee.qualname) == ("m.py", "g")


def test_module_attribute_resolves(tmp_path: Path) -> None:
    """``import m`` plus ``m.g()`` resolves into the project module."""
    _write(tmp_path, "m.py", "def g():\n    return 1\n")
    _write(tmp_path, "a.py", "import m\n\n\ndef f():\n    return m.g()\n")
    refs = discover(tmp_path, ["a.py"])
    (callee,) = call_graph(tmp_path, refs)[refs[0]]
    assert (callee.path, callee.qualname) == ("m.py", "g")


def test_module_alias_resolves(tmp_path: Path) -> None:
    """``import m as mm`` resolves ``mm.g()`` through the alias."""
    _write(tmp_path, "m.py", "def g():\n    return 1\n")
    _write(tmp_path, "a.py", "import m as mm\n\n\ndef f():\n    return mm.g()\n")
    refs = discover(tmp_path, ["a.py"])
    (callee,) = call_graph(tmp_path, refs)[refs[0]]
    assert (callee.path, callee.qualname) == ("m.py", "g")


def test_from_package_import_of_submodule_resolves(tmp_path: Path) -> None:
    """``from . import m`` plus ``m.g()`` resolves the submodule of the same package."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/m.py", "def g():\n    return 1\n")
    _write(tmp_path, "pkg/a.py", "from . import m\n\n\ndef f():\n    return m.g()\n")
    refs = discover(tmp_path, ["pkg/a.py"])
    (callee,) = call_graph(tmp_path, refs)[refs[0]]
    assert (callee.path, callee.qualname) == ("pkg/m.py", "g")


def test_self_call_resolves_to_same_class(tmp_path: Path) -> None:
    """``self.helper()`` resolves to the method of the same class."""
    _write(
        tmp_path,
        "mod.py",
        "class C:\n    def m(self):\n        return self.helper()\n\n    def helper(self):\n        return 1\n",
    )
    assert _edges(tmp_path, "mod.py") == {"C.m": frozenset({"C.helper"}), "C.helper": frozenset()}


def test_recursive_call_is_own_edge(tmp_path: Path) -> None:
    """Edge case: a recursive function (self-call) carries an edge to itself."""
    _write(tmp_path, "mod.py", "def f(n):\n    return f(n - 1) if n else 0\n")
    assert _edges(tmp_path, "mod.py") == {"f": frozenset({"f"})}


def test_other_calls_are_ignored(tmp_path: Path) -> None:
    """Builtins, unknown names, parameters, variables and library calls leave no edge."""
    _write(
        tmp_path,
        "mod.py",
        "import os\n\n\ndef f(cb, args):\n"
        "    n = len([1])\n"
        "    cb()\n"
        "    args.run()\n"
        "    os.getcwd()\n"
        "    return missing() + n\n",
    )
    assert _edges(tmp_path, "mod.py") == {"f": frozenset()}


def test_keys_are_exactly_the_passed_functions(tmp_path: Path) -> None:
    """The graph holds one entry per passed function, even with an empty callee set."""
    _write(tmp_path, "mod.py", "def f():\n    return g()\n\n\ndef g():\n    return 1\n")
    refs = _by_qualname(tmp_path, "mod.py")
    graph = call_graph(tmp_path, [refs["f"]])
    assert list(graph) == [refs["f"]]
    assert graph[refs["f"]] == frozenset({refs["g"]})


def test_closure_calls_belong_to_enclosing_function(tmp_path: Path) -> None:
    """A nested function capturing a variable is checked with its enclosing function."""
    _write(
        tmp_path,
        "mod.py",
        "def outer(n):\n    def inner():\n        return n + g()\n    return inner\n\n\ndef g():\n    return 1\n",
    )
    assert _edges(tmp_path, "mod.py") == {"outer": frozenset({"g"}), "g": frozenset()}


def _chain_repo(repo: Path) -> dict[str, FunctionRef]:
    """``h`` in m.py, ``g`` calling ``h`` in a.py, ``f`` calling ``g`` in b.py."""
    _write(repo, "m.py", "def h():\n    return 1\n")
    _write(repo, "a.py", "from m import h\n\n\ndef g():\n    return h()\n")
    _write(repo, "b.py", "from a import g\n\n\ndef f():\n    return g()\n")
    return {ref.qualname: ref for ref in discover(repo, ["."])}


def test_affected_is_transitive_callers(tmp_path: Path) -> None:
    """Example 3: changing only ``h`` gives ``affected == {h, g, f}``."""
    refs = _chain_repo(tmp_path)
    assert {ref.qualname for ref in affected(tmp_path, [refs["h"]])} == {"h", "g", "f"}


def test_affected_middle_function(tmp_path: Path) -> None:
    """Changing ``g`` affects ``g`` and its caller ``f``, but not its callee ``h``."""
    refs = _chain_repo(tmp_path)
    assert {ref.qualname for ref in affected(tmp_path, [refs["g"]])} == {"g", "f"}


def test_affected_top_function_is_itself(tmp_path: Path) -> None:
    """A function nobody calls affects only itself."""
    refs = _chain_repo(tmp_path)
    assert [ref.qualname for ref in affected(tmp_path, [refs["f"]])] == ["f"]


def test_affected_empty_is_empty(tmp_path: Path) -> None:
    """No changed functions means no affected functions."""
    _chain_repo(tmp_path)
    assert affected(tmp_path, []) == ()


def test_call_order_chain(tmp_path: Path) -> None:
    """Example 1: ``f`` calls ``g``, ``g`` calls ``h``: callees come before callers."""
    refs = _chain_repo(tmp_path)
    assert call_order([refs["f"], refs["g"], refs["h"]]) == (
        (refs["h"],),
        (refs["g"],),
        (refs["f"],),
    )


def test_call_order_mutual_recursion_shares_a_level(tmp_path: Path) -> None:
    """Example 2: ``a`` and ``b`` call each other, so they share one level."""
    _write(tmp_path, "mod.py", "def a():\n    return b()\n\n\ndef b():\n    return a()\n")
    refs = _by_qualname(tmp_path, "mod.py")
    assert call_order([refs["a"], refs["b"]]) == ((refs["a"], refs["b"]),)


def test_call_order_recursive_function_is_its_own_scc(tmp_path: Path) -> None:
    """Edge case: a self-call does not lift the function above its level."""
    _write(
        tmp_path,
        "mod.py",
        "def f(n):\n    return f(n - 1) if n else 0\n\n\ndef g():\n    return 1\n",
    )
    refs = _by_qualname(tmp_path, "mod.py")
    assert call_order([refs["f"], refs["g"]]) == ((refs["f"], refs["g"]),)


def test_call_order_diamond_levels_and_sorting(tmp_path: Path) -> None:
    """Levels count callee depth; functions in one level sort by path, then lineno."""
    _write(tmp_path, "m.py", "def k():\n    return 1\n")
    _write(
        tmp_path,
        "a.py",
        "from m import k\n\n\ndef h():\n    return k()\n\n\ndef g():\n    return k()\n"
        "\n\ndef f():\n    return g() + h()\n",
    )
    refs = {ref.qualname: ref for ref in discover(tmp_path, ["."])}
    assert call_order([refs["f"], refs["g"], refs["h"], refs["k"]]) == (
        (refs["k"],),
        (refs["h"], refs["g"]),
        (refs["f"],),
    )


def test_call_order_sorts_level_by_path_then_lineno(tmp_path: Path) -> None:
    """Two independent functions in one level sort by path before lineno."""
    _write(tmp_path, "b.py", "def zed():\n    return 1\n")
    _write(tmp_path, "a.py", "def mid():\n    return 1\n\n\ndef abc():\n    return 2\n")
    refs = list(discover(tmp_path, ["."]))
    (level,) = call_order(refs)
    assert [ref.qualname for ref in level] == ["mid", "abc", "zed"]


def test_call_order_empty_is_empty() -> None:
    """No functions means no levels."""
    assert call_order([]) == ()
