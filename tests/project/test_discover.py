"""Tests for ``pcc.project.discover``: finding the functions a repository defines."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pcc.core import source_hash
from pcc.project import discover


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_top_level_function_and_method(tmp_path: Path) -> None:
    """Example 1: a plain function and a method give ``["f", "C.m"]``."""
    _write(tmp_path, "mod.py", "def f():\n    return 1\n\n\nclass C:\n    def m(self):\n        return 2\n")
    refs = discover(tmp_path, ["mod.py"])
    assert isinstance(refs, tuple)
    assert [ref.qualname for ref in refs] == ["f", "C.m"]
    assert refs[0].path == "mod.py"
    assert refs[0].lineno == 1
    assert refs[0].source == "def f():\n    return 1"
    assert refs[0].source_hash == source_hash(refs[0].source)
    assert refs[1].lineno == 6
    assert refs[1].source == "def m(self):\n    return 2"
    assert refs[1].source_hash == source_hash(refs[1].source)


def test_nested_function_without_free_variables(tmp_path: Path) -> None:
    """Example 2a: ``inner`` uses only its own argument, so both are found."""
    _write(tmp_path, "mod.py", "def outer():\n    def inner(x):\n        return x\n    return inner\n")
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["outer", "outer.inner"]


def test_nested_function_with_free_variables_is_skipped(tmp_path: Path) -> None:
    """Example 2b: ``inner`` captures ``n``, so only ``outer`` is found."""
    _write(tmp_path, "mod.py", "def outer(n):\n    def inner():\n        return n\n    return inner\n")
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["outer"]


def test_nested_function_may_use_globals_and_builtins(tmp_path: Path) -> None:
    """Globals and builtins are not free variables, so ``inner`` is still found."""
    _write(
        tmp_path,
        "mod.py",
        "LIMIT = 10\n\n\ndef outer():\n    def inner(x):\n        return min(x, LIMIT)\n    return inner\n",
    )
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["outer", "outer.inner"]


def test_named_lambda(tmp_path: Path) -> None:
    """Example 3: ``square = lambda ...`` is found under the bound name."""
    _write(tmp_path, "mod.py", "square = lambda x: x * x\n")
    refs = discover(tmp_path, ["mod.py"])
    assert [ref.qualname for ref in refs] == ["square"]
    assert refs[0].lineno == 1
    assert refs[0].source == "square = lambda x: x * x"
    assert refs[0].source_hash == source_hash(refs[0].source)


def test_annotated_lambda_assignment(tmp_path: Path) -> None:
    """``name: T = lambda ...`` is found under the bound name."""
    _write(tmp_path, "mod.py", "square: int = lambda x: x * x\n")
    refs = discover(tmp_path, ["mod.py"])
    assert [ref.qualname for ref in refs] == ["square"]
    assert refs[0].source == "square: int = lambda x: x * x"


def test_unbound_lambdas_are_skipped(tmp_path: Path) -> None:
    """Lambdas passed as arguments or returned without a name are left out."""
    _write(
        tmp_path,
        "mod.py",
        "def apply(xs):\n    return sorted(xs, key=lambda x: -x)\n\n\ndef make():\n    return lambda: 1\n",
    )
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["apply", "make"]


def test_named_lambda_with_free_variables_is_skipped(tmp_path: Path) -> None:
    """A bound lambda that captures an outer variable is checked with its enclosing function."""
    _write(tmp_path, "mod.py", "def outer(n):\n    f = lambda: n\n    return f()\n")
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["outer"]


def test_decorated_function(tmp_path: Path) -> None:
    """The source starts at the decorator while ``lineno`` is the ``def`` line."""
    _write(
        tmp_path,
        "mod.py",
        "from functools import cache\n\n\n@cache\ndef fib(n):\n    return n\n",
    )
    refs = discover(tmp_path, ["mod.py"])
    assert [ref.qualname for ref in refs] == ["fib"]
    assert refs[0].lineno == 5
    assert refs[0].source == "@cache\ndef fib(n):\n    return n"


def test_async_and_generator_functions_are_found(tmp_path: Path) -> None:
    """``async def`` and generators are discovered; the pipeline sorts them out."""

    _write(
        tmp_path,
        "mod.py",
        "async def fetch(url):\n    return url\n\n\ndef gen(n):\n    yield n\n",
    )
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["fetch", "gen"]


def test_deeply_nested_qualname_has_no_locals(tmp_path: Path) -> None:
    """``__qualname__`` has every ``<locals>.`` removed, through classes too."""
    _write(
        tmp_path,
        "mod.py",
        "def outer():\n    class C:\n        def m(self):\n            return 1\n    return C\n",
    )
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["outer", "outer.C.m"]


def test_nested_function_in_method(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        "class C:\n    def m(self):\n        def h(x):\n            return x\n        return h(1)\n",
    )
    assert [ref.qualname for ref in discover(tmp_path, ["mod.py"])] == ["C.m", "C.m.h"]


def test_path_outside_root_raises(tmp_path: Path) -> None:
    """A path that escapes ``repo_root`` raises ``ValueError``."""
    _write(tmp_path, "sub/mod.py", "def f():\n    return 1\n")
    with pytest.raises(ValueError):
        discover(tmp_path, ["../outside.py"])
    with pytest.raises(ValueError):
        discover(tmp_path, [str(tmp_path.parent / "sibling.py")])


def test_path_staying_inside_root_with_dotdot(tmp_path: Path) -> None:
    """``..`` that resolves back inside the root is fine."""
    _write(tmp_path, "sub/mod.py", "def f():\n    return 1\n")
    assert [ref.qualname for ref in discover(tmp_path, ["sub/../sub/mod.py"])] == ["f"]


def test_empty_directory_gives_empty_tuple(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert discover(tmp_path, ["empty"]) == ()
    assert discover(tmp_path, []) == ()


def test_redefinition_last_wins_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A qualname defined twice keeps the last definition and logs a warning."""
    _write(tmp_path, "mod.py", "def f():\n    return 1\n\n\ndef f():\n    return 2\n")
    with caplog.at_level(logging.WARNING):
        refs = discover(tmp_path, ["mod.py"])
    assert [ref.qualname for ref in refs] == ["f"]
    assert refs[0].lineno == 5
    assert refs[0].source == "def f():\n    return 2"
    assert any("f" in record.getMessage() for record in caplog.records)


def test_property_setter_wins_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A ``@property`` getter and setter share one qualname; the setter wins."""
    _write(
        tmp_path,
        "mod.py",
        "class C:\n    @property\n    def x(self):\n        return self._x\n\n"
        "    @x.setter\n    def x(self, value):\n        self._x = value\n",
    )
    with caplog.at_level(logging.WARNING):
        refs = discover(tmp_path, ["mod.py"])
    assert [ref.qualname for ref in refs] == ["C.x"]
    assert refs[0].lineno == 7
    assert refs[0].source.startswith("@x.setter\ndef x(self, value):")
    assert any("C.x" in record.getMessage() for record in caplog.records)


def test_same_qualname_in_two_files_is_kept(tmp_path: Path) -> None:
    """The last-definition rule applies per file, not across files."""
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    _write(tmp_path, "b.py", "def f():\n    return 2\n")
    refs = discover(tmp_path, ["a.py", "b.py"])
    assert [(ref.path, ref.qualname) for ref in refs] == [("a.py", "f"), ("b.py", "f")]


def test_source_hash_matches_for_lf_and_crlf(tmp_path: Path) -> None:
    """LF and CRLF checkouts of the same file give the same source and hash."""
    text = "def f():\n    return 1\n\n\nclass C:\n    def m(self):\n        return 2\n"
    _write(tmp_path, "lf/mod.py", text)
    crlf = tmp_path / "crlf" / "mod.py"
    crlf.parent.mkdir(parents=True)
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    lf_refs = discover(tmp_path, ["lf"])
    crlf_refs = discover(tmp_path, ["crlf"])
    assert [ref.qualname for ref in lf_refs] == [ref.qualname for ref in crlf_refs] == ["f", "C.m"]
    for lf_ref, crlf_ref in zip(lf_refs, crlf_refs, strict=True):
        assert lf_ref.source == crlf_ref.source
        assert lf_ref.source_hash == crlf_ref.source_hash
        assert "\r" not in crlf_ref.source


def test_syntax_error_is_skipped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A file that does not parse is skipped; the other files are still found."""
    _write(tmp_path, "bad.py", "def broken(:\n")
    _write(tmp_path, "good.py", "def fine():\n    return 1\n")
    with caplog.at_level(logging.WARNING):
        refs = discover(tmp_path, ["."])
    assert [ref.qualname for ref in refs] == ["fine"]
    assert any("bad.py" in record.getMessage() for record in caplog.records)


def test_directory_walk_skips_ignored_directories(tmp_path: Path) -> None:
    """Recursion skips hidden, environment, build and proof directories."""
    _write(tmp_path, "pkg/__init__.py", "def init_fn():\n    return 0\n")
    _write(tmp_path, "pkg/mod.py", "def mod_fn():\n    return 1\n")
    _write(tmp_path, "pkg/sub/deep.py", "def deep_fn():\n    return 2\n")
    _write(tmp_path, "pkg/notes.txt", "def not_python():\n    return 3\n")
    for skipped in (".hidden", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", "proofs"):
        _write(tmp_path, f"{skipped}/skipped.py", "def skipped_fn():\n    return -1\n")
    refs = discover(tmp_path, ["."])
    assert [(ref.path, ref.qualname) for ref in refs] == [
        ("pkg/__init__.py", "init_fn"),
        ("pkg/mod.py", "mod_fn"),
        ("pkg/sub/deep.py", "deep_fn"),
    ]


def test_results_are_ordered_by_path_then_lineno(tmp_path: Path) -> None:
    _write(tmp_path, "b.py", "def zed():\n    return 1\n\n\ndef alpha():\n    return 2\n")
    _write(tmp_path, "a.py", "def mid():\n    return 3\n")
    refs = discover(tmp_path, ["b.py", "a.py"])
    assert [(ref.path, ref.qualname, ref.lineno) for ref in refs] == [
        ("a.py", "mid", 1),
        ("b.py", "zed", 1),
        ("b.py", "alpha", 5),
    ]


def test_method_source_is_dedented(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "class C:\n    def m(self):\n        x = 1\n        return x\n")
    (ref,) = discover(tmp_path, ["mod.py"])
    assert ref.source == "def m(self):\n    x = 1\n    return x"
    assert "\r" not in ref.source
