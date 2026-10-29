"""Tests for ``pcc.sandbox.loader.target_loader``: binding the function under test to ``target``."""

from __future__ import annotations

from pathlib import Path

from pcc.sandbox import run_python
from pcc.sandbox.loader import target_loader
from tests.helpers import function_ref


def _write(tmp_path: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def test_module_path_comes_from_the_file_path() -> None:
    function = function_ref(source="def f(): ...\n", path="strings/palindrome.py", qualname="f")
    assert "importlib.import_module('strings.palindrome')" in target_loader(function)


def test_leading_src_is_dropped_from_the_module_path() -> None:
    function = function_ref(source="def f(): ...\n", path="src/cart.py", qualname="f")
    assert "importlib.import_module('cart')" in target_loader(function)


def test_top_level_function(tmp_path: Path) -> None:
    _write(tmp_path, {"calc.py": "def add(a: int, b: int) -> int:\n    return a + b\n"})
    function = function_ref(
        source="def add(a: int, b: int) -> int:\n    return a + b\n", path="calc.py", qualname="add"
    )
    outcome = run_python(target_loader(function) + "\nprint(target(2, 3))\n", cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "5\n"


def test_method_uses_attribute_access(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "src/cart.py": (
                "class Cart:\n"
                "    def __init__(self, total: int) -> None:\n"
                "        self.total = total\n"
                "    def with_fee(self, fee: int) -> int:\n"
                "        return self.total + fee\n"
            )
        },
    )
    function = function_ref(
        source="def with_fee(self, fee: int) -> int:\n    return self.total + fee\n",
        path="src/cart.py",
        qualname="Cart.with_fee",
    )
    script = "from cart import Cart\n" + target_loader(function) + "\nprint(target(Cart(40), 2))\n"
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "42\n"


def test_nested_function_execs_its_source(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "nest.py": (
                "def outer(x: int) -> int:\n    def inner(y: int) -> int:\n        return y * 2\n    return inner(x)\n"
            )
        },
    )
    function = function_ref(
        source="def inner(y: int) -> int:\n    return y * 2\n", path="nest.py", qualname="outer.inner"
    )
    outcome = run_python(target_loader(function) + "\nprint(target(21))\n", cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "42\n"


def test_locals_marker_is_stripped(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "nest.py": (
                "def outer(x: int) -> int:\n    def inner(y: int) -> int:\n        return y * 2\n    return inner(x)\n"
            )
        },
    )
    function = function_ref(
        source="def inner(y: int) -> int:\n    return y * 2\n",
        path="nest.py",
        qualname="outer.<locals>.inner",
    )
    outcome = run_python(target_loader(function) + "\nprint(target(21))\n", cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "42\n"


def test_named_lambda(tmp_path: Path) -> None:
    _write(tmp_path, {"scale.py": "triple = lambda x: 3 * x\n"})
    function = function_ref(source="triple = lambda x: 3 * x\n", path="scale.py", qualname="triple")
    outcome = run_python(target_loader(function) + "\nprint(target(14))\n", cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "42\n"
