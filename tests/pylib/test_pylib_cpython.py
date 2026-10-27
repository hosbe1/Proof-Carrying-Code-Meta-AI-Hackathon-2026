"""Every PyLib block, compiled to Python with Dafny, must behave exactly like CPython."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pcc.config import Config
from pcc.pylib import PYLIB_FILE

pytestmark = pytest.mark.dafny

nonzero = st.integers().filter(lambda b: b != 0)

# Modules imported by the Dafny-generated Python code. Removed after the fixture so that a later build in the same
# process does not reuse them.
_GENERATED_MODULES = ("PyLib", "_dafny", "System_", "module_")


@pytest.fixture(scope="module")
def pylib(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    out = tmp_path_factory.mktemp("pylib") / "out"
    subprocess.run(
        ["dafny", "build", "--target:py", "--no-verify", "-o", str(out), str(PYLIB_FILE)],
        check=True,
        capture_output=True,
    )
    module_dir = Path(f"{out}-py")
    sys.path.insert(0, str(module_dir))
    try:
        yield importlib.import_module("PyLib").default__
    finally:
        sys.path.remove(str(module_dir))
        for name in _GENERATED_MODULES:
            sys.modules.pop(name, None)


def test_verifies() -> None:
    limit = str(Config().dafny.resource_limit)
    proc = subprocess.run(
        ["dafny", "verify", "--resource-limit", limit, "--verification-time-limit", "0", str(PYLIB_FILE)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@settings(max_examples=2000)
@given(a=st.integers(), b=nonzero)
def test_floordiv_matches_cpython(pylib: ModuleType, a: int, b: int) -> None:
    assert pylib.py__floordiv(a, b) == a // b


@settings(max_examples=2000)
@given(a=st.integers(), b=nonzero)
def test_mod_matches_cpython(pylib: ModuleType, a: int, b: int) -> None:
    assert pylib.py__mod(a, b) == a % b


@pytest.mark.parametrize(("a", "b"), [(7, -2), (-7, 2), (-7, -2), (0, -3), (6, -3), (-10, -11)])
def test_known_differences_from_dafny_division(pylib: ModuleType, a: int, b: int) -> None:
    assert (pylib.py__floordiv(a, b), pylib.py__mod(a, b)) == (a // b, a % b)
