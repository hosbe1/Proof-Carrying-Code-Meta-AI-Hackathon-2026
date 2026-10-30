"""Tests for ``pcc.sandbox.run_python``: the examples and edge cases from the ticket."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from pcc.core import ToolError
from pcc.sandbox import run_python
from pcc.sandbox.loader import target_loader
from tests.helpers import function_ref

PALINDROME_SOURCE = """\
def is_palindrome_recursive(s: str) -> bool:
    if len(s) <= 1:
        return True
    return s[0] == s[-1] and is_palindrome_recursive(s[1:-1])
"""

PALINDROME_MODULE = f'"""Demo module for the sandbox tests."""\n\n\n{PALINDROME_SOURCE}'


def test_print_returns_ok(tmp_path: Path) -> None:
    outcome = run_python("print(1)", cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.returncode == 0
    assert outcome.stdout == "1\n"
    assert outcome.stderr == ""
    assert outcome.timed_out is False
    assert outcome.seconds >= 0.0


def test_timeout_kills_the_process(tmp_path: Path) -> None:
    start = time.monotonic()
    outcome = run_python("while True:\n    pass", cwd=tmp_path, timeout_seconds=1)
    elapsed = time.monotonic() - start
    assert outcome.timed_out is True
    assert outcome.ok is False
    assert elapsed < 3


def test_environment_is_an_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "x")
    monkeypatch.setenv("PCC_SANDBOX_PROBE", "secret")
    script = (
        "import os\n"
        "print(repr(os.environ.get('META_API_KEY')))\n"
        "print(repr(os.environ.get('PCC_SANDBOX_PROBE')))\n"
        "print(os.environ.get('PYTHONHASHSEED'))\n"
        "print('PATH' in os.environ)\n"
    )
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout.splitlines() == ["None", "None", "0", "True"]


def test_palindrome_end_to_end(tmp_path: Path) -> None:
    package = tmp_path / "strings"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "palindrome.py").write_text(PALINDROME_MODULE, encoding="utf-8")
    function = function_ref(
        source=PALINDROME_SOURCE, path="strings/palindrome.py", qualname="is_palindrome_recursive", lineno=4
    )
    script = target_loader(function) + '\nprint(target("aba") is True)\n'
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "True\n"


def test_invalid_utf8_is_replaced(tmp_path: Path) -> None:
    script = "import sys\nsys.stdout.buffer.write(b'\\xff\\xfe\\n')\nsys.stdout.buffer.flush()\n"
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "\ufffd\ufffd\n"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_is_killed_with_the_group(tmp_path: Path) -> None:
    script = (
        "import os\n"
        "import time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(60)\n"
        "else:\n"
        "    print('parent done')\n"
    )
    start = time.monotonic()
    outcome = run_python(script, cwd=tmp_path, timeout_seconds=10)
    elapsed = time.monotonic() - start
    assert outcome.ok is True
    assert "parent done" in outcome.stdout
    assert elapsed < 10


@pytest.mark.skipif(not hasattr(os, "fork") or not hasattr(os, "setsid"), reason="needs os.fork and os.setsid")
def test_setsid_grandchild_does_not_hold_pipes_open(tmp_path: Path) -> None:
    script = (
        "import os\n"
        "import time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.setsid()\n"
        "    time.sleep(60)\n"
        "else:\n"
        "    print('parent done')\n"
    )
    start = time.monotonic()
    outcome = run_python(script, cwd=tmp_path, timeout_seconds=10)
    elapsed = time.monotonic() - start
    assert outcome.ok is True
    assert "parent done" in outcome.stdout
    assert elapsed < 10


@pytest.mark.skipif(not hasattr(os, "fork") or not hasattr(os, "setsid"), reason="needs os.fork and os.setsid")
def test_setsid_timeout_still_returns(tmp_path: Path) -> None:
    script = (
        "import os\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.setsid()\n"
        "    while True:\n"
        "        pass\n"
        "else:\n"
        "    while True:\n"
        "        pass\n"
    )
    start = time.monotonic()
    outcome = run_python(script, cwd=tmp_path, timeout_seconds=1)
    elapsed = time.monotonic() - start
    assert outcome.timed_out is True
    assert outcome.ok is False
    assert elapsed < 10


def test_missing_cwd_raises_tool_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        run_python("print(1)", cwd=tmp_path / "missing")


def test_file_as_cwd_raises_tool_error(tmp_path: Path) -> None:
    marker = tmp_path / "file.txt"
    marker.write_text("x", encoding="utf-8")
    with pytest.raises(ToolError):
        run_python("print(1)", cwd=marker)


@pytest.mark.skipif(sys.platform != "linux", reason="the RLIMIT_AS memory limit is set on Linux only")
def test_memory_limit(tmp_path: Path) -> None:
    outcome = run_python("bytearray(2_000_000_000)", cwd=tmp_path, memory_mb=256)
    assert outcome.ok is False


def test_nonzero_exit_is_not_ok(tmp_path: Path) -> None:
    outcome = run_python("import sys\nsys.exit(3)", cwd=tmp_path)
    assert outcome.ok is False
    assert outcome.returncode == 3
    assert outcome.timed_out is False


def test_stdout_is_cut_off_at_1mb(tmp_path: Path) -> None:
    outcome = run_python('import sys\nsys.stdout.write("x" * 2_000_000)', cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "x" * 1_048_576


def test_stderr_is_cut_off_at_1mb(tmp_path: Path) -> None:
    outcome = run_python('import sys\nsys.stderr.write("e" * 2_000_000)', cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stderr == "e" * 1_048_576


def test_stderr_is_captured(tmp_path: Path) -> None:
    outcome = run_python("import sys\nsys.stderr.write('boom\\n')", cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stderr == "boom\n"


def test_cwd_and_src_are_on_sys_path(tmp_path: Path) -> None:
    (tmp_path / "helper_mod_xyz.py").write_text("VALUE = 42\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "srcmod_xyz.py").write_text("VALUE = 7\n", encoding="utf-8")
    script = "import helper_mod_xyz\nimport srcmod_xyz\nprint(helper_mod_xyz.VALUE + srcmod_xyz.VALUE)"
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "49\n"


def test_pythonpath_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "ppmod_xyz.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    script = "try:\n    import ppmod_xyz\n    print('imported')\nexcept ImportError:\n    print('blocked')\n"
    outcome = run_python(script, cwd=tmp_path)
    assert outcome.ok is True
    assert outcome.stdout == "blocked\n"
