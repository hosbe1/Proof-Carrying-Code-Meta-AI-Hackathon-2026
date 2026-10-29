from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from pcc.core import FunctionRef, source_hash
from pcc.pylib import PYLIB_FILE
from pcc.verifier.layout import assemble, method_name, module_name


def make_ref(path: str, qualname: str) -> FunctionRef:
    source = "def placeholder():\n    return 0\n"
    return FunctionRef(path=path, qualname=qualname, lineno=1, source=source, source_hash=source_hash(source))


def expected_module_name(path: str, qualname: str) -> str:
    stem = path[:-3] if path.endswith(".py") else path
    safe = re.sub(r"[^A-Za-z0-9]", "_", stem)
    safe_qual = re.sub(r"[^A-Za-z0-9]", "_", qualname)
    digest = hashlib.sha256(f"{path}:{qualname}".encode()).hexdigest()[:8]
    return f"PCC_{safe}__{safe_qual}_{digest}"


def test_module_name_has_prefix_and_hex_suffix() -> None:
    assert module_name(make_ref("strings/palindrome.py", "is_palindrome_recursive")).startswith(
        "PCC_strings_palindrome__is_palindrome_recursive_"
    )
    name = module_name(make_ref("strings/palindrome.py", "is_palindrome_recursive"))
    assert re.fullmatch(r"PCC_strings_palindrome__is_palindrome_recursive_[0-9a-f]{8}", name)


def test_module_name_matches_formula() -> None:
    assert module_name(make_ref("strings/palindrome.py", "is_palindrome_recursive")) == expected_module_name(
        "strings/palindrome.py", "is_palindrome_recursive"
    )


def test_module_name_distinguishes_slashed_and_underscored_paths() -> None:
    slashed = module_name(make_ref("a/b.py", "f"))
    underscored = module_name(make_ref("a_b.py", "f"))
    assert slashed != underscored
    assert slashed.startswith("PCC_a_b__f_")
    assert underscored.startswith("PCC_a_b__f_")


def test_method_name_replaces_dots_and_prefixes_leading_underscore() -> None:
    assert method_name(make_ref("cart.py", "Cart.total")) == "Cart_total"
    assert method_name(make_ref("util.py", "_helper")) == "pcc__helper"
    assert method_name(make_ref("util.py", "f")) == "f"


def test_method_name_keeps_dotted_leading_underscore() -> None:
    assert method_name(make_ref("cart.py", "Cart._total")) == "Cart__total"


def test_module_name_sanitizes_dash_and_unicode() -> None:
    name = module_name(make_ref("my-dir/möd-file.py", "Cart.total"))
    assert name.startswith("PCC_my_dir_m_d_file__Cart_total_")
    assert re.fullmatch(r"[A-Za-z0-9_]+", name)


def test_assemble_without_lemmas_has_no_blank_section() -> None:
    function = make_ref("strings/palindrome.py", "is_pal")
    spec = "predicate is_pal_spec(x: int)\n{\n  x >= 0\n}"
    code = "function is_pal(x: int): int\n  requires is_pal_spec(x)\n{\n  x + 1\n}"
    assembled = assemble(function, spec, code)
    assert assembled.startswith(f"module {module_name(function)} {{\n")
    assert "\n  import opened PyLib\n" in assembled
    assert "include" not in assembled
    assert "\n\n\n" not in assembled
    lines = assembled.split("\n")
    assert lines[0] == f"module {module_name(function)} {{"
    assert lines[1] == "  import opened PyLib"
    assert lines[-1] == ""
    assert lines[-2] == "}"
    body = "\n".join(lines[2:-2])
    assert "predicate is_pal_spec(x: int)" in body
    assert "function is_pal(x: int): int" in body
    for line in body.split("\n"):
        if line.strip():
            assert line.startswith("  ")


def test_assemble_indents_lemmas_when_given() -> None:
    function = make_ref("strings/palindrome.py", "is_pal")
    spec = "predicate is_pal_spec(x: int)\n{\n  x >= 0\n}"
    code = "function is_pal(x: int): int\n  requires is_pal_spec(x)\n{\n  x + 1\n}"
    lemmas = "lemma helper(x: int)\n  requires is_pal_spec(x)\n{\n}"
    assembled = assemble(function, spec, code, lemmas=lemmas)
    assert "\n  lemma helper(x: int)\n    requires is_pal_spec(x)\n  {\n  }\n" in assembled


@pytest.mark.dafny
def test_assemble_result_resolves_with_pylib(tmp_path: Path) -> None:
    function = make_ref("strings/palindrome.py", "is_pal")
    name = method_name(function)
    spec = f"predicate {name}_spec(x: int)\n{{\n  x >= 0\n}}"
    code = f"function {name}(x: int): int\n  requires {name}_spec(x)\n{{\n  x + 1\n}}"
    target = tmp_path / "check.dfy"
    target.write_text(assemble(function, spec, code), encoding="utf-8")
    proc = subprocess.run(
        ["dafny", "resolve", str(target), str(PYLIB_FILE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
