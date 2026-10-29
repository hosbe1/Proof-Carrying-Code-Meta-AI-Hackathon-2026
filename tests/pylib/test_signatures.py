"""signatures() must expose every PyLib block header for use in prompts."""

from __future__ import annotations

from pcc.pylib import PYLIB_FILE, signatures


def test_signatures_lists_every_block() -> None:
    sigs = signatures()
    for name in ("py_floordiv", "py_mod", "py_len", "py_index", "py_slice", "py_reversed"):
        # Generic blocks declare as `function py_len<T>(`, plain ones as `function py_mod(`.
        assert f"function {name}(" in sigs or f"function {name}<" in sigs
    assert "requires" in sigs
    assert "ensures" in sigs


def test_signatures_matches_dafny_source() -> None:
    source = PYLIB_FILE.read_text(encoding="utf-8")
    names = [
        line.split("(")[0].removeprefix("function ")
        for line in (raw.strip() for raw in source.splitlines())
        if line.startswith("function ")
    ]
    assert names != []
    sigs = signatures()
    assert sigs.count("function ") == len(names)
    for name in names:
        assert f"function {name}(" in sigs
