"""Python's semantics rebuilt in Dafny, one building block per operation (docs/plan.md, step 4).

Each block lives in ``PyLib.dfy``, is proven against its own spec in Dafny, and is tested against CPython in
``tests/pylib``. The transpiler maps Python operations onto these blocks, so a new rule only lands together with the
blocks it needs.
"""

from __future__ import annotations

from pathlib import Path

PYLIB_DIR = Path(__file__).parent
PYLIB_FILE = PYLIB_DIR / "PyLib.dfy"


def signatures() -> str:
    """The ``function`` headers with their ``requires``/``ensures`` of every PyLib block, for prompts.

    Read from ``PyLib.dfy`` so prompts always quote the proven specs: each block contributes its
    ``function`` line plus every following header line up to the opening brace, dedented. Body lines,
    comments and anything else in the file are left out.
    """
    blocks: list[str] = []
    current: list[str] = []
    for raw in PYLIB_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not current:
            if line.startswith("function "):
                current = [line]
        elif line.startswith("{"):
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(f"  {line}")
    if current:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks)


__all__ = ["PYLIB_DIR", "PYLIB_FILE", "signatures"]
