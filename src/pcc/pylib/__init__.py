"""Python's semantics rebuilt in Dafny, one building block per operation (docs/plan.md, step 4).

Each block lives in ``PyLib.dfy``, is proven against its own spec in Dafny, and is tested against CPython in
``tests/pylib``. The transpiler maps Python operations onto these blocks, so a new rule only lands together with the
blocks it needs.
"""

from __future__ import annotations

from pathlib import Path

PYLIB_DIR = Path(__file__).parent
PYLIB_FILE = PYLIB_DIR / "PyLib.dfy"

__all__ = ["PYLIB_DIR", "PYLIB_FILE"]
