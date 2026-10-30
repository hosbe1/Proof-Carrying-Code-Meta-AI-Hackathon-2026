"""Step 4: the rule-based transpiler (docs/plan.md, step 4). What it supports, it translates; anything else
raises ``UnsupportedConstruct`` and the pipeline falls back to AI translation."""

from __future__ import annotations

from pcc.transpile.code import transpile_function
from pcc.transpile.spec import transpile_spec

__all__ = ["transpile_function", "transpile_spec"]
