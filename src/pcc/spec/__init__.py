"""Step 3: the spec as an executable Python predicate, tested before anything is proven (docs/plan.md, step 3)."""

from __future__ import annotations

from pcc.spec.check import check, spec_cases
from pcc.spec.mutants import rule_mutants
from pcc.spec.write import write

__all__ = ["check", "rule_mutants", "spec_cases", "write"]
