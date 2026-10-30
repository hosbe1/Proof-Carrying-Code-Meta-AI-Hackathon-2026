"""Step 4, checks: is the Dafny translation the same function as the original? (docs/plan.md, step 4)"""

from __future__ import annotations

from pcc.equivalence.check import check
from pcc.equivalence.spec_check import check_spec

__all__ = ["check", "check_spec"]
