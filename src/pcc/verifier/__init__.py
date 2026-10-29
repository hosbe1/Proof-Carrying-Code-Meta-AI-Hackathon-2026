"""The verifier interface ``verify`` with Dafny as its only backend (docs/plan.md, "Result states").

Also the single place that runs the dafny CLI: verifying, compiling back to Python for the equivalence checks, and the
anti-cheat scan. Other modules never call dafny directly.
"""

from __future__ import annotations

from pcc.verifier.anti_cheat import anti_cheat
from pcc.verifier.dafny import VerifyOutcome, build_python, resolve, verify
from pcc.verifier.ghost import executable_part, same_executable
from pcc.verifier.layout import assemble, method_name, module_name

__all__ = [
    "VerifyOutcome",
    "anti_cheat",
    "assemble",
    "build_python",
    "executable_part",
    "method_name",
    "module_name",
    "resolve",
    "same_executable",
    "verify",
]
