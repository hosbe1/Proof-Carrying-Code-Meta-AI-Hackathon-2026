"""The verifier interface ``verify`` with Dafny as its only backend (docs/plan.md, "Result states").

Also the single place that runs the dafny CLI: verifying, compiling back to Python for the equivalence checks, and the
anti-cheat scan. Other modules never call dafny directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pcc.core import DafnyError


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    verified: bool
    errors: tuple[DafnyError, ...]
    seconds: float


def verify(
    dafny_code: str,
    *,
    assumptions_module: str = "",
    resource_limit: int,
    timeout_seconds: float | None = None,
) -> VerifyOutcome:
    """Run ``dafny verify`` on ``dafny_code`` with ``pcc.pylib`` and parse the result.

    The command is ``dafny verify --resource-limit <resource_limit> --verification-time-limit 0``. Pass
    ``Config().dafny.resource_limit``.

    - Never pass ``--allow-warnings``. Dafny exits 2 on warnings without it, so a warning such as a body-less lemma or
      a plain ``assume`` is not verified either.
    - Any non-zero exit code means not verified. Exit 0 is the only way to be verified.
    - ``--verification-time-limit 0`` disables Dafny's own per-batch time limit (``dafny verify --help``: "A value of 0
      indicates no limit"; checked with Dafny 4.11). The resource limit alone decides, so the result does not depend on
      the machine.
    - ``timeout_seconds`` is only a process-level emergency stop for a hung dafny process. A process stopped by it is
      not verified; the timeout is not part of the result.
    - ``assumptions_module`` names the module that may hold library contracts, the same module ``anti_cheat`` takes. A
      violation found by ``anti_cheat`` with it counts as not verified.
    """
    raise NotImplementedError


def build_python(dafny_code: str, out_dir: Path) -> Path:
    """Compile ``dafny_code`` with ``dafny build --target:py`` into ``out_dir`` and return the module directory."""
    raise NotImplementedError


def anti_cheat(dafny_code: str, frozen_spec: str, *, assumptions_module: str = "") -> tuple[str, ...]:
    """Return every violation in ``dafny_code``; an empty tuple means it passes.

    Outside ``assumptions_module`` every one of these forms is a violation (an empty ``assumptions_module`` means no
    module may hold contracts):

    - ``assume``, with or without attributes (``assume {:axiom} false;`` included);
    - ``{:axiom}`` and ``@Axiom`` in any position;
    - body-less lemmas, functions and methods (Dafny warns on them, and they are contracts in disguise);
    - ``{:verify false}`` (Dafny 4.11 has no ``@`` spelling of it: ``@VerifyFalse`` does not resolve);
    - ``{:only}`` and ``@VerifyOnly``, which disable the verification of every other member of the file;
    - ``{:extern}``, a body-less method that Dafny takes on trust;
    - any change to ``frozen_spec``: the frozen spec must still appear unchanged in ``dafny_code``.

    Inside ``assumptions_module`` the only allowed form is a contract: a body-less lemma or function marked ``{:axiom}``
    or ``@Axiom``. Plain ``assume`` statements are reported everywhere, the module included.

    The forms and what Dafny 4.11 does with each are fixed in ``tests/verifier/cheats``.
    """
    raise NotImplementedError


__all__ = ["VerifyOutcome", "anti_cheat", "build_python", "verify"]
