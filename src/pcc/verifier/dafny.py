"""Running the Dafny CLI: verifying, compiling back to Python, parsing errors. Filled in by a later ticket."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pcc.core import DafnyError


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    verified: bool
    errors: tuple[DafnyError, ...]
    seconds: float


def resolve(dafny_code: str, *, timeout_seconds: float = 120.0) -> tuple[DafnyError, ...]:
    """Parse the errors Dafny reports for ``dafny_code``, without deciding verified or not."""
    raise NotImplementedError


def verify(
    dafny_code: str,
    *,
    assumptions_module: str = "",
    resource_limit: int,
    timeout_seconds: float | None = None,
    extra_modules: Sequence[str] = (),
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


__all__ = ["VerifyOutcome", "build_python", "resolve", "verify"]
