"""Running the Dafny CLI: resolving, verifying, compiling back to Python.

``resolve`` and ``verify`` run the Dafny CLI; ``build_python`` is filled in by a later ticket.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pcc.config import Config
from pcc.core import DafnyError, ToolError
from pcc.pylib import PYLIB_FILE
from pcc.verifier.anti_cheat import anti_cheat
from pcc.verifier.errors import parse_errors

#: ``dafny_code`` is written to this file in the temporary directory ``dafny resolve`` runs in.
_MAIN_NAME = "main.dfy"

#: ``PyLib.dfy`` is copied under this name next to it, so the checked code can include it.
_PYLIB_NAME = "PyLib.dfy"

#: One ``anti_cheat`` violation, e.g. ``"line 3: assume"``.
_ANTI_CHEAT_VIOLATION = re.compile(r"^line (\d+): (.*)$")

#: Bound on ``dafny --version``: it must answer quickly or the CLI is broken.
_VERSION_TIMEOUT_SECONDS = 60.0

#: Cap on the fallback error text when ``dafny resolve`` fails without parseable errors.
_MAX_FALLBACK_CHARS = 500


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    verified: bool
    errors: tuple[DafnyError, ...]
    seconds: float


def resolve(dafny_code: str, *, timeout_seconds: float = 120.0) -> tuple[DafnyError, ...]:
    """Run ``dafny resolve`` on ``dafny_code`` with a copy of ``PyLib.dfy`` and parse the errors.

    ``dafny_code`` is written to ``main.dfy`` in a temporary directory next to a copy of
    ``PyLib.dfy``, and ``dafny resolve main.dfy PyLib.dfy`` runs there. An empty result means the
    code resolves. A process stopped by the timeout gives
    ``(DafnyError(0, 0, "timeout after <n> s"),)``. When Dafny fails without parseable errors, the
    result is a single ``DafnyError`` at (0, 0) with Dafny's output, so a failure never looks like a
    clean resolve.

    Raises ``ToolError`` when the dafny CLI is not on ``PATH`` or ``dafny --version`` does not
    report ``Config().dafny.version``.
    """
    _require_dafny()
    with tempfile.TemporaryDirectory(prefix="pcc-resolve-") as tmp:
        root = Path(tmp)
        (root / _MAIN_NAME).write_text(dafny_code, encoding="utf-8")
        shutil.copy(PYLIB_FILE, root / _PYLIB_NAME)
        try:
            proc = _run_dafny(["resolve", _MAIN_NAME, _PYLIB_NAME], timeout_seconds=timeout_seconds, cwd=root)
        except subprocess.TimeoutExpired:
            return (DafnyError(0, 0, f"timeout after {timeout_seconds:g} s"),)
        errors = parse_errors(proc.stdout)
        if proc.returncode != 0 and not errors:
            detail = (proc.stderr or proc.stdout).strip() or (f"dafny resolve exited with code {proc.returncode}")
            errors = (DafnyError(0, 0, detail[:_MAX_FALLBACK_CHARS]),)
        return errors


def verify(
    dafny_code: str,
    *,
    assumptions_module: str = "",
    resource_limit: int,
    timeout_seconds: float | None = None,
    extra_modules: Sequence[str] = (),
) -> VerifyOutcome:
    """Run ``dafny verify`` on ``dafny_code`` with ``pcc.pylib`` and parse the result.

    ``dafny_code`` is written to ``main.dfy`` in a temporary directory next to a copy of
    ``PyLib.dfy``. ``dafny verify --resource-limit <resource_limit> --verification-time-limit 0
    main.dfy PyLib.dfy`` runs there. Pass ``Config().dafny.resource_limit``.

    - Never pass ``--allow-warnings``. Dafny exits 2 on warnings without it, so a warning such as a body-less lemma or
      a plain ``assume`` is not verified either.
    - Any non-zero exit code means not verified. Exit 0 is the only way to be verified.
    - ``--verification-time-limit 0`` disables Dafny's own per-batch time limit (``dafny verify --help``: "A value of 0
      indicates no limit"; checked with Dafny 4.11). The resource limit alone decides, so the result does not depend on
      the machine.
    - ``timeout_seconds`` is only a process-level emergency stop for a hung dafny process. A process stopped by it is
      not verified, with ``errors`` holding ``DafnyError(0, 0, "timeout after <n> s")``.
    - ``assumptions_module`` names the module that may hold library contracts, the same module ``anti_cheat`` takes.
    - Every ``anti_cheat`` violation becomes ``DafnyError(<line>, 0, "anti-cheat: <form>")`` and makes the result
      unverified, even when Dafny exits 0 (Dafny 4.11 verifies ``{:axiom}`` and ``@Axiom`` silently).
    - When Dafny exits non-zero without parseable errors, the result is a single ``DafnyError`` at (0, 0) with
      Dafny's output, so a failure never looks like a clean verify.
    - ``extra_modules`` is reserved for callee modules (a later ticket); it is accepted but not used here.
    - ``seconds`` is the elapsed wall-clock time of this invocation, including the subprocess and the anti-cheat scan.

    Raises ``ToolError`` when the dafny CLI is not on ``PATH`` or ``dafny --version`` does not
    report ``Config().dafny.version``.
    """
    started = time.perf_counter()
    _require_dafny()
    with tempfile.TemporaryDirectory(prefix="pcc-verify-") as tmp:
        root = Path(tmp)
        (root / _MAIN_NAME).write_text(dafny_code, encoding="utf-8")
        shutil.copy(PYLIB_FILE, root / _PYLIB_NAME)
        try:
            proc = _run_dafny(
                [
                    "verify",
                    "--resource-limit",
                    str(resource_limit),
                    "--verification-time-limit",
                    "0",
                    _MAIN_NAME,
                    _PYLIB_NAME,
                ],
                timeout_seconds=timeout_seconds,
                cwd=root,
            )
        except subprocess.TimeoutExpired:
            assert timeout_seconds is not None  # only a set timeout can expire
            elapsed = time.perf_counter() - started
            return VerifyOutcome(False, (DafnyError(0, 0, f"timeout after {timeout_seconds:g} s"),), elapsed)
        errors = parse_errors(proc.stdout)
        if proc.returncode != 0 and not errors:
            detail = (proc.stderr or proc.stdout).strip() or f"dafny verify exited with code {proc.returncode}"
            errors = (DafnyError(0, 0, detail[:_MAX_FALLBACK_CHARS]),)
        errors = (*errors, *_cheat_errors(dafny_code, assumptions_module))
        verified = proc.returncode == 0 and not errors
        return VerifyOutcome(verified, errors, time.perf_counter() - started)


def _cheat_errors(dafny_code: str, assumptions_module: str) -> tuple[DafnyError, ...]:
    """Map ``anti_cheat`` violations in ``dafny_code`` to ``DafnyError`` at column 0."""
    found: list[DafnyError] = []
    for violation in anti_cheat(dafny_code, "", assumptions_module=assumptions_module):
        match = _ANTI_CHEAT_VIOLATION.match(violation)
        if match is None:
            found.append(DafnyError(0, 0, f"anti-cheat: {violation}"))
        else:
            found.append(DafnyError(int(match.group(1)), 0, f"anti-cheat: {match.group(2)}"))
    return tuple(found)


def build_python(dafny_code: str, out_dir: Path) -> Path:
    """Compile ``dafny_code`` with ``dafny build --target:py`` into ``out_dir`` and return the module directory."""
    raise NotImplementedError


def _run_dafny(args: Sequence[str], *, timeout_seconds: float | None, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``dafny <args>`` in ``cwd`` and return the completed process.

    The shared helper for ``resolve`` now and ``verify`` (issue #20) later: Dafny's diagnostics are
    decoded as UTF-8 with replacement, and ``timeout_seconds`` bounds the whole run.
    """
    return subprocess.run(
        ["dafny", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        cwd=str(cwd),
        check=False,
    )


def _require_dafny() -> None:
    """Raise ``ToolError`` unless ``dafny --version`` reports ``Config().dafny.version``."""
    expected = Config().dafny.version
    try:
        proc = _run_dafny(["--version"], timeout_seconds=_VERSION_TIMEOUT_SECONDS, cwd=Path.cwd())
    except OSError as error:
        raise ToolError(f"dafny CLI is not on PATH; install Dafny {expected}") from error
    except subprocess.TimeoutExpired as error:
        raise ToolError(f"dafny --version timed out; install Dafny {expected}") from error
    reported = _release_of(proc.stdout) if proc.returncode == 0 else None
    if reported != expected:
        shown = (proc.stdout or proc.stderr).strip() or "no output"
        raise ToolError(f"dafny --version reports {shown!r}, expected Dafny {expected}; install Dafny {expected}")


def _release_of(output: str) -> str | None:
    """The release number in ``dafny --version`` output, without a build suffix such as ``+abc123``."""
    match = re.search(r"(\d+(?:\.\d+)+)(?:\+\S*)?", output)
    return match.group(1) if match else None


__all__ = ["VerifyOutcome", "build_python", "resolve", "verify"]
