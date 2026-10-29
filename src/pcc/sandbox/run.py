"""Running untrusted Python in a subprocess with time and memory limits."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pcc.core import ToolError

#: Variables passed on to the child process. Everything else from the parent environment is dropped, so API keys and
#: other secrets never reach the untrusted code.
_ALLOWED_ENV_VARS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

#: stdout and stderr are each cut off here.
_MAX_OUTPUT = 1_048_576


@dataclass(frozen=True, slots=True)
class RunOutcome:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float


def run_python(script: str, *, cwd: Path, timeout_seconds: float = 60.0, memory_mb: int = 1024) -> RunOutcome:
    """Run ``script`` with pcc's own interpreter in ``cwd`` under a time and a memory limit.

    The script is written to a temporary directory and run as ``sys.executable -s -P <file>`` with ``cwd`` as the
    working directory. ``-I`` is not used because it implies ``-E``, which would ignore ``PYTHONHASHSEED``. A
    preamble puts ``cwd`` and ``cwd/src`` (when it exists) at the front of ``sys.path``. The checked project's
    third-party dependencies are not installed for pcc's interpreter, so a function that imports one fails here with
    ``ImportError`` and the pipeline reports it as ``not_checkable``.

    The child process gets a minimal environment built from an allowlist (``PATH``, ``HOME``, ``LANG``, ``LC_ALL``,
    ``TMPDIR``) plus ``PYTHONHASHSEED="0"``. Nothing else from the parent environment is passed on; API keys and other
    secrets never reach the untrusted code. Standard input is closed (``/dev/null``), so a script that reads it gets
    end-of-file instead of hanging.

    The process runs in its own session. On timeout the whole process group gets SIGKILL. The whole process group is
    also killed after the main process exits, so a forked child cannot hold the pipes open.

    Memory is limited with ``RLIMIT_AS`` on Linux. On macOS the limit is not set (the platform does not enforce
    ``RLIMIT_AS`` reliably); a script that allocates past the limit raises ``MemoryError`` on Linux and runs out of
    memory on macOS.

    stdout and stderr are each cut off at 1 MB. Output with invalid UTF-8 is decoded with ``errors="replace"``.
    ``ok`` is ``returncode == 0 and not timed_out``.

    There is no network isolation: a script can open sockets. Windows is not supported.

    Only a missing ``cwd`` or an interpreter that cannot be started raises ``ToolError``. Everything else (a crash,
    a timeout, a memory error) is a ``RunOutcome``.
    """
    root = Path(cwd)
    if not root.is_dir():
        raise ToolError(f"sandbox cwd does not exist or is not a directory: {cwd}")
    full_script = _preamble(root) + script
    env = {name: os.environ[name] for name in _ALLOWED_ENV_VARS if name in os.environ}
    env["PYTHONHASHSEED"] = "0"
    preexec = None if sys.platform == "darwin" else _memory_cap(memory_mb)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pcc-sandbox-") as tmp:
        entry = Path(tmp) / "script.py"
        entry.write_text(full_script, encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-s", "-P", str(entry)],
                cwd=str(root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=preexec,
            )
        except OSError as error:
            raise ToolError(f"cannot start the interpreter: {error}") from error
        try:
            # Drain the pipes in the background while the main process is supervised: a forked child that holds the
            # pipes open must not delay noticing that the main process exited (or timed out).
            drained: dict[str, bytes] = {}

            def _drain() -> None:
                stdout_bytes, stderr_bytes = proc.communicate()
                drained["stdout"] = stdout_bytes
                drained["stderr"] = stderr_bytes

            drainer = threading.Thread(target=_drain, daemon=True)
            drainer.start()
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc.pid)
                returncode = proc.wait()
            else:
                timed_out = False
                # The main process exited; kill the group so a forked child cannot hold the pipes open.
                _kill_group(proc.pid)
            drainer.join()
            stdout_bytes = drained["stdout"]
            stderr_bytes = drained["stderr"]
            assert proc.returncode is not None
        finally:
            if proc.poll() is None:
                _kill_group(proc.pid)
                proc.wait()
    seconds = time.monotonic() - start
    return RunOutcome(
        ok=returncode == 0 and not timed_out,
        returncode=returncode,
        stdout=_decode_capped(stdout_bytes),
        stderr=_decode_capped(stderr_bytes),
        timed_out=timed_out,
        seconds=seconds,
    )


def _preamble(root: Path) -> str:
    """``import sys`` plus ``cwd`` and ``cwd/src`` at the front of ``sys.path``."""
    lines = ["import sys"]
    src = root / "src"
    if src.is_dir():
        lines.append(f"sys.path.insert(0, {str(src)!r})")
    lines.append(f"sys.path.insert(0, {str(root)!r})")
    return "\n".join(lines) + "\n"


def _memory_cap(memory_mb: int) -> Callable[[], None]:
    """A ``preexec_fn`` that caps the child's address space at ``memory_mb`` MB (Linux)."""

    def _preexec() -> None:
        import resource

        cap = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    return _preexec


def _kill_group(pid: int) -> None:
    """SIGKILL the whole process group of ``pid``. A group that is already gone is fine."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _decode_capped(data: bytes) -> str:
    """Decode ``data`` with ``errors="replace"``, cut off at 1 MB."""
    return data[:_MAX_OUTPUT].decode("utf-8", errors="replace")


__all__ = ["RunOutcome", "run_python"]
