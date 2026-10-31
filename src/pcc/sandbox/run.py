"""Running untrusted Python in a subprocess with time and memory limits."""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pcc.core import ToolError

#: Variables passed on to the child process. Everything else from the parent environment is dropped, so API keys and
#: other secrets never reach the untrusted code.
_ALLOWED_ENV_VARS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

#: stdout and stderr are each cut off here.
_MAX_OUTPUT = 1_048_576

#: Grace period to wait for the main process after SIGKILL before giving up waiting.
_KILL_GRACE_SECONDS = 2.0

#: Grace period to wait for the output pumps after the main process exited or timed out.
_DRAIN_GRACE_SECONDS = 2.0

#: Size of a single pipe read while draining the child output.
_READ_CHUNK_SIZE = 65_536


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

    The process runs in its own session. On Linux it also receives SIGKILL if its supervising parent dies, so an
    interrupted test or agent cannot leave a CPU-bound sandbox child behind. On timeout the whole process group and
    the main process itself get
    SIGKILL (the direct kill covers a script that left its process group, e.g. via ``fork`` plus ``setsid``). The
    group and the main process are also killed after the main process exits, so a forked child cannot hold the pipes
    open. Waiting for the process and for the output pumps is bounded: even a fully detached descendant cannot hang
    this call forever; in that case the pipes are closed and the output collected so far is returned.

    Memory is limited with ``RLIMIT_AS`` on Linux. On macOS the limit is not set (the platform does not enforce
    ``RLIMIT_AS`` reliably); a script that allocates past the limit raises ``MemoryError`` on Linux and runs out of
    memory on macOS.

    stdout and stderr are each cut off at 1 MB. The pipes are drained incrementally and only the first 1 MB per
    stream are kept; the rest is read and discarded so a verbose child blocks neither on a full pipe nor on parent
    memory. Output with invalid UTF-8 is decoded with ``errors="replace"``. ``ok`` is
    ``returncode == 0 and not timed_out``.

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
    preexec = None if sys.platform == "darwin" else _memory_cap(memory_mb, os.getpid())
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
            # pipes open must not delay noticing that the main process exited (or timed out). Only the first 1 MB
            # per stream are kept; the rest is read and discarded so the parent memory stays bounded.
            assert proc.stdout is not None
            assert proc.stderr is not None
            drained: dict[str, bytes] = {}
            drain_lock = threading.Lock()
            stop_pumps = threading.Event()

            def _pump(stream: BinaryIO, key: str) -> None:
                chunks: list[bytes] = []
                kept = 0
                try:
                    fd = stream.fileno()
                    while not stop_pumps.is_set():
                        try:
                            ready, _, _ = select.select([fd], [], [], 0.2)
                        except (OSError, ValueError):
                            break
                        if stop_pumps.is_set():
                            break
                        if not ready:
                            continue
                        try:
                            data = os.read(fd, _READ_CHUNK_SIZE)
                        except OSError:
                            break
                        if not data:
                            break
                        if kept < _MAX_OUTPUT:
                            piece = data[: _MAX_OUTPUT - kept]
                            chunks.append(piece)
                            kept += len(piece)
                        # Beyond the cap the data is discarded but reading continues so the child
                        # never blocks on a full pipe.
                finally:
                    with drain_lock:
                        drained[key] = b"".join(chunks)

            stdout_pump = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
            stderr_pump = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
            stdout_pump.start()
            stderr_pump.start()
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_tree(proc.pid)
                returncode = _wait_bounded(proc, _KILL_GRACE_SECONDS)
            else:
                timed_out = False
                # The main process exited; kill the group so a forked child cannot hold the pipes open.
                _kill_tree(proc.pid)
            _join_bounded(stdout_pump, stderr_pump, proc, stop_pumps)
            with drain_lock:
                stdout_bytes = drained.get("stdout", b"")
                stderr_bytes = drained.get("stderr", b"")
            assert proc.returncode is not None
        finally:
            if proc.poll() is None:
                _kill_tree(proc.pid)
                _wait_bounded(proc, _KILL_GRACE_SECONDS)
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


def _memory_cap(memory_mb: int, parent_pid: int) -> Callable[[], None]:
    """Cap address space and kill the child if its supervisor dies (Linux)."""

    def _preexec() -> None:
        if sys.platform == "linux":
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
                raise OSError(ctypes.get_errno(), "cannot set sandbox parent-death signal")
            # The parent may have died between fork() and prctl().
            if os.getppid() != parent_pid:
                os._exit(1)
        import resource

        cap = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    return _preexec


def _kill_group(pid: int) -> None:
    """SIGKILL the whole process group of ``pid``. A group that is already gone is fine."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _kill_tree(pid: int) -> None:
    """SIGKILL the process group and the process itself. A missing process or group is fine."""
    _kill_group(pid)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)


def _wait_bounded(proc: subprocess.Popen[bytes], grace_seconds: float) -> int:
    """Wait for ``proc`` at most ``grace_seconds``; never raise ``TimeoutExpired``."""
    try:
        return proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            proc.kill()
        try:
            return proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            if proc.returncode is not None:
                return proc.returncode
            return -signal.SIGKILL


def _join_bounded(
    first: threading.Thread,
    second: threading.Thread,
    proc: subprocess.Popen[bytes],
    stop_pumps: threading.Event,
) -> None:
    """Wait for the output pumps without hanging on a detached descendant.

    The first grace window lets the pumps drain buffered output naturally. Only if a pump is still
    alive (a detached descendant holds the pipes open) are the pumps told to stop; only if they still
    do not finish are the pipes closed and waited on once more.
    """
    first.join(timeout=_DRAIN_GRACE_SECONDS)
    second.join(timeout=_DRAIN_GRACE_SECONDS)
    if first.is_alive() or second.is_alive():
        stop_pumps.set()
        first.join(timeout=_DRAIN_GRACE_SECONDS)
        second.join(timeout=_DRAIN_GRACE_SECONDS)
    if first.is_alive() or second.is_alive():
        with contextlib.suppress(Exception):
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
        first.join(timeout=_DRAIN_GRACE_SECONDS)
        second.join(timeout=_DRAIN_GRACE_SECONDS)


def _decode_capped(data: bytes) -> str:
    """Decode ``data`` with ``errors="replace"``, cut off at 1 MB."""
    return data[:_MAX_OUTPUT].decode("utf-8", errors="replace")


__all__ = ["RunOutcome", "run_python"]
