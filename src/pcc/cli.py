"""Command line entry point ``pcc-check``, also used by the GitHub Action."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from pcc import certificate as certificate_module
from pcc import models, pipeline, project
from pcc import pylib as pylib_module
from pcc import report as report_module
from pcc import verifier as verifier_module
from pcc.config import Config
from pcc.core import Certificate, FunctionRef, ProofOutcome, Result, State, ToolError, serialize


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pcc-check", description="Proof-carrying code for Python functions.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="check functions and write certificates")
    run.add_argument("paths", nargs="*", help="files to check (default: functions changed against --base)")
    run.add_argument("--base", default="origin/main", help="git ref to diff against")
    run.add_argument("--proofs-dir", default="proofs", help="directory holding the proof certificates")
    run.add_argument("--report", default="report.md", help="file the markdown report is written to")
    run.add_argument("--json", default="results.json", help="file the JSON results are written to")
    sub.add_parser("version", help="print the version")
    args = parser.parse_args(argv)
    if args.command == "version":
        from importlib.metadata import version

        print(version("pcc"))
        return 0
    try:
        return _run_command(
            paths=args.paths,
            base=args.base,
            proofs_dir=Path(args.proofs_dir),
            report_path=Path(args.report),
            json_path=Path(args.json),
        )
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


def _run_command(
    *,
    paths: Sequence[str],
    base: str,
    proofs_dir: Path,
    report_path: Path,
    json_path: Path,
) -> int:
    """Check the functions in scope, refresh their certificates and write the report.

    Returns the process exit code: 1 for a counterexample or a stale certificate whose
    function is not proven again (or no longer exists), 2 when a tool breaks, else 0.
    ``open`` results, including budget exhaustion, exit 0.
    """
    repo_root = Path.cwd()
    config = Config()
    initial = project.discover(repo_root, list(paths)) if paths else project.changed(repo_root, base)
    scope = project.affected(repo_root, initial)

    router = models.build_client(config)
    # The run shares one cost cap; calls and time scale with the scope so every
    # function keeps its own per-function budget inside the run.
    breadth = max(1, len(scope))
    client = models.BudgetedClient(
        router,
        max_cost_usd=config.budget.max_cost_usd_per_run,
        max_calls=config.budget.max_calls_per_function * breadth,
        max_seconds=config.budget.max_seconds_per_function * breadth,
    )
    config = models.available_stages(config, router)

    current_pylib = hashlib.sha256(pylib_module.PYLIB_FILE.read_bytes()).hexdigest()
    stored = _load_certificates(proofs_dir)
    by_key = {(cert.function.path, cert.function.qualname): cert for cert in stored}

    cached: dict[FunctionRef, Result] = {}
    to_run: list[FunctionRef] = []
    stale: list[Certificate] = []
    refresh: list[Certificate] = []
    for function in scope:
        cert = by_key.get((function.path, function.qualname))
        if cert is None:
            to_run.append(function)
            continue
        if cert.pylib_sha256 != current_pylib:
            outcome = verifier_module.verify(cert.dafny_source, resource_limit=config.dafny.resource_limit)
            if not outcome.verified:
                stale.append(cert)
                to_run.append(function)
                continue
            if certificate_module.is_stale(cert, function, pylib_sha256=cert.pylib_sha256):
                stale.append(cert)
                to_run.append(function)
                continue
            refresh.append(dataclasses.replace(cert, pylib_sha256=current_pylib))
        elif certificate_module.is_stale(cert, function, pylib_sha256=current_pylib):
            stale.append(cert)
            to_run.append(function)
            continue
        cached[function] = Result(
            function=function,
            state=State.PROVEN,
            reason="certificate",
            proof=ProofOutcome(proven=True, dafny_source=cert.dafny_source, attempts=0, stage="certificate"),
        )
    for cert in stored:
        if (cert.function.path, cert.function.qualname) in {(f.path, f.qualname) for f in scope}:
            continue
        if not _still_exists(repo_root, cert):
            stale.append(cert)

    ran = pipeline.run_many(to_run, repo_root, client=client, config=config) if to_run else ()
    by_result = {(result.function.path, result.function.qualname): result for result in ran}
    results = []
    for function in scope:
        if function in cached:
            results.append(cached[function])
        else:
            results.append(by_result[(function.path, function.qualname)])

    for result in results:
        if result.state is State.PROVEN and result.reason != "certificate":
            certificate_module.write(certificate_module.from_result(result, config), proofs_dir, config=config)
    for cert in refresh:
        certificate_module.write(cert, proofs_dir, config=config)

    by_result_key = {(result.function.path, result.function.qualname): result for result in results}
    finding_stale: list[Certificate] = []
    for cert in stale:
        match = by_result_key.get((cert.function.path, cert.function.qualname))
        # A stale certificate is a finding unless its function was proven again in this run.
        if match is None or match.state is not State.PROVEN:
            finding_stale.append(cert)
    text = report_module.render(results, stale=stale)
    report_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(serialize.dumps(tuple(results)) + "\n", encoding="utf-8")
    print(_summary_line(results))

    if any(result.state is State.COUNTEREXAMPLE for result in results):
        return 1
    if finding_stale:
        return 1
    return 0


def _load_certificates(proofs_dir: Path) -> list[Certificate]:
    """Every certificate stored under ``proofs_dir``, skipping the ``PyLib.dfy`` copy."""
    if not proofs_dir.is_dir():
        return []
    found: list[Certificate] = []
    for path in sorted(proofs_dir.rglob("*.dfy")):
        if path.name == "PyLib.dfy":
            continue
        try:
            found.append(certificate_module.load(path))
        except (FileNotFoundError, ValueError, TypeError, OSError):
            continue
    return found


def _still_exists(repo_root: Path, cert: Certificate) -> bool:
    """Whether ``discover`` still finds the certificate's path and qualname."""
    try:
        refs = project.discover(repo_root, [cert.function.path])
    except (FileNotFoundError, ValueError):
        return False
    return any(ref.qualname == cert.function.qualname for ref in refs)


def _summary_line(results: Sequence[Result]) -> str:
    """The one-line summary printed to stdout, mirroring the report's summary."""
    if not results:
        return "No functions checked."
    parts: list[str] = []
    for state in State:
        count = sum(1 for result in results if result.state is state)
        if not count:
            continue
        word = state.value
        if state is State.COUNTEREXAMPLE and count != 1:
            word = "counterexamples"
        parts.append(f"{count} {word}")
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
