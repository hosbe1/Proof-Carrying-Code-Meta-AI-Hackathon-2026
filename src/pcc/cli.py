"""Command line entry point ``pcc-check``, also used by the GitHub Action."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
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
from pcc.core import (
    Certificate,
    Dependency,
    FunctionRef,
    ProofOutcome,
    Result,
    State,
    ToolError,
    serialize,
)
from pcc.project.calls import subset_edges


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pcc-check", description="Proof-carrying code for Python functions.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="check functions and write certificates")
    run.add_argument("paths", nargs="*", help="files to check (default: functions changed against --base)")
    run.add_argument("--base", default="origin/main", help="git ref to diff against")
    run.add_argument("--proofs-dir", default="proofs", help="directory holding the proof certificates")
    run.add_argument("--report", default="report.md", help="file the markdown report is written to")
    run.add_argument("--json", default="results.json", help="file the JSON results are written to")
    survey = sub.add_parser("survey", help="list functions and whether they would reach step 1")
    survey.add_argument("repo", help="repository to survey")
    survey.add_argument("--json", action="store_true", help="write the JSON list to stdout instead of text lines")
    sub.add_parser("version", help="print the version")
    args = parser.parse_args(argv)
    if args.command == "version":
        from importlib.metadata import version

        print(version("pcc"))
        return 0
    try:
        if args.command == "survey":
            return _survey_command(repo_root=Path(args.repo), as_json=args.json)
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


def _edges_by_key(
    scope: Sequence[FunctionRef], stored: Sequence[Certificate]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Direct project callees of every scope function, keyed by ``(path, qualname)``.

    The universe is the scope plus the stored certificate functions, so callers find
    callees that are cached and not re-run. A snippet that no longer parses contributes
    no edges.
    """
    scope_keys = {(ref.path, ref.qualname) for ref in scope}
    extra = [cert.function for cert in stored if (cert.function.path, cert.function.qualname) not in scope_keys]
    universe = list(scope) + extra
    edges = subset_edges(tuple(universe))
    by_key: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    for ref in universe:
        key = (ref.path, ref.qualname)
        if key in by_key:
            continue
        callees = edges.get(ref, frozenset())
        by_key[key] = tuple(sorted({(callee.path, callee.qualname) for callee in callees}))
    return by_key


def _hashes_for_key(
    key: tuple[str, str],
    edges_by_key: dict[tuple[str, str], tuple[tuple[str, str], ...]],
    specs_by_key: dict[tuple[str, str], str],
) -> tuple[Dependency, ...]:
    """The certificate dependencies of ``key``: the sha256 of each direct callee spec found in ``specs_by_key``.

    Self-calls are skipped, like in ``pipeline._direct_callees``: a function that calls only itself is checked
    normally, with no dependency. Callees without a known spec are skipped, which conservatively marks the
    caller stale through ``is_stale``.
    """
    found: list[Dependency] = []
    for callee_key in edges_by_key.get(key, ()):
        if callee_key == key:
            continue
        spec = specs_by_key.get(callee_key)
        if spec is not None:
            found.append(pipeline.spec_dependency(callee_key[1], spec))
    return tuple(found)


def _expand_callers_of_stale(
    scope: Sequence[FunctionRef],
    edges_by_key: dict[tuple[str, str], tuple[tuple[str, str], ...]],
    cached: dict[FunctionRef, Result],
    to_run: list[FunctionRef],
    stale: list[Certificate],
    by_key: dict[tuple[str, str], Certificate],
) -> None:
    """Move cached callers of re-run callees to ``to_run`` until no more move.

    A caller whose callee is proven again may rest on a changed spec, so its cached
    certificate cannot be reused even when its own source is unchanged.
    """
    to_run_keys = {(ref.path, ref.qualname) for ref in to_run}
    changed = True
    while changed:
        changed = False
        for function in scope:
            key = (function.path, function.qualname)
            if key in to_run_keys or function not in cached:
                continue
            if any(callee in to_run_keys for callee in edges_by_key.get(key, ())):
                to_run_keys.add(key)
                to_run.append(function)
                del cached[function]
                cert = by_key.get(key)
                if cert is not None:
                    stale.append(cert)
                changed = True


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
    stored_specs = {(cert.function.path, cert.function.qualname): cert.spec_dafny for cert in stored}
    edges_by_key = _edges_by_key(scope, stored)

    cached: dict[FunctionRef, Result] = {}
    to_run: list[FunctionRef] = []
    stale: list[Certificate] = []
    refresh: list[Certificate] = []
    for function in scope:
        cert = by_key.get((function.path, function.qualname))
        if cert is None:
            to_run.append(function)
            continue
        current_deps = _hashes_for_key((function.path, function.qualname), edges_by_key, stored_specs)
        if cert.pylib_sha256 != current_pylib:
            outcome = verifier_module.verify(cert.dafny_source, resource_limit=config.dafny.resource_limit)
            if not outcome.verified:
                stale.append(cert)
                to_run.append(function)
                continue
            if certificate_module.is_stale(
                cert, function, dependency_hashes=current_deps, pylib_sha256=cert.pylib_sha256
            ):
                stale.append(cert)
                to_run.append(function)
                continue
            refresh.append(dataclasses.replace(cert, pylib_sha256=current_pylib))
        elif certificate_module.is_stale(cert, function, dependency_hashes=current_deps, pylib_sha256=current_pylib):
            stale.append(cert)
            to_run.append(function)
            continue
        cached[function] = Result(
            function=function,
            state=State.PROVEN,
            reason="certificate",
            proof=ProofOutcome(proven=True, dafny_source=cert.dafny_source, attempts=0, stage="certificate"),
        )
    _expand_callers_of_stale(scope, edges_by_key, cached, to_run, stale, by_key)
    for cert in stored:
        if (cert.function.path, cert.function.qualname) in {(f.path, f.qualname) for f in scope}:
            continue
        if not _still_exists(repo_root, cert):
            stale.append(cert)

    ran = pipeline.run_many(to_run, repo_root, client=client, config=config, callees=cached) if to_run else ()
    by_result = {(result.function.path, result.function.qualname): result for result in ran}
    results = []
    for function in scope:
        if function in cached:
            results.append(cached[function])
        else:
            results.append(by_result[(function.path, function.qualname)])

    fresh_specs = dict(stored_specs)
    for result in ran:
        if result.translation is not None:
            fresh_specs[(result.function.path, result.function.qualname)] = result.translation.dafny_spec
    for result in results:
        if result.state is State.PROVEN and result.reason != "certificate":
            deps = _hashes_for_key((result.function.path, result.function.qualname), edges_by_key, fresh_specs)
            certificate_module.write(
                certificate_module.from_result(result, config, dependency_hashes=deps), proofs_dir, config=config
            )
    for cert in refresh:
        certificate_module.write(cert, proofs_dir, config=config)

    by_result_key = {(result.function.path, result.function.qualname): result for result in results}
    finding_stale: list[Certificate] = []
    for cert in stale:
        match = by_result_key.get((cert.function.path, cert.function.qualname))
        # A stale certificate is a finding unless its function was proven again in this run.
        if match is None or match.state is not State.PROVEN:
            finding_stale.append(cert)
    text = report_module.render(results, stale=finding_stale)
    report_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(serialize.dumps(tuple(results)) + "\n", encoding="utf-8")
    print(_summary_line(results))

    if any(result.state is State.COUNTEREXAMPLE for result in results):
        return 1
    if finding_stale:
        return 1
    return 0


def _is_skipped(path: str) -> bool:
    """Whether ``path`` belongs to the survey skip list (issue #64).

    Matching is case-sensitive on the normalized POSIX relative path: any path with a directory
    component exactly ``tests`` or ``docs`` is skipped, as is a file whose basename is
    ``conftest.py`` or ``setup.py``, starts with ``test_`` and ends with ``.py``, or ends with
    ``_test.py``. There are no substring false positives: ``contest.py`` and ``my_tests/`` stay.
    """
    parts = path.split("/")
    if any(part in ("tests", "docs") for part in parts[:-1]):
        return True
    basename = parts[-1]
    return (
        basename in ("conftest.py", "setup.py")
        or (basename.startswith("test_") and basename.endswith(".py"))
        or basename.endswith("_test.py")
    )


def _survey_command(*, repo_root: Path, as_json: bool) -> int:
    """List every surveyed function and whether it would reach step 1 (issue #64).

    Discovers the functions with ``discover(repo_root, ["."])``, drops the skip list, builds the
    call graph and the import errors once, and reports the ``precheck`` reason per function. Makes
    no model call. With ``as_json`` only the JSON list of ``{"function", "reason"}`` objects goes to
    stdout (``reason`` is null for checkable functions); otherwise one ``<path>::<qualname>`` line per
    function plus a final ``"N of M functions checkable"`` line. Never exits 1: a failing import is
    a result, not an error.
    """
    root = repo_root.resolve()
    functions = tuple(ref for ref in project.discover(root, ["."]) if not _is_skipped(ref.path))
    graph = project.call_graph(root, functions)
    import_errors = pipeline.module_import_errors(functions, root)
    rows = [(ref, pipeline.precheck(ref, root, graph=graph, import_errors=import_errors)) for ref in functions]
    if as_json:
        print(json.dumps([{"function": f"{ref.path}::{ref.qualname}", "reason": reason} for ref, reason in rows]))
    else:
        for ref, reason in rows:
            print(f"{ref.path}::{ref.qualname}: {reason if reason is not None else 'checkable'}")
        checkable = sum(1 for _, reason in rows if reason is None)
        print(f"{checkable} of {len(rows)} functions checkable")
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
