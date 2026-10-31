"""Runs the steps for one function and for a set of functions. The only module that imports every step."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import MappingProxyType

from pcc.config import Config
from pcc.context import collect, search_further
from pcc.core import (
    Assumption,
    AssumptionKind,
    Counterexample,
    Dependency,
    FunctionRef,
    Intent,
    NotCheckable,
    Result,
    Spec,
    State,
    Translation,
    TranslationPath,
    UnsupportedConstruct,
)
from pcc.counterexample import search as search_counterexample
from pcc.equivalence import check as check_equivalence
from pcc.intent import derive
from pcc.models import ModelClient
from pcc.project import call_order
from pcc.project.calls import subset_edges
from pcc.prove import prove
from pcc.second_opinion import review
from pcc.spec import check as check_spec
from pcc.spec import write as write_spec
from pcc.translate_ai import translate
from pcc.transpile import transpile_function, transpile_spec
from pcc.verifier.layout import module_name

#: Reason shared by every result that ends in mutual recursion (docs/plan.md, step 4).
MUTUAL_RECURSION = "mutual recursion"

_METHOD_RE = re.compile(r"(?<![A-Za-z0-9_'])method(?![A-Za-z0-9_'])")


def _sort_key(ref: FunctionRef) -> tuple[str, int]:
    return (ref.path, ref.lineno)


def _direct_callees(function: FunctionRef, callees: Mapping[FunctionRef, Result]) -> tuple[FunctionRef, ...]:
    """The project callees of ``function`` among ``callees``, sorted by path, then by ``lineno``.

    Self-calls are excluded: a function that calls only itself is checked normally. The edges come from the
    functions' sources (see ``subset_edges``); callees ``function`` never calls are ignored.
    """
    if not callees:
        return ()
    edges = subset_edges((function, *callees))
    return tuple(sorted(set(edges.get(function, frozenset())) - {function}, key=_sort_key))


def _reaches(start: FunctionRef, target: FunctionRef, edges: Mapping[FunctionRef, frozenset[FunctionRef]]) -> bool:
    """Whether ``target`` is reachable from ``start`` following ``edges``."""
    seen = {start}
    stack = [start]
    while stack:
        for successor in edges.get(stack.pop(), ()):
            if successor == target:
                return True
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)
    return False


def _callee_gate(
    function: FunctionRef,
    direct: tuple[FunctionRef, ...],
    callees: Mapping[FunctionRef, Result],
) -> Result | None:
    """The early result when a direct callee decides the caller, or ``None`` when the run proceeds.

    A callee reached back from the caller is mutual recursion; a callee that ended that way makes the caller end
    the same way. A callee with a counterexample ends the caller ``open``, because an assumption known to be false
    proves nothing; a callee that is ``not_checkable`` ends the caller ``not_checkable``. No step runs then. The
    first deciding callee in path order wins.
    """
    if not direct:
        return None
    edges = subset_edges((function, *callees))
    for callee in direct:
        outcome = callees[callee]
        if _reaches(callee, function, edges) or (outcome.state is State.OPEN and outcome.reason == MUTUAL_RECURSION):
            return Result(function, State.OPEN, MUTUAL_RECURSION)
        if outcome.state is State.COUNTEREXAMPLE:
            return Result(function, State.OPEN, f"callee {callee.qualname} has a counterexample")
        if outcome.state is State.NOT_CHECKABLE:
            return Result(function, State.NOT_CHECKABLE, f"callee {callee.qualname} not checkable")
    return None


def _match_brace(text: str, opening: int) -> int | None:
    """Index of the ``}`` matching the ``{`` at ``opening``, or ``None`` when unbalanced."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _body_start(text: str) -> int | None:
    """Index of the ``{`` that opens the method body, or ``None`` when there is none.

    A ``{`` inside parentheses or brackets, or a group that more header text follows, holds a contract
    expression (a set display such as ``{0, 1}`` or an attribute argument) rather than the body: only a
    group whose match runs to the end of the text opens the body.
    """
    depth_paren = 0
    depth_brack = 0
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        elif char == "{" and depth_paren == 0 and depth_brack == 0:
            closing = _match_brace(text, index)
            if closing is None:
                return None
            if text[closing + 1 :].strip() == "":
                return index
            index = closing
        index += 1
    return None


def _assumption_method(outcome: Result) -> str | None:
    """The callee's translated method header as a body-less ``{:axiom}`` method.

    Returns ``None`` when the callee has no translated method to assume or no body brace is found, so the
    caller still lists the assumption without Dafny text. The ``{:axiom}`` mark is what ``pcc.verifier.anti_cheat``
    allows inside the generated assumptions module.
    """
    if outcome.translation is None:
        return None
    text = outcome.translation.dafny_function
    body = _body_start(text)
    if body is None:
        return None
    header = text[:body].strip()
    match = _METHOD_RE.search(header)
    if match is None:
        return None
    return f"{header[: match.end()]} {{:axiom}}{header[match.end() :]}"


def _callee_wiring(
    direct: tuple[FunctionRef, ...], callees: Mapping[FunctionRef, Result]
) -> tuple[dict[str, str], tuple[str, ...], tuple[Assumption, ...], str]:
    """The callee info for the transpiler and the prover: module names, extra modules, assumptions, module source.

    A proven callee passes its certificate source as an extra module, under its Dafny module name. An unproven
    (``open``) callee becomes an ``UNPROVEN_CALLEE`` assumption with a body-less method together with the callee's
    Dafny spec in the generated ``PCC_assume_<hash>`` module, where ``<hash>`` is the first 8 hex digits of the
    sha256 of the assumed specs (the same truncation ``verifier.layout.module_name`` uses). An open callee with no
    translated method is still listed, with empty Dafny text, so the caller records what it rests on. With no open
    callee that has a translated method the module source is empty.
    """
    modules: dict[str, str] = {}
    extra: list[str] = []
    assumptions: list[Assumption] = []
    methods: list[str] = []
    specs: list[str] = []
    for callee in direct:
        outcome = callees[callee]
        if outcome.state is State.PROVEN:
            assert outcome.proof is not None  # a proven result always carries its proof
            modules[callee.qualname] = module_name(callee)
            extra.append(outcome.proof.dafny_source)
        elif outcome.state is State.OPEN:
            method = _assumption_method(outcome)
            if method is not None:
                methods.append(method)
            if outcome.translation is not None:
                specs.append(outcome.translation.dafny_spec)
            assumptions.append(
                Assumption(
                    AssumptionKind.UNPROVEN_CALLEE,
                    callee.qualname,
                    f"the spec of {callee.qualname} is assumed",
                    method or "",
                )
            )
    assumptions_module = ""
    if methods:
        digest = hashlib.sha256("\n".join(specs).encode("utf-8")).hexdigest()[:8]
        blocks = [*specs, *methods]
        body = "\n".join(f"  {line}" if line.strip() else "" for block in blocks for line in block.split("\n"))
        assumptions_module = f"module PCC_assume_{digest} {{\n{body}\n}}\n"
    return modules, tuple(extra), tuple(assumptions), assumptions_module


def dependency_hashes(function: FunctionRef, callees: Mapping[FunctionRef, Result]) -> tuple[Dependency, ...]:
    """The certificate dependencies of ``function``: the sha256 of each direct callee's ``dafny_spec``.

    The certificate writer passes these to ``certificate.from_result``; callees without a translated spec are
    skipped.
    """
    found: list[Dependency] = []
    for callee in _direct_callees(function, callees):
        translation = callees[callee].translation
        if translation is not None:
            found.append(
                Dependency(callee.qualname, hashlib.sha256(translation.dafny_spec.encode("utf-8")).hexdigest())
            )
    return tuple(found)


def _closure(start: FunctionRef, edges: Mapping[FunctionRef, frozenset[FunctionRef]]) -> set[FunctionRef]:
    """Every function reachable from ``start`` following ``edges``."""
    seen: set[FunctionRef] = set()
    stack = list(edges.get(start, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edges.get(node, ()))
    return seen


def _mutual_members(level: Sequence[FunctionRef]) -> set[FunctionRef]:
    """Level members that share a multi-member strongly connected component (mutual recursion).

    A function that calls only itself stays out: self-recursion is checked normally.
    """
    members = set(level)
    if len(members) < 2:
        return set()
    edges = subset_edges(tuple(members))
    reachable = {ref: _closure(ref, edges) for ref in members}
    return {
        ref
        for ref in members
        if any(other != ref and other in reachable[ref] and ref in reachable[other] for other in members)
    }


def _translate(
    function: FunctionRef,
    spec: Spec,
    repo_root: Path,
    *,
    client: ModelClient,
    config: Config,
    callee_modules: Mapping[str, str],
) -> Translation:
    """Translate the function, falling back to AI translation with equivalence rechecks (docs/plan.md, step 4).

    The transpiler translates what it supports, with each callee qualname mapped to its Dafny module name; a
    construct it does not support falls back to AI translation. On the AI path the translation is rechecked after
    every attempt and retranslated with the mismatch as a hint, up to ``max_retranslations`` times; after that the
    function is ``not_checkable``.
    """
    dafny_spec = transpile_spec(spec)
    mismatch: Counterexample | None = None
    try:
        # Calls to project functions are not translated yet: the rule that turns them into
        # `<Module>.<name>(...)` calls with the mapping below lands with the subset of #44.
        transpiled: str | None = transpile_function(function, callees=callee_modules)
    except UnsupportedConstruct:
        transpiled = None
    if transpiled is not None:
        translation = Translation(
            function, TranslationPath.TRANSPILER, TranslationPath.TRANSPILER, transpiled, dafny_spec
        )
        report = check_equivalence(
            function,
            translation,
            repo_root,
            inputs=config.equivalence.differential_inputs,
            symbolic_seconds=config.equivalence.symbolic_seconds,
            client=client,
            model=config.models.edge_cases,
        )
        if report.mismatch is None:
            return dataclasses.replace(translation, checks=(*translation.checks, *report.records))
        mismatch = report.mismatch
    for _ in range(config.equivalence.max_retranslations + 1):
        candidate = translate(
            function,
            spec,
            client=client,
            model=config.models.translation,
            dafny_spec=dafny_spec,
            mismatch=mismatch,
            resolve_retries=config.transpile.ai_resolve_retries,
        )
        report = check_equivalence(
            function,
            candidate,
            repo_root,
            inputs=config.equivalence.differential_inputs,
            symbolic_seconds=config.equivalence.symbolic_seconds,
            client=client,
            model=config.models.edge_cases,
        )
        if report.mismatch is None:
            return dataclasses.replace(candidate, checks=(*candidate.checks, *report.records))
        mismatch = report.mismatch
    raise NotCheckable("AI translation differs from the original")


def run(
    function: FunctionRef,
    repo_root: Path,
    *,
    client: ModelClient,
    config: Config,
    callees: Mapping[FunctionRef, Result] = MappingProxyType({}),
) -> Result:
    """Steps 1 to 7 for one function; every expected failure becomes a result state, never an exception.

    ``callees`` holds the results of already-run callees in the project. Callees the function never calls are
    ignored; a callee with a counterexample or a ``not_checkable`` callee ends the run before any step, mutual
    recursion ends it ``open``. Otherwise a proven callee is linked by its certificate source and an unproven one
    by a generated assumptions module, and the run uses only the callees' specs. Expected step failures become
    result states; ``ToolError`` and malformed model answers from the advisory steps escape.
    """
    direct = _direct_callees(function, callees)
    gated = _callee_gate(function, direct, callees)
    if gated is not None:
        return gated
    callee_modules, extra_modules, assumptions, assumptions_module = _callee_wiring(direct, callees)
    intent: Intent | None = None
    spec: Spec | None = None
    translation: Translation | None = None
    try:
        context = collect(function, repo_root)
        if config.pipeline.search_agent:
            context = search_further(context, repo_root, client=client, model=config.models.context_search)
        intent = derive(context, client=client, model=config.models.intent)
        spec = write_spec(intent, client=client, model=config.models.spec)
        spec = check_spec(spec, function, repo_root, client=client, model=config.models.mutants, config=config.spec)
        translation = _translate(function, spec, repo_root, client=client, config=config, callee_modules=callee_modules)
        outcome = prove(
            translation,
            assumptions,
            client=client,
            stages=config.proof.prover_stages,
            resource_limit=config.dafny.resource_limit,
            function_lines=len(function.source.splitlines()),
            extra_modules=extra_modules,
            assumptions_module=assumptions_module,
        )
    except (NotCheckable, UnsupportedConstruct) as error:
        return Result(function, State.NOT_CHECKABLE, str(error) or "not checkable", intent, spec, translation)
    assert intent is not None and spec is not None and translation is not None
    if outcome.proven:
        return Result(function, State.PROVEN, "", intent, spec, translation, assumptions, None, outcome)
    failed_clause = outcome.errors[0].related if outcome.errors else None
    found = search_counterexample(
        function,
        spec,
        repo_root,
        failed_clause=failed_clause or None,
        client=client,
        model=config.models.counterexample,
        config=config.counterexample,
    )
    if found is not None:
        opinion = review(context, spec, client=client, model=config.models.second_opinion)
        return Result(function, State.COUNTEREXAMPLE, "", intent, spec, translation, assumptions, found, None, opinion)
    opinion = review(context, spec, client=client, model=config.models.second_opinion)
    return Result(function, State.OPEN, outcome.last_error or "no proof found", intent, spec, translation, assumptions)


def run_many(
    functions: Sequence[FunctionRef], repo_root: Path, *, client: ModelClient, config: Config
) -> tuple[Result, ...]:
    """Run in ``pcc.project.call_order``, reusing proven callee specs; functions of one level run in parallel.

    Mutually recursive functions of one level end ``open`` without running; every other function runs with the
    results of the earlier levels (plus the mutual ones of its own level) as ``callees``. The result holds one
    entry per passed function, in order. The shared ``client`` is called from several worker threads at once,
    so it must be thread-safe.
    """
    levels = call_order(functions)
    done: dict[FunctionRef, Result] = {}
    for level in levels:
        for ref in _mutual_members(level):
            done[ref] = Result(ref, State.OPEN, MUTUAL_RECURSION)
        rest = [ref for ref in level if ref not in done]
        if rest:
            snapshot: Mapping[FunctionRef, Result] = MappingProxyType(dict(done))
            with ThreadPoolExecutor(max_workers=max(1, config.pipeline.workers)) as pool:
                calls = pool.map(
                    partial(run, repo_root=repo_root, client=client, config=config, callees=snapshot),
                    rest,
                )
                for ref, outcome in zip(rest, calls, strict=True):
                    done[ref] = outcome
    return tuple(done[ref] for ref in functions)


__all__ = ["MUTUAL_RECURSION", "dependency_hashes", "run", "run_many"]
