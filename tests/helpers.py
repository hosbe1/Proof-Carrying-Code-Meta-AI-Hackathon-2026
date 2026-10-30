"""Factories shared by the tests."""

from __future__ import annotations

from pcc.core import (
    Assumption,
    AssumptionKind,
    Certificate,
    CheckKind,
    CheckRecord,
    Claim,
    Contradiction,
    Counterexample,
    Evidence,
    FunctionRef,
    Intent,
    ProofOutcome,
    Result,
    SecondOpinion,
    SourceKind,
    Spec,
    State,
    Translation,
    TranslationPath,
    Usage,
    source_hash,
)

DEFAULT_SOURCE = "def total(self): ..."


def function_ref(
    source: str = DEFAULT_SOURCE,
    *,
    path: str = "src/cart.py",
    qualname: str = "Cart.total",
    lineno: int = 12,
) -> FunctionRef:
    """A ``FunctionRef`` whose ``source_hash`` is computed from ``source``, as the reader does."""
    return FunctionRef(path, qualname, lineno, source, source_hash(source))


def evidence(
    kind: SourceKind = SourceKind.DOCSTRING,
    ref: str = "README.md",
    excerpt: str = "the total includes discounts",
) -> Evidence:
    """One piece of evidence for an intent claim."""
    return Evidence(kind, ref, excerpt)


def claim(
    text: str = "the total includes discounts",
    sources: tuple[Evidence, ...] = (),
) -> Claim:
    """An intent claim, with its source refs."""
    resolved: tuple[Evidence, ...] = sources or (evidence(),)
    return Claim(text, resolved)


def contradiction(
    description: str = "the docstring and the tests disagree on empty carts",
    sources: tuple[Evidence, ...] = (),
    resolved_by: SourceKind | None = SourceKind.TEST,
) -> Contradiction:
    """A contradiction between intent sources."""
    resolved: tuple[Evidence, ...] = sources or (evidence(), evidence(SourceKind.TEST, "tests/test_cart.py"))
    return Contradiction(description, resolved, resolved_by)


def intent(
    function: FunctionRef | None = None,
    *,
    summary: str = "Returns the total price of the cart.",
    claims: tuple[Claim, ...] = (),
    contradictions: tuple[Contradiction, ...] = (),
) -> Intent:
    """An intent with one claim by default."""
    resolved: tuple[Claim, ...] = claims or (claim(),)
    return Intent(function or function_ref(), summary, resolved, contradictions)


def check_record(
    kind: CheckKind = CheckKind.HYPOTHESIS_ORIGINAL,
    passed: bool = True,
    detail: str = "",
    inputs_tried: int | None = None,
    inputs_total: int | None = None,
) -> CheckRecord:
    """A check record, optionally with the share of inputs a precondition accepts."""
    return CheckRecord(kind, passed, detail, inputs_tried, inputs_total)


def spec(
    function: FunctionRef | None = None,
    *,
    predicate: str = "def spec(self, result):\n    return result >= 0",
    precondition: str | None = "len(self.items) > 0",
    checks: tuple[CheckRecord, ...] = (),
) -> Spec:
    """A spec with a precondition by default."""
    resolved_checks: tuple[CheckRecord, ...] = checks or (check_record(),)
    return Spec(function or function_ref(), predicate, precondition, resolved_checks)


def translation(
    function: FunctionRef | None = None,
    *,
    function_path: TranslationPath = TranslationPath.TRANSPILER,
    spec_path: TranslationPath = TranslationPath.TRANSPILER,
    dafny_function: str = "function total(items: seq<int>): int",
    dafny_spec: str = "predicate spec(items: seq<int>, result: int)",
) -> Translation:
    """A translation, by the transpiler by default."""
    return Translation(function or function_ref(), function_path, spec_path, dafny_function, dafny_spec)


def assumption(
    kind: AssumptionKind = AssumptionKind.LIBRARY_CONTRACT,
    subject: str = "json.loads",
    statement: str = "returns the parsed value",
) -> Assumption:
    """One assumption a proof rests on."""
    return Assumption(kind, subject, statement)


def counterexample(
    *,
    arguments: str = "[3, -1], key=None",
    expected: str = "the maximum of the list",
    actual: str = "-1",
    test: str = "def test_maximum_breaks_spec():\n    assert maximum([3, -1]) == 3",
    suggested_fix: str | None = None,
) -> Counterexample:
    """A counterexample with a ready-to-run pytest function."""
    return Counterexample(arguments, expected, actual, test, suggested_fix)


def proof(
    *,
    proven: bool = True,
    dafny_source: str = "module CartTotal {\n  method total() {}\n}",
    attempts: int = 1,
    stage: str = "glimmer",
) -> ProofOutcome:
    """A proof outcome, proven by default."""
    return ProofOutcome(proven, dafny_source, attempts, stage)


def second_opinion(
    *,
    agrees: bool = True,
    alternative_predicate: str | None = None,
    note: str = "agrees.",
) -> SecondOpinion:
    """A second opinion that agrees by default."""
    return SecondOpinion(agrees, alternative_predicate, note)


def usage(
    step: str = "prove",
    model: str | None = "muse-spark-1.3",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.0,
    seconds: float = 1.0,
) -> Usage:
    """One usage record of a pipeline step."""
    return Usage(step, model, input_tokens, output_tokens, cost_usd, seconds)


def result(
    function: FunctionRef | None = None,
    state: State = State.PROVEN,
    reason: str = "",
    *,
    with_intent: bool = True,
    with_translation: bool = True,
) -> Result:
    """A result carrying what its state needs: a proven proof for ``PROVEN``.

    Pass explicit ``None``s through the keyword overrides for states that carry less.
    """
    resolved = function or function_ref()
    if state is State.PROVEN:
        return Result(
            resolved,
            state,
            reason,
            intent(resolved) if with_intent else None,
            spec(resolved),
            translation(resolved) if with_translation else None,
            (),
            None,
            proof(),
        )
    if state is State.COUNTEREXAMPLE:
        return Result(
            resolved,
            state,
            reason,
            intent(resolved) if with_intent else None,
            spec(resolved),
            translation(resolved) if with_translation else None,
            (),
            counterexample(),
        )
    return Result(
        resolved,
        state,
        reason or ("no proof found" if state is State.OPEN else "no type hints"),
        intent(resolved) if with_intent else None,
        spec(resolved) if state is not State.NOT_CHECKABLE else None,
        translation(resolved) if with_translation and state is not State.NOT_CHECKABLE else None,
    )


def certificate(
    function: FunctionRef | None = None,
    *,
    dafny_source: str = "module CartTotal {\n  method total() {}\n}",
) -> Certificate:
    """A stored proof certificate for a proven function."""
    resolved = function or function_ref()
    return Certificate(
        resolved,
        dafny_source,
        "def spec(self, result):\n    return result >= 0",
        "predicate spec(result: int)",
        TranslationPath.TRANSPILER,
        TranslationPath.TRANSPILER,
        (check_record(CheckKind.DAFNY_VERIFY),),
        (),
        "4.11.0",
        "abc123",
        "0.1.0",
    )
