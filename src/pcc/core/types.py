"""Data types shared by every pipeline step.

All types are immutable and JSON-serializable (see ``pcc.core.serialize``), because they end up in certificates,
caches and evaluation logs. Changing or removing a field affects every step and needs a ticket of its own; additive
changes are named in the pull request.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class State(StrEnum):
    """The four results a function can end in (docs/plan.md, "Result states")."""

    PROVEN = "proven"
    COUNTEREXAMPLE = "counterexample"
    OPEN = "open"
    NOT_CHECKABLE = "not_checkable"


class SourceKind(StrEnum):
    """Where a piece of evidence about a function's intent comes from."""

    TEST = "test"
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    COMMIT = "commit"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    CALL_SITE = "call_site"
    DOCS = "docs"
    CONVENTION = "convention"
    SIGNATURE = "signature"
    NAME = "name"


# Ranking used when sources contradict each other (docs/plan.md, step 2). A lower tier wins. Sources in one tier have
# the same rank, so the ranking does not decide a contradiction between them.
SOURCE_RANK: tuple[frozenset[SourceKind], ...] = (
    frozenset({SourceKind.TEST}),
    frozenset({SourceKind.ISSUE, SourceKind.PULL_REQUEST}),
    frozenset({SourceKind.COMMIT}),
    frozenset({SourceKind.DOCSTRING}),
    frozenset({SourceKind.COMMENT}),
    # The plan ranks these below comments, in this order.
    frozenset({SourceKind.CALL_SITE}),
    frozenset({SourceKind.DOCS}),
    frozenset({SourceKind.CONVENTION}),
    frozenset({SourceKind.SIGNATURE}),
    # The plan ranks the name last.
    frozenset({SourceKind.NAME}),
)


def source_rank(kind: SourceKind) -> int:
    """Tier of ``kind`` in ``SOURCE_RANK``. The lower number wins a contradiction."""
    for rank, tier in enumerate(SOURCE_RANK):
        if kind in tier:
            return rank
    raise ValueError(f"no rank for source {kind!r}")


def source_hash(source: str) -> str:
    """sha256 hex digest of ``source`` encoded as UTF-8, with ``\\r\\n`` and ``\\r`` line endings read as ``\\n``.

    Two checkouts that differ only in line endings get the same hash, so a certificate is not stale for that reason.
    """
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class TranslationPath(StrEnum):
    TRANSPILER = "transpiler"
    AI = "ai"


class CheckKind(StrEnum):
    HYPOTHESIS_ORIGINAL = "hypothesis_original"  # spec holds on the original code
    MUTANTS = "mutants"  # spec rejects broken variants
    DIFFERENTIAL = "differential"  # back-compiled Dafny equals the original on many inputs
    SYMBOLIC = "symbolic"  # crosshair diffbehavior found no difference
    DAFNY_VERIFY = "dafny_verify"
    ANTI_CHEAT = "anti_cheat"  # no forbidden form outside the assumptions module, spec unchanged
    CANDIDATE_FIX = "candidate_fix"  # a surviving mutant that fixes the original is a possible fix


class AssumptionKind(StrEnum):
    LIBRARY_CONTRACT = "library_contract"
    UNPROVEN_CALLEE = "unproven_callee"
    RECURSIVE_CALLEE = "recursive_callee"
    SOURCE_CONTRADICTION = "source_contradiction"


@dataclass(frozen=True, slots=True)
class FunctionRef:
    """One Python function, identified by file and qualified name.

    ``source_hash`` is ``source_hash(source)``, computed when the function is read. A certificate is stale when the
    hash of the current source differs from it. Build a ``FunctionRef`` from its source with the hash computed; a wrong
    hash raises ``ValueError``.
    """

    path: str  # relative to the repository root, POSIX separators
    qualname: str  # e.g. "Cart.total" or "maximum"
    lineno: int
    source: str
    source_hash: str  # ``source_hash(source)``; a certificate is stale when this changes

    def __post_init__(self) -> None:
        if self.source_hash != source_hash(self.source):
            raise ValueError(f"source_hash of {self.qualname} does not match its source; use source_hash(source)")


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: SourceKind
    ref: str  # e.g. "tests/test_cart.py::test_discount", "commit a3f9c1e", "#12"
    excerpt: str


@dataclass(frozen=True, slots=True)
class Context:
    function: FunctionRef
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    sources: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class Contradiction:
    description: str
    sources: tuple[Evidence, ...]
    resolved_by: SourceKind | None  # the source that won by SOURCE_RANK; None if unresolvable


@dataclass(frozen=True, slots=True)
class Intent:
    """What the function should do, in plain language, with a source for every claim."""

    function: FunctionRef
    summary: str
    claims: tuple[Claim, ...]
    contradictions: tuple[Contradiction, ...] = ()


@dataclass(frozen=True, slots=True)
class CheckRecord:
    kind: CheckKind
    passed: bool
    detail: str = ""
    inputs_tried: int | None = None
    inputs_total: int | None = None  # with inputs_tried, the share of inputs pre accepts


@dataclass(frozen=True, slots=True)
class Spec:
    """An executable Python predicate ``spec(<args>, result) -> bool`` plus optional preconditions."""

    function: FunctionRef
    predicate: str  # source of a function named ``spec``
    precondition: str | None = None  # source of a function named ``pre``, same arguments as the function
    checks: tuple[CheckRecord, ...] = ()
    precondition_sources: tuple[Claim, ...] = ()  # every clause of pre cites one of these claims


@dataclass(frozen=True, slots=True)
class Translation:
    function: FunctionRef
    function_path: TranslationPath  # how the function was translated
    spec_path: TranslationPath  # how the spec was translated; can differ from the function's path
    dafny_function: str  # the function, translated (code mode)
    dafny_spec: str  # the spec, translated (spec mode)
    checks: tuple[CheckRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class Assumption:
    kind: AssumptionKind
    subject: str  # e.g. "json.loads" or "pricing.discount"
    statement: str  # plain language
    dafny: str = ""  # the contract as Dafny, when there is one


@dataclass(frozen=True, slots=True)
class Dependency:
    """One hashed dependency of a certificate: a callee spec, a contract or PyLib."""

    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Counterexample:
    arguments: str  # Python source of the call arguments, e.g. "[3, -1], key=None"
    expected: str  # what the spec demands, in plain language
    actual: str  # repr of what the code returned or raised
    test: str  # a ready-to-run pytest function
    suggested_fix: str | None = None  # a surviving mutant that fixes the original, if one was found


@dataclass(frozen=True, slots=True)
class EquivalenceReport:
    """Outcome of checking a translation against the original (differential and symbolic)."""

    records: tuple[CheckRecord, ...]
    mismatch: Counterexample | None = None


@dataclass(frozen=True, slots=True)
class DafnyError:
    line: int
    column: int
    message: str
    related: str = ""  # e.g. the ``ensures`` clause that could not be proven


@dataclass(frozen=True, slots=True)
class ProofOutcome:
    proven: bool
    dafny_source: str  # the complete Dafny source that was verified last
    attempts: int
    stage: str  # which prover stage produced the outcome, e.g. "glimmer"
    last_error: str = ""
    errors: tuple[DafnyError, ...] = ()  # Dafny's errors, parsed


@dataclass(frozen=True, slots=True)
class SecondOpinion:
    agrees: bool
    alternative_predicate: str | None
    note: str


@dataclass(frozen=True, slots=True)
class Usage:
    step: str
    model: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Result:
    """The outcome for one function. The state decides what it may carry; ``__post_init__`` checks these rules:

    - a counterexample only with state ``COUNTEREXAMPLE``, and a counterexample result needs one;
    - a proof with ``proven=True`` only with state ``PROVEN``, and a proven result needs one;
    - a ``NOT_CHECKABLE`` result needs a non-empty reason.
    """

    function: FunctionRef
    state: State
    reason: str
    intent: Intent | None = None
    spec: Spec | None = None
    translation: Translation | None = None
    assumptions: tuple[Assumption, ...] = ()
    counterexample: Counterexample | None = None
    proof: ProofOutcome | None = None
    second_opinion: SecondOpinion | None = None
    usage: tuple[Usage, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.state is State.NOT_CHECKABLE and not self.reason.strip():
            raise ValueError("a not_checkable result needs a reason")
        if self.counterexample is not None and self.state is not State.COUNTEREXAMPLE:
            raise ValueError(f"a {self.state.value} result cannot carry a counterexample")
        if self.proof is not None and self.proof.proven and self.state is not State.PROVEN:
            raise ValueError(f"a {self.state.value} result cannot carry a proof with proven=True")
        if self.state is State.COUNTEREXAMPLE and self.counterexample is None:
            raise ValueError("a counterexample result needs a counterexample")
        if self.state is State.PROVEN and (self.proof is None or not self.proof.proven):
            raise ValueError("a proven result needs a proof with proven=True")


@dataclass(frozen=True, slots=True)
class Certificate:
    """What is stored under proofs/ for a proven function; anyone can re-check it with ``dafny verify``.

    ``dafny_source`` is the complete Dafny source that ``dafny verify`` checks.
    """

    function: FunctionRef
    dafny_source: str
    spec_python: str
    spec_dafny: str
    function_path: TranslationPath
    spec_path: TranslationPath
    translation_checks: tuple[CheckRecord, ...]
    assumptions: tuple[Assumption, ...]
    dafny_version: str
    pylib_sha256: str  # hash of the PyLib.dfy the proof was checked against; re-checking needs that exact file
    pcc_version: str
    dependency_hashes: tuple[Dependency, ...] = ()  # hashes of the callee specs, contracts and PyLib this proof uses


# Every public name of this module. ``pcc.core`` re-exports exactly these, so a new type is added here and nowhere else.
__all__ = [
    "SOURCE_RANK",
    "Assumption",
    "AssumptionKind",
    "Certificate",
    "CheckKind",
    "CheckRecord",
    "Claim",
    "Context",
    "Contradiction",
    "Counterexample",
    "DafnyError",
    "Dependency",
    "EquivalenceReport",
    "Evidence",
    "FunctionRef",
    "Intent",
    "ProofOutcome",
    "Result",
    "SecondOpinion",
    "SourceKind",
    "Spec",
    "State",
    "Translation",
    "TranslationPath",
    "Usage",
    "source_hash",
    "source_rank",
]
