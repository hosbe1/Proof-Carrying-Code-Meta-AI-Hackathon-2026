from __future__ import annotations

import pytest

from pcc.core import (
    SOURCE_RANK,
    Counterexample,
    FunctionRef,
    ProofOutcome,
    Result,
    SourceKind,
    State,
    source_hash,
    source_rank,
)
from pcc.core.serialize import from_data, to_data
from tests.helpers import function_ref

FN = function_ref()
COUNTEREXAMPLE = Counterexample("[]", "0", "1", "def test_x(): ...")


def proof(*, proven: bool) -> ProofOutcome:
    return ProofOutcome(proven, "module M {}", 1, "glimmer")


def test_every_source_kind_has_exactly_one_rank() -> None:
    ranked = [kind for tier in SOURCE_RANK for kind in tier]
    assert sorted(ranked) == sorted(SourceKind)


@pytest.mark.parametrize(
    ("higher", "lower"),
    [
        (SourceKind.TEST, SourceKind.ISSUE),
        (SourceKind.ISSUE, SourceKind.COMMIT),
        (SourceKind.COMMIT, SourceKind.DOCSTRING),
        (SourceKind.DOCSTRING, SourceKind.COMMENT),
        (SourceKind.COMMENT, SourceKind.CALL_SITE),
        (SourceKind.SIGNATURE, SourceKind.NAME),
    ],
)
def test_higher_sources_win(higher: SourceKind, lower: SourceKind) -> None:
    assert source_rank(higher) < source_rank(lower)


def test_issue_and_pull_request_share_a_rank() -> None:
    assert source_rank(SourceKind.ISSUE) == source_rank(SourceKind.PULL_REQUEST)


def test_name_is_ranked_last() -> None:
    assert source_rank(SourceKind.NAME) == len(SOURCE_RANK) - 1


def test_counterexample_result_needs_a_counterexample() -> None:
    with pytest.raises(ValueError, match="counterexample"):
        Result(FN, State.COUNTEREXAMPLE, "breaks the spec")


def test_counterexample_result_cannot_carry_a_proven_proof() -> None:
    Result(FN, State.COUNTEREXAMPLE, "breaks the spec", counterexample=COUNTEREXAMPLE, proof=proof(proven=False))
    with pytest.raises(ValueError, match="proven=True"):
        Result(FN, State.COUNTEREXAMPLE, "breaks the spec", counterexample=COUNTEREXAMPLE, proof=proof(proven=True))


def test_proven_result_needs_a_proof() -> None:
    with pytest.raises(ValueError, match="proof"):
        Result(FN, State.PROVEN, "verified")


def test_proven_result_needs_a_proof_that_holds() -> None:
    with pytest.raises(ValueError, match="proven=True"):
        Result(FN, State.PROVEN, "verified", proof=proof(proven=False))


def test_proven_result_cannot_carry_a_counterexample() -> None:
    with pytest.raises(ValueError, match="cannot carry a counterexample"):
        Result(FN, State.PROVEN, "verified", proof=proof(proven=True), counterexample=COUNTEREXAMPLE)


def test_open_and_not_checkable_need_no_proof_or_counterexample() -> None:
    assert Result(FN, State.OPEN, "no proof found").proof is None
    assert Result(FN, State.NOT_CHECKABLE, "the intent contradicts itself").counterexample is None


@pytest.mark.parametrize("state", [State.OPEN, State.NOT_CHECKABLE])
def test_only_a_counterexample_result_carries_a_counterexample(state: State) -> None:
    with pytest.raises(ValueError, match=f"a {state.value} result cannot carry a counterexample"):
        Result(FN, state, "reason", counterexample=COUNTEREXAMPLE)


@pytest.mark.parametrize("state", [State.OPEN, State.NOT_CHECKABLE])
def test_only_a_proven_state_carries_a_proof_with_proven_true(state: State) -> None:
    with pytest.raises(ValueError, match=f"a {state.value} result cannot carry a proof with proven=True"):
        Result(FN, state, "reason", proof=proof(proven=True))


def test_open_result_may_carry_an_unproven_proof() -> None:
    assert Result(FN, State.OPEN, "no proof found", proof=proof(proven=False)).proof is not None


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_not_checkable_needs_a_non_empty_reason(reason: str) -> None:
    with pytest.raises(ValueError, match="not_checkable result needs a reason"):
        Result(FN, State.NOT_CHECKABLE, reason)


def test_loaded_not_checkable_result_needs_a_reason() -> None:
    data = {**to_data(Result(FN, State.NOT_CHECKABLE, "contradiction")), "reason": ""}
    with pytest.raises(ValueError, match="needs a reason"):
        from_data(Result, data)


def test_function_ref_needs_the_hash_of_its_source() -> None:
    with pytest.raises(ValueError, match="source_hash"):
        FunctionRef("src/cart.py", "Cart.total", 12, "def total(self): ...", "ab" * 32)
    assert FN.source_hash == source_hash(FN.source)


def test_function_ref_hash_ignores_line_endings_like_source_hash() -> None:
    lf = "def f():\n    return 1\n"
    ref = FunctionRef("src/f.py", "f", 1, "def f():\r\n    return 1\r\n", source_hash(lf))
    assert ref.source_hash == source_hash(lf)


def test_loaded_function_ref_is_checked_too() -> None:
    data = {**to_data(FN), "source": "def total(self): return 0"}
    with pytest.raises(ValueError, match="source_hash"):
        from_data(FunctionRef, data)


def test_loaded_results_are_checked_too() -> None:
    data = to_data(Result(FN, State.PROVEN, "verified", proof=proof(proven=True)))
    data["proof"] = None
    with pytest.raises(ValueError, match="proof"):
        from_data(Result, data)


def test_source_hash_is_sha256_of_the_utf8_source() -> None:
    assert source_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert source_hash("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert source_hash("é") == "4a99557e4033c3539de2eb65472017cad5f9557f7a0625a09f1c3f6e2ba69c4c"


def test_source_hash_reads_crlf_and_cr_as_lf() -> None:
    lf = source_hash("def f():\n    return 1\n")
    assert source_hash("def f():\r\n    return 1\r\n") == lf
    assert source_hash("def f():\r    return 1\r") == lf


def test_source_hash_keeps_other_differences() -> None:
    assert source_hash("return 1") != source_hash("return 2")
    assert source_hash("x = 1\n") != source_hash("x = 1\n\n")
    assert source_hash("é") != source_hash("e")
