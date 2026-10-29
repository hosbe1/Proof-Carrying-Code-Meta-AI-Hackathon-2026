from __future__ import annotations

import math

import pytest

from pcc.config import Config, EquivalenceConfig, ProofConfig, ProverStage
from pcc.core import (
    Assumption,
    AssumptionKind,
    Certificate,
    CheckKind,
    CheckRecord,
    Claim,
    Context,
    Contradiction,
    Counterexample,
    DafnyError,
    Dependency,
    Evidence,
    FunctionRef,
    Intent,
    ProofOutcome,
    Result,
    SourceKind,
    Spec,
    State,
    Translation,
    TranslationPath,
    Usage,
)
from pcc.core.serialize import dumps, from_data, loads, to_data
from tests.helpers import function_ref

FN = function_ref()
EVIDENCE = Evidence(SourceKind.TEST, "tests/test_cart.py::test_total", "assert cart.total() == 10")


def full_result() -> Result:
    intent = Intent(
        FN,
        "Sum of all item prices.",
        (Claim("Empty cart totals 0.", (EVIDENCE,)),),
        (Contradiction("Docstring says rounded.", (EVIDENCE,), SourceKind.TEST),),
    )
    spec = Spec(FN, "def spec(self, result): ...", None, (CheckRecord(CheckKind.MUTANTS, True, "12/12", 12),))
    return Result(
        function=FN,
        state=State.PROVEN,
        reason="verified",
        intent=intent,
        spec=spec,
        translation=Translation(
            FN, TranslationPath.TRANSPILER, TranslationPath.AI, "function total(...)", "ensures ..."
        ),
        assumptions=(Assumption(AssumptionKind.LIBRARY_CONTRACT, "json.loads", "returns parsed JSON"),),
        proof=ProofOutcome(True, "module M {}", 2, "glimmer"),
        usage=(Usage("prove", "muse-glimmer-pcc", 100, 50, 0.01, 1.5), Usage("context", None)),
    )


def failed_result() -> Result:
    return Result(
        function=FN,
        state=State.COUNTEREXAMPLE,
        reason="the spec breaks on an empty cart",
        counterexample=Counterexample("[]", "0", "1", "def test_x(): ...", "xs.remove(0)"),
        proof=ProofOutcome(
            False,
            "module M {}",
            3,
            "spark",
            "Error: postcondition could not be proved",
            (DafnyError(3, 4, "postcondition could not be proved", "ensures ..."),),
        ),
    )


def full_certificate() -> Certificate:
    return Certificate(
        function=FN,
        dafny_source="module M {}",
        spec_python="def spec(xs, result): ...",
        spec_dafny="ensures ...",
        function_path=TranslationPath.TRANSPILER,
        spec_path=TranslationPath.TRANSPILER,
        translation_checks=(CheckRecord(CheckKind.DIFFERENTIAL, True, "10000/10000", 10000, 10000),),
        assumptions=(Assumption(AssumptionKind.LIBRARY_CONTRACT, "json.loads", "returns parsed JSON"),),
        dafny_version="4.11.0",
        pylib_sha256="abc123",
        pcc_version="0.1.0",
        dependency_hashes=(Dependency("pcc.pylib", "aaa"), Dependency("proj.pricing", "bbb")),
    )


def full_spec() -> Spec:
    return Spec(
        FN,
        "def spec(xs, result): ...",
        "def pre(xs): ...",
        (CheckRecord(CheckKind.MUTANTS, True, "12/12", 12),),
        (Claim("Empty cart totals 0.", (EVIDENCE,)),),
    )


def test_round_trip_of_a_full_result() -> None:
    result = full_result()
    assert loads(Result, dumps(result)) == result


def test_round_trip_of_a_counterexample_result() -> None:
    result = failed_result()
    assert loads(Result, dumps(result)) == result


def test_round_trip_of_a_certificate_with_dependencies() -> None:
    certificate = full_certificate()
    assert loads(Certificate, dumps(certificate)) == certificate


def test_round_trip_of_a_spec_with_precondition_sources() -> None:
    spec = full_spec()
    assert loads(Spec, dumps(spec)) == spec


def test_old_counterexample_json_without_suggested_fix_still_loads() -> None:
    data = {"arguments": "[]", "expected": "0", "actual": "1", "test": "def test_x(): ..."}
    assert from_data(Counterexample, data) == Counterexample("[]", "0", "1", "def test_x(): ...")


def test_old_certificate_json_without_dependency_hashes_still_loads() -> None:
    data = to_data(full_certificate())
    del data["dependency_hashes"]
    assert from_data(Certificate, data).dependency_hashes == ()


def test_old_check_record_json_without_inputs_total_still_loads() -> None:
    data = {**to_data(CheckRecord(CheckKind.MUTANTS, True, "12/12", 12)), "inputs_total": None}
    assert from_data(CheckRecord, data).inputs_total is None
    minimal = {"kind": "mutants", "passed": True}
    assert from_data(CheckRecord, minimal).inputs_total is None


def test_context_round_trip() -> None:
    context = Context(FN, (EVIDENCE,))
    assert from_data(Context, to_data(context)) == context


def test_config_round_trip() -> None:
    assert loads(Config, dumps(Config())) == Config()
    custom = Config(
        equivalence=EquivalenceConfig(differential_inputs=500),
        proof=ProofConfig(prover_stages=(ProverStage("spark", "muse-spark", 2),)),
    )
    assert loads(Config, dumps(custom)) == custom


def test_enums_are_stored_as_values() -> None:
    assert to_data(full_result())["state"] == "proven"


def test_missing_optional_fields_take_defaults() -> None:
    data = to_data(Result(FN, State.OPEN, "no proof"))
    del data["usage"]
    assert from_data(Result, data) == Result(FN, State.OPEN, "no proof")


def test_unsupported_values_are_rejected() -> None:
    with pytest.raises(TypeError):
        to_data({"a": {1, 2}})
    with pytest.raises(TypeError):
        from_data(FunctionRef, {**to_data(FN), "lineno": "12"})


def usage_data(**changes: object) -> dict[str, object]:
    return {**to_data(Usage("prove", None)), **changes}


def test_bool_is_not_an_int() -> None:
    with pytest.raises(TypeError, match="expected int, got bool"):
        from_data(Usage, usage_data(input_tokens=True))


def test_bool_is_not_a_float() -> None:
    with pytest.raises(TypeError, match="expected float, got bool"):
        from_data(Usage, usage_data(cost_usd=True))


def test_float_is_built_from_numbers_only() -> None:
    assert from_data(Usage, usage_data(cost_usd=2)).cost_usd == 2.0
    with pytest.raises(TypeError, match="expected float, got str"):
        from_data(Usage, usage_data(cost_usd="2.0"))


def test_nan_and_infinity_are_not_written() -> None:
    with pytest.raises(ValueError):
        dumps(Usage("prove", None, cost_usd=math.nan))
    with pytest.raises(ValueError):
        dumps(Usage("prove", None, seconds=math.inf))


def test_nan_is_not_read() -> None:
    with pytest.raises(ValueError, match="NaN"):
        loads(Usage, '{"step": "prove", "model": null, "cost_usd": NaN}')


def test_tuple_fields_need_a_json_list() -> None:
    with pytest.raises(TypeError, match="expected a list"):
        from_data(Context, {"function": to_data(FN), "evidence": "abc"})
    with pytest.raises(TypeError, match="expected a list"):
        from_data(Context, {"function": to_data(FN), "evidence": {}})


def test_unknown_enum_values_are_rejected_as_type_errors() -> None:
    data = {**to_data(Result(FN, State.OPEN, "no proof")), "state": "bogus"}
    with pytest.raises(TypeError, match="unknown value for State: 'bogus'"):
        from_data(Result, data)
    with pytest.raises(TypeError, match="unknown value for State"):
        from_data(Result, {**to_data(Result(FN, State.OPEN, "no proof")), "state": ["open"]})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="unknown fields for FunctionRef: lineo"):
        from_data(FunctionRef, {**to_data(FN), "lineo": 12})
