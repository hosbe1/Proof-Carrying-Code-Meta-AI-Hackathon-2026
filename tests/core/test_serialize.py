from __future__ import annotations

import math

import pytest

from pcc.config import Config, EquivalenceConfig, ProofConfig, ProverStage
from pcc.core import (
    Assumption,
    AssumptionKind,
    CheckKind,
    CheckRecord,
    Claim,
    Context,
    Contradiction,
    Counterexample,
    DafnyError,
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
        counterexample=Counterexample("[]", "0", "1", "def test_x(): ..."),
        proof=ProofOutcome(
            False,
            "module M {}",
            3,
            "spark",
            "Error: postcondition could not be proved",
            (DafnyError(3, 4, "postcondition could not be proved", "ensures ..."),),
        ),
    )


def test_round_trip_of_a_full_result() -> None:
    result = full_result()
    assert loads(Result, dumps(result)) == result


def test_round_trip_of_a_counterexample_result() -> None:
    result = failed_result()
    assert loads(Result, dumps(result)) == result


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
        to_data({"a": 1})
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
