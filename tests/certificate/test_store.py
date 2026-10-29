"""Writing, reading and staleness checks of proof certificates (docs/plan.md, section Certificate)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

import pytest

from pcc.certificate import from_result, is_stale, load, write
from pcc.certificate.store import STANDING_ASSUMPTIONS
from pcc.config import Config
from pcc.core import (
    Assumption,
    AssumptionKind,
    CheckKind,
    CheckRecord,
    Counterexample,
    Dependency,
    ProofOutcome,
    Result,
    Spec,
    State,
    Translation,
    TranslationPath,
)
from pcc.pylib import PYLIB_FILE
from tests.helpers import function_ref

PREDICATE = "def spec(xs, result):\n    return result == len(xs)\n"
PRECONDITION = "def pre(xs):\n    return True\n"

DAFNY_SPEC = "predicate SpecFunction(xs: seq<int>, result: int)\n{\n  result == |xs|\n}\n"

CERT_ALPHA_DAFNY = """module CertAlpha {
  function Next(x: int): int
    ensures Next(x) == x + 1
  {
    x + 1
  }
}
"""

CERT_BETA_DAFNY = """module CertBeta {
  function Twice(x: int): int
    ensures Twice(x) == 2 * x
  {
    2 * x
  }
}
"""

AUDIT_DAFNY = """module FxAssumptions {
  // Contract for the unproven callee "fx.external", recorded as an assumption of this certificate.
  function {:axiom} FxExternal(x: int): int
    ensures FxExternal(x) >= x
}

module FxMain {
  import opened FxAssumptions

  function FxId(x: int): int
    requires x >= 0
    ensures FxId(x) >= x
  {
    FxExternal(x)
  }
}
"""


def proven_result(
    *,
    path: str = "src/cart.py",
    qualname: str = "Cart.total",
    source: str = "def total(xs):\n    return len(xs)\n",
    predicate: str | None = PREDICATE,
    precondition: str | None = PRECONDITION,
    dafny_source: str = CERT_ALPHA_DAFNY,
    assumptions: tuple[Assumption, ...] = (),
) -> Result:
    """A minimal ``PROVEN`` result holding ``dafny_source`` as its verified proof."""
    function = function_ref(source, path=path, qualname=qualname)
    spec = Spec(function, predicate or PREDICATE, precondition)
    translation = Translation(
        function,
        TranslationPath.TRANSPILER,
        TranslationPath.TRANSPILER,
        "dafny function",
        DAFNY_SPEC,
        (CheckRecord(CheckKind.DIFFERENTIAL, True),),
    )
    proof = ProofOutcome(True, dafny_source, 1, "spark")
    return Result(
        function, State.PROVEN, "verified", spec=spec, translation=translation, assumptions=assumptions, proof=proof
    )


def test_from_result_builds_certificate() -> None:
    deps = (Dependency("pricing.discount", "ab" * 32),)
    assumptions = (Assumption(AssumptionKind.LIBRARY_CONTRACT, "json.loads", "parses JSON"),)
    result = proven_result(assumptions=assumptions)
    certificate = from_result(result, Config(), dependency_hashes=deps)

    assert certificate.function == result.function
    assert certificate.dafny_source == CERT_ALPHA_DAFNY
    assert certificate.spec_python == PREDICATE + "\n" + PRECONDITION
    assert certificate.spec_dafny == DAFNY_SPEC
    assert certificate.function_path is TranslationPath.TRANSPILER
    assert certificate.spec_path is TranslationPath.TRANSPILER
    assert result.translation is not None
    assert certificate.translation_checks == result.translation.checks
    assert certificate.assumptions == assumptions
    assert certificate.dafny_version == Config().dafny.version
    assert certificate.pylib_sha256 == hashlib.sha256(PYLIB_FILE.read_bytes()).hexdigest()
    assert certificate.pcc_version == importlib.metadata.version("pcc")
    assert certificate.dependency_hashes == deps


def test_from_result_spec_python_without_precondition() -> None:
    certificate = from_result(proven_result(precondition=None), Config())
    assert certificate.spec_python == PREDICATE


def test_from_result_rejects_non_proven() -> None:
    function = function_ref()
    counterexample = Counterexample("[]", "non-empty", "0", "def test_broken():\n    assert False\n")
    cases = [
        Result(function, State.OPEN, "budget"),
        Result(function, State.COUNTEREXAMPLE, "breaks spec", counterexample=counterexample),
        Result(function, State.NOT_CHECKABLE, "no types"),
    ]
    for result in cases:
        with pytest.raises(ValueError, match="proven"):
            from_result(result, Config())


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    certificate = from_result(proven_result(), Config())
    written = write(certificate, tmp_path / "proofs")
    assert written == tmp_path / "proofs" / "src" / "cart" / "Cart.total.dfy"
    assert written.read_text(encoding="utf-8") == CERT_ALPHA_DAFNY
    assert load(written) == certificate


def test_write_layout_and_sidecars(tmp_path: Path) -> None:
    proofs_dir = tmp_path / "proofs"
    certificate = from_result(proven_result(), Config())
    written = write(certificate, proofs_dir)

    sidecar = written.with_suffix(".json")
    assert sidecar.is_file()
    assert "standing_assumptions" in sidecar.read_text(encoding="utf-8")

    pylib_copy = proofs_dir / "PyLib.dfy"
    assert pylib_copy.is_file()
    assert pylib_copy.read_bytes() == PYLIB_FILE.read_bytes()

    toml = proofs_dir / "dfyconfig.toml"
    limit = Config().dafny.resource_limit
    assert toml.read_text(encoding="utf-8") == (
        f'includes = ["**/*.dfy"]\n\n[options]\nresource-limit = {limit}\nverification-time-limit = 0\n'
    )


def test_write_stores_all_other_fields_as_json(tmp_path: Path) -> None:
    certificate = from_result(proven_result(), Config())
    written = write(certificate, tmp_path / "proofs")

    import json

    data = json.loads(written.with_suffix(".json").read_text(encoding="utf-8"))
    assert "dafny_source" not in data
    assert data["standing_assumptions"] == list(STANDING_ASSUMPTIONS)
    assert STANDING_ASSUMPTIONS == ("inputs match the type hints", "strings contain no lone surrogates")
    assert data["spec_python"] == certificate.spec_python
    assert data["pylib_sha256"] == certificate.pylib_sha256


def test_write_twice_is_byte_identical(tmp_path: Path) -> None:
    certificate = from_result(proven_result(), Config())
    proofs_dir = tmp_path / "proofs"
    first = write(certificate, proofs_dir)
    before = {path: path.read_bytes() for path in sorted(proofs_dir.rglob("*")) if path.is_file()}
    second = write(certificate, proofs_dir)
    after = {path: path.read_bytes() for path in sorted(proofs_dir.rglob("*")) if path.is_file()}
    assert second == first
    assert after == before


def test_write_nested_qualname(tmp_path: Path) -> None:
    certificate = from_result(proven_result(path="src/util.py", qualname="outer.inner", source="x = 1\n"), Config())
    written = write(certificate, tmp_path / "proofs")
    assert written == tmp_path / "proofs" / "src" / "util" / "outer.inner.dfy"
    assert load(written) == certificate


def test_load_missing_json_raises(tmp_path: Path) -> None:
    orphan = tmp_path / "orphan.dfy"
    orphan.write_text("module Orphan {}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load(orphan)


def test_is_stale_false_when_nothing_changed() -> None:
    certificate = from_result(proven_result(), Config(), dependency_hashes=(Dependency("pricing.discount", "ab" * 32),))
    assert (
        is_stale(
            certificate,
            certificate.function,
            dependency_hashes=(Dependency("pricing.discount", "ab" * 32),),
            pylib_sha256=certificate.pylib_sha256,
        )
        is False
    )


def test_is_stale_true_when_source_changed() -> None:
    certificate = from_result(proven_result(), Config())
    changed = function_ref("def total(xs):\n    return len(xs) \n")
    assert is_stale(certificate, changed, pylib_sha256=certificate.pylib_sha256) is True


def test_is_stale_true_when_pylib_changed() -> None:
    certificate = from_result(proven_result(), Config())
    assert is_stale(certificate, certificate.function, pylib_sha256="0" * 64) is True


def test_is_stale_dependency_hashes() -> None:
    recorded = (Dependency("pricing.discount", "ab" * 32),)
    certificate = from_result(proven_result(), Config(), dependency_hashes=recorded)
    assert (
        is_stale(
            certificate,
            certificate.function,
            dependency_hashes=(Dependency("pricing.discount", "cd" * 32),),
            pylib_sha256=certificate.pylib_sha256,
        )
        is True
    )
    assert (
        is_stale(certificate, certificate.function, dependency_hashes=(), pylib_sha256=certificate.pylib_sha256) is True
    )
    assert (
        is_stale(
            certificate,
            certificate.function,
            dependency_hashes=(*recorded, Dependency("other.dep", "ef" * 32)),
            pylib_sha256=certificate.pylib_sha256,
        )
        is False
    )


@pytest.mark.dafny
def test_verify_proofs_toml(tmp_path: Path) -> None:
    """Two written certificates verify together via ``dafny verify proofs/dfyconfig.toml``."""
    proofs_dir = tmp_path / "proofs"
    write(from_result(proven_result(qualname="next", dafny_source=CERT_ALPHA_DAFNY), Config()), proofs_dir)
    write(
        from_result(
            proven_result(path="demo/beta.py", qualname="twice", dafny_source=CERT_BETA_DAFNY),
            Config(),
        ),
        proofs_dir,
    )
    proc = subprocess.run(
        ["dafny", "verify", str(proofs_dir / "dfyconfig.toml")],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.dafny
def test_audit_proofs_reports_assumptions(tmp_path: Path) -> None:
    """``dafny audit`` runs on written proofs with an assumptions module; findings are recorded, not asserted."""
    certificate = from_result(
        proven_result(
            path="demo/fx.py",
            qualname="fx_id",
            source="def fx_id(x):\n    return x\n",
            dafny_source=AUDIT_DAFNY,
            assumptions=(Assumption(AssumptionKind.UNPROVEN_CALLEE, "fx.external", "trusted contract"),),
        ),
        Config(),
    )
    proofs_dir = tmp_path / "proofs"
    write(certificate, proofs_dir)
    verify = subprocess.run(
        ["dafny", "verify", str(proofs_dir / "dfyconfig.toml")],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    audit = subprocess.run(
        ["dafny", "audit", str(proofs_dir / "dfyconfig.toml")],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert audit.returncode == 0, audit.stdout + audit.stderr
