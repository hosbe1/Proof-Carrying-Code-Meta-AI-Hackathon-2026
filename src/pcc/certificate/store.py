"""Certificates under proofs/: writing, reading and detecting stale ones (docs/plan.md, section Certificate)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

from pcc.config import Config
from pcc.core import Certificate, Dependency, FunctionRef, Result, State, serialize
from pcc.pylib import PYLIB_FILE

#: The two assumptions every certificate carries: inputs match the type hints, and strings hold no lone surrogates.
#: Stored in the JSON sidecar next to the Dafny file; ``load`` ignores the field because it is not a core field.
STANDING_ASSUMPTIONS: tuple[str, ...] = (
    "inputs match the type hints",
    "strings contain no lone surrogates",
)


def from_result(result: Result, config: Config, *, dependency_hashes: tuple[Dependency, ...] = ()) -> Certificate:
    """Build the certificate of a ``proven`` result. The Dafny version comes from ``config.dafny.version``."""
    if result.state is not State.PROVEN:
        raise ValueError(f"cannot write a certificate for a {result.state.value} result; need a proven one")
    if result.proof is None or not result.proof.proven:
        raise ValueError("cannot write a certificate for a proven result without its proof")
    if result.spec is None or result.translation is None:
        raise ValueError("cannot write a certificate for a proven result without its spec and translation")
    spec_python = result.spec.predicate
    if result.spec.precondition is not None:
        spec_python += "\n" + result.spec.precondition
    return Certificate(
        function=result.function,
        dafny_source=result.proof.dafny_source,
        spec_python=spec_python,
        spec_dafny=result.translation.dafny_spec,
        function_path=result.translation.function_path,
        spec_path=result.translation.spec_path,
        translation_checks=result.translation.checks,
        assumptions=result.assumptions,
        dafny_version=config.dafny.version,
        pylib_sha256=hashlib.sha256(PYLIB_FILE.read_bytes()).hexdigest(),
        pcc_version=importlib.metadata.version("pcc"),
        dependency_hashes=dependency_hashes,
    )


def write(certificate: Certificate, proofs_dir: Path, *, config: Config | None = None) -> Path:
    """Store the certificate (Dafny file plus JSON metadata) and return the path of the Dafny file.

    The Dafny file goes to ``proofs_dir/<path without .py>/<qualname>.dfy`` with its JSON sidecar next to it; the
    sidecar holds every other field in the ``core.serialize`` format plus the constant ``standing_assumptions``.
    ``PyLib.dfy`` is copied to ``proofs_dir/PyLib.dfy`` and ``proofs_dir/dfyconfig.toml`` includes every ``**/*.dfy``
    with the resource limit from ``config.dafny.resource_limit`` (a default ``Config`` when omitted) and
    ``verification-time-limit = 0``, so ``dafny verify proofs/dfyconfig.toml`` re-checks every certificate.
    """
    active = Config() if config is None else config
    stem = certificate.function.path.removesuffix(".py")
    dafny_path = proofs_dir / stem / f"{certificate.function.qualname}.dfy"
    dafny_path.parent.mkdir(parents=True, exist_ok=True)
    dafny_path.write_text(certificate.dafny_source, encoding="utf-8")

    payload = serialize.to_data(certificate)
    if not isinstance(payload, dict):
        raise TypeError(f"certificate serializes to {type(payload).__name__}, not an object")
    del payload["dafny_source"]
    payload["standing_assumptions"] = list(STANDING_ASSUMPTIONS)
    # Same settings as serialize.dumps, which only accepts unconverted values: the payload above holds plain JSON
    # data already, so dumping it again through serialize would reject its lists.
    sidecar_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    dafny_path.with_suffix(".json").write_text(sidecar_text, encoding="utf-8")

    shutil.copyfile(PYLIB_FILE, proofs_dir / "PyLib.dfy")
    proofs_dir.mkdir(parents=True, exist_ok=True)
    (proofs_dir / "dfyconfig.toml").write_text(
        'includes = ["**/*.dfy"]\n\n[options]\n'
        f"resource-limit = {active.dafny.resource_limit}\n"
        "verification-time-limit = 0\n",
        encoding="utf-8",
    )
    return dafny_path


def load(path: Path) -> Certificate:
    """Rebuild the certificate stored at ``path`` (a ``.dfy`` file) together with its JSON sidecar."""
    dafny_source = path.read_text(encoding="utf-8")
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"certificate metadata {sidecar} not found for {path}")
    raw: Any = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"certificate metadata {sidecar} must hold a JSON object")
    raw.pop("standing_assumptions", None)
    raw["dafny_source"] = dafny_source
    return serialize.from_data(Certificate, raw)


def is_stale(
    certificate: Certificate,
    function: FunctionRef,
    *,
    dependency_hashes: tuple[Dependency, ...] = (),
    pylib_sha256: str,
) -> bool:
    """True when the function's source changed since the certificate was written, or a recorded dependency did."""
    if function.source_hash != certificate.function.source_hash:
        return True
    if pylib_sha256 != certificate.pylib_sha256:
        return True
    current = {dependency.name: dependency.sha256 for dependency in dependency_hashes}
    return any(current.get(recorded.name) != recorded.sha256 for recorded in certificate.dependency_hashes)


__all__ = ["from_result", "is_stale", "load", "write"]
