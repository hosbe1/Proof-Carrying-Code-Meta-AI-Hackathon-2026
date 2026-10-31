"""Environment parsing for the Glimmer deployment.

This module is pure: it imports neither Modal nor vLLM, so it can be used and
tested anywhere. The Modal app in ``deploy/modal_glimmer.py`` builds on it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_BASE_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_GPU = "H100"
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MODEL = "glimmer"
DEFAULT_QUANTIZATION = ""

SUPPORTED_QUANTIZATIONS = ("fp8",)

SECRET_NAME = "pcc-glimmer-api"
VOLUME_NAME = "pcc-glimmer"
VOLUME_MOUNT_PATH = "/models/pcc-glimmer"
SERVED_MODEL_NAME = "glimmer"
MAX_LORA_RANK = 16
SCALEDOWN_WINDOW_SECONDS = 15 * 60


@dataclass(frozen=True)
class Adapter:
    """One LoRA adapter served alongside the base model.

    A source that starts with ``/`` is a path under the mounted Modal volume
    :data:`VOLUME_MOUNT_PATH`; any other source is a Hugging Face model id and
    is never checked against the filesystem.
    """

    name: str
    source: str

    @property
    def is_volume_path(self) -> bool:
        """Whether this adapter lives on the mounted Modal volume."""
        return self.source.startswith("/")

    def target(self, mount: str = VOLUME_MOUNT_PATH) -> str:
        """Where vLLM finds this adapter: the container path or the model id."""
        if self.is_volume_path:
            return mount.rstrip("/") + self.source
        return self.source

    def lora_entry(self, mount: str = VOLUME_MOUNT_PATH) -> str:
        """The ``name=path`` entry for vLLM's ``--lora-modules`` flag."""
        return f"{self.name}={self.target(mount)}"


@dataclass(frozen=True)
class GlimmerSettings:
    """Everything the Modal app needs to serve Glimmer."""

    base_model: str = DEFAULT_BASE_MODEL
    gpu: str = DEFAULT_GPU
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    quantization: str = DEFAULT_QUANTIZATION
    model: str = DEFAULT_MODEL
    adapters: tuple[Adapter, ...] = field(default_factory=tuple)


def parse_adapters(raw: str) -> tuple[Adapter, ...]:
    """Parse ``name=source`` adapter entries, in order.

    Only the syntax is validated here; volume paths are checked at deployment
    startup by :func:`check_volume_paths`, which runs inside the container.
    """
    raw = raw.strip()
    if not raw:
        return ()
    adapters = []
    for entry in raw.split(","):
        entry = entry.strip()
        name, separator, source = entry.partition("=")
        name = name.strip()
        source = source.strip()
        if not separator or not name or not source:
            raise ValueError(f"Invalid adapter entry {entry!r}: expected 'name=source'.")
        adapters.append(Adapter(name=name, source=source))
    return tuple(adapters)


def check_volume_paths(adapters: tuple[Adapter, ...], mount: str = VOLUME_MOUNT_PATH) -> None:
    """Fail with a clear error when a volume adapter path does not exist.

    Hugging Face ids are accepted without any path check. This runs at
    deployment startup inside the container, never on the developer machine.
    """
    for adapter in adapters:
        if not adapter.is_volume_path:
            continue
        path = adapter.target(mount)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Adapter {adapter.name!r} not found on volume {VOLUME_NAME!r}: {path} does not exist."
            )


def load_settings(env: Mapping[str, str] | None = None) -> GlimmerSettings:
    """Read the ``PCC_GLIMMER_*`` environment into settings."""
    source = env if env is not None else os.environ
    try:
        max_model_len = int(source.get("PCC_GLIMMER_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))
    except ValueError:
        raise ValueError(
            "Invalid PCC_GLIMMER_MAX_MODEL_LEN "
            f"{source.get('PCC_GLIMMER_MAX_MODEL_LEN')!r}: expected a positive integer."
        ) from None
    if max_model_len <= 0:
        raise ValueError(f"Invalid PCC_GLIMMER_MAX_MODEL_LEN {max_model_len}: expected a positive integer.")
    quantization = source.get("PCC_GLIMMER_QUANTIZATION", DEFAULT_QUANTIZATION).strip()
    if quantization and quantization not in SUPPORTED_QUANTIZATIONS:
        raise ValueError(
            f"Invalid PCC_GLIMMER_QUANTIZATION {quantization!r}: "
            f"expected one of {SUPPORTED_QUANTIZATIONS} or empty (disabled)."
        )
    return GlimmerSettings(
        base_model=source.get("PCC_GLIMMER_BASE_MODEL", DEFAULT_BASE_MODEL).strip() or DEFAULT_BASE_MODEL,
        gpu=source.get("PCC_GLIMMER_GPU", DEFAULT_GPU).strip() or DEFAULT_GPU,
        max_model_len=max_model_len,
        quantization=quantization,
        model=source.get("PCC_GLIMMER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        adapters=parse_adapters(source.get("PCC_GLIMMER_ADAPTERS", "")),
    )


def vllm_server_args(settings: GlimmerSettings, mount: str = VOLUME_MOUNT_PATH) -> list[str]:
    """Command-line arguments for vLLM's OpenAI-compatible server."""
    args = [
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--enable-lora",
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--max-model-len",
        str(settings.max_model_len),
    ]
    if settings.quantization:
        args += ["--quantization", settings.quantization]
    if settings.adapters:
        args.append("--lora-modules")
        args.extend(adapter.lora_entry(mount) for adapter in settings.adapters)
    return args
