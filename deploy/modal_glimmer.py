"""Serve Glimmer with vLLM on Modal.

Deploy from the repository root with the settings exported::

    export PCC_GLIMMER_API_KEY="$(openssl rand -hex 32)"
    modal secret create pcc-glimmer-api PCC_GLIMMER_API_KEY="$PCC_GLIMMER_API_KEY"
    modal deploy deploy/modal_glimmer.py

See ``docs/deploy.md`` for the full procedure. Never import this module in
tests: Modal is an optional dependency that is not installed in CI.
"""

import os
import subprocess
import time

import modal  # type: ignore[import-not-found]  # optional, not a dev dependency

from deploy.glimmer_settings import (
    SCALEDOWN_WINDOW_SECONDS,
    SECRET_NAME,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    check_volume_paths,
    load_settings,
    vllm_server_args,
)

VLLM_PORT = 8000

_SETTINGS = load_settings()

app = modal.App("pcc-glimmer")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub")
    # Cache the weights on the volume so a cold start reuses the download.
    # Build steps after add_local_* are rejected, so this must come first.
    # The slim image has no nvcc, so vLLM must not JIT-compile the FlashInfer
    # sampler during warm-up.
    .env({"HF_HOME": f"{VOLUME_MOUNT_PATH}/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    # Bundle the deploy package so the container resolves its helpers.
    .add_local_python_source("deploy")
)


@app.function(
    image=image,
    gpu=_SETTINGS.gpu,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    # One GPU at most: queued requests during a cold start must not fan out to the GPU limit.
    max_containers=1,
    # Loading the weights takes tens of minutes; the web_server timeout alone
    # does not stop Modal from killing the runner (and the vLLM engine) first.
    startup_timeout=45 * 60,
)
# One container serves many clients at once: without this, Modal passes one
# request at a time and max_containers=1 serialises all clients.
@modal.concurrent(max_inputs=64)
@modal.web_server(port=VLLM_PORT, startup_timeout=45 * 60)
def serve() -> None:
    """Start vLLM's OpenAI-compatible server for the baked-in settings."""
    check_volume_paths(_SETTINGS.adapters)
    cmd = [
        "vllm",
        "serve",
        _SETTINGS.base_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        *vllm_server_args(_SETTINGS),
        "--api-key",
        os.environ["PCC_GLIMMER_API_KEY"],
    ]
    # Log the launch to the container output so a failed start is visible in
    # the deploy logs. Modal health-checks the web port; this print only aids
    # diagnosis and does not change the required background-start contract.
    print(f"Starting vLLM server for {_SETTINGS.base_model} on port {VLLM_PORT}", flush=True)
    proc = subprocess.Popen(cmd)
    # Fail fast when vLLM exits immediately (missing binary, bad flags) by
    # surfacing its exit code instead of a later port health-check failure.
    # A slow weight load keeps running, so only wait a few seconds here.
    time.sleep(5)
    if proc.poll() is not None:
        raise RuntimeError(f"vLLM server exited immediately with code {proc.returncode}")
    return
