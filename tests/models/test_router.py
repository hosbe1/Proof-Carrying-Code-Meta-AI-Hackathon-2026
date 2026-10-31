"""Tests for the endpoint router and stage filtering (issue #19).

Routing tests use ``FakeClient`` as the endpoint client, except the cost test, which uses the real
``OpenAICompatClient`` against a localhost stub.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from pcc.config import Config, ModelsConfig, ProofConfig, ProverStage
from pcc.core import ToolError
from pcc.models import FakeClient, Message, ModelClient
from pcc.models.router import ModelUnavailable, available_stages, build_client

MESSAGES = (Message("user", "hi"),)


def _fakes(n: int = 10) -> dict[str, FakeClient]:
    return {"meta": FakeClient(["meta"] * n), "glimmer": FakeClient(["glimmer"] * n)}


def test_meta_unavailable_without_guard() -> None:
    router = build_client(Config(), {"META_API_KEY": "key"}, clients=_fakes())
    assert router.available("muse-spark-1.3") is False
    with pytest.raises(ModelUnavailable, match="PCC_META_ALLOWED=1 is not set"):
        router.complete("muse-spark-1.3", MESSAGES, step="intent")


def test_model_unavailable_is_a_tool_error() -> None:
    assert issubclass(ModelUnavailable, ToolError)


def test_meta_missing_key_names_the_variable() -> None:
    router = build_client(Config(), {"PCC_META_ALLOWED": "1"}, clients=_fakes())
    assert router.available("muse-spark-1.3") is False
    with pytest.raises(ModelUnavailable, match="META_API_KEY"):
        router.complete("muse-spark-1.3", MESSAGES, step="intent")


def test_meta_dispatch_passes_temperature_and_max_tokens() -> None:
    clients = _fakes()
    env = {"META_API_KEY": "key", "PCC_META_ALLOWED": "1"}
    router = build_client(Config(), env, clients=clients)
    assert router.available("muse-spark-1.3") is True
    assert router.available("muse-spark-1.3-contributor") is True
    assert router.available("muse-wonder-1.0") is True
    completion = router.complete("muse-spark-1.3", MESSAGES, step="intent", temperature=0.5, max_tokens=32)
    assert completion.text == "meta"
    assert completion.usage.step == "intent"
    assert clients["meta"].calls[0].temperature == 0.5
    assert clients["meta"].calls[0].max_tokens == 32
    assert clients["meta"].calls[0].model == "muse-spark-1.3"
    assert clients["glimmer"].calls == []


def test_glimmer_unavailable_without_base_url() -> None:
    router = build_client(Config(), {"META_API_KEY": "key", "PCC_META_ALLOWED": "1"}, clients=_fakes())
    assert router.available("glimmer") is False
    with pytest.raises(ModelUnavailable, match="PCC_GLIMMER_BASE_URL"):
        router.complete("glimmer", MESSAGES, step="prove")


def test_glimmer_missing_key_names_the_variable() -> None:
    router = build_client(Config(), {"PCC_GLIMMER_BASE_URL": "http://127.0.0.1:1"}, clients=_fakes())
    assert router.available("glimmer") is False
    with pytest.raises(ModelUnavailable, match="PCC_GLIMMER_API_KEY"):
        router.complete("glimmer", MESSAGES, step="prove")


def test_available_stages_drops_glimmer_without_its_base_url() -> None:
    env = {"META_API_KEY": "key", "PCC_META_ALLOWED": "1"}
    router = build_client(Config(), env, clients=_fakes())
    filtered = available_stages(Config(), router)
    assert [stage.name for stage in filtered.proof.prover_stages] == ["spark", "wonder"]
    assert [stage.name for stage in Config().proof.prover_stages] == ["glimmer", "spark", "wonder"]


def test_available_stages_keeps_everything_when_configured() -> None:
    env = {
        "META_API_KEY": "key",
        "PCC_META_ALLOWED": "1",
        "PCC_GLIMMER_BASE_URL": "http://127.0.0.1:1",
        "PCC_GLIMMER_API_KEY": "key",
    }
    router = build_client(Config(), env, clients=_fakes())
    filtered = available_stages(Config(), router)
    assert [stage.name for stage in filtered.proof.prover_stages] == ["glimmer", "spark", "wonder"]


def test_models_config_spec_glimmer_goes_to_glimmer() -> None:
    clients = _fakes()
    config = Config(models=ModelsConfig(spec="glimmer"))
    router = build_client(
        config, {"PCC_GLIMMER_BASE_URL": "http://127.0.0.1:1", "PCC_GLIMMER_API_KEY": "key"}, clients=clients
    )
    completion = router.complete("glimmer", MESSAGES, step="spec")
    assert completion.text == "glimmer"
    assert clients["glimmer"].calls[0].step == "spec"
    assert clients["meta"].calls == []


def test_models_config_id_without_stage_goes_to_meta() -> None:
    clients = _fakes()
    config = Config(models=ModelsConfig(intent="custom-model"))
    router = build_client(config, {"META_API_KEY": "key", "PCC_META_ALLOWED": "1"}, clients=clients)
    assert router.complete("custom-model", MESSAGES, step="intent").text == "meta"
    assert clients["glimmer"].calls == []


def test_duplicate_model_with_different_endpoints_raises() -> None:
    stages = (
        ProverStage("a", "m", 1, endpoint="meta"),
        ProverStage("b", "m", 1, endpoint="glimmer"),
    )
    with pytest.raises(ValueError, match="m"):
        build_client(Config(proof=ProofConfig(prover_stages=stages)), {}, clients=_fakes())


def test_meta_price_id_on_glimmer_stage_raises() -> None:
    stages = (ProverStage("x", "muse-spark-1.3", 1, endpoint="glimmer"),)
    with pytest.raises(ValueError, match=r"muse-spark-1\.3"):
        build_client(Config(proof=ProofConfig(prover_stages=stages)), {}, clients=_fakes())


def test_glimmer_sends_override_model_name() -> None:
    clients = _fakes()
    env = {
        "PCC_GLIMMER_BASE_URL": "http://127.0.0.1:1",
        "PCC_GLIMMER_API_KEY": "key",
        "PCC_GLIMMER_MODEL": "deployed-name",
    }
    router = build_client(Config(), env, clients=clients)
    assert router.available("glimmer") is True
    router.complete("glimmer", MESSAGES, step="prove")
    assert clients["glimmer"].calls[0].model == "deployed-name"


def test_unknown_model_is_unavailable() -> None:
    router = build_client(Config(), {"META_API_KEY": "key", "PCC_META_ALLOWED": "1"}, clients=_fakes())
    assert router.available("nope") is False
    with pytest.raises(ModelUnavailable, match="unknown model"):
        router.complete("nope", MESSAGES, step="intent")


def test_build_needs_no_keys() -> None:
    router = build_client(Config(), {}, clients=_fakes())
    assert router.available("muse-spark-1.3") is False
    assert router.available("glimmer") is False


class _StubState:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


def _serve(state: _StubState, prompt_tokens: int, completion_tokens: int) -> ThreadingHTTPServer:
    parent: dict[str, Any] = {"state": state}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            st: _StubState = parent["state"]
            length = int(self.headers.get("Content-Length", "0"))
            st.requests.append(json.loads(self.rfile.read(length).decode("utf-8")))
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _glimmer_env(server: ThreadingHTTPServer) -> dict[str, str]:
    return {
        "PCC_GLIMMER_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
        "PCC_GLIMMER_API_KEY": "test-key",
    }


def test_glimmer_per_token_price_sets_cost() -> None:
    state = _StubState()
    server = _serve(state, 1000, 500)
    try:
        config = Config(models=ModelsConfig(endpoint_prices={"glimmer": (1.25, 4.25)}))
        router = build_client(config, _glimmer_env(server))
        completion = router.complete("glimmer", MESSAGES, step="prove")
    finally:
        server.shutdown()
    assert completion.usage.cost_usd == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)
    assert state.requests[0]["model"] == "glimmer"


def test_glimmer_default_price_costs_zero() -> None:
    state = _StubState()
    server = _serve(state, 1000, 500)
    try:
        router = build_client(Config(), _glimmer_env(server))
        completion = router.complete("glimmer", MESSAGES, step="prove")
    finally:
        server.shutdown()
    assert completion.usage.cost_usd == 0.0


def test_glimmer_override_name_sets_price_key() -> None:
    state = _StubState()
    server = _serve(state, 1000, 500)
    try:
        config = Config(models=ModelsConfig(endpoint_prices={"glimmer": (1.25, 4.25)}))
        env = _glimmer_env(server) | {"PCC_GLIMMER_MODEL": "deployed-name"}
        router = build_client(config, env)
        completion = router.complete("glimmer", MESSAGES, step="prove")
    finally:
        server.shutdown()
    assert completion.usage.cost_usd == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)
    assert state.requests[0]["model"] == "deployed-name"


def test_injected_clients_satisfy_model_client_protocol() -> None:
    clients: Mapping[str, ModelClient] = _fakes()
    router = build_client(Config(), {"META_API_KEY": "k", "PCC_META_ALLOWED": "1"}, clients=clients)
    assert router.complete("muse-spark-1.3", MESSAGES, step="intent").text == "meta"


def _capture_server(state: _StubState) -> ThreadingHTTPServer:
    parent: dict[str, Any] = {"state": state}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            st: _StubState = parent["state"]
            length = int(self.headers.get("Content-Length", "0"))
            st.requests.append(json.loads(self.rfile.read(length).decode("utf-8")))
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_meta_reasoning_effort_wired_from_env() -> None:
    meta_state = _StubState()
    meta_server = _capture_server(meta_state)
    try:
        env = {
            "META_API_KEY": "key",
            "PCC_META_ALLOWED": "1",
            "PCC_META_BASE_URL": f"http://127.0.0.1:{meta_server.server_address[1]}",
            "PCC_META_REASONING_EFFORT": "low",
        }
        router = build_client(Config(), env)
        router.complete("muse-spark-1.3", MESSAGES, step="intent")
    finally:
        meta_server.shutdown()
    assert meta_state.requests[0]["reasoning_effort"] == "low"


def test_meta_reasoning_effort_unset_omits_field() -> None:
    meta_state = _StubState()
    meta_server = _capture_server(meta_state)
    try:
        env = {
            "META_API_KEY": "key",
            "PCC_META_ALLOWED": "1",
            "PCC_META_BASE_URL": f"http://127.0.0.1:{meta_server.server_address[1]}",
        }
        router = build_client(Config(), env)
        router.complete("muse-spark-1.3", MESSAGES, step="intent")
    finally:
        meta_server.shutdown()
    assert "reasoning_effort" not in meta_state.requests[0]


def test_invalid_meta_reasoning_effort_raises() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        build_client(Config(), {"PCC_META_REASONING_EFFORT": "banana"})


def test_glimmer_request_omits_reasoning_effort() -> None:
    meta_state = _StubState()
    glimmer_state = _StubState()
    meta_server = _capture_server(meta_state)
    glimmer_server = _capture_server(glimmer_state)
    try:
        env = {
            "META_API_KEY": "key",
            "PCC_META_ALLOWED": "1",
            "PCC_META_BASE_URL": f"http://127.0.0.1:{meta_server.server_address[1]}",
            "PCC_GLIMMER_BASE_URL": f"http://127.0.0.1:{glimmer_server.server_address[1]}",
            "PCC_GLIMMER_API_KEY": "key",
            "PCC_META_REASONING_EFFORT": "low",
        }
        router = build_client(Config(), env)
        router.complete("glimmer", MESSAGES, step="prove")
        router.complete("muse-spark-1.3", MESSAGES, step="intent")
    finally:
        meta_server.shutdown()
        glimmer_server.shutdown()
    assert "reasoning_effort" not in glimmer_state.requests[0]
    assert meta_state.requests[0]["reasoning_effort"] == "low"
