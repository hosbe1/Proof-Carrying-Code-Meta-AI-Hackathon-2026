"""Tests for the OpenAI-compatible model client (issue #16).

Every test that touches HTTP talks to a localhost stub; the base URL is
injectable through the constructor, so no real network is used. The one test
that calls a real model is marked ``live`` and only runs by hand.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from pcc.core import ToolError
from pcc.models import Message, OpenAICompatClient
from pcc.models.openai_compat import META_PRICES

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

PRICES: dict[str, tuple[float, float]] = {"m": (1.25, 4.25)}

# Each scripted answer is (status, extra headers, payload). A payload of bytes
# is sent raw, anything else is sent as JSON.
Scripted = tuple[int, dict[str, str], Any]


class _StubState:
    def __init__(self, script: list[Scripted]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []


def _ok(text: str, usage: dict[str, int] | None = None) -> Scripted:
    body: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return (200, {}, body)


def _err(status: int, message: str, headers: dict[str, str] | None = None) -> Scripted:
    return (status, dict(headers or {}), {"error": {"message": message}})


def _serve(state: _StubState) -> ThreadingHTTPServer:
    parent: dict[str, Any] = {"state": state}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            st: _StubState = parent["state"]
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload: Any = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                payload = None
            st.requests.append(
                {
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "content_type": self.headers.get("Content-Type"),
                    "body": payload,
                }
            )
            if st.script:
                status, extra, answer = st.script.pop(0)
            else:
                status, extra, answer = (500, {}, {"error": {"message": "stub ran out of scripted responses"}})
            raw_body = answer if isinstance(answer, bytes) else json.dumps(answer).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw_body)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw_body)

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _client(
    server: ThreadingHTTPServer,
    sleeps: list[float],
    *,
    prices: dict[str, tuple[float, float]] | None = PRICES,
    max_retries: int = 4,
    key: str = "test-key",
) -> OpenAICompatClient:
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return OpenAICompatClient(base_url, key, prices=prices, sleep=sleeps.append, max_retries=max_retries)


def _schema_suffix(schema: dict[str, object]) -> str:
    return "\n\nAnswer with JSON matching this schema: " + json.dumps(schema, sort_keys=True, separators=(",", ":"))


def test_meta_prices_match_plan() -> None:
    assert META_PRICES == {
        "muse-spark-1.3": (1.25, 4.25),
        "muse-spark-1.3-contributor": (0.10, 0.20),
        "muse-wonder-1.0": (4.00, 20.00),
    }


def test_success_reports_text_usage_and_cost() -> None:
    state = _StubState([_ok("hi", {"prompt_tokens": 1000, "completion_tokens": 500})])
    server = _serve(state)
    try:
        client = _client(server, [])
        completion = client.complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.text == "hi"
    assert completion.usage.step == "intent"
    assert completion.usage.model == "m"
    assert completion.usage.input_tokens == 1000
    assert completion.usage.output_tokens == 500
    assert completion.usage.cost_usd == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)
    assert completion.usage.seconds >= 0.0


def test_missing_usage_counts_zero_tokens() -> None:
    state = _StubState([_ok("hi")])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.usage.input_tokens == 0
    assert completion.usage.output_tokens == 0
    assert completion.usage.cost_usd == 0.0


def test_prices_none_gives_zero_cost() -> None:
    state = _StubState([_ok("hi", {"prompt_tokens": 1000, "completion_tokens": 500})])
    server = _serve(state)
    try:
        completion = _client(server, [], prices=None).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.usage.cost_usd == 0.0


def test_unknown_model_price_raises() -> None:
    state = _StubState([_ok("hi", {"prompt_tokens": 1, "completion_tokens": 1})])
    server = _serve(state)
    try:
        with pytest.raises(ToolError, match="no price for model"):
            _client(server, [], prices={"other": (1.0, 2.0)}).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()


def test_request_body_carries_model_messages_temperature_and_max_tokens() -> None:
    state = _StubState([_ok("hi")])
    server = _serve(state)
    try:
        _client(server, []).complete("m", (Message("user", "hi"),), step="intent", temperature=0.5, max_tokens=128)
    finally:
        server.shutdown()
    assert len(state.requests) == 1
    request = state.requests[0]
    assert request["path"] == "/chat/completions"
    assert request["auth"] == "Bearer test-key"
    assert request["content_type"] == "application/json"
    assert request["body"]["model"] == "m"
    assert request["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert request["body"]["temperature"] == 0.5
    assert request["body"]["max_tokens"] == 128
    assert "response_format" not in request["body"]


def test_request_body_omits_max_tokens_when_unset() -> None:
    state = _StubState([_ok("hi")])
    server = _serve(state)
    try:
        _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert state.requests[0]["body"]["temperature"] == 0.0
    assert "max_tokens" not in state.requests[0]["body"]


def test_schema_sends_response_format() -> None:
    state = _StubState([_ok('{"answer": "yes"}')])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert completion.text == '{"answer": "yes"}'
    assert state.requests[0]["body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "intent", "schema": SCHEMA, "strict": True},
    }


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_status_then_success(status: int) -> None:
    state = _StubState([_err(status, "busy"), _ok("hi")])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        completion = _client(server, sleeps).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.text == "hi"
    assert len(state.requests) == 2
    assert sleeps == [1]


def test_retry_after_header_overrides_backoff() -> None:
    state = _StubState([_err(429, "slow down", {"Retry-After": "7"}), _ok("hi")])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        _client(server, sleeps).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert sleeps == [7.0]


def test_invalid_retry_after_falls_back_to_backoff() -> None:
    state = _StubState([_err(503, "busy", {"Retry-After": "soon"}), _ok("hi")])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        _client(server, sleeps).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert sleeps == [1]


def test_five_attempts_then_tool_error() -> None:
    state = _StubState([_err(503, "busy")] * 6)
    server = _serve(state)
    sleeps: list[float] = []
    try:
        with pytest.raises(ToolError, match="503"):
            _client(server, sleeps).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 5
    assert sleeps == [1, 2, 4, 8]


def test_max_retries_one_stops_after_two_attempts() -> None:
    state = _StubState([_err(429, "busy")] * 3)
    server = _serve(state)
    sleeps: list[float] = []
    try:
        with pytest.raises(ToolError):
            _client(server, sleeps, max_retries=1).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 2
    assert sleeps == [1]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501])
def test_non_retryable_status_raises_at_once(status: int) -> None:
    state = _StubState([_err(status, "bad key" if status != 400 else "bad request"), _ok("hi")])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        with pytest.raises(ToolError):
            _client(server, sleeps).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 1
    assert sleeps == []


def test_401_message_holds_status_and_error_text_but_not_key() -> None:
    state = _StubState([_err(401, "bad key")])
    server = _serve(state)
    try:
        with pytest.raises(ToolError) as excinfo:
            _client(server, [], key="secret-test-key").complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    message = str(excinfo.value)
    assert "401" in message
    assert "bad key" in message
    assert "secret-test-key" not in message


def test_connection_error_retries_then_raises() -> None:
    state = _StubState([])
    server = _serve(state)
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    sleeps: list[float] = []
    client = OpenAICompatClient(f"http://127.0.0.1:{port}", "test-key", prices=PRICES, sleep=sleeps.append)
    with pytest.raises(ToolError):
        client.complete("m", (Message("user", "hi"),), step="intent", max_tokens=None)
    assert sleeps == [1, 2, 4, 8]


def test_400_mentioning_response_format_falls_back_once_without_sleep() -> None:
    state = _StubState([_err(400, "response_format json_schema is not supported"), _ok('{"answer": "yes"}')])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        completion = _client(server, sleeps).complete(
            "m", (Message("user", "first"), Message("user", "second")), step="intent", schema=SCHEMA
        )
    finally:
        server.shutdown()
    assert completion.text == '{"answer": "yes"}'
    assert len(state.requests) == 2
    assert sleeps == []
    assert state.requests[0]["body"]["response_format"]["type"] == "json_schema"
    fallback = state.requests[1]["body"]
    assert "response_format" not in fallback
    assert fallback["messages"][0] == {"role": "user", "content": "first"}
    assert fallback["messages"][1] == {"role": "user", "content": "second" + _schema_suffix(SCHEMA)}


def test_400_fallback_match_is_case_insensitive() -> None:
    state = _StubState([_err(400, "RESPONSE_FORMAT not allowed here"), _ok('{"answer": "yes"}')])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert completion.text == '{"answer": "yes"}'
    assert len(state.requests) == 2


def test_400_fallback_happens_only_once() -> None:
    state = _StubState(
        [
            _err(400, "response_format unsupported"),
            _err(400, "response_format still unsupported"),
            _ok('{"answer": "yes"}'),
        ]
    )
    server = _serve(state)
    try:
        with pytest.raises(ToolError, match="400"):
            _client(server, []).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert len(state.requests) == 2


def test_fallback_counts_against_retry_budget() -> None:
    state = _StubState([_err(400, "response_format unsupported"), _err(429, "busy"), _ok("hi")])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        with pytest.raises(ToolError, match="429"):
            _client(server, sleeps, max_retries=1).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert len(state.requests) == 2
    assert sleeps == []


def test_invalid_json_retries_without_response_format_and_no_sleep() -> None:
    state = _StubState([_ok("not json"), _ok('{"answer": "yes"}')])
    server = _serve(state)
    sleeps: list[float] = []
    try:
        completion = _client(server, sleeps).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert completion.text == '{"answer": "yes"}'
    assert len(state.requests) == 2
    assert sleeps == []
    assert "response_format" not in state.requests[1]["body"]
    assert state.requests[1]["body"]["messages"] == [{"role": "user", "content": "hi" + _schema_suffix(SCHEMA)}]


def test_second_invalid_json_raises() -> None:
    state = _StubState([_ok("not json"), _ok("still not json"), _ok('{"answer": "yes"}')])
    server = _serve(state)
    try:
        with pytest.raises(ToolError, match="invalid JSON"):
            _client(server, []).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert len(state.requests) == 2


def test_invalid_json_without_schema_is_returned_as_is() -> None:
    state = _StubState([_ok("not json")])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.text == "not json"
    assert len(state.requests) == 1


def test_malformed_success_response_raises_without_retry() -> None:
    state = _StubState([(200, {}, {"unexpected": "shape"}), _ok("hi")])
    server = _serve(state)
    try:
        with pytest.raises(ToolError):
            _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 1


def test_non_json_success_body_raises_without_retry() -> None:
    state = _StubState([(200, {}, b"not a json body"), _ok("hi")])
    server = _serve(state)
    try:
        with pytest.raises(ToolError):
            _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 1


def _ok_null(finish_reason: str | None, usage: dict[str, int] | None = None) -> Scripted:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    choice: dict[str, Any] = {"message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    body: dict[str, Any] = {"choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return (200, {}, body)


def _client_with_effort(
    server: ThreadingHTTPServer,
    sleeps: list[float],
    reasoning_effort: str | None,
) -> OpenAICompatClient:
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return OpenAICompatClient(
        base_url, "test-key", prices=PRICES, sleep=sleeps.append, reasoning_effort=reasoning_effort
    )


def test_reasoning_effort_sent_in_request_body() -> None:
    state = _StubState([_ok("hi")])
    server = _serve(state)
    try:
        _client_with_effort(server, [], "low").complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert state.requests[0]["body"]["reasoning_effort"] == "low"


def test_reasoning_effort_omitted_when_unset() -> None:
    state = _StubState([_ok("hi")])
    server = _serve(state)
    try:
        _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert "reasoning_effort" not in state.requests[0]["body"]


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
def test_valid_reasoning_effort_accepted(effort: str) -> None:
    state = _StubState([_ok("hi")])
    server = _serve(state)
    try:
        _client_with_effort(server, [], effort).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert state.requests[0]["body"]["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["", "none", "None", "LOW", "banana"])
def test_invalid_reasoning_effort_rejected(effort: str) -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        OpenAICompatClient("http://127.0.0.1:1", "test-key", prices=PRICES, reasoning_effort=effort)


def test_null_content_with_length_returns_empty_with_usage() -> None:
    state = _StubState([_ok_null("length", {"prompt_tokens": 10, "completion_tokens": 4096})])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.text == ""
    assert completion.usage.input_tokens == 10
    assert completion.usage.output_tokens == 4096
    assert len(state.requests) == 1


def test_null_content_with_length_and_schema_returns_empty_without_retry() -> None:
    state = _StubState([_ok_null("length", {"prompt_tokens": 10, "completion_tokens": 4096})])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent", schema=SCHEMA)
    finally:
        server.shutdown()
    assert completion.text == ""
    assert completion.usage.output_tokens == 4096
    assert len(state.requests) == 1


def test_null_content_with_length_missing_usage_counts_zero() -> None:
    state = _StubState([_ok_null("length")])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.text == ""
    assert completion.usage.input_tokens == 0
    assert completion.usage.output_tokens == 0


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_null_content_without_length_still_raises(finish_reason: str | None) -> None:
    state = _StubState([_ok_null(finish_reason, {"prompt_tokens": 1, "completion_tokens": 1}), _ok("hi")])
    server = _serve(state)
    try:
        with pytest.raises(ToolError, match="malformed model response"):
            _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 1


def test_nontext_content_without_length_still_raises() -> None:
    state = _StubState([(200, {}, {"choices": [{"message": {"role": "assistant", "content": ["x"]}}]}), _ok("hi")])
    server = _serve(state)
    try:
        with pytest.raises(ToolError, match="malformed model response"):
            _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert len(state.requests) == 1


@pytest.mark.live
def test_live_structured_completion() -> None:
    if os.environ.get("PCC_META_ALLOWED") != "1":
        pytest.skip("needs PCC_META_ALLOWED=1")
    base_url = os.environ.get("PCC_META_BASE_URL")
    api_key = os.environ.get("META_API_KEY")
    if not base_url or not api_key:
        pytest.skip("needs PCC_META_BASE_URL and META_API_KEY")
    client = OpenAICompatClient(base_url, api_key, prices=META_PRICES)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
        "additionalProperties": False,
    }
    completion = client.complete(
        "muse-spark-1.3",
        (Message("user", "Reply with a short greeting."),),
        step="live-check",
        schema=schema,
        max_tokens=64,
    )
    assert json.loads(completion.text)["greeting"]
    assert completion.usage.input_tokens >= 0
    assert completion.usage.output_tokens > 0


@pytest.mark.parametrize(
    ("content", "answer"),
    [
        (" to=userdef f(): pass", "def f(): pass"),
        (" to=selfthink about to=user mentions to=usermethod M() {}", "method M() {}"),
        (" to=selfreasoning cut off by the token limit", ""),
        ("plain answer that mentions to=user inside", "plain answer that mentions to=user inside"),
    ],
)
def test_leaked_channel_headers_are_stripped(content: str, answer: str) -> None:
    state = _StubState([_ok(content)])
    server = _serve(state)
    try:
        completion = _client(server, []).complete("m", (Message("user", "hi"),), step="intent")
    finally:
        server.shutdown()
    assert completion.text == answer
