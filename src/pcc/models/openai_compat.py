"""An OpenAI-compatible chat completions client with structured output.

The client POSTs to ``{base_url}/chat/completions`` and speaks the plain
OpenAI shape, so it works against the model API and any compatible deployment.
With ``schema`` the request carries ``response_format`` with a strict JSON
schema; when the endpoint rejects it, or answers with invalid JSON, the client
falls back to appending the schema to the last user message instead.

Schemas must set ``additionalProperties: false`` on every object, as strict
mode requires; the client does not check this.
"""

from __future__ import annotations

import email.message
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from pcc.core import ToolError, Usage
from pcc.models.client import Completion, Message

META_PRICES: dict[str, tuple[float, float]] = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20),
    "muse-wonder-1.0": (4.00, 20.00),
}
"""Per-million-token (input, output) prices in USD, from docs/plan.md."""

_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
_FALLBACK_PREFIX = "\n\nAnswer with JSON matching this schema: "

REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
"""Valid ``reasoning_effort`` values for the Meta model API; unset omits the field."""

# Errors from the request itself that are worth one retry each: a refused or
# reset connection, a timeout, and a broken HTTP exchange. HTTPError is also a
# URLError, so it is caught separately before this tuple.
_RETRYABLE_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _schema_suffix(schema: dict[str, object]) -> str:
    return _FALLBACK_PREFIX + json.dumps(schema, sort_keys=True, separators=(",", ":"))


def _error_message(raw: str) -> str:
    """The API error ``message`` when the body is JSON with one, else the raw body."""
    try:
        payload: Any = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
    return raw


def _retry_after_seconds(headers: email.message.Message[str, str] | None, retries_used: int) -> float:
    """Nonnegative ``Retry-After`` delta when valid, else 1, 2, 4, 8 s by retry."""
    if headers is not None:
        raw = headers.get("Retry-After")
        if raw is not None:
            try:
                delay = float(raw.strip())
            except ValueError:
                delay = math.nan
            if math.isfinite(delay) and delay >= 0:
                return delay
    return _BACKOFF_SECONDS[min(retries_used, len(_BACKOFF_SECONDS) - 1)]


class OpenAICompatClient:
    """POSTs chat completions to ``{base_url}/chat/completions``.

    ``prices`` maps model names to per-million-token (input, output) prices in
    USD; with ``None`` the reported cost is always 0. A model missing from a
    given ``prices`` map raises ``ToolError``. ``max_retries`` is the number of
    retries after the first request, so at most ``max_retries + 1`` HTTP
    attempts per call; the ``response_format`` fallback and the invalid-JSON
    retry share this budget. Retryable failures (429, 500, 502, 503, 504 and
    connection errors) sleep with backoff 1, 2, 4, 8 s, or ``Retry-After`` when
    valid; the fallbacks never sleep. ``Usage.seconds`` is the wall-clock time
    of the whole call, including sleeps. The API key is sent as
    ``Authorization: Bearer`` and never appears in an error message.

    ``reasoning_effort`` is an opt-in Meta request field (one of ``minimal``,
    ``low``, ``medium``, ``high``, ``xhigh``); with ``None`` the field is
    omitted and the request shape is unchanged. Any other value raises
    ``ValueError``. A response whose message ``content`` is null with
    ``finish_reason`` ``"length"`` is a billed but exhausted attempt: it returns
    an empty ``Completion`` with the reported usage immediately, without the
    invalid-JSON retry, even when ``schema`` was requested. All other null or
    non-text contents remain malformed-response ``ToolError``s.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        prices: Mapping[str, tuple[float, float]] | None = None,
        timeout_seconds: float = 120,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        reasoning_effort: str | None = None,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            valid = ", ".join(sorted(REASONING_EFFORTS))
            raise ValueError(f"invalid reasoning_effort {reasoning_effort!r}; expected one of: {valid}")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._prices = dict(prices) if prices is not None else None
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._sleep = sleep
        self._reasoning_effort = reasoning_effort

    def complete(
        self,
        model: str,
        messages: tuple[Message, ...],
        *,
        step: str,
        schema: dict[str, object] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        """One chat completion; with ``schema`` the answer is JSON text."""
        if self._prices is not None and model not in self._prices:
            raise ToolError(f"no price for model {model!r}")
        use_response_format = schema is not None
        fallback_used = False
        json_retry_used = False
        retries_used = 0
        last_error = ""
        start = time.monotonic()
        while True:
            body = self._request_body(model, messages, step, schema, temperature, max_tokens, use_response_format)
            try:
                raw = self._post(body)
            except urllib.error.HTTPError as exc:
                text = self._read_error(exc)
                last_error = f"status {exc.code}: {text}"
                if (
                    schema is not None
                    and use_response_format
                    and not fallback_used
                    and exc.code == 400
                    and "response_format" in text.lower()
                ):
                    fallback_used = True
                    use_response_format = False
                    retries_used = self._consume_retry(model, step, retries_used, last_error)
                    continue
                if exc.code in _RETRYABLE_STATUSES:
                    delay = _retry_after_seconds(exc.headers, retries_used)
                    retries_used = self._consume_retry(model, step, retries_used, last_error)
                    self._sleep(delay)
                    continue
                raise ToolError(f"model API error for {model!r} (step {step!r}): {last_error}") from exc
            except _RETRYABLE_ERRORS as exc:
                last_error = f"connection error: {exc}"
                delay = _retry_after_seconds(None, retries_used)
                retries_used = self._consume_retry(model, step, retries_used, last_error)
                self._sleep(delay)
                continue
            text, input_tokens, output_tokens, length_exhausted = self._parse_response(model, step, raw)
            if length_exhausted:
                seconds = time.monotonic() - start
                cost = self._cost(model, input_tokens, output_tokens)
                return Completion(text, Usage(step, model, input_tokens, output_tokens, cost, seconds))
            if schema is not None and not json_retry_used and not _is_json(text):
                last_error = f"invalid JSON answer: {text[:200]}"
                json_retry_used = True
                fallback_used = True
                use_response_format = False
                retries_used = self._consume_retry(model, step, retries_used, last_error)
                continue
            if schema is not None and not _is_json(text):
                raise ToolError(f"model {model!r} (step {step!r}) returned invalid JSON twice: {text[:200]}")
            seconds = time.monotonic() - start
            cost = self._cost(model, input_tokens, output_tokens)
            return Completion(text, Usage(step, model, input_tokens, output_tokens, cost, seconds))

    def _consume_retry(self, model: str, step: str, retries_used: int, last_error: str) -> int:
        if retries_used >= self._max_retries:
            raise ToolError(f"model API error for {model!r} (step {step!r}) after retries: {last_error}")
        return retries_used + 1

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if self._prices is None:
            return 0.0
        input_price, output_price = self._prices[model]
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def _request_body(
        self,
        model: str,
        messages: tuple[Message, ...],
        step: str,
        schema: dict[str, object] | None,
        temperature: float,
        max_tokens: int | None,
        use_response_format: bool,
    ) -> dict[str, Any]:
        payload = [{"role": message.role, "content": message.content} for message in messages]
        if schema is not None and not use_response_format and payload:
            index = next((i for i in range(len(payload) - 1, -1, -1) if messages[i].role == "user"), len(payload) - 1)
            payload[index] = {**payload[index], "content": payload[index]["content"] + _schema_suffix(schema)}
        body: dict[str, Any] = {"model": model, "messages": payload, "temperature": temperature}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self._reasoning_effort is not None:
            body["reasoning_effort"] = self._reasoning_effort
        if use_response_format and schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": step, "schema": schema, "strict": True},
            }
        return body

    def _post(self, body: dict[str, Any]) -> bytes:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            data: bytes = response.read()
            return data

    @staticmethod
    def _read_error(exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return ""
        return _error_message(raw)

    @staticmethod
    def _parse_response(model: str, step: str, raw: bytes) -> tuple[str, int, int, bool]:
        try:
            payload: Any = json.loads(raw.decode("utf-8"))
            choice = payload["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            if "content" not in message:
                raise KeyError("content")
            content = message["content"]
            if content is None:
                if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                    prompt, completion_tokens = _usage_tokens(payload)
                    return "", prompt, completion_tokens, True
                raise TypeError("content is not text")
            if not isinstance(content, str):
                raise TypeError("content is not text")
            prompt, completion_tokens = _usage_tokens(payload)
            return content, prompt, completion_tokens, False
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ToolError(f"malformed model response for {model!r} (step {step!r}): {exc}") from exc


def _usage_tokens(payload: Any) -> tuple[int, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        raise TypeError("usage is not an object")
    prompt = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    return int(prompt), int(completion_tokens)


__all__ = ["META_PRICES", "REASONING_EFFORTS", "OpenAICompatClient"]
