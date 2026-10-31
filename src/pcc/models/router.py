"""Routes model ids to their endpoint and filters prover stages by availability.

Each model id is served by one endpoint: ``meta`` (the model API) or ``glimmer`` (a hosted deployment).
``build_client`` maps every id in ``ModelsConfig`` and in the prover stages to its endpoint, using the stage's
endpoint when a ``ModelsConfig`` id matches a stage model and ``meta`` otherwise; every id in ``META_PRICES``
goes to ``meta``. The router dispatches ``complete`` to the endpoint client and reports ``available`` per model.
``available_stages`` drops the prover stages whose model is not available.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

from pcc.config import Config, ModelsConfig
from pcc.core import ToolError
from pcc.models.client import Completion, Message, ModelClient
from pcc.models.openai_compat import META_PRICES, OpenAICompatClient

META_ENDPOINT = "meta"
GLIMMER_ENDPOINT = "glimmer"
META_BASE_URL_DEFAULT = "https://api.meta.ai/v1"


class ModelUnavailable(ToolError):
    """A model cannot be called because its endpoint is not configured or not allowed."""


class Router:
    """Dispatches model calls to the endpoint client serving each model id.

    Built by ``build_client``; a bare ``Router()`` routes nothing, so ``available`` is always false and
    ``complete`` always raises ``ModelUnavailable``.
    """

    __slots__ = ("_clients", "_endpoint_of", "_env", "_sent_model")

    def __init__(
        self,
        clients: Mapping[str, ModelClient] | None = None,
        endpoint_of: Mapping[str, str] | None = None,
        sent_model: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._clients: dict[str, ModelClient] = dict(clients) if clients is not None else {}
        self._endpoint_of: dict[str, str] = dict(endpoint_of) if endpoint_of is not None else {}
        self._sent_model: dict[str, str] = dict(sent_model) if sent_model is not None else {}
        self._env: dict[str, str] = dict(env) if env is not None else {}

    def available(self, model: str) -> bool:
        """True when the endpoint serving ``model`` is configured and, for meta, allowed."""
        endpoint = self._endpoint_of.get(model)
        if endpoint is None:
            return False
        return self._unavailable_reason(endpoint) is None

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
        """Run one completion on the endpoint serving ``model``."""
        endpoint = self._endpoint_of.get(model)
        if endpoint is None:
            raise ModelUnavailable(f"unknown model {model!r}")
        reason = self._unavailable_reason(endpoint)
        if reason is not None:
            raise ModelUnavailable(f"model {model!r} is not available: {reason}")
        return self._clients[endpoint].complete(
            self._sent_model.get(model, model),
            messages,
            step=step,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _unavailable_reason(self, endpoint: str) -> str | None:
        if endpoint == GLIMMER_ENDPOINT:
            if not self._env.get("PCC_GLIMMER_BASE_URL"):
                return "PCC_GLIMMER_BASE_URL is not set"
            if not self._env.get("PCC_GLIMMER_API_KEY"):
                return "PCC_GLIMMER_API_KEY is not set"
            return None
        if not self._env.get("META_API_KEY"):
            return "META_API_KEY is not set"
        if self._env.get("PCC_META_ALLOWED") != "1":
            return "PCC_META_ALLOWED=1 is not set"
        return None


def _models_config_ids(models: ModelsConfig) -> tuple[str, ...]:
    return (
        models.intent,
        models.spec,
        models.mutants,
        models.translation,
        models.edge_cases,
        models.counterexample,
        models.second_opinion,
        models.context_search,
    )


def build_client(
    config: Config,
    env: Mapping[str, str] = os.environ,
    *,
    clients: Mapping[str, ModelClient] | None = None,
) -> Router:
    """Route every model id in ``config`` to its endpoint client.

    ``clients`` overrides the endpoint clients by endpoint name (``"meta"``, ``"glimmer"``); tests pass
    ``FakeClient`` instances there. Without an override the client is a real ``OpenAICompatClient``: meta uses
    ``PCC_META_BASE_URL`` (default ``https://api.meta.ai/v1``), ``META_API_KEY`` and ``META_PRICES``, plus the
    optional ``PCC_META_REASONING_EFFORT`` request field (``minimal``, ``low``, ``medium``, ``high`` or ``xhigh``;
    omitted when unset; any other value raises ``ValueError``); glimmer uses ``PCC_GLIMMER_BASE_URL``,
    ``PCC_GLIMMER_API_KEY`` and
    ``config.models.endpoint_prices["glimmer"]`` keyed by the model name sent, which is ``PCC_GLIMMER_MODEL``
    when set and the routed model id otherwise. Glimmer requests never carry ``reasoning_effort``.
    A model id routed to two different endpoints raises
    ``ValueError``.
    """
    snapshot = dict(env)
    endpoint_of: dict[str, str] = {}

    def assign(model: str, endpoint: str) -> None:
        current = endpoint_of.get(model)
        if current is not None and current != endpoint:
            raise ValueError(f"model {model!r} is routed to both {current!r} and {endpoint!r}")
        endpoint_of[model] = endpoint

    for priced in META_PRICES:
        assign(priced, META_ENDPOINT)
    for stage in config.proof.prover_stages:
        assign(stage.model, stage.endpoint)
    for model in _models_config_ids(config.models):
        match = next((stage.endpoint for stage in config.proof.prover_stages if stage.model == model), META_ENDPOINT)
        assign(model, match)

    given = dict(clients) if clients is not None else {}
    endpoint_clients: dict[str, ModelClient] = {}
    if META_ENDPOINT in given:
        endpoint_clients[META_ENDPOINT] = given[META_ENDPOINT]
    else:
        endpoint_clients[META_ENDPOINT] = OpenAICompatClient(
            snapshot.get("PCC_META_BASE_URL", META_BASE_URL_DEFAULT),
            snapshot.get("META_API_KEY", ""),
            prices=META_PRICES,
            reasoning_effort=snapshot.get("PCC_META_REASONING_EFFORT"),
        )
    glimmer_price = config.models.endpoint_prices.get(GLIMMER_ENDPOINT, (0.0, 0.0))
    override = snapshot.get("PCC_GLIMMER_MODEL") or None
    if override is not None:
        glimmer_prices = {override: glimmer_price}
    else:
        glimmer_prices = {model: glimmer_price for model, name in endpoint_of.items() if name == GLIMMER_ENDPOINT}
    if GLIMMER_ENDPOINT in given:
        endpoint_clients[GLIMMER_ENDPOINT] = given[GLIMMER_ENDPOINT]
    else:
        endpoint_clients[GLIMMER_ENDPOINT] = OpenAICompatClient(
            snapshot.get("PCC_GLIMMER_BASE_URL", ""),
            snapshot.get("PCC_GLIMMER_API_KEY", ""),
            prices=glimmer_prices,
        )
    if override is not None:
        sent_model = {model: override for model, name in endpoint_of.items() if name == GLIMMER_ENDPOINT}
    else:
        sent_model = {}
    return Router(endpoint_clients, endpoint_of, sent_model, snapshot)


def available_stages(config: Config, router: Router) -> Config:
    """The config with only the prover stages whose model is available through ``router``."""
    stages = tuple(stage for stage in config.proof.prover_stages if router.available(stage.model))
    return dataclasses.replace(config, proof=dataclasses.replace(config.proof, prover_stages=stages))


__all__ = ["ModelUnavailable", "Router", "available_stages", "build_client"]
