from __future__ import annotations

from typing import Any

import httpx

from specialization_cache_frontier.backends.base import Backend
from specialization_cache_frontier.cache.cache_models import CacheModel
from specialization_cache_frontier.config import BackendConfig
from specialization_cache_frontier.types import BackendResponse, RequestRecord, RoutingDecision


class VLLMBackend(Backend):
    """OpenAI-compatible vLLM client stub.

    This path is intentionally optional and is not exercised by unit tests unless
    callers opt into integration testing.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url, headers={"Authorization": f"Bearer {config.api_key}"}
        )

    def completion_payload(
        self, request: RequestRecord, decision: RoutingDecision
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": self.config.temperature,
            "extra_body": {**self.config.extra_body, "adapter": decision.adapter_id},
        }
        return payload

    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        del cache_model
        raise NotImplementedError(
            "VLLMBackend is a real-serving integration stub. Use MockBackend for CPU tests, "
            "or implement metric extraction around the OpenAI-compatible response."
        )
