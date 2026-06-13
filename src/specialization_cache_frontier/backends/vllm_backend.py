from __future__ import annotations

import time
from typing import Any

import httpx

from specialization_cache_frontier.backends.base import Backend
from specialization_cache_frontier.cache.cache_models import CacheModel
from specialization_cache_frontier.cache.tokenizer import count_tokens
from specialization_cache_frontier.config import BackendConfig
from specialization_cache_frontier.types import (
    BackendResponse,
    QualityResult,
    RequestMetrics,
    RequestRecord,
    RoutingDecision,
)


class VLLMBackend(Backend):
    """OpenAI-compatible vLLM client stub.

    This path is intentionally optional and is not exercised by unit tests unless
    callers opt into integration testing.
    """

    def __init__(self, config: BackendConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self.client = client or httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=60.0,
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
        cached = cache_model.estimate_cached_prefix_tokens(
            decision.adapter_id,
            request.prompt,
            request.tenant_id,
            request.trust_group_id,
        )
        started = time.perf_counter()
        response = self.client.post(
            "/chat/completions", json=self.completion_payload(request, decision)
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"].get("content", "")
        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens") or count_tokens(request.prompt))
        output_tokens = int(usage.get("completion_tokens") or count_tokens(text) or 1)
        cached = min(cached, prompt_tokens)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            uncached_prompt_tokens=max(0, prompt_tokens - cached),
            prefill_ms=0.0,
            decode_ms=0.0,
            queue_ms=0.0,
            ttft_ms=elapsed_ms,
            itl_ms=0.0,
            tpot_ms=elapsed_ms / max(1, output_tokens),
            e2e_ms=elapsed_ms,
            output_tokens=output_tokens,
        )
        quality = QualityResult(
            task_type=request.task_type,
            adapter_id=decision.adapter_id,
            score=0.0,
        )
        cache_model.observe_request(
            decision.adapter_id,
            request.prompt,
            request.tenant_id,
            request.trust_group_id,
        )
        return BackendResponse(
            request_id=request.request_id,
            adapter_id=decision.adapter_id,
            text=text,
            metrics=metrics,
            quality=quality,
        )
