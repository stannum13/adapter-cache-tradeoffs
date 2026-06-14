from __future__ import annotations

import time
from typing import Any

import httpx

from adapter_cache_bench.backends.base import Backend
from adapter_cache_bench.bench.quality import evaluate_prediction
from adapter_cache_bench.cache.cache_models import CacheModel
from adapter_cache_bench.cache.tokenizer import count_tokens
from adapter_cache_bench.config import BackendConfig
from adapter_cache_bench.types import (
    BackendResponse,
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

    def endpoint(self) -> str:
        return str(self.config.extra_body.get("endpoint", "chat_completions"))

    def model_name(self, decision: RoutingDecision) -> str:
        return self.config.adapter_model_names.get(decision.adapter_id, self.config.model)

    def completion_payload(
        self, request: RequestRecord, decision: RoutingDecision
    ) -> dict[str, Any]:
        extra_body = {
            key: value for key, value in self.config.extra_body.items() if key != "endpoint"
        }
        model_name = self.model_name(decision)
        if self.endpoint() in {"completions", "completion"}:
            payload = {
                "model": model_name,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": self.config.temperature,
            }
            payload.update(extra_body)
            return payload
        adapter_extra_body = dict(extra_body)
        if not self.config.adapter_model_names:
            adapter_extra_body["adapter"] = decision.adapter_id
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": self.config.temperature,
        }
        if adapter_extra_body:
            payload["extra_body"] = adapter_extra_body
        return payload

    def request_path(self) -> str:
        if self.endpoint() in {"completions", "completion"}:
            return "/completions"
        return "/chat/completions"

    def response_text(self, payload: dict[str, Any]) -> str:
        choice = payload["choices"][0]
        if "text" in choice:
            return str(choice.get("text") or "")
        return str(choice.get("message", {}).get("content", ""))

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
            self.request_path(), json=self.completion_payload(request, decision)
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()
        text = self.response_text(payload)
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
        quality = evaluate_prediction(
            task_type=request.task_type,
            adapter_id=decision.adapter_id,
            prediction=text,
            ground_truth=request.ground_truth,
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
