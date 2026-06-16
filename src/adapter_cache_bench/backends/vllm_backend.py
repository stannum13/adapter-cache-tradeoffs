from __future__ import annotations

import json
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

    def __init__(
        self,
        config: BackendConfig,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=60.0,
        )
        self.async_client = async_client

    def endpoint(self) -> str:
        return str(self.config.extra_body.get("endpoint", "chat_completions"))

    def model_name(self, decision: RoutingDecision) -> str:
        return self.config.adapter_model_names.get(decision.adapter_id, self.config.model)

    def completion_payload(
        self, request: RequestRecord, decision: RoutingDecision
    ) -> dict[str, Any]:
        extra_body = {
            key: value
            for key, value in self.config.extra_body.items()
            if key not in {"endpoint", "stream"}
        }
        model_name = self.model_name(decision)
        if self.endpoint() in {"completions", "completion"}:
            payload = {
                "model": model_name,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": self.config.temperature,
            }
            if self.config.stream:
                payload["stream"] = True
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
        if self.config.stream:
            payload["stream"] = True
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

    def stream_delta_text(self, payload: dict[str, Any]) -> str:
        choice = payload.get("choices", [{}])[0]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("content") or "")
        return str(choice.get("text") or "")

    def parse_stream_line(self, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            return None
        return json.loads(line)

    def build_streaming_response(
        self,
        request: RequestRecord,
        decision: RoutingDecision,
        cache_model: CacheModel,
        text: str,
        ttft_ms: float,
        e2e_ms: float,
        cached: int,
    ) -> BackendResponse:
        prompt_tokens = count_tokens(request.prompt)
        output_tokens = count_tokens(text) or 1
        cached = min(cached, prompt_tokens)
        decode_ms = max(0.0, e2e_ms - ttft_ms)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            uncached_prompt_tokens=max(0, prompt_tokens - cached),
            prefill_ms=0.0,
            decode_ms=decode_ms,
            queue_ms=0.0,
            ttft_ms=ttft_ms,
            itl_ms=decode_ms / max(1, output_tokens - 1),
            tpot_ms=e2e_ms / max(1, output_tokens),
            e2e_ms=e2e_ms,
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

    def build_response(
        self,
        request: RequestRecord,
        decision: RoutingDecision,
        cache_model: CacheModel,
        payload: dict[str, Any],
        elapsed_ms: float,
        cached: int,
    ) -> BackendResponse:
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

    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        cached = cache_model.estimate_cached_prefix_tokens(
            decision.adapter_id,
            request.prompt,
            request.tenant_id,
            request.trust_group_id,
        )
        if self.config.stream:
            return self.generate_streaming(request, decision, cache_model, cached)
        started = time.perf_counter()
        response = self.client.post(
            self.request_path(), json=self.completion_payload(request, decision)
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        return self.build_response(
            request,
            decision,
            cache_model,
            response.json(),
            elapsed_ms,
            cached,
        )

    def generate_streaming(
        self,
        request: RequestRecord,
        decision: RoutingDecision,
        cache_model: CacheModel,
        cached: int,
    ) -> BackendResponse:
        started = time.perf_counter()
        first_token_ms: float | None = None
        chunks: list[str] = []
        with self.client.stream(
            "POST", self.request_path(), json=self.completion_payload(request, decision)
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                payload = self.parse_stream_line(line)
                if payload is None:
                    continue
                delta = self.stream_delta_text(payload)
                if not delta:
                    continue
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                chunks.append(delta)
        e2e_ms = (time.perf_counter() - started) * 1000.0
        return self.build_streaming_response(
            request,
            decision,
            cache_model,
            "".join(chunks),
            first_token_ms or e2e_ms,
            e2e_ms,
            cached,
        )

    async def async_generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        cached = cache_model.estimate_cached_prefix_tokens(
            decision.adapter_id,
            request.prompt,
            request.tenant_id,
            request.trust_group_id,
        )
        if self.config.stream:
            return await self.async_generate_streaming(request, decision, cache_model, cached)
        if self.async_client is None:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=60.0,
            ) as client:
                started = time.perf_counter()
                response = await client.post(
                    self.request_path(),
                    json=self.completion_payload(request, decision),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
        else:
            started = time.perf_counter()
            response = await self.async_client.post(
                self.request_path(),
                json=self.completion_payload(request, decision),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        return self.build_response(
            request,
            decision,
            cache_model,
            response.json(),
            elapsed_ms,
            cached,
        )

    async def async_generate_streaming(
        self,
        request: RequestRecord,
        decision: RoutingDecision,
        cache_model: CacheModel,
        cached: int,
    ) -> BackendResponse:
        if self.async_client is None:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=60.0,
            ) as client:
                return await self._async_generate_streaming_with_client(
                    client, request, decision, cache_model, cached
                )
        return await self._async_generate_streaming_with_client(
            self.async_client, request, decision, cache_model, cached
        )

    async def _async_generate_streaming_with_client(
        self,
        client: httpx.AsyncClient,
        request: RequestRecord,
        decision: RoutingDecision,
        cache_model: CacheModel,
        cached: int,
    ) -> BackendResponse:
        started = time.perf_counter()
        first_token_ms: float | None = None
        chunks: list[str] = []
        async with client.stream(
            "POST", self.request_path(), json=self.completion_payload(request, decision)
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                payload = self.parse_stream_line(line)
                if payload is None:
                    continue
                delta = self.stream_delta_text(payload)
                if not delta:
                    continue
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                chunks.append(delta)
        e2e_ms = (time.perf_counter() - started) * 1000.0
        return self.build_streaming_response(
            request,
            decision,
            cache_model,
            "".join(chunks),
            first_token_ms or e2e_ms,
            e2e_ms,
            cached,
        )
