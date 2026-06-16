from __future__ import annotations

import json
import random
from typing import Any

from adapter_cache_bench.backends.base import Backend
from adapter_cache_bench.bench.quality import evaluate_prediction
from adapter_cache_bench.cache.cache_models import CacheModel
from adapter_cache_bench.cache.tokenizer import count_tokens
from adapter_cache_bench.config import BackendConfig
from adapter_cache_bench.routing.scoring import quality_prior
from adapter_cache_bench.types import (
    BackendResponse,
    QualityResult,
    RequestMetrics,
    RequestRecord,
    RoutingDecision,
)


class MockBackend(Backend):
    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()

    def _rng(self, request: RequestRecord, adapter_id: str) -> random.Random:
        seed = f"{self.config.seed}:{request.request_id}:{adapter_id}"
        return random.Random(seed)

    def _prediction_for_quality(
        self, request: RequestRecord, adapter_id: str, target_score: float
    ) -> tuple[str, Any]:
        if request.task_type == "json":
            if target_score > 0.55:
                return json.dumps(request.ground_truth), request.ground_truth
            return "{invalid json", request.ground_truth
        if request.task_type == "code":
            tests = []
            if isinstance(request.ground_truth, dict):
                tests = list(request.ground_truth.get("tests", []))
            passed = tests if target_score > 0.65 else tests[:1]
            return " ".join(str(test) for test in passed), request.ground_truth
        if request.task_type == "summary":
            return str(request.ground_truth), request.ground_truth
        if target_score > 0.65:
            return str(request.ground_truth), request.ground_truth
        return f"partial {request.task_type} answer", request.ground_truth

    def _quality(self, request: RequestRecord, adapter_id: str) -> QualityResult:
        rng = self._rng(request, adapter_id)
        target_score = min(
            1.0, max(0.0, quality_prior(request.task_type, adapter_id) + rng.gauss(0, 0.025))
        )
        prediction, ground_truth = self._prediction_for_quality(request, adapter_id, target_score)
        measured = evaluate_prediction(request.task_type, adapter_id, prediction, ground_truth)
        return measured.model_copy(update={"score": target_score})

    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        rng = self._rng(request, decision.adapter_id)
        prompt_tokens = count_tokens(request.prompt)
        cached = cache_model.estimate_cached_prefix_tokens(
            decision.adapter_id, request.prompt, request.tenant_id, request.trust_group_id
        )
        uncached = max(0, prompt_tokens - cached)
        output_tokens = max(
            1, min(request.max_tokens, int(12 + rng.random() * request.max_tokens / 2))
        )
        queue_ms = rng.uniform(self.config.queue_ms_min, self.config.queue_ms_max)
        prefill_ms = uncached * self.config.prefill_ms_per_token
        decode_ms = output_tokens * self.config.decode_ms_per_token
        ttft_ms = queue_ms + prefill_ms + self.config.first_token_ms
        e2e_ms = ttft_ms + decode_ms
        itl_ms = self.config.decode_ms_per_token
        tpot_ms = e2e_ms / max(1, output_tokens)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            uncached_prompt_tokens=uncached,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            queue_ms=queue_ms,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            tpot_ms=tpot_ms,
            e2e_ms=e2e_ms,
            output_tokens=output_tokens,
        )
        quality = self._quality(request, decision.adapter_id)
        if request.requires_json and quality.score > 0.55:
            text = json.dumps({"answer": request.ground_truth, "adapter": decision.adapter_id})
        else:
            text = f"{decision.adapter_id} response for {request.request_id}"
        cache_model.observe_request(
            decision.adapter_id, request.prompt, request.tenant_id, request.trust_group_id
        )
        return BackendResponse(
            request_id=request.request_id,
            adapter_id=decision.adapter_id,
            text=text,
            metrics=metrics,
            quality=quality,
        )
