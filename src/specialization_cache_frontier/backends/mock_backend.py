from __future__ import annotations

import json
import random

from specialization_cache_frontier.backends.base import Backend
from specialization_cache_frontier.cache.cache_models import CacheModel
from specialization_cache_frontier.cache.tokenizer import count_tokens
from specialization_cache_frontier.config import BackendConfig
from specialization_cache_frontier.routing.scoring import quality_prior
from specialization_cache_frontier.types import (
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

    def _quality(self, request: RequestRecord, adapter_id: str) -> QualityResult:
        rng = self._rng(request, adapter_id)
        score = min(
            1.0, max(0.0, quality_prior(request.task_type, adapter_id) + rng.gauss(0, 0.025))
        )
        kwargs = {}
        if request.task_type == "json":
            kwargs = {
                "valid_json_rate": score,
                "schema_match": score * 0.97,
                "field_f1": score * 0.94,
            }
        elif request.task_type == "qa":
            kwargs = {"exact_match_like_score": score}
        elif request.task_type == "code":
            kwargs = {"unit_test_like_score": score}
        elif request.task_type == "summary":
            kwargs = {"rubric_score": score}
        return QualityResult(
            task_type=request.task_type, adapter_id=adapter_id, score=score, **kwargs
        )

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
