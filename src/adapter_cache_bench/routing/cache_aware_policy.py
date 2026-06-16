from __future__ import annotations

from adapter_cache_bench.routing.base import RouterPolicy
from adapter_cache_bench.routing.scoring import quality_prior
from adapter_cache_bench.types import RequestRecord, RoutingDecision


class CacheAwarePolicy(RouterPolicy):
    name = "cache_aware"

    def choose(self, request: RequestRecord, adapter_ids, cache_model):
        best: tuple[float, str, int] | None = None
        current = self.state.session_adapter.get(request.session_id)
        for adapter_id in adapter_ids:
            cached = cache_model.estimate_cached_prefix_tokens(
                adapter_id, request.prompt, request.tenant_id, request.trust_group_id
            )
            queue_penalty = self.state.adapter_load.get(adapter_id, 0)
            switch_penalty = 1.0 if current and current != adapter_id else 0.0
            tenant_isolation_penalty = 0.0
            cold_penalty = 0.0 if adapter_id in self.state.warm_adapters else 1.0
            score = (
                quality_prior(request.task_type, adapter_id)
                + self.config.alpha * cached
                - self.config.beta * queue_penalty
                - self.config.gamma * switch_penalty
                - self.config.delta * tenant_isolation_penalty
                - self.config.epsilon * cold_penalty
            )
            if best is None or score > best[0]:
                best = (score, adapter_id, cached)
        assert best is not None
        return RoutingDecision(
            request_id=request.request_id,
            adapter_id=best[1],
            policy_name=self.name,
            score=best[0],
            reason="quality/cache/queue/session score",
            estimated_cached_prefix_tokens=best[2],
        )
