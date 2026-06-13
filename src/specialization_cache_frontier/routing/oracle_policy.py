from __future__ import annotations

from specialization_cache_frontier.routing.base import RouterPolicy
from specialization_cache_frontier.routing.scoring import quality_prior
from specialization_cache_frontier.types import RequestRecord, RoutingDecision


class OraclePolicy(RouterPolicy):
    name = "oracle"

    def choose(self, request: RequestRecord, adapter_ids, cache_model):
        best: tuple[float, str, int] | None = None
        for adapter_id in adapter_ids:
            cached = cache_model.estimate_cached_prefix_tokens(
                adapter_id, request.prompt, request.tenant_id, request.trust_group_id
            )
            goodput_bonus = cached / max(1, len(request.prompt.split()))
            expected_quality = quality_prior(request.task_type, adapter_id)
            score = expected_quality + 0.2 * goodput_bonus
            if best is None or score > best[0]:
                best = (score, adapter_id, cached)
        assert best is not None
        return RoutingDecision(
            request_id=request.request_id,
            adapter_id=best[1],
            policy_name=self.name,
            score=best[0],
            reason="simulated quality-adjusted goodput oracle",
            estimated_cached_prefix_tokens=best[2],
        )
