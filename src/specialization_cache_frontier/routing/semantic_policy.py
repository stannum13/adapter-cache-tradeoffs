from __future__ import annotations

from specialization_cache_frontier.routing.base import RouterPolicy
from specialization_cache_frontier.routing.scoring import expected_adapter_for_task, quality_prior
from specialization_cache_frontier.types import RequestRecord, RoutingDecision


class SemanticPolicy(RouterPolicy):
    name = "semantic"

    def choose(self, request: RequestRecord, adapter_ids, cache_model):
        adapter_id = expected_adapter_for_task(request.task_type)
        if adapter_id not in adapter_ids:
            adapter_id = (
                request.expected_adapter
                if request.expected_adapter in adapter_ids
                else adapter_ids[0]
            )
        cached = cache_model.estimate_cached_prefix_tokens(
            adapter_id, request.prompt, request.tenant_id, request.trust_group_id
        )
        return RoutingDecision(
            request_id=request.request_id,
            adapter_id=adapter_id,
            policy_name=self.name,
            score=quality_prior(request.task_type, adapter_id),
            reason="task-type specialist",
            estimated_cached_prefix_tokens=cached,
        )
