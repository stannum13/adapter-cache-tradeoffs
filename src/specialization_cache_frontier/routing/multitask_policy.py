from __future__ import annotations

from specialization_cache_frontier.routing.base import RouterPolicy
from specialization_cache_frontier.routing.scoring import quality_prior
from specialization_cache_frontier.types import RequestRecord, RoutingDecision


class MultitaskPolicy(RouterPolicy):
    name = "multitask"

    def choose(self, request: RequestRecord, adapter_ids, cache_model):
        adapter_id = "multitask" if "multitask" in adapter_ids else adapter_ids[0]
        cached = cache_model.estimate_cached_prefix_tokens(
            adapter_id, request.prompt, request.tenant_id, request.trust_group_id
        )
        return RoutingDecision(
            request_id=request.request_id,
            adapter_id=adapter_id,
            policy_name=self.name,
            score=quality_prior(request.task_type, adapter_id),
            reason="force multitask adapter baseline",
            estimated_cached_prefix_tokens=cached,
        )
