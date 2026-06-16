from __future__ import annotations

from adapter_cache_bench.routing.base import RouterPolicy
from adapter_cache_bench.routing.scoring import expected_adapter_for_task, quality_prior
from adapter_cache_bench.types import RequestRecord, RoutingDecision


class StickySessionPolicy(RouterPolicy):
    name = "sticky_session"

    def choose(self, request: RequestRecord, adapter_ids, cache_model):
        current = self.state.session_adapter.get(request.session_id)
        if current and current in adapter_ids and quality_prior(request.task_type, current) >= 0.70:
            adapter_id = current
            reason = "session affinity"
        else:
            adapter_id = expected_adapter_for_task(request.task_type)
            if adapter_id not in adapter_ids:
                adapter_id = "multitask" if "multitask" in adapter_ids else adapter_ids[0]
            reason = "new or incompatible session adapter"
        cached = cache_model.estimate_cached_prefix_tokens(
            adapter_id, request.prompt, request.tenant_id, request.trust_group_id
        )
        return RoutingDecision(
            request_id=request.request_id,
            adapter_id=adapter_id,
            policy_name=self.name,
            score=quality_prior(request.task_type, adapter_id),
            reason=reason,
            estimated_cached_prefix_tokens=cached,
        )
