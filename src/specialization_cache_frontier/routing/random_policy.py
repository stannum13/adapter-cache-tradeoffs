from __future__ import annotations

import random

from specialization_cache_frontier.routing.base import RouterPolicy
from specialization_cache_frontier.types import RequestRecord, RoutingDecision


class RandomPolicy(RouterPolicy):
    name = "random"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rng = random.Random(self.config.seed)

    def choose(self, request: RequestRecord, adapter_ids, cache_model):
        adapter_id = self.rng.choice(adapter_ids)
        cached = cache_model.estimate_cached_prefix_tokens(
            adapter_id, request.prompt, request.tenant_id, request.trust_group_id
        )
        return RoutingDecision(
            request_id=request.request_id,
            adapter_id=adapter_id,
            policy_name=self.name,
            score=0.0,
            reason="deterministic random sample",
            estimated_cached_prefix_tokens=cached,
        )
