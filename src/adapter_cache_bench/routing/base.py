from __future__ import annotations

from abc import ABC, abstractmethod

from adapter_cache_bench.cache.cache_models import CacheModel
from adapter_cache_bench.config import RouterConfig
from adapter_cache_bench.routing.session_state import SessionState
from adapter_cache_bench.types import RequestRecord, RoutingDecision


class RouterPolicy(ABC):
    name = "base"

    def __init__(
        self, config: RouterConfig | None = None, state: SessionState | None = None
    ) -> None:
        self.config = config or RouterConfig()
        self.state = state or SessionState()

    @abstractmethod
    def choose(
        self, request: RequestRecord, adapter_ids: list[str], cache_model: CacheModel
    ) -> RoutingDecision:
        raise NotImplementedError

    def route(
        self, request: RequestRecord, adapter_ids: list[str], cache_model: CacheModel
    ) -> RoutingDecision:
        decision = self.choose(request, adapter_ids, cache_model)
        self.state.remember(request.session_id, decision.adapter_id)
        return decision


def make_router(config: RouterConfig, state: SessionState | None = None) -> RouterPolicy:
    if config.policy == "random":
        from adapter_cache_bench.routing.random_policy import RandomPolicy

        return RandomPolicy(config, state)
    if config.policy == "semantic":
        from adapter_cache_bench.routing.semantic_policy import SemanticPolicy

        return SemanticPolicy(config, state)
    if config.policy == "multitask":
        from adapter_cache_bench.routing.multitask_policy import MultitaskPolicy

        return MultitaskPolicy(config, state)
    if config.policy == "sticky_session":
        from adapter_cache_bench.routing.sticky_session_policy import StickySessionPolicy

        return StickySessionPolicy(config, state)
    if config.policy == "cache_aware":
        from adapter_cache_bench.routing.cache_aware_policy import CacheAwarePolicy

        return CacheAwarePolicy(config, state)
    if config.policy == "oracle":
        from adapter_cache_bench.routing.oracle_policy import OraclePolicy

        return OraclePolicy(config, state)
    raise ValueError(f"Unknown router policy: {config.policy}")
