from adapter_cache_bench.cache.standard_lora_cache import StandardLoRACache
from adapter_cache_bench.config import CacheConfig, RouterConfig
from adapter_cache_bench.routing.base import make_router
from adapter_cache_bench.routing.session_state import SessionState
from adapter_cache_bench.types import RequestRecord


def _request(task="json", session="s1"):
    return RequestRecord(
        request_id="r1",
        session_id=session,
        tenant_id="t1",
        trust_group_id="g1",
        task_type=task,
        prompt="shared prefix <ADAPTER:json> extract",
        expected_adapter=task,
        ground_truth={},
    )


def test_semantic_routes_by_task_type():
    router = make_router(RouterConfig(policy="semantic"))
    cache = StandardLoRACache(CacheConfig(block_size=2))
    decision = router.route(_request("json"), ["qa", "json", "multitask"], cache)
    assert decision.adapter_id == "json"


def test_multitask_policy_forces_multitask_adapter():
    router = make_router(RouterConfig(policy="multitask"))
    cache = StandardLoRACache(CacheConfig(block_size=2))

    decision = router.route(_request("json"), ["qa", "json", "multitask"], cache)

    assert decision.adapter_id == "multitask"


def test_random_policy_is_deterministic_for_seed():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    first = make_router(RouterConfig(policy="random", seed=3))
    second = make_router(RouterConfig(policy="random", seed=3))

    assert (
        first.route(_request("qa"), ["qa", "json"], cache).adapter_id
        == second.route(_request("qa"), ["qa", "json"], cache).adapter_id
    )


def test_sticky_session_reuses_compatible_adapter():
    router = make_router(RouterConfig(policy="sticky_session"))
    cache = StandardLoRACache(CacheConfig(block_size=2))
    first = router.route(_request("qa"), ["qa", "json", "multitask"], cache)
    second = router.route(_request("qa"), ["qa", "json", "multitask"], cache)
    assert first.adapter_id == second.adapter_id == "qa"


def test_session_state_tracks_live_dispatch_load_separately_from_route_memory():
    state = SessionState()

    state.remember("s1", "qa", tenant_id="t1", trust_group_id="g1")
    assert state.session_adapter["s1"] == "qa"
    assert state.adapter_assignment_count["qa"] == 1
    assert state.active_load("qa") == 0

    state.begin_dispatch("qa")
    state.begin_dispatch("qa")
    assert state.active_load("qa") == 2

    state.end_dispatch("qa")
    assert state.active_load("qa") == 1
    state.end_dispatch("qa")
    state.end_dispatch("qa")
    assert state.active_load("qa") == 0


def test_cache_aware_can_choose_cached_adapter_when_quality_is_close():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    prompt = "shared prefix <ADAPTER:qa> answer"
    cache.observe_request("multitask", prompt, "t1", "g1")
    request = _request("qa")
    request.prompt = prompt
    router = make_router(RouterConfig(policy="cache_aware", alpha=0.10, epsilon=0.0))
    decision = router.route(request, ["qa", "multitask"], cache)
    assert decision.adapter_id == "multitask"
    assert decision.simulated_cached_prefix_tokens == decision.estimated_cached_prefix_tokens


def test_cache_aware_queue_penalty_uses_live_dispatch_load_only():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    state = SessionState(adapter_load={"qa": 2})
    router = make_router(
        RouterConfig(policy="cache_aware", alpha=0.0, beta=1.0, gamma=0.0, epsilon=0.0),
        state,
    )

    decision = router.route(_request("qa"), ["qa", "multitask"], cache)

    assert decision.adapter_id == "multitask"
    assert state.active_load("qa") == 2


def test_cache_aware_penalizes_cross_trust_group_reuse():
    cache = StandardLoRACache(CacheConfig(block_size=2, isolation_scope="trust_group"))
    state = SessionState()
    state.remember("s1", "qa", tenant_id="t1", trust_group_id="g1")
    router = make_router(
        RouterConfig(policy="cache_aware", alpha=0.0, beta=0.0, gamma=0.0, delta=10.0),
        state,
    )
    request = _request("qa", session="s2")
    request.trust_group_id = "g2"

    decision = router.route(request, ["qa", "multitask"], cache)

    assert decision.adapter_id == "multitask"


def test_oracle_prefers_quality_adjusted_goodput_candidate():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    request = _request("summary")
    router = make_router(RouterConfig(policy="oracle"))

    decision = router.route(request, ["qa", "summary", "multitask"], cache)

    assert decision.adapter_id == "summary"
