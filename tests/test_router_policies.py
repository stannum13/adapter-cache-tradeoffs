from specialization_cache_frontier.cache.standard_lora_cache import StandardLoRACache
from specialization_cache_frontier.config import CacheConfig, RouterConfig
from specialization_cache_frontier.routing.base import make_router
from specialization_cache_frontier.types import RequestRecord


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


def test_cache_aware_can_choose_cached_adapter_when_quality_is_close():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    prompt = "shared prefix <ADAPTER:qa> answer"
    cache.observe_request("multitask", prompt, "t1", "g1")
    request = _request("qa")
    request.prompt = prompt
    router = make_router(RouterConfig(policy="cache_aware", alpha=0.10, epsilon=0.0))
    decision = router.route(request, ["qa", "multitask"], cache)
    assert decision.adapter_id == "multitask"


def test_oracle_prefers_quality_adjusted_goodput_candidate():
    cache = StandardLoRACache(CacheConfig(block_size=2))
    request = _request("summary")
    router = make_router(RouterConfig(policy="oracle"))

    decision = router.route(request, ["qa", "summary", "multitask"], cache)

    assert decision.adapter_id == "summary"
