from specialization_cache_frontier.backends.mock_backend import MockBackend
from specialization_cache_frontier.cache.standard_lora_cache import StandardLoRACache
from specialization_cache_frontier.config import CacheConfig
from specialization_cache_frontier.types import RequestRecord, RoutingDecision


def test_mock_backend_returns_metrics_and_observes_cache():
    request = RequestRecord(
        request_id="r1",
        session_id="s1",
        tenant_id="t1",
        trust_group_id="g1",
        task_type="qa",
        prompt="alpha beta gamma <ADAPTER:qa> question",
        expected_adapter="qa",
    )
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")
    cache = StandardLoRACache(CacheConfig(block_size=2))
    response = MockBackend().generate(request, decision, cache)
    assert response.metrics.prompt_tokens == 5
    assert response.metrics.uncached_prompt_tokens == 5
    assert response.quality.score > 0.7
    assert cache.estimate_cached_prefix_tokens("qa", request.prompt, "t1", "g1") == 5
