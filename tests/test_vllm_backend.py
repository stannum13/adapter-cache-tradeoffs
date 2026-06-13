import httpx

from specialization_cache_frontier.backends.vllm_backend import VLLMBackend
from specialization_cache_frontier.cache.standard_lora_cache import StandardLoRACache
from specialization_cache_frontier.config import BackendConfig, CacheConfig
from specialization_cache_frontier.types import RequestRecord, RoutingDecision


def _request() -> RequestRecord:
    return RequestRecord(
        request_id="r1",
        session_id="s1",
        tenant_id="t1",
        trust_group_id="g1",
        task_type="qa",
        prompt="Document: shared text <ADAPTER:qa> answer",
        expected_adapter="qa",
        max_tokens=12,
    )


def test_vllm_payload_uses_openai_compatible_shape():
    backend = VLLMBackend(BackendConfig(model="served-causal-transformer"))
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    payload = backend.completion_payload(_request(), decision)

    assert payload["model"] == "served-causal-transformer"
    assert payload["messages"][0]["content"] == _request().prompt
    assert payload["max_tokens"] == 12
    assert payload["extra_body"]["adapter"] == "qa"


def test_vllm_backend_parses_openai_compatible_response_without_network():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer text"}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 2},
            },
        )

    client = httpx.Client(
        base_url="http://testserver/v1",
        transport=httpx.MockTransport(handler),
    )
    backend = VLLMBackend(BackendConfig(base_url="http://testserver/v1"), client=client)
    cache = StandardLoRACache(CacheConfig(block_size=2))
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    response = backend.generate(_request(), decision, cache)

    assert response.text == "answer text"
    assert response.metrics.prompt_tokens == 6
    assert response.metrics.output_tokens == 2
    assert cache.estimate_cached_prefix_tokens("qa", _request().prompt, "t1", "g1") == 5
