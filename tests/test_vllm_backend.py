import asyncio

import httpx

from adapter_cache_bench.backends.vllm_backend import VLLMBackend
from adapter_cache_bench.cache.standard_lora_cache import StandardLoRACache
from adapter_cache_bench.config import BackendConfig, CacheConfig
from adapter_cache_bench.types import RequestRecord, RoutingDecision


def _request() -> RequestRecord:
    return RequestRecord(
        request_id="r1",
        session_id="s1",
        tenant_id="t1",
        trust_group_id="g1",
        task_type="qa",
        prompt="Document: shared text <ADAPTER:qa> answer",
        expected_adapter="qa",
        ground_truth="answer text",
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


def test_vllm_payload_uses_lora_adapter_as_model_name_when_configured():
    backend = VLLMBackend(
        BackendConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            adapter_model_names={
                "qa": "qa-lora",
                "json": "json-lora",
            },
        )
    )
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    payload = backend.completion_payload(_request(), decision)

    assert payload["model"] == "qa-lora"
    assert "extra_body" not in payload


def test_vllm_payload_falls_back_to_base_model_for_unmapped_adapter():
    backend = VLLMBackend(
        BackendConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            adapter_model_names={"json": "json-lora"},
        )
    )
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    payload = backend.completion_payload(_request(), decision)

    assert payload["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert "extra_body" not in payload


def test_vllm_payload_can_use_completion_endpoint_for_base_models():
    backend = VLLMBackend(
        BackendConfig(
            model="facebook/opt-125m",
            extra_body={"endpoint": "completions"},
        )
    )
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    payload = backend.completion_payload(_request(), decision)

    assert backend.request_path() == "/completions"
    assert payload["model"] == "facebook/opt-125m"
    assert payload["prompt"] == _request().prompt
    assert "messages" not in payload
    assert "endpoint" not in payload


def test_vllm_completion_payload_uses_lora_adapter_as_model_name():
    backend = VLLMBackend(
        BackendConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            adapter_model_names={"qa": "qa-lora"},
            extra_body={"endpoint": "completions"},
        )
    )
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    payload = backend.completion_payload(_request(), decision)

    assert backend.request_path() == "/completions"
    assert payload["model"] == "qa-lora"
    assert payload["prompt"] == _request().prompt


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
    assert response.quality.exact_match_like_score == 1.0
    assert response.quality.score == 1.0
    assert cache.estimate_cached_prefix_tokens("qa", _request().prompt, "t1", "g1") == 5


def test_vllm_backend_parses_completion_response_without_network():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/completions")
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "answer text"}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 2},
            },
        )

    client = httpx.Client(
        base_url="http://testserver/v1",
        transport=httpx.MockTransport(handler),
    )
    backend = VLLMBackend(
        BackendConfig(
            base_url="http://testserver/v1",
            extra_body={"endpoint": "completions"},
        ),
        client=client,
    )
    cache = StandardLoRACache(CacheConfig(block_size=2))
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    response = backend.generate(_request(), decision, cache)

    assert response.text == "answer text"
    assert response.metrics.prompt_tokens == 6
    assert response.metrics.output_tokens == 2


def test_vllm_backend_async_generate_without_network():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer text"}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 2},
            },
        )

    async def run_case():
        async with httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        ) as client:
            backend = VLLMBackend(
                BackendConfig(base_url="http://testserver/v1"),
                async_client=client,
            )
            cache = StandardLoRACache(CacheConfig(block_size=2))
            decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

            return await backend.async_generate(_request(), decision, cache)

    response = asyncio.run(run_case())
    assert response.text == "answer text"
    assert response.metrics.prompt_tokens == 6
    assert response.quality.score == 1.0
