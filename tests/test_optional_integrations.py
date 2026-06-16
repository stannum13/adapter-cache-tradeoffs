import os

import pytest

from adapter_cache_bench.backends.vllm_backend import VLLMBackend
from adapter_cache_bench.config import BackendConfig
from adapter_cache_bench.types import RequestRecord, RoutingDecision

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VLLM_TESTS") != "1",
    reason="vLLM integration tests require RUN_VLLM_TESTS=1",
)


def test_vllm_payload_uses_openai_compatible_shape():
    backend = VLLMBackend(BackendConfig(model="served-causal-transformer"))
    request = RequestRecord(
        request_id="r1",
        session_id="s1",
        tenant_id="t1",
        trust_group_id="g1",
        task_type="qa",
        prompt="Document: shared text <ADAPTER:qa> answer",
        expected_adapter="qa",
        max_tokens=12,
    )
    decision = RoutingDecision(request_id="r1", adapter_id="qa", policy_name="semantic")

    payload = backend.completion_payload(request, decision)

    assert payload["model"] == "served-causal-transformer"
    assert payload["messages"][0]["content"] == request.prompt
    assert payload["max_tokens"] == 12
    assert payload["extra_body"]["adapter"] == "qa"
