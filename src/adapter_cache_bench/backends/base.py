from __future__ import annotations

from abc import ABC, abstractmethod

from adapter_cache_bench.cache.cache_models import CacheModel
from adapter_cache_bench.config import BackendConfig
from adapter_cache_bench.types import BackendResponse, RequestRecord, RoutingDecision


class Backend(ABC):
    @abstractmethod
    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        raise NotImplementedError


def make_backend(config: BackendConfig) -> Backend:
    if config.kind == "mock":
        from adapter_cache_bench.backends.mock_backend import MockBackend

        return MockBackend(config)
    if config.kind == "vllm":
        from adapter_cache_bench.backends.vllm_backend import VLLMBackend

        return VLLMBackend(config)
    raise ValueError(f"Unknown backend kind: {config.kind}")
