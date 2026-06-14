from __future__ import annotations

from abc import ABC, abstractmethod

from specialization_cache_frontier.cache.cache_models import CacheModel
from specialization_cache_frontier.config import BackendConfig
from specialization_cache_frontier.types import BackendResponse, RequestRecord, RoutingDecision


class Backend(ABC):
    @abstractmethod
    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        raise NotImplementedError


def make_backend(config: BackendConfig) -> Backend:
    if config.kind == "mock":
        from specialization_cache_frontier.backends.mock_backend import MockBackend

        return MockBackend(config)
    if config.kind == "vllm":
        from specialization_cache_frontier.backends.vllm_backend import VLLMBackend

        return VLLMBackend(config)
    raise ValueError(f"Unknown backend kind: {config.kind}")
