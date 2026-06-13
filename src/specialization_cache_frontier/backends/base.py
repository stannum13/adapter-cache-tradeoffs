from __future__ import annotations

from abc import ABC, abstractmethod

from specialization_cache_frontier.cache.cache_models import CacheModel
from specialization_cache_frontier.types import BackendResponse, RequestRecord, RoutingDecision


class Backend(ABC):
    @abstractmethod
    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        raise NotImplementedError
