from __future__ import annotations

from adapter_cache_bench.cache.cache_models import CacheModel


class BaseSharedCache(CacheModel):
    name = "base_shared"

    def _segments(self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str):
        del adapter_id
        tokens = self.tokenizer.encode(prompt)
        isolation = self._isolation_id(tenant_id, trust_group_id)
        namespace = [isolation, "base"]
        return [(namespace, namespace, tokens)]
