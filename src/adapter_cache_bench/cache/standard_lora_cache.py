from __future__ import annotations

from adapter_cache_bench.cache.cache_models import CacheModel


class StandardLoRACache(CacheModel):
    name = "standard_lora"

    def _segments(self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str):
        tokens = self.tokenizer.encode(prompt)
        isolation = self._isolation_id(tenant_id, trust_group_id)
        namespace = [isolation, f"adapter:{adapter_id}"]
        logical_namespace = [isolation, "prompt"]
        return [(namespace, logical_namespace, tokens)]
