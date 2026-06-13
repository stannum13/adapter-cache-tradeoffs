from __future__ import annotations

from specialization_cache_frontier.cache.cache_models import CacheModel


class ActivatedLoRACache(CacheModel):
    name = "activated_lora"

    def _find_invocation_index(self, tokens: list[str]) -> int:
        markers = set(self.config.invocation_markers.values())
        for index, token in enumerate(tokens):
            if token in markers:
                return index
        return len(tokens)

    def _segments(self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str):
        tokens = self.tokenizer.encode(prompt)
        split = self._find_invocation_index(tokens)
        isolation = self._isolation_id(tenant_id, trust_group_id)
        base_tokens = tokens[:split]
        adapter_tokens = tokens[split:]
        segments = []
        if base_tokens:
            base_namespace = [isolation, "activated-base"]
            segments.append((base_namespace, base_namespace, base_tokens))
        if adapter_tokens:
            namespace = [isolation, f"adapter:{adapter_id}"]
            logical_namespace = [isolation, "post-invocation"]
            segments.append((namespace, logical_namespace, adapter_tokens))
        return segments
