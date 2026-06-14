from __future__ import annotations

from math import ceil

from adapter_cache_bench.cache.cache_models import CacheModel


class CopyOnWriteCache(CacheModel):
    name = "copy_on_write"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._delta_tokens: dict[str, int] = {}

    def _find_invocation_index(self, tokens: list[str]) -> int:
        markers = set(self.config.invocation_markers.values())
        for index, token in enumerate(tokens):
            if token in markers:
                return index
        return len(tokens)

    def _segments(self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str):
        tokens = self.tokenizer.encode(prompt)
        isolation = self._isolation_id(tenant_id, trust_group_id)
        base_namespace = [isolation, "cow-base"]
        return [(base_namespace, base_namespace, tokens)]

    def observe_request(
        self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str
    ) -> None:
        tokens = self.tokenizer.encode(prompt)
        split = self._find_invocation_index(tokens)
        post_tokens = max(1, len(tokens) - split)
        isolation = self._isolation_id(tenant_id, trust_group_id)
        delta = ceil(post_tokens * self.config.copy_on_write_delta_fraction)
        self._delta_tokens[f"{isolation}:{adapter_id}:{split}:{post_tokens}"] = delta
        super().observe_request(adapter_id, prompt, tenant_id, trust_group_id)

    def memory_tokens(self) -> int:
        return self.table.cached_tokens() + sum(self._delta_tokens.values())
