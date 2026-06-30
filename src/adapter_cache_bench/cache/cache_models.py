from __future__ import annotations

from abc import ABC, abstractmethod

from adapter_cache_bench.cache.prefix_table import CacheBlock, PrefixTable, stable_hash
from adapter_cache_bench.cache.tokenizer import WhitespaceTokenizer
from adapter_cache_bench.config import CacheConfig


class CacheModel(ABC):
    name = "base"
    CONDITIONS = {"warm", "cold", "prefix_disabled"}

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        if self.config.condition not in self.CONDITIONS:
            raise ValueError(f"Unknown cache condition: {self.config.condition}")
        self.tokenizer = WhitespaceTokenizer()
        self.table = PrefixTable(
            max_memory_tokens=self.config.max_memory_tokens,
            eviction_policy=self.config.eviction_policy,
        )
        self.request_count = 0
        self.total_prompt_tokens = 0
        self.total_cached_prefix_tokens = 0

    def _reset_storage(self) -> None:
        self.table = PrefixTable(
            max_memory_tokens=self.config.max_memory_tokens,
            eviction_policy=self.config.eviction_policy,
        )

    def _isolation_id(self, tenant_id: str, trust_group_id: str) -> str:
        if self.config.cache_salt:
            return f"salt:{self.config.cache_salt}"
        if self.config.isolation_scope == "tenant":
            return f"tenant:{tenant_id}"
        if self.config.isolation_scope == "none":
            return "shared"
        return f"trust:{trust_group_id}"

    def _chunks(self, tokens: list[str]) -> list[list[str]]:
        size = max(1, self.config.block_size)
        return [tokens[i : i + size] for i in range(0, len(tokens), size)]

    def _block(
        self,
        namespace: list[str],
        logical_namespace: list[str],
        parent_hash: str,
        logical_parent_hash: str,
        tokens: list[str],
    ) -> CacheBlock:
        token_sig = stable_hash(tokens)
        key = stable_hash([*namespace, parent_hash, token_sig])
        logical_key = stable_hash([*logical_namespace, logical_parent_hash, token_sig])
        return CacheBlock(key=key, logical_key=logical_key, token_count=len(tokens))

    def _estimate_segments(
        self,
        segments: list[tuple[list[str], list[str], list[str]]],
        record_stats: bool = False,
    ) -> int:
        cached = 0
        parent_by_ns: dict[tuple[str, ...], str] = {}
        logical_parent_by_ns: dict[tuple[str, ...], str] = {}
        for namespace, logical_namespace, tokens in segments:
            ns_key = tuple(namespace)
            logical_ns_key = tuple(logical_namespace)
            parent = parent_by_ns.get(ns_key, "root")
            logical_parent = logical_parent_by_ns.get(logical_ns_key, "root")
            for chunk in self._chunks(tokens):
                block = self._block(namespace, logical_namespace, parent, logical_parent, chunk)
                hit = self.table.contains(block) if record_stats else block.key in self.table.blocks
                if not hit:
                    return cached
                cached += len(chunk)
                parent = block.key
                logical_parent = block.logical_key
            parent_by_ns[ns_key] = parent
            logical_parent_by_ns[logical_ns_key] = logical_parent
        return cached

    def _observe_segments(self, segments: list[tuple[list[str], list[str], list[str]]]) -> None:
        parent_by_ns: dict[tuple[str, ...], str] = {}
        logical_parent_by_ns: dict[tuple[str, ...], str] = {}
        for namespace, logical_namespace, tokens in segments:
            ns_key = tuple(namespace)
            logical_ns_key = tuple(logical_namespace)
            parent = parent_by_ns.get(ns_key, "root")
            logical_parent = logical_parent_by_ns.get(logical_ns_key, "root")
            for chunk in self._chunks(tokens):
                block = self._block(namespace, logical_namespace, parent, logical_parent, chunk)
                self.table.add(block)
                parent = block.key
                logical_parent = block.logical_key
            parent_by_ns[ns_key] = parent
            logical_parent_by_ns[logical_ns_key] = logical_parent

    @abstractmethod
    def _segments(
        self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str
    ) -> list[tuple[list[str], list[str], list[str]]]:
        raise NotImplementedError

    def estimate_cached_prefix_tokens(
        self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str
    ) -> int:
        if self.config.condition in {"cold", "prefix_disabled"}:
            return 0
        return self._estimate_segments(
            self._segments(adapter_id, prompt, tenant_id, trust_group_id)
        )

    def observe_request(
        self, adapter_id: str, prompt: str, tenant_id: str, trust_group_id: str
    ) -> None:
        tokens = self.tokenizer.encode(prompt)
        if self.config.condition == "prefix_disabled":
            self.request_count += 1
            self.total_prompt_tokens += len(tokens)
            return
        if self.config.condition == "cold":
            self._reset_storage()
        cached = self._estimate_segments(
            self._segments(adapter_id, prompt, tenant_id, trust_group_id),
            record_stats=True,
        )
        self.request_count += 1
        self.total_prompt_tokens += len(tokens)
        self.total_cached_prefix_tokens += cached
        self._observe_segments(self._segments(adapter_id, prompt, tenant_id, trust_group_id))

    def cache_hit_rate(self) -> float:
        return self.table.hit_rate()

    def cached_tokens(self) -> int:
        return self.total_cached_prefix_tokens

    def cached_prompt_token_ratio(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.total_cached_prefix_tokens / self.total_prompt_tokens

    def memory_tokens(self) -> int:
        return self.table.cached_tokens()

    def fragmentation_index(self) -> float:
        logical = self.table.logical_tokens()
        if logical == 0:
            return 0.0
        return self.memory_tokens() / logical

    def eviction_count(self) -> int:
        return self.table.eviction_count

    def evicted_tokens(self) -> int:
        return self.table.evicted_token_count


def make_cache_model(config: CacheConfig) -> CacheModel:
    if config.model == "standard_lora":
        from adapter_cache_bench.cache.standard_lora_cache import StandardLoRACache

        return StandardLoRACache(config)
    if config.model == "base_shared":
        from adapter_cache_bench.cache.base_shared_cache import BaseSharedCache

        return BaseSharedCache(config)
    if config.model == "activated_lora":
        from adapter_cache_bench.cache.activated_lora_cache import ActivatedLoRACache

        return ActivatedLoRACache(config)
    if config.model == "copy_on_write":
        from adapter_cache_bench.cache.copy_on_write_cache import CopyOnWriteCache

        return CopyOnWriteCache(config)
    raise ValueError(f"Unknown cache model: {config.model}")
