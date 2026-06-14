from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass


def stable_hash(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


@dataclass(frozen=True)
class CacheBlock:
    key: str
    logical_key: str
    token_count: int


class PrefixTable:
    def __init__(self, max_memory_tokens: int | None = None, eviction_policy: str = "lru") -> None:
        self.max_memory_tokens = max_memory_tokens
        self.eviction_policy = eviction_policy
        self.blocks: OrderedDict[str, CacheBlock] = OrderedDict()
        self.logical_blocks: dict[str, int] = {}
        self.lookup_count = 0
        self.hit_count = 0
        self.eviction_count = 0
        self.evicted_token_count = 0

    def contains(self, block: CacheBlock) -> bool:
        self.lookup_count += 1
        hit = block.key in self.blocks
        if hit:
            self.hit_count += 1
            if self.eviction_policy == "lru":
                self.blocks.move_to_end(block.key)
        return hit

    def add(self, block: CacheBlock) -> None:
        if self.max_memory_tokens is not None and block.token_count > self.max_memory_tokens:
            return
        if block.key in self.blocks:
            self.blocks[block.key] = block
            if self.eviction_policy == "lru":
                self.blocks.move_to_end(block.key)
            return
        self.blocks[block.key] = block
        self.logical_blocks.setdefault(block.logical_key, block.token_count)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if self.max_memory_tokens is None:
            return
        if self.eviction_policy != "lru":
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")
        while self.cached_tokens() > self.max_memory_tokens and self.blocks:
            _, evicted = self.blocks.popitem(last=False)
            self.eviction_count += 1
            self.evicted_token_count += evicted.token_count

    def cached_tokens(self) -> int:
        return sum(block.token_count for block in self.blocks.values())

    def logical_tokens(self) -> int:
        return sum(self.logical_blocks.values())

    def hit_rate(self) -> float:
        if self.lookup_count == 0:
            return 0.0
        return self.hit_count / self.lookup_count
