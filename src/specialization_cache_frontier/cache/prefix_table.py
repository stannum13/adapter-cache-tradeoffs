from __future__ import annotations

import hashlib
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
    def __init__(self) -> None:
        self.blocks: dict[str, CacheBlock] = {}
        self.logical_blocks: dict[str, int] = {}
        self.lookup_count = 0
        self.hit_count = 0

    def contains(self, block: CacheBlock) -> bool:
        self.lookup_count += 1
        hit = block.key in self.blocks
        if hit:
            self.hit_count += 1
        return hit

    def add(self, block: CacheBlock) -> None:
        self.blocks[block.key] = block
        self.logical_blocks.setdefault(block.logical_key, block.token_count)

    def cached_tokens(self) -> int:
        return sum(block.token_count for block in self.blocks.values())

    def logical_tokens(self) -> int:
        return sum(self.logical_blocks.values())

    def hit_rate(self) -> float:
        if self.lookup_count == 0:
            return 0.0
        return self.hit_count / self.lookup_count
