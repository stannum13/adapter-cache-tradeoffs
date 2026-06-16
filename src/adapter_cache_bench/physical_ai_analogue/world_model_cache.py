from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldModelCache:
    block_size: int = 8
    include_skill_in_key: bool = True
    seen: set[tuple[str, str, tuple[str, ...]]] = field(default_factory=set)
    lookups: int = 0
    hits: int = 0

    def _blocks(self, tokens: list[str]) -> list[tuple[str, ...]]:
        return [
            tuple(tokens[i : i + self.block_size]) for i in range(0, len(tokens), self.block_size)
        ]

    def estimate(self, skill_id: str, scene_tokens: list[str]) -> int:
        namespace = skill_id if self.include_skill_in_key else "shared-world"
        cached = 0
        for block in self._blocks(scene_tokens):
            self.lookups += 1
            if (namespace, "scene", block) not in self.seen:
                break
            self.hits += 1
            cached += len(block)
        return cached

    def observe(self, skill_id: str, scene_tokens: list[str]) -> None:
        namespace = skill_id if self.include_skill_in_key else "shared-world"
        for block in self._blocks(scene_tokens):
            self.seen.add((namespace, "scene", block))

    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0
