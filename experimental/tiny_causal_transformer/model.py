from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class CausalSelfAttention:
    n_embd: int = 16

    def attend(self, sequence: list[float]) -> list[float]:
        out = []
        for i in range(len(sequence)):
            prefix = sequence[: i + 1]
            out.append(sum(prefix) / len(prefix))
        return out


@dataclass
class TransformerBlock:
    n_embd: int = 16

    def __post_init__(self) -> None:
        self.attn = CausalSelfAttention(self.n_embd)

    def forward(self, sequence: list[float]) -> list[float]:
        attended = self.attn.attend(sequence)
        return [math.tanh(x) for x in attended]


class TinyCausalTransformer:
    """A fundamentals-only decoder-style toy model, not the benchmark contribution."""

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)
        self.block = TransformerBlock()
        self.transition: dict[int, dict[int, int]] = {}

    def fit_next_token_counts(self, sequences: list[list[int]]) -> None:
        for sequence in sequences:
            for current, nxt in zip(sequence, sequence[1:], strict=False):
                self.transition.setdefault(current, {})
                self.transition[current][nxt] = self.transition[current].get(nxt, 0) + 1

    def next_token(self, prefix: list[int]) -> int:
        if not prefix:
            return self.rng.randrange(self.vocab_size)
        options = self.transition.get(prefix[-1])
        if not options:
            return self.rng.randrange(self.vocab_size)
        return max(options, key=options.get)

    def generate(self, prefix: list[int], max_new_tokens: int = 16) -> list[int]:
        output = list(prefix)
        for _ in range(max_new_tokens):
            output.append(self.next_token(output))
        return output
