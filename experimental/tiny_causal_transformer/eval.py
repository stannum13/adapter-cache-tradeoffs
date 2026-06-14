from __future__ import annotations

import math

from experimental.tiny_causal_transformer.dataset import public_domain_tiny_corpus
from experimental.tiny_causal_transformer.train import train


def perplexity() -> float:
    model, tokenizer = train()
    losses = []
    for text in public_domain_tiny_corpus():
        ids = tokenizer.encode(text)
        for i in range(1, len(ids)):
            pred = model.next_token(ids[:i])
            losses.append(0.0 if pred == ids[i] else math.log(len(tokenizer.token_to_id)))
    return math.exp(sum(losses) / max(1, len(losses)))


if __name__ == "__main__":
    print(f"perplexity={perplexity():.3f}")
