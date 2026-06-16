from __future__ import annotations

from experimental.tiny_causal_transformer.dataset import public_domain_tiny_corpus
from experimental.tiny_causal_transformer.model import TinyCausalTransformer
from experimental.tiny_causal_transformer.tokenizer import TinyTokenizer


def train() -> tuple[TinyCausalTransformer, TinyTokenizer]:
    corpus = public_domain_tiny_corpus()
    tokenizer = TinyTokenizer()
    tokenizer.fit(corpus)
    model = TinyCausalTransformer(vocab_size=len(tokenizer.token_to_id))
    model.fit_next_token_counts([tokenizer.encode(text) for text in corpus])
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train()
    print(tokenizer.decode(model.generate(tokenizer.encode("prefix cache"), 8)))
