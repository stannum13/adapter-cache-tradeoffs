from specialization_cache_frontier.tiny_causal_transformer.eval import perplexity
from specialization_cache_frontier.tiny_causal_transformer.train import train


def test_tiny_causal_transformer_trains_and_generates():
    model, tokenizer = train()
    prefix = tokenizer.encode("prefix cache")

    generated = model.generate(prefix, max_new_tokens=4)

    assert len(generated) == len(prefix) + 4
    assert tokenizer.decode(generated)


def test_tiny_causal_transformer_perplexity_is_finite():
    assert perplexity() >= 1.0
