from __future__ import annotations


class WhitespaceTokenizer:
    """Unit-test-friendly tokenization for cache simulation."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def count_tokens(text: str) -> int:
    return len(WhitespaceTokenizer().encode(text))
