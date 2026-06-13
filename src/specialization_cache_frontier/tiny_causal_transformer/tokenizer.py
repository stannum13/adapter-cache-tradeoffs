from __future__ import annotations


class TinyTokenizer:
    def __init__(self) -> None:
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        self.id_to_token = {0: "<pad>", 1: "<unk>"}

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for token in text.split():
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, 1) for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_token.get(idx, "<unk>") for idx in ids)
