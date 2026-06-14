from __future__ import annotations

import argparse

from experimental.tiny_causal_transformer.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    model, tokenizer = train()
    print(tokenizer.decode(model.generate(tokenizer.encode(args.prompt), args.max_new_tokens)))


if __name__ == "__main__":
    main()
