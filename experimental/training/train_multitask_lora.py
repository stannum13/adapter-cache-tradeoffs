from __future__ import annotations

import argparse

from experimental.training.train_lora import train_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    print(
        train_lora(
            args.base_model,
            args.train_file,
            None,
            args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            load_in_4bit=args.load_in_4bit,
        )
    )


if __name__ == "__main__":
    main()
