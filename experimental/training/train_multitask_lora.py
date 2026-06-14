from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.parse_args()
    raise SystemExit(
        "Multitask LoRA training stub. The mock benchmark does not require GPU training."
    )


if __name__ == "__main__":
    main()
