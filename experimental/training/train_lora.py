from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.parse_args()
    raise SystemExit("LoRA training is intentionally out of unit-test scope for the first pass.")


if __name__ == "__main__":
    main()
