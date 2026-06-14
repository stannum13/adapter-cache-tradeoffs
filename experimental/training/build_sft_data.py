from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(f"SFT data export stub. Use workload generator to materialize {args.output}.")


if __name__ == "__main__":
    main()
