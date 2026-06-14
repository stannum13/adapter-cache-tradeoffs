from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise SystemExit("Adapter evaluation stub. Use bench.run_workload for mock evaluation.")


if __name__ == "__main__":
    main()
