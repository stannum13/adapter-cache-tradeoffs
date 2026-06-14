from __future__ import annotations

import argparse
from pathlib import Path

from adapter_cache_bench.config import load_config
from adapter_cache_bench.workloads.generator import generate_workload


def validate_workload_config(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    records = generate_workload(config.workload, config.cache)
    if not records:
        raise ValueError("Workload produced no records")
    missing_ground_truth = [record.request_id for record in records if record.ground_truth is None]
    if missing_ground_truth:
        raise ValueError(f"Records missing ground_truth: {missing_ground_truth}")
    missing_adapter = [record.request_id for record in records if not record.expected_adapter]
    if missing_adapter:
        raise ValueError(f"Records missing expected_adapter: {missing_adapter}")
    return {
        "config_path": str(config_path),
        "workload": config.workload.name,
        "request_count": len(records),
        "task_types": sorted({record.task_type for record in records}),
        "prompt_layouts": sorted({record.prompt_layout for record in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = validate_workload_config(args.config)
    print(result)


if __name__ == "__main__":
    main()
