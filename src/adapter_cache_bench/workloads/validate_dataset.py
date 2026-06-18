from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adapter_cache_bench.config import load_config
from adapter_cache_bench.workloads.generator import generate_workload


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def validate_workload_config(
    config_path: str | Path,
    *,
    min_records: int = 1,
    required_tasks: set[str] | None = None,
    required_layouts: set[str] | None = None,
    balanced_tasks: bool = False,
    min_shared_prefix_groups: int = 1,
    require_tenant_fields: bool = False,
    require_source_fields: bool = False,
    require_public_domain_license: bool = False,
) -> dict[str, object]:
    config = load_config(config_path)
    raw_rows = []
    dataset_path = config.workload.dataset_path
    if dataset_path:
        path = Path(dataset_path)
        if path.suffix in {".jsonl", ".json"} and path.exists():
            raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = generate_workload(config.workload, config.cache)
    if not records:
        raise ValueError("Workload produced no records")
    if len(records) < min_records:
        raise ValueError(f"Expected at least {min_records} records, found {len(records)}")
    missing_ground_truth = [record.request_id for record in records if record.ground_truth is None]
    if missing_ground_truth:
        raise ValueError(f"Records missing ground_truth: {missing_ground_truth}")
    missing_adapter = [record.request_id for record in records if not record.expected_adapter]
    if missing_adapter:
        raise ValueError(f"Records missing expected_adapter: {missing_adapter}")
    if require_tenant_fields:
        missing_tenant = [
            record.request_id
            for record in records
            if not record.tenant_id or not record.trust_group_id
        ]
        if missing_tenant:
            raise ValueError(f"Records missing tenant/trust-group fields: {missing_tenant}")
    if require_source_fields:
        if len(raw_rows) < len(records):
            raise ValueError("Source-field validation requires a JSONL dataset path")
        missing_source_fields = [
            str(row.get("request_id", index))
            for index, row in enumerate(raw_rows[: len(records)])
            if not row.get("source_title")
            or not row.get("source_url")
            or not row.get("source_license")
        ]
        if missing_source_fields:
            raise ValueError(f"Records missing source provenance: {missing_source_fields}")
    if require_public_domain_license:
        if len(raw_rows) < len(records):
            raise ValueError("Public-domain license validation requires a JSONL dataset path")
        non_public_domain = [
            str(row.get("request_id", index))
            for index, row in enumerate(raw_rows[: len(records)])
            if row.get("source_license") != "public-domain"
        ]
        if non_public_domain:
            raise ValueError(f"Records without public-domain license: {non_public_domain}")

    task_counts = Counter(record.task_type for record in records)
    layout_counts = Counter(record.prompt_layout for record in records)
    shared_prefix_counts = Counter(record.shared_prefix_id for record in records)
    tenant_counts = Counter(record.tenant_id for record in records)
    trust_group_counts = Counter(record.trust_group_id for record in records)

    required_tasks = required_tasks or set()
    missing_tasks = required_tasks - set(task_counts)
    if missing_tasks:
        raise ValueError(f"Missing required task types: {sorted(missing_tasks)}")
    required_layouts = required_layouts or set()
    missing_layouts = required_layouts - set(layout_counts)
    if missing_layouts:
        raise ValueError(f"Missing required prompt layouts: {sorted(missing_layouts)}")
    if balanced_tasks and task_counts:
        counts = [task_counts[task] for task in sorted(required_tasks or task_counts)]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"Task counts are not balanced: {dict(task_counts)}")
    repeated_prefix_groups = sum(1 for count in shared_prefix_counts.values() if count > 1)
    if repeated_prefix_groups < min_shared_prefix_groups:
        raise ValueError(
            "Expected at least "
            f"{min_shared_prefix_groups} repeated shared-prefix groups, found "
            f"{repeated_prefix_groups}"
        )
    return {
        "config_path": str(config_path),
        "workload": config.workload.name,
        "request_count": len(records),
        "task_types": sorted(task_counts),
        "task_counts": dict(sorted(task_counts.items())),
        "prompt_layouts": sorted(layout_counts),
        "prompt_layout_counts": dict(sorted(layout_counts.items())),
        "shared_prefix_groups": len(shared_prefix_counts),
        "repeated_shared_prefix_groups": repeated_prefix_groups,
        "tenant_count": len(tenant_counts),
        "trust_group_count": len(trust_group_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--require-tasks", default="")
    parser.add_argument("--require-layouts", default="")
    parser.add_argument("--balanced-tasks", action="store_true")
    parser.add_argument("--min-shared-prefix-groups", type=int, default=1)
    parser.add_argument("--require-tenant-fields", action="store_true")
    parser.add_argument("--require-source-fields", action="store_true")
    parser.add_argument("--require-public-domain-license", action="store_true")
    args = parser.parse_args()
    result = validate_workload_config(
        args.config,
        min_records=args.min_records,
        required_tasks=_parse_csv(args.require_tasks),
        required_layouts=_parse_csv(args.require_layouts),
        balanced_tasks=args.balanced_tasks,
        min_shared_prefix_groups=args.min_shared_prefix_groups,
        require_tenant_fields=args.require_tenant_fields,
        require_source_fields=args.require_source_fields,
        require_public_domain_license=args.require_public_domain_license,
    )
    print(result)


if __name__ == "__main__":
    main()
