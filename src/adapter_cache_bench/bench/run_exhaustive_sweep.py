from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
from pathlib import Path
from typing import Any

from adapter_cache_bench.analysis.report import generate_report
from adapter_cache_bench.bench.run_concurrency_sweep import apply_strategy
from adapter_cache_bench.bench.run_concurrent import run_concurrent
from adapter_cache_bench.config import BenchmarkConfig, load_config

ADAPTER_MODEL_NAMES = {
    "qa": "qa-lora",
    "json": "json-lora",
    "summary": "summary-lora",
    "code": "code-lora",
    "multitask": "multitask-lora",
}

ADAPTER_IDS_BY_COUNT = {
    1: ["qa"],
    2: ["qa", "json"],
    3: ["qa", "json", "summary"],
    4: ["qa", "json", "summary", "code"],
    5: ["qa", "json", "summary", "code", "multitask"],
}


def _matrix_values(matrix: dict[str, list[str | int | float]], key: str, default: Any) -> list[Any]:
    return list(matrix.get(key, [default]))


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p")
    return re.sub(r"[^a-zA-Z0-9_+-]+", "-", text).strip("-")


def apply_adapter_count(config: BenchmarkConfig, strategy: str, adapter_count: int) -> None:
    if strategy == "multitask":
        config.adapters.adapter_ids = ["multitask"]
        config.adapters.default_adapter = "multitask"
        config.backend.adapter_model_names = {"multitask": ADAPTER_MODEL_NAMES["multitask"]}
        return
    if strategy == "base":
        return
    adapter_ids = ADAPTER_IDS_BY_COUNT.get(adapter_count)
    if adapter_ids is None:
        raise ValueError(f"Unsupported adapter_count={adapter_count}; use 1..5")
    config.adapters.adapter_ids = adapter_ids
    config.backend.adapter_model_names = {
        adapter_id: ADAPTER_MODEL_NAMES[adapter_id] for adapter_id in adapter_ids
    }


def record_sweep_dimensions(run_dir: Path, dimensions: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sweep_dimensions"] = dimensions
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def expand_exhaustive_sweep(
    config: BenchmarkConfig,
) -> list[tuple[BenchmarkConfig, dict[str, Any]]]:
    matrix = config.matrix or {}
    strategies = [str(item) for item in _matrix_values(matrix, "strategies", "specialists")]
    concurrencies = [
        int(item)
        for item in _matrix_values(matrix, "concurrencies", config.backend.max_concurrency)
    ]
    workloads = [str(item) for item in _matrix_values(matrix, "workloads", config.workload.name)]
    caches = [str(item) for item in _matrix_values(matrix, "caches", config.cache.model)]
    seeds = [int(item) for item in _matrix_values(matrix, "seeds", config.workload.seed)]
    overlap_fractions = [
        float(item)
        for item in _matrix_values(
            matrix,
            "overlap_fractions",
            config.workload.shared_prefix_fraction,
        )
    ]
    adapter_counts = [int(item) for item in _matrix_values(matrix, "adapter_counts", 4)]
    tenants = [int(item) for item in _matrix_values(matrix, "tenants", config.workload.tenants)]
    isolation_scopes = [
        str(item)
        for item in _matrix_values(matrix, "isolation_scopes", config.cache.isolation_scope)
    ]

    children: list[tuple[BenchmarkConfig, dict[str, Any]]] = []
    for (
        strategy,
        concurrency,
        workload,
        cache,
        seed,
        overlap_fraction,
        adapter_count,
        tenant_count,
        isolation_scope,
    ) in itertools.product(
        strategies,
        concurrencies,
        workloads,
        caches,
        seeds,
        overlap_fractions,
        adapter_counts,
        tenants,
        isolation_scopes,
    ):
        child = apply_strategy(copy.deepcopy(config), strategy)
        child.backend.max_concurrency = concurrency
        child.workload.name = workload
        child.cache.model = cache
        child.workload.seed = seed
        child.router.seed = seed
        child.backend.seed = seed
        child.workload.shared_prefix_fraction = overlap_fraction
        child.workload.tenants = tenant_count
        child.cache.isolation_scope = isolation_scope
        apply_adapter_count(child, strategy, adapter_count)
        dimensions = {
            "strategy": strategy,
            "concurrency": concurrency,
            "workload": workload,
            "cache": cache,
            "seed": seed,
            "overlap_fraction": overlap_fraction,
            "adapter_count": adapter_count
            if strategy == "specialists"
            else len(child.adapters.adapter_ids),
            "tenants": tenant_count,
            "isolation_scope": isolation_scope,
        }
        child.run_name = (
            f"{config.run_name}-{_slug(strategy)}-c{concurrency}-{_slug(workload)}"
            f"-{_slug(cache)}-ov{_slug(overlap_fraction)}-a{dimensions['adapter_count']}"
            f"-t{tenant_count}-{_slug(isolation_scope)}-seed{seed}"
        )
        child.matrix = {}
        children.append((child, dimensions))
    return children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--report-path", default="reports/adapter-cache-bench.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    args = parser.parse_args()
    config = load_config(args.config)
    for child, dimensions in expand_exhaustive_sweep(config):
        run_dir = run_concurrent(child, generate_report_artifacts=False)
        record_sweep_dimensions(run_dir, dimensions)
        print(run_dir)
    generate_report(config.output_dir, report_path=args.report_path, tables_dir=args.tables_dir)


if __name__ == "__main__":
    main()
