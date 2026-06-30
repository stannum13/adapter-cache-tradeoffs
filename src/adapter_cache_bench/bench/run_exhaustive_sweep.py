from __future__ import annotations

import argparse
import copy
import itertools
import re
from typing import Any

from adapter_cache_bench.analysis.report import generate_report
from adapter_cache_bench.bench.run_concurrency_sweep import apply_strategy
from adapter_cache_bench.bench.run_concurrent import run_concurrent
from adapter_cache_bench.bench.sweep_state import (
    SweepChild,
    add_sweep_arguments,
    execute_sweep,
    options_from_args,
    record_sweep_dimensions,
)
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


def _matrix_values(matrix: dict[str, Any], key: str, default: Any) -> list[Any]:
    return list(matrix.get(key, [default]))


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p")
    return re.sub(r"[^a-zA-Z0-9_+-]+", "-", text).strip("-")


def _model_alias(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _model_specs(config: BenchmarkConfig) -> list[dict[str, Any]]:
    raw_specs = _matrix_values(
        config.matrix or {},
        "models",
        {"name": config.backend.model, "alias": _model_alias(config.backend.model)},
    )
    specs = []
    for raw in raw_specs:
        if isinstance(raw, str):
            specs.append({"name": raw, "alias": _model_alias(raw)})
        elif isinstance(raw, dict):
            if "name" not in raw:
                raise ValueError(f"Model sweep entry is missing name: {raw}")
            specs.append({"alias": _model_alias(str(raw["name"])), **raw})
        else:
            raise TypeError(f"Unsupported model sweep entry: {raw!r}")
    return specs


def apply_adapter_count(
    config: BenchmarkConfig,
    strategy: str,
    adapter_count: int,
    adapter_model_names: dict[str, str],
) -> None:
    if strategy == "multitask":
        config.adapters.adapter_ids = ["multitask"]
        config.adapters.default_adapter = "multitask"
        config.backend.adapter_model_names = {"multitask": adapter_model_names["multitask"]}
        return
    if strategy == "base":
        return
    adapter_ids = ADAPTER_IDS_BY_COUNT.get(adapter_count)
    if adapter_ids is None:
        raise ValueError(f"Unsupported adapter_count={adapter_count}; use 1..5")
    config.adapters.adapter_ids = adapter_ids
    config.backend.adapter_model_names = {
        adapter_id: adapter_model_names[adapter_id] for adapter_id in adapter_ids
    }


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
    model_specs = _model_specs(config)

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
        model_spec,
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
        model_specs,
    ):
        child = apply_strategy(copy.deepcopy(config), strategy)
        child.backend.model = str(model_spec["name"])
        adapter_model_names = {
            **ADAPTER_MODEL_NAMES,
            **dict(model_spec.get("adapter_model_names") or {}),
        }
        child.backend.max_concurrency = concurrency
        child.workload.name = workload
        child.cache.model = cache
        child.workload.seed = seed
        child.router.seed = seed
        child.backend.seed = seed
        child.workload.shared_prefix_fraction = overlap_fraction
        child.workload.tenants = tenant_count
        child.cache.isolation_scope = isolation_scope
        apply_adapter_count(child, strategy, adapter_count, adapter_model_names)
        dimensions = {
            "strategy": strategy,
            "concurrency": concurrency,
            "model": child.backend.model,
            "model_alias": str(model_spec["alias"]),
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
            f"{config.run_name}-{_slug(strategy)}-{_slug(model_spec['alias'])}"
            f"-c{concurrency}-{_slug(workload)}"
            f"-{_slug(cache)}-ov{_slug(overlap_fraction)}-a{dimensions['adapter_count']}"
            f"-t{tenant_count}-{_slug(isolation_scope)}-seed{seed}"
        )
        child.matrix = {}
        children.append((child, dimensions))
    return children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    add_sweep_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    children = [
        SweepChild(child, dimensions) for child, dimensions in expand_exhaustive_sweep(config)
    ]
    execute_sweep(
        config=config,
        sweep_name=args.sweep_name or f"{config.run_name}-exhaustive",
        children=children,
        run_child=lambda child_config, run_id: run_concurrent(
            child_config,
            run_id=run_id,
            generate_report_artifacts=False,
        ),
        record_dimensions=record_sweep_dimensions,
        options=options_from_args(args),
        on_complete=lambda: generate_report(
            config.output_dir,
            report_path=args.report_path,
            tables_dir=args.tables_dir,
        ),
    )


if __name__ == "__main__":
    main()
