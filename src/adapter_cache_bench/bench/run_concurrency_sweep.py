from __future__ import annotations

import argparse
import copy

from adapter_cache_bench.analysis.report import generate_report
from adapter_cache_bench.bench.run_concurrent import run_concurrent
from adapter_cache_bench.bench.sweep_state import (
    SweepChild,
    add_sweep_arguments,
    execute_sweep,
    options_from_args,
    record_sweep_dimensions,
)
from adapter_cache_bench.config import BenchmarkConfig, load_config

SPECIALIST_MODEL_NAMES = {
    "qa": "qa-lora",
    "json": "json-lora",
    "summary": "summary-lora",
    "code": "code-lora",
}


def apply_strategy(config: BenchmarkConfig, strategy: str) -> BenchmarkConfig:
    child = copy.deepcopy(config)
    if strategy == "base":
        child.router.policy = "cache_aware"
        child.backend.adapter_model_names = {}
        child.run_name = f"{child.run_name}-base"
        return child
    if strategy == "semantic":
        child.router.policy = "semantic"
        child.backend.adapter_model_names = dict(SPECIALIST_MODEL_NAMES)
        child.run_name = f"{child.run_name}-semantic"
        return child
    if strategy == "sticky_session":
        child.router.policy = "sticky_session"
        child.backend.adapter_model_names = dict(SPECIALIST_MODEL_NAMES)
        child.run_name = f"{child.run_name}-sticky-session"
        return child
    if strategy == "cache_static":
        child.router.policy = "cache_aware"
        child.router.beta = 0.0
        child.backend.adapter_model_names = dict(SPECIALIST_MODEL_NAMES)
        child.run_name = f"{child.run_name}-cache-static"
        return child
    if strategy == "specialists":
        child.router.policy = "cache_aware"
        child.backend.adapter_model_names = dict(SPECIALIST_MODEL_NAMES)
        child.run_name = f"{child.run_name}-specialists"
        return child
    if strategy == "multitask":
        child.router.policy = "multitask"
        child.adapters.adapter_ids = ["multitask"]
        child.adapters.default_adapter = "multitask"
        child.backend.adapter_model_names = {"multitask": "multitask-lora"}
        child.run_name = f"{child.run_name}-multitask"
        return child
    if strategy == "oracle":
        child.router.policy = "oracle"
        child.backend.adapter_model_names = dict(SPECIALIST_MODEL_NAMES)
        child.run_name = f"{child.run_name}-oracle"
        return child
    raise ValueError(f"Unknown sweep strategy: {strategy}")


def expand_concurrency_sweep_children(config: BenchmarkConfig) -> list[SweepChild]:
    matrix = config.matrix or {}
    strategies = [str(item) for item in matrix.get("strategies", ["base", "specialists"])]
    concurrencies = [
        int(item) for item in matrix.get("concurrencies", [config.backend.max_concurrency])
    ]
    cache_conditions = [
        str(item) for item in matrix.get("cache_conditions", [config.cache.condition])
    ]
    include_cache_condition = "cache_conditions" in matrix or config.cache.condition != "warm"
    seeds = [int(item) for item in matrix.get("seeds", [config.workload.seed])]
    children = []
    for strategy in strategies:
        for concurrency in concurrencies:
            for cache_condition in cache_conditions:
                for seed in seeds:
                    child = apply_strategy(config, strategy)
                    child.backend.max_concurrency = concurrency
                    child.cache.condition = cache_condition
                    child.workload.seed = seed
                    child.router.seed = seed
                    child.backend.seed = seed
                    condition_label = f"-{cache_condition}" if include_cache_condition else ""
                    child.run_name = f"{child.run_name}-c{concurrency}{condition_label}-seed{seed}"
                    child.matrix = {}
                    dimensions = {
                        "strategy": strategy,
                        "concurrency": concurrency,
                        "seed": seed,
                    }
                    if include_cache_condition:
                        dimensions["cache_condition"] = cache_condition
                    children.append(SweepChild(child, dimensions))
    return children


def expand_concurrency_sweep(config: BenchmarkConfig) -> list[BenchmarkConfig]:
    return [child.config for child in expand_concurrency_sweep_children(config)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    add_sweep_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    execute_sweep(
        config=config,
        sweep_name=args.sweep_name or f"{config.run_name}-concurrency",
        children=expand_concurrency_sweep_children(config),
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
