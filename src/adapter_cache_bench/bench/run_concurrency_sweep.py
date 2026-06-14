from __future__ import annotations

import argparse
import copy

from adapter_cache_bench.analysis.report import generate_report
from adapter_cache_bench.bench.run_concurrent import run_concurrent
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
    raise ValueError(f"Unknown sweep strategy: {strategy}")


def expand_concurrency_sweep(config: BenchmarkConfig) -> list[BenchmarkConfig]:
    matrix = config.matrix or {}
    strategies = [str(item) for item in matrix.get("strategies", ["base", "specialists"])]
    concurrencies = [
        int(item) for item in matrix.get("concurrencies", [config.backend.max_concurrency])
    ]
    seeds = [int(item) for item in matrix.get("seeds", [config.workload.seed])]
    children = []
    for strategy in strategies:
        for concurrency in concurrencies:
            for seed in seeds:
                child = apply_strategy(config, strategy)
                child.backend.max_concurrency = concurrency
                child.workload.seed = seed
                child.router.seed = seed
                child.backend.seed = seed
                child.run_name = f"{child.run_name}-c{concurrency}-seed{seed}"
                child.matrix = {}
                children.append(child)
    return children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--report-path", default="reports/adapter-cache-bench.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    args = parser.parse_args()
    config = load_config(args.config)
    for child in expand_concurrency_sweep(config):
        print(run_concurrent(child, generate_report_artifacts=False))
    generate_report(config.output_dir, report_path=args.report_path, tables_dir=args.tables_dir)


if __name__ == "__main__":
    main()
