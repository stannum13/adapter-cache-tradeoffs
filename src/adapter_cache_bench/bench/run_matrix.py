from __future__ import annotations

import argparse
import copy
import itertools

from adapter_cache_bench.bench.run_workload import run
from adapter_cache_bench.bench.sweep_state import (
    SweepChild,
    add_sweep_arguments,
    execute_sweep,
    options_from_args,
    record_sweep_dimensions,
)
from adapter_cache_bench.config import BenchmarkConfig, load_config


def expand_matrix_sweep(config: BenchmarkConfig) -> list[SweepChild]:
    matrix = config.matrix or {}
    routers = matrix.get("routers", [config.router.policy])
    caches = matrix.get("caches", [config.cache.model])
    workloads = matrix.get("workloads", [config.workload.name])
    seeds = [int(seed) for seed in matrix.get("seeds", [config.workload.seed])]
    children = []
    for router, cache, workload, seed in itertools.product(routers, caches, workloads, seeds):
        child = copy.deepcopy(config)
        child.router.policy = str(router)
        child.cache.model = str(cache)
        child.workload.name = str(workload)
        child.workload.seed = seed
        child.router.seed = seed
        child.backend.seed = seed
        child.run_name = f"{workload}-{router}-{cache}-seed{seed}"
        child.matrix = {}
        dimensions = {
            "router": str(router),
            "cache": str(cache),
            "workload": str(workload),
            "seed": seed,
        }
        children.append(SweepChild(child, dimensions))
    return children


def expand_matrix(config: BenchmarkConfig) -> list[BenchmarkConfig]:
    return [child.config for child in expand_matrix_sweep(config)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    add_sweep_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    children = expand_matrix_sweep(config)

    from adapter_cache_bench.analysis.report import generate_report

    execute_sweep(
        config=config,
        sweep_name=args.sweep_name or f"{config.run_name}-matrix",
        children=children,
        run_child=lambda child_config, run_id: run(
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
