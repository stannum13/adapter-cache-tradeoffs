from __future__ import annotations

import argparse
import copy
import itertools

from adapter_cache_bench.bench.run_workload import run
from adapter_cache_bench.config import BenchmarkConfig, load_config


def expand_matrix(config: BenchmarkConfig) -> list[BenchmarkConfig]:
    matrix = config.matrix or {}
    routers = matrix.get("routers", [config.router.policy])
    caches = matrix.get("caches", [config.cache.model])
    workloads = matrix.get("workloads", [config.workload.name])
    seeds = [int(seed) for seed in matrix.get("seeds", [config.workload.seed])]
    configs = []
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
        configs.append(child)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument("--report-path", default="reports/adapter-cache-tradeoffs.md")
    parser.add_argument("--tables-dir", default="reports/tables")
    args = parser.parse_args()
    config = load_config(args.config)
    for child in expand_matrix(config):
        print(run(child, generate_report_artifacts=False))

    from adapter_cache_bench.analysis.report import generate_report

    generate_report(config.output_dir, report_path=args.report_path, tables_dir=args.tables_dir)


if __name__ == "__main__":
    main()
