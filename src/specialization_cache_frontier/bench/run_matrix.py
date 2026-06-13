from __future__ import annotations

import argparse
import copy
import itertools

from specialization_cache_frontier.bench.run_workload import run
from specialization_cache_frontier.config import BenchmarkConfig, load_config


def expand_matrix(config: BenchmarkConfig) -> list[BenchmarkConfig]:
    matrix = config.matrix or {}
    routers = matrix.get("routers", [config.router.policy])
    caches = matrix.get("caches", [config.cache.model])
    workloads = matrix.get("workloads", [config.workload.name])
    configs = []
    for router, cache, workload in itertools.product(routers, caches, workloads):
        child = copy.deepcopy(config)
        child.router.policy = router
        child.cache.model = cache
        child.workload.name = workload
        child.run_name = f"{workload}-{router}-{cache}"
        child.matrix = {}
        configs.append(child)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for child in expand_matrix(config):
        print(run(child))


if __name__ == "__main__":
    main()
