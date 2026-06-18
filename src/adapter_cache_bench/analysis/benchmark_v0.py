from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries, with_strategy_columns

BENCHMARK_V0_WORKLOADS = [
    "shared_doc_qa",
    "mixed_tasks_same_doc",
    "prompt_layout_ablation",
    "low_overlap_control",
]
BENCHMARK_V0_ROUTERS = [
    "semantic",
    "multitask",
    "sticky_session",
    "cache_aware",
    "oracle",
]
BENCHMARK_V0_CACHES = [
    "standard_lora",
    "activated_lora",
    "copy_on_write",
]
BENCHMARK_V0_SEEDS = [17, 23, 31]

BENCHMARK_V0_COLUMNS = [
    "benchmark_suite",
    "workload",
    "router_policy",
    "cache_model",
    "adapter_strategy",
    "seed",
    "run_id",
    "request_count",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_e2e_ms",
    "p95_e2e_ms",
    "slo_attainment_rate",
    "request_throughput",
    "token_throughput",
    "mean_quality",
    "quality_adjusted_goodput",
    "quality_adjusted_goodput_per_memory_token",
    "cache_hit_rate",
    "cached_prompt_token_ratio",
    "fragmentation_index",
    "memory_token_footprint",
    "eviction_count",
    "evicted_tokens",
]

_RUN_SUFFIX = re.compile(r"-seed(?P<seed>\d+)-(?P<timestamp>\d+)$")


def _run_timestamp(run_id: object) -> int:
    match = _RUN_SUFFIX.search(str(run_id))
    return int(match.group("timestamp")) if match else -1


def _run_seed(run_id: object) -> int | None:
    match = _RUN_SUFFIX.search(str(run_id))
    return int(match.group("seed")) if match else None


def benchmark_v0_summary(df: pd.DataFrame, *, require_complete: bool = True) -> pd.DataFrame:
    if df.empty:
        if require_complete:
            raise ValueError("no benchmark summaries found")
        return pd.DataFrame(columns=BENCHMARK_V0_COLUMNS)

    rows = with_strategy_columns(df).copy()
    rows["seed"] = rows["run_id"].map(_run_seed)
    rows["run_timestamp"] = rows["run_id"].map(_run_timestamp)
    rows = rows[
        rows["workload"].isin(BENCHMARK_V0_WORKLOADS)
        & rows["router_policy"].isin(BENCHMARK_V0_ROUTERS)
        & rows["cache_model"].isin(BENCHMARK_V0_CACHES)
        & rows["seed"].isin(BENCHMARK_V0_SEEDS)
        & rows["request_count"].eq(96)
        & rows["backend_kind"].eq("mock")
    ].copy()
    rows["seed"] = rows["seed"].astype(int)
    rows = rows.sort_values("run_timestamp").drop_duplicates(
        ["workload", "router_policy", "cache_model", "seed"],
        keep="last",
    )

    expected = set(
        itertools.product(
            BENCHMARK_V0_WORKLOADS,
            BENCHMARK_V0_ROUTERS,
            BENCHMARK_V0_CACHES,
            BENCHMARK_V0_SEEDS,
        )
    )
    observed = set(
        rows[["workload", "router_policy", "cache_model", "seed"]].itertuples(
            index=False,
            name=None,
        )
    )
    missing = sorted(expected - observed)
    if missing and require_complete:
        formatted = ", ".join(
            f"{workload}/{router}/{cache}/seed{seed}"
            for workload, router, cache, seed in missing[:10]
        )
        suffix = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise ValueError(f"benchmark_v0 is incomplete: missing {formatted}{suffix}")

    rows["benchmark_suite"] = "benchmark_v0_mock"
    rows = rows.sort_values(["workload", "router_policy", "cache_model", "seed"])
    return rows[BENCHMARK_V0_COLUMNS].reset_index(drop=True)


def write_benchmark_v0_csv(
    runs_dir: str | Path = "artifacts/runs",
    output_csv: str | Path = "data/results/benchmark_v0_mock.csv",
    *,
    require_complete: bool = True,
) -> Path:
    table = benchmark_v0_summary(load_summaries(runs_dir), require_complete=require_complete)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output-csv", default="data/results/benchmark_v0_mock.csv")
    args = parser.parse_args()
    print(write_benchmark_v0_csv(args.runs_dir, args.output_csv))


if __name__ == "__main__":
    main()
