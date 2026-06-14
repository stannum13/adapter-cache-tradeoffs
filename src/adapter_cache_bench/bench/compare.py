from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapter_cache_bench.bench.aggregate import (
    cache_model_means,
    load_summaries,
    router_means,
    workload_leaders,
)


def compare_runs(runs_dir: str | Path = "artifacts/runs") -> dict[str, pd.DataFrame]:
    summaries = load_summaries(runs_dir)
    return {
        "workload_leaders": workload_leaders(summaries),
        "cache_model_means": cache_model_means(summaries),
        "router_means": router_means(summaries),
    }


def _format_table(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    if df.empty:
        return "No rows."
    visible = df[columns].head(limit).copy()
    for column in visible.select_dtypes(include=["float"]).columns:
        visible[column] = visible[column].map(lambda value: f"{value:.3f}")
    return visible.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    tables = compare_runs(args.runs_dir)
    leaders = tables["workload_leaders"]
    print("Workload leaders")
    print(
        _format_table(
            leaders,
            [
                "workload",
                "router_policy",
                "cache_model",
                "quality_adjusted_goodput",
                "mean_quality",
                "p95_ttft_ms",
            ],
            args.limit,
        )
    )
    print()
    print("Cache model means")
    print(
        _format_table(
            tables["cache_model_means"],
            [
                "cache_model",
                "quality_adjusted_goodput",
                "p95_ttft_ms",
                "cache_hit_rate",
                "fragmentation_index",
            ],
            args.limit,
        )
    )
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        leaders.to_csv(out, index=False)


if __name__ == "__main__":
    main()
