from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries, with_strategy_columns


def pareto_frontier(
    df: pd.DataFrame, quality_col: str = "mean_quality", latency_col: str = "p95_ttft_ms"
) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values([latency_col, quality_col], ascending=[True, False])
    rows = []
    best_quality = -1.0
    for _, row in ordered.iterrows():
        if row[quality_col] > best_quality:
            rows.append(row)
            best_quality = row[quality_col]
    return pd.DataFrame(rows)


def workload_pareto_frontiers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = with_strategy_columns(df)
    frontiers = []
    for workload, group in enriched.groupby("workload"):
        frontier = pareto_frontier(group).copy()
        frontier["pareto_workload"] = workload
        frontiers.append(frontier)
    if not frontiers:
        return pd.DataFrame()
    return pd.concat(frontiers, ignore_index=True)


def write_pareto_frontier(
    runs_dir: str | Path = "artifacts/runs",
    output_csv: str | Path = "reports/tables/pareto_frontier.csv",
) -> Path:
    df = workload_pareto_frontiers(load_summaries(runs_dir))
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output-csv", default="reports/tables/pareto_frontier.csv")
    args = parser.parse_args()
    print(write_pareto_frontier(args.runs_dir, args.output_csv))


if __name__ == "__main__":
    main()
