from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries

MODEL_FAMILY_COLUMNS = [
    "model_alias",
    "strategy",
    "backend_model",
    "runs",
    "requests",
    "mean_quality",
    "mean_quality_std",
    "p95_ttft_ms",
    "p95_ttft_ms_std",
    "quality_adjusted_goodput",
    "quality_adjusted_goodput_std",
    "slo_attainment_rate",
    "server_prefix_cache_hit_rate",
]


def build_model_family_summary(runs_dir: str | Path = "artifacts/runs") -> pd.DataFrame:
    summaries = load_summaries(runs_dir)
    if summaries.empty or "run_id" not in summaries:
        return pd.DataFrame(columns=MODEL_FAMILY_COLUMNS)

    rows = summaries[summaries["run_id"].str.contains("model-family-vllm", na=False)].copy()
    required_columns = {
        "sweep_model_alias",
        "sweep_strategy",
        "backend_model",
        "request_count",
        "mean_quality",
        "p95_ttft_ms",
        "quality_adjusted_goodput",
        "slo_attainment_rate",
        "server_prefix_cache_hit_rate",
    }
    if rows.empty or not required_columns <= set(rows.columns):
        return pd.DataFrame(columns=MODEL_FAMILY_COLUMNS)

    grouped = (
        rows.groupby(["sweep_model_alias", "sweep_strategy", "backend_model"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            requests=("request_count", "sum"),
            mean_quality=("mean_quality", "mean"),
            mean_quality_std=("mean_quality", "std"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            p95_ttft_ms_std=("p95_ttft_ms", "std"),
            quality_adjusted_goodput=("quality_adjusted_goodput", "mean"),
            quality_adjusted_goodput_std=("quality_adjusted_goodput", "std"),
            slo_attainment_rate=("slo_attainment_rate", "mean"),
            server_prefix_cache_hit_rate=("server_prefix_cache_hit_rate", "mean"),
        )
        .rename(
            columns={
                "sweep_model_alias": "model_alias",
                "sweep_strategy": "strategy",
            }
        )
        .fillna(0.0)
        .sort_values(["model_alias", "strategy"], kind="stable")
        .reset_index(drop=True)
    )
    return grouped[MODEL_FAMILY_COLUMNS]


def write_model_family_summary(
    runs_dir: str | Path = "artifacts/runs",
    output: str | Path = "reports/tables/model_family_summary.csv",
) -> Path:
    table = build_model_family_summary(runs_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="reports/tables/model_family_summary.csv")
    args = parser.parse_args()
    print(write_model_family_summary(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
