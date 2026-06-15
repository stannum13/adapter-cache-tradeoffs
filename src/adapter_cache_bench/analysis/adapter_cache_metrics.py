from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_request_rows, load_summaries

ADAPTER_CACHE_COLUMNS = [
    "run_id",
    "backend_kind",
    "backend_model",
    "workload",
    "router_policy",
    "cache_model",
    "adapter_id",
    "request_count",
    "prompt_tokens",
    "cached_prompt_tokens",
    "benchmark_cached_prompt_ratio",
    "mean_quality",
    "mean_ttft_ms",
    "server_prefix_cache_queries",
    "server_prefix_cache_hits",
    "server_prefix_cache_hit_rate",
    "server_prompt_tokens_cached",
    "server_cache_metric_scope",
]


def _server_metric_scope(runs_dir: str | Path) -> dict[str, str]:
    scopes: dict[str, str] = {}
    for manifest_path in Path(runs_dir).glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = set(manifest.get("artifact_files", []))
        run_id = str(manifest.get("run_id", manifest_path.parent.name))
        if "server_reset.log" in artifacts:
            scopes[run_id] = "per_condition_reset"
        elif any(name.startswith("backend_metrics_") for name in artifacts):
            scopes[run_id] = "server_process_window"
        else:
            scopes[run_id] = "unavailable"
    return scopes


def build_adapter_cache_metrics(runs_dir: str | Path = "artifacts/runs") -> pd.DataFrame:
    requests = load_request_rows(runs_dir)
    if requests.empty:
        return pd.DataFrame(columns=ADAPTER_CACHE_COLUMNS)

    grouped = (
        requests.groupby(
            [
                "run_id",
                "backend_kind",
                "backend_model",
                "workload",
                "router_policy",
                "cache_model",
                "adapter_id",
            ],
            as_index=False,
        )
        .agg(
            request_count=("request_id", "count"),
            prompt_tokens=("prompt_tokens", "sum"),
            cached_prompt_tokens=("cached_prompt_tokens", "sum"),
            mean_quality=("quality", "mean"),
            mean_ttft_ms=("ttft_ms", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["benchmark_cached_prompt_ratio"] = (
        grouped["cached_prompt_tokens"] / grouped["prompt_tokens"].clip(lower=1)
    )

    summaries = load_summaries(runs_dir)
    server_columns = [
        "run_id",
        "server_prefix_cache_queries",
        "server_prefix_cache_hits",
        "server_prefix_cache_hit_rate",
        "server_prompt_tokens_cached",
    ]
    if summaries.empty:
        for column in server_columns[1:]:
            grouped[column] = 0.0
    else:
        grouped = grouped.merge(
            summaries[[column for column in server_columns if column in summaries.columns]],
            on="run_id",
            how="left",
        )
        for column in server_columns[1:]:
            if column not in grouped:
                grouped[column] = 0.0
            grouped[column] = grouped[column].fillna(0.0)

    scopes = _server_metric_scope(runs_dir)
    grouped["server_cache_metric_scope"] = grouped["run_id"].map(scopes).fillna("unavailable")
    return grouped[ADAPTER_CACHE_COLUMNS].sort_values(
        ["run_id", "adapter_id"], kind="stable"
    )


def write_adapter_cache_metrics(
    runs_dir: str | Path = "artifacts/runs",
    output: str | Path = "reports/tables/adapter_cache_metrics.csv",
) -> Path:
    table = build_adapter_cache_metrics(runs_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="reports/tables/adapter_cache_metrics.csv")
    args = parser.parse_args()
    print(write_adapter_cache_metrics(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
