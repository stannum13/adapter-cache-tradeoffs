from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_request_rows

DEFAULT_TTFT_THRESHOLDS_MS = [25.0, 50.0, 100.0, 150.0, 250.0]


def slo_sweep(
    request_df: pd.DataFrame,
    thresholds_ms: list[float] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds_ms or DEFAULT_TTFT_THRESHOLDS_MS
    if request_df.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["run_id", "workload", "router_policy", "cache_model"]
    for group_key, group in request_df.groupby(group_columns):
        run_id, workload, router_policy, cache_model = group_key
        duration_s = max(0.001, group["e2e_ms"].sum() / 1000.0)
        mean_quality = group["quality"].mean()
        for threshold in thresholds:
            passing = group[group["ttft_ms"] <= threshold]
            goodput = len(passing) / duration_s
            quality_adjusted_goodput = passing["quality"].sum() / duration_s
            rows.append(
                {
                    "run_id": run_id,
                    "workload": workload,
                    "router_policy": router_policy,
                    "cache_model": cache_model,
                    "ttft_slo_ms": threshold,
                    "requests_under_slo": len(passing),
                    "goodput_under_slo": goodput,
                    "quality_adjusted_goodput": quality_adjusted_goodput,
                    "mean_quality": mean_quality,
                }
            )
    return pd.DataFrame(rows)


def write_slo_sweep(
    runs_dir: str | Path = "artifacts/runs",
    output_csv: str | Path = "reports/tables/slo_sweep.csv",
    thresholds_ms: list[float] | None = None,
) -> Path:
    table = slo_sweep(load_request_rows(runs_dir), thresholds_ms)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output-csv", default="reports/tables/slo_sweep.csv")
    parser.add_argument("--threshold-ms", type=float, nargs="*", default=None)
    args = parser.parse_args()
    print(write_slo_sweep(args.runs_dir, args.output_csv, args.threshold_ms))


if __name__ == "__main__":
    main()
