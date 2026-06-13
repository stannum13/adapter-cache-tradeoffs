from __future__ import annotations

import pandas as pd


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
