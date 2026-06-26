from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries

METRICS = [
    "mean_quality",
    "p95_ttft_ms",
    "slo_attainment_rate",
    "qag",
    "server_prefix_cache_hit_rate",
]

CLAIM_EVIDENCE_COLUMNS = [
    "row_type",
    "claim_group",
    "model_alias",
    "strategy",
    "runs",
    "requests",
    "mean_quality",
    "mean_quality_ci95_low",
    "mean_quality_ci95_high",
    "p95_ttft_ms",
    "p95_ttft_ms_ci95_low",
    "p95_ttft_ms_ci95_high",
    "slo_attainment_rate",
    "slo_attainment_rate_ci95_low",
    "slo_attainment_rate_ci95_high",
    "qag",
    "qag_ci95_low",
    "qag_ci95_high",
    "server_prefix_cache_hit_rate",
    "server_prefix_cache_hit_rate_ci95_low",
    "server_prefix_cache_hit_rate_ci95_high",
    "paired_seed_count",
    "paired_baseline_strategy",
    "paired_comparison_strategy",
]

_SUMMARY_COLUMN_BY_METRIC = {"qag": "quality_adjusted_goodput"}
_NON_COMPARABLE_SWEEP_KEYS = {
    "adapter_count",
    "model",
    "model_alias",
    "seed",
    "strategy",
}


def _t_critical_95(sample_count: int) -> float:
    by_degrees_of_freedom = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if sample_count <= 1:
        return 0.0
    return by_degrees_of_freedom.get(sample_count - 1, 1.96)


def _ci95(values: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    sample_count = len(numeric)
    if sample_count <= 1:
        return None, None
    mean = float(numeric.mean())
    std = float(numeric.std())
    half_width = _t_critical_95(sample_count) * std / (sample_count**0.5)
    return mean - half_width, mean + half_width


def _last_path_part(value: Any) -> str:
    text = "" if value is None else str(value)
    if not text or text == "nan":
        return "unknown"
    return text.rstrip("/").rsplit("/", 1)[-1]


def _first_present(row: pd.Series, columns: list[str], default: Any = None) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]) and row[column] != "":
            return row[column]
    return default


def _infer_strategy(row: pd.Series) -> str:
    strategy = _first_present(row, ["sweep_strategy"])
    if strategy is not None:
        return str(strategy)
    router_policy = str(_first_present(row, ["router_policy"], "unknown"))
    if router_policy == "multitask":
        return "multitask"
    return str(_first_present(row, ["cache_model"], router_policy))


def _format_dimension_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _dimension_columns(df: pd.DataFrame, *, comparable_only: bool) -> list[str]:
    sweep_columns = [column for column in df.columns if column.startswith("sweep_")]
    excluded = (
        _NON_COMPARABLE_SWEEP_KEYS
        if comparable_only
        else {"model", "model_alias", "seed", "strategy"}
    )
    return sorted(
        column for column in sweep_columns if column.removeprefix("sweep_") not in excluded
    )


def _claim_group(row: pd.Series, dimension_columns: list[str]) -> str:
    parts = []
    for column in dimension_columns:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{column.removeprefix('sweep_')}={_format_dimension_value(value)}")
    if parts:
        return "|".join(parts)
    workload = _first_present(row, ["workload", "sweep_workload"], "unknown")
    cache_model = _first_present(row, ["cache_model", "sweep_cache"], "unknown")
    return f"workload={workload}|cache={cache_model}"


def _normalize_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*CLAIM_EVIDENCE_COLUMNS, "seed"])
    normalized = df.copy()
    for metric, source_column in _SUMMARY_COLUMN_BY_METRIC.items():
        if source_column in normalized:
            normalized[metric] = pd.to_numeric(normalized[source_column], errors="coerce")
        elif metric not in normalized:
            normalized[metric] = pd.NA
    for metric in METRICS:
        if metric not in normalized:
            normalized[metric] = pd.NA
        normalized[metric] = pd.to_numeric(normalized[metric], errors="coerce")
    if "request_count" not in normalized:
        normalized["request_count"] = 0
    normalized["request_count"] = pd.to_numeric(
        normalized["request_count"], errors="coerce"
    ).fillna(0)
    normalized["model_alias"] = normalized.apply(
        lambda row: str(
            _first_present(
                row,
                ["sweep_model_alias"],
                _last_path_part(_first_present(row, ["backend_model", "sweep_model"], "unknown")),
            )
        ),
        axis=1,
    )
    normalized["strategy"] = normalized.apply(_infer_strategy, axis=1)
    normalized["seed"] = normalized.apply(
        lambda row: _first_present(row, ["sweep_seed"], pd.NA),
        axis=1,
    )
    evidence_dimensions = _dimension_columns(normalized, comparable_only=False)
    paired_dimensions = _dimension_columns(normalized, comparable_only=True)
    normalized["claim_group"] = normalized.apply(
        lambda row: _claim_group(row, evidence_dimensions),
        axis=1,
    )
    normalized["paired_claim_group"] = normalized.apply(
        lambda row: _claim_group(row, paired_dimensions),
        axis=1,
    )
    return normalized


def _empty_claim_table() -> pd.DataFrame:
    return pd.DataFrame(columns=CLAIM_EVIDENCE_COLUMNS)


def _aggregate_row(group_key: tuple[str, str, str], group: pd.DataFrame) -> dict[str, Any]:
    claim_group, model_alias, strategy = group_key
    row: dict[str, Any] = {
        "row_type": "evidence",
        "claim_group": claim_group,
        "model_alias": model_alias,
        "strategy": strategy,
        "runs": int(len(group)),
        "requests": int(group["request_count"].sum()),
        "paired_seed_count": pd.NA,
        "paired_baseline_strategy": "",
        "paired_comparison_strategy": "",
    }
    for metric in METRICS:
        values = group[metric]
        row[metric] = float(values.mean()) if values.notna().any() else pd.NA
        ci_low, ci_high = _ci95(values)
        row[f"{metric}_ci95_low"] = ci_low if ci_low is not None else pd.NA
        row[f"{metric}_ci95_high"] = ci_high if ci_high is not None else pd.NA
    return row


def _build_evidence_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows
    for group_key, group in df.groupby(["claim_group", "model_alias", "strategy"], dropna=False):
        rows.append(_aggregate_row(group_key, group))
    return rows


def _request_count_for_pair(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if left.empty or right.empty:
        return 0
    return int(min(float(left["request_count"].sum()), float(right["request_count"].sum())))


def _build_paired_delta_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "seed" not in df:
        return []
    paired_candidates = df[df["seed"].notna()].copy()
    if paired_candidates.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_columns = ["paired_claim_group", "model_alias", "seed", "strategy"]
    grouped = paired_candidates.groupby(group_columns, dropna=False).agg(
        request_count=("request_count", "sum"),
        **{metric: (metric, "mean") for metric in METRICS},
    )
    grouped = grouped.reset_index()
    for group_key, group in grouped.groupby(["paired_claim_group", "model_alias"], dropna=False):
        claim_group, model_alias = group_key
        strategies = set(group["strategy"].dropna().astype(str))
        if not {"specialists", "multitask"} <= strategies:
            continue
        deltas = []
        requests = 0
        for seed, seed_group in group.groupby("seed", dropna=False):
            left = seed_group[seed_group["strategy"].eq("specialists")]
            right = seed_group[seed_group["strategy"].eq("multitask")]
            if left.empty or right.empty:
                continue
            delta_row = {"seed": seed}
            for metric in METRICS:
                left_value = left[metric].dropna()
                right_value = right[metric].dropna()
                if left_value.empty or right_value.empty:
                    delta_row[metric] = pd.NA
                else:
                    delta_row[metric] = float(left_value.iloc[0] - right_value.iloc[0])
            requests += _request_count_for_pair(left, right)
            deltas.append(delta_row)
        if not deltas:
            continue
        delta_frame = pd.DataFrame(deltas)
        row: dict[str, Any] = {
            "row_type": "paired_delta",
            "claim_group": claim_group,
            "model_alias": model_alias,
            "strategy": "specialists_vs_multitask_delta",
            "runs": int(len(delta_frame)),
            "requests": requests,
            "paired_seed_count": int(delta_frame["seed"].nunique(dropna=False)),
            "paired_baseline_strategy": "multitask",
            "paired_comparison_strategy": "specialists",
        }
        for metric in METRICS:
            values = pd.to_numeric(delta_frame[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else pd.NA
            ci_low, ci_high = _ci95(values)
            row[f"{metric}_ci95_low"] = ci_low if ci_low is not None else pd.NA
            row[f"{metric}_ci95_high"] = ci_high if ci_high is not None else pd.NA
        rows.append(row)
    return rows


def build_claim_evidence_table(runs_dir: str | Path = "artifacts/runs") -> pd.DataFrame:
    summaries = _normalize_summaries(load_summaries(runs_dir))
    rows = [*_build_evidence_rows(summaries), *_build_paired_delta_rows(summaries)]
    if not rows:
        return _empty_claim_table()
    return (
        pd.DataFrame(rows)
        .reindex(columns=CLAIM_EVIDENCE_COLUMNS)
        .sort_values(["row_type", "claim_group", "model_alias", "strategy"], kind="stable")
        .reset_index(drop=True)
    )


def write_claim_evidence_table(
    runs_dir: str | Path = "artifacts/runs",
    output: str | Path = "reports/tables/claim_evidence.csv",
) -> Path:
    table = build_claim_evidence_table(runs_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="reports/tables/claim_evidence.csv")
    args = parser.parse_args()
    print(write_claim_evidence_table(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
