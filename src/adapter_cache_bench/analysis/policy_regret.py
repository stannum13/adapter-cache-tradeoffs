from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries

POLICY_REGRET_COLUMNS = [
    "regime_id",
    "regime_key",
    "workload",
    "policy",
    "router_policy",
    "cache_model",
    "strategy",
    "runs",
    "requests",
    "qag",
    "best_qag",
    "regret",
    "relative_regret",
    "rank",
    "baseline_policy",
    "baseline_router_policy",
    "baseline_cache_model",
    "baseline_strategy",
    "baseline_source",
    "oracle_present",
]

_NON_REGIME_SWEEP_KEYS = {
    "cache",
    "cache_model",
    "policy",
    "router",
    "router_policy",
    "seed",
    "strategy",
    "workload",
}

_POLICY_COLUMNS = ["policy", "router_policy", "cache_model", "strategy"]


def _first_present(row: pd.Series, columns: list[str], default: Any = None) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]) and row[column] != "":
            return row[column]
    return default


def _dimension_columns(df: pd.DataFrame) -> list[str]:
    sweep_columns = [
        column
        for column in df.columns
        if column.startswith("sweep_")
        and column.removeprefix("sweep_") not in _NON_REGIME_SWEEP_KEYS
    ]
    return ["workload", *sorted(sweep_columns)]


def _format_dimension_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _regime_key(row: pd.Series, dimension_columns: list[str]) -> str:
    return "|".join(
        f"{column.removeprefix('sweep_')}={_format_dimension_value(row.get(column))}"
        for column in dimension_columns
    )


def _infer_strategy(row: pd.Series) -> str:
    strategy = _first_present(row, ["sweep_strategy"])
    if strategy is not None:
        return str(strategy)
    router_policy = str(_first_present(row, ["router_policy"], "unknown"))
    if router_policy == "multitask":
        return "multitask"
    if router_policy == "oracle":
        return "oracle"
    return str(_first_present(row, ["cache_model"], router_policy))


def _infer_policy(row: pd.Series) -> str:
    strategy = str(row["strategy"])
    if strategy not in {"", "unknown"}:
        return strategy
    router_policy = str(row["router_policy"])
    cache_model = str(row["cache_model"])
    return f"{router_policy}/{cache_model}"


def _is_oracle(row: pd.Series) -> bool:
    values = [
        row.get("policy"),
        row.get("router_policy"),
        row.get("strategy"),
        row.get("sweep_strategy"),
    ]
    return any(str(value).lower() == "oracle" for value in values if pd.notna(value))


def _normalize_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*POLICY_REGRET_COLUMNS])

    normalized = df.copy()
    if "quality_adjusted_goodput" in normalized:
        normalized["qag"] = pd.to_numeric(normalized["quality_adjusted_goodput"], errors="coerce")
    elif "qag" in normalized:
        normalized["qag"] = pd.to_numeric(normalized["qag"], errors="coerce")
    else:
        normalized["qag"] = pd.NA

    if "request_count" not in normalized:
        normalized["request_count"] = 0
    normalized["request_count"] = pd.to_numeric(
        normalized["request_count"], errors="coerce"
    ).fillna(0)

    normalized["workload"] = normalized.apply(
        lambda row: str(_first_present(row, ["workload", "sweep_workload"], "unknown")),
        axis=1,
    )
    normalized["router_policy"] = normalized.apply(
        lambda row: str(
            _first_present(row, ["router_policy", "sweep_router", "sweep_router_policy"], "unknown")
        ),
        axis=1,
    )
    normalized["cache_model"] = normalized.apply(
        lambda row: str(
            _first_present(row, ["cache_model", "sweep_cache", "sweep_cache_model"], "unknown")
        ),
        axis=1,
    )
    normalized["strategy"] = normalized.apply(_infer_strategy, axis=1)
    normalized["policy"] = normalized.apply(_infer_policy, axis=1)
    normalized["is_oracle"] = normalized.apply(_is_oracle, axis=1)
    return normalized[normalized["qag"].notna()].reset_index(drop=True)


def _aggregate_policies(
    summaries: pd.DataFrame, dimension_columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    if summaries.empty:
        return pd.DataFrame(), dimension_columns
    for column in dimension_columns:
        if column not in summaries:
            summaries[column] = pd.NA
    grouped = (
        summaries.groupby([*dimension_columns, *_POLICY_COLUMNS], dropna=False)
        .agg(
            runs=("qag", "count"),
            requests=("request_count", "sum"),
            qag=("qag", "mean"),
            oracle_present=("is_oracle", "any"),
        )
        .reset_index()
    )
    grouped["regime_key"] = grouped.apply(lambda row: _regime_key(row, dimension_columns), axis=1)
    return grouped, dimension_columns


def _baseline_for_group(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values(
        ["qag", "policy", "router_policy", "cache_model"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return ordered.iloc[0]


def build_policy_regret_table(runs_dir: str | Path = "artifacts/runs") -> pd.DataFrame:
    summaries = _normalize_summaries(load_summaries(runs_dir))
    dimension_columns = _dimension_columns(summaries)
    dynamic_columns = [column for column in dimension_columns if column != "workload"]
    columns = [*POLICY_REGRET_COLUMNS, *dynamic_columns]
    if summaries.empty:
        return pd.DataFrame(columns=columns)

    policies, dimension_columns = _aggregate_policies(summaries, dimension_columns)
    rows: list[dict[str, Any]] = []
    for regime_index, (_regime_key_value, group) in enumerate(
        policies.groupby("regime_key", sort=True, dropna=False),
        start=1,
    ):
        baseline = _baseline_for_group(group)
        best_qag = float(baseline["qag"])
        oracle_present = bool(group["oracle_present"].any())
        baseline_is_oracle = _is_oracle(baseline)
        ranked = group.copy()
        ranked["rank"] = ranked["qag"].rank(method="dense", ascending=False).astype(int)
        ranked = ranked.sort_values(
            ["rank", "policy", "router_policy", "cache_model"], kind="stable"
        )
        for _, row in ranked.iterrows():
            regret = best_qag - float(row["qag"])
            relative_regret = regret / best_qag if best_qag > 0 else pd.NA
            output_row: dict[str, Any] = {
                "regime_id": f"regime_{regime_index:03d}",
                "regime_key": row["regime_key"],
                "workload": row["workload"],
                "policy": row["policy"],
                "router_policy": row["router_policy"],
                "cache_model": row["cache_model"],
                "strategy": row["strategy"],
                "runs": int(row["runs"]),
                "requests": int(row["requests"]),
                "qag": float(row["qag"]),
                "best_qag": best_qag,
                "regret": regret,
                "relative_regret": relative_regret,
                "rank": int(row["rank"]),
                "baseline_policy": baseline["policy"],
                "baseline_router_policy": baseline["router_policy"],
                "baseline_cache_model": baseline["cache_model"],
                "baseline_strategy": baseline["strategy"],
                "baseline_source": "oracle" if baseline_is_oracle else "best_observed",
                "oracle_present": oracle_present,
            }
            for column in dynamic_columns:
                output_row[column] = row[column]
            rows.append(output_row)
    return pd.DataFrame(rows).reindex(columns=columns).reset_index(drop=True)


def write_policy_regret_table(
    runs_dir: str | Path = "artifacts/runs",
    output: str | Path = "reports/tables/policy_regret.csv",
) -> Path:
    table = build_policy_regret_table(runs_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="reports/tables/policy_regret.csv")
    args = parser.parse_args()
    print(write_policy_regret_table(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
