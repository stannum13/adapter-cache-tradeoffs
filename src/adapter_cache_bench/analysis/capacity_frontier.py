from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REQUIRED_COLUMNS = {
    "condition_id",
    "hardware",
    "gpu_model",
    "model",
    "max_model_len",
    "lora_count",
    "status",
}


def load_capacity_frontier(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}
    records = payload.get("records") or []
    if not isinstance(records, list):
        raise ValueError("capacity frontier file must contain a records list")
    df = pd.DataFrame(records)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"capacity frontier records are missing columns: {sorted(missing)}")
    invalid_status = sorted(set(df["status"]) - {"starts", "fails"})
    if invalid_status:
        raise ValueError(f"capacity frontier status must be starts/fails: {invalid_status}")
    return df


def capacity_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition_id",
        "gpu_model",
        "gpu_memory_gb",
        "max_model_len",
        "lora_count",
        "status",
        "available_kv_cache_gib",
        "gpu_kv_cache_tokens",
        "max_concurrency_at_context",
        "failure_message",
    ]
    return df[columns].sort_values(["gpu_memory_gb", "lora_count"], ascending=[True, True])


def write_capacity_tables(
    input_path: str | Path = "data/results/capacity_frontier.yaml",
    output_csv: str | Path = "reports/tables/capacity_frontier.csv",
) -> Path:
    df = capacity_summary(load_capacity_frontier(input_path))
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _format_markdown(df: pd.DataFrame) -> str:
    visible = df.copy()
    for column in [
        "available_kv_cache_gib",
        "max_concurrency_at_context",
    ]:
        visible[column] = visible[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
        )
    visible["gpu_kv_cache_tokens"] = visible["gpu_kv_cache_tokens"].map(
        lambda value: "" if pd.isna(value) else f"{int(value):,}"
    )
    visible["failure_message"] = visible["failure_message"].fillna("")

    columns = list(visible.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in visible.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                text = ""
            else:
                text = str(value).replace("\n", " ").strip()
            cells.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/capacity_frontier.yaml")
    parser.add_argument("--output-csv", default="reports/tables/capacity_frontier.csv")
    parser.add_argument("--format", choices=["markdown", "csv-path"], default="markdown")
    args = parser.parse_args()

    table = capacity_summary(load_capacity_frontier(args.input))
    path = write_capacity_tables(args.input, args.output_csv)
    if args.format == "csv-path":
        print(path)
    else:
        print(_format_markdown(table))


if __name__ == "__main__":
    main()
