from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

STRATEGY_BY_CACHE_MODEL = {
    "standard_lora": "specialist-adapter",
    "base_shared": "multitask-or-shared-base",
    "activated_lora": "activated-late-specialization",
    "copy_on_write": "copy-on-write-delta",
}


def load_summaries(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(runs_dir).glob("*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return pd.DataFrame(rows)


def with_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    enriched["adapter_strategy"] = (
        enriched["cache_model"].map(STRATEGY_BY_CACHE_MODEL).fillna(enriched["cache_model"])
    )
    enriched["router_cache_pair"] = enriched["router_policy"] + " / " + enriched["cache_model"]
    return enriched


def workload_leaders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = with_strategy_columns(df)
    ordered = enriched.sort_values(
        ["workload", "quality_adjusted_goodput", "mean_quality", "p95_ttft_ms"],
        ascending=[True, False, False, True],
    )
    columns = [
        "workload",
        "router_policy",
        "cache_model",
        "adapter_strategy",
        "quality_adjusted_goodput",
        "mean_quality",
        "p95_ttft_ms",
        "cache_hit_rate",
        "memory_token_footprint",
        "fragmentation_index",
    ]
    return ordered.groupby("workload", as_index=False).head(1)[columns].reset_index(drop=True)


def cache_model_means(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = with_strategy_columns(df)
    return (
        enriched.groupby(["cache_model", "adapter_strategy"], as_index=False)
        .agg(
            quality_adjusted_goodput=("quality_adjusted_goodput", "mean"),
            mean_quality=("mean_quality", "mean"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            cache_hit_rate=("cache_hit_rate", "mean"),
            memory_token_footprint=("memory_token_footprint", "mean"),
            fragmentation_index=("fragmentation_index", "mean"),
            eviction_count=("eviction_count", "mean"),
            evicted_tokens=("evicted_tokens", "mean"),
        )
        .sort_values("quality_adjusted_goodput", ascending=False)
        .reset_index(drop=True)
    )


def router_means(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby("router_policy", as_index=False)
        .agg(
            quality_adjusted_goodput=("quality_adjusted_goodput", "mean"),
            mean_quality=("mean_quality", "mean"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            cache_hit_rate=("cache_hit_rate", "mean"),
        )
        .sort_values("quality_adjusted_goodput", ascending=False)
        .reset_index(drop=True)
    )


def repeated_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["workload", "router_policy", "cache_model"], as_index=False)
        .agg(
            run_count=("quality_adjusted_goodput", "count"),
            quality_adjusted_goodput_mean=("quality_adjusted_goodput", "mean"),
            quality_adjusted_goodput_std=("quality_adjusted_goodput", "std"),
            mean_quality_mean=("mean_quality", "mean"),
            mean_quality_std=("mean_quality", "std"),
            p95_ttft_ms_mean=("p95_ttft_ms", "mean"),
            p95_ttft_ms_std=("p95_ttft_ms", "std"),
        )
        .sort_values("quality_adjusted_goodput_mean", ascending=False)
        .reset_index(drop=True)
        .fillna(0.0)
    )


def layout_ablation_means(request_df: pd.DataFrame) -> pd.DataFrame:
    if request_df.empty or "workload" not in request_df:
        return pd.DataFrame()
    layout_rows = request_df[request_df["workload"].eq("prompt_layout_ablation")]
    if layout_rows.empty:
        return pd.DataFrame()
    return (
        layout_rows.groupby(["prompt_layout", "cache_model"], as_index=False)
        .agg(
            ttft_ms=("ttft_ms", "mean"),
            e2e_ms=("e2e_ms", "mean"),
            quality=("quality", "mean"),
            cached_prompt_tokens=("cached_prompt_tokens", "mean"),
            prompt_tokens=("prompt_tokens", "mean"),
        )
        .sort_values(["prompt_layout", "cache_model"])
        .reset_index(drop=True)
    )


def write_analysis_tables(
    df: pd.DataFrame,
    request_df: pd.DataFrame,
    output_dir: str | Path = "reports/tables",
) -> dict[str, Path]:
    from adapter_cache_bench.analysis.pareto import workload_pareto_frontiers
    from adapter_cache_bench.analysis.slo import slo_sweep

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "summaries": with_strategy_columns(df),
        "workload_leaders": workload_leaders(df),
        "cache_model_means": cache_model_means(df),
        "router_means": router_means(df),
        "repeated_seed_summary": repeated_seed_summary(df),
        "layout_ablation": layout_ablation_means(request_df),
        "pareto_frontier": workload_pareto_frontiers(df),
        "slo_sweep": slo_sweep(request_df),
    }
    paths = {}
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def load_request_rows(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(runs_dir).glob("*/requests.jsonl"):
        summary_path = path.parent / "summary.json"
        metadata = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            metadata = {
                "run_id": summary["run_id"],
                "router_policy": summary["router_policy"],
                "cache_model": summary["cache_model"],
                "workload": summary["workload"],
            }
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                request = record["request"]
                response = record["response"]
                metrics = response["metrics"]
                quality = response["quality"]
                routing = record["routing"]
                rows.append(
                    {
                        **metadata,
                        "request_id": request["request_id"],
                        "prompt_layout": request["prompt_layout"],
                        "task_type": request["task_type"],
                        "adapter_id": routing["adapter_id"],
                        "ttft_ms": metrics["ttft_ms"],
                        "e2e_ms": metrics["e2e_ms"],
                        "cached_prompt_tokens": metrics["cached_prompt_tokens"],
                        "prompt_tokens": metrics["prompt_tokens"],
                        "quality": quality["score"],
                    }
                )
    return pd.DataFrame(rows)
