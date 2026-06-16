from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures",
    request_df: pd.DataFrame | None = None,
) -> list[Path]:
    out = Path(output_dir)
    paths: list[Path] = []
    if df.empty:
        return paths

    fig, ax = plt.subplots(figsize=(7, 4))
    for policy, group in df.groupby("router_policy"):
        sizes = (group["goodput_under_slo"].clip(lower=0.01) * 80).tolist()
        ax.scatter(group["p95_ttft_ms"], group["mean_quality"], s=sizes, label=policy, alpha=0.75)
    ax.set_xlabel("p95 TTFT (ms)")
    ax.set_ylabel("Mean quality")
    ax.set_title("Quality vs p95 TTFT")
    ax.legend(fontsize=8)
    path = out / "quality_vs_p95_ttft.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = df.pivot_table(
        index="router_policy", columns="cache_model", values="cache_hit_rate", aggfunc="mean"
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Cache hit rate")
    ax.set_title("Prefix/cache hit rate")
    path = out / "cache_hit_rate_by_policy_model.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("router_policy")["quality_adjusted_goodput"].mean().sort_values().plot(
        kind="barh", ax=ax
    )
    ax.set_xlabel("Quality-adjusted goodput")
    ax.set_title("Quality-adjusted goodput by router")
    path = out / "quality_adjusted_goodput_by_router.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("cache_model")["memory_token_footprint"].mean().sort_values().plot(kind="bar", ax=ax)
    ax.set_ylabel("Memory tokens")
    ax.set_title("Memory token footprint")
    path = out / "memory_token_footprint_by_cache.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    if request_df is not None and not request_df.empty:
        layout_source = request_df[request_df["workload"].eq("prompt_layout_ablation")]
    else:
        layout_source = pd.DataFrame()
    if not layout_source.empty:
        layout_source.groupby("prompt_layout")[["ttft_ms", "quality"]].mean().plot(
            kind="bar",
            ax=ax,
        )
        ax.set_ylabel("Mean value")
        ax.set_title("Prompt layout ablation")
    else:
        layout_df = df[df["workload"].eq("prompt_layout_ablation")]
        source = layout_df if not layout_df.empty else df
        source.groupby("cache_model")[["p95_ttft_ms", "mean_quality"]].mean().plot(
            kind="bar",
            ax=ax,
        )
        ax.set_title("Prompt layout ablation")
    path = out / "prompt_layout_ablation.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    frontier = df.copy()
    frontier["strategy"] = frontier["cache_model"].replace(
        {
            "standard_lora": "specialists",
            "activated_lora": "activated-style",
            "base_shared": "multitask/base",
        }
    )
    frontier.groupby("strategy")[["mean_quality", "quality_adjusted_goodput"]].mean().plot(
        kind="bar", ax=ax
    )
    ax.set_title("Adapter strategy frontier")
    path = out / "adapter_strategy_frontier.png"
    _save(fig, path)
    paths.append(path)

    if "max_concurrency" in df and df["max_concurrency"].nunique() > 1:
        concurrent = df.copy()
        concurrent["strategy_label"] = (
            concurrent["router_policy"] + " / " + concurrent["cache_model"]
        )
        for metric, ylabel, filename in [
            ("p95_ttft_ms", "p95 TTFT (ms)", "concurrency_p95_ttft.png"),
            ("quality_adjusted_goodput", "Quality-adjusted goodput", "concurrency_qag.png"),
            ("slo_attainment_rate", "SLO attainment", "concurrency_slo_attainment.png"),
            ("request_throughput", "Request throughput", "concurrency_request_throughput.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            pivot = concurrent.pivot_table(
                index="max_concurrency",
                columns="strategy_label",
                values=metric,
                aggfunc="mean",
            ).sort_index()
            pivot.plot(marker="o", ax=ax)
            ax.set_xlabel("Max concurrency")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs concurrency")
            ax.legend(fontsize=7)
            path = out / filename
            _save(fig, path)
            paths.append(path)
    return paths
