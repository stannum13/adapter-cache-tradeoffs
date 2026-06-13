from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_plots(df: pd.DataFrame, output_dir: str | Path = "reports/figures") -> list[Path]:
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
    layout_df = df[df["workload"].eq("prompt_layout_ablation")]
    source = layout_df if not layout_df.empty else df
    source.groupby("cache_model")[["p95_ttft_ms", "mean_quality"]].mean().plot(kind="bar", ax=ax)
    ax.set_title("Prompt layout ablation proxy")
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
    return paths
