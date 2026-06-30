from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adapter_cache_bench.analysis.plot_style import (
    COLORS,
    apply_dark_theme,
    color_for,
    save_figure,
    style_axis,
    style_legend,
)


def _save(fig, path: Path) -> Path:
    return save_figure(fig, path)


def _barh(ax, values: pd.Series, *, title: str, xlabel: str, color: str) -> None:
    values = values.sort_values()
    y = np.arange(len(values))
    ax.barh(y, values.to_numpy(), color=color, alpha=0.94, height=0.56)
    ax.set_yticks(y)
    ax.set_yticklabels([str(label).replace("_", " ") for label in values.index])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    style_axis(ax, xgrid=True, ygrid=False)


def _grouped_barh(ax, data: pd.DataFrame, *, title: str, xlabel: str) -> None:
    data = data.sort_index()
    y = np.arange(len(data.index))
    columns = list(data.columns)
    height = min(0.22, 0.72 / max(len(columns), 1))
    center = (len(columns) - 1) / 2
    for idx, column in enumerate(columns):
        offset = (idx - center) * height * 1.18
        ax.barh(
            y + offset,
            data[column].to_numpy(),
            height=height,
            color=color_for(idx),
            alpha=0.94,
            label=str(column).replace("_", " "),
        )
    ax.set_yticks(y)
    ax.set_yticklabels([str(label).replace("_", " ") for label in data.index])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    style_axis(ax, xgrid=True, ygrid=False)
    style_legend(ax, fontsize=7.5, loc="lower right")


def generate_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures",
    request_df: pd.DataFrame | None = None,
) -> list[Path]:
    out = Path(output_dir)
    paths: list[Path] = []
    if df.empty:
        return paths
    apply_dark_theme()

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    goodput_scale = float(df["goodput_under_slo"].clip(lower=0).quantile(0.95))
    if goodput_scale <= 0:
        goodput_scale = 1.0
    for idx, (policy, group) in enumerate(df.groupby("router_policy", sort=True)):
        sizes = (
            28 + 190 * (group["goodput_under_slo"].clip(lower=0) / goodput_scale).clip(upper=1.0)
        ).tolist()
        ax.scatter(
            group["p95_ttft_ms"],
            group["mean_quality"],
            s=sizes,
            label=str(policy).replace("_", " "),
            alpha=0.82,
            color=color_for(idx),
            edgecolor=COLORS["background"],
            linewidth=0.45,
        )
    ax.set_xlabel("p95 TTFT (ms)")
    ax.set_ylabel("Mean quality")
    ax.set_title("Quality vs p95 TTFT", loc="left", pad=10)
    style_axis(ax, xgrid=True, ygrid=True)
    style_legend(ax, fontsize=7.5, loc="best")
    path = out / "quality_vs_p95_ttft.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    pivot = df.pivot_table(
        index="router_policy", columns="cache_model", values="cache_hit_rate", aggfunc="mean"
    )
    _grouped_barh(ax, pivot, title="Prefix/cache hit rate", xlabel="Cache hit rate")
    ax.set_xlim(0, min(1.0, max(0.05, float(pivot.max().max()) * 1.18)))
    path = out / "cache_hit_rate_by_policy_model.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    _barh(
        ax,
        df.groupby("router_policy")["quality_adjusted_goodput"].mean(),
        title="Quality-adjusted goodput by router",
        xlabel="Quality-adjusted goodput",
        color=COLORS["teal"],
    )
    path = out / "quality_adjusted_goodput_by_router.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    _barh(
        ax,
        df.groupby("cache_model")["memory_token_footprint"].mean(),
        title="Memory token footprint",
        xlabel="Memory tokens",
        color=COLORS["amber"],
    )
    path = out / "memory_token_footprint_by_cache.png"
    _save(fig, path)
    paths.append(path)

    if request_df is not None and not request_df.empty:
        layout_source = request_df[request_df["workload"].eq("prompt_layout_ablation")]
    else:
        layout_source = pd.DataFrame()
    if not layout_source.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
        layout_means = layout_source.groupby("prompt_layout")[["ttft_ms", "quality"]].mean()
        _barh(
            axes[0],
            layout_means["ttft_ms"],
            title="Prompt layout latency",
            xlabel="Mean TTFT (ms)",
            color=COLORS["blue"],
        )
        _barh(
            axes[1],
            layout_means["quality"],
            title="Prompt layout quality",
            xlabel="Mean quality",
            color=COLORS["green"],
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
        layout_df = df[df["workload"].eq("prompt_layout_ablation")]
        source = layout_df if not layout_df.empty else df
        layout_means = source.groupby("cache_model")[["p95_ttft_ms", "mean_quality"]].mean()
        _barh(
            axes[0],
            layout_means["p95_ttft_ms"],
            title="Prompt layout latency",
            xlabel="p95 TTFT (ms)",
            color=COLORS["blue"],
        )
        _barh(
            axes[1],
            layout_means["mean_quality"],
            title="Prompt layout quality",
            xlabel="Mean quality",
            color=COLORS["green"],
        )
    path = out / "prompt_layout_ablation.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    frontier = df.copy()
    frontier["strategy"] = frontier["cache_model"].replace(
        {
            "standard_lora": "specialists",
            "activated_lora": "activated-style",
            "base_shared": "multitask/base",
        }
    )
    _grouped_barh(
        ax,
        frontier.groupby("strategy")[["mean_quality", "quality_adjusted_goodput"]].mean(),
        title="Adapter strategy frontier",
        xlabel="Mean value",
    )
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
            fig, ax = plt.subplots(figsize=(8.4, 4.8))
            pivot = concurrent.pivot_table(
                index="max_concurrency",
                columns="strategy_label",
                values=metric,
                aggfunc="mean",
            ).sort_index()
            highlights = list(pivot.max().sort_values(ascending=False).head(6).index)
            for column in pivot.columns:
                if column in highlights:
                    highlight_idx = highlights.index(column)
                    color = color_for(highlight_idx)
                    alpha = 0.95
                    linewidth = 2.0
                    label = str(column).replace("_", " ")
                    zorder = 3
                else:
                    color = COLORS["faint"]
                    alpha = 0.18
                    linewidth = 0.9
                    label = None
                    zorder = 1
                ax.plot(
                    pivot.index,
                    pivot[column],
                    marker="o",
                    markersize=4.0 if column in highlights else 2.4,
                    linewidth=linewidth,
                    color=color,
                    alpha=alpha,
                    label=label,
                    zorder=zorder,
                )
            ax.set_xlabel("Max concurrency")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs concurrency", loc="left", pad=10)
            style_axis(ax, xgrid=True, ygrid=True)
            style_legend(ax, fontsize=7.2, loc="best")
            path = out / filename
            _save(fig, path)
            paths.append(path)
    return paths
