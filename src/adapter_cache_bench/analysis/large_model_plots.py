from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from adapter_cache_bench.analysis.plot_style import (
    COLORS,
    apply_dark_theme,
    save_figure,
    style_axis,
)
from adapter_cache_bench.bench.aggregate import load_summaries


def latest_large_overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "run_id" not in df:
        return pd.DataFrame()
    sub = df[df["run_id"].str.contains("large-model-overlap-confidence-vllm-streaming", na=False)]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("run_id").drop_duplicates(
        ["sweep_overlap_fraction", "sweep_seed"],
        keep="last",
    )
    return (
        sub.groupby("sweep_overlap_fraction", as_index=False)
        .agg(
            runs=("run_id", "count"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            p95_ttft_std_ms=("p95_ttft_ms", "std"),
            slo_attainment_rate=("slo_attainment_rate", "mean"),
            quality_adjusted_goodput=("quality_adjusted_goodput", "mean"),
            server_prefix_cache_hit_rate=("server_prefix_cache_hit_rate", "mean"),
        )
        .sort_values("sweep_overlap_fraction")
    )


def write_large_model_overlap_plot(
    runs_dir: str | Path = "artifacts/runs",
    output: str | Path = "docs/figures/large_model_overlap_confidence.png",
) -> Path:
    summary = latest_large_overlap_summary(load_summaries(runs_dir))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    apply_dark_theme()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if summary.empty:
        ax.text(
            0.5,
            0.5,
            "No large-model overlap runs found",
            ha="center",
            va="center",
            color=COLORS["muted"],
        )
        ax.set_axis_off()
    else:
        labels = [f"{value:.0%} overlap" for value in summary["sweep_overlap_fraction"]]
        y = np.arange(len(summary))
        bars = ax.barh(
            y,
            summary["p95_ttft_ms"],
            xerr=summary["p95_ttft_std_ms"].fillna(0).to_numpy(),
            capsize=5,
            color=COLORS["blue"],
            alpha=0.94,
            height=0.54,
            label="p95 TTFT",
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.0f} ms" for value in summary["p95_ttft_ms"]],
            fontsize=8,
            padding=4,
            color=COLORS["text"],
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("p95 TTFT (ms)")
        style_axis(ax, xgrid=True, ygrid=False)

        ax2 = ax.twiny()
        ax2.scatter(
            summary["server_prefix_cache_hit_rate"] * 100,
            y,
            marker="o",
            s=58,
            color=COLORS["teal"],
            label="server prefix hit rate",
            zorder=4,
        )
        ax2.scatter(
            summary["slo_attainment_rate"] * 100,
            y,
            marker="s",
            s=56,
            color=COLORS["amber"],
            label="SLO attainment",
            zorder=4,
        )
        ax2.set_xlabel("Rate (%)", color=COLORS["muted"])
        ax2.tick_params(axis="x", colors=COLORS["muted"], labelsize=8.5)
        ax2.spines["top"].set_color(COLORS["spine"])
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.set_xlim(0, 105)

        legend = ax.legend(
            [
                Line2D(
                    [0],
                    [0],
                    color=COLORS["blue"],
                    linewidth=5,
                    label="p95 TTFT",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color=COLORS["teal"],
                    label="server prefix hit rate",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    color=COLORS["amber"],
                    label="SLO attainment",
                ),
            ],
            ["p95 TTFT", "server prefix hit rate", "SLO attainment"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            fontsize=8,
            frameon=False,
        )
        for text in legend.get_texts():
            text.set_color(COLORS["muted"])
        ax.set_title("Qwen2.5-7B reset-isolated two-condition confirmation", loc="left", pad=10)

    return save_figure(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="docs/figures/large_model_overlap_confidence.png")
    args = parser.parse_args()
    print(write_large_model_overlap_plot(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
