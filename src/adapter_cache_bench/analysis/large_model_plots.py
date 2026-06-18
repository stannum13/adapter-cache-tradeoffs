from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if summary.empty:
        ax.text(0.5, 0.5, "No large-model overlap runs found", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [f"{value:.0%} overlap" for value in summary["sweep_overlap_fraction"]]
        x = range(len(summary))
        bars = ax.bar(
            list(x),
            summary["p95_ttft_ms"],
            yerr=summary["p95_ttft_std_ms"].fillna(0).to_numpy(),
            capsize=5,
            color="#2563a9",
            alpha=0.88,
            label="p95 TTFT",
        )
        ax.bar_label(bars, fmt="%.0f ms", fontsize=8, padding=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("p95 TTFT (ms)", color="#2563a9")
        ax.tick_params(axis="y", colors="#2563a9")
        ax.grid(axis="y", color="#d8dee8", linewidth=0.8)

        ax2 = ax.twinx()
        ax2.scatter(
            list(x),
            summary["server_prefix_cache_hit_rate"] * 100,
            marker="s",
            s=70,
            color="#1b8a7a",
            label="server prefix hit rate",
            zorder=4,
        )
        ax2.scatter(
            list(x),
            summary["slo_attainment_rate"] * 100,
            marker="^",
            s=72,
            color="#b7791f",
            label="SLO attainment",
            zorder=4,
        )
        ax2.set_ylabel("rate (%)", color="#1b8a7a")
        ax2.tick_params(axis="y", colors="#1b8a7a")
        ax2.set_ylim(0, 105)

        ax.legend(
            [
                Patch(facecolor="#2563a9", alpha=0.88, label="p95 TTFT"),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    color="#1b8a7a",
                    label="server prefix hit rate",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="None",
                    color="#b7791f",
                    label="SLO attainment",
                ),
            ],
            ["p95 TTFT", "server prefix hit rate", "SLO attainment"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=3,
            fontsize=8,
            frameon=False,
        )
        ax.set_title("Qwen2.5-7B reset-isolated two-condition confirmation")

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="docs/figures/large_model_overlap_confidence.png")
    args = parser.parse_args()
    print(write_large_model_overlap_plot(args.runs_dir, args.output))


if __name__ == "__main__":
    main()
