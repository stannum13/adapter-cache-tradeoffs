from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
        x = summary["sweep_overlap_fraction"] * 100
        ttft_handle = ax.errorbar(
            x,
            summary["p95_ttft_ms"],
            yerr=summary["p95_ttft_std_ms"].fillna(0),
            marker="o",
            linewidth=2.4,
            capsize=4,
            color="#2563a9",
            label="p95 TTFT",
        )
        ax.set_xlabel("shared-prefix overlap (%)")
        ax.set_ylabel("p95 TTFT (ms)", color="#2563a9")
        ax.tick_params(axis="y", colors="#2563a9")
        ax.grid(axis="y", color="#d8dee8", linewidth=0.8)

        ax2 = ax.twinx()
        ax2.plot(
            x,
            summary["server_prefix_cache_hit_rate"] * 100,
            marker="s",
            linewidth=2.2,
            color="#1b8a7a",
            label="server prefix hit rate",
        )
        ax2.plot(
            x,
            summary["slo_attainment_rate"] * 100,
            marker="^",
            linewidth=2.0,
            color="#b7791f",
            label="SLO attainment",
        )
        ax2.set_ylabel("rate (%)", color="#1b8a7a")
        ax2.tick_params(axis="y", colors="#1b8a7a")
        ax2.set_ylim(0, 105)

        handles = [ttft_handle, *ax2.get_lines()]
        labels = ["p95 TTFT", *[line.get_label() for line in ax2.get_lines()]]
        ax.legend(
            handles,
            labels,
            loc="best",
            fontsize=8,
            frameon=False,
        )
        ax.set_title("Qwen2.5-7B reset-isolated cache locality sweep")

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
