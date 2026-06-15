from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries

COLORS = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "grid": "#d7dde3",
    "base": "#d9e6f2",
    "qa": "#2f6f9f",
    "json": "#2a9d8f",
    "summary": "#8d6ab8",
    "code": "#d28c45",
    "specialists": "#1f77b4",
    "multitask": "#2a9d8f",
}


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    ax.title.set_color(COLORS["ink"])


def _block(ax, x: float, y: float, w: float, h: float, color: str, label: str = "") -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=0.6,
        edgecolor="white",
        facecolor=color,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, color="white")


def draw_cache_mechanism(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.98, "A. Cache namespace determines reuse", fontsize=11, weight="bold")
    ax.text(
        0.0,
        0.915,
        "Same document, four task adapters. Standard LoRA repeats the prefix;\n"
        "late specialization shares it before the invocation marker.",
        fontsize=8,
        color=COLORS["muted"],
        va="top",
    )

    ax.text(0.0, 0.79, "standard LoRA cache", fontsize=8, weight="bold", color=COLORS["ink"])
    y0 = 0.68
    adapters = [("qa", "qa"), ("json", "json"), ("summary", "sum"), ("code", "code")]
    for i, (name, label) in enumerate(adapters):
        y = y0 - i * 0.09
        ax.text(0.0, y + 0.025, name, fontsize=7, ha="left", va="center", color=COLORS["muted"])
        for j in range(8):
            _block(ax, 0.13 + j * 0.045, y, 0.038, 0.05, COLORS["base"])
        adapter_color = COLORS[name] if name in COLORS else COLORS["summary"]
        _block(ax, 0.51, y, 0.07, 0.05, adapter_color, label)

    ax.text(0.66, 0.79, "activated-style cache", fontsize=8, weight="bold", color=COLORS["ink"])
    ax.text(0.66, 0.705, "shared prefix", fontsize=7, color=COLORS["muted"])
    for j in range(8):
        _block(ax, 0.66 + j * 0.035, 0.635, 0.03, 0.055, COLORS["base"])
    ax.text(0.66, 0.555, "adapter tails", fontsize=7, color=COLORS["muted"])
    for i, (name, label) in enumerate(adapters):
        x = 0.66 + i * 0.075
        adapter_color = COLORS[name] if name in COLORS else COLORS["summary"]
        _block(ax, x, 0.485, 0.062, 0.052, adapter_color, label)

    ax.annotate(
        "shared once",
        xy=(0.79, 0.665),
        xytext=(0.83, 0.76),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["muted"]},
        fontsize=7,
        color=COLORS["muted"],
    )
    ax.annotate(
        "replicated per adapter",
        xy=(0.28, 0.59),
        xytext=(0.25, 0.29),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["muted"]},
        fontsize=7,
        color=COLORS["muted"],
    )


def _exhaustive(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty or "run_id" not in df:
        return pd.DataFrame()
    return df[df["run_id"].str.contains(name, na=False)].copy()


def draw_overlap_curve(ax, df: pd.DataFrame) -> None:
    ax.set_title(
        "B. Shared-prefix overlap bends the frontier", loc="left", fontsize=11, weight="bold"
    )
    overlap = _exhaustive(df, "exhaustive-overlap")
    if overlap.empty:
        ax.text(0.5, 0.5, "Run make vllm-exhaustive-overlap", ha="center", va="center")
        return
    grouped = (
        overlap.groupby(["sweep_strategy", "sweep_overlap_fraction"], as_index=False)
        .agg(
            qag=("quality_adjusted_goodput", "mean"),
            p95_ttft=("p95_ttft_ms", "mean"),
            server_hit=("server_prefix_cache_hit_rate", "mean"),
        )
        .sort_values("sweep_overlap_fraction")
    )
    for strategy, group in grouped.groupby("sweep_strategy"):
        color = COLORS.get(strategy, COLORS["ink"])
        ax.plot(
            group["sweep_overlap_fraction"] * 100,
            group["qag"],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=color,
            label=strategy,
        )
    ax.set_xlabel("Shared prefix overlap (%)")
    ax.set_ylabel("Quality-adjusted goodput")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    specialists = grouped[grouped["sweep_strategy"].eq("specialists")]
    if not specialists.empty:
        ax2.plot(
            specialists["sweep_overlap_fraction"] * 100,
            specialists["p95_ttft"],
            color="#9ca3af",
            linestyle="--",
            linewidth=1.5,
            label="specialist p95 TTFT",
        )
    ax2.axhline(1000, color="#c2410c", linewidth=1.0, linestyle=":", alpha=0.9)
    ax2.set_ylabel("Specialist p95 TTFT (ms)", color=COLORS["muted"])
    ax2.tick_params(axis="y", colors=COLORS["muted"], labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["grid"])
    _style_axis(ax)


def draw_frontier(ax, df: pd.DataFrame) -> None:
    ax.set_title("C. Repeated held-out frontier", loc="left", fontsize=11, weight="bold")
    confidence = _exhaustive(df, "exhaustive-confidence")
    if confidence.empty:
        ax.text(0.5, 0.5, "Run make vllm-exhaustive-confidence", ha="center", va="center")
        return
    grouped = (
        confidence.groupby(["sweep_strategy", "sweep_concurrency"], as_index=False)
        .agg(
            quality=("mean_quality", "mean"),
            quality_std=("mean_quality", "std"),
            p95_ttft=("p95_ttft_ms", "mean"),
            p95_ttft_std=("p95_ttft_ms", "std"),
            qag=("quality_adjusted_goodput", "mean"),
        )
        .sort_values("sweep_concurrency")
    )
    for _, row in grouped.iterrows():
        strategy = str(row["sweep_strategy"])
        color = COLORS.get(strategy, COLORS["ink"])
        size = 90 + float(row["qag"]) * 16
        ax.scatter(row["p95_ttft"], row["quality"], s=size, color=color, alpha=0.88)
        ax.errorbar(
            row["p95_ttft"],
            row["quality"],
            xerr=row["p95_ttft_std"] if pd.notna(row["p95_ttft_std"]) else 0,
            yerr=row["quality_std"] if pd.notna(row["quality_std"]) else 0,
            color=color,
            linewidth=0.9,
            capsize=2,
            alpha=0.7,
        )
        ax.text(
            row["p95_ttft"] + 8,
            row["quality"] + 0.006,
            f"{strategy} c{int(row['sweep_concurrency'])}\nQAG {row['qag']:.1f}",
            fontsize=7,
            color=COLORS["ink"],
        )
    ax.axvline(1000, color="#c2410c", linestyle=":", linewidth=1.2)
    ax.text(1005, 0.69, "1s TTFT SLO", rotation=90, va="bottom", fontsize=7, color="#c2410c")
    ax.set_xlabel("p95 TTFT (ms)")
    ax.set_ylabel("Mean task quality")
    ax.set_ylim(0.66, 0.88)
    ax.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    _style_axis(ax)


def generate_whitepaper_visual(
    runs_dir: str | Path = "artifacts/runs",
    output_dir: str | Path = "docs/figures",
) -> list[Path]:
    df = load_summaries(runs_dir)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1])
    ax_mechanism = fig.add_subplot(grid[:, 0])
    ax_overlap = fig.add_subplot(grid[0, 1])
    ax_frontier = fig.add_subplot(grid[1, 1])
    draw_cache_mechanism(ax_mechanism)
    draw_overlap_curve(ax_overlap, df)
    draw_frontier(ax_frontier, df)
    fig.suptitle(
        "Specialization is a quality/cache/SLO tradeoff",
        fontsize=15,
        weight="bold",
        x=0.02,
        ha="left",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.012,
        "Adapter Cache Tradeoffs | streamed vLLM sweeps; QAG = quality-adjusted goodput",
        fontsize=8,
        color=COLORS["muted"],
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "whitepaper_specialization_cache_tradeoff.png"
    pdf = out / "whitepaper_specialization_cache_tradeoff.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()
    for path in generate_whitepaper_visual(args.runs_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
