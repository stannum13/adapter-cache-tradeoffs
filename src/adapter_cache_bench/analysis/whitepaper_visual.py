from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries

COLORS = {
    "ink": "#18212f",
    "muted": "#687384",
    "grid": "#e3e8ef",
    "panel": "#fbfcfe",
    "prefix": "#d9e8f4",
    "prefix_edge": "#b7ccdf",
    "qa": "#2d77b8",
    "json": "#2a9d8f",
    "summary": "#8766b5",
    "code": "#d28b35",
    "specialists": "#2878c8",
    "multitask": "#2a9d8f",
    "slo": "#c2410c",
    "good": "#eaf6ef",
}

ADAPTERS = [
    ("qa", COLORS["qa"]),
    ("json", COLORS["json"]),
    ("summary", COLORS["summary"]),
    ("code", COLORS["code"]),
]


def _panel(ax) -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(COLORS["grid"])


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)


def _rounded(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    *,
    edgecolor: str = "white",
    linewidth: float = 0.8,
) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def _prefix_blocks(ax, x: float, y: float, count: int, width: float, height: float) -> None:
    for index in range(count):
        _rounded(
            ax,
            x + index * width * 1.08,
            y,
            width,
            height,
            COLORS["prefix"],
            edgecolor=COLORS["prefix_edge"],
        )


def _adapter_tail(ax, x: float, y: float, label: str, color: str) -> None:
    _rounded(ax, x, y, 0.072, 0.052, color)
    ax.text(
        x + 0.036,
        y + 0.026,
        label,
        color="white",
        fontsize=7.5,
        weight="bold",
        ha="center",
        va="center",
    )


def draw_mechanism(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel(ax)

    ax.text(0.05, 0.93, "1. The hidden cost", fontsize=12, weight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.885,
        "Routing to a specialist can improve quality, but it also changes the KV-cache namespace.",
        fontsize=8.8,
        color=COLORS["muted"],
    )

    ax.text(0.05, 0.79, "standard LoRA", fontsize=9.5, weight="bold", color=COLORS["ink"])
    ax.text(0.05, 0.755, "same prefix cached once per adapter", fontsize=8, color=COLORS["muted"])
    for row, (adapter, color) in enumerate(ADAPTERS):
        y = 0.67 - row * 0.075
        ax.text(0.05, y + 0.026, adapter, fontsize=7.5, color=COLORS["muted"], va="center")
        _prefix_blocks(ax, 0.18, y, 7, 0.034, 0.052)
        _adapter_tail(ax, 0.48, y, adapter[:4], color)

    ax.text(
        0.31,
        0.325,
        "prefix footprint scales with adapter count",
        fontsize=8.2,
        color=COLORS["slo"],
        weight="bold",
        ha="center",
    )

    ax.plot([0.57, 0.57], [0.22, 0.80], color=COLORS["grid"], linewidth=1.0)

    ax.text(0.63, 0.79, "late specialization", fontsize=9.5, weight="bold", color=COLORS["ink"])
    ax.text(0.63, 0.755, "shared prefix, adapter-specific tail", fontsize=8, color=COLORS["muted"])
    _prefix_blocks(ax, 0.63, 0.62, 8, 0.033, 0.058)
    ax.text(0.63, 0.59, "shared document prefix", fontsize=7.5, color=COLORS["muted"])
    for index, (adapter, color) in enumerate(ADAPTERS):
        x = 0.63 + index * 0.078
        _adapter_tail(ax, x, 0.45, adapter[:4], color)
    ax.text(
        0.765,
        0.325,
        "prefix footprint stays nearly flat",
        fontsize=8.2,
        color=COLORS["json"],
        weight="bold",
        ha="center",
    )

    ax.annotate(
        "",
        xy=(0.74, 0.45),
        xytext=(0.76, 0.62),
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 0.9},
    )
    ax.text(0.785, 0.535, "<ADAPTER:task>", fontsize=7.5, color=COLORS["muted"])


def _exhaustive(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty or "run_id" not in df:
        return pd.DataFrame()
    return df[df["run_id"].str.contains(name, na=False)].copy()


def draw_overlap(ax, df: pd.DataFrame) -> None:
    _panel(ax)
    ax.set_title(
        "2. Specialization pays when overlap is high",
        loc="left",
        fontsize=12,
        weight="bold",
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
        )
        .sort_values("sweep_overlap_fraction")
    )

    ax.axvspan(75, 100, color=COLORS["good"], alpha=0.85, zorder=0)
    ax.text(
        77,
        1.30,
        "high reuse",
        fontsize=8,
        color="#166534",
        weight="bold",
        ha="left",
    )

    for strategy, group in grouped.groupby("sweep_strategy", sort=False):
        color = COLORS.get(strategy, COLORS["ink"])
        ax.plot(
            group["sweep_overlap_fraction"] * 100,
            group["qag"],
            marker="o",
            markersize=4.8,
            linewidth=2.4,
            color=color,
            label=strategy,
        )

    specialists = grouped[grouped["sweep_strategy"].eq("specialists")]
    if not specialists.empty:
        ax2 = ax.twinx()
        ax2.plot(
            specialists["sweep_overlap_fraction"] * 100,
            specialists["p95_ttft"],
            color="#9aa4b2",
            linestyle="--",
            linewidth=1.7,
        )
        ax2.axhline(1000, color=COLORS["slo"], linestyle=":", linewidth=1.1)
        ax2.text(
            99,
            1025,
            "1s TTFT SLO",
            fontsize=7.5,
            color=COLORS["slo"],
            ha="right",
            va="bottom",
        )
        ax2.set_ylabel("specialist p95 TTFT (ms)", fontsize=8.5, color=COLORS["muted"])
        ax2.tick_params(axis="y", colors=COLORS["muted"], labelsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(COLORS["grid"])

    ax.set_xlabel("shared-prefix overlap (%)", fontsize=9)
    ax.set_ylabel("quality-adjusted goodput", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-5, 102)
    ax.set_ylim(-0.03, max(1.45, float(grouped["qag"].max()) * 1.08))
    _style_axis(ax)


def draw_takeaway(fig) -> None:
    box = patches.FancyBboxPatch(
        (0.19, 0.025),
        0.64,
        0.050,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#f7fafc",
        edgecolor=COLORS["grid"],
        linewidth=0.8,
        transform=fig.transFigure,
    )
    fig.add_artist(box)
    fig.text(
        0.5,
        0.050,
        "Decision rule: specialize only when quality gain exceeds the cache-footprint "
        "and SLO cost.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
        weight="bold",
    )


def generate_whitepaper_visual(
    runs_dir: str | Path = "artifacts/runs",
    output_dir: str | Path = "docs/figures",
) -> list[Path]:
    df = load_summaries(runs_dir)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.6, 6.6), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        left=0.035,
        right=0.985,
        top=0.84,
        bottom=0.18,
        width_ratios=[1.02, 1.0],
        wspace=0.12,
    )
    ax_mechanism = fig.add_subplot(grid[0, 0])
    ax_overlap = fig.add_subplot(grid[0, 1])
    draw_mechanism(ax_mechanism)
    draw_overlap(ax_overlap, df)
    draw_takeaway(fig)

    fig.text(
        0.035,
        0.935,
        "When is specialization worth its cache footprint?",
        fontsize=20,
        weight="bold",
        color=COLORS["ink"],
        ha="left",
    )
    fig.text(
        0.035,
        0.895,
        "Specialist adapters improve task quality; shared-prefix reuse decides whether "
        "that quality is cheap enough to serve.",
        fontsize=10,
        color=COLORS["muted"],
        ha="left",
    )
    fig.text(
        0.035,
        0.008,
        "Adapter Cache Tradeoffs | streamed vLLM sweeps; QAG = quality-adjusted goodput",
        fontsize=7.8,
        color=COLORS["muted"],
        ha="left",
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "whitepaper_specialization_cache_tradeoff.png"
    pdf = out / "whitepaper_specialization_cache_tradeoff.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
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
