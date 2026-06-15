from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from adapter_cache_bench.bench.aggregate import load_summaries

COLORS = {
    "ink": "#1f2933",
    "deep": "#111827",
    "muted": "#6b7280",
    "light": "#f7f9fb",
    "panel": "#fbfcfd",
    "grid": "#dfe5eb",
    "base": "#d8e7f3",
    "base_edge": "#b7cde0",
    "qa": "#2878a8",
    "json": "#2a9d8f",
    "summary": "#8b6fbc",
    "code": "#d18b35",
    "specialists": "#2878c8",
    "multitask": "#2a9d8f",
    "slo": "#c2410c",
    "good": "#eaf5ef",
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


def _panel(ax) -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#e6ebf0")
        spine.set_linewidth(0.8)


def _block(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    label: str = "",
    *,
    edgecolor: str = "white",
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=color,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color="white",
        )


def _badge(ax, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.5,
        weight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.08",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.9,
        },
    )


def draw_cache_mechanism(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel(ax)
    ax.text(
        0.03,
        0.965,
        "A",
        fontsize=10,
        weight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={"boxstyle": "circle,pad=0.22", "facecolor": COLORS["deep"], "edgecolor": "none"},
    )
    ax.text(0.075, 0.972, "Cache namespace determines reuse", fontsize=13, weight="bold")
    ax.text(
        0.075,
        0.92,
        "Same shared document, different task adapters. The serving choice changes both "
        "quality and prefix locality.",
        fontsize=8.5,
        color=COLORS["muted"],
        va="top",
    )

    ax.text(0.06, 0.80, "standard LoRA namespace", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(0.06, 0.765, "cache key includes adapter_id", fontsize=7.5, color=COLORS["muted"])
    y0 = 0.67
    adapters = [("qa", "qa"), ("json", "json"), ("summary", "sum"), ("code", "code")]
    for i, (name, label) in enumerate(adapters):
        y = y0 - i * 0.085
        ax.text(0.06, y + 0.026, name, fontsize=7.6, ha="left", va="center", color=COLORS["muted"])
        for j in range(7):
            _block(
                ax,
                0.16 + j * 0.041,
                y,
                0.035,
                0.052,
                COLORS["base"],
                edgecolor=COLORS["base_edge"],
            )
        adapter_color = COLORS[name] if name in COLORS else COLORS["summary"]
        _block(ax, 0.475, y, 0.065, 0.052, adapter_color, label)
    _badge(ax, 0.335, 0.255, "prefix stored 4x", COLORS["slo"])

    ax.plot([0.585, 0.585], [0.18, 0.82], color="#e6ebf0", linewidth=1.0)

    ax.text(0.64, 0.80, "activated-style namespace", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(
        0.64,
        0.765,
        "base prefix before invocation marker",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    ax.text(0.64, 0.68, "shared document prefix", fontsize=7.5, color=COLORS["muted"])
    for j in range(8):
        _block(
            ax,
            0.64 + j * 0.034,
            0.605,
            0.029,
            0.058,
            COLORS["base"],
            edgecolor=COLORS["base_edge"],
        )
    ax.text(0.64, 0.525, "<ADAPTER:task> tails", fontsize=7.5, color=COLORS["muted"])
    for i, (name, label) in enumerate(adapters):
        x = 0.64 + i * 0.075
        adapter_color = COLORS[name] if name in COLORS else COLORS["summary"]
        _block(ax, x, 0.455, 0.062, 0.052, adapter_color, label)
        ax.plot(
            [0.74, x + 0.031],
            [0.605, 0.507],
            color="#c7d0da",
            linewidth=0.8,
            alpha=0.8,
        )
    _badge(ax, 0.755, 0.255, "prefix stored 1x", COLORS["json"])

    ax.annotate(
        "quality choice becomes\na cache choice",
        xy=(0.475, 0.545),
        xytext=(0.36, 0.84),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLORS["muted"]},
        fontsize=7.5,
        color=COLORS["muted"],
        ha="center",
    )
    ax.annotate(
        "late specialization preserves\nbase-prefix compatibility",
        xy=(0.75, 0.635),
        xytext=(0.84, 0.38),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLORS["muted"]},
        fontsize=7.5,
        color=COLORS["muted"],
        ha="center",
    )


def _exhaustive(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty or "run_id" not in df:
        return pd.DataFrame()
    return df[df["run_id"].str.contains(name, na=False)].copy()


def draw_overlap_curve(ax, df: pd.DataFrame) -> None:
    _panel(ax)
    ax.set_title(
        "B. Shared-prefix overlap bends the frontier", loc="left", fontsize=12, weight="bold"
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
    ax.axvspan(75, 100, color=COLORS["good"], alpha=0.75, zorder=0)
    ax.text(
        76.5,
        1.31,
        "high-overlap region\nwhere specialists pay off",
        fontsize=7.5,
        color="#166534",
        va="top",
    )
    for strategy, group in grouped.groupby("sweep_strategy"):
        color = COLORS.get(strategy, COLORS["ink"])
        ax.plot(
            group["sweep_overlap_fraction"] * 100,
            group["qag"],
            marker="o",
            linewidth=2.4,
            markersize=5,
            color=color,
            label=strategy,
        )
    ax.set_xlabel("Shared prefix overlap (%)")
    ax.set_ylabel("Quality-adjusted goodput")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=2.8)
    ax2 = ax.twinx()
    specialists = grouped[grouped["sweep_strategy"].eq("specialists")]
    if not specialists.empty:
        ax2.plot(
            specialists["sweep_overlap_fraction"] * 100,
            specialists["p95_ttft"],
            color="#9ca3af",
            linestyle="--",
            linewidth=1.6,
            label="specialist p95 TTFT",
        )
    ax2.axhline(1000, color=COLORS["slo"], linewidth=1.1, linestyle=":", alpha=0.95)
    ax2.text(99, 1025, "1s TTFT SLO", ha="right", va="bottom", fontsize=7, color=COLORS["slo"])
    ax2.set_ylabel("Specialist p95 TTFT (ms)", color=COLORS["muted"])
    ax2.tick_params(axis="y", colors=COLORS["muted"], labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["grid"])
    _style_axis(ax)


def draw_frontier(ax, df: pd.DataFrame) -> None:
    _panel(ax)
    ax.set_title("C. Repeated held-out frontier", loc="left", fontsize=12, weight="bold")
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
    ax.axvspan(0, 1000, color=COLORS["good"], alpha=0.7, zorder=0)
    ax.text(
        690,
        0.828,
        "SLO-feasible side\n(<1s p95 TTFT)",
        fontsize=7.5,
        color="#166534",
        va="top",
    )
    offsets = {
        ("specialists", 8): (-132, 0.014),
        ("specialists", 16): (18, 0.012),
        ("multitask", 8): (10, 0.011),
        ("multitask", 16): (12, 0.008),
    }
    for _, row in grouped.iterrows():
        strategy = str(row["sweep_strategy"])
        color = COLORS.get(strategy, COLORS["ink"])
        size = 90 + float(row["qag"]) * 16
        ax.scatter(
            row["p95_ttft"],
            row["quality"],
            s=size,
            color=color,
            alpha=0.86,
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
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
        dx, dy = offsets.get((strategy, int(row["sweep_concurrency"])), (8, 0.006))
        ax.text(
            row["p95_ttft"] + dx,
            row["quality"] + dy,
            f"{strategy} c{int(row['sweep_concurrency'])}\nQAG {row['qag']:.1f}",
            fontsize=7,
            color=COLORS["ink"],
            bbox={
                "boxstyle": "round,pad=0.15,rounding_size=0.05",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
    ax.axvline(1000, color=COLORS["slo"], linestyle=":", linewidth=1.3)
    ax.text(1005, 0.667, "1s TTFT SLO", rotation=90, va="bottom", fontsize=7, color=COLORS["slo"])
    ax.set_xlabel("p95 TTFT (ms)")
    ax.set_ylabel("Mean task quality")
    ax.set_ylim(0.66, 0.88)
    ax.set_xlim(620, 1560)
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
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(14.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=[1.1, 1.1, 1, 1, 1], height_ratios=[1, 1])
    ax_mechanism = fig.add_subplot(grid[:, :2])
    ax_overlap = fig.add_subplot(grid[0, 2:])
    ax_frontier = fig.add_subplot(grid[1, 2:])
    draw_cache_mechanism(ax_mechanism)
    draw_overlap_curve(ax_overlap, df)
    draw_frontier(ax_frontier, df)
    fig.suptitle(
        "When specialization is worth its cache footprint",
        fontsize=18,
        weight="bold",
        x=0.02,
        ha="left",
        color=COLORS["deep"],
        y=1.02,
    )
    fig.text(
        0.02,
        0.975,
        "Adapter routing improves task quality, but every adapter decision also changes "
        "prefix-cache reuse.",
        fontsize=10,
        color=COLORS["muted"],
        ha="left",
    )
    fig.text(
        0.02,
        -0.01,
        "Adapter Cache Tradeoffs | streamed vLLM sweeps on Qwen2.5-1.5B-Instruct; "
        "QAG = quality-adjusted goodput",
        fontsize=8,
        color=COLORS["muted"],
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
