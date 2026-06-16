from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from adapter_cache_bench.analysis.capacity_frontier import load_capacity_frontier
from adapter_cache_bench.bench.aggregate import load_summaries

COLORS = {
    "ink": "#142033",
    "muted": "#5d6b7c",
    "grid": "#d8dee8",
    "panel": "#fbfcff",
    "blue": "#2563a9",
    "teal": "#1b8a7a",
    "gold": "#b7791f",
    "red": "#b23b3b",
    "green": "#2f7d46",
    "purple": "#6f58a8",
    "prefix": "#dbeafe",
    "prefix_edge": "#9fbfe8",
}


def _panel(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(0.9)
    ax.set_title(title, loc="left", fontsize=11.5, weight="bold", color=COLORS["ink"], pad=12)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.2,
            color=COLORS["muted"],
            ha="left",
            va="bottom",
        )


def _axis_style(ax, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.9)


def _box(ax, x: float, y: float, w: float, h: float, color: str, text: str = "") -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
        )
    )
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=7.2,
            color="white",
            weight="bold",
        )


def _blocks(ax, x: float, y: float, n: int, *, w: float = 0.036, h: float = 0.040) -> None:
    for i in range(n):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + i * w * 1.08, y),
                w,
                h,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                facecolor=COLORS["prefix"],
                edgecolor=COLORS["prefix_edge"],
                linewidth=0.7,
            )
        )


def draw_mechanism(ax) -> None:
    _panel(ax, "A. Cache namespace mechanism")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.05, 0.83, "standard LoRA", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(
        0.05,
        0.78,
        "same prefix, repeated per adapter",
        fontsize=7.8,
        color=COLORS["muted"],
    )
    adapters = [("qa", COLORS["blue"]), ("json", COLORS["teal"]), ("sum", COLORS["purple"])]
    for row, (name, color) in enumerate(adapters):
        y = 0.66 - row * 0.10
        _blocks(ax, 0.06, y, 7)
        _box(ax, 0.36, y - 0.002, 0.060, 0.044, color, name)
    ax.text(
        0.24,
        0.34,
        "prefix memory scales with adapter count",
        ha="center",
        fontsize=8,
        color=COLORS["red"],
    )

    ax.plot([0.50, 0.50], [0.18, 0.86], color=COLORS["grid"], linewidth=1.0)

    ax.text(0.57, 0.83, "late specialization", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(
        0.57,
        0.78,
        "shared base prefix, task-specific tail",
        fontsize=7.8,
        color=COLORS["muted"],
    )
    _blocks(ax, 0.57, 0.61, 8, w=0.034, h=0.050)
    ax.text(0.68, 0.56, "document prefix", ha="center", fontsize=7.6, color=COLORS["muted"])
    for idx, (name, color) in enumerate(adapters):
        _box(ax, 0.58 + idx * 0.09, 0.43, 0.065, 0.048, color, name)
    ax.annotate(
        "",
        xy=(0.68, 0.48),
        xytext=(0.71, 0.61),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLORS["muted"]},
    )
    ax.text(0.75, 0.52, "<ADAPTER:task>", fontsize=7.5, color=COLORS["muted"])
    ax.text(
        0.70,
        0.34,
        "prefix memory stays closer to flat",
        ha="center",
        fontsize=8,
        color=COLORS["green"],
    )


def _large_overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
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
            requests=("request_count", "sum"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            slo=("slo_attainment_rate", "mean"),
            qag=("quality_adjusted_goodput", "mean"),
            server_hit=("server_prefix_cache_hit_rate", "mean"),
        )
        .sort_values("sweep_overlap_fraction")
    )


def draw_cache_frontier(ax, df: pd.DataFrame) -> None:
    _panel(
        ax,
        "B. Real 7B cache-locality effect",
        "reset-isolated vLLM runs on one L4",
    )
    summary = _large_overlap_summary(df)
    if summary.empty:
        ax.text(
            0.5,
            0.5,
            "Run large_model_overlap_confidence_vllm",
            ha="center",
            va="center",
        )
        return

    x = summary["sweep_overlap_fraction"] * 100
    ax.plot(
        x,
        summary["p95_ttft_ms"],
        marker="o",
        linewidth=2.6,
        color=COLORS["blue"],
        label="p95 TTFT",
    )
    ax.set_xlabel("shared-prefix overlap (%)", fontsize=8.8)
    ax.set_ylabel("p95 TTFT (ms)", fontsize=8.8, color=COLORS["blue"])
    ax.tick_params(axis="y", colors=COLORS["blue"])
    ax.set_xlim(45, 100)
    _axis_style(ax)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        summary["server_hit"] * 100,
        marker="s",
        linewidth=2.3,
        color=COLORS["teal"],
        label="server prefix hit",
    )
    ax2.set_ylabel("server prefix hit rate (%)", fontsize=8.8, color=COLORS["teal"])
    ax2.tick_params(axis="y", colors=COLORS["teal"], labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["grid"])
    ax2.set_ylim(0, 100)

    low = summary.iloc[0]
    high = summary.iloc[-1]
    delta = low["p95_ttft_ms"] - high["p95_ttft_ms"]
    hit_delta = (high["server_hit"] - low["server_hit"]) * 100
    ax.text(
        0.04,
        0.08,
        f"-{delta:.0f} ms p95 TTFT\n+{hit_delta:.1f} pp prefix hits",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["ink"],
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": COLORS["grid"],
        },
    )


def _source_frontier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "run_id" not in df:
        return pd.DataFrame()
    sub = df[df["run_id"].str.contains("source-eval-expanded-vllm", na=False)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["condition"] = "base"
    sub.loc[sub["run_id"].str.contains("lora-trained", na=False), "condition"] = "specialists"
    sub.loc[sub["run_id"].str.contains("lora-multitask", na=False), "condition"] = "multitask"
    # Keep the higher-throughput H100 confirmation when duplicate condition rows exist.
    return sub.sort_values("quality_adjusted_goodput").groupby("condition", as_index=False).tail(1)


def draw_quality_frontier(ax, df: pd.DataFrame) -> None:
    _panel(
        ax,
        "C. Quality/goodput frontier",
        "expanded source-backed Qwen2.5-7B eval",
    )
    sub = _source_frontier(df)
    if sub.empty:
        ax.text(0.5, 0.5, "Run source-eval-expanded vLLM conditions", ha="center", va="center")
        return
    colors = {"base": COLORS["muted"], "specialists": COLORS["blue"], "multitask": COLORS["teal"]}
    label_offsets = {
        "base": (7, -0.002),
        "specialists": (7, 0.000),
        "multitask": (7, -0.006),
    }
    for _, row in sub.iterrows():
        condition = str(row["condition"])
        dx, dy = label_offsets.get(condition, (7, 0.0))
        ax.scatter(
            row["p95_ttft_ms"],
            row["mean_quality"],
            s=max(90, float(row["quality_adjusted_goodput"]) * 520),
            color=colors.get(condition, COLORS["gold"]),
            alpha=0.86,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.text(
            row["p95_ttft_ms"] + dx,
            row["mean_quality"] + dy,
            condition,
            fontsize=8.2,
            color=COLORS["ink"],
            va="center",
        )
    ax.set_xlabel("p95 TTFT (ms)", fontsize=8.8)
    ax.set_ylabel("mean quality", fontsize=8.8)
    ax.text(
        0.03,
        0.06,
        "bubble area = QAG",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["muted"],
    )
    ax.margins(x=0.18, y=0.18)
    _axis_style(ax)


def draw_capacity_frontier(ax, path: str | Path = "data/results/capacity_frontier.yaml") -> None:
    _panel(ax, "D. Adapter capacity boundary", "startup feasibility at 4096 context")
    try:
        df = load_capacity_frontier(path)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        ax.text(0.5, 0.5, "No capacity records", ha="center", va="center")
        return

    y_positions = []
    for idx, row in enumerate(df.sort_values(["gpu_memory_gb", "lora_count"]).itertuples()):
        y_positions.append(idx)
        color = COLORS["green"] if row.status == "starts" else COLORS["red"]
        ax.scatter(
            row.lora_count,
            idx,
            s=170,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=1.1,
        )
        label = "starts" if row.status == "starts" else "fails"
        ax.text(row.lora_count + 0.18, idx, label, fontsize=8.5, va="center", color=COLORS["ink"])
        if row.status == "starts" and pd.notna(row.available_kv_cache_gib):
            ax.text(
                row.lora_count + 0.18,
                idx - 0.18,
                f"{float(row.available_kv_cache_gib):.1f} GiB KV",
                fontsize=7.4,
                color=COLORS["muted"],
                va="center",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            str(row.condition_id).replace("qwen7b-", "").replace("-loras", " LoRAs")
            for row in df.sort_values(["gpu_memory_gb", "lora_count"]).itertuples()
        ],
        fontsize=7.6,
    )
    ax.set_xlabel("registered LoRAs", fontsize=8.8)
    ax.set_xlim(4.3, 10.9)
    ax.set_ylim(-0.7, len(y_positions) - 0.3)
    ax.axvspan(7.5, 10.5, color="#fff1f1", alpha=0.75, zorder=0)
    ax.text(8.05, len(y_positions) - 0.72, "L4 failure region", fontsize=7.8, color=COLORS["red"])
    _axis_style(ax, ygrid=False)


def draw_takeaway(fig) -> None:
    box = patches.FancyBboxPatch(
        (0.16, 0.035),
        0.70,
        0.050,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor="#f8fafc",
        edgecolor=COLORS["grid"],
        linewidth=0.9,
        transform=fig.transFigure,
    )
    fig.add_artist(box)
    fig.text(
        0.51,
        0.060,
        "Decision rule: specialize when quality gain exceeds SLO loss, cache "
        "fragmentation, and capacity cost.",
        ha="center",
        va="center",
        fontsize=9.3,
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
            "axes.titlepad": 10,
        }
    )
    fig = plt.figure(figsize=(13.4, 8.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        top=0.840,
        bottom=0.175,
        hspace=0.36,
        wspace=0.22,
    )
    draw_mechanism(fig.add_subplot(grid[0, 0]))
    draw_cache_frontier(fig.add_subplot(grid[0, 1]), df)
    draw_quality_frontier(fig.add_subplot(grid[1, 0]), df)
    draw_capacity_frontier(fig.add_subplot(grid[1, 1]))
    draw_takeaway(fig)

    fig.text(
        0.055,
        0.940,
        "When is specialization worth its cache footprint?",
        fontsize=20,
        weight="bold",
        color=COLORS["ink"],
        ha="left",
    )
    fig.text(
        0.055,
        0.902,
        "Specialist adapters buy quality; shared-prefix reuse and adapter capacity "
        "decide whether that quality is serveable.",
        fontsize=10.2,
        color=COLORS["muted"],
        ha="left",
    )
    fig.text(
        0.055,
        0.012,
        "Adapter Cache Tradeoffs | measured vLLM runs plus simulator mechanism; "
        "QAG = quality-adjusted goodput",
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
