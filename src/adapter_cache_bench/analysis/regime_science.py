from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from adapter_cache_bench.analysis.plot_style import (
    COLORS,
    apply_dark_theme,
    save_figure,
    style_axis,
)
from adapter_cache_bench.analysis.policy_regret import build_policy_regret_table

REGIME_WORKLOAD_ORDER = [
    "regime_uniform",
    "regime_zipfian",
    "regime_bursty_session",
    "regime_phase_shift",
    "regime_adversarial_churn",
]


def _policy_label(row: pd.Series) -> str:
    strategy = str(row.get("strategy", ""))
    router = str(row.get("router_policy", ""))
    cache = str(row.get("cache_model", ""))
    if strategy == "oracle" or router == "oracle":
        return "oracle"
    if router and cache:
        return f"{router} / {cache}"
    return str(row.get("policy", "unknown"))


def _display_label(value: object) -> str:
    text = str(value)
    if text.startswith("regime_"):
        text = text.removeprefix("regime_")
    return text.replace("_", " ")


def _sort_workloads(workloads: list[str]) -> list[str]:
    known = [workload for workload in REGIME_WORKLOAD_ORDER if workload in workloads]
    remaining = sorted(workload for workload in workloads if workload not in set(known))
    return [*known, *remaining]


def build_regime_policy_failure_matrix(regret_table: pd.DataFrame) -> pd.DataFrame:
    required = {"workload", "relative_regret", "router_policy", "cache_model", "strategy"}
    if regret_table.empty or not required <= set(regret_table.columns):
        return pd.DataFrame()

    data = regret_table.copy()
    data = data[data["workload"].astype(str).str.startswith("regime_")]
    if data.empty:
        return pd.DataFrame()

    data["policy_label"] = data.apply(_policy_label, axis=1)
    data["relative_regret"] = pd.to_numeric(data["relative_regret"], errors="coerce")
    data = data[data["relative_regret"].notna()]
    if data.empty:
        return pd.DataFrame()

    matrix = data.pivot_table(
        index="workload",
        columns="policy_label",
        values="relative_regret",
        aggfunc="mean",
    )
    workload_order = _sort_workloads([str(value) for value in matrix.index])
    policy_order = (
        matrix.mean(axis=0)
        .sort_values(kind="stable")
        .index.to_series()
        .sort_values(key=lambda values: values.ne("oracle"), kind="stable")
        .tolist()
    )
    return matrix.reindex(index=workload_order, columns=policy_order)


def write_regime_policy_failure_map(
    runs_dir: str | Path = "artifacts/runs",
    output: str | Path = "reports/figures/regime_policy_failure_map.png",
    *,
    regret_table: pd.DataFrame | None = None,
) -> Path | None:
    table = regret_table if regret_table is not None else build_policy_regret_table(runs_dir)
    matrix = build_regime_policy_failure_matrix(table)
    if matrix.empty:
        return None

    apply_dark_theme()
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    vmax = float(matrix.stack().quantile(0.95)) if matrix.stack().size else 0.0
    vmax = max(0.02, min(vmax, 0.5))
    cmap = LinearSegmentedColormap.from_list(
        "regime_regret",
        [COLORS["panel_alt"], COLORS["teal"], COLORS["amber"], COLORS["rose"]],
    ).with_extremes(bad=COLORS["panel"])

    width = max(8.4, 0.78 * len(matrix.columns) + 2.8)
    height = max(4.6, 0.55 * len(matrix.index) + 2.1)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(
        [_display_label(label).replace(" / ", "\n") for label in matrix.columns],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([_display_label(label) for label in matrix.index])
    ax.set_title("Policy regret by workload regime", loc="left", pad=12)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Workload regime")
    style_axis(ax, xgrid=False, ygrid=False)
    ax.tick_params(axis="x", labelsize=7.2, pad=8)
    ax.tick_params(axis="y", labelsize=8.5)

    for row_index, (_, row) in enumerate(matrix.iterrows()):
        for column_index, value in enumerate(row):
            if pd.isna(value):
                label = "-"
                color = COLORS["faint"]
            else:
                label = f"{value * 100:.1f}%"
                color = COLORS["text"] if value < vmax * 0.72 else COLORS["background"]
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=7.1,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.025)
    cbar.set_label("Relative regret vs best observed QAG", color=COLORS["muted"])
    cbar.ax.tick_params(colors=COLORS["muted"], labelsize=7.5)
    cbar.outline.set_edgecolor(COLORS["spine"])
    return save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--output", default="reports/figures/regime_policy_failure_map.png")
    args = parser.parse_args()
    path = write_regime_policy_failure_map(args.runs_dir, args.output)
    if path is not None:
        print(path)


if __name__ == "__main__":
    main()
