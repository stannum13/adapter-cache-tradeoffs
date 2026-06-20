from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COLORS = {
    "background": "#090B10",
    "panel": "#11141A",
    "panel_alt": "#151922",
    "text": "#E7EAF0",
    "muted": "#9AA4B2",
    "faint": "#6F7887",
    "grid": "#2A303A",
    "spine": "#353C49",
    "blue": "#86A8FF",
    "teal": "#67D4C0",
    "amber": "#D7B96F",
    "rose": "#D97E7E",
    "green": "#8BD69A",
    "violet": "#B4A5FF",
    "slate": "#7F8EA3",
}

PALETTE = (
    COLORS["blue"],
    COLORS["teal"],
    COLORS["amber"],
    COLORS["violet"],
    COLORS["green"],
    COLORS["rose"],
    COLORS["slate"],
)


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["spine"],
            "axes.labelcolor": COLORS["muted"],
            "axes.titlecolor": COLORS["text"],
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.facecolor": COLORS["panel"],
            "legend.edgecolor": COLORS["grid"],
            "legend.labelcolor": COLORS["muted"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def style_axis(ax: Axes, *, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8.5)
    ax.title.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    ax.grid(False)
    if xgrid:
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.62)
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.62)
    ax.set_axisbelow(True)


def style_legend(ax: Axes, **kwargs) -> None:
    legend = ax.legend(frameon=False, labelcolor=COLORS["muted"], **kwargs)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(COLORS["muted"])


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 220) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=COLORS["background"])
    plt.close(fig)
    return out
