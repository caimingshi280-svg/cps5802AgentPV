"""Presentation-grade matplotlib style + figure helpers (Component 3 robustness suite).

Centralising rcParams here means every figure (confusion matrix, tradeoff
plot, robustness plots) shares the same look: clean spines, soft dashed
grid, a colour-blind-aware palette, and high-DPI output suitable for
slide decks.

Public API:

* :func:`apply_presentation_style`  — call once before any ``plt.subplots``.
* :data:`PALETTE`                   — primary categorical colours.
* :data:`SEVERITY_COLORS`           — semantic palette for severity / status.
* :func:`make_fig`                  — create a figure with a consistent footer.
* :func:`save_fig`                  — write a high-DPI PNG and close the figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Tableau-10 inspired but tuned for slide projection (high contrast, AA-safe).
PALETTE: tuple[str, ...] = (
    "#2E86AB",   # blue   — baseline / primary
    "#A23B72",   # plum   — degraded
    "#F18F01",   # orange — warning / drift
    "#3B9C5A",   # green  — improvement
    "#5E60CE",   # indigo — secondary system
    "#C73E1D",   # red    — failure
    "#6E6E6E",   # grey   — reference line
)

SEVERITY_COLORS: dict[str, str] = {
    "ok":       "#3B9C5A",
    "monitor":  "#2E86AB",
    "warning":  "#F18F01",
    "critical": "#C73E1D",
}

_STYLE_RC = {
    "figure.figsize":    (8.0, 4.8),
    "figure.dpi":        110,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    1.0,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10.0,
    "axes.labelsize":    11,
    "axes.labelweight":  "regular",
    "axes.labelcolor":   "#222222",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.color":        "#888888",
    "xtick.color":       "#222222",
    "ytick.color":       "#222222",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.frameon":    False,
    "lines.linewidth":   2.0,
    "lines.markersize":  6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}


def apply_presentation_style() -> None:
    """Apply the project's matplotlib style. Safe to call repeatedly."""

    import matplotlib

    matplotlib.use("Agg")  # headless CI / scripts
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE_RC)


def palette(n: int) -> list[str]:
    """Return ``n`` colours from :data:`PALETTE`, cycling if needed."""

    if n <= 0:
        return []
    out: list[str] = []
    for i in range(n):
        out.append(PALETTE[i % len(PALETTE)])
    return out


def annotate_bars(
    ax,
    values: Iterable[float],
    *,
    fmt: str = "{:.3f}",
    fontsize: int = 9,
    color: str = "#222222",
    dy_frac: float = 0.015,
) -> None:
    """Write the numeric value above each bar in ``ax.containers[-1]``."""

    bars = ax.containers[-1]
    ymax = ax.get_ylim()[1]
    dy = ymax * dy_frac
    for bar, value in zip(bars, values, strict=True):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(
            x,
            y + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def save_fig(fig, output_path: Path) -> Path:
    """Save the figure as PNG (200 DPI, tight) and close it."""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
