"""Matplotlib figures with a consistent, colorblind-checked palette."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Fixed entity -> color mapping (never re-assigned by rank). Baselines are neutral gray.
MODEL_COLORS = {
    "AutoETS": "#2a78d6",
    "AutoARIMA": "#eb6834",
    "LightGBM": "#1baf7a",
    "LightGBM+exog": "#eda100",
    "Ensemble": "#6250d6",
    "Naive": "#8a8984",
    "SeasonalNaive": "#8a8984",
}
MODEL_STYLE = {"SeasonalNaive": "--"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "font.size": 10, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110,
    })


def metric_by_horizon(per_h: pd.DataFrame, metric: str, title: str, ax=None):
    """Line chart of a metric (rows: model x horizon) by forecast horizon."""
    ax = ax or plt.gca()
    for model, g in per_h.reset_index().groupby("model"):
        ax.plot(g["horizon"], g[metric], color=MODEL_COLORS.get(model, INK_2),
                ls=MODEL_STYLE.get(model, "-"), marker="o", ms=4, label=model)
    ax.set_xlabel("Horizon (months ahead)")
    ax.set_ylabel(metric)
    ax.set_xticks(range(1, 13))
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9)
    return ax


def fan_chart(history: pd.Series, fc: pd.DataFrame, title: str, ax=None, color: str = "#2a78d6",
              history_months: int = 72, ylabel: str = "% of household portfolio 90+ days past due"):
    """History plus point forecast and 80/95% bands."""
    ax = ax or plt.gca()
    h = history.iloc[-history_months:]
    ax.plot(h.index, h.values, color=INK, lw=1.8, label="Observed")
    ds = pd.to_datetime(fc["ds"])
    ax.fill_between(ds, fc["lo_95"], fc["hi_95"], color=color, alpha=0.15, lw=0, label="95% interval")
    ax.fill_between(ds, fc["lo_80"], fc["hi_80"], color=color, alpha=0.30, lw=0, label="80% interval")
    ax.plot([h.index[-1], *ds], [h.values[-1], *fc["mean"]], color=color, lw=2, label="Forecast")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    return ax
