"""Static matplotlib figures for the README (consistent palette, thin marks, recessive axes)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order: colour follows the model, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#8a8985"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
POS, NEG = "#2a78d6", "#d95926"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_ate(ate: pd.DataFrame, path: Path) -> None:
    """Relative lift with 95% CI per arm and outcome."""
    outcomes = list(ate["outcome"].unique())
    fig, axes = plt.subplots(1, len(outcomes), figsize=(10, 3.2))
    for ax, out in zip(axes, outcomes, strict=True):
        sub = ate[ate["outcome"] == out].reset_index(drop=True)
        ypos = np.arange(len(sub))
        for i, r in sub.iterrows():
            ax.errorbar(
                r["ate"],
                i,
                xerr=[[r["ate"] - r["ci_low"]], [r["ci_high"] - r["ate"]]],
                fmt="o",
                color=SERIES[i],
                ms=7,
                capsize=3,
                lw=2,
            )
        ax.axvline(0, color=NEUTRAL, lw=1, ls="--")
        ax.set_yticks(ypos, sub["arm"])
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(f"ATE on {out} (95% CI)", fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    _save(fig, path)


def plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, ylabel: str, path: Path) -> None:
    """Qini / uplift curves with the random-targeting diagonal."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (name, (x, v)) in enumerate(curves.items()):
        ax.plot(x / x[-1], v, color=SERIES[i], lw=2, label=name)
    x, v = next(iter(curves.values()))
    ax.plot([0, 1], [0, v[-1]], color=NEUTRAL, lw=1.5, ls="--", label="Random targeting")
    ax.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, path)


def plot_deciles(dec: pd.DataFrame, overall_ate: float, model_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    colors = [POS if u >= 0 else NEG for u in dec["observed_uplift"]]
    ax.bar(dec["bin"], dec["observed_uplift"] * 100, color=colors, width=0.7)
    ax.errorbar(
        dec["bin"],
        dec["observed_uplift"] * 100,
        yerr=[(dec["observed_uplift"] - dec["ci_low"]) * 100, (dec["ci_high"] - dec["observed_uplift"]) * 100],
        fmt="none",
        ecolor=INK_2,
        lw=1,
        capsize=2,
    )
    ax.plot(dec["bin"], dec["predicted_uplift"] * 100, "o", color=INK, ms=5, label="Mean predicted uplift")
    ax.axhline(overall_ate * 100, color=NEUTRAL, ls="--", lw=1.2, label="Average effect (random targeting)")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(dec["bin"])
    ax.set_xlabel("Decile of predicted uplift (1 = most persuadable)", color=INK_2)
    ax.set_ylabel("Observed uplift on test set (pp)", color=INK_2)
    ax.set_title(f"Uplift by decile - {model_name} (95% CI)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, path)


def plot_profit(curves: dict[str, pd.DataFrame], best: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["fraction"] * 100, c["profit"] / 1000, color=SERIES[i], lw=2, label=name)
    ax.axhline(0, color=NEUTRAL, lw=1.2, ls=":", label="Contact no one")
    ax.axvline(best["fraction"] * 100, color=INK_2, lw=1, ls="--")
    ax.text(
        best["fraction"] * 100 + 1.5,
        0.04,
        f"chosen k = {best['fraction']:.0%}\n(selected on train CV)",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
        color=INK,
        va="bottom",
    )
    ax.set_xlabel("% of customers e-mailed (top-k by predicted uplift)", color=INK_2)
    ax.set_ylabel("Incremental profit, $k per 100k customers", color=INK_2)
    ax.set_title("Expected campaign profit by targeting depth (test set)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    _save(fig, path)


def plot_segments(seg: pd.DataFrame, path: Path) -> None:
    """Model-free uplift by customer segment with CIs."""
    seg = seg.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(seg) + 1.2))
    y = np.arange(len(seg))[::-1]
    colors = [POS if v >= 0 else NEG for v in seg["ate"]]
    ax.barh(y, seg["ate"] * 100, color=colors, height=0.6)
    ax.errorbar(
        seg["ate"] * 100,
        y,
        xerr=[(seg["ate"] - seg["ci_low"]) * 100, (seg["ci_high"] - seg["ate"]) * 100],
        fmt="none",
        ecolor=INK_2,
        lw=1,
        capsize=2,
    )
    ax.set_yticks(
        y, [f"{v}: {lv}".replace("$", r"\$") for v, lv in zip(seg["variable"], seg["level"], strict=True)], fontsize=8
    )
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("Observed uplift in visit rate (pp, 95% CI)", color=INK_2)
    ax.set_title("Who responds to the women's e-mail? (all data, model-free)", fontsize=11, color=INK, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    _save(fig, path)
