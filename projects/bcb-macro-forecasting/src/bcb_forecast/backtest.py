"""Rolling-origin (expanding window) backtesting and evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .data import TARGET

Forecaster = Callable[[pd.DataFrame], pd.DataFrame]


def make_cutoffs(index: pd.DatetimeIndex, first: str, horizon: int, step: int = 3) -> list[pd.Timestamp]:
    """Forecast origins from ``first`` every ``step`` months, keeping only origins whose full
    ``horizon`` is observed (so every cutoff contributes to every horizon)."""
    last_allowed = index.max() - pd.DateOffset(months=horizon)
    cutoffs = pd.date_range(pd.Timestamp(first), last_allowed, freq=f"{step}MS")
    return [c for c in cutoffs if c in index]


def mase_scale(y_train: pd.Series) -> float:
    """In-sample MAE of the one-step naive forecast (the MASE denominator)."""
    return float(np.mean(np.abs(np.diff(y_train.dropna().to_numpy()))))


def rolling_origin(
    panel: pd.DataFrame,
    forecaster: Forecaster,
    cutoffs: list[pd.Timestamp],
    horizon: int = 12,
    target: str = TARGET,
) -> pd.DataFrame:
    """Expanding-window backtest.

    For each cutoff the forecaster only sees ``panel.loc[:cutoff]``: data after the origin is
    physically removed, which rules out look-ahead leakage by construction.
    """
    results = []
    for cutoff in cutoffs:
        hist = panel.loc[:cutoff]
        fc = forecaster(hist)
        fc = fc[fc["horizon"] <= horizon].copy()
        fc["cutoff"] = cutoff
        fc["actual"] = panel[target].reindex(pd.DatetimeIndex(fc["ds"])).to_numpy()
        fc["scale"] = mase_scale(hist[target])
        results.append(fc)
    return pd.concat(results, ignore_index=True)


def add_errors(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt.copy()
    bt["error"] = bt["actual"] - bt["mean"]
    bt["abs_error"] = bt["error"].abs()
    bt["ape"] = bt["abs_error"] / bt["actual"].abs() * 100
    bt["scaled_abs_error"] = bt["abs_error"] / bt["scale"]
    for lv in (80, 95):
        if f"lo_{lv}" in bt:
            covered = (bt["actual"] >= bt[f"lo_{lv}"]) & (bt["actual"] <= bt[f"hi_{lv}"])
            # NaN bounds (e.g. conformal warm-up) -> excluded from coverage, not counted as misses
            bt[f"covered_{lv}"] = covered.astype(float).where(bt[f"lo_{lv}"].notna())
            bt[f"width_{lv}"] = bt[f"hi_{lv}"] - bt[f"lo_{lv}"]
    return bt


def summarize(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """MAE, RMSE, MAPE, MASE and empirical interval coverage, grouped by ``by``."""
    by = by or ["model"]
    bt = add_errors(bt.dropna(subset=["actual"]))
    agg = bt.groupby(by).agg(
        MAE=("abs_error", "mean"),
        RMSE=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
        MAPE=("ape", "mean"),
        MASE=("scaled_abs_error", "mean"),
        bias=("error", "mean"),
        coverage_80=("covered_80", "mean"),
        coverage_95=("covered_95", "mean"),
        width_80=("width_80", "mean"),
        n=("error", "size"),
        n_intervals=("covered_80", "count"),
    )
    return agg


def skill_vs(summary: pd.DataFrame, baseline: str, metric: str = "MAE") -> pd.Series:
    """Relative improvement of each model over ``baseline`` (positive = better)."""
    base = summary.loc[baseline, metric]
    return 1 - summary[metric] / base


def add_ensemble(bt: pd.DataFrame, members: list[str], name: str = "Ensemble") -> pd.DataFrame:
    """Append an equal-weight average of ``members``' point forecasts (intervals left empty:
    they are produced by :func:`conformalize`)."""
    sub = bt[bt["model"].isin(members)]
    keys = ["cutoff", "ds", "horizon"]
    ens = sub.groupby(keys, as_index=False).agg(mean=("mean", "mean"), actual=("actual", "first"),
                                                scale=("scale", "first"), n=("model", "nunique"))
    ens = ens[ens["n"] == len(members)].drop(columns="n")
    ens["model"] = name
    return pd.concat([bt, ens], ignore_index=True)


def conformal_quantile(abs_errors: np.ndarray, level: float) -> float:
    """Finite-sample-corrected empirical quantile of absolute errors (split-conformal style)."""
    e = np.sort(np.asarray(abs_errors, dtype=float))
    k = int(np.ceil((len(e) + 1) * level))  # k-th smallest error; capped at the maximum
    return float(e[min(k, len(e)) - 1])


def conformalize(bt: pd.DataFrame, levels: tuple[int, ...] = (80, 95), min_history: int = 8) -> pd.DataFrame:
    """Replace interval bounds with *online* conformal bands: ``mean +/- q``.

    For a forecast made at ``cutoff``, ``q`` is the empirical quantile of the absolute errors of
    the same model and horizon from earlier forecasts whose target month is ``<= cutoff``
    (i.e. errors that were already observable at forecast time). Rows with fewer than
    ``min_history`` past errors get NaN bounds and are excluded from coverage statistics.
    """
    out = bt.copy()
    for lv in levels:
        out[f"lo_{lv}"] = np.nan
        out[f"hi_{lv}"] = np.nan
    ds = pd.to_datetime(out["ds"])
    cut = pd.to_datetime(out["cutoff"])
    abs_err = (out["actual"] - out["mean"]).abs()
    for (_, _), idx in out.groupby(["model", "horizon"]).groups.items():
        for i in idx:
            past = abs_err[idx][(ds[idx] <= cut[i]).to_numpy()].dropna().to_numpy()
            if len(past) < min_history:
                continue
            for lv in levels:
                q = conformal_quantile(past, lv / 100)
                out.at[i, f"lo_{lv}"] = out.at[i, "mean"] - q
                out.at[i, f"hi_{lv}"] = out.at[i, "mean"] + q
    return out


def final_conformal_bands(bt: pd.DataFrame, model: str, levels: tuple[int, ...] = (80, 95)) -> pd.DataFrame:
    """Per-horizon half-widths from *all* backtest errors of ``model`` (used for the live forecast)."""
    sub = bt[(bt["model"] == model) & bt["actual"].notna()]
    rows = []
    for h, g in sub.groupby("horizon"):
        e = (g["actual"] - g["mean"]).abs().to_numpy()
        rows.append({"horizon": h, **{f"q_{lv}": conformal_quantile(e, lv / 100) for lv in levels}})
    return pd.DataFrame(rows)
