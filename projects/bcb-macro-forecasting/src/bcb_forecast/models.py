"""Forecasting models with a common output format.

Every forecaster takes the panel *truncated at the forecast origin* and returns a long DataFrame:
``model, ds, horizon, mean, lo_80, hi_80, lo_95, hi_95``.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive

from .data import TARGET
from .features import DEFAULT_EXOG, make_supervised

LEVELS = (80, 95)
SEED = 42
STAT_MODELS = ("Naive", "SeasonalNaive", "AutoETS", "AutoARIMA")
OUTPUT_COLS = ["model", "ds", "horizon", "mean"] + [f"{b}_{lv}" for lv in LEVELS for b in ("lo", "hi")]

LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "n_jobs": 1,
    "verbose": -1,
}


def forecast_stats(y: pd.Series, h: int = 12, models: tuple[str, ...] = STAT_MODELS) -> pd.DataFrame:
    """Naive, seasonal-naive, ETS and ARIMA forecasts (with native prediction intervals)."""
    registry = {
        "Naive": Naive(),
        "SeasonalNaive": SeasonalNaive(season_length=12),
        "AutoETS": AutoETS(season_length=12),
        "AutoARIMA": AutoARIMA(season_length=12),
    }
    df = pd.DataFrame({"unique_id": "y", "ds": y.index, "y": y.to_numpy(dtype=float)})
    sf = StatsForecast(models=[registry[m] for m in models], freq="MS", n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = sf.forecast(df=df, h=h, level=list(LEVELS))
    out = []
    for m in models:
        part = pd.DataFrame({"model": m, "ds": fc["ds"].to_numpy(), "horizon": np.arange(1, h + 1),
                             "mean": fc[m].to_numpy()})
        for lv in LEVELS:
            part[f"lo_{lv}"] = fc[f"{m}-lo-{lv}"].to_numpy()
            part[f"hi_{lv}"] = fc[f"{m}-hi-{lv}"].to_numpy()
        out.append(part)
    return pd.concat(out, ignore_index=True)[OUTPUT_COLS]


def _fit_predict(X: pd.DataFrame, yl: pd.Series, x_new: pd.DataFrame, **kw) -> float:
    model = lgb.LGBMRegressor(**{**LGBM_PARAMS, **kw})
    model.fit(X, yl)
    return float(model.predict(x_new)[0])


def forecast_lgbm(
    panel: pd.DataFrame,
    h: int = 12,
    target: str = TARGET,
    exog: list[str] | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Direct multi-horizon LightGBM: one model per horizon, label = change vs. origin.

    Point forecast uses L2 loss; interval bounds come from separate quantile-loss models
    (alpha = 0.025/0.10/0.90/0.975). Bounds are sorted to avoid quantile crossing.
    ``panel`` must end at the forecast origin (its last row).
    """
    origin = panel.index[-1]
    y_origin = float(panel[target].iloc[-1])
    rows = []
    for k in range(1, h + 1):
        X, label = make_supervised(panel, k, target, exog)
        train = label.notna() & X["y"].notna()
        Xt, yt = X[train], label[train]
        x_new = X.loc[[origin]]
        mean = _fit_predict(Xt, yt, x_new)
        qs = {}
        for lv in LEVELS:
            a = (1 - lv / 100) / 2
            qs[f"lo_{lv}"] = _fit_predict(Xt, yt, x_new, objective="quantile", alpha=a)
            qs[f"hi_{lv}"] = _fit_predict(Xt, yt, x_new, objective="quantile", alpha=1 - a)
        bounds = np.sort([qs["lo_95"], qs["lo_80"], qs["hi_80"], qs["hi_95"]])
        # Make sure the point forecast lies inside the 80% band.
        lo80, hi80 = min(bounds[1], mean), max(bounds[2], mean)
        lo95, hi95 = min(bounds[0], lo80), max(bounds[3], hi80)
        rows.append({
            "model": name or ("LightGBM+exog" if exog else "LightGBM"),
            "ds": origin + pd.DateOffset(months=k),
            "horizon": k,
            "mean": y_origin + mean,
            "lo_80": y_origin + lo80, "hi_80": y_origin + hi80,
            "lo_95": y_origin + lo95, "hi_95": y_origin + hi95,
        })
    return pd.DataFrame(rows)[OUTPUT_COLS]


def forecast_all(panel: pd.DataFrame, h: int = 12, target: str = TARGET,
                 include_lgbm: bool = True) -> pd.DataFrame:
    """Run every model from a single origin (the last row of ``panel``)."""
    y = panel[target].dropna()
    parts = [forecast_stats(y, h)]
    if include_lgbm:
        parts.append(forecast_lgbm(panel, h, target, exog=None))
        parts.append(forecast_lgbm(panel, h, target, exog=DEFAULT_EXOG))
    return pd.concat(parts, ignore_index=True)
