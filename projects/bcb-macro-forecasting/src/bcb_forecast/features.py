"""Leakage-safe feature engineering for direct multi-horizon forecasting.

Convention: a forecast is made at *origin* ``t`` = the last month for which the target has been
published. Every feature for origin ``t`` uses only information available at that moment:

* target values up to ``t``;
* each driver up to ``t - availability_lag`` (e.g. IBC-Br is published ~2 weeks after the credit
  statistics, so only its value for ``t - 1`` is known);
* the calendar month of the month being forecast (deterministic).

For horizon ``h`` the label is the *change* ``y[t+h] - y[t]``; predicting changes instead of levels
lets tree models extrapolate beyond the historical range of the level.
"""

from __future__ import annotations

import pandas as pd

from .data import SERIES, TARGET

DEFAULT_EXOG = [
    "selic_target",
    "ipca_12m",
    "lending_rate_hh",
    "unemployment",
    "ibc_br_sa",
    "debt_to_income_hh",
    "concessions_hh",
]


def availability_lag(col: str) -> int:
    """Months a driver lags behind the target at publication time (``ipca_12m`` inherits IPCA's)."""
    base = "ipca_mom" if col == "ipca_12m" else col
    spec = SERIES.get(base)
    return spec.availability_lag if spec is not None else 0


def origin_features(panel: pd.DataFrame, target: str = TARGET, exog: list[str] | None = None) -> pd.DataFrame:
    """Features indexed by forecast origin ``t`` (one row per month)."""
    y = panel[target]
    f = pd.DataFrame(index=panel.index)
    f["y"] = y
    for k in (1, 3, 6, 12):
        f[f"y_diff_{k}"] = y - y.shift(k)
    f["y_dev_ma12"] = y - y.rolling(12).mean()
    f["y_diff_12_lag12"] = y.shift(12) - y.shift(24)  # same month last year: seasonal pattern
    for col in exog or []:
        x = panel[col].shift(availability_lag(col))
        f[f"{col}"] = x
        f[f"{col}_diff_3"] = x - x.shift(3)
        f[f"{col}_diff_12"] = x - x.shift(12)
    return f


def make_supervised(
    panel: pd.DataFrame, horizon: int, target: str = TARGET, exog: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, label)`` for a direct model of the given horizon.

    ``X`` holds every origin (including the latest ones whose label is still unknown -> NaN);
    callers must restrict training rows to origins with ``origin + horizon <= cutoff``.
    """
    X = origin_features(panel, target, exog)
    X["target_month"] = (panel.index + pd.DateOffset(months=horizon)).month
    label = panel[target].shift(-horizon) - panel[target]
    return X, label
