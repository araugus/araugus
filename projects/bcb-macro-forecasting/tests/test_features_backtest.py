from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bcb_forecast.backtest import (
    add_ensemble,
    conformal_quantile,
    conformalize,
    make_cutoffs,
    mase_scale,
    rolling_origin,
    summarize,
)
from bcb_forecast.features import make_supervised, origin_features
from bcb_forecast.models import OUTPUT_COLS, forecast_lgbm, forecast_stats

TARGET = "default_rate_hh"


# ----------------------------------------------------------------------------- features
def test_features_do_not_depend_on_future_values(synthetic_panel):
    t = synthetic_panel.index[60]
    base = origin_features(synthetic_panel, exog=["selic_target", "ibc_br_sa"])
    shocked = synthetic_panel.copy()
    shocked.loc[shocked.index > t] *= 100  # wreck the future
    after = origin_features(shocked, exog=["selic_target", "ibc_br_sa"])
    pd.testing.assert_series_equal(base.loc[t], after.loc[t])


def test_availability_lag_is_applied(synthetic_panel):
    f = origin_features(synthetic_panel, exog=["selic_target", "ibc_br_sa"])
    t, prev = synthetic_panel.index[30], synthetic_panel.index[29]
    assert f.loc[t, "selic_target"] == synthetic_panel.loc[t, "selic_target"]  # lag 0
    assert f.loc[t, "ibc_br_sa"] == synthetic_panel.loc[prev, "ibc_br_sa"]  # published one month later


def test_label_is_future_change(synthetic_panel):
    X, label = make_supervised(synthetic_panel, 3)
    y = synthetic_panel[TARGET]
    assert label.iloc[10] == pytest.approx(y.iloc[13] - y.iloc[10])
    assert label.iloc[-3:].isna().all()
    assert X["target_month"].iloc[0] == 4  # Jan origin + 3 months -> April


# ----------------------------------------------------------------------------- backtest
def test_make_cutoffs_keep_full_horizon(synthetic_panel):
    cuts = make_cutoffs(synthetic_panel.index, "2014-01-01", horizon=12, step=6)
    assert cuts[0] == pd.Timestamp("2014-01-01")
    assert cuts[-1] <= synthetic_panel.index.max() - pd.DateOffset(months=12)
    assert all((b - a).days > 150 for a, b in zip(cuts[:-1], cuts[1:], strict=True))


def test_rolling_origin_never_shows_future_data(synthetic_panel):
    seen = []

    def spy_forecaster(hist: pd.DataFrame) -> pd.DataFrame:
        seen.append(hist.index.max())
        last = hist[TARGET].iloc[-1]
        ds = pd.date_range(hist.index[-1], periods=4, freq="MS")[1:]
        return pd.DataFrame({"model": "spy", "ds": ds, "horizon": [1, 2, 3], "mean": last,
                             "lo_80": last - 1, "hi_80": last + 1, "lo_95": last - 2, "hi_95": last + 2})

    cuts = make_cutoffs(synthetic_panel.index, "2015-01-01", horizon=3, step=12)
    bt = rolling_origin(synthetic_panel, spy_forecaster, cuts, horizon=3)
    assert seen == cuts
    assert len(bt) == 3 * len(cuts)
    row = bt.iloc[1]
    assert row["actual"] == synthetic_panel.loc[row["ds"], TARGET]


def test_metrics_on_hand_computed_example():
    bt = pd.DataFrame({
        "model": ["a", "a", "b", "b"], "horizon": [1, 1, 1, 1],
        "mean": [1.0, 2.0, 1.0, 1.0], "actual": [2.0, 2.0, 2.0, 4.0], "scale": [0.5] * 4,
        "lo_80": [0.0, 1.5, 0.0, 0.0], "hi_80": [3.0, 2.5, 1.5, 1.5],
        "lo_95": [0.0, 1.0, 0.0, 0.0], "hi_95": [3.0, 3.0, 3.0, 3.0],
    })
    s = summarize(bt)
    assert s.loc["a", "MAE"] == pytest.approx(0.5)
    assert s.loc["b", "RMSE"] == pytest.approx(np.sqrt((1 + 9) / 2))
    assert s.loc["a", "MAPE"] == pytest.approx(25.0)
    assert s.loc["a", "MASE"] == pytest.approx(1.0)
    assert s.loc["a", "coverage_80"] == 1.0 and s.loc["b", "coverage_80"] == 0.0
    assert s.loc["b", "coverage_95"] == 0.5


def test_mase_scale_is_mean_abs_first_difference():
    assert mase_scale(pd.Series([1.0, 2.0, 4.0, 3.0])) == pytest.approx((1 + 2 + 1) / 3)


def test_conformal_quantile_finite_sample():
    e = np.arange(1, 11, dtype=float)  # 10 errors
    assert conformal_quantile(e, 0.8) == 9.0  # ceil(11 * 0.8) = 9th smallest
    assert conformal_quantile(e, 0.95) == 10.0


def test_conformalize_uses_only_errors_observed_before_cutoff():
    cut = pd.date_range("2020-01-01", periods=12, freq="MS")
    bt = pd.DataFrame({"model": "m", "horizon": 1, "cutoff": cut, "ds": cut + pd.DateOffset(months=1),
                       "mean": 0.0, "actual": np.r_[np.ones(11), 100.0]})
    out = conformalize(bt, min_history=3)
    # First rows: not enough past errors -> no interval
    assert out["lo_80"].iloc[:3].isna().all()
    # Row 3 (cutoff Apr-20) may use errors whose target month <= Apr-20: rows 0..2 -> |e| = 1
    assert out.loc[3, "hi_80"] == pytest.approx(1.0)
    # The huge last error never leaks into any band (its target month is after every cutoff)
    assert out["hi_95"].max() == pytest.approx(1.0)


def test_add_ensemble_averages_members():
    base = pd.DataFrame({"model": ["a", "b"], "cutoff": ["2020-01-01"] * 2, "ds": ["2020-02-01"] * 2,
                         "horizon": [1, 1], "mean": [1.0, 3.0], "actual": [2.0, 2.0], "scale": [1.0, 1.0]})
    out = add_ensemble(base, ["a", "b"], "E")
    assert out.loc[out["model"] == "E", "mean"].item() == 2.0


# ----------------------------------------------------------------------------- models
def test_forecast_lgbm_output_contract(synthetic_panel):
    fc = forecast_lgbm(synthetic_panel, h=3, exog=["selic_target", "ibc_br_sa"])
    assert list(fc.columns) == OUTPUT_COLS
    assert fc["horizon"].tolist() == [1, 2, 3]
    assert fc["ds"].iloc[0] == synthetic_panel.index[-1] + pd.DateOffset(months=1)
    assert (fc["lo_95"] <= fc["lo_80"]).all() and (fc["lo_80"] <= fc["mean"]).all()
    assert (fc["mean"] <= fc["hi_80"]).all() and (fc["hi_80"] <= fc["hi_95"]).all()


def test_forecast_stats_naive_is_last_value(synthetic_panel):
    y = synthetic_panel[TARGET]
    fc = forecast_stats(y, h=4, models=("Naive", "SeasonalNaive"))
    naive = fc[fc["model"] == "Naive"]
    assert np.allclose(naive["mean"], y.iloc[-1])
    snaive = fc[fc["model"] == "SeasonalNaive"]
    assert np.allclose(snaive["mean"], y.iloc[-12:-8])
    assert (naive["hi_95"] > naive["hi_80"]).all()
