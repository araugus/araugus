import numpy as np
import pandas as pd
import pytest

from uplift_campaign.business import CampaignEconomics, bootstrap_profit_gain, optimal_fraction, profit_curve
from uplift_campaign.data import binary_setup, make_features
from uplift_campaign.stats import ate_with_ci, balance_table, standardized_mean_difference


def test_profit_curve_endpoints():
    rng = np.random.default_rng(0)
    n = 5000
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.1 + 0.05 * t)
    econ = CampaignEconomics(value_per_response=3.0, cost_per_contact=0.1, population=10_000)
    curve = profit_curve(y, t, rng.random(n), econ)
    assert curve.iloc[0]["profit"] == 0
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert curve.iloc[-1]["profit"] == pytest.approx(10_000 * (ate * 3.0 - 0.1))
    assert econ.break_even_uplift == pytest.approx(0.1 / 3.0)


def test_targeting_persuadables_beats_blanket(synthetic):
    X, t, y, tau = synthetic
    econ = CampaignEconomics(value_per_response=1.0, cost_per_contact=0.05)
    curve = profit_curve(y, t, tau, econ)
    best = optimal_fraction(curve)
    assert 0.4 <= best["fraction"] <= 0.6  # half of the customers are sleeping dogs
    assert best["profit"] > curve.iloc[-1]["profit"]
    lo, _ = bootstrap_profit_gain(y, t, tau, econ, k=0.5, n_boot=30)
    assert lo > 0


def test_ate_with_ci_known_values():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    res = ate_with_ci(y, t)
    assert res["ate"] == pytest.approx(0.25)
    assert res["ci_low"] < 0.25 < res["ci_high"]


def test_balance_on_identical_groups():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert standardized_mean_difference(x, x) == 0
    X = pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]})
    arm = pd.Series(["c", "c", "t", "t", "u", "u"])
    bal = balance_table(X, arm, control="c")
    assert bal.loc[0, "smd_t"] == 0 and bal.loc[0, "anova_p"] == pytest.approx(1.0)


def test_features_and_binary_setup():
    raw = pd.DataFrame(
        {
            "recency": [1, 5, 9],
            "history": [10.0, 200.0, 50.0],
            "mens": [1, 0, 1],
            "womens": [0, 1, 1],
            "newbie": [1, 0, 0],
            "zip_code": ["Urban", "Rural", "Surburban"],
            "channel": ["Web", "Phone", "Web"],
            "segment": ["Womens E-Mail", "No E-Mail", "Mens E-Mail"],
            "visit": [1, 0, 1],
            "conversion": [0, 0, 1],
            "spend": [0.0, 0.0, 30.0],
        }
    )
    X = make_features(raw)
    assert "zip_code_urban" in X.columns and X.shape == (3, 10)
    d = binary_setup(raw, "Womens E-Mail", "visit")
    assert len(d.y) == 2 and d.treatment.tolist() == [1, 0]
    with pytest.raises(ValueError):
        binary_setup(raw, "Bogus", "visit")
