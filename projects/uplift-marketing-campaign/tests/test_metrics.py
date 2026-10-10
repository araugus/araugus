import numpy as np
import pytest

from uplift_campaign.metrics import (
    auuc,
    bootstrap_metric,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_by_bin,
    uplift_curve,
)


@pytest.fixture
def random_data():
    rng = np.random.default_rng(1)
    n = 3000
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.2 + 0.1 * t)
    score = rng.normal(size=n)
    return y, t, score


def test_matches_scikit_uplift(random_data):
    sk = pytest.importorskip("sklift.metrics")
    y, t, s = random_data
    assert qini_coefficient(y, t, s) == pytest.approx(sk.qini_auc_score(y, s, t), abs=1e-10)
    assert auuc(y, t, s) == pytest.approx(sk.uplift_auc_score(y, s, t), abs=1e-10)
    pairs = [(qini_curve(y, t, s), sk.qini_curve(y, s, t)), (uplift_curve(y, t, s), sk.uplift_curve(y, s, t))]
    for ours, theirs in pairs:
        np.testing.assert_allclose(ours[0], theirs[0])
        np.testing.assert_allclose(ours[1], theirs[1])


def test_qini_curve_endpoints(random_data):
    y, t, s = random_data
    x, q = qini_curve(y, t, s)
    assert x[0] == 0 and q[0] == 0 and x[-1] == len(y)
    # Final value = treated responders - control responders rescaled to the treated group size.
    expected = y[t == 1].sum() - y[t == 0].sum() * (t == 1).sum() / (t == 0).sum()
    assert q[-1] == pytest.approx(expected)


def test_perfect_and_reversed_rankings():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.uniform(-1, 1, n)
    tau = np.where(x > 0, 0.3, -0.1)
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.3 + tau * t)
    assert qini_coefficient(y, t, y * t - y * (1 - t)) == pytest.approx(1.0)
    good, bad = qini_coefficient(y, t, x), qini_coefficient(y, t, -x)
    assert good > 0.1 and bad < -0.1
    assert auuc(y, t, x) > 0 > auuc(y, t, -x)


def test_random_score_is_near_zero():
    rng = np.random.default_rng(3)
    n = 20000
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.2 + 0.05 * t)
    vals = [qini_coefficient(y, t, rng.random(n)) for _ in range(5)]
    assert abs(np.mean(vals)) < 0.05


def test_uplift_by_bin_and_at_k(random_data):
    y, t, s = random_data
    dec = uplift_by_bin(y, t, s, n_bins=10)
    assert dec["n"].sum() == len(y) and len(dec) == 10
    assert (dec["predicted_uplift"].diff().dropna() <= 0).all()  # bin 1 = highest score
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert uplift_at_k(y, t, s, 1.0) == pytest.approx(ate)


def test_bootstrap_interval_contains_point(random_data):
    y, t, s = random_data
    lo, hi = bootstrap_metric(qini_coefficient, y, t, s, n_boot=50, seed=0)
    assert lo <= qini_coefficient(y, t, s) <= hi
