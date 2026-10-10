import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr

from uplift_campaign.learners import ClassTransformation, SLearner, TLearner, XLearner

PARAMS = dict(
    n_estimators=100, learning_rate=0.05, num_leaves=4, min_child_samples=50, random_state=0, verbose=-1, n_jobs=1
)


def _learners():
    return [
        SLearner(LGBMClassifier(**PARAMS)),
        TLearner(LGBMClassifier(**PARAMS)),
        XLearner(LGBMClassifier(**PARAMS), LGBMRegressor(**PARAMS)),
        ClassTransformation(LGBMClassifier(**PARAMS)),
    ]


@pytest.mark.parametrize("learner", _learners(), ids=lambda m: m.name)
def test_learner_recovers_heterogeneous_effect(synthetic, learner):
    X, t, y, tau = synthetic
    pred = learner.fit(X, t, y).predict_uplift(X)
    # The sign of the effect is recovered for the vast majority of customers ...
    assert np.mean(np.sign(pred) == np.sign(tau)) > 0.85
    # ... the ranking is informative ...
    assert spearmanr(pred, tau).statistic > 0.6
    # ... and group averages are close to the truth. The S-learner is known to shrink
    # effects towards zero (the treatment flag competes with other splits), so it gets
    # a looser tolerance.
    tol = 0.1 if learner.name == "S-learner" else 0.07
    assert abs(pred[tau > 0].mean() - 0.25) < tol
    assert abs(pred[tau < 0].mean() + 0.15) < tol


def test_class_transformation_handles_unbalanced_arms(synthetic):
    X, t, y, tau = synthetic
    keep = (t == 0) | (np.arange(len(t)) % 3 == 0)  # roughly 25% treated
    model = ClassTransformation(LGBMClassifier(**PARAMS)).fit(X[keep], t[keep], y[keep])
    pred = model.predict_uplift(X)
    assert abs(pred[tau > 0].mean() - 0.25) < 0.1
    assert abs(pred[tau < 0].mean() + 0.15) < 0.1


def test_learners_reject_single_arm(synthetic):
    X, _, y, _ = synthetic
    with pytest.raises(ValueError):
        TLearner(LGBMClassifier(**PARAMS)).fit(X, np.ones(len(y), dtype=int), y)
