"""Meta-learners for conditional average treatment effect (CATE / uplift) estimation.

Implemented on top of any scikit-learn compatible estimator so they work the
same with LightGBM, logistic regression or anything else. All learners expose
`fit(X, treatment, y)` and `predict_uplift(X)` returning tau(x) = E[Y|X,T=1] - E[Y|X,T=0].

References
----------
* Kunzel, Sekhon, Bickel & Yu (2019), "Metalearners for estimating heterogeneous
  treatment effects using machine learning" (S-, T-, X-learner).
* Jaskowski & Jaroszewicz (2012), "Uplift modeling for clinical trial data"
  (class transformation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

ArrayLike = np.ndarray | pd.DataFrame


def _as_array(X: ArrayLike) -> np.ndarray:
    return X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)


def _proba(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _check(treatment: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    treatment, y = np.asarray(treatment).astype(int), np.asarray(y)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("treatment must contain both 0 (control) and 1 (treated)")
    return treatment, y


class UpliftLearner(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: ArrayLike, treatment: np.ndarray, y: np.ndarray) -> UpliftLearner: ...

    @abstractmethod
    def predict_uplift(self, X: ArrayLike) -> np.ndarray: ...


class SLearner(UpliftLearner):
    """One model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0).

    Simple and data-efficient, but regularisation can shrink the treatment
    effect towards zero when the flag is a weak split candidate.
    """

    name = "S-learner"

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: ArrayLike, treatment: np.ndarray, y: np.ndarray) -> SLearner:
        treatment, y = _check(treatment, y)
        Xa = _as_array(X)
        self.model_ = clone(self.estimator).fit(np.column_stack([Xa, treatment]), y)
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        Xa = _as_array(X)
        ones, zeros = np.ones(len(Xa)), np.zeros(len(Xa))
        return _proba(self.model_, np.column_stack([Xa, ones])) - _proba(self.model_, np.column_stack([Xa, zeros]))


class TLearner(UpliftLearner):
    """Separate response models for treated and control; uplift = mu1(x) - mu0(x)."""

    name = "T-learner"

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: ArrayLike, treatment: np.ndarray, y: np.ndarray) -> TLearner:
        treatment, y = _check(treatment, y)
        Xa = _as_array(X)
        self.model_t_ = clone(self.estimator).fit(Xa[treatment == 1], y[treatment == 1])
        self.model_c_ = clone(self.estimator).fit(Xa[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        Xa = _as_array(X)
        return _proba(self.model_t_, Xa) - _proba(self.model_c_, Xa)


class XLearner(UpliftLearner):
    """X-learner: T-learner first stage, then regress imputed individual effects.

    D1 = y - mu0(x) on treated units, D0 = mu1(x) - y on control units; tau1 and
    tau0 are fitted on D1 and D0 and combined with the propensity score g:
    tau(x) = g * tau0(x) + (1 - g) * tau1(x). In a randomised test g is the
    (constant) share of treated units, so no propensity model is needed.
    """

    name = "X-learner"

    def __init__(self, outcome_estimator: BaseEstimator, effect_estimator: BaseEstimator) -> None:
        self.outcome_estimator = outcome_estimator
        self.effect_estimator = effect_estimator

    def fit(self, X: ArrayLike, treatment: np.ndarray, y: np.ndarray) -> XLearner:
        treatment, y = _check(treatment, y)
        Xa = _as_array(X)
        Xt, yt, Xc, yc = Xa[treatment == 1], y[treatment == 1], Xa[treatment == 0], y[treatment == 0]
        mu1 = clone(self.outcome_estimator).fit(Xt, yt)
        mu0 = clone(self.outcome_estimator).fit(Xc, yc)
        d1 = yt - _proba(mu0, Xt)
        d0 = _proba(mu1, Xc) - yc
        self.tau1_ = clone(self.effect_estimator).fit(Xt, d1)
        self.tau0_ = clone(self.effect_estimator).fit(Xc, d0)
        self.propensity_ = float(treatment.mean())
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        Xa = _as_array(X)
        g = self.propensity_
        return g * self.tau0_.predict(Xa) + (1 - g) * self.tau1_.predict(Xa)


class ClassTransformation(UpliftLearner):
    """Class transformation (Jaskowski & Jaroszewicz, 2012).

    Z = 1 if (T=1, Y=1) or (T=0, Y=0). With balanced arms, uplift = 2 P(Z=1|x) - 1.
    Arms are re-weighted to 50/50 so the identity holds for any treated share.
    """

    name = "Class transformation"

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: ArrayLike, treatment: np.ndarray, y: np.ndarray) -> ClassTransformation:
        treatment, y = _check(treatment, y)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("ClassTransformation requires a binary outcome")
        p = treatment.mean()
        z = (y == treatment).astype(int)
        weights = np.where(treatment == 1, 0.5 / p, 0.5 / (1 - p))
        self.model_ = clone(self.estimator).fit(_as_array(X), z, sample_weight=weights)
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        return 2 * _proba(self.model_, _as_array(X)) - 1
