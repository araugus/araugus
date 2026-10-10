"""Model zoo, cross-validated selection and an interpretable surrogate of the uplift scores."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor, export_text

from .learners import ClassTransformation, SLearner, TLearner, UpliftLearner, XLearner
from .metrics import qini_coefficient

SEED = 42

# Deliberately conservative trees: the signal (a few points of uplift) is small
# relative to the noise, so heavy regularisation beats capacity here.
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=8,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=1,
)


def make_learners() -> dict[str, Callable[[], UpliftLearner]]:
    """Factories so that each CV fold gets a fresh, unfitted learner."""
    clf = lambda: LGBMClassifier(**LGBM_PARAMS)  # noqa: E731
    reg = lambda: LGBMRegressor(**LGBM_PARAMS)  # noqa: E731
    return {
        "S-learner": lambda: SLearner(clf()),
        "T-learner": lambda: TLearner(clf()),
        "X-learner": lambda: XLearner(clf(), reg()),
        "Class transformation": lambda: ClassTransformation(clf()),
    }


def stratum(treatment: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Joint treatment x outcome label used to stratify splits."""
    return 2 * np.asarray(treatment, int) + (np.asarray(y) > 0).astype(int)


def out_of_fold_uplift(
    factory: Callable[[], UpliftLearner],
    X: pd.DataFrame,
    treatment: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    """Out-of-fold uplift predictions on the training set (used for model and k selection)."""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, stratum(treatment, y)):
        model = factory().fit(X.iloc[tr], treatment[tr], y[tr])
        oof[va] = model.predict_uplift(X.iloc[va])
    return oof


def cv_qini(oof: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> float:
    return qini_coefficient(y, treatment, oof)


def surrogate_tree(
    X: pd.DataFrame, uplift: np.ndarray, max_depth: int = 3, min_samples_leaf: int = 1000
) -> tuple[DecisionTreeRegressor, str]:
    """Shallow regression tree fitted to predicted uplift -> human-readable segment rules."""
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED).fit(
        X, uplift
    )
    return tree, export_text(tree, feature_names=list(X.columns), decimals=3)


def leaf_rules(tree: DecisionTreeRegressor, feature_names: list[str]) -> dict[int, str]:
    """Readable conjunction of conditions for every leaf of a fitted tree."""
    t = tree.tree_
    rules: dict[int, str] = {}

    def walk(node: int, conds: list[str]) -> None:
        if t.children_left[node] == -1:
            rules[node] = " AND ".join(conds) if conds else "all customers"
            return
        name, thr = feature_names[t.feature[node]], t.threshold[node]
        if thr == 0.5:  # split on a 0/1 dummy (all binary features split at 0.5)
            left, right = f"{name} = 0", f"{name} = 1"
        else:
            left, right = f"{name} <= {thr:.1f}", f"{name} > {thr:.1f}"
        walk(t.children_left[node], conds + [left])
        walk(t.children_right[node], conds + [right])

    walk(0, [])
    return rules
