"""Uplift evaluation metrics: Qini / uplift curves, normalised areas, uplift by decile.

Uplift cannot be observed per customer (we never see both potential outcomes),
so every metric compares treated vs control *groups* among the customers
ranked highest by the model. Curve definitions follow Radcliffe (2007) and
match scikit-uplift's conventions (verified in the test-suite).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cumulative_counts(
    y: np.ndarray, treatment: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative counts after sorting by descending score, at each distinct score threshold."""
    y, treatment, score = np.asarray(y, float), np.asarray(treatment, int), np.asarray(score, float)
    order = np.argsort(score, kind="mergesort")[::-1]
    y, treatment, score = y[order], treatment[order], score[order]
    thresholds = np.r_[np.where(np.diff(score))[0], score.size - 1]
    n_t = np.cumsum(treatment)[thresholds].astype(float)
    y_t = np.cumsum(y * treatment)[thresholds]
    y_c = np.cumsum(y * (1 - treatment))[thresholds]
    n_all = (thresholds + 1).astype(float)
    return n_all, n_t, n_all - n_t, y_t, y_c


def _prepend_origin(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or x[0] != 0 or v[0] != 0:
        return np.r_[0.0, x], np.r_[0.0, v]
    return x, v


def qini_curve(y, treatment, score) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: incremental responders among the top-n, Q(n) = Y_t(n) - Y_c(n) * N_t(n) / N_c(n)."""
    n_all, n_t, n_c, y_t, y_c = _cumulative_counts(y, treatment, score)
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c != 0)
    return _prepend_origin(n_all, y_t - y_c * ratio)


def uplift_curve(y, treatment, score) -> tuple[np.ndarray, np.ndarray]:
    """Uplift (gain) curve: U(n) = (Y_t/N_t - Y_c/N_c) * n, i.e. incremental responders if the top-n were treated."""
    n_all, n_t, n_c, y_t, y_c = _cumulative_counts(y, treatment, score)
    rate_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t != 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c != 0)
    return _prepend_origin(n_all, (rate_t - rate_c) * n_all)


def _area_above_random(x: np.ndarray, v: np.ndarray) -> float:
    return float(np.trapezoid(v, x) - 0.5 * x[-1] * v[-1])


def qini_coefficient(y, treatment, score) -> float:
    """Normalised Qini coefficient: area between model and random Qini curves divided by the
    same area for the perfect ranking (negative effects allowed). 0 = random, 1 = perfect."""
    y, treatment = np.asarray(y), np.asarray(treatment)
    perfect_score = y * treatment - y * (1 - treatment)
    model = _area_above_random(*qini_curve(y, treatment, score))
    perfect = _area_above_random(*qini_curve(y, treatment, perfect_score))
    return model / perfect


def auuc(y, treatment, score) -> float:
    """Normalised area under the uplift curve (same normalisation idea as `qini_coefficient`)."""
    y, treatment = np.asarray(y), np.asarray(treatment)
    cr = np.sum((y == 1) & (treatment == 0))
    tn = np.sum((y == 0) & (treatment == 1))
    summand = y if cr > tn else treatment
    perfect_score = 2 * (y == treatment) + summand
    model = _area_above_random(*uplift_curve(y, treatment, score))
    perfect = _area_above_random(*uplift_curve(y, treatment, perfect_score))
    return model / perfect


def uplift_at_k(y, treatment, score, k: float) -> float:
    """Observed uplift (treated rate - control rate) among the top-k fraction by score."""
    y, treatment, score = np.asarray(y, float), np.asarray(treatment), np.asarray(score)
    top = np.argsort(-score, kind="mergesort")[: max(1, int(np.ceil(k * len(score))))]
    yt, tt = y[top], treatment[top]
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def uplift_by_bin(y, treatment, score, n_bins: int = 10) -> pd.DataFrame:
    """Observed uplift per score bin (bin 1 = highest predicted uplift) with a 95% normal CI."""
    y, treatment, score = np.asarray(y, float), np.asarray(treatment), np.asarray(score)
    order = np.argsort(-score, kind="mergesort")
    bins = np.empty(len(score), dtype=int)
    bins[order] = np.arange(len(score)) * n_bins // len(score) + 1
    rows = []
    for b in range(1, n_bins + 1):
        m = bins == b
        y1, y0 = y[m & (treatment == 1)], y[m & (treatment == 0)]
        uplift = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "predicted_uplift": float(score[m].mean()),
                "rate_treated": y1.mean(),
                "rate_control": y0.mean(),
                "observed_uplift": uplift,
                "ci_low": uplift - 1.96 * se,
                "ci_high": uplift + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric(metric, y, treatment, score, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% bootstrap interval of a metric(y, treatment, score) on a fixed test set."""
    rng = np.random.default_rng(seed)
    y, treatment, score = np.asarray(y), np.asarray(treatment), np.asarray(score)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric(y[idx], treatment[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
