"""Experiment diagnostics: covariate balance and average treatment effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def standardized_mean_difference(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """SMD = (mean_t - mean_c) / pooled sd. |SMD| < 0.1 is the usual balance rule of thumb."""
    pooled = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((np.mean(x_t) - np.mean(x_c)) / pooled)


def balance_table(X: pd.DataFrame, arm: pd.Series, control: str) -> pd.DataFrame:
    """SMD of every covariate for each arm vs control, plus an omnibus p-value per covariate.

    The p-value comes from a one-way ANOVA across all arms (equivalent to a
    chi-square-like test for dummies at this sample size).
    """
    arm = pd.Series(np.asarray(arm), index=X.index)
    arms = [a for a in arm.unique() if a != control]
    rows = []
    for col in X.columns:
        groups = [X.loc[arm == a, col].to_numpy() for a in arm.unique()]
        row = {"covariate": col, "control_mean": X.loc[arm == control, col].mean()}
        for a in sorted(arms):
            row[f"smd_{a}"] = standardized_mean_difference(
                X.loc[arm == a, col].to_numpy(), X.loc[arm == control, col].to_numpy()
            )
        row["anova_p"] = float(stats.f_oneway(*groups).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def ate_with_ci(y: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Difference in means with a Welch (unequal variance) normal-approximation CI.

    For a binary outcome this is the classic two-proportion Wald interval.
    """
    y, treatment = np.asarray(y, dtype=float), np.asarray(treatment)
    y1, y0 = y[treatment == 1], y[treatment == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(ate) / se)) if se > 0 else np.nan
    return {
        "mean_treated": y1.mean(),
        "mean_control": y0.mean(),
        "ate": ate,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
        "relative_lift": ate / y0.mean() if y0.mean() else np.nan,
        "p_value": p_value,
        "n_treated": len(y1),
        "n_control": len(y0),
    }


def ate_table(df: pd.DataFrame, arms: list[str], control: str, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for a in arms:
        sub = df[df["segment"].isin([a, control])]
        t = (sub["segment"] == a).to_numpy(dtype=int)
        for out in outcomes:
            rows.append({"arm": a, "outcome": out, **ate_with_ci(sub[out].to_numpy(), t)})
    return pd.DataFrame(rows)


def segment_effects(df: pd.DataFrame, treatment: np.ndarray, y: np.ndarray, by: str) -> pd.DataFrame:
    """Model-free ATE (with CI) within each level of a pre-treatment variable."""
    rows = []
    for level, idx in df.groupby(by).indices.items():
        res = ate_with_ci(y[idx], treatment[idx])
        rows.append({"variable": by, "level": level, "n": len(idx), **res})
    return pd.DataFrame(rows)
