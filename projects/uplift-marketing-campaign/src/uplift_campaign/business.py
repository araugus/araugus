"""Translate uplift rankings into campaign profit under explicit assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignEconomics:
    """Unit economics of one e-mail contact. All values are assumptions, in USD.

    value_per_response: contribution margin of one *incremental* response (visit).
        Default: the experiment shows ~$9.4 of incremental spend per incremental
        visit for the women's e-mail; at a 30% gross margin that is ~$2.8, rounded
        down to $2.50 to stay conservative.
    cost_per_contact: fully loaded cost of sending one e-mail - delivery, creative,
        coupon/offer cost and the opportunity cost of list fatigue / unsubscribes.
    population: customer base the campaign is sized for (results are scaled to it).
    """

    value_per_response: float = 2.50
    cost_per_contact: float = 0.10
    population: int = 100_000

    @property
    def break_even_uplift(self) -> float:
        """Minimum uplift (in response-rate points) a customer needs to be worth contacting."""
        return self.cost_per_contact / self.value_per_response


def profit_curve(y, treatment, score, econ: CampaignEconomics, fractions: np.ndarray | None = None) -> pd.DataFrame:
    """Expected incremental profit of contacting the top-k fraction ranked by `score`.

    For each k, incremental responses are estimated from the randomised test set as
    (treated rate - control rate among the top-k) * k * population. Profit is
    incremental responses * value - contacts * cost. k = 0 means "contact no one"
    (profit 0 by definition) and k = 1 means "contact everyone".
    """
    y, treatment, score = np.asarray(y, float), np.asarray(treatment, int), np.asarray(score, float)
    if fractions is None:
        fractions = np.linspace(0, 1, 101)
    order = np.argsort(-score, kind="mergesort")
    y_s, t_s = y[order], treatment[order]
    cum_t, cum_c = np.cumsum(t_s), np.cumsum(1 - t_s)
    cum_yt, cum_yc = np.cumsum(y_s * t_s), np.cumsum(y_s * (1 - t_s))
    n = len(y)
    rows = []
    for k in fractions:
        m = int(round(k * n))
        if m == 0:
            uplift = 0.0
        else:
            nt, nc = cum_t[m - 1], cum_c[m - 1]
            rt = cum_yt[m - 1] / nt if nt else 0.0
            rc = cum_yc[m - 1] / nc if nc else 0.0
            uplift = rt - rc
        contacts = k * econ.population
        incremental = uplift * contacts
        cost = contacts * econ.cost_per_contact
        revenue = incremental * econ.value_per_response
        rows.append(
            {
                "fraction": float(k),
                "contacts": contacts,
                "uplift_in_targeted": uplift,
                "incremental_responses": incremental,
                "incremental_value": revenue,
                "campaign_cost": cost,
                "profit": revenue - cost,
                "roi": (revenue - cost) / cost if cost > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def optimal_fraction(curve: pd.DataFrame) -> pd.Series:
    """Row of `profit_curve` with the highest profit."""
    return curve.loc[curve["profit"].idxmax()]


def bootstrap_profit_gain(
    y, treatment, score, econ: CampaignEconomics, k: float, n_boot: int = 200, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap CI of profit(top-k) - profit(everyone) on a fixed test set.

    The ranking (`score`) is held fixed; only the sampling noise of the
    randomised test set is propagated.
    """
    rng = np.random.default_rng(seed)
    y, treatment, score = np.asarray(y), np.asarray(treatment), np.asarray(score)
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        c = profit_curve(y[idx], treatment[idx], score[idx], econ, fractions=np.array([k, 1.0]))
        gains.append(c["profit"].iloc[0] - c["profit"].iloc[1])
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return float(lo), float(hi)
