from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Replays a list of responses (or a callable) and records every request."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, **(params or {})})
        if callable(self.responder):
            return self.responder(url, params)
        return self.responder.pop(0)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Small monthly panel shaped like the real one (trend + noise + one driver)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2012-01-01", periods=96, freq="MS")
    driver = 10 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    target = 4 + 0.05 * np.r_[np.zeros(6), driver[:-6] - 10] + np.cumsum(rng.normal(0, 0.03, len(idx)))
    return pd.DataFrame(
        {"default_rate_hh": target, "selic_target": driver, "ibc_br_sa": 100 + rng.normal(0, 1, len(idx))},
        index=pd.DatetimeIndex(idx, freq="MS", name="month"),
    )
