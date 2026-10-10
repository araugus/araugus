"""Client for the Banco Central do Brasil SGS API and construction of a tidy monthly panel.

API docs: https://dadosabertos.bcb.gov.br/ (SGS - Sistema Gerenciador de Series Temporais).
Endpoint: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json
          &dataInicial=dd/mm/yyyy&dataFinal=dd/mm/yyyy

Quirks handled here:
* Daily series reject windows longer than 10 years (HTTP 406), so every request is paginated
  into windows shorter than 10 years.
* The gateway occasionally returns an HTML error page with status 200; such responses are
  retried with exponential backoff, like 429/5xx and connection errors.
* A window with no observations returns HTTP 404 ("Value(s) not found"); treated as empty.
* The Selic target series (432) contains dates in the future (the current target is repeated
  until the next Copom meeting), so observations after "today" are dropped.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAX_WINDOW_DAYS = 1826  # ~5 years: well below the 10-year limit for daily series
DEFAULT_START = date(2000, 1, 1)


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata for one SGS series.

    availability_lag: how many months the series lags behind the *target* at the moment the
    target for month T is published. E.g. IBC-Br for month T is released ~2 weeks after the
    credit statistics for month T, so at forecast time only IBC-Br up to T-1 is known (lag 1).
    """

    code: int
    column: str
    name: str
    frequency: str  # "daily" or "monthly"
    unit: str
    availability_lag: int = 0


TARGET = "default_rate_hh"

SERIES: dict[str, SeriesSpec] = {
    s.column: s
    for s in [
        SeriesSpec(21084, TARGET, "Credit portfolio default rate (90+ days) - Households - Total",
                   "monthly", "% of portfolio"),
        SeriesSpec(21112, "default_rate_hh_free",
                   "Credit default rate (90+ days), non-earmarked credit - Households - Total",
                   "monthly", "% of portfolio"),
        SeriesSpec(432, "selic_target", "Selic target rate set by Copom (daily, monthly mean)",
                   "daily", "% p.a."),
        SeriesSpec(433, "ipca_mom", "IPCA consumer price index - monthly change",
                   "monthly", "% m/m"),
        SeriesSpec(20716, "lending_rate_hh", "Average interest rate on credit - Households - Total",
                   "monthly", "% p.a."),
        SeriesSpec(20633, "concessions_hh", "Credit concessions (new loans) - Households - Total",
                   "monthly", "BRL million"),
        SeriesSpec(24369, "unemployment", "Unemployment rate - PNAD Continua (rolling quarter)",
                   "monthly", "%"),
        SeriesSpec(24364, "ibc_br_sa", "IBC-Br economic activity index - seasonally adjusted",
                   "monthly", "index (2002=100)", availability_lag=1),
        SeriesSpec(29037, "debt_to_income_hh",
                   "Household debt to the financial system / 12-month income",
                   "monthly", "%", availability_lag=1),
    ]
}


class SGSError(RuntimeError):
    """Raised when the SGS API cannot be reached after all retries."""


def date_windows(start: date, end: date, max_days: int = MAX_WINDOW_DAYS) -> Iterator[tuple[date, date]]:
    """Split [start, end] into consecutive, non-overlapping windows of at most ``max_days``."""
    if start > end:
        return
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


class SGSClient:
    """Small, dependency-light SGS client with pagination, retries and on-disk caching."""

    def __init__(
        self,
        cache_dir: Path | None = RAW_DIR,
        session: requests.Session | None = None,
        max_retries: int = 4,
        backoff: float = 1.5,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self._sleep = sleep

    # ------------------------------------------------------------------ HTTP
    def _get_window(self, code: int, start: date, end: date) -> list[dict[str, Any]]:
        params = {
            "formato": "json",
            "dataInicial": start.strftime("%d/%m/%Y"),
            "dataFinal": end.strftime("%d/%m/%Y"),
        }
        url = BASE_URL.format(code=code)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 404:  # SGS answers 404 for an empty window
                    return []
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                # The gateway intermittently answers 200 with an HTML error page: retry it.
                payload = resp.json()
                if not isinstance(payload, list):
                    raise SGSError(f"Unexpected payload for series {code}: {payload!r:.200}")
                return payload
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError, ValueError) as err:
                status = getattr(getattr(err, "response", None), "status_code", None)
                if status is not None and status not in (429, 500, 502, 503, 504):
                    raise SGSError(f"Series {code}: HTTP {status}") from err
                last_err = err
                wait = self.backoff * (2**attempt)
                logger.warning("SGS %s %s-%s failed (%s); retry in %.1fs", code, start, end, err, wait)
                if attempt < self.max_retries:
                    self._sleep(wait)
        raise SGSError(f"Series {code}: giving up after {self.max_retries + 1} attempts") from last_err

    def fetch_raw(self, code: int, start: date = DEFAULT_START, end: date | None = None) -> pd.DataFrame:
        """Download a series over [start, end], paginating, and return columns ``date``, ``value``."""
        end = end or date.today()
        records: list[dict[str, Any]] = []
        for w_start, w_end in date_windows(start, end):
            records.extend(self._get_window(code, w_start, w_end))
        return parse_records(records, today=end)

    # ----------------------------------------------------------------- cache
    def _cache_path(self, code: int) -> Path | None:
        return self.cache_dir / f"sgs_{code}.json" if self.cache_dir is not None else None

    def get_series(
        self, code: int, start: date = DEFAULT_START, end: date | None = None, refresh: bool = False
    ) -> pd.DataFrame:
        """Return a series, reading the JSON cache in ``data/raw`` when available."""
        path = self._cache_path(code)
        if path is not None and path.exists() and not refresh:
            records = json.loads(path.read_text(encoding="utf-8"))
            return parse_records(records, today=end)
        df = self.fetch_raw(code, start, end)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            out = [
                {"data": d.strftime("%d/%m/%Y"), "valor": str(v)}
                for d, v in zip(df["date"], df["value"], strict=True)
            ]
            path.write_text(json.dumps(out), encoding="utf-8")
        return df


def parse_records(records: list[dict[str, Any]], today: date | None = None) -> pd.DataFrame:
    """Convert SGS JSON records into a sorted, de-duplicated DataFrame (``date``, ``value``)."""
    if not records:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})
    df = pd.DataFrame(records).rename(columns={"data": "date", "valor": "value"})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).drop_duplicates("date", keep="last").sort_values("date")
    if today is not None:
        df = df[df["date"] <= pd.Timestamp(today)]
    return df[["date", "value"]].reset_index(drop=True)


def to_monthly(df: pd.DataFrame, how: str = "mean") -> pd.Series:
    """Aggregate a (date, value) frame to a month-start indexed series."""
    s = df.set_index("date")["value"]
    s = getattr(s.resample("MS"), how)()
    return s.dropna()


def ipca_12m(ipca_mom: pd.Series) -> pd.Series:
    """Accumulated 12-month inflation (%) from monthly % changes."""
    factor = (1 + ipca_mom / 100).rolling(12).apply(np.prod, raw=True)
    return (factor - 1) * 100


def build_panel(
    raw: dict[str, pd.DataFrame], specs: dict[str, SeriesSpec] = SERIES, start: str | None = None
) -> pd.DataFrame:
    """Build the monthly modeling panel (month-start index, one column per series).

    Daily series are averaged within the month. Adds ``ipca_12m``. The panel is trimmed to start
    at the first observation of the target (or ``start``) and ends at the last target month;
    drivers that are published later than the target keep their trailing NaNs (handled by the
    feature builder through ``availability_lag``).
    """
    cols = {}
    for col, spec in specs.items():
        if col not in raw:
            continue
        cols[col] = to_monthly(raw[col], "mean" if spec.frequency == "daily" else "last")
    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.DatetimeIndex(panel.index, freq="MS", name="month")
    if "ipca_mom" in panel:
        panel["ipca_12m"] = ipca_12m(panel["ipca_mom"])
    target = panel[TARGET].dropna()
    first = pd.Timestamp(start) if start else target.index.min()
    panel = panel.loc[first : target.index.max()]
    return panel.asfreq("MS")


def download_all(client: SGSClient | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download (or load from cache) every series in :data:`SERIES`."""
    client = client or SGSClient()
    out = {}
    for col, spec in SERIES.items():
        out[col] = client.get_series(spec.code, refresh=refresh)
        logger.info("%s (SGS %s): %d obs", col, spec.code, len(out[col]))
    return out


def load_panel(refresh: bool = False, client: SGSClient | None = None) -> pd.DataFrame:
    """Convenience: download/cache all series and return the monthly panel."""
    return build_panel(download_all(client, refresh=refresh))
