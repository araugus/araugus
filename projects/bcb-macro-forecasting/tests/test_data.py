from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import FakeResponse, FakeSession

from bcb_forecast.data import (
    SGSClient,
    SGSError,
    build_panel,
    date_windows,
    ipca_12m,
    parse_records,
)


def no_sleep(_: float) -> None:
    pass


def test_date_windows_cover_range_without_gaps_or_overlap():
    start, end = date(2000, 1, 1), date(2026, 10, 10)
    wins = list(date_windows(start, end, max_days=1826))
    assert wins[0][0] == start and wins[-1][1] == end
    for (_, b1), (a2, _) in zip(wins[:-1], wins[1:], strict=True):
        assert (a2 - b1).days == 1
    assert all((b - a).days + 1 <= 1826 for a, b in wins)
    assert list(date_windows(end, start)) == []


def test_fetch_paginates_and_treats_404_as_empty():
    def responder(url, params):
        if params["dataInicial"] == "01/01/2000":
            return FakeResponse(404, {"erro": "Value(s) not found"})
        return FakeResponse(200, [{"data": params["dataInicial"], "valor": "1.5"}])

    session = FakeSession(responder)
    client = SGSClient(cache_dir=None, session=session, sleep=no_sleep)
    df = client.fetch_raw(432, date(2000, 1, 1), date(2012, 12, 31))
    assert len(session.calls) == 3  # 13 years in ~5-year windows
    assert all("bcdata.sgs.432" in c["url"] for c in session.calls)
    assert len(df) == 2 and df["value"].eq(1.5).all()


def test_retries_transient_errors_then_succeeds():
    responses = [
        FakeResponse(503),
        FakeResponse(200, None, text="<html>gateway error</html>"),  # 200 + HTML page
        FakeResponse(200, [{"data": "01/01/2020", "valor": "4.5"}]),
    ]
    waits: list[float] = []
    client = SGSClient(cache_dir=None, session=FakeSession(responses), sleep=waits.append, backoff=1.0)
    df = client.fetch_raw(433, date(2020, 1, 1), date(2020, 1, 31))
    assert df["value"].tolist() == [4.5]
    assert waits == [1.0, 2.0]  # exponential backoff


def test_gives_up_after_max_retries():
    client = SGSClient(cache_dir=None, session=FakeSession(lambda u, p: FakeResponse(500)),
                       sleep=no_sleep, max_retries=2)
    with pytest.raises(SGSError):
        client.fetch_raw(433, date(2020, 1, 1), date(2020, 1, 31))


def test_client_error_is_not_retried():
    session = FakeSession(lambda u, p: FakeResponse(406, {"error": "window too large"}))
    client = SGSClient(cache_dir=None, session=session, sleep=no_sleep)
    with pytest.raises(SGSError):
        client.fetch_raw(432, date(2020, 1, 1), date(2020, 1, 31))
    assert len(session.calls) == 1


def test_cache_avoids_second_download(tmp_path):
    session = FakeSession(lambda u, p: FakeResponse(200, [{"data": "01/03/2011", "valor": "4.62"}]))
    client = SGSClient(cache_dir=tmp_path, session=session, sleep=no_sleep)
    first = client.get_series(21084, date(2011, 1, 1), date(2011, 12, 31))
    n_calls = len(session.calls)
    second = client.get_series(21084, date(2011, 1, 1), date(2011, 12, 31))
    assert len(session.calls) == n_calls
    assert (tmp_path / "sgs_21084.json").exists()
    pd.testing.assert_frame_equal(first, second)


def test_parse_records_drops_future_duplicates_and_bad_values():
    recs = [
        {"data": "02/01/2020", "valor": "2"},
        {"data": "01/01/2020", "valor": "1"},
        {"data": "01/01/2020", "valor": "1.1"},
        {"data": "03/01/2020", "valor": ""},
        {"data": "05/01/2030", "valor": "9"},  # Selic target series lists future dates
    ]
    df = parse_records(recs, today=date(2026, 1, 1))
    assert df["date"].dt.day.tolist() == [1, 2]
    assert df["value"].tolist() == [1.1, 2.0]
    assert parse_records([]).empty


def test_ipca_12m_compounds_monthly_changes():
    s = pd.Series([1.0] * 12, index=pd.date_range("2020-01-01", periods=12, freq="MS"))
    assert ipca_12m(s).iloc[-1] == pytest.approx((1.01**12 - 1) * 100)
    assert np.isnan(ipca_12m(s).iloc[-2])


def test_build_panel_aggregates_daily_and_trims_to_target():
    days = pd.date_range("2019-01-01", "2020-04-30", freq="D")
    raw = {
        "default_rate_hh": pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                                         "value": [3.0, 3.1, 3.2]}),
        "selic_target": pd.DataFrame({"date": days, "value": np.where(days.day <= 15, 4.0, 5.0)}),
        "ipca_mom": pd.DataFrame({"date": pd.date_range("2019-01-01", periods=16, freq="MS"),
                                  "value": 0.5}),
    }
    panel = build_panel(raw)
    assert panel.index.min() == pd.Timestamp("2020-01-01")
    assert panel.index.max() == pd.Timestamp("2020-03-01")  # ends at the last target month
    assert panel.index.freqstr == "MS"
    jan = panel.loc["2020-01-01", "selic_target"]
    assert jan == pytest.approx((15 * 4 + 16 * 5) / 31)
    assert panel["ipca_12m"].notna().all()
