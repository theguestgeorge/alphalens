import numpy as np
import pandas as pd
import pytest

import data
from data import DataError, clean_prices, download_prices, drop_unfinished_session, normalise_ticker


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("aapl", "AAPL"), (" msft ", "MSFT"), ("BRK-B", "BRK-B"), ("vod.l", "VOD.L"), ("^GSPC", "^GSPC")],
)
def test_normalise_ticker_accepts_common_symbols(raw, expected):
    assert normalise_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "AAPL; rm", "A B", "ABCDEFGHIJKLMNOPQ"])
def test_normalise_ticker_rejects_invalid_symbols(raw):
    with pytest.raises(DataError):
        normalise_ticker(raw)


def _raw_frame(index, closes):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 100, "Dividends": 0.0},
        index=index,
    )


def test_clean_prices_sorts_removes_timezone_and_drops_missing_closes():
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04", "2024-01-05"]).tz_localize("America/New_York")
    raw = _raw_frame(index, [101.0, 100.0, np.nan, 103.0])

    prices = clean_prices(raw)

    assert list(prices.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert prices.index.tz is None
    assert prices.index.is_monotonic_increasing
    # The missing close is dropped, not filled with an invented price
    assert list(prices["Close"]) == [100.0, 101.0, 103.0]


def test_clean_prices_raises_when_columns_are_missing():
    raw = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(DataError, match="missing"):
        clean_prices(raw)


def test_clean_prices_raises_when_no_close_is_valid():
    raw = _raw_frame(pd.bdate_range("2024-01-01", periods=2), [np.nan, np.nan])
    with pytest.raises(DataError):
        clean_prices(raw)


def _session_frame():
    index = pd.DatetimeIndex(["2024-03-04", "2024-03-05"]).tz_localize("America/New_York")
    return _raw_frame(index, [100.0, 101.0])


def test_drop_unfinished_session_removes_todays_intraday_row():
    session_end = pd.Timestamp("2024-03-05 16:00", tz="America/New_York")
    during_session = pd.Timestamp("2024-03-05 11:00", tz="America/New_York")

    result = drop_unfinished_session(_session_frame(), session_end, during_session)

    assert list(result["Close"]) == [100.0]


def test_drop_unfinished_session_keeps_completed_sessions():
    session_end = pd.Timestamp("2024-03-05 16:00", tz="America/New_York")
    after_close = pd.Timestamp("2024-03-05 18:00", tz="America/New_York")
    next_session_end = pd.Timestamp("2024-03-06 16:00", tz="America/New_York")
    before_next_open = pd.Timestamp("2024-03-06 08:00", tz="America/New_York")

    assert len(drop_unfinished_session(_session_frame(), session_end, after_close)) == 2
    assert len(drop_unfinished_session(_session_frame(), next_session_end, before_next_open)) == 2
    assert len(drop_unfinished_session(_session_frame(), None, before_next_open)) == 2


class _FailingTicker:
    def __init__(self, symbol):
        pass

    def history(self, **kwargs):
        raise ConnectionError("network unreachable")


class _EmptyTicker:
    history_metadata = {}

    def __init__(self, symbol):
        pass

    def history(self, **kwargs):
        return pd.DataFrame()


def test_download_prices_reports_provider_failures(monkeypatch):
    monkeypatch.setattr(data.yf, "Ticker", _FailingTicker)
    with pytest.raises(DataError, match="network unreachable"):
        download_prices("AAPL")


def test_download_prices_reports_empty_results(monkeypatch):
    monkeypatch.setattr(data.yf, "Ticker", _EmptyTicker)
    with pytest.raises(DataError, match="no prices"):
        download_prices("AAPL")
