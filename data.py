"""Download and clean daily stock prices from Yahoo Finance."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

# Sidebar label -> yfinance period string
PERIODS = {"1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y"}
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
# Yahoo search result types that have ordinary daily prices (no futures, options or currencies)
SEARCH_TYPES = {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}

# Letters, digits and the separators Yahoo uses, e.g. AAPL, BRK-B, RDSA.AS, ^GSPC
_TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")

# Make yfinance raise errors instead of logging them and returning an empty frame,
# so the app can tell the user what actually went wrong.
yf.config.debug.hide_exceptions = False


class DataError(Exception):
    """Raised when price data cannot be retrieved or is unusable. The message is shown to the user."""


def normalise_ticker(ticker: str) -> str:
    """Return an upper-case ticker symbol, or raise DataError if it cannot be a valid symbol."""
    symbol = ticker.strip().upper()
    if not _TICKER_PATTERN.match(symbol):
        raise DataError(f"'{ticker}' is not a valid ticker symbol. Enter a symbol such as AAPL, MSFT or NVDA.")
    return symbol


@dataclass(frozen=True)
class Listing:
    """A security found by search: its Yahoo symbol, company name and exchange."""

    symbol: str
    name: str
    exchange: str

    def label(self) -> str:
        return f"{self.symbol} · {self.name} ({self.exchange})"


def search_listings(query: str, max_results: int = 10) -> list[Listing]:
    """Find listings whose company name or ticker matches the query, exact symbol match first.

    Raises DataError if the query is empty, the search fails or nothing matches.
    """
    query = query.strip()
    if not query:
        raise DataError("Enter a company name or ticker symbol, for example Apple or AAPL.")
    try:
        quotes = yf.Search(
            query, max_results=max_results, news_count=0, lists_count=0, include_cb=False, recommended=0
        ).quotes
    except Exception as exc:
        detail = str(exc).strip().rstrip(":") or type(exc).__name__
        raise DataError(f"Could not search Yahoo Finance for '{query}' ({detail}).") from exc

    listings = [
        Listing(
            symbol=quote["symbol"],
            name=quote.get("longname") or quote.get("shortname") or quote["symbol"],
            exchange=quote.get("exchDisp") or quote.get("exchange") or "unknown exchange",
        )
        for quote in quotes
        if quote.get("quoteType") in SEARCH_TYPES and _TICKER_PATTERN.match(quote["symbol"])
    ]
    if not listings:
        raise DataError(
            f"No stocks, ETFs or indices on Yahoo Finance match '{query}'. "
            "Try the company's full name or its ticker symbol."
        )
    # Stable sort: a listing whose symbol is exactly what was typed comes first, the rest keep Yahoo's order.
    listings.sort(key=lambda listing: listing.symbol != query.upper())
    return listings


def download_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV prices, adjusted for splits and dividends, keeping completed sessions only."""
    symbol = normalise_ticker(ticker)
    try:
        source = yf.Ticker(symbol)
        raw = source.history(period=period, interval="1d", auto_adjust=True)
        session_end = _regular_session_end(source.history_metadata)
    except Exception as exc:
        # yfinance raises many exception types (unknown symbol, network failure, rate limiting).
        # Re-raise them as one error type with a readable message, keeping the original as the cause.
        detail = str(exc).strip().rstrip(":") or type(exc).__name__
        raise DataError(
            f"Could not download prices for {symbol} from Yahoo Finance ({detail}). "
            "Check that the symbol exists on finance.yahoo.com."
        ) from exc

    if raw.empty:
        raise DataError(f"Yahoo Finance returned no prices for {symbol}. Check that the symbol is listed there.")
    raw = drop_unfinished_session(raw, session_end, pd.Timestamp.now(tz="UTC"))
    return clean_prices(raw)


def _regular_session_end(metadata: dict) -> pd.Timestamp | None:
    end = (metadata or {}).get("currentTradingPeriod", {}).get("regular", {}).get("end")
    return end if isinstance(end, pd.Timestamp) else None


def drop_unfinished_session(raw: pd.DataFrame, session_end: pd.Timestamp | None, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the latest row while its trading session is still open: its 'Close' is only an intraday price."""
    if session_end is None or raw.empty:
        return raw
    if now < session_end and raw.index[-1].date() == session_end.date():
        return raw.iloc[:-1]
    return raw


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep OHLCV columns, use a plain date index and drop rows without a valid closing price.

    Missing prices are dropped, never filled in, so the data never contains invented prices.
    """
    missing = [column for column in PRICE_COLUMNS if column not in raw.columns]
    if missing:
        raise DataError(f"The downloaded data is missing these columns: {', '.join(missing)}.")

    prices = raw[PRICE_COLUMNS].copy()
    index = pd.DatetimeIndex(prices.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    prices.index = index.normalize()
    prices.index.name = "Date"

    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices[prices["Close"].notna() & (prices["Close"] > 0)]
    if prices.empty:
        raise DataError("The downloaded data contains no valid closing prices.")
    return prices
