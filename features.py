"""Technical indicators, model features and the next-day prediction target."""

from __future__ import annotations

import pandas as pd

SHORT_WINDOW = 10
LONG_WINDOW = 20

# Inputs to the model. Each value on day t uses closing prices up to and including day t only.
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "close_vs_ma10",
    "close_vs_ma20",
    "ma10_vs_ma20",
    "volatility_20d",
]


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily returns, 10- and 20-day moving averages and 20-day rolling volatility."""
    df = prices.copy()
    close = df["Close"]
    df["return_1d"] = close / close.shift(1) - 1
    df["ma_10"] = close.rolling(SHORT_WINDOW).mean()
    df["ma_20"] = close.rolling(LONG_WINDOW).mean()
    # Standard deviation of daily returns (not annualised)
    df["volatility_20d"] = df["return_1d"].rolling(LONG_WINDOW).std()
    return df


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Return indicators, features and the target, without the warm-up rows that lack full features.

    target on day t is 1 if the close on day t+1 is higher than the close on day t, otherwise 0.
    The last row has no next close yet, so its target is missing. That row is the one used
    for the next-day forecast and is never used for training or testing.
    """
    df = add_indicators(prices)
    close = df["Close"]
    df["return_5d"] = close / close.shift(5) - 1
    df["close_vs_ma10"] = close / df["ma_10"] - 1
    df["close_vs_ma20"] = close / df["ma_20"] - 1
    df["ma10_vs_ma20"] = df["ma_10"] / df["ma_20"] - 1

    next_close = close.shift(-1)
    df["target"] = (next_close > close).astype(float).where(next_close.notna())

    return df.dropna(subset=FEATURE_COLUMNS)
