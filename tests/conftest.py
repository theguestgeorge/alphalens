import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_prices():
    """Factory for synthetic daily OHLCV prices following a seeded random walk (tests only)."""

    def _make(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days))
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000},
            index=dates,
        )

    return _make
