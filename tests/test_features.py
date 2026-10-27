import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, LONG_WINDOW, add_indicators, build_dataset


def test_indicators_match_hand_calculation():
    close = pd.Series(np.arange(1.0, 31.0), index=pd.bdate_range("2024-01-01", periods=30))
    df = add_indicators(pd.DataFrame({"Close": close}))

    assert np.isnan(df["return_1d"].iloc[0])
    assert df["return_1d"].iloc[1] == pytest.approx(2 / 1 - 1)
    assert df["return_1d"].iloc[29] == pytest.approx(30 / 29 - 1)

    assert df["ma_10"].iloc[:9].isna().all()
    assert df["ma_10"].iloc[9] == pytest.approx(np.mean(np.arange(1, 11)))
    assert df["ma_20"].iloc[:19].isna().all()
    assert df["ma_20"].iloc[19] == pytest.approx(np.mean(np.arange(1, 21)))


def test_volatility_is_rolling_std_of_daily_returns(make_prices):
    df = add_indicators(make_prices(60))
    returns = df["Close"] / df["Close"].shift(1) - 1

    expected = np.std(returns.iloc[21:41], ddof=1)
    assert df["volatility_20d"].iloc[40] == pytest.approx(expected)


def test_target_is_next_day_direction():
    # 25 flat days for warm-up, then up, down, unchanged, up
    closes = [100.0] * 25 + [101.0, 100.0, 100.0, 102.0]
    prices = pd.DataFrame({"Close": closes}, index=pd.bdate_range("2024-01-01", periods=len(closes)))

    target = build_dataset(prices)["target"]

    # Day 24 (100) -> day 25 (101) is up; 101 -> 100 down; 100 -> 100 unchanged counts as not up
    assert target.iloc[-5:-1].tolist() == [1.0, 0.0, 0.0, 1.0]
    # The latest day has no next close, so its target is unknown rather than guessed
    assert np.isnan(target.iloc[-1])


def test_target_matches_following_close_on_random_walk(make_prices):
    dataset = build_dataset(make_prices())
    close = dataset["Close"]

    for i in range(len(dataset) - 1):
        assert dataset["target"].iloc[i] == float(close.iloc[i + 1] > close.iloc[i])


def test_features_do_not_use_future_prices(make_prices):
    prices = make_prices()
    changed = prices.copy()
    cutoff = prices.index[200]
    changed.loc[changed.index >= cutoff, "Close"] *= 1.5

    original = build_dataset(prices)
    altered = build_dataset(changed)

    before = original.index < cutoff
    pd.testing.assert_frame_equal(original.loc[before, FEATURE_COLUMNS], altered.loc[before, FEATURE_COLUMNS])


def test_target_and_future_values_are_not_features():
    assert "target" not in FEATURE_COLUMNS
    assert "Close" not in FEATURE_COLUMNS


def test_build_dataset_drops_only_warm_up_rows(make_prices):
    prices = make_prices(100)
    dataset = build_dataset(prices)

    # 20-day volatility needs 20 daily returns, and the first return needs a previous close
    assert len(dataset) == len(prices) - LONG_WINDOW
    assert dataset.index[0] == prices.index[LONG_WINDOW]
    assert not dataset[FEATURE_COLUMNS].isna().any().any()
    assert dataset["target"].iloc[:-1].notna().all()


def test_build_dataset_with_short_history_is_empty(make_prices):
    assert build_dataset(make_prices(15)).empty
