import numpy as np
import pandas as pd
import pytest

from backtest import TRADING_DAYS_PER_YEAR, performance_metrics, run_backtest


def _close(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_signal_earns_the_following_days_return():
    close = _close([100, 110, 99, 99, 108.9])
    # Signals generated after the closes of days 0-3
    signals = pd.Series([1, 0, 1, 1], index=close.index[:4])

    result = run_backtest(close, signals, initial_capital=1000, cost_rate=0)

    # Day 1 held (+10%), day 2 in cash (avoids -10%), day 3 held (0%), day 4 held (+10%)
    assert result["position"].tolist() == [0, 1, 0, 1, 1]
    assert result["strategy_value"].tolist() == pytest.approx([1000, 1100, 1100, 1100, 1210])
    assert result["benchmark_value"].iloc[-1] == pytest.approx(1000 * 108.9 / 100)


def test_transaction_costs_are_charged_on_each_position_change():
    close = _close([100, 100, 100, 100, 100])  # flat prices isolate the effect of costs
    signals = pd.Series([1, 0, 1, 1], index=close.index[:4])

    result = run_backtest(close, signals, initial_capital=1000, cost_rate=0.01)

    # Buy, sell, buy: three trades, each costing 1% of the portfolio value at that time
    assert result["strategy_value"].iloc[-1] == pytest.approx(1000 * 0.99**3)
    # Buy-and-hold pays one entry cost
    assert result["benchmark_value"].iloc[-1] == pytest.approx(1000 * 0.99)
    assert performance_metrics(result["strategy_value"], result["position"])["trades"] == 3


def test_cost_is_paid_before_the_days_return():
    close = _close([100, 110])
    result = run_backtest(close, pd.Series([1], index=close.index[:1]), initial_capital=1000, cost_rate=0.01)
    assert result["strategy_value"].iloc[-1] == pytest.approx(1000 * 0.99 * 1.10)


def test_always_invested_strategy_matches_buy_and_hold():
    close = _close([100, 101, 99, 103, 104])
    signals = pd.Series(1, index=close.index[:4])

    result = run_backtest(close, signals, initial_capital=1000, cost_rate=0.001)

    np.testing.assert_allclose(result["strategy_value"], result["benchmark_value"])


def test_strategy_and_benchmark_share_dates_and_capital():
    close = _close(np.linspace(100, 120, 10))
    signals = pd.Series([1, 0, 1, 0, 1], index=close.index[3:8])

    result = run_backtest(close, signals, initial_capital=5000, cost_rate=0.001)

    # From the first signal date to the day after the last signal
    assert result.index.equals(close.index[3:9])
    assert result["strategy_value"].iloc[0] == 5000
    assert result["benchmark_value"].iloc[0] == 5000


def test_backtest_rejects_a_signal_without_a_following_close():
    close = _close([100, 101, 102])
    with pytest.raises(ValueError):
        run_backtest(close, pd.Series([1, 1, 1], index=close.index), 1000, 0.001)


def test_backtest_rejects_gaps_between_signals():
    close = _close([100, 101, 102, 103])
    with pytest.raises(ValueError):
        run_backtest(close, pd.Series([1, 1], index=close.index[[0, 2]]), 1000, 0.001)


@pytest.mark.parametrize(("capital", "cost"), [(0, 0.001), (1000, -0.01), (1000, 1.0)])
def test_backtest_rejects_invalid_settings(capital, cost):
    close = _close([100, 101])
    with pytest.raises(ValueError):
        run_backtest(close, pd.Series([1], index=close.index[:1]), capital, cost)


def test_backtest_rejects_empty_signals():
    with pytest.raises(ValueError):
        run_backtest(_close([100, 101]), pd.Series(dtype=float), 1000, 0.001)


def test_metrics_on_a_known_series():
    value = pd.Series([100.0, 120.0, 90.0, 135.0])
    position = pd.Series([0.0, 1.0, 1.0, 0.0])

    metrics = performance_metrics(value, position)

    daily = np.array([0.2, -0.25, 0.5])
    assert metrics["total_return"] == pytest.approx(0.35)
    assert metrics["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert metrics["sharpe_ratio"] == pytest.approx(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    assert metrics["trades"] == 2
    assert metrics["time_in_market"] == pytest.approx(2 / 3)
    # Three days is far too short to annualise meaningfully
    assert metrics["annualised_return"] is None


def test_annualised_return_over_two_years_of_steady_growth():
    n_days = 2 * TRADING_DAYS_PER_YEAR
    value = pd.Series(100 * 1.0005 ** np.arange(n_days + 1))

    metrics = performance_metrics(value, pd.Series(1.0, index=value.index))

    assert metrics["annualised_return"] == pytest.approx(1.0005**252 - 1)
    assert metrics["max_drawdown"] == 0


def test_sharpe_ratio_is_undefined_without_volatility():
    value = pd.Series([1000.0] * 10)  # always in cash

    metrics = performance_metrics(value, pd.Series(0.0, index=value.index))

    assert metrics["sharpe_ratio"] is None
    assert metrics["total_return"] == 0
    assert metrics["trades"] == 0
    assert metrics["time_in_market"] == 0
