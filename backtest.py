"""Transaction-cost-aware backtest of a long-or-cash strategy against buy-and-hold."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def run_backtest(close: pd.Series, signals: pd.Series, initial_capital: float, cost_rate: float) -> pd.DataFrame:
    """Simulate the strategy and buy-and-hold over the same dates with the same starting capital.

    Execution convention: a signal dated t is generated after the close on day t. A signal of 1
    means holding the stock from that close to the next trading day's close; 0 means holding cash
    (earning nothing). So the position on each date is the signal from the previous trading day.

    Costs: cost_rate x |change in position| is charged on the portfolio value at every trade,
    before the following day's return. The strategy starts in cash. Buy-and-hold buys at the
    first close and pays one entry cost. Neither is sold at the end (values are marked to market).

    The returned frame starts at the first signal date (the starting point, value = initial
    capital) and ends on the trading day after the last signal.
    """
    if signals.empty:
        raise ValueError("There are no signals to backtest.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if not 0 <= cost_rate < 1:
        raise ValueError("The transaction cost rate must be between 0 and 1.")
    if signals.index[0] not in close.index:
        raise ValueError("The first signal date has no closing price.")

    start = close.index.get_loc(signals.index[0])
    period = close.index[start : start + len(signals) + 1]
    if len(period) != len(signals) + 1 or not period[:-1].equals(signals.index):
        raise ValueError("Signals must cover consecutive trading days, followed by one more known close.")

    daily_return = (close / close.shift(1) - 1).loc[period]
    daily_return.iloc[0] = 0.0  # starting point: nothing is held before the first signal

    # Shift by one trading day: the signal from period[i] is the position held on period[i + 1]
    position = pd.Series(signals.to_numpy(dtype=float), index=period[1:]).reindex(period, fill_value=0.0)
    benchmark_position = pd.Series(1.0, index=period)
    benchmark_position.iloc[0] = 0.0

    return pd.DataFrame(
        {
            "close": close.loc[period],
            "daily_return": daily_return,
            "position": position,
            "strategy_value": _portfolio_value(position, daily_return, initial_capital, cost_rate),
            "benchmark_position": benchmark_position,
            "benchmark_value": _portfolio_value(benchmark_position, daily_return, initial_capital, cost_rate),
        }
    )


def _portfolio_value(position: pd.Series, daily_return: pd.Series, initial_capital: float, cost_rate: float) -> pd.Series:
    trades = position.diff().abs().fillna(0.0)
    growth = (1 - cost_rate * trades) * (1 + position * daily_return)
    return initial_capital * growth.cumprod()


def performance_metrics(value: pd.Series, position: pd.Series) -> dict[str, float | int | None]:
    """Summarise an equity curve. None means the metric is not meaningful for this series.

    - total_return: final value / initial value - 1
    - annualised_return: compounded over 252 trading days per year; None for periods under a year
    - max_drawdown: largest fall from a previous peak, as a negative fraction
    - sharpe_ratio: mean / standard deviation of daily returns x sqrt(252), risk-free rate assumed 0
    - trades: number of position changes (buys plus sells)
    - time_in_market: fraction of days holding the stock
    """
    n_days = len(value) - 1
    if n_days < 1:
        raise ValueError("At least two portfolio values are needed.")

    daily = (value / value.shift(1) - 1).iloc[1:]
    total_return = value.iloc[-1] / value.iloc[0] - 1

    annualised_return = None
    if n_days >= TRADING_DAYS_PER_YEAR:
        annualised_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    std = daily.std()
    sharpe_ratio = None
    if n_days >= 2 and std > 1e-12:
        sharpe_ratio = daily.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "total_return": float(total_return),
        "annualised_return": None if annualised_return is None else float(annualised_return),
        "max_drawdown": float((value / value.cummax() - 1).min()),
        "sharpe_ratio": None if sharpe_ratio is None else float(sharpe_ratio),
        "trades": int(round(position.diff().abs().fillna(0.0).sum())),
        "time_in_market": float(position.iloc[1:].mean()),
    }
