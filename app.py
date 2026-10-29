"""AlphaLens: a simple stock market prediction and backtesting tool.

This file only builds the Streamlit interface. Data, features, models and the backtest
live in data.py, features.py, model.py and backtest.py.
"""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import TRADING_DAYS_PER_YEAR, performance_metrics, run_backtest
from data import PERIODS, DataError, Listing, download_prices, search_listings
from features import add_indicators, build_dataset
from model import MODEL_NAMES, Evaluation, Forecast, InsufficientDataError, evaluate_model, forecast_next_day

PRIMARY_COLOUR = "#2a78d6"  # close price, strategy
SECONDARY_COLOUR = "#eb6834"  # moving average, buy-and-hold
SHADE_COLOUR = "#898781"


@st.cache_data(ttl=86400, show_spinner=False)
def load_listings(query: str) -> list[Listing]:
    return search_listings(query)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(symbol: str, period: str) -> pd.DataFrame:
    return download_prices(symbol, period)


@st.cache_data(show_spinner=False)
def load_model_results(dataset: pd.DataFrame, model_name: str) -> tuple[Evaluation, Forecast]:
    return evaluate_model(dataset, model_name), forecast_next_day(dataset, model_name)


def stop_with_error(message: str) -> NoReturn:
    st.error(message)
    st.write(
        "Check the company name or ticker in the sidebar and try again. If it is correct, Yahoo Finance may be "
        "temporarily unavailable or limiting requests, so wait a minute before retrying."
    )
    st.button("Try again")
    st.stop()


def date_text(timestamp: pd.Timestamp) -> str:
    return f"{timestamp.day} {timestamp:%b %Y}"


def signed_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def line_chart(series: dict[str, tuple[pd.Series, str]], y_title: str, shade_from: pd.Timestamp | None = None) -> go.Figure:
    """Line chart with one trace per series and an optional shaded region from shade_from to the end."""
    fig = go.Figure()
    for name, (values, colour) in series.items():
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values,
                name=name,
                mode="lines",
                line={"color": colour, "width": 2},
                hovertemplate="%{y:,.2f}",
            )
        )
    if shade_from is not None:
        end = max(values.index[-1] for values, _ in series.values())
        fig.add_vrect(x0=shade_from, x1=end, fillcolor=SHADE_COLOUR, opacity=0.1, line_width=0, layer="below")
    fig.update_layout(
        height=360,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis={"title": y_title, "tickformat": ",.0f"},
    )
    return fig


def format_metrics(metrics: dict) -> dict[str, str]:
    sharpe = metrics["sharpe_ratio"]
    return {
        "Total return": signed_percent(metrics["total_return"]),
        "Annualised return": signed_percent(metrics["annualised_return"]),
        "Maximum drawdown": f"{metrics['max_drawdown']:.1%}",
        "Sharpe ratio": "n/a" if sharpe is None else f"{sharpe:.2f}",
        "Trades": str(metrics["trades"]),
        "Time in market": f"{metrics['time_in_market']:.0%}",
    }


st.set_page_config(page_title="AlphaLens", layout="centered")

with st.sidebar:
    query = st.text_input(
        "Company or ticker",
        value="AAPL",
        help="Search Yahoo Finance by company name (for example Apple) or ticker symbol (for example AAPL), "
        "then press Enter.",
    )
    search_error = None
    try:
        listings = load_listings(query.strip())
    except DataError as exc:
        listings, search_error = [], str(exc)
    listing = st.selectbox(
        "Stock",
        listings,
        format_func=Listing.label,
        disabled=not listings,
        help="Matching stocks, ETFs and indices. An exact ticker match is listed first.",
    )
    period_label = st.selectbox("History", list(PERIODS), index=2)
    initial_capital = st.number_input(
        "Initial investment",
        min_value=100.0,
        value=10_000.0,
        step=1_000.0,
        format="%.0f",
        help="In the same currency as the stock's prices.",
    )
    cost_percent = st.number_input(
        "Transaction cost per trade (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.10,
        step=0.05,
        format="%.2f",
        help="Charged on the portfolio value every time the strategy buys or sells.",
    )
    model_name = st.selectbox("Model", MODEL_NAMES)
    st.caption("Prices from Yahoo Finance via yfinance. For education only, not investment advice.")

st.title("AlphaLens")
st.write("A simple stock market prediction and backtesting tool.")

if listing is None:
    stop_with_error(search_error)

symbol = listing.symbol
display_name = f"{listing.name} ({symbol})"
try:
    with st.spinner(f"Downloading daily prices for {display_name}..."):
        prices = load_prices(symbol, PERIODS[period_label])
except DataError as exc:
    stop_with_error(str(exc))

dataset = build_dataset(prices)
evaluation, forecast, model_error = None, None, None
try:
    with st.spinner(f"Training the {model_name.lower()}..."):
        evaluation, forecast = load_model_results(dataset, model_name)
except InsufficientDataError as exc:
    model_error = str(exc)

# Summary and price history
indicators = add_indicators(prices)
close = prices["Close"]
last_date = prices.index[-1]

st.subheader(display_name)
col1, col2, col3 = st.columns(3)
col1.metric(
    f"Close on {date_text(last_date)}",
    f"{close.iloc[-1]:,.2f}",
    delta=f"{close.iloc[-1] / close.iloc[0] - 1:+.1%} over the period",
)
col2.metric(
    "Annualised volatility",
    f"{indicators['return_1d'].std() * np.sqrt(TRADING_DAYS_PER_YEAR):.0%}",
    help="Standard deviation of daily returns over the selected history, multiplied by √252.",
)
col3.metric("Trading days", f"{len(prices):,}")
st.caption(
    f"Daily prices from {date_text(prices.index[0])} to {date_text(last_date)}, adjusted for splits and dividends."
)

test_start = evaluation.test.index[0] if evaluation is not None else None
st.plotly_chart(
    line_chart(
        {"Close": (close, PRIMARY_COLOUR), "20-day moving average": (indicators["ma_20"], SECONDARY_COLOUR)},
        "Price",
        shade_from=test_start,
    )
)
if test_start is not None:
    st.caption("The shaded area is the test period used for the backtest below.")

with st.expander("Price data"):
    price_table = pd.DataFrame(
        {
            "Close": close,
            "10-day MA": indicators["ma_10"],
            "20-day MA": indicators["ma_20"],
            "Daily return (%)": indicators["return_1d"] * 100,
            "Volume": prices["Volume"],
        }
    ).round(2)
    price_table.index = price_table.index.date
    st.dataframe(price_table.iloc[::-1])

# Next-day forecast
st.subheader("Next trading day")
if forecast is None:
    st.warning(model_error)
else:
    col1, col2 = st.columns(2)
    col1.metric("Predicted direction", "Higher" if forecast.prob_up > 0.5 else "Lower or flat")
    col2.metric("Model-estimated probability of a higher close", f"{forecast.prob_up:.0%}")
    st.caption(
        f"Based on data up to the close on {date_text(forecast.as_of)}. For this forecast the "
        f"{model_name.lower()} was refit on all {forecast.training_rows:,} days with a known outcome. "
        "A probability near 50% means the model has little to go on. This is not a recommendation to trade."
    )

# Backtest
st.subheader("Backtest on unseen data")
if evaluation is None:
    st.warning(model_error)
else:
    train, test = evaluation.train, evaluation.test
    result = run_backtest(close, test["prediction"], initial_capital, cost_percent / 100)
    strategy = performance_metrics(result["strategy_value"], result["position"])
    benchmark = performance_metrics(result["benchmark_value"], result["benchmark_position"])

    st.write(
        f"The {model_name.lower()} was trained on {len(train):,} days from {date_text(train.index[0])} to "
        f"{date_text(train.index[-1])}, then tested on the following {len(test):,} days, which it never saw "
        "during training. After each close, the strategy holds the stock for the next day if the model-estimated "
        "probability of a higher close is above 50%, and holds cash otherwise."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Strategy return",
        signed_percent(strategy["total_return"]),
        delta=f"{(strategy['total_return'] - benchmark['total_return']) * 100:+.1f} pp vs buy and hold",
    )
    col2.metric("Buy-and-hold return", signed_percent(benchmark["total_return"]))
    col3.metric(
        "Test accuracy",
        f"{evaluation.accuracy:.1%}",
        delta=f"{(evaluation.accuracy - evaluation.baseline_accuracy) * 100:+.1f} pp vs always 'up'",
        help="Share of test days where the predicted direction was right, compared with always predicting a higher close.",
    )

    st.plotly_chart(
        line_chart(
            {
                "Strategy": (result["strategy_value"], PRIMARY_COLOUR),
                "Buy and hold": (result["benchmark_value"], SECONDARY_COLOUR),
            },
            "Portfolio value",
        )
    )
    st.table(pd.DataFrame({"Strategy": format_metrics(strategy), "Buy and hold": format_metrics(benchmark)}))
    st.caption(
        f"Both start with {initial_capital:,.0f} on {date_text(result.index[0])} and end on "
        f"{date_text(result.index[-1])}. Each buy or sell costs {cost_percent:.2f}% of the portfolio value; "
        "buy-and-hold pays this once when it buys. Sharpe ratio uses daily returns, annualised with √252 and a 0% "
        "risk-free rate. Annualised return is only shown for test periods of at least one year."
    )

    with st.expander("Daily backtest values"):
        backtest_table = pd.DataFrame(
            {
                "Close": result["close"],
                "Position (1 = stock)": result["position"].astype(int),
                "Strategy value": result["strategy_value"],
                "Buy-and-hold value": result["benchmark_value"],
            }
        ).round(2)
        backtest_table.index = backtest_table.index.date
        st.dataframe(backtest_table.iloc[::-1])

# Method and limitations
st.subheader("Method and limitations")
st.markdown(
    """
- **Features.** For each day: the 1-day and 5-day returns, the close relative to its 10- and 20-day moving
  averages, the 10-day average relative to the 20-day average, and the 20-day volatility of daily returns.
  Each uses prices up to that day's close only.
- **Target.** 1 if the next trading day's close is higher than today's close, otherwise 0.
- **Training and testing.** The earliest 70% of days train the model and the latest 30% test it, in time order
  with no shuffling. Logistic regression scales features using training data only.
- **Strategy.** A prediction made after one day's close decides whether the next day is spent in the stock or
  in cash. Cash earns no interest.
- **Limitations.** Results cover one stock over one historical period and ignore taxes, bid-ask spreads,
  slippage and interest on cash. Accuracy close to 50% is normal for daily stock direction, and a good backtest
  can be luck. Past performance does not guarantee future results.
"""
)
