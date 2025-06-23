import sys
from typing import Dict
import numpy as np

import pandas as pd
import yfinance as yf


def fetch_data(ticker, start="2020-01-01", end="2024-01-01"):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        print("Using simulated data instead...")
        dates = pd.date_range(start=start, end=end, freq="B")
        prices = 100 + np.cumsum(np.random.randn(len(dates)))
        return pd.DataFrame({"Close": prices}, index=dates)


def generate_signals(df):
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Signal"] = 0
    df.loc[df["SMA10"] > df["SMA50"], "Signal"] = 1
    df.loc[df["SMA10"] < df["SMA50"], "Signal"] = -1
    return df


def simulate_returns(df: pd.DataFrame, initial_capital: float = 10_000) -> pd.DataFrame:
    """Simulate daily strategy returns and resulting equity curve."""
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy"] = df["Signal"].shift(1).fillna(0) * df["Return"]
    df["Equity"] = initial_capital * (1 + df["Strategy"]).cumprod()
    return df


def calculate_metrics(df: pd.DataFrame, initial_capital: float = 10_000) -> Dict[str, float]:
    """Compute performance metrics for the strategy."""
    # CAGR
    total_days = (df.index[-1] - df.index[0]).days
    years = total_days / 365.25 if total_days else 0
    final_equity = df["Equity"].iloc[-1]
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years else 0.0

    # Sharpe Ratio (risk free rate assumed 0)
    strategy_std = df["Strategy"].std(ddof=0)
    sharpe = ((df["Strategy"].mean() / strategy_std) * (252 ** 0.5)) if strategy_std else 0.0

    # Max Drawdown
    roll_max = df["Equity"].cummax()
    drawdown = df["Equity"] / roll_max - 1
    max_drawdown = drawdown.min()

    return {"CAGR": cagr, "Sharpe": sharpe, "Max Drawdown": max_drawdown}


def run_backtest(ticker: str = "AAPL", initial_capital: float = 10_000, start="2020-01-01", end="2024-01-01"):
    df = fetch_data(ticker, start=start, end=end)
    df = generate_signals(df)
    df = simulate_returns(df, initial_capital)
    metrics = calculate_metrics(df, initial_capital)
    for name, value in metrics.items():
        print(f"{name}: {value:.2%}")
    return df, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run momentum strategy backtest.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df, _ = run_backtest(args.ticker, args.capital, args.start, args.end)
    df.to_csv("backtest_results.csv")
    print(df.tail())
