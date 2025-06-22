import pandas as pd
import yfinance as yf


def fetch_data(ticker, start="2020-01-01", end="2024-01-01"):
    return yf.download(ticker, start=start, end=end)


def generate_signals(df):
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Signal"] = 0
    df.loc[df["SMA10"] > df["SMA50"], "Signal"] = 1
    df.loc[df["SMA10"] < df["SMA50"], "Signal"] = -1
    return df


def simulate_returns(df):
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Return"]
    return df


def run_backtest(ticker):
    df = fetch_data(ticker)
    df = generate_signals(df)
    df = simulate_returns(df)
    return df


if __name__ == "__main__":
    df = run_backtest("AAPL")
    df.to_csv("backtest_results.csv")
    print(df.tail())
