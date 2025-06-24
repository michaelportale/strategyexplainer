import sys
import os
from typing import Dict
from datetime import datetime
import logging

OUTPUT_DIR = "backend/outputs"

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from backend.simulate import TradingSimulator
from backend.metrics import PerformanceMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fetch_data(ticker, start="2020-01-01", end="2024-01-01"):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        # FLATTEN and CLEAN columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in data.columns.values]
        data.columns = [col.lower().replace(f"_{ticker.lower()}", "") for col in data.columns]
        logger.info(f"DF COLS AFTER FETCH: {data.columns}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        logger.info("Using simulated data instead...")
        dates = pd.date_range(start=start, end=end, freq="B")
        prices = 100 + np.cumsum(np.random.randn(len(dates)))
        return pd.DataFrame({"close": prices}, index=dates)


def generate_signals(df):
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["signal"] = 0
    df.loc[df["sma10"] > df["sma50"], "signal"] = 1
    df.loc[df["sma10"] < df["sma50"], "signal"] = -1
    return df


## Removed unused simulate_returns function


def calculate_metrics(df: pd.DataFrame, initial_capital: float = 10_000) -> Dict[str, float]:
    """Compute performance metrics for the strategy."""
    # CAGR
    total_days = (df.index[-1] - df.index[0]).days
    years = total_days / 365.25 if total_days else 0
    final_equity = df["equity"].iloc[-1]
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years else 0.0

    # Sharpe Ratio (risk free rate assumed 0)
    strategy_std = df["strategy"].std(ddof=0)
    sharpe = ((df["strategy"].mean() / strategy_std) * (252 ** 0.5)) if strategy_std else 0.0

    # Max Drawdown
    roll_max = df["equity"].cummax()
    drawdown = df["equity"] / roll_max - 1
    max_drawdown = drawdown.min()

    return {"CAGR": cagr, "Sharpe": sharpe, "Max Drawdown": max_drawdown}


def run_backtest(ticker: str = "AAPL",
                 initial_capital: float = 10_000,
                 start: str = "2020-01-01",
                 end: str = "2024-01-01",
                 stop_loss_pct: float = None,
                 take_profit_pct: float = None,
                 max_drawdown_pct: float = None,
                 position_size_pct: float = 1.0):
    
    df = fetch_data(ticker, start=start, end=end)
    df = generate_signals(df)

    sim = TradingSimulator(initial_capital=initial_capital)
    equity_curve, trade_log = sim.simulate_strategy(df, df[["signal"]])
    perf = PerformanceMetrics()
    metrics = perf.calculate_all_metrics(equity_curve, trade_log)

    # Save with timestamped filenames in OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    equity_curve.to_csv(f"{OUTPUT_DIR}/equity_curve_{timestamp}.csv", index=False)
    trade_log.to_csv(f"{OUTPUT_DIR}/trade_log_{timestamp}.csv", index=False)
    return equity_curve, trade_log, metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run momentum strategy backtest.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--stop_loss_pct", type=float, default=None, help="Stop loss percentage as decimal (e.g., 0.02 for 2%)")
    parser.add_argument("--take_profit_pct", type=float, default=None, help="Take profit percentage as decimal (e.g., 0.05 for 5%)")
    parser.add_argument("--max_drawdown_pct", type=float, default=None, help="Max portfolio drawdown as decimal (e.g., 0.15 for 15%)")
    parser.add_argument("--position_size_pct", type=float, default=1.0, help="Position size as decimal (e.g., 0.5 for 50% of capital per trade)")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Instantiate TradingSimulator with new arguments
    df = fetch_data(args.ticker, start=args.start, end=args.end)
    df = generate_signals(df)
    sim = TradingSimulator(
        initial_capital=args.capital,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_drawdown_pct=args.max_drawdown_pct,
        position_size_pct=args.position_size_pct
    )
    equity_curve, trade_log = sim.simulate_strategy(df, df[["signal"]])
    perf = PerformanceMetrics()
    metrics = perf.calculate_all_metrics(equity_curve, trade_log)

    # Save with timestamped filenames in OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    equity_curve.to_csv(f"{OUTPUT_DIR}/equity_curve_{timestamp}.csv", index=False)
    trade_log.to_csv(f"{OUTPUT_DIR}/trade_log_{timestamp}.csv", index=False)

    logger.info("Performance Metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    logger.info("\nEquity Curve Tail:")
    logger.info(f"\n{equity_curve.tail()}")

    logger.info("\nTrade Log Tail:")
    logger.info(f"\n{trade_log.tail()}")