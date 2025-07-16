import sys
import os
from typing import Dict
from datetime import datetime
import logging

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load OUTPUT_DIR from configuration
from config.config_manager import get_config_manager
config = get_config_manager()
OUTPUT_DIR = config.get('paths.outputs_dir', 'backend/outputs')

import numpy as np
import pandas as pd
import yfinance as yf
from backend.simulate import TradingSimulator
from backend.metrics import PerformanceMetrics

# Import new modular strategy system
from backend.strategy_engine import StrategyEngine
from backend.strategies import SmaEmaRsiStrategy, CrossoverStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fetch_data(ticker, start="2020-01-01", end="2024-01-01"):
    """Legacy function - maintained for backward compatibility."""
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
    """Legacy signal generation - now uses modular strategy system."""
    logger.info("Using legacy signal generation (SMA10/SMA50 crossover)")
    
    # Use the new CrossoverStrategy for better consistency
    strategy = CrossoverStrategy({'fast_period': 10, 'slow_period': 50})
    signals_df = strategy.generate_signals(df)
    
    # Copy required columns for backward compatibility
    df['sma10'] = signals_df.get('fast_ma', df['close'].rolling(10).mean())
    df['sma50'] = signals_df.get('slow_ma', df['close'].rolling(50).mean())
    df['signal'] = signals_df['signal']
    
    return df


def run_backtest_legacy(ticker: str = "AAPL",
                       initial_capital: float = 10_000,
                       start: str = "2020-01-01",
                       end: str = "2024-01-01",
                       stop_loss_pct: float = None,
                       take_profit_pct: float = None,
                       max_drawdown_pct: float = None,
                       position_size_pct: float = 1.0):
    """Legacy backtest function - maintained for backward compatibility."""
    
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


def run_backtest_modular(ticker: str = "AAPL",
                        strategy_name: str = "sma_ema_rsi",
                        strategy_params: Dict = None,
                        initial_capital: float = 10_000,
                        start: str = "2020-01-01",
                        end: str = "2024-01-01",
                        config_path: str = None):
    """Enhanced backtest using modular strategy system."""
    
    logger.info(f"Running modular backtest: {strategy_name} on {ticker}")
    
    # Initialize strategy engine
    engine = StrategyEngine(config_path)
    
    # Create strategy
    if strategy_params:
        strategy = engine.create_strategy(strategy_name, strategy_params)
    else:
        # Use config defaults
        strategy = engine.create_strategy(strategy_name)
    
    # Run backtest
    result = engine.run_single_strategy(strategy, ticker, start, end)
    
    if 'error' in result:
        logger.error(f"Backtest failed: {result['error']}")
        return None, None, {}
    
    return result['equity_curve'], result['trade_log'], result['metrics']


def run_backtest(ticker: str = "AAPL",
                initial_capital: float = 10_000,
                start: str = "2020-01-01",
                end: str = "2024-01-01",
                stop_loss_pct: float = None,
                take_profit_pct: float = None,
                max_drawdown_pct: float = None,
                position_size_pct: float = 1.0,
                use_modular: bool = True,
                strategy_name: str = "crossover",
                strategy_params: Dict = None):
    """Main backtest function with option to use legacy or modular system."""
    
    if use_modular:
        # Use new modular system
        if strategy_params is None:
            strategy_params = {'fast_period': 10, 'slow_period': 50}
        
        return run_backtest_modular(
            ticker=ticker,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            initial_capital=initial_capital,
            start=start,
            end=end
        )
    else:
        # Use legacy system
        return run_backtest_legacy(
            ticker=ticker,
            initial_capital=initial_capital,
            start=start,
            end=end,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_drawdown_pct=max_drawdown_pct,
            position_size_pct=position_size_pct
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run momentum strategy backtest.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Legacy parameters
    parser.add_argument("--stop_loss_pct", type=float, default=None, help="Stop loss percentage as decimal")
    parser.add_argument("--take_profit_pct", type=float, default=None, help="Take profit percentage as decimal")
    parser.add_argument("--max_drawdown_pct", type=float, default=None, help="Max portfolio drawdown as decimal")
    parser.add_argument("--position_size_pct", type=float, default=1.0, help="Position size as decimal")
    
    # New modular parameters
    parser.add_argument("--legacy", action="store_true", help="Use legacy backtest system")
    parser.add_argument("--strategy", type=str, default="crossover", help="Strategy name")
    parser.add_argument("--fast-period", type=int, default=10, help="Fast MA period")
    parser.add_argument("--slow-period", type=int, default=50, help="Slow MA period")
    parser.add_argument("--config", type=str, help="Path to strategy config file")
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Set up strategy parameters
    strategy_params = {
        'fast_period': args.fast_period,
        'slow_period': args.slow_period,
        'ma_type': 'sma'
    }

    # Run backtest
    equity_curve, trade_log, metrics = run_backtest(
        ticker=args.ticker,
        initial_capital=args.capital,
        start=args.start,
        end=args.end,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_drawdown_pct=args.max_drawdown_pct,
        position_size_pct=args.position_size_pct,
        use_modular=not args.legacy,
        strategy_name=args.strategy,
        strategy_params=strategy_params
    )

    if metrics:
        logger.info("Performance Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")

        if equity_curve is not None and not equity_curve.empty:
            logger.info("\nEquity Curve Tail:")
            logger.info(f"\n{equity_curve.tail()}")

        if trade_log is not None and not trade_log.empty:
            logger.info("\nTrade Log Tail:")
            logger.info(f"\n{trade_log.tail()}")
    else:
        logger.error("Backtest failed - no results to display")