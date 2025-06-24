"""Strategy Engine - Core runner for modular trading strategies.

Takes config, executes selected strategies, plugs into existing backtest/sim framework.
"""

import sys
import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
import logging
import pandas as pd

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.strategies.base import BaseStrategy, StrategyComposer
from backend.strategies.sma_ema_rsi import SmaEmaRsiStrategy, CrossoverStrategy, RsiStrategy
from backend.strategies.vol_breakout import VolatilityBreakoutStrategy, ChannelBreakoutStrategy, VolumeBreakoutStrategy
from backend.strategies.mean_reversion import (
    BollingerBandMeanReversionStrategy, ZScoreMeanReversionStrategy, 
    RSIMeanReversionStrategy, MeanReversionComboStrategy
)
from backend.strategies.regime_switch import RegimeGatedStrategy, RegimeSwitchStrategy, RegimeDetector
from backend.strategies.sentiment_overlay import SentimentOverlayStrategy, SentimentMeanReversionStrategy, MockSentimentProvider
from backend.simulate import TradingSimulator
from backend.metrics import PerformanceMetrics
from backend.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class StrategyEngine:
    """Core strategy engine for executing modular trading strategies."""
    
    # Strategy registry mapping string names to classes
    STRATEGY_REGISTRY = {
        'sma_ema_rsi': SmaEmaRsiStrategy,
        'crossover': CrossoverStrategy,
        'rsi': RsiStrategy,
        'volatility_breakout': VolatilityBreakoutStrategy,
        'channel_breakout': ChannelBreakoutStrategy,
        'volume_breakout': VolumeBreakoutStrategy,
        'bollinger_mean_reversion': BollingerBandMeanReversionStrategy,
        'zscore_mean_reversion': ZScoreMeanReversionStrategy,
        'rsi_mean_reversion': RSIMeanReversionStrategy,
        'mean_reversion_combo': MeanReversionComboStrategy,
        'sentiment_mean_reversion': SentimentMeanReversionStrategy,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize strategy engine.
        
        Args:
            config_path: Path to strategy configuration file
        """
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.logger = logger
    
    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            Date object
        """
        return datetime.strptime(date_str, '%Y-%m-%d').date()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load strategy configuration.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Default config
            return {
                'strategies': {
                    'sma_ema_rsi': {
                        'enabled': True,
                        'parameters': {
                            'fast_period': 10,
                            'slow_period': 50,
                            'rsi_period': 14,
                            'use_rsi_filter': True
                        }
                    }
                },
                'backtest': {
                    'initial_capital': 10000,
                    'start_date': '2020-01-01',
                    'end_date': '2024-01-01',
                    'commission': 0.001,
                    'slippage': 0.0005
                },
                'risk_management': {
                    'stop_loss_pct': None,
                    'take_profit_pct': None,
                    'max_drawdown_pct': 0.20,
                    'position_size_pct': 1.0
                }
            }
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return self._load_config(None)  # Fall back to default
    
    def create_strategy(self, strategy_name: str, parameters: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance.
        
        Args:
            strategy_name: Name of strategy to create
            parameters: Strategy parameters
            
        Returns:
            Strategy instance
        """
        if strategy_name not in self.STRATEGY_REGISTRY:
            available = list(self.STRATEGY_REGISTRY.keys())
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
        
        strategy_class = self.STRATEGY_REGISTRY[strategy_name]
        
        # Special handling for strategies that need additional setup
        if strategy_name == 'sentiment_mean_reversion':
            # Use mock sentiment provider for testing
            provider = MockSentimentProvider()
            return strategy_class(sentiment_provider=provider, parameters=parameters)
        else:
            return strategy_class(parameters=parameters)
    
    def get_enabled_strategies(self) -> List[BaseStrategy]:
        """Get list of enabled strategies from config.
        
        Returns:
            List of strategy instances
        """
        strategies = []
        
        for strategy_name, config in self.config.get('strategies', {}).items():
            if config.get('enabled', False):
                try:
                    parameters = config.get('parameters', {})
                    strategy = self.create_strategy(strategy_name, parameters)
                    strategies.append(strategy)
                    self.logger.info(f"Loaded strategy: {strategy}")
                except Exception as e:
                    self.logger.error(f"Failed to create strategy {strategy_name}: {e}")
        
        return strategies
    
    def run_single_strategy(self, 
                           strategy: BaseStrategy,
                           ticker: str,
                           start_date: str = None,
                           end_date: str = None) -> Dict[str, Any]:
        """Run backtest for a single strategy.
        
        Args:
            strategy: Strategy instance
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with results
        """
        # Get dates from config if not provided
        backtest_config = self.config.get('backtest', {})
        start_date = start_date or backtest_config.get('start_date', '2020-01-01')
        end_date = end_date or backtest_config.get('end_date', '2024-01-01')
        
        try:
            # Load data
            self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
            start_date_obj = self._parse_date(start_date)
            end_date_obj = self._parse_date(end_date)
            data = self.data_loader.load_stock_data(ticker, start_date_obj, end_date_obj)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Generate signals
            self.logger.info(f"Generating signals with {strategy.name}")
            signals_df = strategy.generate_signals(data)
            
            # Set up simulator
            risk_config = self.config.get('risk_management', {})
            sim = TradingSimulator(
                initial_capital=backtest_config.get('initial_capital', 10000),
                commission=backtest_config.get('commission', 0.001),
                slippage=backtest_config.get('slippage', 0.0005),
                stop_loss_pct=risk_config.get('stop_loss_pct'),
                take_profit_pct=risk_config.get('take_profit_pct'),
                max_drawdown_pct=risk_config.get('max_drawdown_pct'),
                position_size_pct=risk_config.get('position_size_pct', 1.0)
            )
            
            # Run simulation
            self.logger.info("Running backtest simulation")
            equity_curve, trade_log = sim.simulate_strategy(signals_df, signals_df[['signal']])
            
            # Calculate metrics
            perf = PerformanceMetrics()
            metrics = perf.calculate_all_metrics(equity_curve, trade_log)
            
            return {
                'strategy': strategy.name,
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'equity_curve': equity_curve,
                'trade_log': trade_log,
                'metrics': metrics,
                'signals_df': signals_df,
                'strategy_info': strategy.get_info()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run strategy {strategy.name} on {ticker}: {e}")
            return {
                'strategy': strategy.name,
                'ticker': ticker,
                'error': str(e),
                'success': False
            }
    
    def run_strategy_combo(self,
                          strategies: List[BaseStrategy],
                          ticker: str,
                          combination_method: str = 'majority',
                          start_date: str = None,
                          end_date: str = None) -> Dict[str, Any]:
        """Run backtest for a combination of strategies.
        
        Args:
            strategies: List of strategy instances
            ticker: Stock ticker symbol
            combination_method: How to combine signals ('majority', 'unanimous', 'any')
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with results
        """
        if not strategies:
            raise ValueError("No strategies provided")
        
        # Get dates from config if not provided
        backtest_config = self.config.get('backtest', {})
        start_date = start_date or backtest_config.get('start_date', '2020-01-01')
        end_date = end_date or backtest_config.get('end_date', '2024-01-01')
        
        try:
            # Load data
            self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
            start_date_obj = self._parse_date(start_date)
            end_date_obj = self._parse_date(end_date)
            data = self.data_loader.load_stock_data(ticker, start_date_obj, end_date_obj)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Create strategy composer
            composer = StrategyComposer(strategies, combination_method)
            
            # Generate combined signals
            combo_name = f"{combination_method.title()} of {len(strategies)} strategies"
            self.logger.info(f"Generating combined signals: {combo_name}")
            signals_df = composer.generate_combined_signals(data)
            
            # Set up simulator
            risk_config = self.config.get('risk_management', {})
            sim = TradingSimulator(
                initial_capital=backtest_config.get('initial_capital', 10000),
                commission=backtest_config.get('commission', 0.001),
                slippage=backtest_config.get('slippage', 0.0005),
                stop_loss_pct=risk_config.get('stop_loss_pct'),
                take_profit_pct=risk_config.get('take_profit_pct'),
                max_drawdown_pct=risk_config.get('max_drawdown_pct'),
                position_size_pct=risk_config.get('position_size_pct', 1.0)
            )
            
            # Run simulation
            self.logger.info("Running combination backtest simulation")
            equity_curve, trade_log = sim.simulate_strategy(signals_df, signals_df[['signal']])
            
            # Calculate metrics
            perf = PerformanceMetrics()
            metrics = perf.calculate_all_metrics(equity_curve, trade_log)
            
            return {
                'strategy': combo_name,
                'strategies': [s.name for s in strategies],
                'combination_method': combination_method,
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'equity_curve': equity_curve,
                'trade_log': trade_log,
                'metrics': metrics,
                'signals_df': signals_df,
                'strategy_info': {
                    'individual_strategies': [s.get_info() for s in strategies],
                    'combination_method': combination_method
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run strategy combo on {ticker}: {e}")
            return {
                'strategy': combo_name,
                'ticker': ticker,
                'error': str(e),
                'success': False
            }
    
    def run_batch_backtest(self, 
                          tickers: List[str],
                          strategies: List[str] = None,
                          include_combinations: bool = True) -> List[Dict[str, Any]]:
        """Run batch backtest across multiple tickers and strategies.
        
        Args:
            tickers: List of ticker symbols
            strategies: List of strategy names (None = use all enabled)
            include_combinations: Whether to include strategy combinations
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # Get strategies to test
        if strategies is None:
            strategy_instances = self.get_enabled_strategies()
        else:
            strategy_instances = []
            for strategy_name in strategies:
                try:
                    config = self.config.get('strategies', {}).get(strategy_name, {})
                    parameters = config.get('parameters', {})
                    strategy = self.create_strategy(strategy_name, parameters)
                    strategy_instances.append(strategy)
                except Exception as e:
                    self.logger.error(f"Failed to create strategy {strategy_name}: {e}")
        
        if not strategy_instances:
            self.logger.error("No valid strategies to test")
            return results
        
        self.logger.info(f"Running batch backtest: {len(tickers)} tickers, "
                        f"{len(strategy_instances)} strategies")
        
        # Test individual strategies
        for ticker in tickers:
            for strategy in strategy_instances:
                self.logger.info(f"Testing {strategy.name} on {ticker}")
                result = self.run_single_strategy(strategy, ticker)
                result['test_type'] = 'single_strategy'
                results.append(result)
        
        # Test strategy combinations if requested
        if include_combinations and len(strategy_instances) > 1:
            combination_methods = ['majority', 'unanimous', 'any']
            
            for ticker in tickers:
                for method in combination_methods:
                    self.logger.info(f"Testing {method} combination on {ticker}")
                    result = self.run_strategy_combo(strategy_instances, ticker, method)
                    result['test_type'] = 'combination'
                    results.append(result)
        
        self.logger.info(f"Batch backtest complete: {len(results)} total tests")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "backend/outputs"):
        """Save backtest results to files.
        
        Args:
            results: List of result dictionaries
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_data = []
        for result in results:
            if 'error' not in result:
                metrics = result.get('metrics', {})
                summary_data.append({
                    'strategy': result['strategy'],
                    'ticker': result['ticker'],
                    'test_type': result.get('test_type', 'single'),
                    'total_return': metrics.get('Total Return', 0),
                    'cagr': metrics.get('CAGR', 0),
                    'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
                    'max_drawdown': metrics.get('Max Drawdown', 0),
                    'win_rate': metrics.get('Win Rate', 0),
                    'profit_factor': metrics.get('Profit Factor', 0),
                    'total_trades': metrics.get('Total Trades', 0)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = f"{output_dir}/strategy_results_summary_{timestamp}.csv"
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Saved results summary to {summary_path}")
        
        # Save detailed results
        for i, result in enumerate(results):
            if 'error' not in result and 'equity_curve' in result:
                strategy_name = result['strategy'].replace(' ', '_').replace('/', '_')
                ticker = result['ticker']
                
                # Save equity curve
                equity_path = f"{output_dir}/equity_{strategy_name}_{ticker}_{timestamp}_{i}.csv"
                result['equity_curve'].to_csv(equity_path, index=False)
                
                # Save trade log
                if not result['trade_log'].empty:
                    trades_path = f"{output_dir}/trades_{strategy_name}_{ticker}_{timestamp}_{i}.csv"
                    result['trade_log'].to_csv(trades_path, index=False)
    
    def get_strategy_leaderboard(self, results: List[Dict[str, Any]], 
                                metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Get leaderboard of strategies ranked by performance metric.
        
        Args:
            results: List of result dictionaries
            metric: Metric to rank by ('sharpe_ratio', 'cagr', 'total_return', etc.)
            
        Returns:
            DataFrame with strategy rankings
        """
        leaderboard_data = []
        
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                
                # Map metric names
                metric_mapping = {
                    'sharpe_ratio': 'Sharpe Ratio',
                    'cagr': 'CAGR',
                    'total_return': 'Total Return',
                    'max_drawdown': 'Max Drawdown',
                    'win_rate': 'Win Rate',
                    'profit_factor': 'Profit Factor'
                }
                
                metric_key = metric_mapping.get(metric, metric)
                metric_value = metrics.get(metric_key, 0)
                
                leaderboard_data.append({
                    'rank': 0,  # Will be set after sorting
                    'strategy': result['strategy'],
                    'ticker': result['ticker'],
                    'test_type': result.get('test_type', 'single'),
                    metric: metric_value,
                    'total_return': metrics.get('Total Return', 0),
                    'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
                    'max_drawdown': metrics.get('Max Drawdown', 0),
                    'total_trades': metrics.get('Total Trades', 0)
                })
        
        if not leaderboard_data:
            return pd.DataFrame()
        
        # Create DataFrame and sort
        df = pd.DataFrame(leaderboard_data)
        
        # Sort descending for most metrics, ascending for drawdown
        ascending = metric in ['max_drawdown']
        df = df.sort_values(metric, ascending=ascending)
        df['rank'] = range(1, len(df) + 1)
        
        return df


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Engine - Run modular trading strategies")
    parser.add_argument("--config", type=str, help="Path to strategy config file")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--strategy", type=str, help="Strategy name to run")
    parser.add_argument("--batch", nargs="+", help="Run batch test on multiple tickers")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--combinations", action="store_true", help="Include strategy combinations")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = StrategyEngine(args.config)
    
    # List strategies if requested
    if args.list_strategies:
        print("Available strategies:")
        for name in engine.STRATEGY_REGISTRY.keys():
            print(f"  - {name}")
        return
    
    # Run batch test
    if args.batch:
        results = engine.run_batch_backtest(
            tickers=args.batch,
            strategies=[args.strategy] if args.strategy else None,
            include_combinations=args.combinations
        )
        
        # Save results
        engine.save_results(results)
        
        # Show leaderboard
        leaderboard = engine.get_strategy_leaderboard(results, 'sharpe_ratio')
        if not leaderboard.empty:
            print("\n=== STRATEGY LEADERBOARD (by Sharpe Ratio) ===")
            print(leaderboard.to_string(index=False))
    
    # Run single strategy
    elif args.strategy:
        strategy = engine.create_strategy(args.strategy)
        result = engine.run_single_strategy(strategy, args.ticker, args.start, args.end)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n=== {result['strategy']} on {result['ticker']} ===")
            for metric, value in result['metrics'].items():
                print(f"{metric}: {value}")
    
    # Run enabled strategies from config
    else:
        strategies = engine.get_enabled_strategies()
        if not strategies:
            print("No strategies enabled in config")
            return
        
        for strategy in strategies:
            result = engine.run_single_strategy(strategy, args.ticker, args.start, args.end)
            
            if 'error' in result:
                print(f"Error running {strategy.name}: {result['error']}")
            else:
                print(f"\n=== {result['strategy']} on {result['ticker']} ===")
                for metric, value in result['metrics'].items():
                    print(f"{metric}: {value}")


if __name__ == "__main__":
    main() 