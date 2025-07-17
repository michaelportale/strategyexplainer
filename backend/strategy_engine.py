"""Strategy Engine - Core runner for modular trading strategies.

This module provides the central orchestration layer for the strategy explainer application.
It handles strategy instantiation, execution, backtesting, and results aggregation.

Key Features:
- Unified strategy execution framework
- Modular strategy composition and combination
- Batch backtesting across multiple assets and timeframes
- Risk management integration
- Performance metrics calculation and reporting
- Both legacy and modern configuration support

The StrategyEngine serves as the bridge between the strategy implementations
and the various frontends (CLI, Streamlit app, batch processing).

Classes:
    StrategyEngine: Main orchestration class for strategy execution

Functions:
    main: CLI interface for direct strategy execution

Usage Examples:
    # Single strategy execution
    engine = StrategyEngine()
    strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10})
    result = engine.run_single_strategy(strategy, 'AAPL')
    
    # Batch testing
    results = engine.run_batch_backtest(['AAPL', 'MSFT'], include_combinations=True)
    
    # Strategy combinations
    strategies = [engine.create_strategy('sma_ema_rsi'), engine.create_strategy('rsi')]
    combo_result = engine.run_strategy_combo(strategies, 'AAPL', 'majority')
"""

import sys
import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
import logging
import pandas as pd

# Add the repository root to the Python path for proper module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration management imports
from config.config_manager import get_config_manager

# Strategy implementation imports - all available strategy types
from backend.strategies.base import BaseStrategy, StrategyComposer, StrategyRegistry
# Import all strategy modules to trigger auto-registration
from backend.strategies import momentum, breakout, mean_reversion, meta

# Core infrastructure imports
from backend.simulate import TradingSimulator
from backend.metrics import PerformanceMetrics
from backend.data_loader import DataLoader

# Set up logging with standardized format
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class StrategyEngine:
    """Core strategy engine for executing modular trading strategies.
    
    This class provides the central orchestration layer for strategy execution,
    handling everything from configuration management to result aggregation.
    It supports both individual strategy testing and complex multi-strategy
    combinations with various risk management features.
    
    The engine is designed to be framework-agnostic and can be used from:
    - CLI scripts for automated testing
    - Streamlit frontend for interactive analysis  
    - Batch processing systems for large-scale backtesting
    - Jupyter notebooks for research
    
    Attributes:
        config_manager: Configuration manager instance (if using new config system)
        config (Dict[str, Any]): Configuration dictionary
        data_loader (DataLoader): Data fetching and caching instance
        logger: Logging instance for operation tracking
        
    Note:
        Strategy registry is now handled dynamically via StrategyRegistry singleton.
        All strategies are automatically registered when their modules are imported.
    
    Example:
        >>> engine = StrategyEngine()
        >>> strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10})
        >>> result = engine.run_single_strategy(strategy, 'AAPL')
        >>> print(f"Strategy return: {result['metrics']['Total Return']:.2%}")
    """
    
    # Strategy registry is now handled dynamically via the StrategyRegistry singleton
    # All strategies are automatically registered when their modules are imported
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize strategy engine with configuration management.
        
        The engine supports both the new unified configuration system (config.yaml)
        and legacy configuration files (JSON/YAML) for backward compatibility.
        
        Args:
            config_path: Path to strategy configuration file. If None or points to
                        config.yaml, uses the new ConfigManager system. Otherwise,
                        loads legacy configuration format.
        
        Raises:
            FileNotFoundError: If specified config file doesn't exist
            ValueError: If config file format is unsupported
        """
        # Determine which configuration system to use
        if config_path is None or config_path == "config/config.yaml":
            # Use new unified configuration manager
            self.config_manager = get_config_manager()
            self.config = self.config_manager.config
            logger.info("Using unified configuration system (config.yaml)")
        else:
            # Legacy support for old JSON/YAML configuration files
            self.config_manager = None
            self.config = self._load_legacy_config(config_path)
            logger.info(f"Using legacy configuration from {config_path}")
        
        # Initialize core components
        self.data_loader = DataLoader()  # Handles data fetching and caching
        self.logger = logger             # Consistent logging across the engine
    
    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object with standardized format.
        
        This utility method ensures consistent date parsing across the engine.
        All date inputs should be in YYYY-MM-DD format for compatibility.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            date: Python date object
            
        Raises:
            ValueError: If date string format is invalid
            
        Example:
            >>> engine._parse_date('2023-01-15')
            datetime.date(2023, 1, 15)
        """
        return datetime.strptime(date_str, '%Y-%m-%d').date()
        
    def _load_legacy_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load strategy configuration from legacy JSON/YAML files.
        
        This method provides backward compatibility with older configuration
        formats while the system transitions to the unified config.yaml approach.
        It includes sensible defaults if no configuration is provided.
        
        Args:
            config_path: Path to legacy config file (JSON or YAML format)
            
        Returns:
            Dict[str, Any]: Configuration dictionary with strategy definitions,
                           backtest parameters, and risk management settings
                           
        Note:
            Falls back to default configuration if file loading fails,
            ensuring the engine can always operate even with missing configs.
        """
        if config_path is None:
            # Provide sensible default configuration for standalone operation
            logger.info("No config provided, using default configuration")
            return {
                'strategies': {
                    'sma_ema_rsi': {
                        'enabled': True,
                        'parameters': {
                            'fast_period': 10,      # Short-term moving average period
                            'slow_period': 50,      # Long-term moving average period
                            'rsi_period': 14,       # RSI calculation period
                            'use_rsi_filter': True  # Enable RSI-based signal filtering
                        }
                    }
                },
                'backtest': {
                    'initial_capital': 10000,    # Starting capital for backtests
                    'start_date': '2020-01-01',  # Default backtest start date
                    'end_date': '2024-01-01',    # Default backtest end date
                    'commission': 0.001,         # Trading commission as percentage
                    'slippage': 0.0005          # Market slippage as percentage
                },
                'risk_management': {
                    'stop_loss_pct': None,       # Stop loss percentage (None = disabled)
                    'take_profit_pct': None,     # Take profit percentage (None = disabled)
                    'max_drawdown_pct': 0.20,    # Maximum portfolio drawdown threshold
                    'position_size_pct': 1.0     # Position sizing as percentage of capital
                }
            }
        
        try:
            # Attempt to load configuration file based on extension
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                    logger.info(f"Loaded JSON configuration from {config_path}")
                elif config_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded YAML configuration from {config_path}")
                else:
                    raise ValueError(f"Unsupported config format: {config_path}")
            
            return config
            
        except Exception as e:
            # Fall back to default configuration if loading fails
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            self.logger.warning("Falling back to default configuration")
            return self._load_legacy_config(None)  # Recursive call for default config
    
    def create_strategy(self, strategy_name: str, parameters: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance from the dynamic registry.
        
        This factory method handles the instantiation of strategy objects,
        including any special setup required for specific strategy types.
        It serves as the central point for strategy creation across the engine.
        
        Args:
            strategy_name: Name of strategy to create (must be in StrategyRegistry)
            parameters: Strategy-specific parameters to override defaults
            
        Returns:
            BaseStrategy: Configured strategy instance ready for signal generation
            
        Raises:
            ValueError: If strategy_name is not found in registry
            
        Example:
            >>> strategy = engine.create_strategy('sma_ema_rsi', {
            ...     'fast_period': 10,
            ...     'slow_period': 50,
            ...     'use_rsi_filter': True
            ... })
            >>> print(strategy.name)
            'SMA/EMA with RSI Strategy'
        """
        # Get the dynamic registry instance
        registry = StrategyRegistry.get_instance()
        
        # Get strategy class from dynamic registry  
        strategy_class = registry.get_strategy_class(strategy_name)
        
        # Handle special cases that require additional setup
        if strategy_name == 'sentiment_mean_reversion':
            # Import here to avoid circular imports
            from backend.strategies.meta import MockSentimentProvider
            # Sentiment strategies need a sentiment provider instance
            # Use mock provider for testing/demo purposes
            provider = MockSentimentProvider()
            logger.info("Creating sentiment strategy with mock provider")
            return strategy_class(sentiment_provider=provider, parameters=parameters)
        else:
            # Standard strategy instantiation
            return strategy_class(parameters=parameters)
    
    def get_enabled_strategies(self) -> List[BaseStrategy]:
        """Get list of enabled strategies from configuration.
        
        This method reads the configuration and instantiates all strategies
        that are marked as enabled. It's used for batch testing and automatic
        strategy loading without manual specification.
        
        Returns:
            List[BaseStrategy]: List of configured and enabled strategy instances
            
        Note:
            Logs warnings for strategies that fail to instantiate but continues
            processing other strategies to maintain robust operation.
        """
        strategies = []
        
        # Iterate through configured strategies
        for strategy_name, config in self.config.get('strategies', {}).items():
            if config.get('enabled', False):
                try:
                    # Extract parameters and create strategy instance
                    parameters = config.get('parameters', {})
                    strategy = self.create_strategy(strategy_name, parameters)
                    strategies.append(strategy)
                    self.logger.info(f"Loaded strategy: {strategy}")
                except Exception as e:
                    # Log error but continue with other strategies
                    self.logger.error(f"Failed to create strategy {strategy_name}: {e}")
        
        if not strategies:
            self.logger.warning("No enabled strategies found in configuration")
        
        return strategies
    
    def run_single_strategy(self, 
                           strategy: BaseStrategy,
                           ticker: str,
                           start_date: str = None,
                           end_date: str = None) -> Dict[str, Any]:
        """Run complete backtest for a single strategy on specified asset.
        
        This is the core method for individual strategy testing. It orchestrates
        the complete workflow from data loading through results generation,
        providing a comprehensive performance analysis.
        
        Workflow:
        1. Load and validate market data for the specified ticker and date range
        2. Generate trading signals using the provided strategy
        3. Configure trading simulator with risk management parameters
        4. Execute simulated trading based on the generated signals
        5. Calculate comprehensive performance metrics
        6. Return structured results for analysis or display
        
        Args:
            strategy: Configured strategy instance to test
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Backtest start date in YYYY-MM-DD format (uses config default if None)
            end_date: Backtest end date in YYYY-MM-DD format (uses config default if None)
            
        Returns:
            Dict[str, Any]: Comprehensive results dictionary containing:
                - strategy: Strategy name and configuration
                - ticker: Asset symbol tested
                - start_date/end_date: Actual dates used for testing
                - equity_curve: Portfolio value progression over time
                - trade_log: Detailed record of all trades executed
                - metrics: Performance metrics (returns, Sharpe ratio, drawdown, etc.)
                - signals_df: Raw trading signals generated by strategy
                - strategy_info: Strategy configuration and metadata
                - error: Error message if execution failed
                - success: Boolean indicating execution success
                
        Example:
            >>> strategy = engine.create_strategy('sma_ema_rsi')
            >>> result = engine.run_single_strategy(strategy, 'AAPL', '2023-01-01', '2024-01-01')
            >>> if 'error' not in result:
            ...     print(f"Total Return: {result['metrics']['Total Return']:.2%}")
            ...     print(f"Sharpe Ratio: {result['metrics']['Sharpe Ratio']:.2f}")
        """
        # Extract configuration settings with defaults
        backtest_config = self.config.get('backtest', {})
        start_date = start_date or backtest_config.get('start_date', '2020-01-01')
        end_date = end_date or backtest_config.get('end_date', '2024-01-01')
        
        try:
            # Step 1: Load market data
            self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
            start_date_obj = self._parse_date(start_date)
            end_date_obj = self._parse_date(end_date)
            data = self.data_loader.load_stock_data(ticker, start_date_obj, end_date_obj)
            
            # Validate data availability
            if data.empty:
                raise ValueError(f"No data available for {ticker} in specified date range")
            
            # Step 2: Generate trading signals
            self.logger.info(f"Generating signals with {strategy.name}")
            signals_df = strategy.generate_signals(data)
            
            # Validate signal generation
            if signals_df.empty:
                raise ValueError(f"Strategy {strategy.name} generated no signals")
            
            # Step 3: Configure trading simulator with risk management
            risk_config = self.config.get('risk_management', {})
            sim = TradingSimulator(
                initial_capital=backtest_config.get('initial_capital', 10000),
                commission=backtest_config.get('commission', 0.001),        # Trading costs
                slippage=backtest_config.get('slippage', 0.0005),          # Market impact
                stop_loss_pct=risk_config.get('stop_loss_pct'),            # Risk management
                take_profit_pct=risk_config.get('take_profit_pct'),        # Profit taking
                max_drawdown_pct=risk_config.get('max_drawdown_pct'),      # Portfolio protection
                position_size_pct=risk_config.get('position_size_pct', 1.0) # Position sizing
            )
            
            # Step 4: Execute simulated trading
            self.logger.info("Running backtest simulation")
            equity_curve, trade_log = sim.simulate_strategy(signals_df, signals_df[['signal']])
            
            # Step 5: Calculate comprehensive performance metrics
            perf = PerformanceMetrics()
            metrics = perf.calculate_all_metrics(equity_curve, trade_log)
            
            # Return comprehensive results package
            return {
                'strategy': strategy.name,
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'equity_curve': equity_curve,      # Portfolio value over time
                'trade_log': trade_log,            # Individual trade records
                'metrics': metrics,                # Performance statistics
                'signals_df': signals_df,          # Raw strategy signals
                'strategy_info': strategy.get_info(),  # Strategy configuration
                'success': True                    # Execution success flag
            }
            
        except Exception as e:
            # Return error information for debugging
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
        """Run backtest for a combination of strategies using specified voting logic.
        
        This method enables sophisticated multi-strategy approaches where signals
        from multiple strategies are combined using various logical operations.
        This can improve robustness and reduce false signals compared to single strategies.
        
        Combination Methods:
        - 'majority': Requires >50% of strategies to agree (e.g., 2 out of 3)
        - 'unanimous': Requires ALL strategies to agree (high confidence, fewer signals)
        - 'any': Triggers if ANY strategy signals (high sensitivity, more signals)
        
        Args:
            strategies: List of configured strategy instances to combine
            ticker: Stock ticker symbol to test
            combination_method: Logic for combining signals ('majority', 'unanimous', 'any')
            start_date: Backtest start date (uses config default if None)
            end_date: Backtest end date (uses config default if None)
            
        Returns:
            Dict[str, Any]: Results dictionary similar to run_single_strategy but with:
                - strategies: List of individual strategy names
                - combination_method: Method used for signal combination
                - strategy_info: Information about all component strategies
                
        Raises:
            ValueError: If no strategies provided or invalid combination method
            
        Example:
            >>> momentum_strategy = engine.create_strategy('sma_ema_rsi')
            >>> breakout_strategy = engine.create_strategy('volatility_breakout')
            >>> strategies = [momentum_strategy, breakout_strategy]
            >>> result = engine.run_strategy_combo(strategies, 'AAPL', 'majority')
        """
        # Validate inputs
        if not strategies:
            raise ValueError("No strategies provided for combination")
        
        # Extract configuration with defaults
        backtest_config = self.config.get('backtest', {})
        start_date = start_date or backtest_config.get('start_date', '2020-01-01')
        end_date = end_date or backtest_config.get('end_date', '2024-01-01')
        
        try:
            # Step 1: Load market data (same as single strategy)
            self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
            start_date_obj = self._parse_date(start_date)
            end_date_obj = self._parse_date(end_date)
            data = self.data_loader.load_stock_data(ticker, start_date_obj, end_date_obj)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Step 2: Create strategy composer for signal combination
            composer = StrategyComposer(strategies, combination_method)
            
            # Step 3: Generate combined signals using specified logic
            combo_name = f"{combination_method.title()} of {len(strategies)} strategies"
            self.logger.info(f"Generating combined signals: {combo_name}")
            signals_df = composer.generate_combined_signals(data)
            
            # Step 4: Set up simulator (same configuration as single strategy)
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
            
            # Step 5: Execute combined strategy simulation
            self.logger.info("Running combination backtest simulation")
            equity_curve, trade_log = sim.simulate_strategy(signals_df, signals_df[['signal']])
            
            # Step 6: Calculate performance metrics for combination
            perf = PerformanceMetrics()
            metrics = perf.calculate_all_metrics(equity_curve, trade_log)
            
            # Return comprehensive combination results
            return {
                'strategy': combo_name,
                'strategies': [s.name for s in strategies],      # Individual strategy names
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
                },
                'success': True
            }
            
        except Exception as e:
            # Return error information with combination context
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
        
        This method orchestrates comprehensive backtesting across multiple assets
        and strategies, enabling systematic performance comparison and analysis.
        It can execute both individual strategies and sophisticated strategy
        combinations, providing a complete picture of strategy effectiveness.
        
        The batch backtest process:
        1. Validates and instantiates the requested strategies
        2. Runs each strategy individually on each ticker
        3. Optionally creates and tests strategy combinations
        4. Aggregates all results for comprehensive analysis
        
        Args:
            tickers: List of stock ticker symbols to test (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            strategies: List of strategy names to include in batch testing.
                       If None, uses all enabled strategies from configuration.
            include_combinations: Boolean indicating whether to test strategy
                                combinations using different voting methods.
            
        Returns:
            List[Dict[str, Any]]: Comprehensive list of result dictionaries,
                                 each containing complete backtest results for
                                 a specific strategy-ticker combination.
                                 
        Note:
            Results include both successful backtests and failed attempts
            (marked with 'error' key) to provide complete transparency.
            
        Example:
            >>> # Test specific strategies on multiple assets
            >>> results = engine.run_batch_backtest(
            ...     tickers=['AAPL', 'MSFT', 'GOOGL'],
            ...     strategies=['sma_ema_rsi', 'volatility_breakout'],
            ...     include_combinations=True
            ... )
            >>> print(f"Completed {len(results)} total backtests")
        """
        results = []
        
        # Step 1: Determine which strategies to test
        if strategies is None:
            # Use all enabled strategies from configuration
            strategy_instances = self.get_enabled_strategies()
        else:
            # Create strategy instances based on provided strategy names
            strategy_instances = []
            for strategy_name in strategies:
                try:
                    # Extract strategy configuration and parameters
                    config = self.config.get('strategies', {}).get(strategy_name, {})
                    parameters = config.get('parameters', {})
                    strategy = self.create_strategy(strategy_name, parameters)
                    strategy_instances.append(strategy)
                except Exception as e:
                    # Log errors but continue with other strategies
                    self.logger.error(f"Failed to create strategy {strategy_name}: {e}")
        
        # Validate that we have strategies to test
        if not strategy_instances:
            self.logger.error("No valid strategies available for batch testing")
            return results
        
        self.logger.info(f"Starting batch backtest: {len(tickers)} tickers, "
                        f"{len(strategy_instances)} strategies")
        
        # Step 2: Test individual strategies on each ticker
        total_individual_tests = len(tickers) * len(strategy_instances)
        current_test = 0
        
        for ticker in tickers:
            for strategy in strategy_instances:
                current_test += 1
                self.logger.info(f"Testing {strategy.name} on {ticker} "
                               f"({current_test}/{total_individual_tests})")
                
                # Run individual strategy backtest
                result = self.run_single_strategy(strategy, ticker)
                result['test_type'] = 'single_strategy'  # Mark as individual test
                results.append(result)
        
        # Step 3: Test strategy combinations if requested and feasible
        if include_combinations and len(strategy_instances) > 1:
            self.logger.info("Starting strategy combination tests")
            
            # Test different combination methods for robustness analysis
            combination_methods = ['majority', 'unanimous', 'any']
            total_combo_tests = len(tickers) * len(combination_methods)
            current_combo_test = 0
            
            for ticker in tickers:
                for method in combination_methods:
                    current_combo_test += 1
                    self.logger.info(f"Testing {method} combination on {ticker} "
                                   f"({current_combo_test}/{total_combo_tests})")
                    
                    # Run strategy combination backtest
                    result = self.run_strategy_combo(strategy_instances, ticker, method)
                    result['test_type'] = 'combination'  # Mark as combination test
                    results.append(result)
        
        # Log completion statistics
        successful_tests = len([r for r in results if 'error' not in r])
        failed_tests = len(results) - successful_tests
        
        self.logger.info(f"Batch backtest complete: {len(results)} total tests "
                        f"({successful_tests} successful, {failed_tests} failed)")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "backend/outputs"):
        """Save comprehensive backtest results to organized file structure.
        
        This method creates a structured output directory with multiple file formats
        to facilitate different types of analysis. It saves both summary data for
        quick overview and detailed results for in-depth investigation.
        
        Output Structure:
        - Summary CSV: High-level metrics for all tests in tabular format
        - Individual CSV files: Detailed equity curves and trade logs per test
        - Timestamped filenames: Prevent overwrites and track analysis runs
        
        Args:
            results: List of result dictionaries from batch backtesting
            output_dir: Target directory for saving results. Created if doesn't exist.
                       Defaults to 'backend/outputs' for consistent organization.
                       
        Files Created:
            - strategy_results_summary_TIMESTAMP.csv: Consolidated performance metrics
            - equity_STRATEGY_TICKER_TIMESTAMP_INDEX.csv: Portfolio value progression
            - trades_STRATEGY_TICKER_TIMESTAMP_INDEX.csv: Individual trade records
            
        Example:
            >>> results = engine.run_batch_backtest(['AAPL', 'MSFT'])
            >>> engine.save_results(results, 'analysis_output')
            # Creates organized file structure in analysis_output/
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Create summary DataFrame from all successful results
        summary_data = []
        for result in results:
            if 'error' not in result:  # Only include successful backtests
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
        
        # Save summary CSV if we have any successful results
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = f"{output_dir}/strategy_results_summary_{timestamp}.csv"
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Saved results summary to {summary_path}")
        
        # Step 2: Save detailed results for each successful backtest
        for i, result in enumerate(results):
            if 'error' not in result and 'equity_curve' in result:
                # Create safe filename from strategy and ticker names
                strategy_name = result['strategy'].replace(' ', '_').replace('/', '_')
                ticker = result['ticker']
                
                # Save equity curve (portfolio value progression)
                equity_path = f"{output_dir}/equity_{strategy_name}_{ticker}_{timestamp}_{i}.csv"
                result['equity_curve'].to_csv(equity_path, index=False)
                
                # Save trade log if trades were executed
                if not result['trade_log'].empty:
                    trades_path = f"{output_dir}/trades_{strategy_name}_{ticker}_{timestamp}_{i}.csv"
                    result['trade_log'].to_csv(trades_path, index=False)
    
    def get_strategy_leaderboard(self, results: List[Dict[str, Any]], 
                                metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Generate a comprehensive leaderboard ranking strategies by performance metrics.
        
        This method creates a systematic ranking of all tested strategies based on
        a specified performance metric. It's essential for identifying the most
        effective strategies and understanding relative performance across different
        assets and market conditions.
        
        Supported Ranking Metrics:
        - 'sharpe_ratio': Risk-adjusted returns (higher is better)
        - 'cagr': Compound Annual Growth Rate (higher is better)
        - 'total_return': Absolute return percentage (higher is better)
        - 'max_drawdown': Maximum portfolio decline (lower is better)
        - 'win_rate': Percentage of profitable trades (higher is better)
        - 'profit_factor': Ratio of gross profit to gross loss (higher is better)
        
        Args:
            results: List of result dictionaries from batch backtesting
            metric: Performance metric to use for ranking. Defaults to 'sharpe_ratio'
                   as it provides the best risk-adjusted performance measure.
            
        Returns:
            pd.DataFrame: Ranked DataFrame with columns:
                - rank: Numerical ranking (1 = best performing)
                - strategy: Strategy name
                - ticker: Asset symbol
                - test_type: 'single_strategy' or 'combination'
                - [metric]: Value of the ranking metric
                - Additional metrics for context (total_return, sharpe_ratio, etc.)
                
        Note:
            Returns empty DataFrame if no successful results are available.
            Automatically handles ascending vs descending sort based on metric type.
            
        Example:
            >>> results = engine.run_batch_backtest(['AAPL', 'MSFT'])
            >>> leaderboard = engine.get_strategy_leaderboard(results, 'sharpe_ratio')
            >>> print(leaderboard.head())  # Show top 5 performers
        """
        leaderboard_data = []
        
        # Extract performance data from all successful results
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                
                # Map user-friendly metric names to internal metric keys
                metric_mapping = {
                    'sharpe_ratio': 'Sharpe Ratio',
                    'cagr': 'CAGR',
                    'total_return': 'Total Return',
                    'max_drawdown': 'Max Drawdown',
                    'win_rate': 'Win Rate',
                    'profit_factor': 'Profit Factor'
                }
                
                # Get the actual metric value using mapped name
                metric_key = metric_mapping.get(metric, metric)
                metric_value = metrics.get(metric_key, 0)
                
                # Build comprehensive leaderboard entry
                leaderboard_data.append({
                    'rank': 0,  # Will be assigned after sorting
                    'strategy': result['strategy'],
                    'ticker': result['ticker'],
                    'test_type': result.get('test_type', 'single'),
                    metric: metric_value,  # Primary ranking metric
                    'total_return': metrics.get('Total Return', 0),
                    'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
                    'max_drawdown': metrics.get('Max Drawdown', 0),
                    'total_trades': metrics.get('Total Trades', 0)
                })
        
        # Return empty DataFrame if no data available
        if not leaderboard_data:
            self.logger.warning("No successful results available for leaderboard")
            return pd.DataFrame()
        
        # Create DataFrame and apply appropriate sorting
        df = pd.DataFrame(leaderboard_data)
        
        # Determine sort order based on metric type
        # Most metrics: higher is better (descending sort)
        # Drawdown metrics: lower is better (ascending sort)
        ascending = metric in ['max_drawdown']
        df = df.sort_values(metric, ascending=ascending)
        
        # Assign ranks after sorting (1 = best performing)
        df['rank'] = range(1, len(df) + 1)
        
        return df


def main():
    """Main function providing comprehensive CLI interface for strategy testing.
    
    This function enables command-line execution of the strategy engine with
    various options for different use cases. It supports single strategy testing,
    batch processing, strategy combinations, and performance analysis.
    
    Command Line Options:
        --config: Path to custom configuration file
        --ticker: Stock symbol for single strategy testing
        --strategy: Specific strategy name to test
        --batch: Multiple tickers for batch testing
        --list-strategies: Display all available strategies
        --start/--end: Custom date range for backtesting
        --combinations: Include strategy combinations in batch tests
    
    Usage Examples:
        # List available strategies
        python backend/strategy_engine.py --list-strategies
        
        # Test single strategy on one asset
        python backend/strategy_engine.py --strategy sma_ema_rsi --ticker AAPL
        
        # Batch test on multiple assets
        python backend/strategy_engine.py --batch AAPL MSFT GOOGL --combinations
        
        # Custom date range and configuration
        python backend/strategy_engine.py --config custom.yaml --start 2023-01-01 --end 2024-01-01
    """
    import argparse
    
    # Set up comprehensive command-line argument parser
    parser = argparse.ArgumentParser(
        description="Strategy Engine - Advanced modular trading strategy testing framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-strategies
  %(prog)s --strategy sma_ema_rsi --ticker AAPL
  %(prog)s --batch AAPL MSFT GOOGL --combinations
  %(prog)s --config custom.yaml --start 2023-01-01 --end 2024-01-01
        """
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, 
                       help="Path to strategy configuration file (JSON/YAML)")
    
    # Single strategy testing options
    parser.add_argument("--ticker", type=str, default="AAPL", 
                       help="Stock ticker symbol for testing (default: AAPL)")
    parser.add_argument("--strategy", type=str, 
                       help="Specific strategy name to test")
    
    # Batch testing options
    parser.add_argument("--batch", nargs="+", 
                       help="Run batch test on multiple tickers (space-separated)")
    parser.add_argument("--combinations", action="store_true", 
                       help="Include strategy combinations in batch testing")
    
    # Date range options
    parser.add_argument("--start", type=str, 
                       help="Start date for backtesting (YYYY-MM-DD format)")
    parser.add_argument("--end", type=str, 
                       help="End date for backtesting (YYYY-MM-DD format)")
    
    # Information options
    parser.add_argument("--list-strategies", action="store_true", 
                       help="Display all available strategies and exit")
    
    args = parser.parse_args()
    
    # Initialize strategy engine with specified or default configuration
    try:
        engine = StrategyEngine(args.config)
        logger.info(f"Strategy engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize strategy engine: {e}")
        return 1
    
    # Handle strategy listing request
    if args.list_strategies:
        print("\n=== AVAILABLE STRATEGIES ===")
        print("Strategy Name                | Description")
        print("-" * 50)
        
        # Get the dynamic registry instance
        registry = StrategyRegistry.get_instance()
        
        for name in sorted(registry.list_strategies()):
            # Get brief description from strategy class if available
            strategy_class = registry.get_strategy_class(name)
            description = getattr(strategy_class, '__doc__', 'No description').split('\n')[0]
            print(f"{name:<28} | {description}")
        return 0
    
    # Handle batch testing request
    if args.batch:
        logger.info(f"Starting batch backtest on {len(args.batch)} tickers")
        
        # Run comprehensive batch backtest
        results = engine.run_batch_backtest(
            tickers=args.batch,
            strategies=[args.strategy] if args.strategy else None,
            include_combinations=args.combinations
        )
        
        # Save detailed results to files
        engine.save_results(results)
        logger.info(f"Saved detailed results to backend/outputs/")
        
        # Display performance leaderboard
        leaderboard = engine.get_strategy_leaderboard(results, 'sharpe_ratio')
        if not leaderboard.empty:
            print("\n" + "="*80)
            print("STRATEGY PERFORMANCE LEADERBOARD (Ranked by Sharpe Ratio)")
            print("="*80)
            # Display top 10 results with formatted output
            display_cols = ['rank', 'strategy', 'ticker', 'sharpe_ratio', 'total_return', 'max_drawdown']
            print(leaderboard[display_cols].head(10).to_string(index=False, float_format='{:.3f}'.format))
            print("="*80)
        
        return 0
    
    # Handle single strategy testing request
    elif args.strategy:
        logger.info(f"Testing single strategy: {args.strategy}")
        
        try:
            # Create and test specified strategy
            strategy = engine.create_strategy(args.strategy)
            result = engine.run_single_strategy(strategy, args.ticker, args.start, args.end)
            
            # Display results
            if 'error' in result:
                print(f"\nERROR: {result['error']}")
                return 1
            else:
                print(f"\n" + "="*60)
                print(f"STRATEGY PERFORMANCE: {result['strategy']} on {result['ticker']}")
                print("="*60)
                
                # Display key performance metrics
                metrics = result['metrics']
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric_name:<25}: {value:>12.4f}")
                    else:
                        print(f"{metric_name:<25}: {value:>12}")
                
                print("="*60)
        
        except Exception as e:
            logger.error(f"Failed to test strategy {args.strategy}: {e}")
            return 1
    
    # Default: run all enabled strategies from configuration
    else:
        logger.info("Running all enabled strategies from configuration")
        
        # Get enabled strategies and test each one
        strategies = engine.get_enabled_strategies()
        if not strategies:
            print("No strategies enabled in configuration")
            return 1
        
        # Test each enabled strategy on the specified ticker
        for strategy in strategies:
            result = engine.run_single_strategy(strategy, args.ticker, args.start, args.end)
            
            if 'error' in result:
                print(f"\nError running {strategy.name}: {result['error']}")
            else:
                print(f"\n=== {result['strategy']} on {result['ticker']} ===")
                metrics = result['metrics']
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 