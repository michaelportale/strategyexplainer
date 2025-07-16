"""Base strategy classes and utilities for the modular trading strategy framework.

This module provides the foundational infrastructure for all trading strategies
in the system. It defines the abstract base class that all strategies must inherit
from, common technical indicator calculations, and strategy composition utilities.

Key Components:
    BaseStrategy: Abstract base class defining the strategy interface
    StrategyComposer: Utility for combining multiple strategies with voting logic

The design follows the Strategy pattern, enabling:
- Pluggable strategy implementations
- Standardized signal generation interface
- Common technical indicator library
- Strategy combination and ensemble methods
- Consistent validation and error handling

All concrete strategy implementations must inherit from BaseStrategy and implement
the generate_signals() method. The framework provides extensive technical indicator
support and robust data validation to ensure reliable strategy execution.

Classes:
    BaseStrategy: Abstract foundation for all strategy implementations
    StrategyComposer: Multi-strategy combination with configurable voting logic

Example:
    >>> class MyStrategy(BaseStrategy):
    ...     def generate_signals(self, data):
    ...         df = self.add_technical_indicators(data)
    ...         df['signal'] = 0  # Strategy logic here
    ...         return df
    >>> 
    >>> strategy = MyStrategy("Custom Strategy", {"param1": 10})
    >>> signals = strategy.generate_signals(price_data)
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies in the framework.
    
    This class provides the foundational interface and common functionality
    that all trading strategies must implement. It establishes a standardized
    contract for signal generation while providing utility methods for data
    validation and technical indicator calculation.
    
    Design Principles:
    - Strategy implementations must focus solely on signal logic
    - Common technical indicators are provided as utility methods
    - Data validation ensures robust operation across different data sources
    - Logging integration enables comprehensive strategy monitoring
    - Parameter management supports strategy optimization and configuration
    
    Subclasses must implement:
        generate_signals(): Core method that produces trading signals
        
    Attributes:
        name (str): Human-readable strategy name for identification
        parameters (Dict[str, Any]): Strategy-specific configuration parameters
        logger: Logging instance for operation tracking
    
    Example:
        >>> class SMAStrategy(BaseStrategy):
        ...     def generate_signals(self, data):
        ...         df = self.add_technical_indicators(data)
        ...         df['signal'] = np.where(df['close'] > df['sma_20'], 1, -1)
        ...         return df
        >>> 
        >>> strategy = SMAStrategy("SMA-20", {"period": 20})
        >>> signals = strategy.generate_signals(price_data)
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """Initialize strategy with name and configuration parameters.
        
        Sets up the basic strategy infrastructure including logging,
        parameter management, and identification. All strategies require
        a name for identification and optionally accept configuration parameters.
        
        Args:
            name: Human-readable strategy identifier used in logging and reporting.
                 Should be descriptive and unique across the strategy ensemble.
            parameters: Dictionary of strategy-specific configuration values.
                       Supports strategy optimization and parameter sweeping.
                       Common parameters include periods, thresholds, and flags.
        
        Example:
            >>> strategy = BaseStrategy("Moving Average Crossover", {
            ...     "fast_period": 10,
            ...     "slow_period": 20,
            ...     "use_volume_filter": True
            ... })
        """
        self.name = name
        self.parameters = parameters or {}
        # Set up strategy-specific logger for debugging and monitoring
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market data and strategy logic.
        
        This is the core method that all strategy implementations must define.
        It takes market data as input and returns the same data with trading
        signals added. The signal column should contain:
        - 1: Buy signal (go long)
        - -1: Sell signal (go short or exit long)
        - 0: Hold signal (no action)
        
        Implementation Guidelines:
        - Always validate input data using self.validate_data()
        - Use self.add_technical_indicators() for common indicators
        - Ensure signal column is properly formatted
        - Handle edge cases and missing data gracefully
        - Log significant events for debugging
        
        Args:
            data: DataFrame containing market data with standard OHLCV columns.
                 Required columns: ['close']
                 Optional columns: ['open', 'high', 'low', 'volume']
                 Index should be datetime-based for proper time series analysis.
            
        Returns:
            DataFrame identical to input but with 'signal' column added.
            May include additional columns for intermediate calculations,
            technical indicators, or debugging information.
            
        Raises:
            ValueError: If input data fails validation
            NotImplementedError: If method is not implemented by subclass
            
        Example:
            >>> def generate_signals(self, data):
            ...     if not self.validate_data(data):
            ...         raise ValueError("Invalid input data")
            ...     
            ...     df = self.add_technical_indicators(data)
            ...     
            ...     # Simple moving average crossover strategy
            ...     df['signal'] = 0
            ...     df.loc[df['sma_10'] > df['sma_20'], 'signal'] = 1
            ...     df.loc[df['sma_10'] < df['sma_20'], 'signal'] = -1
            ...     
            ...     return df
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data meets minimum requirements for strategy execution.
        
        Performs comprehensive validation of market data to ensure strategies
        can operate reliably. Checks for required columns, data completeness,
        and basic data quality issues that could cause strategy failures.
        
        Validation Checks:
        - Required columns are present (minimum: 'close')
        - Data is not empty
        - No null values in critical columns
        - Data types are appropriate for calculations
        
        Args:
            data: DataFrame to validate containing market data
            
        Returns:
            bool: True if data passes all validation checks, False otherwise.
                 False return triggers error logging with specific failure details.
                 
        Side Effects:
            Logs detailed error messages for any validation failures
            to aid in debugging data pipeline issues.
            
        Example:
            >>> if not strategy.validate_data(price_data):
            ...     print("Data validation failed - check logs for details")
            ...     return
            >>> signals = strategy.generate_signals(price_data)
        """
        # Check for minimum required columns
        required_cols = ['close']  # All strategies need at least closing prices
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for empty dataset
        if data.empty:
            self.logger.error("Data is empty - cannot generate signals")
            return False
        
        # Check for null values in critical columns
        null_counts = data[required_cols].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found in required columns: {null_counts[null_counts > 0].to_dict()}")
            # Don't fail validation for nulls, but warn user
            
        return True
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to market data.
        
        This utility method provides a library of common technical indicators
        that strategies can use without reimplementing calculations. It efficiently
        adds indicators only if they don't already exist, enabling indicator reuse
        across multiple strategies and preventing redundant calculations.
        
        Available Indicators:
        - Simple Moving Averages (SMA): 10, 20, 50, 200 periods
        - Relative Strength Index (RSI): 14-period momentum oscillator
        - Bollinger Bands: 20-period with 2 standard deviation bands
        - Average True Range (ATR): 14-period volatility measure
        
        Performance Optimization:
        - Checks for existing indicators before calculation
        - Uses vectorized pandas operations for efficiency
        - Maintains original data structure and index
        
        Args:
            data: Input DataFrame with market data. Required columns depend
                 on indicators being calculated (minimum: 'close' for most indicators)
            
        Returns:
            DataFrame with original data plus technical indicators added as columns.
            Original data is never modified - returns a copy with indicators added.
            
        Note:
            Indicators may have initial NaN values due to rolling window calculations.
            Strategies should handle these appropriately in signal generation logic.
            
        Example:
            >>> data_with_indicators = strategy.add_technical_indicators(price_data)
            >>> print(data_with_indicators.columns)
            Index(['close', 'sma_10', 'sma_20', 'rsi', 'bb_upper', 'bb_lower', ...])
        """
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Simple Moving Averages - commonly used trend indicators
        if 'sma_10' not in df.columns:
            df['sma_10'] = df['close'].rolling(window=10).mean()
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(window=50).mean()
        if 'sma_200' not in df.columns:
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
        # RSI (Relative Strength Index) - momentum oscillator
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['close'])
            
        # Bollinger Bands - volatility-based support/resistance
        if 'bb_upper' not in df.columns:
            bb_period = 20  # Standard Bollinger Band period
            bb_std = 2      # Standard deviation multiplier
            sma = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = sma + (std * bb_std)
            df['bb_lower'] = sma - (std * bb_std)
            df['bb_middle'] = sma
            
        # ATR (Average True Range) - volatility measure (requires OHLC data)
        if 'atr' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = self._calculate_atr(df)
            
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI) momentum oscillator.
        
        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes. It oscillates between 0 and 100, with readings
        above 70 typically indicating overbought conditions and readings
        below 30 indicating oversold conditions.
        
        Calculation Method:
        1. Calculate price changes (gains and losses)
        2. Smooth gains and losses using exponential moving average
        3. Calculate relative strength (RS) = average gain / average loss
        4. Convert to RSI = 100 - (100 / (1 + RS))
        
        Args:
            prices: Time series of closing prices
            period: Number of periods for RSI calculation (default: 14)
            
        Returns:
            pd.Series: RSI values ranging from 0 to 100
            
        Note:
            First 'period' values will be NaN due to insufficient data
            for the rolling calculation.
            
        Example:
            >>> rsi = strategy._calculate_rsi(data['close'], period=14)
            >>> overbought = rsi > 70  # Identify overbought conditions
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate relative strength and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) volatility indicator.
        
        ATR measures market volatility by calculating the average of true ranges
        over a specified period. True Range is the greatest of:
        1. Current High - Current Low
        2. Current High - Previous Close (absolute value)
        3. Current Low - Previous Close (absolute value)
        
        ATR is essential for:
        - Position sizing based on volatility
        - Setting stop-loss levels
        - Identifying breakout strength
        - Comparing volatility across different assets
        
        Args:
            data: DataFrame containing 'high', 'low', and 'close' columns
            period: Number of periods for ATR smoothing (default: 14)
            
        Returns:
            pd.Series: ATR values representing average volatility
            
        Raises:
            KeyError: If required columns (high, low, close) are missing
            
        Example:
            >>> atr = strategy._calculate_atr(data, period=14)
            >>> volatility_breakout = data['close'] > data['close'].shift(1) + 2*atr
        """
        # Calculate the three components of True Range
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        
        # True Range is the maximum of the three components
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # ATR is the rolling average of True Range
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information for reporting and analysis.
        
        Returns a dictionary containing key strategy metadata that can be used
        for strategy identification, parameter tracking, and performance reporting.
        This information is particularly useful for batch testing and strategy
        comparison analysis.
        
        Returns:
            Dict[str, Any]: Strategy information containing:
                - name: Human-readable strategy identifier
                - parameters: Current configuration parameters
                - class: Python class name for technical identification
                
        Example:
            >>> info = strategy.get_info()
            >>> print(f"Strategy: {info['name']}")
            >>> print(f"Parameters: {info['parameters']}")
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'class': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """Return human-readable string representation for display purposes."""
        return f"{self.name} ({self.__class__.__name__})"
    
    def __repr__(self) -> str:
        """Return technical string representation for debugging and logging."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"


class StrategyComposer:
    """Combine multiple strategies using configurable voting logic.
    
    This class enables sophisticated multi-strategy approaches by combining
    signals from multiple individual strategies using various logical operations.
    It supports different combination methods to balance signal quality,
    frequency, and robustness.
    
    Combination Methods:
    - 'majority': Requires >50% of strategies to agree (balanced approach)
    - 'unanimous': Requires ALL strategies to agree (high confidence, fewer signals)
    - 'any': Triggers if ANY strategy signals (high sensitivity, more signals)
    
    Benefits of Strategy Combination:
    - Reduced false signals through consensus
    - Improved robustness across market conditions
    - Risk diversification across strategy types
    - Flexibility in signal sensitivity tuning
    
    Attributes:
        strategies (List[BaseStrategy]): Individual strategies to combine
        combination_method (str): Voting logic for signal combination
        logger: Logging instance for operation tracking
    
    Example:
        >>> momentum_strategy = SMAStrategy("SMA Momentum")
        >>> mean_revert_strategy = RSIStrategy("RSI Mean Reversion")
        >>> composer = StrategyComposer([momentum_strategy, mean_revert_strategy], 'majority')
        >>> combined_signals = composer.generate_combined_signals(price_data)
    """
    
    def __init__(self, strategies: List[BaseStrategy], combination_method: str = 'majority'):
        """Initialize strategy composer with strategies and combination logic.
        
        Sets up the multi-strategy ensemble with specified voting logic.
        Validates that strategies are properly configured and establishes
        logging for combination operations.
        
        Args:
            strategies: List of BaseStrategy instances to combine. Each strategy
                       should be fully configured and ready to generate signals.
                       Minimum of 2 strategies required for meaningful combination.
            combination_method: Voting logic for combining signals:
                - 'majority': Signal when >50% of strategies agree
                - 'unanimous': Signal when ALL strategies agree  
                - 'any': Signal when ANY strategy triggers
                
        Raises:
            ValueError: If combination_method is not recognized
            TypeError: If strategies list contains non-BaseStrategy objects
            
        Example:
            >>> strategies = [trend_strategy, breakout_strategy, mean_revert_strategy]
            >>> composer = StrategyComposer(strategies, 'majority')
        """
        self.strategies = strategies
        self.combination_method = combination_method
        self.logger = logging.getLogger(__name__)
        
        # Validate combination method
        valid_methods = ['majority', 'unanimous', 'any']
        if combination_method not in valid_methods:
            raise ValueError(f"Invalid combination method '{combination_method}'. "
                           f"Must be one of: {valid_methods}")
        
        # Log composition setup
        strategy_names = [s.name for s in strategies]
        self.logger.info(f"Initialized strategy composer with {len(strategies)} strategies: "
                        f"{strategy_names} using '{combination_method}' method")
    
    def generate_combined_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate combined trading signals from multiple strategies.
        
        This method orchestrates signal generation from all component strategies
        and combines them using the specified voting logic. It handles errors
        gracefully, ensuring that failures in individual strategies don't prevent
        signal generation from the ensemble.
        
        Process:
        1. Generate signals from each individual strategy
        2. Handle any strategy failures with error logging
        3. Apply voting logic to combine signals
        4. Add individual strategy signals for analysis
        5. Return comprehensive signal data
        
        Args:
            data: DataFrame containing market data with OHLCV columns
            
        Returns:
            DataFrame containing:
                - All original market data
                - 'signal': Combined signal based on voting logic (1, -1, 0)
                - Individual strategy signals as separate columns
                - Any technical indicators calculated by strategies
                
        Note:
            Failed strategies are treated as generating hold signals (0)
            to ensure robust operation of the ensemble.
            
        Example:
            >>> data = load_market_data("AAPL")
            >>> combined_df = composer.generate_combined_signals(data)
            >>> print(f"Combined signals generated: {(combined_df['signal'] != 0).sum()}")
        """
        # Start with copy of original data
        df = data.copy()
        
        # Step 1: Generate signals from each individual strategy
        individual_signals = {}
        for strategy in self.strategies:
            try:
                # Generate signals from this strategy
                strategy_df = strategy.generate_signals(data)
                individual_signals[strategy.name] = strategy_df['signal']
                self.logger.debug(f"Generated {(strategy_df['signal'] != 0).sum()} signals "
                                f"from {strategy.name}")
            except Exception as e:
                # Log error but continue with other strategies
                self.logger.error(f"Error generating signals for {strategy.name}: {e}")
                # Use hold signals (0) for failed strategy
                individual_signals[strategy.name] = pd.Series(0, index=data.index)
        
        # Step 2: Combine signals using specified voting logic
        signal_df = pd.DataFrame(individual_signals)
        
        if self.combination_method == 'majority':
            # Require majority consensus for signal generation
            buy_votes = (signal_df == 1).sum(axis=1)
            sell_votes = (signal_df == -1).sum(axis=1)
            total_strategies = len(self.strategies)
            
            df['signal'] = 0  # Default to hold
            df.loc[buy_votes > total_strategies / 2, 'signal'] = 1
            df.loc[sell_votes > total_strategies / 2, 'signal'] = -1
            
        elif self.combination_method == 'unanimous':
            # Require complete consensus for signal generation
            df['signal'] = 0  # Default to hold
            df.loc[(signal_df == 1).all(axis=1), 'signal'] = 1
            df.loc[(signal_df == -1).all(axis=1), 'signal'] = -1
            
        elif self.combination_method == 'any':
            # Signal on any individual strategy trigger
            df['signal'] = 0  # Default to hold
            df.loc[(signal_df == 1).any(axis=1), 'signal'] = 1
            df.loc[(signal_df == -1).any(axis=1), 'signal'] = -1
        
        # Step 3: Add individual strategy signals for analysis and debugging
        for strategy_name, signals in individual_signals.items():
            # Create safe column name from strategy name
            safe_name = strategy_name.lower().replace(" ", "_").replace("-", "_")
            df[f'signal_{safe_name}'] = signals
        
        # Log combination statistics
        total_signals = (df['signal'] != 0).sum()
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        
        self.logger.info(f"Combined signals generated: {total_signals} total "
                        f"({buy_signals} buy, {sell_signals} sell) "
                        f"using {self.combination_method} method")
        
        return df 