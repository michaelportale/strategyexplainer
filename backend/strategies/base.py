"""Base strategy classes and utilities for the modular trading strategy framework.

Abstract base class for all strategy implementations using the Strategy pattern.
Provides technical indicator calculations and strategy composition utilities.

Classes:
    BaseStrategy: Abstract base class defining strategy interface
    StrategyComposer: Multi-strategy combination with voting logic
    StrategyRegistry: Dynamic strategy registration and discovery
    
Key requirement: All strategies must implement generate_signals() method.
"""

from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Type
import logging
import sys
import os

# Add utils to path for logging framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_config import (
    LoggerManager, 
    StrategyError, 
    DataError, 
    ValidationError,
    with_error_handling,
    with_performance_logging,
    log_function_entry,
    log_function_exit,
    safe_execute
)


class StrategyRegistry:
    """Dynamic strategy registry that automatically tracks all strategy implementations.
    
    This singleton class maintains a registry of all strategy classes that inherit from
    BaseStrategy. It provides thread-safe registration and lookup capabilities, enabling
    dynamic strategy instantiation without manual registry maintenance.
    
    The registry automatically handles:
    - Strategy registration during class definition
    - Name collision detection and resolution
    - Category-based organization
    - Lookup by name or class
    - Listing available strategies
    
    Example:
        >>> registry = StrategyRegistry.get_instance()
        >>> registry.list_strategies()
        ['sma_ema_rsi', 'crossover', 'volatility_breakout', ...]
        >>> strategy_class = registry.get_strategy_class('sma_ema_rsi')
        >>> strategy = strategy_class(parameters={'fast_period': 10})
    """
    
    _instance = None
    _strategies = {}
    
    def __new__(cls):
        """Singleton pattern to ensure one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'StrategyRegistry':
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, name: str, strategy_class: Type['BaseStrategy'], 
                 category: str = 'general') -> None:
        """Register a strategy class with the registry.
        
        Args:
            name: Unique identifier for the strategy (snake_case recommended)
            strategy_class: The strategy class to register
            category: Strategy category for organization (default: 'general')
            
        Raises:
            ValueError: If strategy name already exists with different class
        """
        if name in self._strategies:
            existing_class = self._strategies[name]['class']
            if existing_class != strategy_class:
                logging.warning(
                    f"Strategy name '{name}' already registered with class {existing_class.__name__}. "
                    f"Skipping registration of {strategy_class.__name__}."
                )
                return
        
        self._strategies[name] = {
            'class': strategy_class,
            'category': category,
            'module': strategy_class.__module__
        }
        logging.debug(f"Registered strategy '{name}' -> {strategy_class.__name__}")
    
    def get_strategy_class(self, name: str) -> Type['BaseStrategy']:
        """Get strategy class by name.
        
        Args:
            name: Strategy name to lookup
            
        Returns:
            Strategy class
            
        Raises:
            ValueError: If strategy name not found
        """
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        
        return self._strategies[name]['class']
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def list_strategies_by_category(self) -> Dict[str, List[str]]:
        """Group strategies by category."""
        categories = {}
        for name, info in self._strategies.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        return categories
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a registered strategy."""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy '{name}'")
        return self._strategies[name].copy()


class StrategyMeta(ABCMeta):
    """Metaclass that automatically registers strategies with the registry.
    
    This metaclass intercepts strategy class creation and automatically registers
    each concrete strategy class (non-abstract) with the StrategyRegistry. It uses
    naming conventions and strategy attributes to determine registration details.
    
    Registration Logic:
    - Only registers concrete classes (not abstract base classes)
    - Uses 'strategy_name' class attribute if present, otherwise derives from class name
    - Uses 'strategy_category' class attribute if present, otherwise 'general'
    - Converts CamelCase class names to snake_case for registry names
    
    This approach eliminates the need for manual registry maintenance while ensuring
    all strategy implementations are automatically discoverable.
    """
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        """Create class and register it if it's a concrete strategy."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # Only register concrete strategy classes, not abstract base classes
        if (hasattr(cls, 'generate_signals') and 
            getattr(cls.generate_signals, '__isabstractmethod__', False) is False and
            name != 'BaseStrategy'):
            
            # Get or derive strategy name
            strategy_name = getattr(cls, 'strategy_name', None)
            if strategy_name is None:
                # Convert CamelCase to snake_case
                strategy_name = ''.join(
                    ['_' + c.lower() if c.isupper() and i > 0 else c.lower() 
                     for i, c in enumerate(name)]
                )
                # Remove 'strategy' suffix if present
                if strategy_name.endswith('_strategy'):
                    strategy_name = strategy_name[:-9]
            
            # Get category from class attribute or default
            category = getattr(cls, 'strategy_category', 'general')
            
            # Register with the global registry
            registry = StrategyRegistry.get_instance()
            registry.register(strategy_name, cls, category)
        
        return cls


class BaseStrategy(ABC, metaclass=StrategyMeta):
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
        # Set up strategy-specific logger using the centralized logging framework
        self.logger = LoggerManager.get_logger(f"strategy.{self.__class__.__name__}")
        
        # Log strategy initialization
        self.logger.info(f"Initialized strategy '{name}' with {len(self.parameters)} parameters")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Strategy parameters: {self.parameters}")
        
        # Validate parameters on initialization
        self._validate_parameters()
        
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
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters for common issues.
        
        This method provides basic parameter validation that all strategies can use.
        Subclasses should override this method to add strategy-specific validation.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            # Check for common parameter issues
            for key, value in self.parameters.items():
                # Check for negative periods (but allow zero for delay parameters)
                if 'period' in key.lower() and isinstance(value, (int, float)):
                    if 'delay' not in key.lower() and value <= 0:
                        raise ValidationError(
                            f"Parameter '{key}' must be positive, got {value}",
                            context={'parameter': key, 'value': value, 'strategy': self.name}
                        )
                    elif 'delay' in key.lower() and value < 0:
                        raise ValidationError(
                            f"Parameter '{key}' must be non-negative, got {value}",
                            context={'parameter': key, 'value': value, 'strategy': self.name}
                        )
                
                # Check for invalid thresholds
                if 'threshold' in key.lower() and isinstance(value, (int, float)):
                    if not (0 <= value <= 1) and 'pct' in key.lower():
                        self.logger.warning(f"Parameter '{key}' = {value} may be outside normal range [0, 1]")
                
                # Check for None values where they shouldn't be
                if value is None and key in ['period', 'window', 'lookback']:
                    raise ValidationError(
                        f"Parameter '{key}' cannot be None",
                        context={'parameter': key, 'strategy': self.name}
                    )
            
            self.logger.debug("Parameter validation completed successfully")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                # Wrap unexpected errors
                raise ValidationError(
                    f"Unexpected error during parameter validation: {str(e)}",
                    context={'strategy': self.name, 'parameters': self.parameters}
                ) from e
    
    @with_error_handling(fallback_value=False, log_errors=True)
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data meets minimum requirements for strategy execution.
        
        Performs comprehensive validation of market data to ensure strategies
        can operate reliably. Checks for required columns, data completeness,
        and basic data quality issues that could cause strategy failures.
        
        Validation Checks:
        - Input is actually a DataFrame
        - Required columns are present (minimum: 'close')
        - Data is not empty and has sufficient rows
        - No excessive null values in critical columns
        - Data types are appropriate for calculations
        - Basic data quality checks (no infinite values, reasonable ranges)
        
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
        log_function_entry(self.logger, "validate_data", data_shape=getattr(data, 'shape', 'unknown'))
        
        try:
            # Check if input is a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise DataError(
                    f"Expected pandas DataFrame, got {type(data).__name__}",
                    context={'strategy': self.name, 'data_type': type(data).__name__}
                )
            
            # Check for minimum required columns
            required_cols = ['close']  # All strategies need at least closing prices
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise DataError(
                    f"Missing required columns: {missing_cols}",
                    context={
                        'strategy': self.name,
                        'missing_columns': missing_cols,
                        'available_columns': list(data.columns)
                    }
                )
            
            # Check for empty dataset
            if data.empty:
                raise DataError(
                    "Data is empty - cannot generate signals",
                    context={'strategy': self.name}
                )
            
            # Check for insufficient data
            min_rows = self._get_minimum_data_rows()
            if len(data) < min_rows:
                self.logger.warning(
                    f"Data has only {len(data)} rows, but strategy needs at least {min_rows} rows"
                )
            
            # Check for null values in critical columns
            null_counts = data[required_cols].isnull().sum()
            if null_counts.any():
                null_percentage = (null_counts / len(data) * 100).round(2)
                self.logger.warning(
                    f"Null values found in critical columns: {dict(zip(null_counts.index, null_percentage))}%"
                )
                
                # Fail validation if too many nulls
                if any(null_percentage > 50):  # More than 50% nulls
                    raise DataError(
                        "Too many null values in critical columns",
                        context={
                            'strategy': self.name,
                            'null_percentages': dict(zip(null_counts.index, null_percentage))
                        }
                    )
            
            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            inf_counts = np.isinf(data[numeric_cols]).sum()
            if inf_counts.any():
                self.logger.warning(f"Infinite values found: {inf_counts[inf_counts > 0].to_dict()}")
            
            # Check for reasonable price ranges (prices should be positive)
            if 'close' in data.columns:
                if (data['close'] <= 0).any():
                    negative_count = (data['close'] <= 0).sum()
                    self.logger.warning(f"Found {negative_count} non-positive close prices")
                    
                    # Fail validation if all prices are non-positive
                    if (data['close'] <= 0).all():
                        raise DataError(
                            "All close prices are non-positive",
                            context={'strategy': self.name}
                        )
            
            # Data type validation
            expected_numeric_cols = ['close', 'open', 'high', 'low', 'volume']
            available_numeric_cols = [col for col in expected_numeric_cols if col in data.columns]
            
            for col in available_numeric_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        # Try to convert to numeric
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        self.logger.info(f"Converted column '{col}' to numeric type")
                    except Exception:
                        self.logger.warning(f"Column '{col}' is not numeric and cannot be converted")
            
            self.logger.debug(f"Data validation passed: {len(data)} rows, {len(data.columns)} columns")
            return True
            
        except (DataError, ValidationError) as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(f"Unexpected error during data validation: {str(e)}", exc_info=True)
            return False
    
    def _get_minimum_data_rows(self) -> int:
        """Get minimum number of data rows needed for this strategy.
        
        This method can be overridden by strategies that need more data points
        for their calculations (e.g., strategies with long moving averages).
        
        Returns:
            int: Minimum number of data rows needed
        """
        # Look for period parameters to estimate minimum data needed
        periods = []
        for key, value in self.parameters.items():
            if 'period' in key.lower() and isinstance(value, (int, float)) and value > 0:
                periods.append(int(value))
        
        # Return the largest period plus buffer, or default minimum
        if periods:
            return max(periods) + 10  # Add buffer for rolling calculations
        else:
            return 20  # Default minimum
    
    @with_error_handling(fallback_value=None, log_errors=True)
    @with_performance_logging(threshold_ms=500.0)
    def add_technical_indicators(self, data: pd.DataFrame, custom_periods: Dict[str, Any] = None) -> pd.DataFrame:
        """Add comprehensive technical indicators to market data.
        
        This utility method provides a library of common technical indicators
        that strategies can use without reimplementing calculations. It efficiently
        adds indicators only if they don't already exist, enabling indicator reuse
        across multiple strategies and preventing redundant calculations.
        
        Available Indicators:
        - Simple Moving Averages (SMA): Configurable periods
        - Exponential Moving Averages (EMA): Configurable periods
        - Relative Strength Index (RSI): Configurable period momentum oscillator
        - Bollinger Bands: Configurable period and deviation bands
        - Average True Range (ATR): Configurable period volatility measure
        - Volume indicators: Volume averages and ratios
        - Z-Score: Statistical price normalization
        
        Performance Optimization:
        - Checks for existing indicators before calculation
        - Uses vectorized pandas operations for efficiency
        - Maintains original data structure and index
        
        Args:
            data: Input DataFrame with market data. Required columns depend
                 on indicators being calculated (minimum: 'close' for most indicators)
            custom_periods: Optional dictionary to override default periods
                           e.g., {'sma_periods': [5, 10], 'rsi_period': 21}
            
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
        
        # Set up default periods, allowing customization
        periods = {
            'sma_periods': [10, 20, 50, 200],
            'ema_periods': [10, 20, 50],
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'atr_period': 14,
            'volume_period': 20
        }
        
        if custom_periods:
            periods.update(custom_periods)
        
        # Simple Moving Averages - commonly used trend indicators
        for period in periods['sma_periods']:
            col_name = f'sma_{period}'
            if col_name not in df.columns:
                df[col_name] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages - more responsive trend indicators
        for period in periods['ema_periods']:
            col_name = f'ema_{period}'
            if col_name not in df.columns:
                df[col_name] = df['close'].ewm(span=period).mean()
            
        # RSI (Relative Strength Index) - momentum oscillator
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['close'], periods['rsi_period'])
            
        # Bollinger Bands - volatility-based support/resistance
        if 'bb_upper' not in df.columns:
            bb_period = periods['bb_period']
            bb_std = periods['bb_std_dev']
            sma = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = sma + (std * bb_std)
            df['bb_lower'] = sma - (std * bb_std)
            df['bb_middle'] = sma
            # Band position for analysis
            band_width = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / band_width
            df['bb_position'] = df['bb_position'].fillna(0.5)
            
        # ATR (Average True Range) - volatility measure (requires OHLC data)
        if 'atr' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = self._calculate_atr(df, periods['atr_period'])
        
        # Volume indicators (if volume data available)
        if 'volume' in df.columns and 'volume_avg' not in df.columns:
            volume_period = periods['volume_period']
            df['volume_avg'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg']
            
        # Z-Score calculation for statistical analysis
        if 'z_score' not in df.columns:
            z_period = periods.get('z_score_period', 50)
            df['price_mean'] = df['close'].rolling(window=z_period).mean()
            df['price_std'] = df['close'].rolling(window=z_period).std()
            df['z_score'] = (df['close'] - df['price_mean']) / df['price_std']
            df['z_score'] = df['z_score'].fillna(0)
            
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
    
    def _calculate_moving_average(self, prices: pd.Series, period: int, ma_type: str = 'sma') -> pd.Series:
        """Calculate moving average with configurable type.
        
        Provides a unified interface for calculating different types of moving averages,
        reducing code duplication across strategies that need flexible MA calculations.
        
        Args:
            prices: Time series of prices
            period: Number of periods for MA calculation
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            
        Returns:
            pd.Series: Moving average values
            
        Raises:
            ValueError: If ma_type is not supported
        """
        if ma_type.lower() == 'sma':
            return prices.rolling(window=period).mean()
        elif ma_type.lower() == 'ema':
            return prices.ewm(span=period).mean()
        elif ma_type.lower() == 'wma':
            # Weighted moving average - linear weights
            weights = np.arange(1, period + 1)
            return prices.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        else:
            raise ValueError(f"Unsupported moving average type: {ma_type}. Use 'sma', 'ema', or 'wma'")
    
    def _calculate_crossover_signals(self, fast_series: pd.Series, slow_series: pd.Series) -> Dict[str, pd.Series]:
        """Calculate crossover signals between two time series.
        
        Common utility for generating crossover-based signals used by many strategies.
        Identifies when one series crosses above or below another, providing both
        directional signals and specific crossover event detection.
        
        Args:
            fast_series: Faster/more responsive time series (e.g., short MA)
            slow_series: Slower/more stable time series (e.g., long MA)
            
        Returns:
            Dict containing:
                - 'bullish_signal': Boolean series for bullish crossover events
                - 'bearish_signal': Boolean series for bearish crossover events
                - 'trend_signal': Integer series (1=bullish trend, -1=bearish trend, 0=neutral)
        """
        # Get previous period values for crossover detection
        fast_prev = fast_series.shift(1)
        slow_prev = slow_series.shift(1)
        
        # Bullish crossover: fast crosses above slow
        bullish_crossover = (fast_series > slow_series) & (fast_prev <= slow_prev)
        
        # Bearish crossover: fast crosses below slow
        bearish_crossover = (fast_series < slow_series) & (fast_prev >= slow_prev)
        
        # Trend signal based on current relationship
        trend_signal = pd.Series(0, index=fast_series.index)
        trend_signal.loc[fast_series > slow_series] = 1
        trend_signal.loc[fast_series < slow_series] = -1
        
        return {
            'bullish_signal': bullish_crossover,
            'bearish_signal': bearish_crossover,
            'trend_signal': trend_signal
        }
    
    def _apply_volume_filter(self, data: pd.DataFrame, signals: pd.Series, 
                           volume_multiplier: float = 1.5, volume_period: int = 20) -> pd.Series:
        """Apply volume confirmation filter to trading signals.
        
        Common utility for filtering signals based on volume confirmation.
        Only allows signals to pass through when volume exceeds a threshold
        relative to recent average volume.
        
        Args:
            data: DataFrame containing 'volume' column
            signals: Boolean or integer series of trading signals
            volume_multiplier: Minimum volume relative to average (default: 1.5)
            volume_period: Period for volume average calculation (default: 20)
            
        Returns:
            pd.Series: Filtered signals that meet volume requirements
            
        Note:
            If 'volume' column is not present, returns original signals unchanged
        """
        if 'volume' not in data.columns:
            self.logger.warning("Volume column not found - skipping volume filter")
            return signals
        
        # Calculate volume threshold
        volume_avg = data['volume'].rolling(window=volume_period).mean()
        volume_threshold = volume_avg * volume_multiplier
        volume_filter = data['volume'] > volume_threshold
        
        # Apply filter to signals
        filtered_signals = signals & volume_filter
        
        self.logger.debug(f"Volume filter applied: {signals.sum()} -> {filtered_signals.sum()} signals")
        return filtered_signals
    
    def _calculate_channel_levels(self, data: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """Calculate price channel levels (Donchian channels).
        
        Common utility for calculating support and resistance levels based on
        highest highs and lowest lows over a specified period.
        
        Args:
            data: DataFrame containing 'high' and 'low' columns
            period: Lookback period for channel calculation
            
        Returns:
            Dict containing:
                - 'channel_high': Upper channel boundary (resistance)
                - 'channel_low': Lower channel boundary (support)
                - 'channel_mid': Middle channel line
                - 'channel_width': Channel width (volatility proxy)
        """
        # Calculate channel boundaries
        channel_high = data['high'].rolling(window=period).max()
        channel_low = data['low'].rolling(window=period).min()
        channel_mid = (channel_high + channel_low) / 2
        channel_width = channel_high - channel_low
        
        return {
            'channel_high': channel_high,
            'channel_low': channel_low,
            'channel_mid': channel_mid,
            'channel_width': channel_width
        }
    
    def _detect_breakouts(self, data: pd.DataFrame, levels: Dict[str, pd.Series], 
                         threshold: float = 0.01) -> Dict[str, pd.Series]:
        """Detect price breakouts from support/resistance levels.
        
        Common utility for detecting when price breaks above resistance or
        below support levels with optional threshold buffer.
        
        Args:
            data: DataFrame containing 'close' column
            levels: Dictionary with 'channel_high' and 'channel_low' levels
            threshold: Percentage buffer for breakout confirmation (default: 0.01)
            
        Returns:
            Dict containing:
                - 'upside_breakout': Boolean series for upside breakouts
                - 'downside_breakout': Boolean series for downside breakouts
        """
        # Calculate breakout thresholds with buffer
        upper_threshold = levels['channel_high'] * (1 + threshold)
        lower_threshold = levels['channel_low'] * (1 - threshold)
        
        # Detect breakouts
        upside_breakout = data['close'] > upper_threshold
        downside_breakout = data['close'] < lower_threshold
        
        return {
            'upside_breakout': upside_breakout,
            'downside_breakout': downside_breakout
        }
    
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
        successful_strategies = 0
        
        for strategy in self.strategies:
            strategy_result = safe_execute(
                strategy.generate_signals,
                data,
                fallback=None,
                context={
                    'strategy_name': strategy.name,
                    'strategy_class': strategy.__class__.__name__,
                    'combination_method': self.combination_method
                }
            )
            
            if strategy_result is not None and 'signal' in strategy_result.columns:
                individual_signals[strategy.name] = strategy_result['signal']
                signal_count = (strategy_result['signal'] != 0).sum()
                self.logger.debug(f"Generated {signal_count} signals from {strategy.name}")
                successful_strategies += 1
            else:
                # Use hold signals (0) for failed strategy
                individual_signals[strategy.name] = pd.Series(0, index=data.index)
                self.logger.warning(f"Strategy {strategy.name} failed - using hold signals")
        
        # Check if we have enough successful strategies
        if successful_strategies == 0:
            raise StrategyError(
                "All strategies failed to generate signals",
                context={
                    'total_strategies': len(self.strategies),
                    'successful_strategies': successful_strategies
                }
            )
        elif successful_strategies < len(self.strategies) / 2:
            self.logger.warning(
                f"Only {successful_strategies}/{len(self.strategies)} strategies succeeded"
            )
        
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