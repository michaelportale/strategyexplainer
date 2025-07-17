"""Example enhanced MACD strategy demonstrating the new parameter management architecture.

This module shows how to create a modern strategy using the enhanced base class
with comprehensive parameter management, metadata, and UI integration capabilities.

The strategy demonstrates:
- Declarative parameter schemas with rich metadata
- Type-safe parameter access
- Automatic UI form generation
- Configuration validation
- Enhanced logging and error handling
- Composable overlay integration

Example Usage:
    >>> # Create with defaults
    >>> strategy = EnhancedMacdStrategy()
    
    >>> # Create with custom parameters
    >>> params = {'fast_period': 8, 'slow_period': 21, 'signal_period': 5}
    >>> strategy = EnhancedMacdStrategy(parameters=params)
    
    >>> # Generate UI schema for frontend
    >>> ui_schema = EnhancedMacdStrategy.get_ui_schema()
    
    >>> # Apply overlays
    >>> from ..overlays import RegimeFilterOverlay, PositionSizingOverlay
    >>> regime_filtered = RegimeFilterOverlay(strategy)
    >>> final_strategy = PositionSizingOverlay(regime_filtered)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from ..enhanced_base import EnhancedBaseStrategy, StrategyMetadata
from ..parameters import ParameterDefinition, ParameterSchema


class EnhancedMacdStrategy(EnhancedBaseStrategy):
    """Enhanced MACD strategy with comprehensive parameter management.
    
    This strategy implements the classic MACD (Moving Average Convergence Divergence)
    signal generation with modern parameter management, extensive validation, and
    rich metadata for UI integration.
    
    Strategy Logic:
    - Calculate MACD line (fast EMA - slow EMA)
    - Calculate signal line (EMA of MACD line)
    - Generate buy signals when MACD crosses above signal line
    - Generate sell signals when MACD crosses below signal line
    - Optional zero-line filter for trend confirmation
    - Optional histogram momentum filter
    
    Key Enhancements:
    - Type-safe parameter definitions with validation
    - Rich UI metadata for automatic form generation
    - Comprehensive logging and error handling
    - Configuration synchronization support
    - Parameter sensitivity analysis
    - Performance optimization tracking
    """
    
    # Rich parameter schema with comprehensive validation and UI metadata
    PARAMETER_SCHEMA = ParameterSchema({
        'fast_period': ParameterDefinition(
            default=12,
            type=int,
            range=(2, 50),
            description="Fast EMA period for MACD calculation",
            display_name="Fast Period",
            category="Technical Indicators",
            tooltip="Shorter periods make the MACD more responsive to price changes. Standard value is 12.",
            ui_widget="slider"
        ),
        
        'slow_period': ParameterDefinition(
            default=26,
            type=int,
            range=(10, 100),
            description="Slow EMA period for MACD calculation",
            display_name="Slow Period", 
            category="Technical Indicators",
            tooltip="Longer periods provide smoother signals with less noise. Standard value is 26.",
            ui_widget="slider"
        ),
        
        'signal_period': ParameterDefinition(
            default=9,
            type=int,
            range=(3, 30),
            description="EMA period for MACD signal line",
            display_name="Signal Period",
            category="Technical Indicators",
            tooltip="Period for smoothing the MACD line to create the signal line. Standard value is 9.",
            ui_widget="slider"
        ),
        
        'crossover_type': ParameterDefinition(
            default='signal_line',
            type=str,
            choices=['signal_line', 'zero_line', 'both'],
            description="Type of MACD crossover to use for signals",
            display_name="Crossover Type",
            category="Signal Generation",
            tooltip="Signal line: MACD vs Signal crossovers, Zero line: MACD vs 0 crossovers, Both: Requires both conditions",
            ui_widget="dropdown"
        ),
        
        'use_histogram_filter': ParameterDefinition(
            default=False,
            type=bool,
            description="Filter signals based on MACD histogram momentum",
            display_name="Use Histogram Filter",
            category="Signal Generation",
            tooltip="Only generate signals when histogram confirms momentum direction",
            ui_widget="checkbox"
        ),
        
        'min_histogram_threshold': ParameterDefinition(
            default=0.001,
            type=float,
            range=(0.0, 0.01),
            description="Minimum histogram value for signal confirmation",
            display_name="Histogram Threshold",
            category="Signal Generation",
            tooltip="Minimum absolute histogram value required for signal generation",
            ui_widget="number"
        ),
        
        'trend_confirmation_period': ParameterDefinition(
            default=20,
            type=int,
            range=(5, 50),
            description="Period for trend confirmation using price moving average",
            display_name="Trend Confirmation Period",
            category="Trend Analysis",
            tooltip="Only trade in direction of this moving average trend",
            ui_widget="slider"
        ),
        
        'enable_trend_filter': ParameterDefinition(
            default=True,
            type=bool,
            description="Enable trend direction filtering",
            display_name="Enable Trend Filter",
            category="Trend Analysis",
            tooltip="Filter signals to trade only in direction of the trend",
            ui_widget="checkbox"
        ),
        
        'signal_delay_periods': ParameterDefinition(
            default=0,
            type=int,
            range=(0, 5),
            description="Number of periods to delay signal generation",
            display_name="Signal Delay",
            category="Advanced",
            tooltip="Delay signals by N periods to avoid false breakouts",
            ui_widget="number"
        ),
        
        'min_price_change_pct': ParameterDefinition(
            default=0.0,
            type=float,
            range=(0.0, 0.05),
            description="Minimum price change percentage to generate signal",
            display_name="Min Price Change %",
            category="Advanced",
            tooltip="Require minimum price movement to filter noise",
            ui_widget="number"
        )
    }, strategy_name="Enhanced MACD Strategy", description="Advanced MACD implementation with comprehensive parameter management")
    
    # Strategy metadata for enhanced introspection
    STRATEGY_DESCRIPTION = "Enhanced MACD crossover strategy with configurable signal generation, trend filtering, and histogram confirmation"
    STRATEGY_VERSION = "2.0"
    TAGS = ["momentum", "trend-following", "macd", "crossover", "enhanced"]
    
    def __init__(self, parameters: Dict[str, Any] = None, name: str = None):
        """Initialize enhanced MACD strategy with parameter validation."""
        super().__init__(parameters, name)
        
        # Validate parameter relationships
        self._validate_parameter_relationships()
        
        # Initialize performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'crossovers_detected': 0,
            'trend_filtered_signals': 0,
            'histogram_filtered_signals': 0
        }
    
    def _validate_parameter_relationships(self) -> None:
        """Validate parameter relationships and dependencies."""
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        
        if fast_period >= slow_period:
            raise ValueError(
                f"Fast period ({fast_period}) must be less than slow period ({slow_period})"
            )
        
        # Warn about unusual parameter combinations
        if slow_period / fast_period > 5:
            self.logger.warning(
                f"Large period ratio ({slow_period}/{fast_period} = {slow_period/fast_period:.1f}) "
                "may produce very slow signals"
            )
        
        if self.get_parameter('use_histogram_filter') and self.get_parameter('min_histogram_threshold') == 0:
            self.logger.warning("Histogram filter enabled but threshold is 0 - may not filter effectively")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced MACD signals with comprehensive analysis.
        
        This implementation provides detailed signal analysis, performance tracking,
        and multiple filtering options for robust signal generation.
        
        Args:
            data: Market data containing 'close' column minimum
            
        Returns:
            DataFrame with signals and comprehensive MACD analysis
        """
        # Validate input data
        if not self.validate_data(data):
            self.logger.error("Data validation failed")
            return self._create_empty_result(data)
        
        # Initialize result dataframe
        df = data.copy()
        
        # Calculate MACD components
        macd_data = self._calculate_macd_components(df)
        for key, series in macd_data.items():
            df[key] = series
        
        # Generate base crossover signals
        crossover_signals = self._generate_crossover_signals(df)
        for key, series in crossover_signals.items():
            df[key] = series
        
        # Apply filtering layers
        filtered_signals = self._apply_signal_filters(df)
        for key, series in filtered_signals.items():
            df[key] = series
        
        # Add performance metrics
        self._update_performance_metrics(df)
        
        # Add analysis columns for debugging and visualization
        self._add_analysis_columns(df)
        
        self.logger.info(f"Generated {(df['signal'] != 0).sum()} signals from {len(df)} data points")
        
        return df
    
    def _calculate_macd_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD, signal line, and histogram."""
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        signal_period = self.get_parameter('signal_period')
        
        # Calculate EMAs
        fast_ema = data['close'].ewm(span=fast_period).mean()
        slow_ema = data['close'].ewm(span=slow_period).mean()
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema
        }
    
    def _generate_crossover_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate crossover signals based on configuration."""
        crossover_type = self.get_parameter('crossover_type')
        
        signals = pd.Series(0, index=data.index)
        signal_line_cross = pd.Series(False, index=data.index)
        zero_line_cross = pd.Series(False, index=data.index)
        
        # Signal line crossovers
        if crossover_type in ['signal_line', 'both']:
            macd_above_signal = (data['macd'] > data['macd_signal']).fillna(False)
            macd_above_signal_prev = macd_above_signal.shift(1).fillna(False)
            
            # Bullish crossover: MACD crosses above signal line
            signal_line_bull_cross = macd_above_signal & ~macd_above_signal_prev
            
            # Bearish crossover: MACD crosses below signal line
            signal_line_bear_cross = ~macd_above_signal & macd_above_signal_prev
            
            signal_line_cross = signal_line_bull_cross | signal_line_bear_cross
            
            if crossover_type == 'signal_line':
                signals.loc[signal_line_bull_cross] = 1
                signals.loc[signal_line_bear_cross] = -1
        
        # Zero line crossovers
        if crossover_type in ['zero_line', 'both']:
            macd_above_zero = (data['macd'] > 0).fillna(False)
            macd_above_zero_prev = macd_above_zero.shift(1).fillna(False)
            
            # Bullish crossover: MACD crosses above zero
            zero_line_bull_cross = macd_above_zero & ~macd_above_zero_prev
            
            # Bearish crossover: MACD crosses below zero
            zero_line_bear_cross = ~macd_above_zero & macd_above_zero_prev
            
            zero_line_cross = zero_line_bull_cross | zero_line_bear_cross
            
            if crossover_type == 'zero_line':
                signals.loc[zero_line_bull_cross] = 1
                signals.loc[zero_line_bear_cross] = -1
            elif crossover_type == 'both':
                # Both conditions required
                signals.loc[signal_line_bull_cross & zero_line_bull_cross] = 1
                signals.loc[signal_line_bear_cross & zero_line_bear_cross] = -1
        
        return {
            'base_signal': signals,
            'signal_line_crossover': signal_line_cross,
            'zero_line_crossover': zero_line_cross
        }
    
    def _apply_signal_filters(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Apply various signal filters based on configuration."""
        filtered_signal = data['base_signal'].copy()
        filter_reasons = pd.Series('', index=data.index)
        
        # Histogram filter
        if self.get_parameter('use_histogram_filter'):
            histogram_threshold = self.get_parameter('min_histogram_threshold')
            
            # Only allow signals when histogram confirms direction
            histogram_confirms_long = (
                (data['base_signal'] == 1) & 
                (data['macd_histogram'] > histogram_threshold)
            )
            histogram_confirms_short = (
                (data['base_signal'] == -1) & 
                (data['macd_histogram'] < -histogram_threshold)
            )
            
            histogram_filter = histogram_confirms_long | histogram_confirms_short
            
            # Filter out signals that don't meet histogram criteria
            filtered_out = (data['base_signal'] != 0) & ~histogram_filter
            filtered_signal.loc[filtered_out] = 0
            filter_reasons.loc[filtered_out] += 'histogram_filter '
        
        # Trend filter
        if self.get_parameter('enable_trend_filter'):
            trend_period = self.get_parameter('trend_confirmation_period')
            trend_ma = data['close'].rolling(trend_period).mean()
            
            # Only allow signals in direction of trend
            uptrend = data['close'] > trend_ma
            trend_confirms_long = (data['base_signal'] == 1) & uptrend
            trend_confirms_short = (data['base_signal'] == -1) & ~uptrend
            
            trend_filter = trend_confirms_long | trend_confirms_short
            
            # Filter out counter-trend signals
            counter_trend = (data['base_signal'] != 0) & ~trend_filter
            filtered_signal.loc[counter_trend] = 0
            filter_reasons.loc[counter_trend] += 'trend_filter '
        
        # Price change filter
        min_price_change = self.get_parameter('min_price_change_pct')
        if min_price_change > 0:
            price_change = data['close'].pct_change().abs()
            insufficient_change = (
                (data['base_signal'] != 0) & 
                (price_change < min_price_change)
            )
            filtered_signal.loc[insufficient_change] = 0
            filter_reasons.loc[insufficient_change] += 'price_change_filter '
        
        # Signal delay
        delay_periods = self.get_parameter('signal_delay_periods')
        if delay_periods > 0:
            filtered_signal = filtered_signal.shift(delay_periods)
            filter_reasons = filter_reasons.shift(delay_periods)
        
        return {
            'signal': filtered_signal.fillna(0).astype(int),
            'filter_reasons': filter_reasons.fillna('')
        }
    
    def _update_performance_metrics(self, data: pd.DataFrame) -> None:
        """Update internal performance tracking metrics."""
        self.performance_metrics['signals_generated'] = (data['signal'] != 0).sum()
        self.performance_metrics['crossovers_detected'] = (
            data['signal_line_crossover'] | data['zero_line_crossover']
        ).sum()
        
        # Count filtered signals
        filtered_by_trend = (data['filter_reasons'].str.contains('trend_filter')).sum()
        filtered_by_histogram = (data['filter_reasons'].str.contains('histogram_filter')).sum()
        
        self.performance_metrics['trend_filtered_signals'] = filtered_by_trend
        self.performance_metrics['histogram_filtered_signals'] = filtered_by_histogram
    
    def _add_analysis_columns(self, data: pd.DataFrame) -> None:
        """Add additional columns for analysis and visualization."""
        # MACD momentum analysis
        data['macd_momentum'] = data['macd_histogram'] - data['macd_histogram'].shift(1)
        data['macd_acceleration'] = data['macd_momentum'] - data['macd_momentum'].shift(1)
        
        # Signal strength
        data['signal_strength'] = np.abs(data['macd_histogram']) / data['macd'].abs()
        data['signal_strength'] = data['signal_strength'].fillna(0)
        
        # Trend alignment
        if self.get_parameter('enable_trend_filter'):
            trend_period = self.get_parameter('trend_confirmation_period')
            trend_ma = data['close'].rolling(trend_period).mean()
            data['trend_alignment'] = np.where(
                data['close'] > trend_ma, 1, -1
            )
        else:
            data['trend_alignment'] = 0
    
    def _create_empty_result(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create empty result dataframe when data validation fails."""
        df = data.copy()
        df['signal'] = 0
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_histogram'] = np.nan
        return df
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary including signal statistics."""
        summary = self.performance_metrics.copy()
        
        # Calculate filtering efficiency
        if summary['crossovers_detected'] > 0:
            summary['signal_efficiency'] = (
                summary['signals_generated'] / summary['crossovers_detected']
            )
        else:
            summary['signal_efficiency'] = 0
        
        # Add parameter summary
        summary['current_parameters'] = {
            'fast_period': self.get_parameter('fast_period'),
            'slow_period': self.get_parameter('slow_period'),
            'signal_period': self.get_parameter('signal_period'),
            'crossover_type': self.get_parameter('crossover_type')
        }
        
        return summary
    
    def optimize_parameters(self, 
                          data: pd.DataFrame,
                          optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Basic parameter optimization using grid search.
        
        This is a simplified optimization example. In practice, you would
        use more sophisticated optimization algorithms.
        
        Args:
            data: Historical data for optimization
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Dictionary with optimization results
        """
        # Define parameter ranges for optimization
        param_ranges = {
            'fast_period': [8, 10, 12, 15],
            'slow_period': [21, 26, 30, 35],
            'signal_period': [7, 9, 11, 14]
        }
        
        best_score = float('-inf')
        best_params = None
        results = []
        
        # Simple grid search
        for fast in param_ranges['fast_period']:
            for slow in param_ranges['slow_period']:
                for signal in param_ranges['signal_period']:
                    if fast >= slow:  # Skip invalid combinations
                        continue
                    
                    # Create strategy with test parameters
                    test_params = {
                        'fast_period': fast,
                        'slow_period': slow,
                        'signal_period': signal
                    }
                    
                    try:
                        test_strategy = self.clone(**test_params)
                        signals_df = test_strategy.generate_signals(data)
                        
                        # Calculate simple performance metric
                        returns = data['close'].pct_change()
                        strategy_returns = returns * signals_df['signal'].shift(1)
                        
                        if optimization_metric == 'sharpe_ratio':
                            score = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
                        elif optimization_metric == 'total_return':
                            score = (1 + strategy_returns).prod() - 1
                        else:
                            score = strategy_returns.sum()
                        
                        results.append({
                            'parameters': test_params,
                            'score': score,
                            'signals_count': (signals_df['signal'] != 0).sum()
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = test_params
                            
                    except Exception as e:
                        self.logger.warning(f"Optimization failed for params {test_params}: {e}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_metric': optimization_metric,
            'total_combinations_tested': len(results),
            'all_results': results
        }