"""MACD (Moving Average Convergence Divergence) crossover strategy for momentum trading.

This module implements a sophisticated momentum strategy based on the MACD indicator,
which measures the relationship between two moving averages of a security's price.
MACD is one of the most popular and effective momentum oscillators in technical analysis.

Strategy Logic:
- MACD Line = EMA(12) - EMA(26) [difference between fast and slow EMAs]
- Signal Line = EMA(9) of MACD Line [smoothed version of MACD]
- Histogram = MACD Line - Signal Line [convergence/divergence indicator]

Signal Generation:
- BUY: MACD line crosses above signal line (bullish momentum)
- SELL: MACD line crosses below signal line (bearish momentum)
- Alternative: MACD crosses above/below zero line for trend confirmation

Classes:
    MacdCrossoverStrategy: MACD signal line crossover implementation
    MacdZeroCrossStrategy: MACD zero line crossover implementation

Example:
    >>> strategy = MacdCrossoverStrategy({
    ...     'fast_period': 12,
    ...     'slow_period': 26,
    ...     'signal_period': 9
    ... })
    >>> signals = strategy.generate_signals(price_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class MacdCrossoverStrategy(BaseStrategy):
    """MACD signal line crossover strategy for momentum identification.
    
    This strategy uses the classic MACD (Moving Average Convergence Divergence) 
    indicator to identify momentum shifts and trend changes. The MACD measures
    the relationship between two EMAs and generates signals when the MACD line
    crosses above or below its signal line.
    
    Strategy Components:
    1. MACD Line: Difference between fast EMA and slow EMA
    2. Signal Line: EMA of the MACD line 
    3. Histogram: Difference between MACD and Signal lines
    
    Signal Rules:
    - BUY: MACD line crosses above signal line (bullish momentum shift)
    - SELL: MACD line crosses below signal line (bearish momentum shift)
    - Optional: Zero line filter for trend confirmation
    
    Key Features:
    - Configurable EMA periods for different sensitivities
    - Optional zero line confirmation filter
    - Histogram analysis for momentum strength
    - Detailed signal analysis and logging
    
    Best Use Cases:
    - Trending markets with clear momentum shifts
    - Medium-term position trading
    - Momentum confirmation for other strategies
    - Market timing for entry/exit decisions
    
    Attributes:
        strategy_category: 'momentum' for auto-registration
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize MACD crossover strategy with configurable parameters.
        
        Sets up the MACD strategy with standard parameters optimized for daily
        timeframes. The default settings (12, 26, 9) are the most widely used
        MACD configuration in technical analysis.
        
        Args:
            parameters: Strategy configuration dictionary:
                - fast_period (int): Fast EMA period (default: 12)
                    Shorter periods = more responsive to price changes
                - slow_period (int): Slow EMA period (default: 26)  
                    Longer periods = smoother, less noisy signals
                - signal_period (int): Signal line EMA period (default: 9)
                    Smoothing period for MACD line to generate signal line
                - use_zero_filter (bool): Require MACD above/below zero (default: False)
                    True = only trade in direction of MACD zero line
                - min_histogram_threshold (float): Minimum histogram value for signals (default: 0.0)
                    Filters weak momentum signals
                    
        Example:
            >>> # Sensitive short-term MACD
            >>> params = {
            ...     'fast_period': 8,
            ...     'slow_period': 21,
            ...     'signal_period': 5,
            ...     'use_zero_filter': True
            ... }
            >>> strategy = MacdCrossoverStrategy(params)
        """
        # Standard MACD parameters based on Gerald Appel's original work
        default_params = {
            'fast_period': 12,           # Fast EMA period (standard)
            'slow_period': 26,           # Slow EMA period (standard)
            'signal_period': 9,          # Signal line EMA period (standard)
            'use_zero_filter': False,    # Optional zero line confirmation
            'min_histogram_threshold': 0.0  # Minimum momentum threshold
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MACD Crossover", default_params)
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicator components.
        
        Computes the complete MACD indicator including the main line, signal line,
        and histogram. This provides all components needed for signal generation
        and technical analysis.
        
        Args:
            prices: Time series of closing prices
            
        Returns:
            Dict containing:
                - 'macd': MACD line (fast EMA - slow EMA)
                - 'signal': Signal line (EMA of MACD)
                - 'histogram': MACD histogram (MACD - Signal)
                
        Example:
            >>> macd_data = strategy._calculate_macd(data['close'])
            >>> print(f"MACD range: {macd_data['macd'].min():.4f} to {macd_data['macd'].max():.4f}")
        """
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        
        # Calculate EMAs using base class utility
        fast_ema = self._calculate_moving_average(prices, fast_period, 'ema')
        slow_ema = self._calculate_moving_average(prices, slow_period, 'ema')
        
        # MACD line is the difference between fast and slow EMAs
        macd_line = fast_ema - slow_ema
        
        # Signal line is EMA of MACD line
        signal_line = self._calculate_moving_average(macd_line, signal_period, 'ema')
        
        # Histogram shows convergence/divergence
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD crossover signals with momentum analysis.
        
        Implements the classic MACD signal generation methodology while providing
        comprehensive momentum analysis. The strategy identifies when the MACD line
        crosses above or below the signal line, indicating momentum shifts.
        
        Implementation Steps:
        1. Validate input data for quality and completeness
        2. Calculate MACD components (line, signal, histogram)
        3. Identify crossover events using base class utilities
        4. Apply optional zero line and histogram filters
        5. Generate final buy/sell/hold signals
        6. Add analysis columns for evaluation
        
        Args:
            data: DataFrame containing market data with 'close' column
            
        Returns:
            DataFrame with original data plus MACD analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'macd': MACD line values
                - 'macd_signal': Signal line values
                - 'macd_histogram': Histogram values
                - 'macd_bullish_cross': Boolean bullish crossover events
                - 'macd_bearish_cross': Boolean bearish crossover events
                - 'macd_above_zero': Boolean MACD above zero line
                
        Signal Logic:
            - BUY: MACD crosses above signal line (+ optional filters)
            - SELL: MACD crosses below signal line (+ optional filters)
            - HOLD: No crossover or filtered out by conditions
            
        Example:
            >>> signals = strategy.generate_signals(daily_data)
            >>> crossovers = signals[signals['macd_bullish_cross'] | signals['macd_bearish_cross']]
            >>> print(f"Found {len(crossovers)} MACD crossover events")
        """
        # Step 1: Validate input data
        if not self.validate_data(data):
            self.logger.error("Data validation failed for MACD strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with copy of original data
        df = data.copy()
        
        # Step 2: Calculate MACD components
        macd_components = self._calculate_macd(df['close'])
        
        # Add MACD components to dataframe
        df['macd'] = macd_components['macd']
        df['macd_signal'] = macd_components['signal']
        df['macd_histogram'] = macd_components['histogram']
        
        self.logger.debug(f"Calculated MACD({self.parameters['fast_period']},"
                         f"{self.parameters['slow_period']},"
                         f"{self.parameters['signal_period']})")
        
        # Step 3: Calculate crossover signals using base class utility
        crossover_signals = self._calculate_crossover_signals(df['macd'], df['macd_signal'])
        
        # Add crossover event indicators
        df['macd_bullish_cross'] = crossover_signals['bullish_signal']
        df['macd_bearish_cross'] = crossover_signals['bearish_signal']
        
        # Step 4: Initialize signal column
        df['signal'] = 0
        
        # Step 5: Generate basic crossover signals
        buy_condition = df['macd_bullish_cross']
        sell_condition = df['macd_bearish_cross']
        
        # Step 6: Apply optional zero line filter
        if self.parameters['use_zero_filter']:
            df['macd_above_zero'] = df['macd'] > 0
            
            # Only buy when MACD is above zero (uptrend confirmation)
            buy_condition = buy_condition & df['macd_above_zero']
            
            # Only sell when MACD is below zero (downtrend confirmation)
            sell_condition = sell_condition & (~df['macd_above_zero'])
            
            self.logger.debug("Applied MACD zero line filter")
        else:
            df['macd_above_zero'] = df['macd'] > 0
        
        # Step 7: Apply histogram threshold filter
        histogram_threshold = self.parameters['min_histogram_threshold']
        if histogram_threshold > 0:
            # Require minimum momentum strength
            strong_momentum_up = df['macd_histogram'] > histogram_threshold
            strong_momentum_down = df['macd_histogram'] < -histogram_threshold
            
            buy_condition = buy_condition & strong_momentum_up
            sell_condition = sell_condition & strong_momentum_down
            
            self.logger.debug(f"Applied histogram threshold filter: {histogram_threshold}")
        
        # Step 8: Apply final signal conditions
        df.loc[buy_condition, 'signal'] = 1   # Buy signal
        df.loc[sell_condition, 'signal'] = -1  # Sell signal
        
        # Step 9: Log signal statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        total_crossovers = df['macd_bullish_cross'].sum() + df['macd_bearish_cross'].sum()
        
        self.logger.info(f"MACD signals: {buy_count} buys, {sell_count} sells "
                        f"from {total_crossovers} total crossovers")
        
        if self.parameters['use_zero_filter']:
            above_zero_pct = (df['macd_above_zero'].sum() / len(df) * 100).round(1)
            self.logger.debug(f"MACD above zero: {above_zero_pct}% of time")
        
        return df


class MacdZeroCrossStrategy(BaseStrategy):
    """MACD zero line crossover strategy for trend confirmation.
    
    This alternative MACD strategy generates signals when the MACD line crosses
    above or below the zero line, providing trend confirmation signals rather
    than momentum shift signals. This approach typically generates fewer but
    potentially higher-quality signals.
    
    Signal Rules:
    - BUY: MACD line crosses above zero (bullish trend confirmation)
    - SELL: MACD line crosses below zero (bearish trend confirmation)
    
    Comparison with Signal Line Strategy:
    - Zero line crossovers are less frequent but more significant
    - Better for trend-following rather than momentum trading
    - Reduces whipsaws but may lag trend changes
    """
    
    # Strategy metadata for auto-registration  
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize MACD zero line crossover strategy."""
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,  # Still calculated for analysis
            'min_duration': 1    # Minimum periods above/below zero
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MACD Zero Cross", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD zero line crossover signals."""
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate MACD using the same method as signal line strategy
        strategy_temp = MacdCrossoverStrategy(self.parameters)
        macd_components = strategy_temp._calculate_macd(df['close'])
        
        df['macd'] = macd_components['macd']
        df['macd_signal'] = macd_components['signal']
        df['macd_histogram'] = macd_components['histogram']
        
        # Generate zero line crossover signals
        zero_crossover = self._calculate_crossover_signals(df['macd'], pd.Series(0, index=df.index))
        
        df['macd_zero_bull_cross'] = zero_crossover['bullish_signal']
        df['macd_zero_bear_cross'] = zero_crossover['bearish_signal']
        
        # Apply signals
        df['signal'] = 0
        df.loc[df['macd_zero_bull_cross'], 'signal'] = 1
        df.loc[df['macd_zero_bear_cross'], 'signal'] = -1
        
        # Log statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        
        self.logger.info(f"MACD zero line signals: {buy_count} buys, {sell_count} sells")
        
        return df