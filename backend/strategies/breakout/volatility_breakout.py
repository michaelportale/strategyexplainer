"""Sophisticated volatility breakout strategy with multi-factor confirmation.

This module implements an advanced approach to breakout trading by combining
price action, volume analysis, and volatility measures. It aims to identify
significant price movements that indicate genuine trend changes or acceleration
while filtering out false breakouts that plague simpler systems.

Strategy Components:
1. Price Breakout: Breaking above/below recent highs/lows
2. Volume Confirmation: Unusual volume validates the move
3. Volatility Filter: ATR ensures moves are significant relative to recent volatility
4. Minimum Move Filter: Prevents tiny breakouts from generating signals

Signal Logic:
- BUY: Price > N-day high + volume > threshold + volatility confirmation
- SELL: Price < N-day low + volume > threshold + volatility confirmation
- HOLD: Insufficient confirmation factors present

Classes:
    VolatilityBreakoutStrategy: Multi-factor breakout with volume and volatility confirmation

Example:
    >>> # Conservative breakout strategy for daily timeframe
    >>> strategy = VolatilityBreakoutStrategy({
    ...     'breakout_period': 30,      # Longer lookback for significance
    ...     'volume_multiplier': 2.5,   # High volume requirement
    ...     'atr_multiplier': 1.5,      # Moderate volatility filter
    ...     'min_price_change': 0.02    # 2% minimum move
    ... })
    >>> signals = strategy.generate_signals(daily_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Sophisticated volatility breakout strategy with multi-factor confirmation.
    
    This strategy represents an advanced approach to breakout trading by combining
    price action, volume analysis, and volatility measures. It aims to identify
    significant price movements that indicate genuine trend changes or acceleration
    while filtering out false breakouts that plague simpler systems.
    
    Strategy Components:
    1. Price Breakout: Breaking above/below recent highs/lows
    2. Volume Confirmation: Unusual volume validates the move
    3. Volatility Filter: ATR ensures moves are significant relative to recent volatility
    4. Minimum Move Filter: Prevents tiny breakouts from generating signals
    
    Signal Logic:
    - BUY: Price > N-day high + volume > threshold + volatility confirmation
    - SELL: Price < N-day low + volume > threshold + volatility confirmation
    - HOLD: Insufficient confirmation factors present
    
    Key Features:
    - Configurable lookback periods for different timeframes
    - Volume threshold prevents low-conviction breakouts
    - ATR filter adapts to changing market volatility
    - Minimum price change filter reduces noise
    
    Best Use Cases:
    - Liquid markets with reliable volume data
    - Trend continuation after consolidation
    - Breakout from chart patterns (triangles, rectangles)
    - News-driven momentum events
    
    Risk Considerations:
    - False breakouts in choppy markets
    - Gap openings can trigger premature signals
    - Requires proper position sizing for volatility
    
    Example:
        >>> # Conservative breakout strategy for daily timeframe
        >>> strategy = VolatilityBreakoutStrategy({
        ...     'breakout_period': 30,      # Longer lookback for significance
        ...     'volume_multiplier': 2.5,   # High volume requirement
        ...     'atr_multiplier': 1.5,      # Moderate volatility filter
        ...     'min_price_change': 0.02    # 2% minimum move
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'breakout'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize volatility breakout strategy with comprehensive filtering.
        
        Sets up the breakout strategy with multiple confirmation factors and
        configurable parameters for different market conditions and timeframes.
        Default parameters are optimized for daily data and medium-term breakouts.
        
        Args:
            parameters: Strategy configuration dictionary with options:
                - breakout_period (int): Lookback period for high/low calculation (default: 20)
                    Shorter periods = more sensitive to recent levels
                    Longer periods = more significant breakouts required
                - volume_period (int): Period for volume average calculation (default: 10)
                    Used to establish normal volume baseline
                - volume_multiplier (float): Volume threshold multiplier (default: 1.5)
                    Higher values = require more unusual volume for confirmation
                    Lower values = more volume confirmations but potentially false signals
                - use_atr_filter (bool): Enable ATR volatility filter (default: True)
                    True = additional volatility confirmation required
                    False = pure price/volume breakout system
                - atr_period (int): ATR calculation period (default: 14)
                    Standard volatility measurement period
                - atr_multiplier (float): ATR threshold multiplier (default: 2.0)
                    Higher values = require larger moves relative to volatility
                - min_price_change (float): Minimum price change % for signal (default: 0.01)
                    Prevents micro-breakouts from generating signals
                    
        Parameter Optimization Tips:
        - Volatile markets: Increase atr_multiplier and min_price_change
        - Liquid markets: Decrease volume_multiplier
        - Trending markets: Decrease breakout_period for faster signals
        - Range-bound markets: Increase all filters to reduce false signals
        
        Example:
            >>> # Aggressive intraday breakout setup
            >>> params = {
            ...     'breakout_period': 10,
            ...     'volume_multiplier': 1.2,
            ...     'atr_multiplier': 1.0,
            ...     'min_price_change': 0.005
            ... }
            >>> strategy = VolatilityBreakoutStrategy(params)
        """
        # Research-based default parameters for robust breakout detection
        default_params = {
            'breakout_period': 20,          # Standard 20-day breakout period
            'volume_period': 10,            # 10-day volume baseline
            'volume_multiplier': 1.5,       # 50% above average volume
            'use_atr_filter': True,         # Enable volatility confirmation
            'atr_period': 14,               # Standard ATR period
            'atr_multiplier': 2.0,          # 2x ATR move requirement
            'min_price_change': 0.01        # 1% minimum breakout magnitude
        }
        
        # Update defaults with user parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with descriptive name
        super().__init__("Volatility Breakout", default_params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data contains required columns for breakout analysis.
        
        Breakout strategies require OHLCV data for comprehensive analysis.
        This method ensures all necessary columns are present before processing.
        
        Args:
            data: DataFrame to validate for breakout strategy requirements
            
        Returns:
            bool: True if data contains required columns, False otherwise
            
        Required Columns:
            - 'high': Daily high prices for resistance level calculation
            - 'low': Daily low prices for support level calculation  
            - 'close': Closing prices for breakout detection
            - 'volume': Trading volume for confirmation analysis
        """
        # Define columns specific to breakout strategy needs
        required_cols = ['high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns for breakout strategy: {missing_cols}")
            return False
        
        # Call parent validation for additional checks
        return super().validate_data(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate sophisticated volatility breakout signals with multi-factor confirmation.
        
        This method implements the complete breakout strategy logic including price level
        calculation, volume analysis, volatility filtering, and signal generation.
        It provides comprehensive breakout detection while minimizing false signals
        through multiple confirmation factors.
        
        Implementation Process:
        1. Validate input data for required OHLCV columns
        2. Calculate rolling high/low breakout levels
        3. Establish volume thresholds for confirmation
        4. Apply ATR volatility filter if enabled
        5. Generate breakout signals with all confirmations
        6. Add analysis columns for strategy evaluation
        7. Log comprehensive breakout statistics
        
        Args:
            data: DataFrame containing OHLCV market data with datetime index
            
        Returns:
            DataFrame with original data plus breakout analysis:
                - 'signal': Primary trading signal (1=buy breakout, -1=sell breakout, 0=hold)
                - 'high_breakout': Rolling high resistance level
                - 'low_breakout': Rolling low support level
                - 'volume_avg': Average volume baseline
                - 'volume_threshold': Volume confirmation threshold
                - 'atr': Average True Range (if ATR filter enabled)
                - 'atr_threshold': ATR-based move threshold
                - 'upside_breakout': Boolean upside breakout indicator
                - 'downside_breakout': Boolean downside breakout indicator
                - 'volume_ratio': Current volume / average volume ratio
                
        Signal Confirmation Requirements:
            Upside Breakout (BUY):
            - Close > N-day high
            - Volume > volume threshold
            - Price change > minimum threshold
            - ATR confirmation (if enabled)
            
            Downside Breakout (SELL):
            - Close < N-day low
            - Volume > volume threshold  
            - Price change > minimum threshold
            - ATR confirmation (if enabled)
            
        Example:
            >>> data = load_market_data('AAPL', '2023-01-01', '2024-01-01')
            >>> signals = strategy.generate_signals(data)
            >>> breakouts = signals[signals['signal'] != 0]
            >>> print(f"Found {len(breakouts)} confirmed breakouts")
        """
        # Step 1: Validate input data contains required columns
        if not self.validate_data(data):
            self.logger.error("Data validation failed for breakout strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Calculate breakout levels using base class utility
        breakout_period = self.parameters['breakout_period']
        
        # Use base class channel calculation utility
        channel_levels = self._calculate_channel_levels(df, breakout_period)
        
        # Shift to avoid look-ahead bias
        df['high_breakout'] = channel_levels['channel_high'].shift(1)
        df['low_breakout'] = channel_levels['channel_low'].shift(1)
        
        # Step 3: Add technical indicators including volume analysis
        volume_period = self.parameters['volume_period']
        volume_multiplier = self.parameters['volume_multiplier']
        
        custom_periods = {'volume_period': volume_period}
        df = self.add_technical_indicators(df, custom_periods)
        
        # Volume must exceed this threshold for confirmation
        df['volume_threshold'] = df['volume_avg'] * volume_multiplier
        
        # Step 4: Calculate ATR volatility filter if enabled (already added by add_technical_indicators)
        if self.parameters['use_atr_filter']:
            # Moves must exceed this threshold relative to recent volatility
            df['atr_threshold'] = df['atr'] * self.parameters['atr_multiplier']
        
        # Step 5: Initialize signal column with hold positions
        df['signal'] = 0
        
        # Step 6: Define upside breakout conditions
        
        # Primary condition: price breaks above resistance
        price_breaks_high = df['close'] > df['high_breakout']
        
        # Volume confirmation: unusual volume validates the breakout
        volume_confirms = df['volume'] > df['volume_threshold']
        
        # Minimum price change filter: prevents micro-breakouts
        min_change = self.parameters['min_price_change']
        price_change_up = (df['close'] - df['high_breakout']) / df['high_breakout'] > min_change
        
        # Combine core upside breakout conditions
        upside_breakout = price_breaks_high & volume_confirms & price_change_up
        
        # Step 7: Define downside breakout conditions
        
        # Primary condition: price breaks below support
        price_breaks_low = df['close'] < df['low_breakout']
        
        # Minimum price change filter for downside moves
        price_change_down = (df['low_breakout'] - df['close']) / df['low_breakout'] > min_change
        
        # Combine core downside breakout conditions
        downside_breakout = price_breaks_low & volume_confirms & price_change_down
        
        # Step 8: Apply ATR volatility filter if enabled
        if self.parameters['use_atr_filter']:
            # Additional filter: move must be significant relative to recent volatility
            
            # Upside move significance: today's gain > ATR threshold
            significant_move_up = (df['close'] - df['close'].shift(1)) > df['atr_threshold']
            
            # Downside move significance: today's loss > ATR threshold
            significant_move_down = (df['close'].shift(1) - df['close']) > df['atr_threshold']
            
            # Apply volatility filters to breakout conditions
            upside_breakout = upside_breakout & significant_move_up
            downside_breakout = downside_breakout & significant_move_down
            
            self.logger.debug("Applied ATR volatility filter to breakout signals")
        
        # Step 9: Generate final trading signals
        df.loc[upside_breakout, 'signal'] = 1   # Buy on confirmed upside breakout
        df.loc[downside_breakout, 'signal'] = -1 # Sell on confirmed downside breakout
        
        # Step 10: Add comprehensive analysis columns for evaluation
        df['upside_breakout'] = upside_breakout
        df['downside_breakout'] = downside_breakout
        df['volume_ratio'] = df['volume'] / df['volume_avg']  # Volume strength indicator
        
        # Step 11: Log detailed breakout statistics
        upside_count = upside_breakout.sum()
        downside_count = downside_breakout.sum()
        total_signals = upside_count + downside_count
        
        # Additional statistics for strategy evaluation
        avg_volume_ratio = df['volume_ratio'].mean()
        max_breakout_high = df['high_breakout'].max()
        min_breakout_low = df['low_breakout'].min()
        
        self.logger.info(f"Breakout analysis complete: {total_signals} total signals "
                        f"({upside_count} upside, {downside_count} downside)")
        self.logger.debug(f"Breakout statistics: avg_volume_ratio={avg_volume_ratio:.2f}, "
                         f"period={breakout_period}, volume_multiplier={volume_multiplier}")
        
        return df 