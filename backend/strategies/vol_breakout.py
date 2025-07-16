"""Volatility and breakout strategies for capturing momentum and trend changes.

This module implements three sophisticated breakout strategies designed to capture
significant price movements and trend reversals. Breakout strategies are fundamental
to momentum trading and work by identifying when price breaks through key support
or resistance levels with confirming factors.

Strategy Categories:
1. VolatilityBreakoutStrategy: Multi-factor breakout with volume and volatility confirmation
2. ChannelBreakoutStrategy: Classic Donchian channel breakout system
3. VolumeBreakoutStrategy: Volume spike-driven momentum capture

Design Philosophy:
- Breakouts signal potential trend changes or acceleration
- Volume confirmation reduces false signals
- Volatility filters ensure significant moves
- Multiple timeframes provide robustness
- Risk management through minimum move thresholds

Key Concepts:
- Support/Resistance: Price levels where buying/selling pressure emerges
- Volume Confirmation: High volume validates breakout strength
- False Breakouts: Price briefly exceeds levels then reverses
- Volatility Filtering: ATR helps distinguish noise from real moves

These strategies excel in:
- Trending markets with clear directional moves
- High-volume trading environments
- Breakout from consolidation patterns
- Momentum continuation after news events

Classes:
    VolatilityBreakoutStrategy: Comprehensive multi-factor breakout system
    ChannelBreakoutStrategy: Simple but effective channel breakouts
    VolumeBreakoutStrategy: Volume-driven momentum identification

Example:
    >>> # Create sophisticated breakout strategy
    >>> strategy = VolatilityBreakoutStrategy({
    ...     'breakout_period': 20,
    ...     'volume_multiplier': 2.0,
    ...     'use_atr_filter': True
    ... })
    >>> signals = strategy.generate_signals(price_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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
        
        # Step 2: Calculate rolling breakout levels (support/resistance)
        breakout_period = self.parameters['breakout_period']
        
        # Resistance level: highest high over lookback period (shifted to avoid look-ahead bias)
        df['high_breakout'] = df['high'].rolling(window=breakout_period).max().shift(1)
        
        # Support level: lowest low over lookback period (shifted to avoid look-ahead bias)
        df['low_breakout'] = df['low'].rolling(window=breakout_period).min().shift(1)
        
        # Step 3: Calculate volume confirmation thresholds
        volume_period = self.parameters['volume_period']
        volume_multiplier = self.parameters['volume_multiplier']
        
        # Establish average volume baseline
        df['volume_avg'] = df['volume'].rolling(window=volume_period).mean()
        
        # Volume must exceed this threshold for confirmation
        df['volume_threshold'] = df['volume_avg'] * volume_multiplier
        
        # Step 4: Calculate ATR volatility filter if enabled
        if self.parameters['use_atr_filter']:
            # Calculate Average True Range for volatility measurement
            df['atr'] = self._calculate_atr(df, self.parameters['atr_period'])
            
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


class ChannelBreakoutStrategy(BaseStrategy):
    """Classic Donchian channel breakout strategy for trend identification.
    
    This strategy implements the time-tested Donchian channel approach popularized
    by the Turtle Traders. It identifies breakouts from price channels defined by
    the highest high and lowest low over a specified period, providing simple
    yet effective trend identification signals.
    
    Strategy Philosophy:
    - Price channels define support and resistance zones
    - Breakouts from channels signal potential trend changes
    - Simple and robust approach with minimal parameters
    - Works well across different timeframes and asset classes
    
    Channel Construction:
    - Upper Channel: Highest high over N periods
    - Lower Channel: Lowest low over N periods  
    - Middle Channel: Average of upper and lower bounds
    - Breakout Buffer: Additional threshold to reduce false signals
    
    Signal Logic:
    - BUY: Close > Upper Channel + Buffer (upside breakout)
    - SELL: Close < Lower Channel - Buffer (downside breakout)
    - HOLD: Price remains within channel bounds
    
    Advantages:
    - Simple to understand and implement
    - Self-adjusting to market volatility
    - Effective in trending markets
    - Minimal parameter optimization required
    
    Limitations:
    - Generates whipsaws in range-bound markets
    - Signals often lag significant moves
    - No volume or momentum confirmation
    
    Example:
        >>> # Classic 20-day Donchian channel
        >>> strategy = ChannelBreakoutStrategy({
        ...     'channel_period': 20,
        ...     'breakout_threshold': 0.005  # 0.5% buffer
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize classic Donchian channel breakout strategy.
        
        Sets up the channel breakout strategy with configurable period and
        threshold parameters. Default values are based on the original
        Turtle Trading system but can be adjusted for different markets.
        
        Args:
            parameters: Strategy configuration dictionary:
                - channel_period (int): Period for channel calculation (default: 20)
                    Shorter periods = more sensitive to recent price action
                    Longer periods = more significant breakouts required
                    Common values: 10 (short-term), 20 (medium-term), 55 (long-term)
                - breakout_threshold (float): % threshold for breakout confirmation (default: 0.01)
                    Prevents minor fluctuations from triggering false signals
                    Higher values = fewer signals but higher quality
                    Lower values = more signals but potentially more noise
                    
        Historical Context:
        The 20-day Donchian channel was popularized by Richard Dennis and
        the Turtle Traders in the 1980s. The 55-day channel was used for
        longer-term trend identification.
        
        Example:
            >>> # Turtle Trading inspired setup
            >>> params = {
            ...     'channel_period': 55,    # Long-term trend identification
            ...     'breakout_threshold': 0  # No buffer for pure system
            ... }
            >>> strategy = ChannelBreakoutStrategy(params)
        """
        # Turtle Trading inspired default parameters
        default_params = {
            'channel_period': 20,           # Classic 20-day Donchian channel
            'breakout_threshold': 0.01      # 1% buffer to reduce false signals
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with channel focus
        super().__init__("Channel Breakout", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Donchian channel breakout signals with configurable buffers.
        
        Implements the classic channel breakout methodology with optional
        threshold buffers to reduce false signals. This method calculates
        dynamic support and resistance levels and generates signals when
        price breaks decisively outside the established channel.
        
        Implementation Steps:
        1. Validate input data for minimum requirements
        2. Calculate rolling channel boundaries (high, low, middle)
        3. Apply breakout thresholds to reduce noise
        4. Generate signals based on channel breakouts
        5. Add channel analysis for strategy evaluation
        6. Log channel breakout statistics
        
        Args:
            data: DataFrame containing market data with 'high', 'low', 'close' columns
            
        Returns:
            DataFrame with original data plus channel analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'channel_high': Upper channel boundary (resistance)
                - 'channel_low': Lower channel boundary (support)
                - 'channel_mid': Middle channel line (trend reference)
                - 'channel_width': Channel width (volatility measure)
                - 'breakout_buffer': Dynamic breakout threshold
                
        Channel Interpretation:
            - Narrow channels: Low volatility, potential breakout setup
            - Wide channels: High volatility, established trend
            - Price at channel_mid: Neutral/undecided market
            - Breakouts: Potential trend initiation or continuation
            
        Example:
            >>> signals = strategy.generate_signals(price_data)
            >>> upper_breakouts = signals[signals['signal'] == 1]
            >>> channel_width = signals['channel_width'].mean()
            >>> print(f"Average channel width: {channel_width:.2f}")
        """
        # Step 1: Validate input data for channel calculation
        if not self.validate_data(data):
            self.logger.error("Data validation failed for channel breakout strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Calculate Donchian channel boundaries
        period = self.parameters['channel_period']
        
        # Upper channel: highest high over period (resistance level)
        df['channel_high'] = df['high'].rolling(window=period).max()
        
        # Lower channel: lowest low over period (support level)
        df['channel_low'] = df['low'].rolling(window=period).min()
        
        # Middle channel: average of upper and lower bounds (trend reference)
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
        
        # Step 3: Calculate dynamic breakout thresholds
        threshold = self.parameters['breakout_threshold']
        
        # Channel width as volatility proxy
        df['channel_width'] = df['channel_high'] - df['channel_low']
        
        # Breakout buffer based on channel width (adapts to volatility)
        df['breakout_buffer'] = df['channel_width'] * threshold
        
        # Step 4: Initialize signals with hold position
        df['signal'] = 0
        
        # Step 5: Generate breakout signals with threshold buffers
        
        # BUY: Close breaks above upper channel + buffer (upside breakout)
        upper_breakout = df['close'] > (df['channel_high'] + df['breakout_buffer'])
        
        # SELL: Close breaks below lower channel - buffer (downside breakout)
        lower_breakout = df['close'] < (df['channel_low'] - df['breakout_buffer'])
        
        # Apply breakout signals
        df.loc[upper_breakout, 'signal'] = 1   # Buy on upper channel breakout
        df.loc[lower_breakout, 'signal'] = -1  # Sell on lower channel breakout
        
        # Step 6: Log comprehensive channel statistics
        upper_count = upper_breakout.sum()
        lower_count = lower_breakout.sum()
        total_signals = upper_count + lower_count
        
        # Channel analysis statistics
        avg_channel_width = df['channel_width'].mean()
        max_channel_width = df['channel_width'].max()
        min_channel_width = df['channel_width'].min()
        
        self.logger.info(f"Channel breakout analysis: {total_signals} total signals "
                        f"({upper_count} upper, {lower_count} lower)")
        self.logger.debug(f"Channel statistics: period={period}, avg_width={avg_channel_width:.2f}, "
                         f"threshold={threshold:.3f}")
        
        return df


class VolumeBreakoutStrategy(BaseStrategy):
    """Volume-driven breakout strategy focusing on unusual trading activity.
    
    This strategy takes a unique approach to breakout detection by prioritizing
    volume analysis over price levels. It identifies periods of unusually high
    trading activity combined with significant price moves, capturing momentum
    driven by institutional activity or news events.
    
    Strategy Philosophy:
    - Volume precedes price movement
    - Unusual volume indicates informed trading
    - Price moves with volume confirmation are more sustainable
    - Institutional activity creates detectable volume patterns
    
    Volume Analysis Components:
    1. Volume Spike Detection: Identifying unusually high volume
    2. Price Move Validation: Ensuring volume accompanies significant price change
    3. Direction Determination: Matching volume with price direction
    
    Signal Logic:
    - BUY: Volume spike + significant positive price move
    - SELL: Volume spike + significant negative price move
    - HOLD: Normal volume or insufficient price movement
    
    Key Features:
    - Adapts to changing volume patterns
    - Captures news-driven moves early
    - Works well in liquid markets
    - Complementary to price-based strategies
    
    Best Applications:
    - High-volume liquid stocks
    - News and earnings event trading
    - Intraday momentum capture
    - Institutional flow detection
    
    Limitations:
    - Requires reliable volume data
    - Less effective in thin markets
    - Can generate false signals on settlement days
    - Sensitive to unusual market conditions
    
    Example:
        >>> # Sensitive volume breakout for active stocks
        >>> strategy = VolumeBreakoutStrategy({
        ...     'volume_period': 10,            # Short baseline
        ...     'volume_spike_multiplier': 2.5, # Moderate spike requirement
        ...     'price_move_threshold': 0.015   # 1.5% price move minimum
        ... })
        >>> signals = strategy.generate_signals(intraday_data)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize volume-driven breakout strategy with spike detection.
        
        Configures the volume breakout strategy to identify unusual trading
        activity combined with significant price movements. Parameters can
        be adjusted based on the typical volume patterns of target assets.
        
        Args:
            parameters: Strategy configuration dictionary:
                - volume_period (int): Period for volume average baseline (default: 20)
                    Shorter periods = more sensitive to recent volume changes
                    Longer periods = more stable baseline, fewer false signals
                - volume_spike_multiplier (float): Volume spike threshold (default: 3.0)
                    Multiple of average volume required for spike detection
                    Higher values = require more extreme volume for signals
                    Lower values = more sensitive to volume increases
                - price_move_threshold (float): Minimum price move % (default: 0.02)
                    Percentage price change required to confirm breakout
                    Higher values = reduce noise but may miss smaller moves
                    Lower values = more signals but potentially more false positives
                    
        Parameter Guidelines:
        - High-volume stocks: Use higher volume_spike_multiplier (4.0+)
        - Volatile stocks: Use higher price_move_threshold (0.03+)
        - Intraday trading: Use shorter volume_period (10-15)
        - Daily trading: Use longer volume_period (20-30)
        
        Example:
            >>> # Conservative setup for large-cap stocks
            >>> params = {
            ...     'volume_period': 30,        # Stable baseline
            ...     'volume_spike_multiplier': 4.0,  # High threshold
            ...     'price_move_threshold': 0.025     # 2.5% minimum move
            ... }
            >>> strategy = VolumeBreakoutStrategy(params)
        """
        # Volume-focused default parameters for momentum detection
        default_params = {
            'volume_period': 20,            # 20-day volume baseline
            'volume_spike_multiplier': 3.0, # 3x average volume spike
            'price_move_threshold': 0.02    # 2% minimum price move
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with volume focus
        super().__init__("Volume Breakout", default_params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data contains volume column for volume analysis.
        
        Volume breakout strategies specifically require volume data for analysis.
        This method ensures volume information is available before processing.
        
        Args:
            data: DataFrame to validate for volume requirements
            
        Returns:
            bool: True if volume data is available, False otherwise
        """
        if 'volume' not in data.columns:
            self.logger.error("Volume column required for volume breakout strategy")
            return False
        
        # Call parent validation for additional standard checks
        return super().validate_data(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-driven breakout signals based on trading activity spikes.
        
        Implements volume-centric breakout detection by identifying periods of
        unusual trading activity combined with significant price movements.
        This approach captures momentum driven by institutional activity or
        news events that may not be apparent in price-only analysis.
        
        Implementation Process:
        1. Validate input data contains required volume information
        2. Calculate rolling volume baseline and spike thresholds
        3. Identify volume spikes exceeding normal activity
        4. Calculate price movement magnitudes
        5. Generate signals combining volume spikes with price moves
        6. Add volume analysis columns for evaluation
        7. Log volume breakout statistics and patterns
        
        Args:
            data: DataFrame containing market data with 'close' and 'volume' columns
            
        Returns:
            DataFrame with original data plus volume analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'volume_avg': Rolling average volume baseline
                - 'volume_spike': Boolean indicator of volume spike
                - 'price_change': Period-over-period price change percentage
                - 'volume_ratio': Current volume / average volume ratio
                
        Signal Conditions:
            BUY Signal:
            - Volume > (average volume × spike multiplier)
            - Price change > +price_move_threshold
            
            SELL Signal:
            - Volume > (average volume × spike multiplier)
            - Price change < -price_move_threshold
            
        Volume Spike Analysis:
        The strategy identifies volume spikes as periods where trading volume
        significantly exceeds recent averages, indicating increased interest
        from institutions or informed traders.
        
        Example:
            >>> signals = strategy.generate_signals(stock_data)
            >>> volume_breakouts = signals[signals['volume_spike']]
            >>> avg_volume_ratio = signals['volume_ratio'].mean()
            >>> print(f"Found {len(volume_breakouts)} volume spikes")
        """
        # Step 1: Validate input data contains volume information
        if not self.validate_data(data):
            self.logger.error("Data validation failed for volume breakout strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Extract strategy parameters for volume analysis
        period = self.parameters['volume_period']
        multiplier = self.parameters['volume_spike_multiplier']
        
        # Step 3: Calculate volume baseline and spike detection
        
        # Establish normal volume baseline using rolling average
        df['volume_avg'] = df['volume'].rolling(window=period).mean()
        
        # Identify volume spikes: current volume exceeds threshold
        df['volume_spike'] = df['volume'] > (df['volume_avg'] * multiplier)
        
        # Step 4: Calculate price movement analysis
        
        # Period-over-period price change percentage
        df['price_change'] = df['close'].pct_change()
        
        # Extract minimum price move threshold
        threshold = self.parameters['price_move_threshold']
        
        # Step 5: Initialize signals with hold position
        df['signal'] = 0
        
        # Step 6: Generate volume-confirmed breakout signals
        
        # BUY: Volume spike with significant positive price movement
        buy_condition = df['volume_spike'] & (df['price_change'] > threshold)
        
        # SELL: Volume spike with significant negative price movement
        sell_condition = df['volume_spike'] & (df['price_change'] < -threshold)
        
        # Apply volume breakout signals
        df.loc[buy_condition, 'signal'] = 1   # Buy on volume-confirmed upside move
        df.loc[sell_condition, 'signal'] = -1 # Sell on volume-confirmed downside move
        
        # Step 7: Add comprehensive volume analysis columns
        
        # Volume ratio: current volume relative to average (strength indicator)
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        # Step 8: Log detailed volume breakout statistics
        buy_signals = buy_condition.sum()
        sell_signals = sell_condition.sum()
        total_signals = buy_signals + sell_signals
        total_volume_spikes = df['volume_spike'].sum()
        
        # Volume pattern analysis
        avg_volume_ratio = df['volume_ratio'].mean()
        max_volume_ratio = df['volume_ratio'].max()
        spike_rate = (total_volume_spikes / len(df)) * 100
        
        self.logger.info(f"Volume breakout analysis: {total_signals} total signals "
                        f"({buy_signals} volume buy spikes, {sell_signals} volume sell spikes)")
        self.logger.info(f"Volume patterns: {total_volume_spikes} total spikes "
                        f"({spike_rate:.1f}% spike rate)")
        self.logger.debug(f"Volume statistics: avg_ratio={avg_volume_ratio:.2f}, "
                         f"max_ratio={max_volume_ratio:.2f}, multiplier={multiplier}")
        
        return df
