"""Classic Donchian channel breakout strategy for trend identification.

This module implements the time-tested Donchian channel approach popularized
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

Classes:
    ChannelBreakoutStrategy: Classic Donchian channel breakout system

Example:
    >>> # Classic 20-day Donchian channel
    >>> strategy = ChannelBreakoutStrategy({
    ...     'channel_period': 20,
    ...     'breakout_threshold': 0.005  # 0.5% buffer
    ... })
    >>> signals = strategy.generate_signals(daily_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


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
    
    # Strategy metadata for auto-registration
    strategy_category = 'breakout'
    
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
        
        # Step 2: Calculate Donchian channel boundaries using base class utility
        period = self.parameters['channel_period']
        
        # Use base class channel calculation
        channel_levels = self._calculate_channel_levels(df, period)
        df['channel_high'] = channel_levels['channel_high']
        df['channel_low'] = channel_levels['channel_low']
        df['channel_mid'] = channel_levels['channel_mid']
        df['channel_width'] = channel_levels['channel_width']
        
        # Step 3: Calculate dynamic breakout thresholds
        threshold = self.parameters['breakout_threshold']
        
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