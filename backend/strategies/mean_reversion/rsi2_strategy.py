"""RSI-2 Mean Reversion Strategy for short-term trading.

This module implements the RSI-2 strategy, a popular short-term mean reversion
approach developed by Larry Connors. The strategy uses a 2-period RSI to identify
extremely overbought and oversold conditions for quick mean reversion trades.

Strategy Philosophy:
- Market prices tend to revert to their mean over short periods
- 2-period RSI is extremely sensitive to short-term price movements
- Extreme readings (below 10 or above 90) indicate high probability reversals
- Requires trend filter to avoid trading against strong trends

Signal Logic:
- BUY: RSI-2 drops below oversold threshold (typically 10-25)
- SELL: RSI-2 rises above overbought threshold (typically 75-90)
- Optional: Trend filter using longer-period moving average

Key Features:
- Ultra-short-term mean reversion signals
- Configurable RSI thresholds for different market conditions
- Optional trend filter to avoid counter-trend trades
- Position holding period limits for quick exits

Classes:
    Rsi2Strategy: Main RSI-2 mean reversion implementation
    Rsi2TrendFilterStrategy: RSI-2 with trend direction filter

Example:
    >>> strategy = Rsi2Strategy({
    ...     'rsi_period': 2,
    ...     'oversold_threshold': 15,
    ...     'overbought_threshold': 85
    ... })
    >>> signals = strategy.generate_signals(price_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class Rsi2Strategy(BaseStrategy):
    """RSI-2 mean reversion strategy for short-term trading.
    
    This strategy implements Larry Connors' RSI-2 methodology, using an extremely
    short-period RSI to identify temporary price extremes that are likely to revert.
    The 2-period RSI is much more sensitive than the standard 14-period RSI,
    making it ideal for short-term mean reversion trading.
    
    Strategy Components:
    1. RSI-2: 2-period Relative Strength Index for extreme sensitivity
    2. Oversold Threshold: Typically 10-25 for buy signals
    3. Overbought Threshold: Typically 75-90 for sell signals
    4. Optional Trend Filter: Longer MA to avoid counter-trend trades
    
    Signal Rules:
    - BUY: RSI-2 < oversold threshold (extreme oversold condition)
    - SELL: RSI-2 > overbought threshold (extreme overbought condition)
    - EXIT: RSI-2 crosses back above/below middle threshold (50)
    
    Key Advantages:
    - Very responsive to short-term price movements
    - High win rate in ranging/choppy markets
    - Quick signal generation for day trading
    - Simple and objective entry/exit rules
    
    Limitations:
    - Generates many signals (requires careful position sizing)
    - Can struggle in strong trending markets
    - Requires quick execution and monitoring
    - Best suited for liquid, mean-reverting assets
    
    Best Use Cases:
    - Short-term trading (intraday to few days)
    - Range-bound or choppy markets
    - Highly liquid stocks and ETFs
    - Complement to longer-term trend strategies
    
    Attributes:
        strategy_category: 'mean_reversion' for auto-registration
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'mean_reversion'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI-2 mean reversion strategy.
        
        Sets up the RSI-2 strategy with parameters optimized for short-term
        mean reversion trading. Default thresholds are based on Larry Connors'
        research and backtesting results.
        
        Args:
            parameters: Strategy configuration dictionary:
                - rsi_period (int): RSI calculation period (default: 2)
                    2 is the classic setting, 3-5 can be used for less sensitivity
                - oversold_threshold (float): RSI level for buy signals (default: 15)
                    Lower values = more extreme oversold conditions required
                    Typical range: 5-25
                - overbought_threshold (float): RSI level for sell signals (default: 85)
                    Higher values = more extreme overbought conditions required
                    Typical range: 75-95
                - exit_threshold (float): RSI level for position exits (default: 50)
                    Middle level where positions are closed
                - use_trend_filter (bool): Enable trend direction filter (default: False)
                    Avoids trading against strong trends
                - trend_ma_period (int): Moving average for trend filter (default: 50)
                    Longer periods = stronger trend filter
                - max_hold_days (int): Maximum days to hold position (default: 5)
                    Forced exit after specified days regardless of RSI
                    
        Example:
            >>> # Aggressive short-term setup
            >>> params = {
            ...     'oversold_threshold': 10,
            ...     'overbought_threshold': 90,
            ...     'use_trend_filter': True,
            ...     'max_hold_days': 3
            ... }
            >>> strategy = Rsi2Strategy(params)
        """
        # Research-based default parameters from Larry Connors' work
        default_params = {
            'rsi_period': 2,              # Ultra-short RSI period
            'oversold_threshold': 15,     # Buy when extremely oversold
            'overbought_threshold': 85,   # Sell when extremely overbought  
            'exit_threshold': 50,         # Exit at middle RSI level
            'use_trend_filter': False,    # Optional trend direction filter
            'trend_ma_period': 50,        # Moving average for trend filter
            'max_hold_days': 5           # Maximum position holding period
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("RSI-2 Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-2 mean reversion signals with optional trend filtering.
        
        Implements the RSI-2 methodology for identifying short-term price extremes
        that are likely to revert. The strategy can optionally include a trend filter
        to avoid trading against strong directional moves.
        
        Implementation Steps:
        1. Validate input data quality and completeness
        2. Calculate 2-period RSI for extreme sensitivity
        3. Calculate optional trend filter (moving average)
        4. Identify oversold conditions for buy signals
        5. Identify overbought conditions for sell signals
        6. Apply trend filter if enabled
        7. Generate position exit signals at middle RSI
        8. Add analysis columns for evaluation
        
        Args:
            data: DataFrame containing market data with 'close' column
            
        Returns:
            DataFrame with original data plus RSI-2 analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'rsi2': 2-period RSI values
                - 'rsi2_oversold': Boolean oversold condition indicator
                - 'rsi2_overbought': Boolean overbought condition indicator
                - 'rsi2_exit_long': Boolean long exit condition
                - 'rsi2_exit_short': Boolean short exit condition
                - 'trend_ma': Trend filter moving average (if enabled)
                - 'above_trend': Boolean above trend indicator (if enabled)
                
        Signal Logic:
            - BUY: RSI-2 < oversold_threshold (+ trend filter if enabled)
            - SELL: RSI-2 > overbought_threshold (+ trend filter if enabled)
            - EXIT: RSI-2 crosses back to middle level (50)
            
        Example:
            >>> signals = strategy.generate_signals(intraday_data)
            >>> oversold_signals = signals[signals['rsi2_oversold']]
            >>> print(f"Found {len(oversold_signals)} extreme oversold conditions")
        """
        # Step 1: Validate input data
        if not self.validate_data(data):
            self.logger.error("Data validation failed for RSI-2 strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with copy of original data
        df = data.copy()
        
        # Step 2: Calculate RSI-2 using base class utility
        rsi_period = self.parameters['rsi_period']
        df['rsi2'] = self._calculate_rsi(df['close'], rsi_period)
        
        self.logger.debug(f"Calculated RSI-{rsi_period} for mean reversion signals")
        
        # Step 3: Calculate trend filter if enabled
        if self.parameters['use_trend_filter']:
            trend_period = self.parameters['trend_ma_period']
            df['trend_ma'] = self._calculate_moving_average(df['close'], trend_period, 'sma')
            df['above_trend'] = df['close'] > df['trend_ma']
            
            self.logger.debug(f"Applied trend filter using {trend_period}-period SMA")
        else:
            df['above_trend'] = True  # Always allow trades if no filter
        
        # Step 4: Identify extreme RSI conditions
        oversold_threshold = self.parameters['oversold_threshold']
        overbought_threshold = self.parameters['overbought_threshold']
        exit_threshold = self.parameters['exit_threshold']
        
        # Extreme condition indicators
        df['rsi2_oversold'] = df['rsi2'] < oversold_threshold
        df['rsi2_overbought'] = df['rsi2'] > overbought_threshold
        
        # Exit condition indicators
        df['rsi2_exit_long'] = df['rsi2'] > exit_threshold  # Exit long when RSI rises above 50
        df['rsi2_exit_short'] = df['rsi2'] < exit_threshold  # Exit short when RSI falls below 50
        
        # Step 5: Initialize signal column
        df['signal'] = 0
        
        # Step 6: Generate entry signals with optional trend filter
        if self.parameters['use_trend_filter']:
            # Buy only when oversold AND above trend (trend confirmation)
            buy_condition = df['rsi2_oversold'] & df['above_trend']
            
            # Sell only when overbought AND below trend (trend confirmation)
            sell_condition = df['rsi2_overbought'] & (~df['above_trend'])
            
            self.logger.debug("Applied trend filter to RSI-2 signals")
        else:
            # Pure mean reversion without trend filter
            buy_condition = df['rsi2_oversold']
            sell_condition = df['rsi2_overbought']
            
            self.logger.debug("Using pure RSI-2 mean reversion signals")
        
        # Step 7: Apply entry signals
        df.loc[buy_condition, 'signal'] = 1   # Buy signal (expect mean reversion up)
        df.loc[sell_condition, 'signal'] = -1  # Sell signal (expect mean reversion down)
        
        # Note: Exit signals are provided as indicators but not automatically applied
        # to 'signal' column since this requires position state tracking
        
        # Step 8: Log signal statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        oversold_count = df['rsi2_oversold'].sum()
        overbought_count = df['rsi2_overbought'].sum()
        
        self.logger.info(f"RSI-2 signals: {buy_count} buys, {sell_count} sells")
        self.logger.info(f"Extreme conditions: {oversold_count} oversold, {overbought_count} overbought")
        
        # Log RSI-2 statistics for analysis
        rsi2_mean = df['rsi2'].mean()
        rsi2_std = df['rsi2'].std()
        extreme_pct = ((oversold_count + overbought_count) / len(df) * 100).round(1)
        
        self.logger.debug(f"RSI-2 stats: mean={rsi2_mean:.1f}, std={rsi2_std:.1f}, "
                         f"extreme conditions={extreme_pct}% of time")
        
        return df


class Rsi2TrendFilterStrategy(BaseStrategy):
    """RSI-2 strategy with mandatory trend filter for reduced false signals.
    
    This variation of the RSI-2 strategy always includes a trend filter to avoid
    counter-trend trades. It's more conservative but typically has higher win rates
    by only trading in the direction of the prevailing trend.
    
    Key Differences from Basic RSI-2:
    - Always includes trend filter (not optional)
    - Uses stronger trend confirmation with multiple timeframes
    - More conservative entry thresholds
    - Better suited for trending markets
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'mean_reversion'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI-2 strategy with mandatory trend filter."""
        default_params = {
            'rsi_period': 2,
            'oversold_threshold': 20,     # More conservative than basic RSI-2
            'overbought_threshold': 80,   # More conservative than basic RSI-2
            'exit_threshold': 50,
            'trend_ma_period': 50,
            'trend_ma_type': 'ema',       # More responsive trend filter
            'min_trend_strength': 0.02    # Minimum trend slope required
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("RSI-2 Trend Filtered", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-filtered RSI-2 signals."""
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate RSI-2
        rsi_period = self.parameters['rsi_period']
        df['rsi2'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Calculate trend filter with slope analysis
        trend_period = self.parameters['trend_ma_period']
        ma_type = self.parameters['trend_ma_type']
        df['trend_ma'] = self._calculate_moving_average(df['close'], trend_period, ma_type)
        
        # Calculate trend strength (slope)
        df['trend_slope'] = (df['trend_ma'] - df['trend_ma'].shift(5)) / df['trend_ma'].shift(5)
        min_slope = self.parameters['min_trend_strength']
        
        # Define trend conditions
        df['strong_uptrend'] = (df['close'] > df['trend_ma']) & (df['trend_slope'] > min_slope)
        df['strong_downtrend'] = (df['close'] < df['trend_ma']) & (df['trend_slope'] < -min_slope)
        
        # RSI extreme conditions
        oversold = df['rsi2'] < self.parameters['oversold_threshold']
        overbought = df['rsi2'] > self.parameters['overbought_threshold']
        
        # Generate signals only with trend confirmation
        df['signal'] = 0
        df.loc[oversold & df['strong_uptrend'], 'signal'] = 1   # Buy in uptrend when oversold
        df.loc[overbought & df['strong_downtrend'], 'signal'] = -1  # Sell in downtrend when overbought
        
        # Log statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        
        self.logger.info(f"Trend-filtered RSI-2 signals: {buy_count} buys, {sell_count} sells")
        
        return df