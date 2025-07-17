"""Bollinger Band-based mean reversion strategy for volatility-adjusted contrarian trading.

This module leverages the statistical properties of Bollinger Bands to identify
potential mean reversion opportunities. Bollinger Bands create dynamic support and
resistance levels that expand and contract with market volatility, providing
context-aware entry and exit points for contrarian trades.

Strategy Philosophy:
- Price touches of band extremes indicate temporary overextension
- Volatility bands automatically adjust to market conditions
- Mean reversion is more likely when RSI confirms oversold/overbought conditions
- Middle band serves as natural profit-taking level

Bollinger Band Construction:
- Middle Band: N-period simple moving average (typically 20)
- Upper Band: Middle band + (K × standard deviation)
- Lower Band: Middle band - (K × standard deviation)
- Band Width: Measure of current market volatility

Classes:
    BollingerBandMeanReversionStrategy: Volatility band-based reversion

Example:
    >>> # Conservative Bollinger Band setup for daily trading
    >>> strategy = BollingerBandMeanReversionStrategy({
    ...     'bb_period': 20,        # Standard period
    ...     'bb_std_dev': 2.5,      # Wider bands for quality signals
    ...     'use_rsi_filter': True, # Momentum confirmation
    ...     'rsi_oversold': 25,     # Strict oversold level
    ...     'rsi_overbought': 75    # Strict overbought level
    ... })
    >>> signals = strategy.generate_signals(daily_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class BollingerBandMeanReversionStrategy(BaseStrategy):
    """Bollinger Band-based mean reversion strategy for volatility-adjusted contrarian trading.
    
    This strategy leverages the statistical properties of Bollinger Bands to identify
    potential mean reversion opportunities. Bollinger Bands create dynamic support and
    resistance levels that expand and contract with market volatility, providing
    context-aware entry and exit points for contrarian trades.
    
    Strategy Philosophy:
    - Price touches of band extremes indicate temporary overextension
    - Volatility bands automatically adjust to market conditions
    - Mean reversion is more likely when RSI confirms oversold/overbought conditions
    - Middle band serves as natural profit-taking level
    
    Bollinger Band Construction:
    - Middle Band: N-period simple moving average (typically 20)
    - Upper Band: Middle band + (K × standard deviation)
    - Lower Band: Middle band - (K × standard deviation)
    - Band Width: Measure of current market volatility
    
    Signal Logic:
    - BUY: Price touches/crosses below lower band (oversold condition)
    - SELL: Price touches/crosses above upper band (overbought condition)
    - EXIT: Price returns to middle band (mean reversion complete)
    - FILTER: RSI confirmation prevents counter-trend signals
    
    Key Features:
    - Self-adjusting volatility bands
    - Optional RSI momentum filter
    - Configurable exit strategies
    - Band position analysis for signal strength
    
    Advantages:
    - Adapts automatically to changing volatility
    - Clear visual representation of extremes
    - Well-tested statistical foundation
    - Effective in ranging markets
    
    Limitations:
    - Poor performance in strong trends
    - Whipsaws during band expansion/contraction
    - Requires sufficient volatility for band separation
    
    Example:
        >>> # Conservative Bollinger Band setup for daily trading
        >>> strategy = BollingerBandMeanReversionStrategy({
        ...     'bb_period': 20,        # Standard period
        ...     'bb_std_dev': 2.5,      # Wider bands for quality signals
        ...     'use_rsi_filter': True, # Momentum confirmation
        ...     'rsi_oversold': 25,     # Strict oversold level
        ...     'rsi_overbought': 75    # Strict overbought level
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'mean_reversion'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Bollinger Band mean reversion strategy with volatility adaptation.
        
        Configures the strategy with Bollinger Band parameters and optional RSI filter
        for enhanced signal quality. Default parameters are optimized for daily
        timeframes but can be adjusted for different market conditions.
        
        Args:
            parameters: Strategy configuration dictionary with options:
                - bb_period (int): Bollinger Band calculation period (default: 20)
                    Shorter periods = more responsive but noisier signals
                    Longer periods = smoother bands but delayed signals
                    Standard: 20 for daily, 10 for hourly, 50 for weekly
                - bb_std_dev (float): Standard deviation multiplier (default: 2.0)
                    Higher values = wider bands, fewer but higher quality signals
                    Lower values = tighter bands, more signals but potentially more noise
                    Common values: 1.5 (tight), 2.0 (standard), 2.5 (wide)
                - use_rsi_filter (bool): Enable RSI momentum filter (default: True)
                    True = additional confirmation required, fewer false signals
                    False = pure Bollinger Band signals, more aggressive
                - rsi_period (int): RSI calculation period (default: 14)
                    Standard momentum oscillator period
                - rsi_oversold (float): RSI oversold threshold (default: 30)
                    Lower values = more extreme oversold conditions required
                - rsi_overbought (float): RSI overbought threshold (default: 70)
                    Higher values = more extreme overbought conditions required
                - exit_at_middle (bool): Exit positions at middle band (default: True)
                    True = take profits at mean reversion completion
                    False = hold until opposite band signal
                    
        Parameter Optimization Tips:
        - Volatile markets: Increase bb_std_dev to 2.5 or 3.0
        - Trending markets: Disable RSI filter or use asymmetric thresholds
        - Short-term trading: Decrease bb_period and enable exit_at_middle
        - Position trading: Increase bb_period and disable exit_at_middle
        
        Example:
            >>> # Aggressive intraday mean reversion setup
            >>> params = {
            ...     'bb_period': 14,        # Shorter period for responsiveness
            ...     'bb_std_dev': 1.8,      # Tighter bands for more signals
            ...     'use_rsi_filter': False, # No momentum filter
            ...     'exit_at_middle': True   # Quick profit taking
            ... }
            >>> strategy = BollingerBandMeanReversionStrategy(params)
        """
        # Bollinger Band standard parameters based on John Bollinger's research
        default_params = {
            'bb_period': 20,            # Standard 20-period moving average
            'bb_std_dev': 2.0,          # 2 standard deviation bands (95% confidence)
            'use_rsi_filter': True,     # Enable momentum confirmation
            'rsi_period': 14,           # Standard RSI period
            'rsi_oversold': 30,         # Standard oversold threshold
            'rsi_overbought': 70,       # Standard overbought threshold
            'exit_at_middle': True      # Exit at mean reversion completion
        }
        
        # Update defaults with user parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with Bollinger Band focus
        super().__init__("Bollinger Band Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band mean reversion signals with volatility analysis.
        
        Implements comprehensive Bollinger Band analysis including band calculation,
        position analysis, optional RSI filtering, and exit signal generation.
        This method provides detailed mean reversion signals with statistical context.
        
        Implementation Process:
        1. Validate input data for required price information
        2. Calculate Bollinger Bands (middle, upper, lower)
        3. Compute band position for signal strength analysis
        4. Apply RSI momentum filter if enabled
        5. Generate buy signals at lower band touches
        6. Generate sell signals at upper band touches
        7. Create exit signals at middle band if enabled
        8. Add comprehensive analysis columns
        
        Args:
            data: DataFrame containing market data with 'close' column required
            
        Returns:
            DataFrame with original data plus Bollinger Band analysis:
                - 'signal': Primary trading signal (1=buy, -1=sell, 0=hold)
                - 'bb_middle': Middle Bollinger Band (moving average)
                - 'bb_upper': Upper Bollinger Band (resistance)
                - 'bb_lower': Lower Bollinger Band (support)
                - 'bb_std': Rolling standard deviation
                - 'bb_position': Relative position within bands (0-1 scale)
                - 'rsi': RSI values (if RSI filter enabled)
                - 'at_lower_band': Boolean lower band touch indicator
                - 'at_upper_band': Boolean upper band touch indicator
                - 'middle_exit_long': Middle band exit for long positions
                - 'middle_exit_short': Middle band exit for short positions
                
        Band Position Interpretation:
            - 0.0: Price at lower band (maximum oversold)
            - 0.5: Price at middle band (fair value)
            - 1.0: Price at upper band (maximum overbought)
            
        Signal Quality Indicators:
        Band position provides context for signal strength:
        - Positions < 0.2 or > 0.8 indicate strong mean reversion potential
        - RSI confirmation adds momentum validation to band signals
        
        Example:
            >>> signals = strategy.generate_signals(stock_data)
            >>> lower_band_signals = signals[signals['at_lower_band']]
            >>> avg_band_position = signals['bb_position'].mean()
            >>> print(f"Average band position: {avg_band_position:.2f}")
        """
        # Step 1: Validate input data quality
        if not self.validate_data(data):
            self.logger.error("Data validation failed for Bollinger Band strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Add Bollinger Bands using base class utility
        custom_periods = {
            'bb_period': self.parameters['bb_period'],
            'bb_std_dev': self.parameters['bb_std_dev']
        }
        df = self.add_technical_indicators(df, custom_periods)
        
        # Add custom BB calculations
        df['bb_std'] = df['close'].rolling(window=self.parameters['bb_period']).std()
        
        # Step 3: Add RSI if filter is enabled (already included in add_technical_indicators if needed)
        if self.parameters['use_rsi_filter']:
            # RSI already calculated by add_technical_indicators
            pass
        
        # Step 5: Initialize signal column with hold positions
        df['signal'] = 0
        
        # Step 6: Define mean reversion entry conditions
        
        # BUY condition: Price touches or penetrates lower Bollinger Band
        buy_condition = df['close'] <= df['bb_lower']
        
        # SELL condition: Price touches or penetrates upper Bollinger Band
        sell_condition = df['close'] >= df['bb_upper']
        
        # Step 7: Apply RSI momentum filter if enabled
        if self.parameters['use_rsi_filter']:
            # Only buy when RSI confirms oversold condition
            rsi_oversold = df['rsi'] <= self.parameters['rsi_oversold']
            buy_condition = buy_condition & rsi_oversold
            
            # Only sell when RSI confirms overbought condition
            rsi_overbought = df['rsi'] >= self.parameters['rsi_overbought']
            sell_condition = sell_condition & rsi_overbought
            
            self.logger.debug("Applied RSI filter to Bollinger Band signals")
        
        # Step 8: Generate primary trading signals
        df.loc[buy_condition, 'signal'] = 1   # Buy at lower band
        df.loc[sell_condition, 'signal'] = -1 # Sell at upper band
        
        # Step 9: Generate exit signals at middle band if enabled
        if self.parameters['exit_at_middle']:
            # Exit long positions when price crosses above middle band
            middle_cross_up = (df['close'] > df['bb_middle']) & \
                            (df['close'].shift(1) <= df['bb_middle'])
            
            # Exit short positions when price crosses below middle band
            middle_cross_down = (df['close'] < df['bb_middle']) & \
                              (df['close'].shift(1) >= df['bb_middle'])
            
            # Add exit indicators for analysis
            df['middle_exit_long'] = middle_cross_up
            df['middle_exit_short'] = middle_cross_down
            
            self.logger.debug("Enabled middle band exit signals")
        
        # Step 10: Add comprehensive analysis columns
        df['at_lower_band'] = buy_condition      # Lower band touch indicator
        df['at_upper_band'] = sell_condition     # Upper band touch indicator
        
        # Step 11: Log detailed Bollinger Band statistics
        buy_signals = buy_condition.sum()
        sell_signals = sell_condition.sum()
        total_signals = buy_signals + sell_signals
        
        # Band analysis statistics
        avg_band_position = df['bb_position'].mean()
        avg_band_width = (df['bb_upper'] - df['bb_lower']).mean()
        
        self.logger.info(f"Bollinger Band analysis: {total_signals} total signals "
                        f"({buy_signals} lower band, {sell_signals} upper band)")
        self.logger.debug(f"Band statistics: avg_position={avg_band_position:.3f}, "
                         f"avg_width={avg_band_width:.2f}, std_dev={self.parameters['bb_std_dev']}")
        
        return df 