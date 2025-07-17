"""Moving average crossover strategies for trend identification and momentum trading.

This module implements sophisticated trend-following strategies based on moving average
relationships. These strategies form the foundation of systematic trading and excel
in trending market conditions.

Strategy Categories:
1. SmaEmaRsiStrategy: Advanced trend following with RSI momentum filter
2. CrossoverStrategy: Classic moving average crossover system

Design Philosophy:
- Trend-following strategies work best in trending markets
- Moving averages smooth price noise and identify direction
- Multiple timeframes provide robustness across market conditions
- RSI filters help avoid false signals during consolidation

These strategies represent the foundation of systematic trading and can be:
- Used independently for single-factor exposure
- Combined with other strategies for ensemble approaches
- Enhanced with regime detection and sentiment overlays
- Optimized through parameter sweeping and walk-forward analysis

Classes:
    SmaEmaRsiStrategy: Combined trend and momentum strategy
    CrossoverStrategy: Simple moving average crossover

Example:
    >>> # Create trend-following strategy
    >>> strategy = SmaEmaRsiStrategy({
    ...     'fast_period': 10,
    ...     'slow_period': 50,
    ...     'use_rsi_filter': True
    ... })
    >>> signals = strategy.generate_signals(price_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class SmaEmaRsiStrategy(BaseStrategy):
    """Advanced trend-following strategy combining moving averages with RSI momentum filter.
    
    This strategy represents a sophisticated approach to trend identification by combining
    the directional clarity of moving average crossovers with the momentum validation
    of RSI. It aims to capture sustained trends while avoiding whipsaws and false signals
    common in pure trend-following systems.
    
    Strategy Logic:
    1. Trend Identification: Fast MA vs Slow MA determines market direction
    2. Momentum Filter: RSI prevents buying at tops and selling at bottoms
    3. Signal Generation: Combines trend direction with momentum conditions
    
    Signal Rules:
    - BUY: Fast MA > Slow MA AND RSI < overbought threshold (trend + momentum alignment)
    - SELL: Fast MA < Slow MA AND RSI > oversold threshold (downtrend + not oversold)
    - HOLD: Mixed signals or extreme RSI conditions
    
    Key Features:
    - Configurable MA types (SMA or EMA) for different responsiveness
    - RSI filter can be disabled for pure trend following
    - Detailed signal analysis with trend condition tracking
    - Robust parameter validation and error handling
    
    Best Use Cases:
    - Trending markets with clear directional moves
    - Medium to long-term position holding
    - Part of multi-strategy ensemble for diversification
    - Base strategy for regime-gated implementations
    
    Attributes:
        name (str): Strategy identifier for reporting
        parameters (Dict): Configuration parameters for optimization
        
    Example:
        >>> # Conservative trend following with RSI filter
        >>> strategy = SmaEmaRsiStrategy({
        ...     'fast_period': 20,
        ...     'slow_period': 100,
        ...     'rsi_period': 14,
        ...     'use_rsi_filter': True
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize SMA/EMA + RSI trend-following strategy with configurable parameters.
        
        Sets up the strategy with default parameters optimized for daily timeframes
        and medium-term trend following. Parameters can be customized for different
        timeframes, market conditions, and risk preferences.
        
        Args:
            parameters: Strategy configuration dictionary with the following options:
                - fast_period (int): Fast moving average lookback period (default: 10)
                    Shorter periods = more responsive to price changes
                    Longer periods = smoother signals, fewer whipsaws
                - slow_period (int): Slow moving average lookback period (default: 50)
                    Should be significantly longer than fast_period for clear signals
                - rsi_period (int): RSI calculation period (default: 14)
                    Standard RSI period, shorter = more sensitive to momentum changes
                - rsi_oversold (float): RSI oversold threshold (default: 30)
                    Lower values = more extreme oversold conditions required
                - rsi_overbought (float): RSI overbought threshold (default: 70)
                    Higher values = more extreme overbought conditions required
                - ma_type (str): Moving average type 'sma' or 'ema' (default: 'sma')
                    SMA = equal weight to all periods, EMA = more weight to recent prices
                - use_rsi_filter (bool): Enable RSI momentum filter (default: True)
                    False = pure trend following, True = momentum-filtered signals
                    
        Note:
            Default parameters are optimized for daily data and medium-term trends.
            For intraday trading, consider shorter periods. For longer-term investing,
            consider longer periods and wider RSI thresholds.
            
        Example:
            >>> # Aggressive short-term strategy
            >>> params = {
            ...     'fast_period': 5,
            ...     'slow_period': 20,
            ...     'ma_type': 'ema',
            ...     'rsi_overbought': 75,
            ...     'rsi_oversold': 25
            ... }
            >>> strategy = SmaEmaRsiStrategy(params)
        """
        # Define robust default parameters based on market research
        default_params = {
            'fast_period': 10,          # Short-term trend detection
            'slow_period': 50,          # Medium-term trend confirmation
            'rsi_period': 14,           # Standard RSI calculation period
            'rsi_oversold': 30,         # Standard oversold threshold
            'rsi_overbought': 70,       # Standard overbought threshold
            'ma_type': 'sma',           # Simple moving average (more stable)
            'use_rsi_filter': True      # Enable momentum filtering by default
        }
        
        # Update defaults with user-provided parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with name and parameters
        super().__init__("SMA/EMA + RSI", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate sophisticated trend-following signals with momentum validation.
        
        This method implements the core strategy logic by calculating moving averages
        and RSI, then combining them according to the strategy rules. It provides
        comprehensive signal generation with detailed intermediate calculations
        for analysis and debugging.
        
        Implementation Steps:
        1. Validate input data quality and completeness
        2. Calculate fast and slow moving averages (SMA or EMA)
        3. Calculate RSI momentum oscillator
        4. Determine trend direction from MA relationship
        5. Apply RSI filter to avoid extreme momentum conditions
        6. Generate final buy/sell/hold signals
        7. Add analysis columns for strategy evaluation
        
        Args:
            data: DataFrame containing market data with required columns:
                - 'close': Closing prices (required for all calculations)
                - 'open', 'high', 'low', 'volume': Optional for enhanced analysis
                Index should be datetime-based for proper time series handling
            
        Returns:
            DataFrame containing original data plus strategy indicators:
                - 'signal': Primary trading signal (1=buy, -1=sell, 0=hold)
                - 'fast_ma': Fast moving average values
                - 'slow_ma': Slow moving average values
                - 'rsi': RSI momentum oscillator values
                - 'trend_bullish': Boolean trend direction indicator
                - 'trend_bearish': Boolean trend direction indicator
                
        Signal Interpretation:
            - 1 (BUY): Uptrend confirmed + momentum not overbought
            - -1 (SELL): Downtrend confirmed + momentum not oversold
            - 0 (HOLD): Mixed signals or extreme momentum conditions
            
        Example:
            >>> data = load_daily_data('AAPL', '2023-01-01', '2024-01-01')
            >>> signals = strategy.generate_signals(data)
            >>> buy_signals = signals[signals['signal'] == 1]
            >>> print(f"Generated {len(buy_signals)} buy signals")
        """
        # Step 1: Validate input data quality
        if not self.validate_data(data):
            self.logger.error("Data validation failed - returning zero signals")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with copy of original data to preserve input
        df = data.copy()
        
        # Step 2: Calculate moving averages using base class utility
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters['ma_type']
        
        df['fast_ma'] = self._calculate_moving_average(df['close'], fast_period, ma_type)
        df['slow_ma'] = self._calculate_moving_average(df['close'], slow_period, ma_type)
        self.logger.debug(f"Calculated {ma_type.upper()}: fast={fast_period}, slow={slow_period}")
        
        # Step 3: Add technical indicators including RSI
        custom_periods = {'rsi_period': self.parameters['rsi_period']}
        df = self.add_technical_indicators(df, custom_periods)
        
        # Step 4: Initialize signal column with hold signals
        df['signal'] = 0
        
        # Step 5: Use base class crossover utility for trend signals
        crossover_signals = self._calculate_crossover_signals(df['fast_ma'], df['slow_ma'])
        df['trend_bullish'] = crossover_signals['trend_signal'] == 1
        df['trend_bearish'] = crossover_signals['trend_signal'] == -1
        
        # Step 6: Apply RSI momentum filter if enabled
        if self.parameters['use_rsi_filter']:
            # Momentum filter prevents buying at overbought levels and selling at oversold levels
            rsi_not_overbought = df['rsi'] < self.parameters['rsi_overbought']
            rsi_not_oversold = df['rsi'] > self.parameters['rsi_oversold']
            
            # BUY: Confirmed uptrend + momentum not overbought (avoid buying at tops)
            buy_condition = df['trend_bullish'] & rsi_not_overbought
            
            # SELL: Confirmed downtrend + momentum not oversold (avoid selling at bottoms)
            sell_condition = df['trend_bearish'] & rsi_not_oversold
            
            self.logger.debug("Applied RSI momentum filter to trend signals")
        else:
            # Pure trend following without momentum filter
            buy_condition = df['trend_bullish']
            sell_condition = df['trend_bearish']
            
            self.logger.debug("Using pure trend following without RSI filter")
        
        # Step 7: Apply final signal conditions
        df.loc[buy_condition, 'signal'] = 1   # Buy signal
        df.loc[sell_condition, 'signal'] = -1  # Sell signal
        # Hold signals remain 0 (default)
        
        # Step 9: Log signal generation statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        hold_count = (df['signal'] == 0).sum()
        
        self.logger.info(f"Generated signals: {buy_count} buys, {sell_count} sells, {hold_count} holds")
        
        return df


class CrossoverStrategy(BaseStrategy):
    """Classic moving average crossover strategy for trend identification.
    
    This strategy implements one of the most fundamental and widely-used technical
    analysis approaches. It identifies trend changes through the relationship between
    fast and slow moving averages, providing clear and objective entry/exit signals.
    
    Strategy Philosophy:
    - Simple and robust approach to trend identification
    - Time-tested method used by traders for decades
    - Clear visual representation of market direction
    - Forms the foundation for more complex strategies
    
    Signal Logic:
    - Golden Cross: Fast MA crosses above Slow MA → BUY signal
    - Death Cross: Fast MA crosses below Slow MA → SELL signal
    - Trend Continuation: Maintain position while MAs remain aligned
    
    Advantages:
    - Easy to understand and implement
    - Works well in strongly trending markets
    - Objective signals reduce emotional decision-making
    - Can be applied to any timeframe or asset class
    
    Limitations:
    - Generates whipsaws in sideways markets
    - Signals often lag significant price moves
    - No built-in risk management or position sizing
    
    Example:
        >>> # Classic 50/200 day golden cross strategy
        >>> strategy = CrossoverStrategy({
        ...     'fast_period': 50,
        ...     'slow_period': 200,
        ...     'ma_type': 'sma'
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize classic moving average crossover strategy.
        
        Sets up the crossover strategy with configurable moving average periods
        and types. Default parameters are suitable for daily timeframes but
        can be adjusted for different market conditions and timeframes.
        
        Args:
            parameters: Strategy configuration with options:
                - fast_period (int): Fast moving average period (default: 10)
                    Shorter periods provide earlier signals but more noise
                - slow_period (int): Slow moving average period (default: 50)
                    Longer periods provide smoother signals but with more lag
                - ma_type (str): Moving average type 'sma' or 'ema' (default: 'sma')
                    SMA for stability, EMA for responsiveness
                    
        Note:
            The ratio between fast and slow periods affects signal quality.
            Common ratios: 2:1 (10/20), 2.5:1 (10/25), 5:1 (10/50), 4:1 (50/200)
            
        Example:
            >>> # Short-term scalping setup
            >>> params = {'fast_period': 5, 'slow_period': 15, 'ma_type': 'ema'}
            >>> strategy = CrossoverStrategy(params)
        """
        # Research-based default parameters for medium-term trend following
        default_params = {
            'fast_period': 10,   # Short-term trend detection
            'slow_period': 50,   # Medium-term trend confirmation
            'ma_type': 'sma'     # Simple moving average for stability
        }
        
        # Merge user parameters with defaults
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy
        super().__init__("MA Crossover", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate moving average crossover signals with detailed analysis.
        
        Implements the classic crossover methodology while providing enhanced
        signal analysis including crossover detection and trend confirmation.
        This method not only generates trading signals but also identifies
        specific crossover events for detailed strategy evaluation.
        
        Implementation Process:
        1. Validate input data for required columns and quality
        2. Calculate fast and slow moving averages based on configuration
        3. Generate basic trend-following signals from MA relationship
        4. Identify specific crossover events (golden cross, death cross)
        5. Add analysis columns for strategy evaluation and visualization
        6. Log comprehensive signal statistics for monitoring
        
        Args:
            data: DataFrame with market data containing 'close' column minimum
            
        Returns:
            DataFrame with original data plus crossover analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'fast_ma': Fast moving average values
                - 'slow_ma': Slow moving average values
                - 'golden_cross': Boolean indicator of bullish crossover
                - 'death_cross': Boolean indicator of bearish crossover
                
        Signal Details:
            - Golden Cross: Fast MA crosses above slow MA (bullish reversal)
            - Death Cross: Fast MA crosses below slow MA (bearish reversal)
            - Trend signals maintain position while MAs remain aligned
            
        Example:
            >>> signals = strategy.generate_signals(price_data)
            >>> golden_crosses = signals[signals['golden_cross']]
            >>> print(f"Found {len(golden_crosses)} golden cross signals")
        """
        # Step 1: Validate input data quality
        if not self.validate_data(data):
            self.logger.error("Data validation failed for crossover strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of input data
        df = data.copy()
        
        # Step 2: Extract configuration parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters['ma_type']
        
        # Step 3: Calculate moving averages using base class utility
        df['fast_ma'] = self._calculate_moving_average(df['close'], fast_period, ma_type)
        df['slow_ma'] = self._calculate_moving_average(df['close'], slow_period, ma_type)
        self.logger.debug(f"Using {ma_type.upper()} crossover: {fast_period}/{slow_period}")
        
        # Step 4: Use base class crossover utility for signal generation
        crossover_signals = self._calculate_crossover_signals(df['fast_ma'], df['slow_ma'])
        
        # Apply trend-following signals
        df['signal'] = crossover_signals['trend_signal']
        
        # Step 5: Add specific crossover event indicators
        df['golden_cross'] = crossover_signals['bullish_signal']  # Golden Cross events
        df['death_cross'] = crossover_signals['bearish_signal']   # Death Cross events
        
        # Step 6: Log comprehensive signal statistics
        total_signals = (df['signal'] != 0).sum()
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        golden_crosses = df['golden_cross'].sum()
        death_crosses = df['death_cross'].sum()
        
        self.logger.info(f"Crossover analysis complete: {total_signals} total signals "
                        f"({buy_signals} bullish, {sell_signals} bearish)")
        self.logger.info(f"Crossover events: {golden_crosses} golden crosses, "
                        f"{death_crosses} death crosses")
        
        return df 