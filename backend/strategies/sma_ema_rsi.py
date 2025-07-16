"""Trend and momentum strategies using moving averages and RSI indicators.

This module implements three fundamental trend-following and momentum strategies
that form the core of many quantitative trading systems. These strategies leverage
technical analysis principles to identify market direction and momentum.

Strategy Categories:
1. SmaEmaRsiStrategy: Sophisticated trend following with momentum filter
2. CrossoverStrategy: Classic moving average crossover system
3. RsiStrategy: Pure momentum-based mean reversion approach

Design Philosophy:
- Trend-following strategies work best in trending markets
- Momentum filters help avoid false signals during consolidation
- Multiple timeframes provide robustness across market conditions
- Risk management through overbought/oversold filters

These strategies represent the foundation of systematic trading and can be:
- Used independently for single-factor exposure
- Combined with other strategies for ensemble approaches
- Enhanced with regime detection and sentiment overlays
- Optimized through parameter sweeping and walk-forward analysis

Classes:
    SmaEmaRsiStrategy: Combined trend and momentum strategy
    CrossoverStrategy: Simple moving average crossover
    RsiStrategy: RSI-based momentum strategy

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
from .base import BaseStrategy


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
        
        # Step 2: Calculate moving averages based on configuration
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters['ma_type']
        
        if ma_type.lower() == 'ema':
            # Exponential Moving Average - more responsive to recent prices
            df['fast_ma'] = df['close'].ewm(span=fast_period).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period).mean()
            self.logger.debug(f"Calculated EMA: fast={fast_period}, slow={slow_period}")
        else:
            # Simple Moving Average - equal weight to all periods
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
            self.logger.debug(f"Calculated SMA: fast={fast_period}, slow={slow_period}")
        
        # Step 3: Calculate RSI momentum oscillator
        df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Step 4: Initialize signal column with hold signals
        df['signal'] = 0
        
        # Step 5: Determine basic trend direction from moving average relationship
        trend_bullish = df['fast_ma'] > df['slow_ma']  # Fast MA above slow MA = uptrend
        trend_bearish = df['fast_ma'] < df['slow_ma']  # Fast MA below slow MA = downtrend
        
        # Step 6: Apply RSI momentum filter if enabled
        if self.parameters['use_rsi_filter']:
            # Momentum filter prevents buying at overbought levels and selling at oversold levels
            rsi_not_overbought = df['rsi'] < self.parameters['rsi_overbought']
            rsi_not_oversold = df['rsi'] > self.parameters['rsi_oversold']
            
            # BUY: Confirmed uptrend + momentum not overbought (avoid buying at tops)
            buy_condition = trend_bullish & rsi_not_overbought
            
            # SELL: Confirmed downtrend + momentum not oversold (avoid selling at bottoms)
            sell_condition = trend_bearish & rsi_not_oversold
            
            self.logger.debug("Applied RSI momentum filter to trend signals")
        else:
            # Pure trend following without momentum filter
            buy_condition = trend_bullish
            sell_condition = trend_bearish
            
            self.logger.debug("Using pure trend following without RSI filter")
        
        # Step 7: Apply final signal conditions
        df.loc[buy_condition, 'signal'] = 1   # Buy signal
        df.loc[sell_condition, 'signal'] = -1  # Sell signal
        # Hold signals remain 0 (default)
        
        # Step 8: Add trend analysis columns for strategy evaluation
        df['trend_bullish'] = trend_bullish
        df['trend_bearish'] = trend_bearish
        
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
        
        # Step 3: Calculate moving averages based on type selection
        if ma_type.lower() == 'ema':
            # Exponential Moving Average - more weight to recent prices
            df['fast_ma'] = df['close'].ewm(span=fast_period).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period).mean()
            self.logger.debug(f"Using EMA crossover: {fast_period}/{slow_period}")
        else:
            # Simple Moving Average - equal weight to all periods
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
            self.logger.debug(f"Using SMA crossover: {fast_period}/{slow_period}")
        
        # Step 4: Generate basic trend-following signals
        df['signal'] = 0  # Initialize with hold signals
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1   # Bullish alignment
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Bearish alignment
        
        # Step 5: Identify specific crossover events for enhanced analysis
        # Get previous period values for crossover detection
        fast_ma_prev = df['fast_ma'].shift(1)
        slow_ma_prev = df['slow_ma'].shift(1)
        
        # Golden Cross: Fast MA crosses above slow MA (bullish reversal signal)
        golden_cross = (df['fast_ma'] > df['slow_ma']) & (fast_ma_prev <= slow_ma_prev)
        
        # Death Cross: Fast MA crosses below slow MA (bearish reversal signal)
        death_cross = (df['fast_ma'] < df['slow_ma']) & (fast_ma_prev >= slow_ma_prev)
        
        # Add crossover indicators to output
        df['golden_cross'] = golden_cross
        df['death_cross'] = death_cross
        
        # Step 6: Log comprehensive signal statistics
        total_signals = (df['signal'] != 0).sum()
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        golden_crosses = golden_cross.sum()
        death_crosses = death_cross.sum()
        
        self.logger.info(f"Crossover analysis complete: {total_signals} total signals "
                        f"({buy_signals} bullish, {sell_signals} bearish)")
        self.logger.info(f"Crossover events: {golden_crosses} golden crosses, "
                        f"{death_crosses} death crosses")
        
        return df


class RsiStrategy(BaseStrategy):
    """Pure RSI momentum strategy for mean reversion trading.
    
    This strategy implements a momentum-based approach using the Relative Strength
    Index (RSI) to identify overbought and oversold conditions. Unlike trend-following
    strategies, this approach assumes prices will revert to mean levels after
    extreme momentum readings.
    
    Strategy Philosophy:
    - Markets tend to revert after extreme momentum moves
    - RSI effectively identifies momentum extremes
    - Counter-trend approach complements trend-following strategies
    - Works best in range-bound or mean-reverting markets
    
    Signal Logic:
    - BUY: RSI crosses above oversold threshold (momentum recovery)
    - SELL: RSI crosses below overbought threshold (momentum exhaustion)
    - HOLD: RSI in neutral zone between thresholds
    
    Key Features:
    - Pure momentum-based signals without trend bias
    - Configurable thresholds for different market volatilities
    - Crossover detection prevents premature signals
    - Excellent for range-bound market conditions
    
    Best Applications:
    - Complement to trend-following strategies in ensemble
    - Short-term mean reversion trading
    - Range-bound market environments
    - Counter-trend opportunities during corrections
    
    Example:
        >>> # Conservative mean reversion strategy
        >>> strategy = RsiStrategy({
        ...     'rsi_period': 14,
        ...     'oversold_threshold': 25,
        ...     'overbought_threshold': 75
        ... })
        >>> signals = strategy.generate_signals(hourly_data)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI momentum strategy with configurable thresholds.
        
        Configures the RSI strategy with parameters optimized for mean reversion
        trading. The default thresholds work well for most markets but can be
        adjusted based on asset volatility and market conditions.
        
        Args:
            parameters: Strategy configuration dictionary:
                - rsi_period (int): RSI calculation period (default: 14)
                    Shorter periods = more sensitive to price changes
                    Longer periods = smoother signals, fewer false signals
                - oversold_threshold (float): Buy signal threshold (default: 30)
                    Lower values = more extreme oversold conditions required
                    Higher values = earlier buy signals, more false positives
                - overbought_threshold (float): Sell signal threshold (default: 70)
                    Higher values = more extreme overbought conditions required
                    Lower values = earlier sell signals, more false positives
                    
        Threshold Guidelines:
        - Volatile markets: Use wider thresholds (20/80)
        - Stable markets: Use tighter thresholds (30/70)
        - Trending markets: Use asymmetric thresholds (25/75)
        
        Example:
            >>> # Aggressive mean reversion for volatile crypto markets
            >>> params = {
            ...     'rsi_period': 10,
            ...     'oversold_threshold': 20,
            ...     'overbought_threshold': 80
            ... }
            >>> strategy = RsiStrategy(params)
        """
        # Standard RSI parameters based on Wilder's original research
        default_params = {
            'rsi_period': 14,               # Original Wilder RSI period
            'oversold_threshold': 30,       # Standard oversold level
            'overbought_threshold': 70      # Standard overbought level
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with RSI focus
        super().__init__("RSI Momentum", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based momentum signals for mean reversion trading.
        
        Implements pure momentum-based signal generation using RSI crossover logic.
        This method focuses on identifying momentum exhaustion points rather than
        trend direction, making it complementary to trend-following approaches.
        
        Signal Generation Process:
        1. Validate input data quality and completeness
        2. Calculate RSI momentum oscillator for specified period
        3. Identify RSI crossover events at threshold levels
        4. Generate buy signals on oversold recovery
        5. Generate sell signals on overbought exhaustion
        6. Log signal statistics for performance monitoring
        
        Args:
            data: DataFrame containing market data with 'close' column required
            
        Returns:
            DataFrame with original data plus RSI analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'rsi': RSI momentum oscillator values (0-100 scale)
                
        Signal Timing:
        - BUY signals occur when RSI crosses ABOVE oversold threshold
        - SELL signals occur when RSI crosses BELOW overbought threshold
        - This timing captures momentum shifts rather than extreme levels
        
        Risk Considerations:
        - RSI signals can persist in trending markets
        - Consider combining with trend filters for robustness
        - Backtest thoroughly in different market regimes
        
        Example:
            >>> rsi_signals = strategy.generate_signals(price_data)
            >>> oversold_bounces = rsi_signals[rsi_signals['signal'] == 1]
            >>> print(f"Found {len(oversold_bounces)} oversold bounce opportunities")
        """
        # Step 1: Validate input data for RSI calculation
        if not self.validate_data(data):
            self.logger.error("Data validation failed for RSI strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Calculate RSI momentum oscillator
        rsi_period = self.parameters['rsi_period']
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Step 3: Initialize signals with hold (neutral) position
        df['signal'] = 0
        
        # Step 4: Extract threshold parameters for signal generation
        oversold_threshold = self.parameters['oversold_threshold']
        overbought_threshold = self.parameters['overbought_threshold']
        
        # Step 5: Generate crossover-based signals to avoid premature entries
        
        # BUY: RSI crosses ABOVE oversold threshold (momentum recovery)
        # This indicates potential bottom formation and upward momentum shift
        buy_condition = (df['rsi'] > oversold_threshold) & \
                       (df['rsi'].shift(1) <= oversold_threshold)
        
        # SELL: RSI crosses BELOW overbought threshold (momentum exhaustion)
        # This indicates potential top formation and downward momentum shift
        sell_condition = (df['rsi'] < overbought_threshold) & \
                        (df['rsi'].shift(1) >= overbought_threshold)
        
        # Step 6: Apply signal conditions to generate trading signals
        df.loc[buy_condition, 'signal'] = 1   # Buy on oversold recovery
        df.loc[sell_condition, 'signal'] = -1 # Sell on overbought exhaustion
        
        # Step 7: Log comprehensive signal analysis
        buy_signals = buy_condition.sum()
        sell_signals = sell_condition.sum()
        total_signals = buy_signals + sell_signals
        
        # Additional RSI statistics for strategy evaluation
        avg_rsi = df['rsi'].mean()
        rsi_range = df['rsi'].max() - df['rsi'].min()
        
        self.logger.info(f"RSI momentum analysis: {total_signals} total signals "
                        f"({buy_signals} oversold bounces, {sell_signals} overbought reversals)")
        self.logger.debug(f"RSI statistics: avg={avg_rsi:.1f}, range={rsi_range:.1f}, "
                         f"period={rsi_period}")
        
        return df
