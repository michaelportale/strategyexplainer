"""RSI momentum strategy for mean reversion and momentum trading.

This module implements a pure RSI-based momentum strategy that identifies
overbought and oversold conditions. Unlike trend-following strategies,
this approach assumes prices will revert to mean levels after extreme
momentum readings.

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

Classes:
    RsiStrategy: RSI-based momentum strategy

Example:
    >>> # Create conservative mean reversion strategy
    >>> strategy = RsiStrategy({
    ...     'rsi_period': 14,
    ...     'oversold_threshold': 25,
    ...     'overbought_threshold': 75
    ... })
    >>> signals = strategy.generate_signals(hourly_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


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
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
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
        
        # Step 2: Add technical indicators including RSI
        custom_periods = {'rsi_period': self.parameters['rsi_period']}
        df = self.add_technical_indicators(df, custom_periods)
        
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
                         f"period={self.parameters['rsi_period']}")
        
        return df 