"""RSI-based mean reversion strategy with advanced divergence detection.

This module uses the Relative Strength Index (RSI) as the primary mean reversion
indicator, enhanced with divergence analysis for higher-quality signals. RSI is
particularly effective for mean reversion because it oscillates between fixed
bounds (0-100) and has well-established overbought/oversold levels.

Strategy Philosophy:
- RSI extremes indicate temporary momentum exhaustion
- Divergences between price and RSI reveal underlying weakness/strength
- Multiple RSI levels provide signal gradation and risk management
- Mean reversion works best when momentum indicators confirm price extremes

RSI Analysis Components:
1. Standard Levels: Traditional 30/70 oversold/overbought thresholds
2. Extreme Levels: More conservative 15/85 thresholds for high-confidence signals
3. Divergence Detection: Price vs. RSI direction mismatches
4. Signal Strength: Graduated signals based on RSI level and divergence presence

Classes:
    RSIMeanReversionStrategy: RSI-based momentum strategy with divergence detection

Example:
    >>> # Comprehensive RSI mean reversion with divergence
    >>> strategy = RSIMeanReversionStrategy({
    ...     'rsi_period': 14,              # Standard RSI period
    ...     'oversold_threshold': 25,      # Conservative oversold
    ...     'overbought_threshold': 75,    # Conservative overbought
    ...     'extreme_oversold': 15,        # High-confidence level
    ...     'extreme_overbought': 85,      # High-confidence level
    ...     'use_divergence': True,        # Enable divergence detection
    ...     'divergence_lookback': 12      # Divergence detection window
    ... })
    >>> signals = strategy.generate_signals(daily_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based mean reversion strategy with advanced divergence detection.
    
    This strategy uses the Relative Strength Index (RSI) as the primary mean reversion
    indicator, enhanced with divergence analysis for higher-quality signals. RSI is
    particularly effective for mean reversion because it oscillates between fixed
    bounds (0-100) and has well-established overbought/oversold levels.
    
    Strategy Philosophy:
    - RSI extremes indicate temporary momentum exhaustion
    - Divergences between price and RSI reveal underlying weakness/strength
    - Multiple RSI levels provide signal gradation and risk management
    - Mean reversion works best when momentum indicators confirm price extremes
    
    RSI Analysis Components:
    1. Standard Levels: Traditional 30/70 oversold/overbought thresholds
    2. Extreme Levels: More conservative 15/85 thresholds for high-confidence signals
    3. Divergence Detection: Price vs. RSI direction mismatches
    4. Signal Strength: Graduated signals based on RSI level and divergence presence
    
    Signal Rules:
    - BUY: RSI < oversold threshold (standard or extreme)
    - SELL: RSI > overbought threshold (standard or extreme)
    - STRONG BUY/SELL: RSI at extreme levels (higher confidence)
    - DIVERGENCE SIGNALS: Price/RSI divergence at overbought/oversold levels
    
    Key Features:
    - Multiple signal strength levels for position sizing
    - Bullish/bearish divergence detection
    - Configurable thresholds for different market conditions
    - Comprehensive RSI analysis beyond basic oscillator signals
    
    Best Applications:
    - Range-bound markets with clear momentum cycles
    - Swing trading with 3-10 day holding periods
    - Divergence trading for trend reversal signals
    - Risk-graded position sizing based on signal strength
    
    Advanced Features:
    Signal strength gradation enables sophisticated position management:
    - Level 1: Standard RSI signals
    - Level 2: Extreme RSI signals (higher confidence)
    - Level 3: Divergence signals (highest confidence)
    
    Example:
        >>> # Comprehensive RSI mean reversion with divergence
        >>> strategy = RSIMeanReversionStrategy({
        ...     'rsi_period': 14,              # Standard RSI period
        ...     'oversold_threshold': 25,      # Conservative oversold
        ...     'overbought_threshold': 75,    # Conservative overbought
        ...     'extreme_oversold': 15,        # High-confidence level
        ...     'extreme_overbought': 85,      # High-confidence level
        ...     'use_divergence': True,        # Enable divergence detection
        ...     'divergence_lookback': 12      # Divergence detection window
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'mean_reversion'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI mean reversion strategy with multi-level thresholds.
        
        Configures the strategy with RSI parameters including standard and extreme
        thresholds, plus optional divergence detection for enhanced signal quality.
        Default parameters provide conservative mean reversion signals.
        
        Args:
            parameters: Strategy configuration dictionary with options:
                - rsi_period (int): RSI calculation period (default: 14)
                    Standard Wilder RSI period, shorter = more sensitive
                    Common alternatives: 9 (faster), 21 (slower)
                - oversold_threshold (float): Standard oversold level (default: 25)
                    Lower than traditional 30 for more conservative signals
                    Range: 20-35 depending on market volatility
                - overbought_threshold (float): Standard overbought level (default: 75)
                    Higher than traditional 70 for more conservative signals
                    Range: 65-80 depending on market volatility
                - extreme_oversold (float): Extreme oversold level (default: 15)
                    Very low RSI indicating high-confidence reversal setup
                    Typically 10-20 for rare but high-quality signals
                - extreme_overbought (float): Extreme overbought level (default: 85)
                    Very high RSI indicating high-confidence reversal setup
                    Typically 80-90 for rare but high-quality signals
                - use_divergence (bool): Enable divergence detection (default: True)
                    True = scan for bullish/bearish divergences
                    False = pure RSI level-based signals
                - divergence_lookback (int): Divergence detection window (default: 10)
                    Periods to scan for price/RSI directional mismatches
                    Shorter = more sensitive, longer = more significant divergences
                    
        Threshold Selection Guidelines:
        - Volatile markets: Use wider thresholds (20/80, 10/90)
        - Stable markets: Use tighter thresholds (30/70, 20/80)
        - Trending markets: Use asymmetric thresholds (25/80 in uptrend)
        
        Example:
            >>> # Aggressive RSI setup for active trading
            >>> params = {
            ...     'rsi_period': 9,            # Faster RSI
            ...     'oversold_threshold': 30,   # Traditional level
            ...     'overbought_threshold': 70, # Traditional level
            ...     'extreme_oversold': 20,     # Less extreme
            ...     'extreme_overbought': 80,   # Less extreme
            ...     'use_divergence': False     # Disable for speed
            ... }
            >>> strategy = RSIMeanReversionStrategy(params)
        """
        # Conservative RSI parameters optimized for mean reversion
        default_params = {
            'rsi_period': 14,               # Standard Wilder RSI period
            'oversold_threshold': 25,       # Conservative oversold (vs. 30 standard)
            'overbought_threshold': 75,     # Conservative overbought (vs. 70 standard)
            'extreme_oversold': 15,         # High-confidence oversold
            'extreme_overbought': 85,       # High-confidence overbought
            'use_divergence': True,         # Enable advanced divergence detection
            'divergence_lookback': 10       # 10-period divergence analysis window
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with RSI focus
        super().__init__("RSI Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced RSI mean reversion signals with divergence analysis.
        
        Implements comprehensive RSI analysis including multi-level thresholds,
        signal strength gradation, and optional divergence detection. This method
        provides sophisticated momentum-based mean reversion signals with context.
        
        Implementation Process:
        1. Validate input data for RSI calculation
        2. Calculate RSI momentum oscillator
        3. Define multiple RSI threshold levels
        4. Generate graduated signals based on RSI extremes
        5. Detect bullish/bearish divergences if enabled
        6. Apply divergence-enhanced signals
        7. Add comprehensive RSI analysis columns
        8. Log RSI distribution and signal statistics
        
        Args:
            data: DataFrame containing market data with 'close' column required
            
        Returns:
            DataFrame with original data plus RSI analysis:
                - 'signal': Primary trading signal (1=buy, -1=sell, 0=hold)
                - 'signal_strength': Signal confidence level (1-3 scale)
                - 'rsi': RSI momentum oscillator values (0-100 scale)
                - 'rsi_oversold': Boolean standard oversold indicator
                - 'rsi_overbought': Boolean standard overbought indicator
                - 'rsi_extreme_oversold': Boolean extreme oversold indicator
                - 'rsi_extreme_overbought': Boolean extreme overbought indicator
                - 'bullish_divergence': Boolean bullish divergence indicator
                - 'bearish_divergence': Boolean bearish divergence indicator
                
        Signal Strength Levels:
            - Level 1: Standard RSI oversold/overbought (signal = ±1)
            - Level 2: Extreme RSI levels (signal_strength = 2)
            - Level 3: Divergence signals (signal_strength = 3)
            
        Divergence Detection:
        The strategy identifies two types of divergences:
        - Bullish: Price makes lower low, RSI makes higher low (buy signal)
        - Bearish: Price makes higher high, RSI makes lower high (sell signal)
        
        Example:
            >>> signals = strategy.generate_signals(stock_data)
            >>> divergence_signals = signals[
            ...     (signals['bullish_divergence']) | (signals['bearish_divergence'])
            ... ]
            >>> high_confidence = signals[signals['signal_strength'] >= 2]
            >>> print(f"Found {len(divergence_signals)} divergence signals")
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
        
        # Step 3: Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0  # Track signal confidence level
        
        # Step 4: Define RSI threshold conditions
        oversold_threshold = self.parameters['oversold_threshold']
        overbought_threshold = self.parameters['overbought_threshold']
        extreme_oversold = self.parameters['extreme_oversold']
        extreme_overbought = self.parameters['extreme_overbought']
        
        # Standard RSI mean reversion conditions
        oversold = df['rsi'] < oversold_threshold
        overbought = df['rsi'] > overbought_threshold
        
        # Extreme RSI conditions for higher confidence signals
        extreme_oversold_condition = df['rsi'] < extreme_oversold
        extreme_overbought_condition = df['rsi'] > extreme_overbought
        
        # Step 5: Generate basic RSI signals with strength gradation
        
        # Level 1: Standard RSI signals
        df.loc[oversold, 'signal'] = 1              # Buy on oversold
        df.loc[overbought, 'signal'] = -1           # Sell on overbought
        df.loc[oversold | overbought, 'signal_strength'] = 1
        
        # Level 2: Extreme RSI signals (higher confidence)
        df.loc[extreme_oversold_condition, 'signal'] = 1     # Strong buy
        df.loc[extreme_overbought_condition, 'signal'] = -1  # Strong sell
        df.loc[extreme_oversold_condition | extreme_overbought_condition, 'signal_strength'] = 2
        
        # Step 6: RSI Divergence detection if enabled
        if self.parameters['use_divergence']:
            lookback = self.parameters['divergence_lookback']
            
            # Calculate rolling highs and lows for divergence analysis
            price_low = df['close'].rolling(window=lookback).min()
            rsi_low = df['rsi'].rolling(window=lookback).min()
            price_high = df['close'].rolling(window=lookback).max()
            rsi_high = df['rsi'].rolling(window=lookback).max()
            
            # Bullish divergence: Price makes lower low, RSI makes higher low
            # Indicates underlying strength despite price weakness
            price_lower_low = df['close'] < price_low.shift(1)
            rsi_higher_low = df['rsi'] > rsi_low.shift(1)
            bullish_divergence = price_lower_low & rsi_higher_low & oversold
            
            # Bearish divergence: Price makes higher high, RSI makes lower high
            # Indicates underlying weakness despite price strength
            price_higher_high = df['close'] > price_high.shift(1)
            rsi_lower_high = df['rsi'] < rsi_high.shift(1)
            bearish_divergence = price_higher_high & rsi_lower_high & overbought
            
            # Step 7: Apply divergence signals (highest confidence level)
            df.loc[bullish_divergence, 'signal'] = 1    # Divergence buy
            df.loc[bearish_divergence, 'signal'] = -1   # Divergence sell
            df.loc[bullish_divergence | bearish_divergence, 'signal_strength'] = 3
            
            # Add divergence analysis columns
            df['bullish_divergence'] = bullish_divergence
            df['bearish_divergence'] = bearish_divergence
            
            self.logger.debug("Applied RSI divergence analysis")
        
        # Step 8: Add comprehensive RSI analysis columns
        df['rsi_oversold'] = oversold
        df['rsi_overbought'] = overbought
        df['rsi_extreme_oversold'] = extreme_oversold_condition
        df['rsi_extreme_overbought'] = extreme_overbought_condition
        
        # Step 9: Log detailed RSI statistics
        standard_buy = oversold.sum()
        standard_sell = overbought.sum()
        extreme_buy = extreme_oversold_condition.sum()
        extreme_sell = extreme_overbought_condition.sum()
        
        # RSI distribution analysis
        avg_rsi = df['rsi'].mean()
        rsi_volatility = df['rsi'].std()
        
        if self.parameters['use_divergence']:
            divergence_buy = df['bullish_divergence'].sum()
            divergence_sell = df['bearish_divergence'].sum()
            
            self.logger.info(f"RSI analysis: {standard_buy + standard_sell} standard signals, "
                            f"{extreme_buy + extreme_sell} extreme signals, "
                            f"{divergence_buy + divergence_sell} divergence signals")
        else:
            self.logger.info(f"RSI analysis: {standard_buy + standard_sell} standard signals, "
                            f"{extreme_buy + extreme_sell} extreme signals")
        
        self.logger.debug(f"RSI statistics: avg={avg_rsi:.1f}, volatility={rsi_volatility:.1f}, "
                         f"period={rsi_period}")
        
        return df 