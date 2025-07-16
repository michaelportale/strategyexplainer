"""Mean reversion strategies for contrarian trading and market normalization capture.

This module implements sophisticated mean reversion strategies based on the principle
that prices tend to revert to their statistical mean over time. These strategies
excel in range-bound markets and during periods of market overreaction.

Strategy Philosophy:
"Buy blood, sell euphoria" - Mean reversion strategies profit from temporary price
dislocations by betting on the return to normalcy. They work on the assumption that
extreme price movements are often temporary and will be corrected by market forces.

Strategy Categories:
1. BollingerBandMeanReversionStrategy: Volatility-based mean reversion using Bollinger Bands
2. ZScoreMeanReversionStrategy: Statistical mean reversion using Z-score analysis  
3. RSIMeanReversionStrategy: Momentum-based mean reversion with divergence detection
4. MeanReversionComboStrategy: Multi-indicator consensus approach

Key Concepts:
- Statistical Mean: Long-term average price level that acts as gravitational center
- Price Extremes: Temporary deviations from mean due to overreaction or sentiment
- Support/Resistance: Price levels where mean reversion forces typically emerge
- Volatility Bands: Dynamic boundaries that expand/contract with market volatility

Design Principles:
- Contrarian approach: Buy oversold, sell overbought conditions
- Statistical validation: Use multiple timeframes and indicators
- Risk management: Tight stops and position sizing for mean reversion trades
- Market regime awareness: Adapt to trending vs. ranging market conditions

Best Market Conditions:
- Range-bound markets with clear support/resistance
- Post-earnings or news event overreactions
- High-volatility periods with rapid price swings
- Liquid markets with efficient price discovery

Classes:
    BollingerBandMeanReversionStrategy: Volatility band-based reversion
    ZScoreMeanReversionStrategy: Pure statistical mean reversion
    RSIMeanReversionStrategy: Momentum oscillator with divergence signals
    MeanReversionComboStrategy: Multi-indicator consensus approach

Example:
    >>> # Create conservative Bollinger Band mean reversion strategy
    >>> strategy = BollingerBandMeanReversionStrategy({
    ...     'bb_period': 20,
    ...     'bb_std_dev': 2.5,      # Wider bands for less noise
    ...     'use_rsi_filter': True,  # Additional momentum confirmation
    ...     'exit_at_middle': True   # Take profits at center line
    ... })
    >>> signals = strategy.generate_signals(price_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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
        
        # Step 2: Calculate Bollinger Bands components
        period = self.parameters['bb_period']
        std_dev_multiplier = self.parameters['bb_std_dev']
        
        # Middle band: Simple moving average (center line)
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Rolling standard deviation for volatility measurement
        df['bb_std'] = df['close'].rolling(window=period).std()
        
        # Upper band: Middle + (K × standard deviation)
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev_multiplier)
        
        # Lower band: Middle - (K × standard deviation)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev_multiplier)
        
        # Step 3: Calculate band position for signal strength analysis
        # Position ranges from 0 (at lower band) to 1 (at upper band)
        band_width = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / band_width
        
        # Handle division by zero for very narrow bands
        df['bb_position'] = df['bb_position'].fillna(0.5)
        
        # Step 4: Calculate RSI momentum filter if enabled
        if self.parameters['use_rsi_filter']:
            df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
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
                         f"avg_width={avg_band_width:.2f}, std_dev={std_dev_multiplier}")
        
        return df


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score based mean reversion strategy for statistical price normalization trading.
    
    This strategy applies pure statistical analysis to identify mean reversion opportunities
    by calculating how many standard deviations the current price deviates from its
    historical mean. Z-scores provide a standardized measure of price extremes that
    is independent of the absolute price level.
    
    Statistical Foundation:
    - Z-Score = (Current Price - Mean Price) / Standard Deviation
    - Measures price deviation in standard deviation units
    - Normal distribution assumes 95% of values within ±2 standard deviations
    - Extreme Z-scores indicate temporary price dislocations
    
    Strategy Logic:
    - Prices with extreme Z-scores tend to revert to mean
    - High positive Z-scores indicate overbought conditions
    - High negative Z-scores indicate oversold conditions
    - Exit signals generated as Z-score approaches zero
    
    Signal Rules:
    - BUY: Z-score < -threshold (statistically oversold)
    - SELL: Z-score > +threshold (statistically overbought)
    - EXIT: Z-score approaches zero (mean reversion in progress)
    - FILTER: Optional volume confirmation for signal quality
    
    Key Features:
    - Pure statistical approach without subjective indicators
    - Automatically adapts to changing price levels
    - Configurable lookback periods for different timeframes
    - Optional volume filtering for enhanced signal quality
    
    Best Applications:
    - Range-bound markets with stable volatility
    - Statistical arbitrage and pairs trading
    - Post-event mean reversion capture
    - Quantitative trading systems
    
    Limitations:
    - Assumes normal distribution of returns
    - Poor performance in trending markets
    - Sensitive to outliers in historical data
    - Requires sufficient historical data for stability
    
    Example:
        >>> # Conservative Z-score setup for weekly mean reversion
        >>> strategy = ZScoreMeanReversionStrategy({
        ...     'lookback_period': 100,     # Longer history for stability
        ...     'z_threshold': 2.5,         # Conservative threshold
        ...     'exit_z_threshold': 0.3,    # Early exit for profit protection
        ...     'use_volume_filter': True   # Volume confirmation
        ... })
        >>> signals = strategy.generate_signals(weekly_data)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Z-Score mean reversion strategy with statistical parameters.
        
        Configures the strategy with Z-score calculation parameters and optional
        volume filtering. Default parameters provide a balanced approach suitable
        for most mean reversion applications.
        
        Args:
            parameters: Strategy configuration dictionary with options:
                - lookback_period (int): Period for mean/std calculation (default: 50)
                    Shorter periods = more responsive to recent changes
                    Longer periods = more stable statistical base
                    Recommended: 30-60 for daily, 100-200 for weekly
                - z_threshold (float): Z-score threshold for signals (default: 2.0)
                    Higher values = more extreme conditions required (fewer signals)
                    Lower values = earlier signals but potentially more noise
                    Statistical guide: 2.0 (95%), 2.5 (99%), 3.0 (99.7%)
                - exit_z_threshold (float): Z-score threshold for exits (default: 0.5)
                    Controls profit-taking timing as price reverts to mean
                    Lower values = earlier exits, higher values = let profits run
                - use_volume_filter (bool): Enable volume confirmation (default: False)
                    True = require unusual volume for signal confirmation
                    False = pure price-based Z-score signals
                - volume_period (int): Volume average period (default: 20)
                    Period for establishing normal volume baseline
                - volume_threshold (float): Volume multiplier threshold (default: 1.2)
                    Minimum volume relative to average for confirmation
                    
        Statistical Considerations:
        - Lookback period should be long enough for stable statistics
        - Z-threshold selection affects signal frequency vs. quality tradeoff
        - Volume filter can improve signal quality in liquid markets
        
        Example:
            >>> # Aggressive short-term Z-score strategy
            >>> params = {
            ...     'lookback_period': 30,      # Short-term statistical base
            ...     'z_threshold': 1.5,         # Sensitive threshold
            ...     'exit_z_threshold': 0.2,    # Quick profit taking
            ...     'use_volume_filter': False  # No volume requirement
            ... }
            >>> strategy = ZScoreMeanReversionStrategy(params)
        """
        # Statistically-grounded default parameters
        default_params = {
            'lookback_period': 50,          # 50-period statistical window
            'z_threshold': 2.0,             # 2 standard deviation threshold (95% confidence)
            'exit_z_threshold': 0.5,        # Exit when halfway back to mean
            'use_volume_filter': False,     # Pure statistical approach by default
            'volume_period': 20,            # 20-period volume baseline
            'volume_threshold': 1.2         # 20% above average volume requirement
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with Z-score focus
        super().__init__("Z-Score Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Z-Score based mean reversion signals with statistical analysis.
        
        Implements comprehensive Z-score analysis including statistical calculation,
        threshold comparison, optional volume filtering, and exit signal generation.
        This method provides pure statistical mean reversion signals.
        
        Implementation Process:
        1. Validate input data for price information
        2. Calculate rolling mean and standard deviation
        3. Compute Z-scores for statistical analysis
        4. Apply volume filter if enabled
        5. Generate entry signals at extreme Z-scores
        6. Generate exit signals as Z-scores normalize
        7. Add statistical analysis columns
        8. Log Z-score distribution statistics
        
        Args:
            data: DataFrame containing market data with 'close' column required,
                 'volume' column required if volume filter enabled
            
        Returns:
            DataFrame with original data plus Z-score analysis:
                - 'signal': Primary trading signal (1=buy, -1=sell, 0=hold)
                - 'price_mean': Rolling mean price (statistical center)
                - 'price_std': Rolling standard deviation (volatility measure)
                - 'z_score': Current Z-score (standardized price deviation)
                - 'volume_avg': Average volume (if volume filter enabled)
                - 'oversold': Boolean oversold condition indicator
                - 'overbought': Boolean overbought condition indicator
                - 'exit_long': Exit signal for long positions
                - 'exit_short': Exit signal for short positions
                
        Z-Score Interpretation:
            - Z > +2.0: Price is 2+ standard deviations above mean (overbought)
            - Z < -2.0: Price is 2+ standard deviations below mean (oversold)
            - Z ≈ 0: Price is near statistical mean (fair value)
            
        Statistical Quality Checks:
        The strategy monitors Z-score distribution to ensure statistical validity:
        - Extreme Z-scores should be rare (consistent with normal distribution)
        - Rolling statistics should be stable and meaningful
        
        Example:
            >>> signals = strategy.generate_signals(price_data)
            >>> extreme_scores = signals[abs(signals['z_score']) > 2.5]
            >>> mean_reversion_rate = len(extreme_scores) / len(signals)
            >>> print(f"Extreme Z-score rate: {mean_reversion_rate:.1%}")
        """
        # Step 1: Validate input data for Z-score calculation
        if not self.validate_data(data):
            self.logger.error("Data validation failed for Z-Score strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Calculate rolling statistical measures
        period = self.parameters['lookback_period']
        
        # Rolling mean: statistical center for Z-score calculation
        df['price_mean'] = df['close'].rolling(window=period).mean()
        
        # Rolling standard deviation: volatility measure for normalization
        df['price_std'] = df['close'].rolling(window=period).std()
        
        # Step 3: Calculate Z-scores (standardized price deviations)
        df['z_score'] = (df['close'] - df['price_mean']) / df['price_std']
        
        # Handle division by zero for periods with no volatility
        df['z_score'] = df['z_score'].fillna(0)
        
        # Step 4: Apply volume filter if enabled
        if self.parameters['use_volume_filter']:
            # Calculate volume baseline and threshold
            volume_period = self.parameters['volume_period']
            volume_threshold = self.parameters['volume_threshold']
            
            df['volume_avg'] = df['volume'].rolling(window=volume_period).mean()
            volume_filter = df['volume'] > (df['volume_avg'] * volume_threshold)
            
            self.logger.debug("Applied volume filter to Z-score signals")
        else:
            # No volume filter: all signals pass through
            volume_filter = True
        
        # Step 5: Initialize signal column with hold positions
        df['signal'] = 0
        
        # Step 6: Define Z-score threshold conditions
        z_threshold = self.parameters['z_threshold']
        
        # BUY: Extremely oversold (large negative Z-score)
        buy_condition = (df['z_score'] < -z_threshold) & volume_filter
        
        # SELL: Extremely overbought (large positive Z-score)
        sell_condition = (df['z_score'] > z_threshold) & volume_filter
        
        # Step 7: Generate primary mean reversion signals
        df.loc[buy_condition, 'signal'] = 1   # Buy on statistical oversold
        df.loc[sell_condition, 'signal'] = -1 # Sell on statistical overbought
        
        # Step 8: Generate exit signals as Z-scores normalize
        exit_threshold = self.parameters['exit_z_threshold']
        
        # Exit long positions when Z-score recovers toward mean
        df['exit_long'] = (df['z_score'] > -exit_threshold) & \
                         (df['z_score'].shift(1) <= -exit_threshold)
        
        # Exit short positions when Z-score recovers toward mean
        df['exit_short'] = (df['z_score'] < exit_threshold) & \
                          (df['z_score'].shift(1) >= exit_threshold)
        
        # Step 9: Add comprehensive analysis columns
        df['oversold'] = buy_condition       # Statistical oversold indicator
        df['overbought'] = sell_condition    # Statistical overbought indicator
        
        # Step 10: Log detailed Z-score statistics
        buy_signals = buy_condition.sum()
        sell_signals = sell_condition.sum()
        total_signals = buy_signals + sell_signals
        
        # Z-score distribution analysis
        mean_z_score = df['z_score'].mean()
        std_z_score = df['z_score'].std()
        max_z_score = df['z_score'].max()
        min_z_score = df['z_score'].min()
        
        # Extreme Z-score frequency (quality check)
        extreme_positive = (df['z_score'] > z_threshold).sum()
        extreme_negative = (df['z_score'] < -z_threshold).sum()
        extreme_rate = (extreme_positive + extreme_negative) / len(df) * 100
        
        self.logger.info(f"Z-Score analysis: {total_signals} total signals "
                        f"({buy_signals} oversold, {sell_signals} overbought)")
        self.logger.info(f"Z-Score distribution: mean={mean_z_score:.3f}, "
                        f"std={std_z_score:.3f}, range=[{min_z_score:.2f}, {max_z_score:.2f}]")
        self.logger.debug(f"Extreme Z-score rate: {extreme_rate:.1f}% "
                         f"(threshold=±{z_threshold})")
        
        return df


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


class MeanReversionComboStrategy(BaseStrategy):
    """Multi-indicator mean reversion strategy combining Bollinger Bands, Z-Score, and RSI.
    
    This strategy represents the ultimate mean reversion approach by combining the
    strengths of three complementary indicators: Bollinger Bands (volatility-based),
    Z-Score (statistical), and RSI (momentum-based). The combination approach
    significantly improves signal quality by requiring consensus across different
    analytical dimensions.
    
    Strategy Philosophy:
    - Multiple indicators reduce false signals through consensus
    - Different indicator types capture various aspects of mean reversion
    - Weighted or voting systems provide signal strength gradation
    - Robust performance across different market conditions
    
    Component Indicators:
    1. Bollinger Bands: Volatility-adjusted extremes (adaptive to market conditions)
    2. Z-Score: Pure statistical mean reversion (objective price deviation)
    3. RSI: Momentum exhaustion (behavioral finance aspect)
    
    Combination Methods:
    - Consensus: Require 2+ indicators to agree (high quality, fewer signals)
    - Weighted: Combine indicator strengths with configurable weights
    - Adaptive: Adjust weights based on recent performance or market conditions
    
    Signal Logic:
    - BUY: Multiple indicators confirm oversold condition
    - SELL: Multiple indicators confirm overbought condition
    - STRENGTH: Number of agreeing indicators determines signal confidence
    
    Key Features:
    - Configurable consensus requirements
    - Individual indicator weight adjustment
    - Comprehensive analysis of all component signals
    - Robust performance across market regimes
    
    Advantages:
    - Higher signal quality through multi-factor confirmation
    - Reduced false signals compared to single indicators
    - Adaptable to different market conditions
    - Provides insight into signal source composition
    
    Limitations:
    - Fewer total signals due to consensus requirements
    - More complex parameter optimization
    - Potential for over-fitting with too many parameters
    - Slower signal generation due to multiple calculations
    
    Example:
        >>> # Conservative consensus-based combo strategy
        >>> strategy = MeanReversionComboStrategy({
        ...     'bb_period': 20,           # Standard Bollinger parameters
        ...     'bb_std_dev': 2.5,         # Wider bands for quality
        ...     'z_lookback': 60,          # Longer Z-score history
        ...     'z_threshold': 2.2,        # Conservative Z-score threshold
        ...     'rsi_period': 14,          # Standard RSI
        ...     'rsi_oversold': 25,        # Conservative RSI levels
        ...     'rsi_overbought': 75,
        ...     'require_consensus': True   # Require 2+ indicators
        ... })
        >>> signals = strategy.generate_signals(daily_data)
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize multi-indicator mean reversion combination strategy.
        
        Configures all component indicators and combination logic to create
        a robust mean reversion system. Default parameters balance signal
        quality with reasonable frequency.
        
        Args:
            parameters: Strategy configuration dictionary combining all indicators:
                
                Bollinger Band Parameters:
                - bb_period (int): Bollinger Band period (default: 20)
                - bb_std_dev (float): Standard deviation multiplier (default: 2.0)
                
                Z-Score Parameters:
                - z_lookback (int): Z-score calculation period (default: 50)
                - z_threshold (float): Z-score signal threshold (default: 2.0)
                
                RSI Parameters:
                - rsi_period (int): RSI calculation period (default: 14)
                - rsi_oversold (float): RSI oversold threshold (default: 25)
                - rsi_overbought (float): RSI overbought threshold (default: 75)
                
                Combination Logic:
                - require_consensus (bool): Require 2+ indicators to agree (default: True)
                    True = high-quality signals with multi-indicator confirmation
                    False = weighted combination approach for more signals
                - weight_bb (float): Bollinger Band weight in combination (default: 1.0)
                - weight_zscore (float): Z-Score weight in combination (default: 1.0)
                - weight_rsi (float): RSI weight in combination (default: 1.0)
                
        Combination Approaches:
        1. Consensus Mode (require_consensus=True):
           - Requires at least 2 out of 3 indicators to agree
           - High signal quality but lower frequency
           - Recommended for conservative mean reversion trading
           
        2. Weighted Mode (require_consensus=False):
           - Combines indicator signals using specified weights
           - Requires 60% of total weighted score for signal generation
           - More flexible but potentially more false signals
           
        Example:
            >>> # Weighted combination favoring RSI and Z-score
            >>> params = {
            ...     'require_consensus': False,  # Use weighted approach
            ...     'weight_bb': 0.5,           # Lower Bollinger weight
            ...     'weight_zscore': 1.5,       # Higher Z-score weight
            ...     'weight_rsi': 1.5,          # Higher RSI weight
            ...     'z_threshold': 1.8,         # More sensitive Z-score
            ...     'rsi_oversold': 30          # Standard RSI levels
            ... }
            >>> strategy = MeanReversionComboStrategy(params)
        """
        # Balanced default parameters for multi-indicator combination
        default_params = {
            # Bollinger Band configuration
            'bb_period': 20,                # Standard 20-period bands
            'bb_std_dev': 2.0,              # Standard 2-sigma bands
            
            # Z-Score configuration
            'z_lookback': 50,               # 50-period statistical window
            'z_threshold': 2.0,             # 2-sigma Z-score threshold
            
            # RSI configuration
            'rsi_period': 14,               # Standard RSI period
            'rsi_oversold': 25,             # Conservative oversold level
            'rsi_overbought': 75,           # Conservative overbought level
            
            # Combination logic configuration
            'require_consensus': True,      # Require multi-indicator agreement
            'weight_bb': 1.0,               # Equal weight to Bollinger Bands
            'weight_zscore': 1.0,           # Equal weight to Z-Score
            'weight_rsi': 1.0               # Equal weight to RSI
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with combination focus
        super().__init__("Mean Reversion Combo", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-indicator consensus mean reversion signals.
        
        Implements comprehensive multi-indicator analysis by calculating all
        component indicators and combining them using the specified logic.
        This method provides the highest quality mean reversion signals through
        cross-validation across different analytical approaches.
        
        Implementation Process:
        1. Validate input data for all indicator calculations
        2. Calculate Bollinger Bands with configured parameters
        3. Calculate Z-Score statistical analysis
        4. Calculate RSI momentum analysis
        5. Generate individual indicator signals
        6. Combine signals using consensus or weighted approach
        7. Add comprehensive multi-indicator analysis
        8. Log combination statistics and signal sources
        
        Args:
            data: DataFrame containing market data with 'close' column required
            
        Returns:
            DataFrame with original data plus comprehensive analysis:
                - 'signal': Combined trading signal (1=buy, -1=sell, 0=hold)
                - Bollinger Band columns: bb_middle, bb_upper, bb_lower
                - Z-Score columns: price_mean, price_std, z_score
                - RSI column: rsi
                - Individual signals: bb_signal, z_signal, rsi_signal
                
        Combination Logic Details:
        
        Consensus Mode:
        - Counts agreeing indicators for buy/sell decisions
        - Requires majority (2+ out of 3) for signal generation
        - Provides highest signal quality with lowest false positive rate
        
        Weighted Mode:
        - Calculates weighted score for buy/sell conditions
        - Requires 60% of maximum possible score for signal generation
        - Allows fine-tuning of indicator importance
        
        Signal Quality Analysis:
        The strategy tracks which indicators contribute to each signal,
        enabling analysis of signal source composition and quality.
        
        Example:
            >>> signals = strategy.generate_signals(market_data)
            >>> consensus_signals = signals[signals['signal'] != 0]
            >>> bb_contribution = consensus_signals['bb_signal'].abs().mean()
            >>> print(f"Bollinger Band contribution: {bb_contribution:.1%}")
        """
        # Step 1: Validate input data for all indicators
        if not self.validate_data(data):
            self.logger.error("Data validation failed for combo strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Calculate Bollinger Bands component
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_series = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_series * bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std_series * bb_std_dev)
        
        # Step 3: Calculate Z-Score component
        z_period = self.parameters['z_lookback']
        z_threshold = self.parameters['z_threshold']
        
        df['price_mean'] = df['close'].rolling(window=z_period).mean()
        df['price_std'] = df['close'].rolling(window=z_period).std()
        df['z_score'] = (df['close'] - df['price_mean']) / df['price_std']
        
        # Step 4: Calculate RSI component
        rsi_period = self.parameters['rsi_period']
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Step 5: Generate individual indicator signals
        
        # Bollinger Band signals
        bb_buy = df['close'] <= df['bb_lower']
        bb_sell = df['close'] >= df['bb_upper']
        
        # Z-Score signals
        z_buy = df['z_score'] < -z_threshold
        z_sell = df['z_score'] > z_threshold
        
        # RSI signals
        rsi_oversold = self.parameters['rsi_oversold']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_buy = df['rsi'] < rsi_oversold
        rsi_sell = df['rsi'] > rsi_overbought
        
        # Step 6: Combine signals using specified logic
        if self.parameters['require_consensus']:
            # Consensus approach: Require majority agreement (2+ indicators)
            
            # Count agreeing indicators for each direction
            buy_votes = bb_buy.astype(int) + z_buy.astype(int) + rsi_buy.astype(int)
            sell_votes = bb_sell.astype(int) + z_sell.astype(int) + rsi_sell.astype(int)
            
            # Generate signals only when majority agrees
            buy_signal = buy_votes >= 2   # At least 2 out of 3 indicators
            sell_signal = sell_votes >= 2
            
            self.logger.debug("Using consensus combination approach")
            
        else:
            # Weighted approach: Combine using indicator weights
            
            # Extract weights for each indicator
            w_bb = self.parameters['weight_bb']
            w_z = self.parameters['weight_zscore']
            w_rsi = self.parameters['weight_rsi']
            
            # Calculate weighted scores for each direction
            buy_score = (bb_buy * w_bb + z_buy * w_z + rsi_buy * w_rsi)
            sell_score = (bb_sell * w_bb + z_sell * w_z + rsi_sell * w_rsi)
            
            # Require 60% of maximum possible weighted score
            total_weight = w_bb + w_z + w_rsi
            threshold = total_weight * 0.6
            
            buy_signal = buy_score >= threshold
            sell_signal = sell_score >= threshold
            
            self.logger.debug(f"Using weighted combination: BB={w_bb}, Z={w_z}, RSI={w_rsi}")
        
        # Step 7: Apply combined signals
        df['signal'] = 0
        df.loc[buy_signal, 'signal'] = 1   # Combined buy signal
        df.loc[sell_signal, 'signal'] = -1 # Combined sell signal
        
        # Step 8: Add individual signal analysis columns
        df['bb_signal'] = 0
        df.loc[bb_buy, 'bb_signal'] = 1
        df.loc[bb_sell, 'bb_signal'] = -1
        
        df['z_signal'] = 0
        df.loc[z_buy, 'z_signal'] = 1
        df.loc[z_sell, 'z_signal'] = -1
        
        df['rsi_signal'] = 0
        df.loc[rsi_buy, 'rsi_signal'] = 1
        df.loc[rsi_sell, 'rsi_signal'] = -1
        
        # Step 9: Log comprehensive combination statistics
        combined_buy = buy_signal.sum()
        combined_sell = sell_signal.sum()
        total_combined = combined_buy + combined_sell
        
        # Individual indicator signal counts
        bb_signals = (bb_buy | bb_sell).sum()
        z_signals = (z_buy | z_sell).sum()
        rsi_signals = (rsi_buy | rsi_sell).sum()
        
        # Signal source analysis
        if total_combined > 0:
            bb_contribution = df.loc[df['signal'] != 0, 'bb_signal'].abs().mean()
            z_contribution = df.loc[df['signal'] != 0, 'z_signal'].abs().mean()
            rsi_contribution = df.loc[df['signal'] != 0, 'rsi_signal'].abs().mean()
        else:
            bb_contribution = z_contribution = rsi_contribution = 0
        
        self.logger.info(f"Combo strategy: {total_combined} consensus signals "
                        f"({combined_buy} buy, {combined_sell} sell)")
        self.logger.info(f"Individual signals: BB={bb_signals}, Z={z_signals}, RSI={rsi_signals}")
        self.logger.debug(f"Signal contributions: BB={bb_contribution:.1%}, "
                         f"Z={z_contribution:.1%}, RSI={rsi_contribution:.1%}")
        
        return df 