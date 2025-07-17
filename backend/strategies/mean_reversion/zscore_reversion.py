"""Z-Score based mean reversion strategy for statistical price normalization trading.

This module applies pure statistical analysis to identify mean reversion opportunities
by calculating how many standard deviations the current price deviates from its
historical mean. Z-scores provide a standardized measure of price extremes that
is independent of the absolute price level.

Statistical Foundation:
- Z-Score = (Current Price - Mean Price) / Standard Deviation
- Measures price deviation in standard deviation units
- Normal distribution assumes 95% of values within ±2 standard deviations
- Extreme Z-scores indicate temporary price dislocations

Classes:
    ZScoreMeanReversionStrategy: Pure statistical mean reversion

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

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


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
    
    # Strategy metadata for auto-registration
    strategy_category = 'mean_reversion'
    
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
        
        # Step 2: Add technical indicators including Z-score
        custom_periods = {'z_score_period': self.parameters['lookback_period']}
        df = self.add_technical_indicators(df, custom_periods)
        
        # Step 3: Apply volume filter if enabled (volume indicators already calculated)
        if self.parameters['use_volume_filter']:
            volume_threshold = self.parameters['volume_threshold']
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