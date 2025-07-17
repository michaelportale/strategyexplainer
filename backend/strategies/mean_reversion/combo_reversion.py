"""Multi-indicator mean reversion strategy combining Bollinger Bands, Z-Score, and RSI.

This module represents the ultimate mean reversion approach by combining the
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

Classes:
    MeanReversionComboStrategy: Multi-indicator consensus approach

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

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


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
    
    # Strategy metadata for auto-registration
    strategy_category = 'mean_reversion'
    
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