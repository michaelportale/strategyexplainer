"""Mean Reversion Strategies.

Phase 4: "Buy blood, sell euphoria" - mean reversion strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class BollingerBandMeanReversionStrategy(BaseStrategy):
    """Mean reversion using Bollinger Bands.
    
    Signal Logic:
    - Buy: Price touches or crosses below lower Bollinger Band
    - Sell: Price touches or crosses above upper Bollinger Band
    - Optional RSI filter to avoid counter-trend moves
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Bollinger Band mean reversion strategy.
        
        Args:
            parameters: Strategy parameters
                - bb_period: Bollinger Band period (default: 20)
                - bb_std_dev: Standard deviation multiplier (default: 2.0)
                - use_rsi_filter: Use RSI filter (default: True)
                - rsi_period: RSI period for filter (default: 14)
                - rsi_oversold: RSI oversold threshold (default: 30)
                - rsi_overbought: RSI overbought threshold (default: 70)
                - exit_at_middle: Exit at middle band (default: True)
        """
        default_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'use_rsi_filter': True,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'exit_at_middle': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Bollinger Band Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band mean reversion signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column (1=buy, -1=sell, 0=hold)
        """
        if not self.validate_data(data):
            self.logger.error("Data validation failed")
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate Bollinger Bands
        period = self.parameters['bb_period']
        std_dev = self.parameters['bb_std_dev']
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate Bollinger Band position (0 = lower band, 1 = upper band)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI filter if enabled
        if self.parameters['use_rsi_filter']:
            df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Initialize signals
        df['signal'] = 0
        
        # Buy conditions: Price at/below lower band
        buy_condition = df['close'] <= df['bb_lower']
        
        # Sell conditions: Price at/above upper band
        sell_condition = df['close'] >= df['bb_upper']
        
        # Apply RSI filter if enabled
        if self.parameters['use_rsi_filter']:
            # Only buy when RSI confirms oversold
            buy_condition = buy_condition & (df['rsi'] <= self.parameters['rsi_oversold'])
            # Only sell when RSI confirms overbought
            sell_condition = sell_condition & (df['rsi'] >= self.parameters['rsi_overbought'])
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Exit at middle band if enabled
        if self.parameters['exit_at_middle']:
            # Exit long positions when price crosses above middle
            middle_cross_up = (df['close'] > df['bb_middle']) & (df['close'].shift(1) <= df['bb_middle'])
            # Exit short positions when price crosses below middle
            middle_cross_down = (df['close'] < df['bb_middle']) & (df['close'].shift(1) >= df['bb_middle'])
            
            df['middle_exit_long'] = middle_cross_up
            df['middle_exit_short'] = middle_cross_down
        
        # Add analysis columns
        df['at_lower_band'] = buy_condition
        df['at_upper_band'] = sell_condition
        
        self.logger.info(f"BB Mean Reversion signals: {buy_condition.sum()} lower band touches, "
                        f"{sell_condition.sum()} upper band touches")
        
        return df


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score based mean reversion strategy.
    
    Signal Logic:
    - Buy: Price Z-score < -threshold (oversold)
    - Sell: Price Z-score > +threshold (overbought)
    - Exit: Z-score approaches zero
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Z-Score mean reversion strategy.
        
        Args:
            parameters: Strategy parameters
                - lookback_period: Period for Z-score calculation (default: 50)
                - z_threshold: Z-score threshold for signals (default: 2.0)
                - exit_z_threshold: Z-score for exits (default: 0.5)
                - use_volume_filter: Filter by volume (default: False)
                - volume_period: Volume average period (default: 20)
                - volume_threshold: Volume multiplier (default: 1.2)
        """
        default_params = {
            'lookback_period': 50,
            'z_threshold': 2.0,
            'exit_z_threshold': 0.5,
            'use_volume_filter': False,
            'volume_period': 20,
            'volume_threshold': 1.2
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Z-Score Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Z-Score mean reversion signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column (1=buy, -1=sell, 0=hold)
        """
        if not self.validate_data(data):
            self.logger.error("Data validation failed")
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate Z-Score
        period = self.parameters['lookback_period']
        df['price_mean'] = df['close'].rolling(window=period).mean()
        df['price_std'] = df['close'].rolling(window=period).std()
        df['z_score'] = (df['close'] - df['price_mean']) / df['price_std']
        
        # Volume filter if enabled
        if self.parameters['use_volume_filter']:
            volume_period = self.parameters['volume_period']
            df['volume_avg'] = df['volume'].rolling(window=volume_period).mean()
            volume_filter = df['volume'] > (df['volume_avg'] * self.parameters['volume_threshold'])
        else:
            volume_filter = True
        
        # Initialize signals
        df['signal'] = 0
        
        # Buy: Extremely oversold (negative Z-score)
        z_threshold = self.parameters['z_threshold']
        buy_condition = (df['z_score'] < -z_threshold) & volume_filter
        
        # Sell: Extremely overbought (positive Z-score)
        sell_condition = (df['z_score'] > z_threshold) & volume_filter
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Exit signals when Z-score normalizes
        exit_threshold = self.parameters['exit_z_threshold']
        df['exit_long'] = (df['z_score'] > -exit_threshold) & (df['z_score'].shift(1) <= -exit_threshold)
        df['exit_short'] = (df['z_score'] < exit_threshold) & (df['z_score'].shift(1) >= exit_threshold)
        
        # Add analysis columns
        df['oversold'] = buy_condition
        df['overbought'] = sell_condition
        
        self.logger.info(f"Z-Score signals: {buy_condition.sum()} oversold, "
                        f"{sell_condition.sum()} overbought")
        
        return df


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based mean reversion strategy.
    
    Signal Logic:
    - Buy: RSI < oversold threshold
    - Sell: RSI > overbought threshold
    - Enhanced with RSI divergence detection
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI mean reversion strategy.
        
        Args:
            parameters: Strategy parameters
                - rsi_period: RSI calculation period (default: 14)
                - oversold_threshold: RSI oversold level (default: 25)
                - overbought_threshold: RSI overbought level (default: 75)
                - extreme_oversold: Extreme oversold level (default: 15)
                - extreme_overbought: Extreme overbought level (default: 85)
                - use_divergence: Use RSI divergence (default: True)
                - divergence_lookback: Periods to look for divergence (default: 10)
        """
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 25,
            'overbought_threshold': 75,
            'extreme_oversold': 15,
            'extreme_overbought': 85,
            'use_divergence': True,
            'divergence_lookback': 10
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("RSI Mean Reversion", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI mean reversion signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column (1=buy, -1=sell, 0=hold)  
        """
        if not self.validate_data(data):
            self.logger.error("Data validation failed")
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Initialize signals
        df['signal'] = 0
        
        # Basic mean reversion signals
        oversold = df['rsi'] < self.parameters['oversold_threshold']
        overbought = df['rsi'] > self.parameters['overbought_threshold']
        
        # Extreme levels for stronger signals
        extreme_oversold = df['rsi'] < self.parameters['extreme_oversold']
        extreme_overbought = df['rsi'] > self.parameters['extreme_overbought']
        
        # Apply signals
        df.loc[oversold, 'signal'] = 1
        df.loc[overbought, 'signal'] = -1
        
        # Stronger signals at extreme levels
        df.loc[extreme_oversold, 'signal'] = 2  # Strong buy
        df.loc[extreme_overbought, 'signal'] = -2  # Strong sell
        
        # RSI Divergence detection if enabled
        if self.parameters['use_divergence']:
            lookback = self.parameters['divergence_lookback']
            
            # Bullish divergence: Price makes lower low, RSI makes higher low
            price_low = df['close'].rolling(window=lookback).min()
            rsi_low = df['rsi'].rolling(window=lookback).min()
            
            price_lower_low = df['close'] < price_low.shift(1)
            rsi_higher_low = df['rsi'] > rsi_low.shift(1)
            
            bullish_divergence = price_lower_low & rsi_higher_low & oversold
            
            # Bearish divergence: Price makes higher high, RSI makes lower high
            price_high = df['close'].rolling(window=lookback).max()
            rsi_high = df['rsi'].rolling(window=lookback).max()
            
            price_higher_high = df['close'] > price_high.shift(1)
            rsi_lower_high = df['rsi'] < rsi_high.shift(1)
            
            bearish_divergence = price_higher_high & rsi_lower_high & overbought
            
            # Apply divergence signals
            df.loc[bullish_divergence, 'signal'] = 3  # Divergence buy
            df.loc[bearish_divergence, 'signal'] = -3  # Divergence sell
            
            # Add divergence columns for analysis
            df['bullish_divergence'] = bullish_divergence
            df['bearish_divergence'] = bearish_divergence
        
        # Add analysis columns
        df['rsi_oversold'] = oversold
        df['rsi_overbought'] = overbought
        df['rsi_extreme_oversold'] = extreme_oversold
        df['rsi_extreme_overbought'] = extreme_overbought
        
        # Normalize signals to standard -1, 0, 1 format for compatibility
        df['signal_strength'] = df['signal'].copy()  # Keep original signal strength
        df['signal'] = np.clip(df['signal'], -1, 1)  # Normalize for standard interface
        
        self.logger.info(f"RSI Mean Reversion signals: {oversold.sum()} oversold, "
                        f"{overbought.sum()} overbought, "
                        f"{extreme_oversold.sum()} extreme oversold, "
                        f"{extreme_overbought.sum()} extreme overbought")
        
        return df


class MeanReversionComboStrategy(BaseStrategy):
    """Combination mean reversion strategy using multiple indicators."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize combination mean reversion strategy.
        
        Args:
            parameters: Strategy parameters combining BB, Z-Score, and RSI
        """
        default_params = {
            # Bollinger Band params
            'bb_period': 20,
            'bb_std_dev': 2.0,
            # Z-Score params
            'z_lookback': 50,
            'z_threshold': 2.0,
            # RSI params
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            # Combination logic
            'require_consensus': True,  # Require 2+ indicators to agree
            'weight_bb': 1.0,
            'weight_zscore': 1.0,
            'weight_rsi': 1.0
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Mean Reversion Combo", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate combination mean reversion signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column (1=buy, -1=sell, 0=hold)
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate all indicators
        # Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std_dev']
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_series = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_series * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_series * bb_std)
        
        # Z-Score
        z_period = self.parameters['z_lookback']
        df['price_mean'] = df['close'].rolling(window=z_period).mean()
        df['price_std'] = df['close'].rolling(window=z_period).std()
        df['z_score'] = (df['close'] - df['price_mean']) / df['price_std']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Individual signals
        bb_buy = df['close'] <= df['bb_lower']
        bb_sell = df['close'] >= df['bb_upper']
        
        z_buy = df['z_score'] < -self.parameters['z_threshold']
        z_sell = df['z_score'] > self.parameters['z_threshold']
        
        rsi_buy = df['rsi'] < self.parameters['rsi_oversold']
        rsi_sell = df['rsi'] > self.parameters['rsi_overbought']
        
        # Combine signals
        if self.parameters['require_consensus']:
            # Require at least 2 indicators to agree
            buy_votes = bb_buy.astype(int) + z_buy.astype(int) + rsi_buy.astype(int)
            sell_votes = bb_sell.astype(int) + z_sell.astype(int) + rsi_sell.astype(int)
            
            buy_signal = buy_votes >= 2
            sell_signal = sell_votes >= 2
        else:
            # Weighted combination
            w_bb = self.parameters['weight_bb']
            w_z = self.parameters['weight_zscore']
            w_rsi = self.parameters['weight_rsi']
            
            buy_score = (bb_buy * w_bb + z_buy * w_z + rsi_buy * w_rsi)
            sell_score = (bb_sell * w_bb + z_sell * w_z + rsi_sell * w_rsi)
            
            total_weight = w_bb + w_z + w_rsi
            buy_signal = buy_score > (total_weight * 0.6)  # 60% threshold
            sell_signal = sell_score > (total_weight * 0.6)
        
        # Apply signals
        df['signal'] = 0
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        # Add individual signal columns for analysis
        df['bb_signal'] = 0
        df.loc[bb_buy, 'bb_signal'] = 1
        df.loc[bb_sell, 'bb_signal'] = -1
        
        df['z_signal'] = 0
        df.loc[z_buy, 'z_signal'] = 1
        df.loc[z_sell, 'z_signal'] = -1
        
        df['rsi_signal'] = 0
        df.loc[rsi_buy, 'rsi_signal'] = 1
        df.loc[rsi_sell, 'rsi_signal'] = -1
        
        self.logger.info(f"Mean Reversion Combo signals: {buy_signal.sum()} buys, "
                        f"{sell_signal.sum()} sells")
        
        return df 