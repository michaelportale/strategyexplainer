"""Volatility Breakout Strategy.

Phase 2 core signal: Breakout strategy with volume confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy with volume confirmation.
    
    Signal Logic:
    - Buy: Price breaks above X-day high with volume > volume threshold
    - Sell: Price breaks below X-day low with volume > volume threshold
    - Optional: Add ATR filter for volatility confirmation
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize volatility breakout strategy.
        
        Args:
            parameters: Strategy parameters
                - breakout_period: Period for high/low calculation (default: 20)
                - volume_period: Period for volume average (default: 10)
                - volume_multiplier: Volume threshold multiplier (default: 1.5)
                - use_atr_filter: Use ATR filter for volatility (default: True)
                - atr_period: ATR calculation period (default: 14)
                - atr_multiplier: ATR threshold multiplier (default: 2.0)
                - min_price_change: Minimum price change % for signal (default: 0.01)
        """
        default_params = {
            'breakout_period': 20,
            'volume_period': 10,
            'volume_multiplier': 1.5,
            'use_atr_filter': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_price_change': 0.01  # 1% minimum move
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Volatility Breakout", default_params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data has required columns for breakout strategy."""
        required_cols = ['high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns for breakout strategy: {missing_cols}")
            return False
            
        return super().validate_data(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals.
        
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
        
        # Calculate breakout levels
        breakout_period = self.parameters['breakout_period']
        df['high_breakout'] = df['high'].rolling(window=breakout_period).max().shift(1)
        df['low_breakout'] = df['low'].rolling(window=breakout_period).min().shift(1)
        
        # Calculate volume threshold
        volume_period = self.parameters['volume_period']
        volume_multiplier = self.parameters['volume_multiplier']
        df['volume_avg'] = df['volume'].rolling(window=volume_period).mean()
        df['volume_threshold'] = df['volume_avg'] * volume_multiplier
        
        # ATR filter if enabled
        if self.parameters['use_atr_filter']:
            df['atr'] = self._calculate_atr(df, self.parameters['atr_period'])
            df['atr_threshold'] = df['atr'] * self.parameters['atr_multiplier']
        
        # Initialize signals
        df['signal'] = 0
        
        # Upside breakout conditions
        price_breaks_high = df['close'] > df['high_breakout']
        volume_confirms = df['volume'] > df['volume_threshold']
        
        # Price change filter
        min_change = self.parameters['min_price_change']
        price_change_up = (df['close'] - df['high_breakout']) / df['high_breakout'] > min_change
        
        upside_breakout = price_breaks_high & volume_confirms & price_change_up
        
        # Downside breakout conditions
        price_breaks_low = df['close'] < df['low_breakout']
        price_change_down = (df['low_breakout'] - df['close']) / df['low_breakout'] > min_change
        
        downside_breakout = price_breaks_low & volume_confirms & price_change_down
        
        # Apply ATR filter if enabled
        if self.parameters['use_atr_filter']:
            # Only trade if current move is significant relative to recent volatility
            significant_move_up = (df['close'] - df['close'].shift(1)) > df['atr_threshold']
            significant_move_down = (df['close'].shift(1) - df['close']) > df['atr_threshold']
            
            upside_breakout = upside_breakout & significant_move_up
            downside_breakout = downside_breakout & significant_move_down
        
        # Apply signals
        df.loc[upside_breakout, 'signal'] = 1
        df.loc[downside_breakout, 'signal'] = -1
        
        # Add analysis columns
        df['upside_breakout'] = upside_breakout
        df['downside_breakout'] = downside_breakout
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        self.logger.info(f"Breakout signals: {upside_breakout.sum()} upside breakouts, "
                        f"{downside_breakout.sum()} downside breakouts")
        
        return df


class ChannelBreakoutStrategy(BaseStrategy):
    """Simple price channel breakout strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize channel breakout strategy.
        
        Args:
            parameters: Strategy parameters
                - channel_period: Period for channel calculation (default: 20)
                - breakout_threshold: % threshold for breakout (default: 0.01)
        """
        default_params = {
            'channel_period': 20, 
            'breakout_threshold': 0.01
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Channel Breakout", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate channel breakout signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate channel bounds
        period = self.parameters['channel_period']
        df['channel_high'] = df['high'].rolling(window=period).max()
        df['channel_low'] = df['low'].rolling(window=period).min()
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
        
        # Calculate channel width for threshold
        threshold = self.parameters['breakout_threshold']
        df['channel_width'] = df['channel_high'] - df['channel_low']
        df['breakout_buffer'] = df['channel_width'] * threshold
        
        # Generate signals
        df['signal'] = 0
        
        # Buy on upper channel breakout
        upper_breakout = df['close'] > (df['channel_high'] + df['breakout_buffer'])
        
        # Sell on lower channel breakout
        lower_breakout = df['close'] < (df['channel_low'] - df['breakout_buffer'])
        
        df.loc[upper_breakout, 'signal'] = 1
        df.loc[lower_breakout, 'signal'] = -1
        
        self.logger.info(f"Channel breakout signals: {upper_breakout.sum()} upper, "
                        f"{lower_breakout.sum()} lower")
        
        return df


class VolumeBreakoutStrategy(BaseStrategy):
    """Volume-based breakout strategy focusing on volume spikes."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize volume breakout strategy.
        
        Args:
            parameters: Strategy parameters
                - volume_period: Period for volume average (default: 20)
                - volume_spike_multiplier: Volume spike threshold (default: 3.0)
                - price_move_threshold: Minimum price move % (default: 0.02)
        """
        default_params = {
            'volume_period': 20,
            'volume_spike_multiplier': 3.0,
            'price_move_threshold': 0.02
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("Volume Breakout", default_params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data has volume column."""
        if 'volume' not in data.columns:
            self.logger.error("Volume column required for volume breakout strategy")
            return False
        return super().validate_data(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume breakout signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signal column
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate volume metrics
        period = self.parameters['volume_period']
        multiplier = self.parameters['volume_spike_multiplier']
        
        df['volume_avg'] = df['volume'].rolling(window=period).mean()
        df['volume_spike'] = df['volume'] > (df['volume_avg'] * multiplier)
        
        # Calculate price moves
        df['price_change'] = df['close'].pct_change()
        threshold = self.parameters['price_move_threshold']
        
        # Generate signals
        df['signal'] = 0
        
        # Buy on volume spike with positive price move
        buy_condition = df['volume_spike'] & (df['price_change'] > threshold)
        
        # Sell on volume spike with negative price move
        sell_condition = df['volume_spike'] & (df['price_change'] < -threshold)
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Add analysis columns
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        self.logger.info(f"Volume breakout signals: {buy_condition.sum()} volume buy spikes, "
                        f"{sell_condition.sum()} volume sell spikes")
        
        return df
