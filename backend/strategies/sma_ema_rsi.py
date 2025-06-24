"""SMA/EMA + RSI Trend Strategy.

Phase 1 core signal: Combines moving average trend with RSI momentum filter.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class SmaEmaRsiStrategy(BaseStrategy):
    """Trend strategy using SMA/EMA crossover with RSI filter.
    
    Signal Logic:
    - Buy: Fast MA > Slow MA AND RSI < overbought threshold
    - Sell: Fast MA < Slow MA AND RSI > oversold threshold
    - Hold: Otherwise
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize SMA/EMA + RSI strategy.
        
        Args:
            parameters: Strategy parameters
                - fast_period: Fast moving average period (default: 10)
                - slow_period: Slow moving average period (default: 50)
                - rsi_period: RSI calculation period (default: 14)
                - rsi_oversold: RSI oversold threshold (default: 30)
                - rsi_overbought: RSI overbought threshold (default: 70)
                - ma_type: Moving average type 'sma' or 'ema' (default: 'sma')
                - use_rsi_filter: Whether to use RSI as filter (default: True)
        """
        default_params = {
            'fast_period': 10,
            'slow_period': 50,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'ma_type': 'sma',
            'use_rsi_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("SMA/EMA + RSI", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using SMA/EMA crossover with RSI filter.
        
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
        
        # Calculate moving averages
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters['ma_type']
        
        if ma_type.lower() == 'ema':
            df['fast_ma'] = df['close'].ewm(span=fast_period).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period).mean()
        else:  # SMA
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Initialize signals
        df['signal'] = 0
        
        # Basic trend signals: Fast MA vs Slow MA
        trend_bullish = df['fast_ma'] > df['slow_ma']
        trend_bearish = df['fast_ma'] < df['slow_ma']
        
        if self.parameters['use_rsi_filter']:
            # Apply RSI filter to avoid buying at overbought/selling at oversold
            rsi_not_overbought = df['rsi'] < self.parameters['rsi_overbought']
            rsi_not_oversold = df['rsi'] > self.parameters['rsi_oversold']
            
            # Buy: Uptrend + RSI not overbought
            buy_condition = trend_bullish & rsi_not_overbought
            
            # Sell: Downtrend + RSI not oversold  
            sell_condition = trend_bearish & rsi_not_oversold
        else:
            # Pure trend following without RSI filter
            buy_condition = trend_bullish
            sell_condition = trend_bearish
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Add additional columns for analysis
        df['trend_bullish'] = trend_bullish
        df['trend_bearish'] = trend_bearish
        
        self.logger.info(f"Generated signals: {(df['signal'] == 1).sum()} buys, "
                        f"{(df['signal'] == -1).sum()} sells, "
                        f"{(df['signal'] == 0).sum()} holds")
        
        return df


class CrossoverStrategy(BaseStrategy):
    """Simple MA crossover strategy (subset of SmaEmaRsiStrategy)."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize crossover strategy.
        
        Args:
            parameters: Strategy parameters
                - fast_period: Fast MA period (default: 10)
                - slow_period: Slow MA period (default: 50)
                - ma_type: 'sma' or 'ema' (default: 'sma')
        """
        default_params = {
            'fast_period': 10,
            'slow_period': 50,
            'ma_type': 'sma'
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MA Crossover", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals.
        
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
        
        # Calculate moving averages
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        ma_type = self.parameters['ma_type']
        
        if ma_type.lower() == 'ema':
            df['fast_ma'] = df['close'].ewm(span=fast_period).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period).mean()
        else:
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        # Generate crossover signals
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Bullish crossover
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Bearish crossover
        
        # Detect actual crossover points (optional enhancement)
        fast_ma_prev = df['fast_ma'].shift(1)
        slow_ma_prev = df['slow_ma'].shift(1)
        
        # Golden cross: fast MA crosses above slow MA
        golden_cross = (df['fast_ma'] > df['slow_ma']) & (fast_ma_prev <= slow_ma_prev)
        # Death cross: fast MA crosses below slow MA
        death_cross = (df['fast_ma'] < df['slow_ma']) & (fast_ma_prev >= slow_ma_prev)
        
        df['golden_cross'] = golden_cross
        df['death_cross'] = death_cross
        
        self.logger.info(f"Generated crossover signals: {golden_cross.sum()} golden crosses, "
                        f"{death_cross.sum()} death crosses")
        
        return df


class RsiStrategy(BaseStrategy):
    """Pure RSI momentum strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI strategy.
        
        Args:
            parameters: Strategy parameters
                - rsi_period: RSI period (default: 14)
                - oversold_threshold: Buy threshold (default: 30)
                - overbought_threshold: Sell threshold (default: 70)
        """
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("RSI Momentum", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI momentum signals.
        
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
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when RSI crosses above oversold
        buy_condition = (df['rsi'] > self.parameters['oversold_threshold']) & \
                       (df['rsi'].shift(1) <= self.parameters['oversold_threshold'])
        
        # Sell when RSI crosses below overbought
        sell_condition = (df['rsi'] < self.parameters['overbought_threshold']) & \
                        (df['rsi'].shift(1) >= self.parameters['overbought_threshold'])
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        self.logger.info(f"RSI signals: {buy_condition.sum()} oversold bounces, "
                        f"{sell_condition.sum()} overbought reversals")
        
        return df
