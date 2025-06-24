"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.
    
    All strategies must implement generate_signals() method.
    Provides common functionality and standardized interface.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            parameters: Dictionary of strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on price data.
        
        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with signal column added (1=buy, -1=sell, 0=hold)
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        required_cols = ['close']  # Most basic requirement
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        if data.empty:
            self.logger.error("Data is empty")
            return False
            
        return True
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        df = data.copy()
        
        # Simple Moving Averages
        if 'sma_10' not in df.columns:
            df['sma_10'] = df['close'].rolling(window=10).mean()
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(window=50).mean()
        if 'sma_200' not in df.columns:
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
        # RSI
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['close'])
            
        # Bollinger Bands
        if 'bb_upper' not in df.columns:
            bb_period = 20
            bb_std = 2
            sma = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = sma + (std * bb_std)
            df['bb_lower'] = sma - (std * bb_std)
            df['bb_middle'] = sma
            
        # ATR (Average True Range)
        if 'atr' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = self._calculate_atr(df)
            
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range.
        
        Args:
            data: DataFrame with high, low, close columns
            period: ATR period
            
        Returns:
            ATR series
        """
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information.
        
        Returns:
            Dictionary with strategy info
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'class': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.__class__.__name__})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"


class StrategyComposer:
    """Compose multiple strategies together."""
    
    def __init__(self, strategies: List[BaseStrategy], combination_method: str = 'majority'):
        """Initialize strategy composer.
        
        Args:
            strategies: List of strategy instances
            combination_method: How to combine signals ('majority', 'unanimous', 'any')
        """
        self.strategies = strategies
        self.combination_method = combination_method
        self.logger = logging.getLogger(__name__)
    
    def generate_combined_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate combined signals from multiple strategies.
        
        Args:
            data: Input price data
            
        Returns:
            DataFrame with combined signal column
        """
        df = data.copy()
        
        # Generate signals from each strategy
        individual_signals = {}
        for strategy in self.strategies:
            try:
                strategy_df = strategy.generate_signals(data)
                individual_signals[strategy.name] = strategy_df['signal']
            except Exception as e:
                self.logger.error(f"Error generating signals for {strategy.name}: {e}")
                individual_signals[strategy.name] = pd.Series(0, index=data.index)
        
        # Combine signals
        signal_df = pd.DataFrame(individual_signals)
        
        if self.combination_method == 'majority':
            # Take majority vote
            buy_votes = (signal_df == 1).sum(axis=1)
            sell_votes = (signal_df == -1).sum(axis=1)
            total_strategies = len(self.strategies)
            
            df['signal'] = 0
            df.loc[buy_votes > total_strategies / 2, 'signal'] = 1
            df.loc[sell_votes > total_strategies / 2, 'signal'] = -1
            
        elif self.combination_method == 'unanimous':
            # All strategies must agree
            df['signal'] = 0
            df.loc[(signal_df == 1).all(axis=1), 'signal'] = 1
            df.loc[(signal_df == -1).all(axis=1), 'signal'] = -1
            
        elif self.combination_method == 'any':
            # Any strategy triggers signal
            df['signal'] = 0
            df.loc[(signal_df == 1).any(axis=1), 'signal'] = 1
            df.loc[(signal_df == -1).any(axis=1), 'signal'] = -1
        
        # Add individual signals as columns for analysis
        for strategy_name, signals in individual_signals.items():
            df[f'signal_{strategy_name.lower().replace(" ", "_")}'] = signals
        
        return df 