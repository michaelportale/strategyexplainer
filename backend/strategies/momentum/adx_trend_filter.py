"""ADX (Average Directional Index) trend filter strategy for trend strength analysis.

This module implements strategies based on the ADX indicator, which measures the
strength of a trend regardless of direction. ADX is particularly useful for
determining when to trade trend-following strategies vs when to avoid trading
due to weak or non-existent trends.

ADX Components:
- ADX: Average Directional Index (trend strength, 0-100)
- +DI: Plus Directional Indicator (bullish directional movement)  
- -DI: Minus Directional Indicator (bearish directional movement)

Strategy Philosophy:
- Only trade when trends are strong (high ADX)
- Use directional indicators (+DI, -DI) for signal direction
- Avoid trading in low ADX environments (choppy/sideways markets)
- Combine with other indicators for trend confirmation

Signal Logic:
- Strong Trend: ADX > threshold (typically 25-30)
- Bullish Signal: +DI > -DI in strong trend environment
- Bearish Signal: -DI > +DI in strong trend environment
- No Signal: ADX below threshold (weak trend)

Classes:
    AdxTrendFilterStrategy: ADX-based trend strength filter with MA crossover
    AdxDirectionalStrategy: Pure ADX with +DI/-DI directional signals
    AdxComboStrategy: ADX combined with other trend indicators

Example:
    >>> strategy = AdxTrendFilterStrategy({
    ...     'adx_period': 14,
    ...     'adx_threshold': 25,
    ...     'ma_fast': 10,
    ...     'ma_slow': 30
    ... })
    >>> signals = strategy.generate_signals(price_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class AdxTrendFilterStrategy(BaseStrategy):
    """ADX trend filter strategy combining trend strength with moving average signals.
    
    This strategy uses the ADX (Average Directional Index) to filter trading signals
    based on trend strength. It only generates signals when the ADX indicates a strong
    trend is present, thereby avoiding whipsaw trades in choppy market conditions.
    
    Strategy Components:
    1. ADX: Measures trend strength (0-100 scale)
    2. +DI/-DI: Directional indicators for trend direction
    3. Moving Averages: Primary signal generation mechanism
    4. Trend Filter: ADX threshold to enable/disable trading
    
    Signal Rules:
    - Trade Condition: ADX > threshold (strong trend present)
    - BUY: Fast MA > Slow MA AND ADX > threshold AND +DI > -DI
    - SELL: Fast MA < Slow MA AND ADX > threshold AND -DI > +DI
    - HOLD: ADX < threshold (weak trend, avoid trading)
    
    Key Features:
    - Reduces false signals in ranging markets
    - Configurable ADX threshold for different market conditions
    - Optional directional confirmation with +DI/-DI
    - Detailed trend strength analysis
    - Works with any base trend-following strategy
    
    Best Use Cases:
    - Trending markets with clear directional bias
    - Filter for other trend-following strategies
    - Medium to long-term position trading
    - Risk management for trend strategies
    
    Limitations:
    - May miss early trend moves (ADX lags)
    - Reduces trading frequency significantly
    - Less effective in consistently trending markets
    - Requires parameter optimization for different assets
    
    Attributes:
        strategy_category: 'momentum' for auto-registration
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize ADX trend filter strategy.
        
        Sets up the ADX strategy with parameters optimized for trend strength
        filtering. The default ADX threshold of 25 is widely used in technical
        analysis to distinguish between trending and non-trending markets.
        
        Args:
            parameters: Strategy configuration dictionary:
                - adx_period (int): ADX calculation period (default: 14)
                    Standard ADX period, shorter = more sensitive
                - adx_threshold (float): Minimum ADX for trading (default: 25)
                    Higher values = only trade in very strong trends
                    Typical range: 20-35
                - ma_fast (int): Fast moving average period (default: 10)
                    Primary trend signal generation
                - ma_slow (int): Slow moving average period (default: 30)
                    Trend confirmation
                - ma_type (str): Moving average type 'sma' or 'ema' (default: 'ema')
                - use_di_filter (bool): Use +DI/-DI for direction (default: True)
                    Additional confirmation using directional indicators
                - min_di_separation (float): Minimum +DI/-DI difference (default: 2.0)
                    Requires clear directional bias
                    
        Example:
            >>> # Conservative high-strength trend filter
            >>> params = {
            ...     'adx_threshold': 35,
            ...     'ma_fast': 20,
            ...     'ma_slow': 50,
            ...     'use_di_filter': True
            ... }
            >>> strategy = AdxTrendFilterStrategy(params)
        """
        # Research-based default parameters for trend strength filtering
        default_params = {
            'adx_period': 14,           # Standard ADX calculation period
            'adx_threshold': 25,        # Minimum trend strength for trading
            'ma_fast': 10,              # Fast MA for trend signals
            'ma_slow': 30,              # Slow MA for trend confirmation
            'ma_type': 'ema',           # More responsive moving averages
            'use_di_filter': True,      # Use directional indicator confirmation
            'min_di_separation': 2.0    # Minimum +DI/-DI difference required
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("ADX Trend Filter", default_params)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate ADX and directional indicators.
        
        Computes the complete ADX system including the ADX line itself and the
        directional indicators (+DI and -DI). This provides comprehensive trend
        strength and direction analysis.
        
        Calculation Steps:
        1. Calculate True Range (TR)
        2. Calculate Directional Movement (+DM, -DM)
        3. Smooth TR, +DM, -DM using exponential moving averages
        4. Calculate +DI and -DI from smoothed values
        5. Calculate DX from +DI and -DI
        6. Calculate ADX as smoothed DX
        
        Args:
            data: DataFrame containing 'high', 'low', 'close' columns
            period: Smoothing period for ADX calculation (default: 14)
            
        Returns:
            Dict containing:
                - 'adx': ADX trend strength values (0-100)
                - 'plus_di': +DI bullish directional indicator
                - 'minus_di': -DI bearish directional indicator
                - 'dx': DX intermediate calculation
                
        Raises:
            KeyError: If required OHLC columns are missing
            
        Example:
            >>> adx_data = strategy._calculate_adx(data, period=14)
            >>> strong_trend = adx_data['adx'] > 25
            >>> bullish_direction = adx_data['plus_di'] > adx_data['minus_di']
        """
        # Validate required columns
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns for ADX calculation: {missing_cols}")
        
        # Step 1: Calculate True Range components
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        
        # True Range is the maximum of the three components
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Step 2: Calculate Directional Movement
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        
        # Only keep positive directional movements
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Step 3: Smooth using Wilder's smoothing (modified EMA)
        alpha = 1.0 / period
        
        # Smooth True Range
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        
        # Smooth Directional Movements
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
        
        # Step 4: Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # Step 5: Calculate DX (Directional Index)
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * (di_diff / di_sum)
        dx = dx.fillna(0)  # Handle division by zero
        
        # Step 6: Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ADX-filtered trend signals with comprehensive analysis.
        
        Implements the ADX trend filtering methodology by combining traditional
        moving average signals with ADX trend strength analysis. Only generates
        signals when the ADX indicates sufficient trend strength is present.
        
        Implementation Steps:
        1. Validate input data for required OHLC columns
        2. Calculate ADX, +DI, and -DI indicators
        3. Calculate moving averages for base trend signals
        4. Apply ADX threshold filter for trend strength
        5. Apply optional directional indicator filter
        6. Generate final buy/sell/hold signals
        7. Add comprehensive analysis columns
        
        Args:
            data: DataFrame containing OHLC data ('high', 'low', 'close' required)
            
        Returns:
            DataFrame with original data plus ADX analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'adx': ADX trend strength values
                - 'plus_di': +DI bullish directional indicator
                - 'minus_di': -DI bearish directional indicator
                - 'ma_fast': Fast moving average
                - 'ma_slow': Slow moving average
                - 'strong_trend': Boolean ADX above threshold
                - 'bullish_direction': Boolean +DI > -DI
                - 'bearish_direction': Boolean -DI > +DI
                - 'trend_signal': Base trend signal before ADX filter
                
        Signal Logic:
            - BUY: MA trend bullish + ADX > threshold + directional confirmation
            - SELL: MA trend bearish + ADX > threshold + directional confirmation  
            - HOLD: ADX below threshold (insufficient trend strength)
            
        Example:
            >>> signals = strategy.generate_signals(ohlc_data)
            >>> strong_trends = signals[signals['strong_trend']]
            >>> print(f"Strong trend periods: {len(strong_trends)} / {len(signals)}")
        """
        # Step 1: Validate input data for ADX requirements
        if not self.validate_data(data):
            self.logger.error("Data validation failed for ADX strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Check for required OHLC columns
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns for ADX: {missing_cols}")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with copy of original data
        df = data.copy()
        
        # Step 2: Calculate ADX system
        adx_period = self.parameters['adx_period']
        adx_data = self._calculate_adx(df, adx_period)
        
        # Add ADX components to dataframe
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']
        
        self.logger.debug(f"Calculated ADX({adx_period}) and directional indicators")
        
        # Step 3: Calculate moving averages for base trend signals
        ma_fast = self.parameters['ma_fast']
        ma_slow = self.parameters['ma_slow']
        ma_type = self.parameters['ma_type']
        
        df['ma_fast'] = self._calculate_moving_average(df['close'], ma_fast, ma_type)
        df['ma_slow'] = self._calculate_moving_average(df['close'], ma_slow, ma_type)
        
        # Generate base trend signals using crossover utility
        crossover_signals = self._calculate_crossover_signals(df['ma_fast'], df['ma_slow'])
        df['trend_signal'] = crossover_signals['trend_signal']
        
        # Step 4: Apply ADX trend strength filter
        adx_threshold = self.parameters['adx_threshold']
        df['strong_trend'] = df['adx'] > adx_threshold
        
        # Step 5: Apply directional indicator analysis
        df['bullish_direction'] = df['plus_di'] > df['minus_di']
        df['bearish_direction'] = df['minus_di'] > df['plus_di']
        
        # Calculate directional strength if filter enabled
        if self.parameters['use_di_filter']:
            min_separation = self.parameters['min_di_separation']
            df['strong_bullish'] = (df['plus_di'] - df['minus_di']) > min_separation
            df['strong_bearish'] = (df['minus_di'] - df['plus_di']) > min_separation
            
            self.logger.debug(f"Applied DI filter with minimum separation: {min_separation}")
        else:
            df['strong_bullish'] = df['bullish_direction']
            df['strong_bearish'] = df['bearish_direction']
        
        # Step 6: Initialize signal column
        df['signal'] = 0
        
        # Step 7: Generate filtered signals
        # BUY: Bullish MA trend + Strong ADX + Bullish direction
        buy_condition = (
            (df['trend_signal'] == 1) &           # Bullish MA crossover
            df['strong_trend'] &                  # ADX above threshold
            df['strong_bullish']                  # Strong bullish direction
        )
        
        # SELL: Bearish MA trend + Strong ADX + Bearish direction
        sell_condition = (
            (df['trend_signal'] == -1) &          # Bearish MA crossover
            df['strong_trend'] &                  # ADX above threshold
            df['strong_bearish']                  # Strong bearish direction
        )
        
        # Apply final signals
        df.loc[buy_condition, 'signal'] = 1    # Buy signal
        df.loc[sell_condition, 'signal'] = -1  # Sell signal
        
        # Step 8: Log comprehensive statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        total_base_signals = (df['trend_signal'] != 0).sum()
        strong_trend_periods = df['strong_trend'].sum()
        
        self.logger.info(f"ADX filtered signals: {buy_count} buys, {sell_count} sells")
        self.logger.info(f"Filtered {total_base_signals} base signals to {buy_count + sell_count} ADX signals")
        
        # ADX statistics for analysis
        adx_mean = df['adx'].mean()
        strong_trend_pct = (strong_trend_periods / len(df) * 100).round(1)
        
        self.logger.debug(f"ADX stats: mean={adx_mean:.1f}, strong trend={strong_trend_pct}% of time")
        
        return df


class AdxDirectionalStrategy(BaseStrategy):
    """Pure ADX directional strategy using only +DI/-DI signals.
    
    This strategy relies purely on the directional indicators (+DI and -DI)
    for signal generation, with ADX used only as a filter for trend strength.
    It's more responsive than MA-based approaches but may generate more noise.
    
    Signal Rules:
    - BUY: +DI crosses above -DI AND ADX > threshold
    - SELL: -DI crosses above +DI AND ADX > threshold
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize pure ADX directional strategy."""
        default_params = {
            'adx_period': 14,
            'adx_threshold': 25,
            'di_crossover_threshold': 5.0,  # Minimum DI difference for signals
            'use_adx_slope_filter': True,   # Require rising ADX
            'adx_slope_period': 3          # Periods for ADX slope calculation
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("ADX Directional", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pure directional indicator signals."""
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Check for OHLC data
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            self.logger.error(f"Missing OHLC data for ADX calculation")
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate ADX system using the main strategy's method
        main_strategy = AdxTrendFilterStrategy(self.parameters)
        adx_data = main_strategy._calculate_adx(df, self.parameters['adx_period'])
        
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']
        
        # Apply ADX strength filter
        df['strong_trend'] = df['adx'] > self.parameters['adx_threshold']
        
        # Optional ADX slope filter (rising ADX shows strengthening trend)
        if self.parameters['use_adx_slope_filter']:
            slope_period = self.parameters['adx_slope_period']
            df['adx_slope'] = df['adx'] - df['adx'].shift(slope_period)
            df['adx_rising'] = df['adx_slope'] > 0
        else:
            df['adx_rising'] = True
        
        # Calculate DI crossovers
        di_crossover = self._calculate_crossover_signals(df['plus_di'], df['minus_di'])
        
        # Apply crossover threshold
        di_threshold = self.parameters['di_crossover_threshold']
        strong_bullish_cross = di_crossover['bullish_signal'] & ((df['plus_di'] - df['minus_di']) > di_threshold)
        strong_bearish_cross = di_crossover['bearish_signal'] & ((df['minus_di'] - df['plus_di']) > di_threshold)
        
        # Generate signals
        df['signal'] = 0
        df.loc[strong_bullish_cross & df['strong_trend'] & df['adx_rising'], 'signal'] = 1
        df.loc[strong_bearish_cross & df['strong_trend'] & df['adx_rising'], 'signal'] = -1
        
        # Log statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        
        self.logger.info(f"ADX directional signals: {buy_count} buys, {sell_count} sells")
        
        return df


class AdxComboStrategy(BaseStrategy):
    """Combined ADX strategy with multiple confirmation signals.
    
    This advanced strategy combines ADX trend strength filtering with multiple
    technical indicators for robust signal generation. It's designed for traders
    who want high-confidence signals with multiple layers of confirmation.
    
    Components:
    - ADX for trend strength
    - Moving averages for trend direction
    - RSI for momentum confirmation
    - Volume analysis for institutional confirmation
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'momentum'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize combined ADX strategy with multiple confirmations."""
        default_params = {
            'adx_period': 14,
            'adx_threshold': 25,
            'ma_period': 20,
            'rsi_period': 14,
            'rsi_neutral_low': 45,
            'rsi_neutral_high': 55,
            'volume_ma_period': 20,
            'volume_threshold': 1.2,
            'require_all_confirmations': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("ADX Combo", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-confirmation ADX signals."""
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Calculate all indicators
        adx_strategy = AdxTrendFilterStrategy({'adx_period': self.parameters['adx_period']})
        adx_data = adx_strategy._calculate_adx(df, self.parameters['adx_period'])
        
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']
        
        # Add other technical indicators
        df = self.add_technical_indicators(df, {
            'rsi_period': self.parameters['rsi_period'],
            'sma_periods': [self.parameters['ma_period']]
        })
        
        # Volume confirmation if available
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(self.parameters['volume_ma_period']).mean()
            df['volume_confirmation'] = df['volume'] > (volume_ma * self.parameters['volume_threshold'])
        else:
            df['volume_confirmation'] = True
        
        # Individual confirmations
        df['adx_strong'] = df['adx'] > self.parameters['adx_threshold']
        df['price_above_ma'] = df['close'] > df[f"sma_{self.parameters['ma_period']}"]
        df['price_below_ma'] = df['close'] < df[f"sma_{self.parameters['ma_period']}"]
        df['rsi_bullish'] = df['rsi'] > self.parameters['rsi_neutral_high']
        df['rsi_bearish'] = df['rsi'] < self.parameters['rsi_neutral_low']
        df['di_bullish'] = df['plus_di'] > df['minus_di']
        df['di_bearish'] = df['minus_di'] > df['plus_di']
        
        # Combined signals
        if self.parameters['require_all_confirmations']:
            # All confirmations required
            buy_condition = (
                df['adx_strong'] & 
                df['price_above_ma'] & 
                df['rsi_bullish'] & 
                df['di_bullish'] & 
                df['volume_confirmation']
            )
            sell_condition = (
                df['adx_strong'] & 
                df['price_below_ma'] & 
                df['rsi_bearish'] & 
                df['di_bearish'] & 
                df['volume_confirmation']
            )
        else:
            # Majority confirmations (3 out of 4)
            buy_confirmations = (
                df['adx_strong'].astype(int) +
                df['price_above_ma'].astype(int) +
                df['rsi_bullish'].astype(int) +
                df['di_bullish'].astype(int)
            )
            sell_confirmations = (
                df['adx_strong'].astype(int) +
                df['price_below_ma'].astype(int) +
                df['rsi_bearish'].astype(int) +
                df['di_bearish'].astype(int)
            )
            
            buy_condition = (buy_confirmations >= 3) & df['volume_confirmation']
            sell_condition = (sell_confirmations >= 3) & df['volume_confirmation']
        
        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Log statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        
        self.logger.info(f"ADX combo signals: {buy_count} buys, {sell_count} sells")
        
        return df