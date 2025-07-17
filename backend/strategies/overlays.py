"""Strategy overlay decorators for composable functionality enhancement.

This module implements the Decorator pattern for trading strategies, providing
reusable overlays that can be composed with any base strategy. Each overlay
adds specific functionality without modifying the underlying strategy logic.

Available Overlays:
- RegimeFilterOverlay: Filter signals based on market regime detection
- PositionSizingOverlay: Apply dynamic position sizing rules
- RiskManagementOverlay: Add stop-loss and take-profit functionality
- VolumeFilterOverlay: Filter signals based on volume confirmation
- VolatilityFilterOverlay: Filter signals based on volatility conditions
- TimeFilterOverlay: Apply time-based trading restrictions

Design Principles:
- Each overlay can wrap any BaseStrategy or EnhancedBaseStrategy
- Overlays can be chained together for complex behaviors
- Each overlay maintains the same interface as the base strategy
- Parameter management is integrated for configuration
- Comprehensive logging for overlay-specific analysis

Example:
    >>> base_strategy = MacdCrossoverStrategy()
    >>> regime_filtered = RegimeFilterOverlay(base_strategy, regime_detector)
    >>> risk_managed = RiskManagementOverlay(regime_filtered, {'stop_loss_pct': 0.05})
    >>> final_strategy = PositionSizingOverlay(risk_managed, {'position_size': 0.02})
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union
from abc import ABC, abstractmethod
from datetime import datetime, time
from enum import Enum

from .enhanced_base import StrategyDecorator, EnhancedBaseStrategy
from .parameters import ParameterDefinition, ParameterSchema

# Import logging utilities with absolute path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_config import LoggerManager, StrategyError


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    UNKNOWN = "unknown"


class RegimeDetector(ABC):
    """Abstract base class for market regime detection."""
    
    @abstractmethod
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regime for each data point.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Series with MarketRegime values for each timestamp
        """
        pass


class SimpleRegimeDetector(RegimeDetector):
    """Simple regime detector based on moving average slope and volatility."""
    
    def __init__(self, ma_period: int = 50, volatility_period: int = 20):
        self.ma_period = ma_period
        self.volatility_period = volatility_period
    
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect regime using moving average slope and volatility."""
        # Calculate moving average and slope
        ma = data['close'].rolling(self.ma_period).mean()
        ma_slope = ma.diff(5) / ma.shift(5)  # 5-period slope
        
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.volatility_period).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)
        
        # Classify regimes
        regime = pd.Series(MarketRegime.UNKNOWN.value, index=data.index)
        
        # High volatility regime
        regime.loc[vol_percentile > 0.8] = MarketRegime.HIGH_VOLATILITY.value
        
        # Low volatility regime
        regime.loc[vol_percentile < 0.2] = MarketRegime.LOW_VOLATILITY.value
        
        # Trend regimes (when volatility is normal)
        normal_vol = (vol_percentile >= 0.2) & (vol_percentile <= 0.8)
        regime.loc[normal_vol & (ma_slope > 0.001)] = MarketRegime.BULL.value
        regime.loc[normal_vol & (ma_slope < -0.001)] = MarketRegime.BEAR.value
        regime.loc[normal_vol & (ma_slope.abs() <= 0.001)] = MarketRegime.SIDEWAYS.value
        
        return regime


class RegimeFilterOverlay(StrategyDecorator):
    """Strategy overlay that filters signals based on market regime.
    
    This overlay allows trading only in favorable market regimes, helping
    strategies avoid unfavorable market conditions. For example, a trend-following
    strategy might only trade during bull or bear regimes, avoiding sideways markets.
    
    Parameters:
        allowed_regimes: List of market regimes where trading is permitted
        regime_detector: RegimeDetector instance for regime classification
        override_threshold: Confidence threshold for regime override
        
    Example:
        >>> regime_filter = RegimeFilterOverlay(
        ...     base_strategy,
        ...     allowed_regimes=[MarketRegime.BULL, MarketRegime.BEAR],
        ...     regime_detector=SimpleRegimeDetector()
        ... )
    """
    
    PARAMETER_SCHEMA = ParameterSchema({
        'allowed_regimes': ParameterDefinition(
            default=[MarketRegime.BULL.value, MarketRegime.BEAR.value],
            type=list,
            description="Market regimes where trading is allowed",
            category="Regime Filter"
        ),
        'regime_confidence_threshold': ParameterDefinition(
            default=0.7,
            type=float,
            range=(0.0, 1.0),
            description="Minimum confidence for regime detection",
            category="Regime Filter"
        ),
        'transition_buffer_periods': ParameterDefinition(
            default=3,
            type=int,
            range=(0, 10),
            description="Buffer periods during regime transitions",
            category="Regime Filter"
        )
    }, strategy_name="Regime Filter Overlay")
    
    def __init__(self, 
                 wrapped_strategy: EnhancedBaseStrategy,
                 regime_detector: Optional[RegimeDetector] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        """Initialize regime filter overlay.
        
        Args:
            wrapped_strategy: Strategy to wrap with regime filtering
            regime_detector: Optional custom regime detector
            parameters: Overlay configuration parameters
        """
        super().__init__(wrapped_strategy)
        
        # Set up regime detector
        self.regime_detector = regime_detector or SimpleRegimeDetector()
        
        # Validate and set parameters
        user_params = parameters or {}
        self.overlay_parameters = self.PARAMETER_SCHEMA.validate(user_params)
        
        self.logger.info(f"Initialized regime filter overlay with {len(self.overlay_parameters)} parameters")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-filtered signals.
        
        Args:
            data: Market data for signal generation
            
        Returns:
            DataFrame with regime-filtered signals
        """
        # Generate base signals
        signals_df = self.wrapped_strategy.generate_signals(data)
        
        # Detect market regimes
        regimes = self.regime_detector.detect_regime(data)
        signals_df['market_regime'] = regimes
        
        # Apply regime filter
        allowed_regimes = self.overlay_parameters['allowed_regimes']
        buffer_periods = self.overlay_parameters['transition_buffer_periods']
        
        # Create regime filter mask
        regime_allowed = signals_df['market_regime'].isin(allowed_regimes)
        
        # Apply transition buffer - don't trade for N periods after regime change
        if buffer_periods > 0:
            regime_changed = signals_df['market_regime'] != signals_df['market_regime'].shift(1)
            buffer_mask = ~regime_changed.rolling(buffer_periods).sum().astype(bool)
            regime_allowed = regime_allowed & buffer_mask
        
        # Filter signals based on regime
        original_signals = signals_df['signal'].copy()
        signals_df.loc[~regime_allowed, 'signal'] = 0
        
        # Add regime analysis columns
        signals_df['regime_allowed'] = regime_allowed
        signals_df['regime_filtered'] = original_signals != signals_df['signal']
        
        # Log filtering statistics
        total_signals = (original_signals != 0).sum()
        filtered_signals = (signals_df['signal'] != 0).sum()
        filtered_count = total_signals - filtered_signals
        
        self.logger.info(f"Regime filter: kept {filtered_signals}/{total_signals} signals "
                        f"({filtered_count} filtered out)")
        
        return signals_df


class PositionSizingOverlay(StrategyDecorator):
    """Strategy overlay for dynamic position sizing.
    
    This overlay applies sophisticated position sizing rules to base strategy
    signals, supporting fixed, volatility-adjusted, and Kelly criterion sizing.
    
    Sizing Methods:
        - fixed: Fixed percentage of capital per trade
        - volatility_adjusted: Size inversely proportional to volatility
        - kelly: Kelly criterion based on win rate and average return
        - equal_weight: Equal weight across all positions
        
    Example:
        >>> position_overlay = PositionSizingOverlay(
        ...     base_strategy,
        ...     {'sizing_method': 'volatility_adjusted', 'base_position_size': 0.02}
        ... )
    """
    
    PARAMETER_SCHEMA = ParameterSchema({
        'sizing_method': ParameterDefinition(
            default='fixed',
            type=str,
            choices=['fixed', 'volatility_adjusted', 'kelly', 'equal_weight'],
            description="Position sizing method",
            category="Position Sizing"
        ),
        'base_position_size': ParameterDefinition(
            default=0.02,
            type=float,
            range=(0.001, 0.5),
            description="Base position size as fraction of capital",
            category="Position Sizing"
        ),
        'max_position_size': ParameterDefinition(
            default=0.1,
            type=float,
            range=(0.001, 1.0),
            description="Maximum allowed position size",
            category="Position Sizing"
        ),
        'volatility_window': ParameterDefinition(
            default=20,
            type=int,
            range=(5, 100),
            description="Lookback window for volatility calculation",
            category="Position Sizing"
        ),
        'target_volatility': ParameterDefinition(
            default=0.02,
            type=float,
            range=(0.005, 0.1),
            description="Target volatility for vol-adjusted sizing",
            category="Position Sizing"
        ),
        'kelly_lookback': ParameterDefinition(
            default=100,
            type=int,
            range=(20, 500),
            description="Lookback period for Kelly criterion calculation",
            category="Position Sizing"
        )
    }, strategy_name="Position Sizing Overlay")
    
    def __init__(self, 
                 wrapped_strategy: EnhancedBaseStrategy,
                 parameters: Optional[Dict[str, Any]] = None):
        """Initialize position sizing overlay."""
        super().__init__(wrapped_strategy)
        
        user_params = parameters or {}
        self.overlay_parameters = self.PARAMETER_SCHEMA.validate(user_params)
        
        self.logger.info(f"Initialized position sizing overlay: {self.overlay_parameters['sizing_method']}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with position sizing information."""
        # Generate base signals
        signals_df = self.wrapped_strategy.generate_signals(data)
        
        # Calculate position sizes
        sizing_method = self.overlay_parameters['sizing_method']
        
        if sizing_method == 'fixed':
            position_sizes = self._calculate_fixed_sizing(signals_df)
        elif sizing_method == 'volatility_adjusted':
            position_sizes = self._calculate_volatility_adjusted_sizing(data, signals_df)
        elif sizing_method == 'kelly':
            position_sizes = self._calculate_kelly_sizing(data, signals_df)
        elif sizing_method == 'equal_weight':
            position_sizes = self._calculate_equal_weight_sizing(signals_df)
        else:
            raise ValueError(f"Unknown sizing method: {sizing_method}")
        
        # Add position sizing information
        signals_df['position_size'] = position_sizes
        signals_df['sizing_method'] = sizing_method
        
        # Apply maximum position size limit
        max_size = self.overlay_parameters['max_position_size']
        signals_df['position_size'] = signals_df['position_size'].clip(upper=max_size)
        
        # Log sizing statistics
        non_zero_sizes = signals_df[signals_df['position_size'] > 0]['position_size']
        if len(non_zero_sizes) > 0:
            avg_size = non_zero_sizes.mean()
            max_used_size = non_zero_sizes.max()
            self.logger.info(f"Position sizing: avg={avg_size:.3f}, max={max_used_size:.3f}")
        
        return signals_df
    
    def _calculate_fixed_sizing(self, signals_df: pd.DataFrame) -> pd.Series:
        """Calculate fixed position sizes."""
        base_size = self.overlay_parameters['base_position_size']
        sizes = pd.Series(0.0, index=signals_df.index)
        sizes.loc[signals_df['signal'] != 0] = base_size
        return sizes
    
    def _calculate_volatility_adjusted_sizing(self, data: pd.DataFrame, signals_df: pd.DataFrame) -> pd.Series:
        """Calculate volatility-adjusted position sizes."""
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        vol_window = self.overlay_parameters['volatility_window']
        volatility = returns.rolling(vol_window).std() * np.sqrt(252)  # Annualized
        
        # Calculate vol-adjusted sizes
        target_vol = self.overlay_parameters['target_volatility']
        base_size = self.overlay_parameters['base_position_size']
        
        # Size inversely proportional to volatility
        vol_multiplier = target_vol / volatility.clip(lower=0.001)  # Avoid division by zero
        adjusted_sizes = base_size * vol_multiplier
        
        # Apply only to signal periods
        sizes = pd.Series(0.0, index=signals_df.index)
        signal_mask = signals_df['signal'] != 0
        sizes.loc[signal_mask] = adjusted_sizes.loc[signal_mask]
        
        return sizes
    
    def _calculate_kelly_sizing(self, data: pd.DataFrame, signals_df: pd.DataFrame) -> pd.Series:
        """Calculate Kelly criterion position sizes."""
        lookback = self.overlay_parameters['kelly_lookback']
        base_size = self.overlay_parameters['base_position_size']
        
        # Calculate historical returns for signals
        returns = data['close'].pct_change()
        
        sizes = pd.Series(0.0, index=signals_df.index)
        
        for i in range(lookback, len(signals_df)):
            if signals_df.iloc[i]['signal'] == 0:
                continue
            
            # Get historical signal performance
            historical_signals = signals_df.iloc[i-lookback:i]['signal']
            historical_returns = returns.iloc[i-lookback:i]
            
            # Calculate win rate and average returns
            signal_returns = historical_returns[historical_signals != 0]
            
            if len(signal_returns) < 10:  # Need minimum history
                sizes.iloc[i] = base_size
                continue
            
            win_rate = (signal_returns > 0).mean()
            avg_win = signal_returns[signal_returns > 0].mean() if (signal_returns > 0).any() else 0
            avg_loss = abs(signal_returns[signal_returns < 0].mean()) if (signal_returns < 0).any() else 0.01
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_fraction = (b * win_rate - (1 - win_rate)) / b
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = base_size
            
            sizes.iloc[i] = kelly_fraction
        
        return sizes
    
    def _calculate_equal_weight_sizing(self, signals_df: pd.DataFrame) -> pd.Series:
        """Calculate equal weight position sizes."""
        # Count simultaneous signals to determine equal weights
        signal_count = (signals_df['signal'] != 0).rolling(window=1).sum()
        
        # For simplicity, use base size (could be enhanced to consider portfolio)
        base_size = self.overlay_parameters['base_position_size']
        sizes = pd.Series(0.0, index=signals_df.index)
        sizes.loc[signals_df['signal'] != 0] = base_size
        
        return sizes


class RiskManagementOverlay(StrategyDecorator):
    """Strategy overlay for risk management with stop-loss and take-profit.
    
    This overlay adds sophisticated risk management to any base strategy,
    including trailing stops, time-based exits, and dynamic risk adjustment.
    
    Features:
        - Fixed and percentage-based stop losses
        - Multiple take-profit levels
        - Trailing stop functionality
        - Time-based exit rules
        - Dynamic risk adjustment based on volatility
        
    Example:
        >>> risk_overlay = RiskManagementOverlay(
        ...     base_strategy,
        ...     {'stop_loss_pct': 0.05, 'take_profit_pct': 0.10, 'use_trailing_stop': True}
        ... )
    """
    
    PARAMETER_SCHEMA = ParameterSchema({
        'stop_loss_pct': ParameterDefinition(
            default=0.05,
            type=float,
            range=(0.01, 0.5),
            description="Stop loss percentage",
            category="Risk Management"
        ),
        'take_profit_pct': ParameterDefinition(
            default=0.10,
            type=float,
            range=(0.01, 1.0),
            description="Take profit percentage",
            category="Risk Management"
        ),
        'use_trailing_stop': ParameterDefinition(
            default=False,
            type=bool,
            description="Enable trailing stop loss",
            category="Risk Management"
        ),
        'trailing_stop_distance': ParameterDefinition(
            default=0.03,
            type=float,
            range=(0.01, 0.2),
            description="Trailing stop distance percentage",
            category="Risk Management"
        ),
        'max_holding_periods': ParameterDefinition(
            default=20,
            type=int,
            range=(1, 100),
            description="Maximum periods to hold a position",
            category="Risk Management"
        ),
        'partial_profit_levels': ParameterDefinition(
            default=[0.05, 0.10],
            type=list,
            description="Partial profit taking levels",
            category="Risk Management"
        ),
        'volatility_adjustment': ParameterDefinition(
            default=True,
            type=bool,
            description="Adjust stops based on volatility",
            category="Risk Management"
        )
    }, strategy_name="Risk Management Overlay")
    
    def __init__(self, 
                 wrapped_strategy: EnhancedBaseStrategy,
                 parameters: Optional[Dict[str, Any]] = None):
        """Initialize risk management overlay."""
        super().__init__(wrapped_strategy)
        
        user_params = parameters or {}
        self.overlay_parameters = self.PARAMETER_SCHEMA.validate(user_params)
        
        # State tracking for risk management
        self.position_entry_price = None
        self.position_entry_date = None
        self.trailing_stop_level = None
        self.current_position = 0  # 1 for long, -1 for short, 0 for flat
        
        self.logger.info("Initialized risk management overlay")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate risk-managed signals."""
        # Generate base signals
        signals_df = self.wrapped_strategy.generate_signals(data)
        
        # Add risk management columns
        signals_df['risk_managed_signal'] = signals_df['signal'].copy()
        signals_df['stop_loss_price'] = np.nan
        signals_df['take_profit_price'] = np.nan
        signals_df['exit_reason'] = ''
        signals_df['position_pnl'] = 0.0
        
        # Apply risk management rules
        self._apply_risk_management(data, signals_df)
        
        return signals_df
    
    def _apply_risk_management(self, data: pd.DataFrame, signals_df: pd.DataFrame):
        """Apply risk management rules to signals."""
        stop_loss_pct = self.overlay_parameters['stop_loss_pct']
        take_profit_pct = self.overlay_parameters['take_profit_pct']
        use_trailing = self.overlay_parameters['use_trailing_stop']
        max_holding = self.overlay_parameters['max_holding_periods']
        
        for i, (idx, row) in enumerate(signals_df.iterrows()):
            current_price = data.loc[idx, 'close']
            
            # Check for new entry signal
            if row['signal'] != 0 and self.current_position == 0:
                self._enter_position(idx, row['signal'], current_price, signals_df.loc[idx])
            
            # Check for exit conditions if in position
            elif self.current_position != 0:
                exit_signal, exit_reason = self._check_exit_conditions(
                    idx, current_price, i, max_holding
                )
                
                if exit_signal:
                    self._exit_position(idx, current_price, exit_reason, signals_df.loc[idx])
                else:
                    # Update trailing stop if enabled
                    if use_trailing:
                        self._update_trailing_stop(current_price)
                    
                    # Update P&L
                    if self.position_entry_price:
                        pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price
                        if self.current_position == -1:
                            pnl_pct = -pnl_pct
                        signals_df.loc[idx, 'position_pnl'] = pnl_pct
    
    def _enter_position(self, timestamp, signal, price, row_ref):
        """Enter a new position."""
        self.current_position = signal
        self.position_entry_price = price
        self.position_entry_date = timestamp
        
        # Set initial stop loss and take profit
        stop_loss_pct = self.overlay_parameters['stop_loss_pct']
        take_profit_pct = self.overlay_parameters['take_profit_pct']
        
        if signal == 1:  # Long position
            row_ref['stop_loss_price'] = price * (1 - stop_loss_pct)
            row_ref['take_profit_price'] = price * (1 + take_profit_pct)
            self.trailing_stop_level = price * (1 - stop_loss_pct)
        else:  # Short position
            row_ref['stop_loss_price'] = price * (1 + stop_loss_pct)
            row_ref['take_profit_price'] = price * (1 - take_profit_pct)
            self.trailing_stop_level = price * (1 + stop_loss_pct)
        
        self.logger.debug(f"Entered {signal} position at {price}")
    
    def _check_exit_conditions(self, timestamp, current_price, period_count, max_holding):
        """Check if any exit conditions are met."""
        if not self.position_entry_price:
            return False, ""
        
        # Check stop loss
        if self.current_position == 1 and current_price <= self.trailing_stop_level:
            return True, "stop_loss"
        elif self.current_position == -1 and current_price >= self.trailing_stop_level:
            return True, "stop_loss"
        
        # Check take profit
        take_profit_pct = self.overlay_parameters['take_profit_pct']
        if self.current_position == 1:
            take_profit_price = self.position_entry_price * (1 + take_profit_pct)
            if current_price >= take_profit_price:
                return True, "take_profit"
        else:
            take_profit_price = self.position_entry_price * (1 - take_profit_pct)
            if current_price <= take_profit_price:
                return True, "take_profit"
        
        # Check maximum holding period
        if period_count >= max_holding:
            return True, "max_holding"
        
        return False, ""
    
    def _update_trailing_stop(self, current_price):
        """Update trailing stop level."""
        if not self.position_entry_price:
            return
        
        trailing_distance = self.overlay_parameters['trailing_stop_distance']
        
        if self.current_position == 1:  # Long position
            new_stop = current_price * (1 - trailing_distance)
            self.trailing_stop_level = max(self.trailing_stop_level, new_stop)
        else:  # Short position
            new_stop = current_price * (1 + trailing_distance)
            self.trailing_stop_level = min(self.trailing_stop_level, new_stop)
    
    def _exit_position(self, timestamp, price, reason, row_ref):
        """Exit the current position."""
        if self.position_entry_price:
            pnl_pct = (price - self.position_entry_price) / self.position_entry_price
            if self.current_position == -1:
                pnl_pct = -pnl_pct
            row_ref['position_pnl'] = pnl_pct
        
        row_ref['exit_reason'] = reason
        row_ref['risk_managed_signal'] = -self.current_position  # Exit signal
        
        self.logger.debug(f"Exited position at {price}, reason: {reason}")
        
        # Reset position state
        self.current_position = 0
        self.position_entry_price = None
        self.position_entry_date = None
        self.trailing_stop_level = None


class VolumeFilterOverlay(StrategyDecorator):
    """Strategy overlay that filters signals based on volume confirmation.
    
    This overlay only allows signals when volume confirms the price movement,
    helping to filter out weak signals that lack institutional support.
    """
    
    PARAMETER_SCHEMA = ParameterSchema({
        'volume_ma_period': ParameterDefinition(
            default=20,
            type=int,
            range=(5, 100),
            description="Volume moving average period",
            category="Volume Filter"
        ),
        'volume_multiplier': ParameterDefinition(
            default=1.5,
            type=float,
            range=(1.0, 5.0),
            description="Minimum volume relative to average",
            category="Volume Filter"
        ),
        'require_volume_trend': ParameterDefinition(
            default=True,
            type=bool,
            description="Require increasing volume trend",
            category="Volume Filter"
        )
    }, strategy_name="Volume Filter Overlay")
    
    def __init__(self, 
                 wrapped_strategy: EnhancedBaseStrategy,
                 parameters: Optional[Dict[str, Any]] = None):
        """Initialize volume filter overlay."""
        super().__init__(wrapped_strategy)
        
        user_params = parameters or {}
        self.overlay_parameters = self.PARAMETER_SCHEMA.validate(user_params)
        
        self.logger.info("Initialized volume filter overlay")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-filtered signals."""
        # Check if volume data is available
        if 'volume' not in data.columns:
            self.logger.warning("Volume data not available - passing through unfiltered signals")
            return self.wrapped_strategy.generate_signals(data)
        
        # Generate base signals
        signals_df = self.wrapped_strategy.generate_signals(data)
        
        # Calculate volume metrics
        volume_ma_period = self.overlay_parameters['volume_ma_period']
        volume_multiplier = self.overlay_parameters['volume_multiplier']
        
        volume_ma = data['volume'].rolling(volume_ma_period).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Volume confirmation filter
        volume_confirmed = volume_ratio > volume_multiplier
        
        # Optional volume trend filter
        if self.overlay_parameters['require_volume_trend']:
            volume_trend = data['volume'] > data['volume'].shift(1)
            volume_confirmed = volume_confirmed & volume_trend
        
        # Apply volume filter
        original_signals = signals_df['signal'].copy()
        signals_df.loc[~volume_confirmed, 'signal'] = 0
        
        # Add analysis columns
        signals_df['volume_ratio'] = volume_ratio
        signals_df['volume_confirmed'] = volume_confirmed
        signals_df['volume_filtered'] = original_signals != signals_df['signal']
        
        # Log filtering results
        total_signals = (original_signals != 0).sum()
        filtered_signals = (signals_df['signal'] != 0).sum()
        self.logger.info(f"Volume filter: kept {filtered_signals}/{total_signals} signals")
        
        return signals_df


class TimeFilterOverlay(StrategyDecorator):
    """Strategy overlay for time-based trading restrictions.
    
    This overlay allows trading only during specified time periods, useful for
    avoiding market open/close volatility or trading only during high-liquidity hours.
    """
    
    PARAMETER_SCHEMA = ParameterSchema({
        'allowed_hours': ParameterDefinition(
            default=[9, 10, 11, 14, 15],
            type=list,
            description="Hours of day when trading is allowed",
            category="Time Filter"
        ),
        'avoid_market_open_minutes': ParameterDefinition(
            default=30,
            type=int,
            range=(0, 120),
            description="Minutes to avoid after market open",
            category="Time Filter"
        ),
        'avoid_market_close_minutes': ParameterDefinition(
            default=30,
            type=int,
            range=(0, 120),
            description="Minutes to avoid before market close",
            category="Time Filter"
        ),
        'allowed_weekdays': ParameterDefinition(
            default=[0, 1, 2, 3, 4],  # Monday=0 to Friday=4
            type=list,
            description="Weekdays when trading is allowed (0=Monday)",
            category="Time Filter"
        )
    }, strategy_name="Time Filter Overlay")
    
    def __init__(self, 
                 wrapped_strategy: EnhancedBaseStrategy,
                 parameters: Optional[Dict[str, Any]] = None):
        """Initialize time filter overlay."""
        super().__init__(wrapped_strategy)
        
        user_params = parameters or {}
        self.overlay_parameters = self.PARAMETER_SCHEMA.validate(user_params)
        
        self.logger.info("Initialized time filter overlay")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-filtered signals."""
        # Generate base signals
        signals_df = self.wrapped_strategy.generate_signals(data)
        
        # Apply time filters
        time_allowed = self._calculate_time_filter(signals_df.index)
        
        # Filter signals
        original_signals = signals_df['signal'].copy()
        signals_df.loc[~time_allowed, 'signal'] = 0
        
        # Add analysis columns
        signals_df['time_allowed'] = time_allowed
        signals_df['time_filtered'] = original_signals != signals_df['signal']
        
        # Log filtering results
        total_signals = (original_signals != 0).sum()
        filtered_signals = (signals_df['signal'] != 0).sum()
        self.logger.info(f"Time filter: kept {filtered_signals}/{total_signals} signals")
        
        return signals_df
    
    def _calculate_time_filter(self, index: pd.DatetimeIndex) -> pd.Series:
        """Calculate time-based filter mask."""
        time_allowed = pd.Series(True, index=index)
        
        # Filter by hour of day
        allowed_hours = self.overlay_parameters['allowed_hours']
        if allowed_hours:
            hour_mask = index.hour.isin(allowed_hours)
            time_allowed = time_allowed & hour_mask
        
        # Filter by weekday
        allowed_weekdays = self.overlay_parameters['allowed_weekdays']
        if allowed_weekdays:
            weekday_mask = index.weekday.isin(allowed_weekdays)
            time_allowed = time_allowed & weekday_mask
        
        # TODO: Add market open/close filtering (requires market calendar)
        # This would need integration with trading calendars for different exchanges
        
        return time_allowed