"""Regime Switching Logic.

Phase 5: Wrapper/filter to only let signals fire when regime is "on".
Can be used as decorator or wrapper class to "regime-gate" any signal.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from .base import BaseStrategy
import logging


class RegimeDetector:
    """Detect market regime using various methods."""
    
    def __init__(self, method: str = 'sma_slope', parameters: Dict[str, Any] = None):
        """Initialize regime detector.
        
        Args:
            method: Detection method ('sma_slope', 'atr_volatility', 'vix', 'combined')
            parameters: Method-specific parameters
        """
        self.method = method
        self.parameters = parameters or {}
        self.logger = logging.getLogger(__name__)
        
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regime.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Boolean Series where True = favorable regime, False = unfavorable
        """
        if self.method == 'sma_slope':
            return self._sma_slope_regime(data)
        elif self.method == 'atr_volatility':
            return self._atr_volatility_regime(data)
        elif self.method == 'vix':
            return self._vix_regime(data)
        elif self.method == 'combined':
            return self._combined_regime(data)
        else:
            self.logger.warning(f"Unknown regime method: {self.method}")
            return pd.Series(True, index=data.index)  # Default to all favorable
    
    def _sma_slope_regime(self, data: pd.DataFrame) -> pd.Series:
        """Regime based on SMA slope.
        
        Favorable regime when SMA is trending up (positive slope).
        """
        sma_period = self.parameters.get('sma_period', 200)
        slope_threshold = self.parameters.get('slope_threshold', 0.001)
        
        sma = data['close'].rolling(window=sma_period).mean()
        
        # Calculate slope (rate of change)
        sma_slope = sma.pct_change(periods=5)  # 5-day slope
        
        favorable_regime = sma_slope > slope_threshold
        
        self.logger.info(f"SMA Slope Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime
    
    def _atr_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Regime based on ATR volatility.
        
        Favorable regime when volatility is moderate (not too high or too low).
        """
        atr_period = self.parameters.get('atr_period', 14)
        atr_lookback = self.parameters.get('atr_lookback', 50)
        low_vol_percentile = self.parameters.get('low_vol_percentile', 20)
        high_vol_percentile = self.parameters.get('high_vol_percentile', 80)
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()
        
        # Calculate percentiles of ATR over lookback period
        atr_percentile = atr.rolling(window=atr_lookback).rank(pct=True) * 100
        
        # Favorable when volatility is moderate
        favorable_regime = (atr_percentile > low_vol_percentile) & (atr_percentile < high_vol_percentile)
        
        self.logger.info(f"ATR Volatility Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime
    
    def _vix_regime(self, data: pd.DataFrame) -> pd.Series:
        """Regime based on VIX levels.
        
        Requires VIX data in the DataFrame.
        Favorable regime when VIX is below threshold.
        """
        vix_threshold = self.parameters.get('vix_threshold', 25.0)
        
        if 'vix' not in data.columns:
            self.logger.warning("VIX column not found, using mock VIX based on price volatility")
            # Create mock VIX from price volatility
            returns = data['close'].pct_change()
            mock_vix = returns.rolling(window=20).std() * np.sqrt(252) * 100
            favorable_regime = mock_vix < vix_threshold
        else:
            favorable_regime = data['vix'] < vix_threshold
        
        self.logger.info(f"VIX Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime
    
    def _combined_regime(self, data: pd.DataFrame) -> pd.Series:
        """Combined regime using multiple indicators."""
        # Get individual regime signals
        sma_regime = self._sma_slope_regime(data)
        atr_regime = self._atr_volatility_regime(data)
        vix_regime = self._vix_regime(data)
        
        # Require majority consensus
        regime_votes = sma_regime.astype(int) + atr_regime.astype(int) + vix_regime.astype(int)
        favorable_regime = regime_votes >= 2
        
        self.logger.info(f"Combined Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime


class RegimeGatedStrategy(BaseStrategy):
    """Wrapper that applies regime filter to any strategy."""
    
    def __init__(self, base_strategy: BaseStrategy, regime_detector: RegimeDetector):
        """Initialize regime-gated strategy.
        
        Args:
            base_strategy: The underlying strategy to wrap
            regime_detector: Regime detection instance
        """
        self.base_strategy = base_strategy
        self.regime_detector = regime_detector
        
        # Combine names
        name = f"Regime-Gated {base_strategy.name}"
        parameters = {
            'base_strategy': base_strategy.get_info(),
            'regime_method': regime_detector.method,
            'regime_parameters': regime_detector.parameters
        }
        
        super().__init__(name, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-gated signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime-filtered signals
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Get base strategy signals
        df = self.base_strategy.generate_signals(data)
        
        # Detect regime
        favorable_regime = self.regime_detector.detect_regime(data)
        
        # Store original signals for analysis
        df['base_signal'] = df['signal'].copy()
        df['regime_favorable'] = favorable_regime
        
        # Apply regime filter - only allow signals in favorable regime
        df.loc[~favorable_regime, 'signal'] = 0
        
        # Count filtered signals
        original_buys = (df['base_signal'] == 1).sum()
        original_sells = (df['base_signal'] == -1).sum()
        filtered_buys = (df['signal'] == 1).sum()
        filtered_sells = (df['signal'] == -1).sum()
        
        self.logger.info(f"Regime filter: {original_buys} -> {filtered_buys} buys, "
                        f"{original_sells} -> {filtered_sells} sells")
        
        return df


class RegimeSwitchStrategy(BaseStrategy):
    """Strategy that switches between different strategies based on regime."""
    
    def __init__(self, 
                 trend_strategy: BaseStrategy,
                 mean_revert_strategy: BaseStrategy,
                 regime_detector: RegimeDetector):
        """Initialize regime-switching strategy.
        
        Args:
            trend_strategy: Strategy to use in trending markets
            mean_revert_strategy: Strategy to use in ranging markets
            regime_detector: Regime detection instance
        """
        self.trend_strategy = trend_strategy
        self.mean_revert_strategy = mean_revert_strategy
        self.regime_detector = regime_detector
        
        name = f"Regime Switch ({trend_strategy.name} / {mean_revert_strategy.name})"
        parameters = {
            'trend_strategy': trend_strategy.get_info(),
            'mean_revert_strategy': mean_revert_strategy.get_info(),
            'regime_detector': {
                'method': regime_detector.method,
                'parameters': regime_detector.parameters
            }
        }
        
        super().__init__(name, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-switching signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime-switched signals
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Generate signals from both strategies
        trend_df = self.trend_strategy.generate_signals(data)
        mean_revert_df = self.mean_revert_strategy.generate_signals(data)
        
        # Detect regime
        favorable_regime = self.regime_detector.detect_regime(data)
        
        # Start with base data
        df = data.copy()
        df['signal'] = 0
        
        # Store individual signals for analysis
        df['trend_signal'] = trend_df['signal']
        df['mean_revert_signal'] = mean_revert_df['signal']
        df['regime_favorable'] = favorable_regime
        
        # Apply regime switching logic
        # Favorable regime = use trend strategy
        # Unfavorable regime = use mean reversion strategy
        df.loc[favorable_regime, 'signal'] = trend_df.loc[favorable_regime, 'signal']
        df.loc[~favorable_regime, 'signal'] = mean_revert_df.loc[~favorable_regime, 'signal']
        
        # Add regime transition markers
        regime_change = favorable_regime != favorable_regime.shift(1)
        df['regime_change'] = regime_change
        
        trend_periods = favorable_regime.sum()
        mean_revert_periods = (~favorable_regime).sum()
        total_periods = len(favorable_regime)
        
        self.logger.info(f"Regime switching: {trend_periods}/{total_periods} trend periods "
                        f"({trend_periods/total_periods:.1%}), "
                        f"{mean_revert_periods}/{total_periods} mean-revert periods "
                        f"({mean_revert_periods/total_periods:.1%})")
        
        return df


def regime_gate_decorator(regime_detector: RegimeDetector):
    """Decorator to add regime gating to any strategy.
    
    Args:
        regime_detector: Regime detection instance
        
    Returns:
        Decorator function
    """
    def decorator(strategy_class):
        class RegimeGatedStrategyClass(strategy_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.regime_detector = regime_detector
                self.name = f"Regime-Gated {self.name}"
            
            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                # Get base signals
                df = super().generate_signals(data)
                
                # Apply regime filter
                favorable_regime = self.regime_detector.detect_regime(data)
                df['base_signal'] = df['signal'].copy()
                df['regime_favorable'] = favorable_regime
                df.loc[~favorable_regime, 'signal'] = 0
                
                return df
        
        return RegimeGatedStrategyClass
    
    return decorator


# Example usage functions
def create_regime_gated_strategy(base_strategy: BaseStrategy, 
                                regime_method: str = 'sma_slope',
                                regime_params: Dict[str, Any] = None) -> RegimeGatedStrategy:
    """Helper function to create regime-gated strategy.
    
    Args:
        base_strategy: Strategy to wrap
        regime_method: Regime detection method
        regime_params: Regime detection parameters
        
    Returns:
        Regime-gated strategy instance
    """
    detector = RegimeDetector(regime_method, regime_params)
    return RegimeGatedStrategy(base_strategy, detector)


def create_regime_switch_strategy(trend_strategy: BaseStrategy,
                                 mean_revert_strategy: BaseStrategy,
                                 regime_method: str = 'sma_slope',
                                 regime_params: Dict[str, Any] = None) -> RegimeSwitchStrategy:
    """Helper function to create regime-switching strategy.
    
    Args:
        trend_strategy: Strategy for trending markets
        mean_revert_strategy: Strategy for ranging markets
        regime_method: Regime detection method
        regime_params: Regime detection parameters
        
    Returns:
        Regime-switching strategy instance
    """
    detector = RegimeDetector(regime_method, regime_params)
    return RegimeSwitchStrategy(trend_strategy, mean_revert_strategy, detector) 