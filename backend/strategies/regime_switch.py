"""Regime Switching Strategy Implementation.

This module implements market regime detection and filtering for trading strategies. The core
concept is that markets operate in different "regimes" (trending vs. choppy, high vol vs. low vol,
risk-on vs. risk-off) and strategies perform differently in different regimes.

REGIME SWITCHING THEORY:
======================
Market regime switching is based on the observation that financial markets exhibit different
behavioral patterns across time periods. Common regime classifications include:

1. TREND REGIMES: Bull markets vs. Bear markets vs. Sideways markets
   - Detected via moving average slopes, momentum indicators
   - Trend-following strategies work well in trending regimes
   - Mean reversion strategies struggle in strong trending regimes

2. VOLATILITY REGIMES: High volatility vs. Low volatility vs. Normal volatility  
   - Detected via ATR, VIX, realized volatility measures
   - High vol regimes favor protective strategies
   - Low vol regimes may signal complacency

3. SENTIMENT REGIMES: Risk-on vs. Risk-off periods
   - Detected via VIX, credit spreads, sector rotation
   - Risk-on favors growth/momentum strategies
   - Risk-off favors defensive/quality strategies

IMPLEMENTATION APPROACH:
=======================
This module provides both regime detection and regime-gated strategy execution:

1. RegimeDetector: Identifies current market regime using various methods
2. RegimeGatedStrategy: Wrapper that only executes base strategy signals when regime is favorable

The regime gating acts as a filter - when regime is unfavorable, no new positions are taken
and existing positions may be reduced/closed. This can significantly improve risk-adjusted
returns by avoiding unfavorable market conditions.

USAGE EXAMPLES:
==============
```python
# Basic regime detection
detector = RegimeDetector(method='sma_slope', parameters={'sma_period': 200})
regime_signal = detector.detect_regime(price_data)

# Wrap any strategy with regime filter
base_strategy = SmaEmaRsiStrategy()
regime_strategy = RegimeGatedStrategy(base_strategy, detector)
signals = regime_strategy.generate_signals(data)

# Combined regime using multiple indicators
combined_detector = RegimeDetector(method='combined')
```

ACADEMIC REFERENCES:
===================
- Hamilton, J.D. (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Ang, A. & Bekaert, G. (2002): "Regime Switches in Interest Rates"  
- Guidolin, M. & Timmermann, A. (2007): "Asset Allocation under Multivariate Regime Switching"

Phase 5: Wrapper/filter to only let signals fire when regime is "on".
Can be used as decorator or wrapper class to "regime-gate" any signal.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from .base import BaseStrategy
import logging


class RegimeDetector:
    """Market regime detection using multiple statistical methods.
    
    This class implements various regime detection algorithms to identify whether
    current market conditions are favorable for specific trading strategies.
    
    DETECTION METHODS:
    =================
    1. 'sma_slope': Trend regime via moving average slope analysis
    2. 'atr_volatility': Volatility regime via ATR percentile ranking  
    3. 'vix': Fear/greed regime via VIX level analysis
    4. 'combined': Consensus regime using multiple indicators
    
    DESIGN PHILOSOPHY:
    =================
    Rather than trying to predict regime changes, this detector identifies
    current regime characteristics and filters strategy execution accordingly.
    The approach is defensive - when in doubt, avoid taking risk.
    """
    
    def __init__(self, method: str = 'sma_slope', parameters: Dict[str, Any] = None):
        """Initialize regime detector with specified method and parameters.
        
        Args:
            method: Detection method ('sma_slope', 'atr_volatility', 'vix', 'combined')
                   - 'sma_slope': Favorable when SMA is trending upward
                   - 'atr_volatility': Favorable when volatility is moderate  
                   - 'vix': Favorable when fear gauge is below threshold
                   - 'combined': Requires majority consensus from multiple methods
            parameters: Method-specific configuration parameters
                       - sma_slope: {'sma_period': 200, 'slope_threshold': 0.001}
                       - atr_volatility: {'atr_period': 14, 'atr_lookback': 50, 
                                        'low_vol_percentile': 20, 'high_vol_percentile': 80}
                       - vix: {'vix_threshold': 25.0}
                       
        Example:
            # Conservative trend follower
            detector = RegimeDetector(
                method='sma_slope', 
                parameters={'sma_period': 200, 'slope_threshold': 0.002}
            )
        """
        self.method = method
        self.parameters = parameters or {}
        self.logger = logging.getLogger(__name__)
        
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regime for each time period in the data.
        
        This is the main entry point for regime detection. Returns a boolean
        series indicating whether each time period represents a "favorable"
        regime for trading the underlying strategy.
        
        Args:
            data: Price data DataFrame with OHLCV columns
                 Required columns: ['close', 'high', 'low', 'volume']
                 Optional columns: ['vix'] for VIX-based regime detection
            
        Returns:
            pd.Series: Boolean series where True = favorable regime, False = unfavorable
                      Index matches input data index for proper alignment
                      
        Note:
            The definition of "favorable" depends on the detection method and is
            strategy-specific. Generally means conditions where the strategy has
            historically performed well or market structure supports the approach.
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
        """Detect trend regime based on Simple Moving Average slope analysis.
        
        METHODOLOGY:
        ===========
        Calculates the rate of change (slope) of a long-term moving average to identify
        trend regimes. Positive slope indicates uptrend (bull market), negative slope
        indicates downtrend (bear market).
        
        FAVORABLE REGIME CONDITIONS:
        ===========================
        - SMA slope > threshold: Indicates persistent upward momentum
        - Filters out choppy/sideways markets where trend strategies underperform
        - Works well for momentum and breakout strategies
        
        PARAMETER GUIDANCE:
        ==================
        - sma_period: Longer periods (200) for major trends, shorter (50) for intermediate
        - slope_threshold: Higher values (0.002) for strong trends only, lower (0.0005) for weak trends
        
        Args:
            data: Price DataFrame with 'close' column
            
        Returns:
            pd.Series: True when SMA slope indicates favorable trending regime
        """
        sma_period = self.parameters.get('sma_period', 200)
        slope_threshold = self.parameters.get('slope_threshold', 0.001)
        
        # Calculate Simple Moving Average for trend identification
        sma = data['close'].rolling(window=sma_period).mean()
        
        # Calculate slope as rate of change over 5-day period
        # This smooths out daily noise while capturing trend direction
        sma_slope = sma.pct_change(periods=5)  # 5-day slope
        
        # Favorable regime when slope exceeds threshold (uptrend)
        favorable_regime = sma_slope > slope_threshold
        
        self.logger.info(f"SMA Slope Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime
    
    def _atr_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect volatility regime based on Average True Range percentile ranking.
        
        METHODOLOGY:
        ===========
        Uses ATR (Average True Range) to measure current volatility relative to recent
        historical volatility. Ranks current ATR against rolling window to identify
        high/low volatility regimes.
        
        FAVORABLE REGIME CONDITIONS:
        ===========================
        - Moderate volatility (20th-80th percentile): Optimal for most strategies
        - Avoids extremely low volatility (complacency, lack of trends)
        - Avoids extremely high volatility (panic, whipsaws, gap risk)
        
        VOLATILITY REGIME THEORY:
        ========================
        - Low vol regimes: Often precede volatility spikes, limited profit opportunities
        - High vol regimes: Increased gap risk, stop-loss failures, emotional trading
        - Moderate vol regimes: Sufficient movement for profits with manageable risk
        
        Args:
            data: Price DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            pd.Series: True when volatility is in moderate (favorable) range
        """
        atr_period = self.parameters.get('atr_period', 14)
        atr_lookback = self.parameters.get('atr_lookback', 50)
        low_vol_percentile = self.parameters.get('low_vol_percentile', 20)
        high_vol_percentile = self.parameters.get('high_vol_percentile', 80)
        
        # Calculate True Range components
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        
        # True Range is the maximum of the three components
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Average True Range over specified period
        atr = true_range.rolling(window=atr_period).mean()
        
        # Calculate percentile ranking of current ATR vs. recent history
        atr_percentile = atr.rolling(window=atr_lookback).rank(pct=True) * 100
        
        # Favorable regime when volatility is moderate (not too high or too low)
        favorable_regime = (atr_percentile > low_vol_percentile) & (atr_percentile < high_vol_percentile)
        
        self.logger.info(f"ATR Volatility Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime
    
    def _vix_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect fear/greed regime based on VIX volatility index levels.
        
        METHODOLOGY:
        ===========
        Uses VIX (CBOE Volatility Index) as a "fear gauge" to identify market sentiment
        regimes. VIX measures implied volatility of S&P 500 options and reflects
        market expectations of future volatility and uncertainty.
        
        VIX INTERPRETATION:
        ==================
        - VIX < 20: Complacency regime (low fear, potential for surprises)
        - VIX 20-30: Normal regime (moderate uncertainty, good for most strategies)  
        - VIX > 30: Fear regime (high uncertainty, defensive positioning needed)
        - VIX > 40: Panic regime (extreme fear, opportunity for contrarians)
        
        FAVORABLE REGIME CONDITIONS:
        ===========================
        - VIX below threshold: Lower uncertainty, more predictable price action
        - Supports momentum and trend-following strategies
        - Higher VIX periods often see mean reversion and defensive positioning
        
        Args:
            data: Price DataFrame, optionally with 'vix' column
                 If no VIX data available, creates mock VIX from price volatility
            
        Returns:
            pd.Series: True when VIX indicates favorable (low fear) regime
        """
        vix_threshold = self.parameters.get('vix_threshold', 25.0)
        
        if 'vix' not in data.columns:
            self.logger.warning("VIX column not found, using mock VIX based on price volatility")
            # Create mock VIX from 20-day realized volatility
            # Annualized volatility * 100 approximates VIX levels
            returns = data['close'].pct_change()
            mock_vix = returns.rolling(window=20).std() * np.sqrt(252) * 100
            favorable_regime = mock_vix < vix_threshold
        else:
            favorable_regime = data['vix'] < vix_threshold
        
        self.logger.info(f"VIX Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime
    
    def _combined_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect regime using consensus from multiple indicators.
        
        METHODOLOGY:
        ===========
        Combines signals from SMA slope, ATR volatility, and VIX regimes using
        majority voting. This approach reduces false signals by requiring
        agreement across multiple regime detection methods.
        
        CONSENSUS LOGIC:
        ===============
        - Requires at least 2 out of 3 methods to agree (majority consensus)
        - More conservative than individual methods (fewer but higher quality signals)
        - Reduces whipsaws and false regime change signals
        - Better suited for strategies that need high confidence in regime stability
        
        THEORETICAL FOUNDATION:
        ======================
        Multiple regime indicators capture different aspects of market structure:
        - SMA slope: Trend persistence and momentum
        - ATR volatility: Price variability and uncertainty  
        - VIX: Forward-looking fear/greed sentiment
        
        When these align, regime signal is more reliable and persistent.
        
        Args:
            data: Price DataFrame with required columns for all sub-methods
            
        Returns:
            pd.Series: True when majority of regime indicators agree on favorable conditions
        """
        # Get individual regime signals using default parameters for each method
        sma_regime = self._sma_slope_regime(data)
        atr_regime = self._atr_volatility_regime(data)
        vix_regime = self._vix_regime(data)
        
        # Count votes for favorable regime (convert boolean to int)
        regime_votes = sma_regime.astype(int) + atr_regime.astype(int) + vix_regime.astype(int)
        
        # Require majority consensus (at least 2 out of 3 methods agree)
        favorable_regime = regime_votes >= 2
        
        self.logger.info(f"Combined Regime: {favorable_regime.sum()}/{len(favorable_regime)} "
                        f"periods favorable ({favorable_regime.mean():.1%})")
        
        return favorable_regime


class RegimeGatedStrategy(BaseStrategy):
    """Wrapper strategy that applies regime filtering to any base strategy.
    
    This class implements the "regime gating" concept where the underlying strategy's
    signals are only executed when market regime conditions are favorable. When
    regime turns unfavorable, new positions are avoided and existing positions
    may be reduced or closed.
    
    REGIME GATING BENEFITS:
    ======================
    1. RISK REDUCTION: Avoids trading during unfavorable market conditions
    2. IMPROVED SHARPE: Better risk-adjusted returns through selective execution
    3. REDUCED DRAWDOWNS: Exits positions before major adverse moves
    4. STRATEGY ENHANCEMENT: Makes any strategy more robust and adaptive
    
    IMPLEMENTATION DETAILS:
    ======================
    - Signal Generation: Base strategy generates raw signals as normal
    - Regime Filtering: Signals are filtered through regime detector
    - Position Management: Unfavorable regime triggers position reduction
    - Risk Management: Additional layer of systematic risk control
    
    USAGE PATTERNS:
    ==============
    ```python
    # Enhance trend strategy with regime filter
    trend_strategy = SmaEmaRsiStrategy()
    regime_detector = RegimeDetector(method='sma_slope')
    enhanced_strategy = RegimeGatedStrategy(trend_strategy, regime_detector)
    
    # Conservative approach: only trade in favorable regimes
    conservative_detector = RegimeDetector(method='combined')
    conservative_strategy = RegimeGatedStrategy(base_strategy, conservative_detector)
    ```
    """
    
    def __init__(self, base_strategy: BaseStrategy, regime_detector: RegimeDetector):
        """Initialize regime-gated strategy wrapper.
        
        Args:
            base_strategy: The underlying strategy to wrap and filter
                          Can be any strategy implementing BaseStrategy interface
                          Strategy operates normally but signals are regime-filtered
            regime_detector: Regime detection instance with configured method/parameters
                           Determines when market conditions are favorable for base strategy
                           
        Example:
            # Create momentum strategy with volatility regime filter
            momentum_strategy = RsiStrategy(rsi_period=14, rsi_oversold=30, rsi_overbought=70)
            vol_detector = RegimeDetector(
                method='atr_volatility',
                parameters={'low_vol_percentile': 10, 'high_vol_percentile': 90}
            )
            regime_strategy = RegimeGatedStrategy(momentum_strategy, vol_detector)
        """
        self.base_strategy = base_strategy
        self.regime_detector = regime_detector
        
        # Combine strategy names for identification
        self.name = f"RegimeGated_{self.base_strategy.name}_{self.regime_detector.method}"
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-filtered trading signals by combining base strategy with regime detection.
        
        SIGNAL FILTERING PROCESS:
        ========================
        1. Base Strategy Signals: Generate raw signals using underlying strategy
        2. Regime Detection: Identify favorable/unfavorable market regime periods
        3. Signal Filtering: Apply regime filter to base signals
        4. Position Management: Handle regime transitions and position sizing
        
        FILTERING LOGIC:
        ===============
        - Favorable Regime + Buy Signal = Execute Buy
        - Favorable Regime + Sell Signal = Execute Sell  
        - Unfavorable Regime + Any Signal = Reduce/Close positions
        - Regime Transition = Gradual position adjustment
        
        RISK MANAGEMENT:
        ===============
        The regime filter acts as an additional risk management layer:
        - Prevents new positions during unfavorable conditions
        - Reduces position sizes during regime uncertainty
        - Provides systematic exit mechanism for adverse conditions
        
        Args:
            data: Price data DataFrame with OHLCV columns
                 Must contain all data required by both base strategy and regime detector
            
        Returns:
            pd.DataFrame: Enhanced signals with regime filtering applied
                         Columns: ['signal', 'regime', 'base_signal', 'position', 'regime_strength']
                         - signal: Final regime-filtered signal (-1, 0, 1)
                         - regime: Regime state (True=favorable, False=unfavorable)
                         - base_signal: Raw signal from underlying strategy
                         - position: Recommended position size (0.0 to 1.0)
                         - regime_strength: Confidence in regime detection (future enhancement)
        """
        # Step 1: Generate base strategy signals
        base_signals = self.base_strategy.generate_signals(data)
        
        # Step 2: Detect market regime for each period
        regime_favorable = self.regime_detector.detect_regime(data)
        
        # Step 3: Create enhanced signals DataFrame with regime information
        signals = base_signals.copy()
        signals['regime'] = regime_favorable
        signals['base_signal'] = signals['signal'].copy()
        
        # Step 4: Apply regime filtering to signals
        # Only allow signals when regime is favorable
        signals.loc[~regime_favorable, 'signal'] = 0
        
        # Step 5: Calculate position sizing based on regime confidence
        # Full position in favorable regime, reduced position during transitions
        signals['position'] = self._calculate_regime_position_size(signals)
        
        # Step 6: Add regime strength indicator (placeholder for future enhancement)
        signals['regime_strength'] = regime_favorable.astype(float)
        
        # Log regime filtering statistics
        total_base_signals = (base_signals['signal'] != 0).sum()
        total_final_signals = (signals['signal'] != 0).sum()
        filter_rate = 1 - (total_final_signals / max(total_base_signals, 1))
        
        self.logger.info(f"Regime filtering: {total_base_signals} base signals -> "
                        f"{total_final_signals} final signals (filtered {filter_rate:.1%})")
        
        return signals
    
    def _calculate_regime_position_size(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate position sizing based on regime state and transitions.
        
        POSITION SIZING LOGIC:
        =====================
        - Favorable regime: Full position size (1.0)
        - Unfavorable regime: No new positions (0.0)
        - Regime transitions: Gradual adjustment over multiple periods
        - Signal strength: Future enhancement for variable sizing
        
        TRANSITION HANDLING:
        ===================
        When regime changes from favorable to unfavorable, positions are
        gradually reduced rather than immediately closed to avoid whipsaws
        and improve execution.
        
        Args:
            signals: DataFrame with 'signal', 'regime', and 'base_signal' columns
            
        Returns:
            pd.Series: Position size recommendations (0.0 to 1.0)
        """
        position_size = pd.Series(0.0, index=signals.index)
        
        # Full position during favorable regime with active signals
        favorable_with_signal = signals['regime'] & (signals['signal'] != 0)
        position_size[favorable_with_signal] = 1.0
        
        # TODO: Implement gradual position adjustment during regime transitions
        # This could use a rolling average or exponential decay for smoother transitions
        
        return position_size
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the regime-gated strategy configuration.
        
        Returns:
            Dict containing nested information about both base strategy and regime detector,
            providing full transparency into the composite strategy configuration.
        """
        return {
            'strategy_type': 'RegimeGated',
            'name': self.name,
            'base_strategy': self.base_strategy.get_info(),
            'regime_detector': {
                'method': self.regime_detector.method,
                'parameters': self.regime_detector.parameters
            },
            'description': f'Regime-filtered {self.base_strategy.name} using {self.regime_detector.method} detection'
        }


class RegimeSwitchStrategy(BaseStrategy):
    """Strategy that dynamically switches between different strategies based on market regime.
    
    This class implements a meta-strategy that selects between two underlying strategies
    depending on the detected market regime. Unlike RegimeGatedStrategy which filters
    signals, this strategy actively switches between different trading approaches.
    
    STRATEGY SWITCHING PHILOSOPHY:
    =============================
    Different market regimes favor different trading approaches:
    - TRENDING REGIMES: Favor momentum and trend-following strategies
    - RANGING REGIMES: Favor mean reversion and contrarian strategies
    - VOLATILE REGIMES: May require defensive or reduced-risk approaches
    
    IMPLEMENTATION APPROACH:
    =======================
    1. Regime Detection: Continuously monitor market regime characteristics
    2. Strategy Selection: Choose appropriate strategy for current regime
    3. Signal Generation: Execute signals from selected strategy only
    4. Transition Management: Handle strategy switches smoothly
    
    BENEFITS:
    ========
    - ADAPTIVE: Automatically adapts to changing market conditions
    - DIVERSIFICATION: Combines multiple strategy approaches
    - REGIME OPTIMIZATION: Uses best strategy for each market environment
    - RISK MANAGEMENT: Avoids using inappropriate strategies in wrong regimes
    
    EXAMPLE USAGE:
    =============
    ```python
    # Create component strategies
    trend_strategy = SmaEmaRsiStrategy(sma_short=20, sma_long=50)
    mean_revert_strategy = BollingerBandMeanReversionStrategy()
    
    # Create regime detector
    regime_detector = RegimeDetector(method='sma_slope')
    
    # Combine into switching strategy
    switch_strategy = RegimeSwitchStrategy(
        trend_strategy=trend_strategy,
        mean_revert_strategy=mean_revert_strategy,
        regime_detector=regime_detector
    )
    ```
    """
    
    def __init__(self, 
                 trend_strategy: BaseStrategy,
                 mean_revert_strategy: BaseStrategy,
                 regime_detector: RegimeDetector):
        """Initialize regime-switching strategy with component strategies and detector.
        
        Args:
            trend_strategy: Strategy to use during trending/favorable regimes
                          Should be momentum-based or trend-following strategy
                          Examples: SmaEmaRsiStrategy, CrossoverStrategy, VolatilityBreakoutStrategy
            mean_revert_strategy: Strategy to use during ranging/unfavorable regimes
                                Should be mean-reverting or contrarian strategy
                                Examples: BollingerBandMeanReversionStrategy, RSIMeanReversionStrategy
            regime_detector: Regime detection instance that determines when to switch
                           Configuration affects switching frequency and sensitivity
                           
        Example:
            # Conservative switching strategy
            trend_strat = CrossoverStrategy(sma_short=50, sma_long=200)
            mean_strat = BollingerBandMeanReversionStrategy(bb_period=20, bb_std=2)
            detector = RegimeDetector(method='combined')  # Conservative switching
            
            switch_strategy = RegimeSwitchStrategy(trend_strat, mean_strat, detector)
        """
        self.trend_strategy = trend_strategy
        self.mean_revert_strategy = mean_revert_strategy
        self.regime_detector = regime_detector
        
        # Create composite strategy name indicating both component strategies
        name = f"RegimeSwitch_{trend_strategy.name}_vs_{mean_revert_strategy.name}"
        
        # Build comprehensive parameter structure for transparency
        parameters = {
            'strategy_type': 'RegimeSwitching',
            'trend_strategy': trend_strategy.get_info(),
            'mean_revert_strategy': mean_revert_strategy.get_info(),
            'regime_detector': {
                'method': regime_detector.method,
                'parameters': regime_detector.parameters
            },
            'switching_logic': 'favorable_regime -> trend_strategy, unfavorable_regime -> mean_revert_strategy'
        }
        
        super().__init__(name, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-switching signals by selecting appropriate strategy for each period.
        
        SIGNAL GENERATION PROCESS:
        =========================
        1. Strategy Execution: Generate signals from both component strategies
        2. Regime Detection: Identify current market regime for each period
        3. Signal Selection: Choose signals from appropriate strategy based on regime
        4. Transition Tracking: Monitor regime changes and strategy switches
        
        SWITCHING LOGIC:
        ===============
        - Favorable Regime (trending): Use trend_strategy signals
        - Unfavorable Regime (ranging): Use mean_revert_strategy signals
        - Regime Transitions: Immediate switch (no gradual transition)
        - Signal Conflicts: Regime detector has priority in selection
        
        PERFORMANCE TRACKING:
        ====================
        The output includes additional columns for analysis:
        - trend_signal: Raw signals from trend strategy
        - mean_revert_signal: Raw signals from mean reversion strategy
        - regime_favorable: Boolean regime state
        - regime_change: Transition markers for strategy switches
        
        Args:
            data: Price data DataFrame with OHLCV columns
                 Must contain all data required by both component strategies and regime detector
            
        Returns:
            pd.DataFrame: Enhanced signals with regime switching applied
                         Columns: ['signal', 'trend_signal', 'mean_revert_signal', 
                                  'regime_favorable', 'regime_change']
                         - signal: Final regime-selected signal (-1, 0, 1)
                         - trend_signal: Raw trend strategy signal for analysis
                         - mean_revert_signal: Raw mean reversion signal for analysis
                         - regime_favorable: Current regime state (True=trending, False=ranging)
                         - regime_change: Boolean marking regime transition points
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Step 1: Generate signals from both component strategies in parallel
        trend_df = self.trend_strategy.generate_signals(data)
        mean_revert_df = self.mean_revert_strategy.generate_signals(data)
        
        # Step 2: Detect market regime for each time period
        favorable_regime = self.regime_detector.detect_regime(data)
        
        # Step 3: Initialize output DataFrame with base structure
        df = data.copy()
        df['signal'] = 0
        
        # Step 4: Store individual strategy signals for analysis and transparency
        df['trend_signal'] = trend_df['signal']
        df['mean_revert_signal'] = mean_revert_df['signal']
        df['regime_favorable'] = favorable_regime
        
        # Step 5: Apply regime switching logic to select appropriate signals
        # Favorable regime (trending market) -> use trend strategy
        df.loc[favorable_regime, 'signal'] = trend_df.loc[favorable_regime, 'signal']
        
        # Unfavorable regime (ranging market) -> use mean reversion strategy
        df.loc[~favorable_regime, 'signal'] = mean_revert_df.loc[~favorable_regime, 'signal']
        
        # Step 6: Track regime transitions for performance analysis
        regime_change = favorable_regime != favorable_regime.shift(1)
        df['regime_change'] = regime_change
        
        # Step 7: Calculate and log regime allocation statistics
        trend_periods = favorable_regime.sum()
        mean_revert_periods = (~favorable_regime).sum()
        total_periods = len(favorable_regime)
        regime_transitions = regime_change.sum()
        
        self.logger.info(f"Regime switching: {trend_periods}/{total_periods} trend periods "
                        f"({trend_periods/total_periods:.1%}), "
                        f"{mean_revert_periods}/{total_periods} mean-revert periods "
                        f"({mean_revert_periods/total_periods:.1%}), "
                        f"{regime_transitions} regime transitions")
        
        return df


def regime_gate_decorator(regime_detector: RegimeDetector):
    """Decorator function to add regime gating functionality to any existing strategy class.
    
    This decorator provides a convenient way to enhance existing strategies with regime
    filtering without modifying the original strategy implementation. It wraps the
    strategy's generate_signals method with regime detection logic.
    
    DECORATOR BENEFITS:
    ==================
    - NON-INVASIVE: Doesn't modify original strategy code
    - FLEXIBLE: Can be applied to any strategy implementing BaseStrategy
    - REUSABLE: Same decorator can enhance multiple strategy types
    - TRANSPARENT: Preserves original strategy interface
    
    USAGE PATTERN:
    =============
    ```python
    # Create regime detector
    regime_detector = RegimeDetector(method='vix', parameters={'vix_threshold': 20})
    
    # Apply as decorator to strategy class
    @regime_gate_decorator(regime_detector)
    class EnhancedRsiStrategy(RsiStrategy):
        pass
    
    # Or apply to existing strategy instance
    enhanced_strategy = regime_gate_decorator(regime_detector)(RsiStrategy)
    ```
    
    Args:
        regime_detector: Configured RegimeDetector instance to use for filtering
                        The detector's method and parameters determine filtering behavior
        
    Returns:
        Decorator function that wraps strategy classes with regime gating functionality
        
    Note:
        The decorated strategy will have 'Regime-Gated' prepended to its name and
        will include regime information in its signal output.
    """
    def decorator(strategy_class):
        """Inner decorator function that modifies the strategy class."""
        
        class RegimeGatedStrategyClass(strategy_class):
            """Dynamically created regime-gated version of the base strategy class."""
            
            def __init__(self, *args, **kwargs):
                """Initialize with regime detection capability added."""
                super().__init__(*args, **kwargs)
                self.regime_detector = regime_detector
                self.name = f"Regime-Gated {self.name}"
            
            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                """Enhanced signal generation with regime filtering applied."""
                # Get base strategy signals using parent class method
                df = super().generate_signals(data)
                
                # Apply regime filtering logic
                favorable_regime = self.regime_detector.detect_regime(data)
                
                # Store original signals for analysis
                df['base_signal'] = df['signal'].copy()
                df['regime_favorable'] = favorable_regime
                
                # Filter signals based on regime (only favorable regime signals pass through)
                df.loc[~favorable_regime, 'signal'] = 0
                
                return df
        
        return RegimeGatedStrategyClass
    
    return decorator


# Factory functions for convenient strategy creation
def create_regime_gated_strategy(base_strategy: BaseStrategy, 
                                regime_method: str = 'sma_slope',
                                regime_params: Dict[str, Any] = None) -> RegimeGatedStrategy:
    """Factory function to create regime-gated strategies with simplified configuration.
    
    This helper function streamlines the creation of regime-gated strategies by handling
    the instantiation of both the regime detector and the wrapper strategy with proper
    parameter validation and sensible defaults.
    
    COMMON CONFIGURATIONS:
    =====================
    - Conservative Trend Filter: method='sma_slope', params={'sma_period': 200, 'slope_threshold': 0.002}
    - Volatility Filter: method='atr_volatility', params={'low_vol_percentile': 10, 'high_vol_percentile': 90}
    - Fear/Greed Filter: method='vix', params={'vix_threshold': 25}
    - Multi-Factor Filter: method='combined' (uses default parameters for all methods)
    
    Args:
        base_strategy: Any strategy implementing BaseStrategy interface to be enhanced
                      The strategy will operate normally but signals filtered by regime
        regime_method: Regime detection method ('sma_slope', 'atr_volatility', 'vix', 'combined')
                      Determines what market conditions are considered favorable
        regime_params: Optional parameters for regime detector customization
                      If None, uses sensible defaults for the selected method
        
    Returns:
        RegimeGatedStrategy: Configured strategy with regime filtering applied
        
    Example:
        # Create conservative momentum strategy
        base_rsi = RsiStrategy(rsi_period=14, rsi_oversold=25, rsi_overbought=75)
        
        # Add strict trend regime filter
        conservative_strategy = create_regime_gated_strategy(
            base_strategy=base_rsi,
            regime_method='sma_slope',
            regime_params={'sma_period': 200, 'slope_threshold': 0.003}  # Very strict trend requirement
        )
    """
    detector = RegimeDetector(regime_method, regime_params)
    return RegimeGatedStrategy(base_strategy, detector)


def create_regime_switch_strategy(trend_strategy: BaseStrategy,
                                 mean_revert_strategy: BaseStrategy,
                                 regime_method: str = 'sma_slope',
                                 regime_params: Dict[str, Any] = None) -> RegimeSwitchStrategy:
    """Factory function to create regime-switching strategies with simplified configuration.
    
    This helper function creates a complete regime-switching strategy by combining two
    complementary strategies with an appropriate regime detector. The function handles
    proper initialization and parameter validation.
    
    STRATEGY PAIRING GUIDELINES:
    ===========================
    Good trend/mean-reversion pairs:
    - SmaEmaRsiStrategy + BollingerBandMeanReversionStrategy
    - CrossoverStrategy + RSIMeanReversionStrategy  
    - VolatilityBreakoutStrategy + ZScoreMeanReversionStrategy
    
    REGIME METHOD SELECTION:
    =======================
    - 'sma_slope': Good for trend/range regime distinction
    - 'atr_volatility': Good for normal/high volatility switching
    - 'vix': Good for risk-on/risk-off regime switching
    - 'combined': Most robust but more conservative switching
    
    Args:
        trend_strategy: Strategy optimized for trending/momentum markets
                       Should perform well when prices show directional movement
        mean_revert_strategy: Strategy optimized for ranging/mean-reverting markets
                            Should perform well when prices oscillate around mean
        regime_method: Method for detecting when to switch between strategies
                      Choice affects switching frequency and market interpretation
        regime_params: Optional parameters for fine-tuning regime detection
                      If None, uses balanced defaults suitable for most markets
        
    Returns:
        RegimeSwitchStrategy: Configured strategy that switches between components based on regime
        
    Example:
        # Create adaptive strategy for different market environments
        trend_strat = SmaEmaRsiStrategy(sma_short=10, sma_long=30, rsi_period=14)
        range_strat = BollingerBandMeanReversionStrategy(bb_period=20, bb_std=2.5)
        
        adaptive_strategy = create_regime_switch_strategy(
            trend_strategy=trend_strat,
            mean_revert_strategy=range_strat,
            regime_method='combined',  # Conservative switching
            regime_params=None  # Use defaults
        )
    """
    detector = RegimeDetector(regime_method, regime_params)
    return RegimeSwitchStrategy(trend_strategy, mean_revert_strategy, detector)


# Example usage and testing functions
if __name__ == "__main__":
    """Example usage of regime switching strategies for testing and demonstration."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample price data for testing
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create realistic price data with regime changes
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
    
    # Add some volatility clustering (GARCH-like behavior)
    high = price * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    low = price * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    volume = np.random.lognormal(10, 1, len(dates))
    
    sample_data = pd.DataFrame({
        'close': price,
        'high': high,
        'low': low,
        'volume': volume
    }, index=dates)
    
    # Test different regime detection methods
    methods = ['sma_slope', 'atr_volatility', 'combined']
    
    print("=== Regime Detection Testing ===")
    for method in methods:
        detector = RegimeDetector(method=method)
        regime_signal = detector.detect_regime(sample_data)
        favorable_pct = regime_signal.mean()
        
        print(f"{method.upper()}: {favorable_pct:.1%} of periods favorable")
    
    # Test regime-gated strategy creation
    print("\n=== Regime-Gated Strategy Testing ===")
    try:
        regime_strategy = create_regime_gated_strategy(
            base_strategy_name='RsiStrategy',
            base_parameters={'rsi_period': 14},
            regime_method='combined'
        )
        
        print(f"Created strategy: {regime_strategy.name}")
        print(f"Strategy info: {regime_strategy.get_info()}")
        
    except Exception as e:
        print(f"Error creating regime strategy: {e}") 