"""Enhanced performance metrics calculation with additional professional-grade analytics.

This module extends the existing PerformanceMetrics class with additional metrics
commonly used by portfolio managers and institutional investors:

- Exposure (Time in Market) analysis
- Enhanced Calmar and MAR ratio calculations
- Tail risk metrics (Tail Ratio, enhanced Skew/Kurtosis analysis)
- Recovery Factor with detailed analysis
- Enhanced benchmark Alpha/Beta analysis with risk attribution
- Professional-grade metric presentation and formatting

Key Enhancements:
- Exposure metrics showing efficiency of capital deployment
- Tail risk analysis for understanding return distribution characteristics
- Enhanced benchmark comparison with risk attribution
- Professional metric categorization and presentation
- Robust error handling and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy import stats
import warnings
import sys
import os
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Import base metrics class
from .metrics import PerformanceMetrics


class EnhancedPerformanceMetrics(PerformanceMetrics):
    """Enhanced performance metrics calculator with professional-grade analytics.
    
    This class extends the base PerformanceMetrics with additional metrics
    crucial for institutional analysis and portfolio management.
    
    New Metrics Categories:
    1. Exposure Analytics: Time in market, capital efficiency
    2. Enhanced Risk Metrics: Tail ratios, distribution analysis
    3. Enhanced Benchmark Analysis: Risk attribution, factor exposure
    4. Professional Presentation: Categorized, formatted metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize enhanced metrics calculator."""
        super().__init__(risk_free_rate)
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_metrics(self, 
                            equity_curve: pd.DataFrame, 
                            trades: pd.DataFrame,
                            signals: Optional[pd.DataFrame] = None,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: Optional[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive enhanced performance metrics.
        
        Args:
            equity_curve: DataFrame with equity values over time
            trades: DataFrame with individual trade records
            signals: Optional DataFrame with strategy signals for exposure analysis
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Override default risk-free rate
            
        Returns:
            Dictionary with all calculated metrics organized by category
        """
        # Get base metrics
        base_metrics = super().calculate_all_metrics(
            equity_curve, trades, benchmark_returns, risk_free_rate
        )
        
        if equity_curve.empty:
            return self._enhanced_empty_metrics()
        
        # Calculate returns for enhanced analysis
        returns = self._calculate_returns(equity_curve)
        
        if returns.empty or len(returns) < 2:
            return self._enhanced_empty_metrics()
        
        # Enhanced metrics
        enhanced_metrics = {
            'basic_metrics': base_metrics,
            'exposure_metrics': self._calculate_exposure_metrics(equity_curve, trades, signals),
            'enhanced_risk_metrics': self._calculate_enhanced_risk_metrics(returns),
            'tail_risk_metrics': self._calculate_tail_risk_metrics(returns),
            'enhanced_drawdown_metrics': self._calculate_enhanced_drawdown_metrics(equity_curve),
            'efficiency_metrics': self._calculate_efficiency_metrics(equity_curve, returns),
        }
        
        # Enhanced benchmark analysis if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            enhanced_metrics['enhanced_benchmark_metrics'] = self._calculate_enhanced_benchmark_metrics(
                returns, benchmark_returns
            )
        
        # Calculate summary metrics that combine various factors
        enhanced_metrics['summary_metrics'] = self._calculate_summary_metrics(enhanced_metrics)
        
        # Add metric metadata for presentation
        enhanced_metrics['metric_metadata'] = self._get_metric_metadata()
        
        return enhanced_metrics
    
    def _calculate_exposure_metrics(self, 
                                  equity_curve: pd.DataFrame, 
                                  trades: pd.DataFrame,
                                  signals: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate exposure (time in market) metrics.
        
        Exposure metrics help understand capital efficiency by showing how much
        time the strategy was actually invested vs. sitting in cash.
        
        Args:
            equity_curve: DataFrame with equity values
            trades: DataFrame with trade records
            signals: Optional DataFrame with signals for position analysis
            
        Returns:
            Dictionary with exposure metrics
        """
        exposure_metrics = {}
        
        # Method 1: From trades (if available and has position tracking)
        if not trades.empty and 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            exposure_metrics.update(self._calculate_exposure_from_trades(trades, equity_curve))
        
        # Method 2: From signals (if available)
        if signals is not None and 'signal' in signals.columns:
            exposure_metrics.update(self._calculate_exposure_from_signals(signals))
        
        # Method 3: From equity curve changes (basic heuristic)
        if not exposure_metrics:  # Fallback if other methods not available
            exposure_metrics.update(self._calculate_exposure_from_equity_changes(equity_curve))
        
        return exposure_metrics
    
    def _calculate_exposure_from_trades(self, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate exposure metrics from trade records."""
        if trades.empty:
            return {}
        
        # Total time period
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        total_days = (end_date - start_date).days
        
        # Calculate total time in positions
        total_position_days = 0
        overlapping_days = 0
        
        # Sort trades by entry date
        sorted_trades = trades.sort_values('entry_date')
        
        # Track overlapping positions
        active_positions = []
        
        for _, trade in sorted_trades.iterrows():
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            position_days = (exit_date - entry_date).days
            
            # Remove expired positions
            active_positions = [pos for pos in active_positions if pos['exit'] > entry_date]
            
            # Check for overlap
            if active_positions:
                overlapping_days += position_days
            
            active_positions.append({'entry': entry_date, 'exit': exit_date})
            total_position_days += position_days
        
        # Calculate exposure metrics
        time_in_market = total_position_days / total_days if total_days > 0 else 0
        time_in_cash = 1 - time_in_market
        
        # Position concentration metrics
        avg_position_duration = trades['duration'].mean() if 'duration' in trades.columns else 0
        max_position_duration = trades['duration'].max() if 'duration' in trades.columns else 0
        
        # Overlap analysis
        overlap_ratio = overlapping_days / total_position_days if total_position_days > 0 else 0
        
        return {
            'time_in_market': time_in_market,
            'time_in_cash': time_in_cash,
            'exposure_efficiency': time_in_market,  # Same as time_in_market but conceptually different
            'avg_position_duration_days': avg_position_duration,
            'max_position_duration_days': max_position_duration,
            'position_overlap_ratio': overlap_ratio,
            'total_position_periods': len(trades),
        }
    
    def _calculate_exposure_from_signals(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate exposure metrics from signal DataFrame."""
        if signals.empty or 'signal' in signals.columns:
            return {}
        
        # Count periods with active signals
        active_signals = signals['signal'] != 0
        total_periods = len(signals)
        active_periods = active_signals.sum()
        
        time_in_market = active_periods / total_periods if total_periods > 0 else 0
        time_in_cash = 1 - time_in_market
        
        # Signal analysis
        long_signals = (signals['signal'] == 1).sum()
        short_signals = (signals['signal'] == -1).sum()
        
        long_exposure = long_signals / total_periods if total_periods > 0 else 0
        short_exposure = short_signals / total_periods if total_periods > 0 else 0
        
        # Signal concentration
        signal_changes = (signals['signal'] != signals['signal'].shift()).sum()
        avg_signal_duration = active_periods / signal_changes if signal_changes > 0 else 0
        
        return {
            'time_in_market': time_in_market,
            'time_in_cash': time_in_cash,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'exposure_efficiency': time_in_market,
            'avg_signal_duration_periods': avg_signal_duration,
            'total_signal_changes': signal_changes,
        }
    
    def _calculate_exposure_from_equity_changes(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic exposure metrics from equity curve changes (heuristic method)."""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}
        
        # Heuristic: assume position when equity changes significantly
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Define "significant" change threshold (could be parameterized)
        threshold = returns.std() * 0.5  # Half standard deviation
        
        active_periods = (returns.abs() > threshold).sum()
        total_periods = len(returns)
        
        time_in_market = active_periods / total_periods if total_periods > 0 else 0
        
        return {
            'time_in_market_estimated': time_in_market,
            'time_in_cash_estimated': 1 - time_in_market,
            'exposure_estimation_method': 'equity_curve_heuristic',
            'estimation_threshold': threshold,
        }
    
    def _calculate_enhanced_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate enhanced risk metrics beyond the basic implementation."""
        if returns.empty:
            return {}
        
        # Enhanced volatility analysis
        rolling_vol_30 = returns.rolling(30).std()
        rolling_vol_90 = returns.rolling(90).std()
        
        vol_stability = rolling_vol_30.std() / rolling_vol_30.mean() if rolling_vol_30.mean() != 0 else 0
        vol_trend = np.polyfit(range(len(rolling_vol_30.dropna())), rolling_vol_30.dropna(), 1)[0] if len(rolling_vol_30.dropna()) > 1 else 0
        
        # Enhanced downside risk metrics
        negative_returns = returns[returns < 0]
        downside_frequency = len(negative_returns) / len(returns)
        
        # Pain index (average drawdown over all periods)
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown_series = (cumulative_returns - running_max) / running_max
        pain_index = drawdown_series.mean()
        
        # Ulcer index (RMS of drawdowns)
        ulcer_index = np.sqrt((drawdown_series ** 2).mean())
        
        return {
            'volatility_stability': vol_stability,
            'volatility_trend': vol_trend,
            'downside_frequency': downside_frequency,
            'pain_index': pain_index,
            'ulcer_index': ulcer_index,
            'vol_30d_avg': rolling_vol_30.mean(),
            'vol_90d_avg': rolling_vol_90.mean(),
        }
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive tail risk metrics.
        
        These metrics help understand the behavior of returns in extreme conditions.
        """
        if returns.empty:
            return {}
        
        # Percentile analysis
        percentiles = [1, 5, 10, 90, 95, 99]
        percentile_values = {f'percentile_{p}': np.percentile(returns, p) for p in percentiles}
        
        # Tail ratios
        upside_95 = np.percentile(returns, 95)
        downside_5 = np.percentile(returns, 5)
        tail_ratio = abs(upside_95 / downside_5) if downside_5 != 0 else np.inf
        
        upside_99 = np.percentile(returns, 99)
        downside_1 = np.percentile(returns, 1)
        extreme_tail_ratio = abs(upside_99 / downside_1) if downside_1 != 0 else np.inf
        
        # Enhanced skewness and kurtosis analysis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        excess_kurtosis = kurtosis  # scipy.stats.kurtosis returns excess kurtosis by default
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
        except:
            jb_stat, jb_pvalue = np.nan, np.nan
        
        # Maximum loss streaks
        negative_streak = 0
        max_negative_streak = 0
        for ret in returns:
            if ret < 0:
                negative_streak += 1
                max_negative_streak = max(max_negative_streak, negative_streak)
            else:
                negative_streak = 0
        
        # Positive returns analysis
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        positive_ratio = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        return {
            **percentile_values,
            'tail_ratio_95_5': tail_ratio,
            'tail_ratio_99_1': extreme_tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': excess_kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal_distribution': jb_pvalue > 0.05 if not np.isnan(jb_pvalue) else False,
            'max_negative_streak': max_negative_streak,
            'positive_return_ratio': positive_ratio,
            'avg_positive_return': positive_returns.mean() if len(positive_returns) > 0 else 0,
            'avg_negative_return': negative_returns.mean() if len(negative_returns) > 0 else 0,
        }
    
    def _calculate_enhanced_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate enhanced drawdown metrics with detailed analysis."""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}
        
        equity = equity_curve['equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        # Enhanced drawdown analysis
        underwater_periods = (drawdown < 0).sum()
        total_periods = len(drawdown)
        underwater_ratio = underwater_periods / total_periods
        
        # Drawdown depth distribution
        dd_percentiles = [10, 25, 50, 75, 90, 95, 99]
        dd_distribution = {f'dd_percentile_{p}': np.percentile(drawdown[drawdown < 0], p) 
                          for p in dd_percentiles if (drawdown < 0).any()}
        
        # Recovery analysis
        recovery_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_periods.append(i - drawdown_start)
        
        avg_recovery_periods = np.mean(recovery_periods) if recovery_periods else 0
        max_recovery_periods = max(recovery_periods) if recovery_periods else 0
        
        # Lake ratio (time underwater / total time)
        lake_ratio = underwater_ratio
        
        # Burke ratio (excess return / drawdown standard deviation)
        dd_std = drawdown.std()
        mean_return = equity.pct_change().mean()
        burke_ratio = mean_return / dd_std if dd_std != 0 else 0
        
        return {
            'underwater_ratio': underwater_ratio,
            'avg_recovery_periods': avg_recovery_periods,
            'max_recovery_periods': max_recovery_periods,
            'lake_ratio': lake_ratio,
            'burke_ratio': burke_ratio,
            'drawdown_std': dd_std,
            'total_drawdown_periods': len([p for p in recovery_periods]),
            **dd_distribution,
        }
    
    def _calculate_efficiency_metrics(self, equity_curve: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """Calculate capital and return efficiency metrics."""
        if equity_curve.empty or returns.empty:
            return {}
        
        # Return efficiency metrics
        total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1
        num_periods = len(returns)
        
        # Efficiency per period
        return_per_period = total_return / num_periods if num_periods > 0 else 0
        
        # Risk-adjusted efficiency
        return_to_risk = returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Consistency metrics
        positive_periods = (returns > 0).sum()
        consistency_ratio = positive_periods / num_periods if num_periods > 0 else 0
        
        # Rolling performance stability
        rolling_returns = returns.rolling(30).sum()
        performance_stability = rolling_returns.std() / rolling_returns.mean() if rolling_returns.mean() != 0 else 0
        
        return {
            'return_per_period': return_per_period,
            'return_to_risk_ratio': return_to_risk,
            'consistency_ratio': consistency_ratio,
            'performance_stability': performance_stability,
        }
    
    def _calculate_enhanced_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate enhanced benchmark comparison metrics with risk attribution."""
        base_benchmark = super()._calculate_benchmark_metrics(returns, benchmark_returns)
        
        if returns.empty or benchmark_returns.empty:
            return base_benchmark
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')
        strategy_returns = aligned_returns[0].dropna()
        bench_returns = aligned_returns[1].dropna()
        
        if strategy_returns.empty or bench_returns.empty:
            return base_benchmark
        
        # Enhanced alpha/beta analysis
        enhanced_metrics = {}
        
        # Rolling beta analysis
        rolling_beta = self._calculate_rolling_beta(strategy_returns, bench_returns, window=60)
        enhanced_metrics['beta_stability'] = rolling_beta.std()
        enhanced_metrics['avg_beta'] = rolling_beta.mean()
        enhanced_metrics['beta_trend'] = np.polyfit(range(len(rolling_beta)), rolling_beta, 1)[0] if len(rolling_beta) > 1 else 0
        
        # Alpha decomposition
        total_return_strategy = (1 + strategy_returns).prod() - 1
        total_return_benchmark = (1 + bench_returns).prod() - 1
        
        # Jensen's alpha (already calculated in base)
        jensen_alpha = base_benchmark.get('alpha', 0)
        
        # Return attribution
        beta_return = base_benchmark.get('beta', 1) * total_return_benchmark
        alpha_return = total_return_strategy - beta_return
        
        enhanced_metrics['alpha_contribution'] = alpha_return
        enhanced_metrics['beta_contribution'] = beta_return
        enhanced_metrics['total_excess_return'] = total_return_strategy - total_return_benchmark
        
        # Risk attribution
        strategy_vol = strategy_returns.std()
        benchmark_vol = bench_returns.std()
        excess_vol = (strategy_returns - bench_returns).std()
        
        enhanced_metrics['relative_volatility'] = strategy_vol / benchmark_vol if benchmark_vol != 0 else 1
        enhanced_metrics['excess_volatility'] = excess_vol
        
        # Market timing metrics
        up_market = bench_returns > bench_returns.median()
        down_market = bench_returns <= bench_returns.median()
        
        strategy_up_performance = strategy_returns[up_market].mean()
        strategy_down_performance = strategy_returns[down_market].mean()
        benchmark_up_performance = bench_returns[up_market].mean()
        benchmark_down_performance = bench_returns[down_market].mean()
        
        enhanced_metrics['up_market_capture'] = strategy_up_performance / benchmark_up_performance if benchmark_up_performance != 0 else 0
        enhanced_metrics['down_market_capture'] = strategy_down_performance / benchmark_down_performance if benchmark_down_performance != 0 else 0
        
        # Market timing ability (positive if outperforms in up markets and underperforms less in down markets)
        timing_ability = enhanced_metrics['up_market_capture'] - enhanced_metrics['down_market_capture']
        enhanced_metrics['market_timing_ability'] = timing_ability
        
        # Combine with base metrics
        return {**base_benchmark, **enhanced_metrics}
    
    def _calculate_rolling_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling beta over specified window."""
        rolling_beta = pd.Series(index=strategy_returns.index, dtype=float)
        
        for i in range(window, len(strategy_returns)):
            strat_window = strategy_returns.iloc[i-window:i]
            bench_window = benchmark_returns.iloc[i-window:i]
            
            if len(strat_window) > 10 and len(bench_window) > 10:  # Minimum data points
                covariance = np.cov(strat_window, bench_window)[0, 1]
                benchmark_variance = np.var(bench_window)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
                rolling_beta.iloc[i] = beta
        
        return rolling_beta.dropna()
    
    def _calculate_summary_metrics(self, enhanced_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate high-level summary metrics combining various factors."""
        summary = {}
        
        # Extract key metrics from different categories
        basic = enhanced_metrics.get('basic_metrics', {})
        exposure = enhanced_metrics.get('exposure_metrics', {})
        tail_risk = enhanced_metrics.get('tail_risk_metrics', {})
        efficiency = enhanced_metrics.get('efficiency_metrics', {})
        
        # Exposure-adjusted returns
        total_return = basic.get('total_return', 0)
        time_in_market = exposure.get('time_in_market', 1)  # Default to fully invested
        
        if time_in_market > 0:
            exposure_adjusted_return = total_return / time_in_market
            summary['exposure_adjusted_return'] = exposure_adjusted_return
            summary['capital_efficiency'] = exposure_adjusted_return / total_return if total_return != 0 else 1
        
        # Risk-adjusted efficiency score
        sharpe = basic.get('sharpe_ratio', 0)
        tail_ratio = tail_risk.get('tail_ratio_95_5', 1)
        consistency = efficiency.get('consistency_ratio', 0)
        
        # Composite efficiency score (0-100 scale)
        efficiency_score = min(100, max(0, (
            sharpe * 20 +  # Sharpe component (0-20)
            min(tail_ratio, 5) * 10 +  # Tail ratio component (0-50), capped at 5
            consistency * 30  # Consistency component (0-30)
        )))
        
        summary['strategy_efficiency_score'] = efficiency_score
        
        # Market regime adaptability (if benchmark metrics available)
        benchmark = enhanced_metrics.get('enhanced_benchmark_metrics', {})
        if benchmark:
            timing_ability = benchmark.get('market_timing_ability', 0)
            summary['market_adaptability'] = min(1, max(-1, timing_ability))
        
        return summary
    
    def _get_metric_metadata(self) -> Dict[str, Dict[str, str]]:
        """Get metadata for metric presentation and interpretation."""
        return {
            'exposure_metrics': {
                'category': 'Capital Efficiency',
                'description': 'Metrics showing how efficiently capital is deployed',
                'interpretation': 'Higher time in market with good returns indicates efficient capital use'
            },
            'tail_risk_metrics': {
                'category': 'Risk Analysis',
                'description': 'Analysis of extreme return behavior and distribution characteristics',
                'interpretation': 'Tail ratios > 1 indicate better upside than downside extremes'
            },
            'enhanced_benchmark_metrics': {
                'category': 'Relative Performance',
                'description': 'Detailed comparison against benchmark with risk attribution',
                'interpretation': 'Positive alpha indicates outperformance beyond beta exposure'
            },
            'efficiency_metrics': {
                'category': 'Performance Quality',
                'description': 'Metrics measuring consistency and quality of returns',
                'interpretation': 'Higher consistency and stability indicate more reliable performance'
            },
            'summary_metrics': {
                'category': 'Overall Assessment',
                'description': 'High-level composite metrics for strategy evaluation',
                'interpretation': 'Efficiency scores above 60 indicate strong overall performance'
            }
        }
    
    def _enhanced_empty_metrics(self) -> Dict[str, Any]:
        """Return enhanced empty metrics structure."""
        base_empty = super()._empty_metrics()
        
        return {
            'basic_metrics': base_empty,
            'exposure_metrics': {
                'time_in_market': 0.0,
                'time_in_cash': 1.0,
                'exposure_efficiency': 0.0,
            },
            'enhanced_risk_metrics': {
                'volatility_stability': 0.0,
                'pain_index': 0.0,
                'ulcer_index': 0.0,
            },
            'tail_risk_metrics': {
                'tail_ratio_95_5': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
            },
            'efficiency_metrics': {
                'return_per_period': 0.0,
                'consistency_ratio': 0.0,
            },
            'summary_metrics': {
                'strategy_efficiency_score': 0.0,
                'exposure_adjusted_return': 0.0,
            }
        }
    
    def format_metrics_for_display(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Format metrics for professional display with proper units and descriptions.
        
        Args:
            metrics: Raw metrics dictionary
            
        Returns:
            Formatted metrics with descriptions and proper formatting
        """
        formatter = MetricsFormatter()
        return formatter.format_all_metrics(metrics)


class MetricsFormatter:
    """Professional metrics formatting for display and reporting."""
    
    def __init__(self):
        self.percentage_metrics = {
            'total_return', 'annual_return', 'monthly_return', 'max_drawdown',
            'time_in_market', 'time_in_cash', 'win_rate', 'exposure_efficiency',
            'consistency_ratio', 'positive_return_ratio', 'underwater_ratio'
        }
        
        self.ratio_metrics = {
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor',
            'tail_ratio_95_5', 'tail_ratio_99_1', 'return_to_risk_ratio'
        }
        
        self.currency_metrics = {
            'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
            'gross_profit', 'gross_loss', 'expectancy'
        }
    
    def format_all_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Format all metric categories for display."""
        formatted = {}
        
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                formatted[category] = self.format_category(category_metrics)
            
        return formatted
    
    def format_category(self, category_metrics: Dict[str, float]) -> Dict[str, str]:
        """Format metrics within a category."""
        formatted = {}
        
        for metric_name, value in category_metrics.items():
            if isinstance(value, (int, float)):
                formatted[metric_name] = self.format_metric(metric_name, value)
            else:
                formatted[metric_name] = str(value)
        
        return formatted
    
    def format_metric(self, metric_name: str, value: float) -> str:
        """Format individual metric with appropriate units and precision."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        
        if metric_name in self.percentage_metrics:
            return f"{value:.2%}"
        elif metric_name in self.ratio_metrics:
            return f"{value:.3f}"
        elif metric_name in self.currency_metrics:
            return f"${value:,.2f}"
        elif 'days' in metric_name or 'periods' in metric_name:
            return f"{value:.0f}"
        elif 'pvalue' in metric_name:
            return f"{value:.4f}"
        else:
            # Default formatting
            if abs(value) >= 1000:
                return f"{value:,.0f}"
            elif abs(value) >= 1:
                return f"{value:.3f}"
            else:
                return f"{value:.6f}"