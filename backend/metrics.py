"""Performance metrics calculation for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
import warnings
import sys
import os
import logging
warnings.filterwarnings('ignore')

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


class PerformanceMetrics:
    """Calculate comprehensive performance metrics for trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = settings.TRADING_DAYS_PER_YEAR
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_metrics(self, 
                            equity_curve: pd.DataFrame, 
                            trades: pd.DataFrame,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: Optional[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: DataFrame with equity values over time
            trades: DataFrame with individual trade records
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Override default risk-free rate
            
        Returns:
            Dictionary with all calculated metrics
        """
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        
        if equity_curve.empty:
            return self._empty_metrics()
        
        # Calculate returns
        returns = self._calculate_returns(equity_curve)
        
        if returns.empty or len(returns) < 2:
            return self._empty_metrics()
        
        # Basic return metrics
        metrics = self._calculate_return_metrics(equity_curve, returns)
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_curve, returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(equity_curve))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Trade-based metrics
        if not trades.empty:
            metrics.update(self._calculate_trade_metrics(trades))
        
        # Benchmark comparison metrics
        if benchmark_returns is not None and not benchmark_returns.empty:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return metrics
    
    def _calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """Calculate returns from equity curve.
        
        Args:
            equity_curve: DataFrame with equity values
            
        Returns:
            Series of period returns
        """
        if 'equity' not in equity_curve.columns:
            return pd.Series(dtype=float)
        
        return equity_curve['equity'].pct_change().dropna()
    
    def _calculate_return_metrics(self, equity_curve: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic return metrics.
        
        Args:
            equity_curve: DataFrame with equity values
            returns: Series of returns
            
        Returns:
            Dictionary with return metrics
        """
        if returns.empty:
            return {}
        
        # Total return
        initial_value = equity_curve['equity'].iloc[0]
        final_value = equity_curve['equity'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        num_periods = len(returns)
        periods_per_year = self._estimate_periods_per_year(equity_curve)
        years = num_periods / periods_per_year
        
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0
        
        # Average returns
        daily_return_avg = returns.mean()
        monthly_return = (1 + daily_return_avg) ** (periods_per_year / 12) - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': monthly_return,
            'daily_return_avg': daily_return_avg,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum() / len(returns),
            'negative_days': (returns < 0).sum() / len(returns),
        }
    
    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics.
        
        Args:
            equity_curve: DataFrame with equity values
            returns: Series of returns
            
        Returns:
            Dictionary with risk metrics
        """
        if returns.empty:
            return {}
        
        periods_per_year = self._estimate_periods_per_year(equity_curve)
        
        # Volatility
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(periods_per_year)
        
        # Downside deviation (volatility of negative returns)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 5% VaR
        var_99 = np.percentile(returns, 1)  # 1% VaR
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'daily_volatility': daily_volatility,
            'annual_volatility': annual_volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown metrics.
        
        Args:
            equity_curve: DataFrame with equity values
            
        Returns:
            Dictionary with drawdown metrics
        """
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}
        
        equity = equity_curve['equity']
        
        # Calculate running maximum (peak)
        peak = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - peak) / peak
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Validation: Check for impossible drawdown values
        if max_drawdown < -1.0:  # Drawdown worse than -100%
            self.logger.warning(f"⚠️  IMPOSSIBLE DRAWDOWN DETECTED: {max_drawdown:.1%}")
            self.logger.warning(f"   This indicates portfolio went below zero (risk of ruin)")
            self.logger.warning(f"   Equity range: {equity.min():.2f} to {equity.max():.2f}")
            # Cap at -100% for reporting purposes
            max_drawdown = max(max_drawdown, -1.0)
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Drawdown duration analysis
        in_drawdown = drawdown < 0
        drawdown_periods = self._get_drawdown_periods(in_drawdown)
        
        max_dd_duration = max([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0
        avg_dd_duration = np.mean([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0
        
        # Recovery factor (total return / max drawdown)
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = self._calculate_return_metrics(equity_curve, equity.pct_change().dropna()).get('annual_return', 0)
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_dd_duration': max_dd_duration,
            'avg_dd_duration': avg_dd_duration,
            'recovery_factor': recovery_factor,
            'calmar_ratio': calmar_ratio,
            'drawdown_periods': len(drawdown_periods),
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        if returns.empty:
            return {}
        
        # Sharpe ratio
        excess_returns = returns.mean() - self.risk_free_rate / self.trading_days_per_year
        sharpe_ratio = excess_returns / returns.std() if returns.std() != 0 else 0
        sharpe_ratio_annual = sharpe_ratio * np.sqrt(self.trading_days_per_year)
        
        # Sortino ratio (uses downside deviation instead of total volatility)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        sortino_ratio = excess_returns / downside_std if downside_std != 0 else 0
        sortino_ratio_annual = sortino_ratio * np.sqrt(self.trading_days_per_year)
        
        # Treynor ratio (requires beta, will calculate if benchmark provided)
        # For now, we'll use a placeholder
        treynor_ratio = 0
        
        # Information ratio (will calculate if benchmark provided)
        information_ratio = 0
        
        return {
            'sharpe_ratio': sharpe_ratio_annual,
            'sortino_ratio': sortino_ratio_annual,
            'treynor_ratio': treynor_ratio,
            'information_ratio': information_ratio,
        }
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based metrics.

        Args:
            trades: DataFrame with trade records

        Returns:
            Dictionary with trade metrics
        """
        if trades.empty:
            return {}

        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = trades[trades['net_pnl'] > 0]
        losing_trades = trades[trades['net_pnl'] < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0

        # P&L statistics
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        largest_win = winning_trades['net_pnl'].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades['net_pnl'].min() if len(losing_trades) > 0 else 0

        # Profit factor
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

        # Expectancy
        expectancy = win_rate * avg_win + loss_rate * avg_loss

        # Kelly criterion
        if avg_win != 0 and avg_loss != 0 and win_rate != 0 and loss_rate != 0:
            kelly_pct = win_rate - (loss_rate * (abs(avg_loss) / avg_win))
        else:
            kelly_pct = 0

        # Average trade duration
        avg_duration = trades['duration'].mean() if 'duration' in trades.columns else 0

        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_outcomes(trades, 'win')
        consecutive_losses = self._calculate_consecutive_outcomes(trades, 'loss')

        # MAR ratio (if available)
        mar_ratio = 0
        if 'duration' in trades.columns and not trades.empty:
            total_return = trades['net_pnl'].sum()
            max_drawdown = min(0, trades['net_pnl'].min())  # Approximation
            if max_drawdown != 0:
                mar_ratio = total_return / abs(max_drawdown)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'kelly_criterion': kelly_pct,
            'avg_trade_duration': avg_duration,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'mar_ratio': mar_ratio,
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate metrics relative to benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with benchmark comparison metrics
        """
        if returns.empty or benchmark_returns.empty:
            return {}
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')
        strategy_returns = aligned_returns[0].dropna()
        bench_returns = aligned_returns[1].dropna()
        
        if strategy_returns.empty or bench_returns.empty:
            return {}
        
        # Beta calculation
        covariance = np.cov(strategy_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        
        # Alpha calculation (Jensen's Alpha)
        strategy_mean = strategy_returns.mean() * self.trading_days_per_year
        benchmark_mean = bench_returns.mean() * self.trading_days_per_year
        alpha = strategy_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        # Information ratio
        excess_returns = strategy_returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)
        information_ratio = excess_returns.mean() * self.trading_days_per_year / tracking_error if tracking_error != 0 else 0
        
        # Correlation
        correlation = np.corrcoef(strategy_returns, bench_returns)[0, 1]
        
        # Up/Down capture ratios
        up_market = bench_returns > 0
        down_market = bench_returns < 0
        
        up_capture = (strategy_returns[up_market].mean() / bench_returns[up_market].mean()) if up_market.any() and bench_returns[up_market].mean() != 0 else 0
        down_capture = (strategy_returns[down_market].mean() / bench_returns[down_market].mean()) if down_market.any() and bench_returns[down_market].mean() != 0 else 0
        
        return {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'up_capture': up_capture,
            'down_capture': down_capture,
        }
    
    def _get_drawdown_periods(self, in_drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Identify drawdown periods.

        Args:
            in_drawdown: Boolean series indicating drawdown periods

        Returns:
            List of drawdown period dictionaries
        """
        periods = []
        start_idx = None

        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                periods.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'start_date': in_drawdown.index[start_idx],
                    'end_date': in_drawdown.index[i - 1],
                    'duration': i - start_idx
                })
                start_idx = None

        # Handle case where drawdown continues to the end
        if start_idx is not None:
            periods.append({
                'start_idx': start_idx,
                'end_idx': len(in_drawdown) - 1,
                'start_date': in_drawdown.index[start_idx],
                'end_date': in_drawdown.index[-1],
                'duration': len(in_drawdown) - start_idx
            })

        return periods
    
    def _calculate_consecutive_outcomes(self, trades: pd.DataFrame, outcome_type: str) -> int:
        """Calculate maximum consecutive wins or losses.
        
        Args:
            trades: DataFrame with trade records
            outcome_type: 'win' or 'loss'
            
        Returns:
            Maximum consecutive outcomes
        """
        if trades.empty or 'net_pnl' not in trades.columns:
            return 0
        
        if outcome_type == 'win':
            outcomes = trades['net_pnl'] > 0
        else:
            outcomes = trades['net_pnl'] < 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for outcome in outcomes:
            if outcome:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _estimate_periods_per_year(self, equity_curve: pd.DataFrame) -> float:
        """Estimate the number of periods per year based on data frequency.
        
        Args:
            equity_curve: DataFrame with timestamp index
            
        Returns:
            Estimated periods per year
        """
        if len(equity_curve) < 2:
            return self.trading_days_per_year
        
        # Calculate average time difference
        time_diffs = pd.Series(equity_curve.index).diff().dropna()
        avg_diff = time_diffs.mean()
        # If index is not datetime, fallback to trading days
        if not hasattr(avg_diff, 'days'):
            return self.trading_days_per_year
        if avg_diff.days >= 1:
            return 365.25 / avg_diff.days
        else:
            hours_per_period = avg_diff.total_seconds() / 3600
            return (365.25 * 24) / hours_per_period
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return dictionary with zero/null metrics for empty data.
        
        Returns:
            Dictionary with default metric values
        """
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'monthly_return': 0.0,
            'daily_return_avg': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
        } 