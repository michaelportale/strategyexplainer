"""Performance Metrics Display Component for Strategy Explainer Dashboard.

This module provides a comprehensive suite of performance metric visualizations designed
for trading strategy evaluation. It creates intuitive, information-rich metric cards
and displays that help users quickly understand strategy performance characteristics.

METRICS PHILOSOPHY:
==================
The metrics system follows key principles for effective performance communication:

1. HIERARCHY: Most important metrics displayed prominently with visual emphasis
2. CONTEXT: Metrics include benchmarks, percentiles, and historical context
3. CLARITY: Complex calculations presented in accessible, intuitive formats
4. ACTIONABILITY: Metrics guide decision-making with clear good/bad indicators
5. COMPLETENESS: Comprehensive coverage of risk, return, and efficiency measures

METRIC CATEGORIES:
=================

1. RETURN METRICS: Core performance measures
   - Total Return: Cumulative strategy performance over period
   - Annualized Return: Standardized yearly performance measure
   - Benchmark-Relative Return: Outperformance vs. market indices
   - Rolling Returns: Time-varying performance analysis

2. RISK METRICS: Downside protection and volatility measures
   - Maximum Drawdown: Worst peak-to-trough decline
   - Volatility: Standard deviation of returns (annualized)
   - Value at Risk (VaR): Potential loss estimates
   - Downside Deviation: Volatility of negative returns only

3. RISK-ADJUSTED METRICS: Return per unit of risk measures
   - Sharpe Ratio: Excess return per unit of total risk
   - Sortino Ratio: Excess return per unit of downside risk
   - Calmar Ratio: Annual return divided by maximum drawdown
   - Information Ratio: Active return per unit of tracking error

4. EFFICIENCY METRICS: Trading and execution effectiveness
   - Win Rate: Percentage of profitable trades
   - Profit Factor: Ratio of gross profit to gross loss
   - Average Trade: Mean profit/loss per trade
   - Recovery Factor: Return divided by maximum drawdown

VISUAL DESIGN PRINCIPLES:
========================
- COLOR CODING: Green for positive/good, red for negative/poor, neutral gray
- TYPOGRAPHY: Clear hierarchy with large primary values, smaller context text
- LAYOUT: Grid-based responsive design adapting to screen sizes
- ANIMATIONS: Subtle transitions and hover effects for engagement
- ACCESSIBILITY: High contrast, screen reader support, keyboard navigation

USAGE PATTERNS:
==============
```python
# Display key performance overview
display_key_metrics(strategy_metrics)

# Show comprehensive metrics grid
display_detailed_metrics(strategy_metrics)

# Create performance comparison table
display_metrics_comparison([strategy1_metrics, strategy2_metrics], 
                          names=["Strategy A", "Strategy B"])

# Risk analysis focused display
display_risk_metrics(strategy_metrics, include_var=True)
```
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from config.config_manager import get_config_manager


# Load metrics styling from configuration
config = get_config_manager()
metrics_styling_config = config.get('ui.metrics_cards.styling', {})

# Define consistent styling for metrics display
METRICS_STYLING = {
    'positive_color': metrics_styling_config.get('positive_color', '#26A69A'),    # Green for good performance
    'negative_color': metrics_styling_config.get('negative_color', '#EF5350'),    # Red for poor performance  
    'neutral_color': metrics_styling_config.get('neutral_color', '#9E9E9E'),     # Gray for neutral values
    'background_good': '#E8F5E8',   # Light green background
    'background_bad': '#FFEBEE',    # Light red background
    'background_neutral': '#F5F5F5', # Light gray background
    'text_primary': metrics_styling_config.get('text_color', '#333333'),      # Dark gray for primary text
    'text_secondary': metrics_styling_config.get('header_color', '#666666'),    # Medium gray for secondary text
    'border_color': metrics_styling_config.get('border_color', '#E0E0E0')       # Light gray for borders
}


def display_key_metrics(metrics: Dict[str, Any]) -> None:
    """Display essential performance metrics with enhanced formatting and color coding.
    
    This function creates a visually appealing overview of the most critical
    performance metrics that users need to evaluate strategy effectiveness.
    The layout emphasizes the key metrics while providing context and comparisons.
    
    KEY METRICS DISPLAYED:
    =====================
    - Total Return: Overall strategy performance as percentage
    - Annualized Return: Standardized yearly performance measure
    - Sharpe Ratio: Risk-adjusted return efficiency measure
    - Maximum Drawdown: Worst-case loss scenario
    
    ENHANCED VISUAL FEATURES:
    ========================
    - Professional percentage formatting (15.23%)
    - Color-coded metrics (green=positive, red=negative)
    - Delta indicators with benchmark comparisons
    - Contextual status indicators with proper thresholds
    - Responsive grid layout with enhanced styling
    
    Args:
        metrics (Dict[str, Any]): Performance metrics dictionary
    """
    st.subheader("📊 Key Performance Metrics")
    
    # Create four-column layout for primary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total Return with enhanced color coding
        total_return = metrics.get('total_return', 0)
        total_return_pct = total_return * 100
        
        # Enhanced color coding based on performance with consistent formatting
        metric_color = METRICS_STYLING['positive_color'] if total_return_pct > 0 else METRICS_STYLING['negative_color']
        bg_color = METRICS_STYLING['background_good'] if total_return_pct > 0 else METRICS_STYLING['background_bad']
        
        st.markdown(
            f"<div style='padding: 12px; background-color: {bg_color}; "
            f"border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {metric_color};'>"
            f"<h4 style='color: {METRICS_STYLING['text_primary']}; margin: 0; font-size: 14px;'>📈 Total Return</h4>"
            f"<h2 style='color: {metric_color}; margin: 5px 0; font-size: 28px; font-weight: 600;'>{total_return_pct:.2f}%</h2>"
            f"</div>", unsafe_allow_html=True
        )
        
        # Performance status with enhanced thresholds
        if total_return_pct > 15:
            st.success("🚀 Exceptional Performance")
        elif total_return_pct > 8:
            st.success("🎯 Strong Performance")
        elif total_return_pct > 0:
            st.info("📊 Positive Performance") 
        elif total_return_pct > -5:
            st.warning("⚠️ Slight Underperformance")
        else:
            st.error("🔻 Poor Performance")
    
    with col2:
        # Annualized Return with enhanced benchmark context
        annual_return = metrics.get('annual_return', 0)
        annual_return_pct = annual_return * 100
        benchmark_return = metrics.get('benchmark_return', 0)
        benchmark_return_pct = benchmark_return * 100 if benchmark_return else None
        
        # Calculate excess return if benchmark available
        excess_return = annual_return_pct - benchmark_return_pct if benchmark_return_pct is not None else None
        
        # Enhanced metric display with delta coloring
        delta_text = None
        delta_color = None
        if excess_return is not None:
            delta_text = f"{excess_return:+.2f}% vs benchmark"
            delta_color = "normal" if excess_return >= 0 else "inverse"
        
        st.metric(
            label="📅 Annual Return",
            value=f"{annual_return_pct:.2f}%",
            delta=delta_text,
            delta_color=delta_color,
            help="Annualized return standardized to yearly performance"
        )
        
        # Benchmark comparison status
        if excess_return is not None:
            if excess_return > 3:
                st.success(f"🏆 Outperforming by {excess_return:.1f}%")
            elif excess_return > 0:
                st.info(f"📈 Beating benchmark by {excess_return:.1f}%")
            elif excess_return > -2:
                st.warning(f"📊 Close to benchmark ({excess_return:+.1f}%)")
            else:
                st.error(f"📉 Underperforming by {abs(excess_return):.1f}%")
    
    with col3:
        # Sharpe Ratio with enhanced interpretation and coloring
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        # Color-coded Sharpe ratio display
        if sharpe_ratio > 1:
            sharpe_color = METRICS_STYLING['positive_color']
            bg_color = METRICS_STYLING['background_good']
        elif sharpe_ratio > 0:
            sharpe_color = METRICS_STYLING['neutral_color']
            bg_color = METRICS_STYLING['background_neutral']
        else:
            sharpe_color = METRICS_STYLING['negative_color']
            bg_color = METRICS_STYLING['background_bad']
        
        st.markdown(
            f"<div style='padding: 12px; background-color: {bg_color}; "
            f"border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {sharpe_color};'>"
            f"<h4 style='color: {METRICS_STYLING['text_primary']}; margin: 0; font-size: 14px;'>⚖️ Sharpe Ratio</h4>"
            f"<h2 style='color: {sharpe_color}; margin: 5px 0; font-size: 28px; font-weight: 600;'>{sharpe_ratio:.2f}</h2>"
            f"</div>", unsafe_allow_html=True
        )
        
        # Enhanced Sharpe ratio interpretation
        if sharpe_ratio > 2:
            st.success("🌟 Excellent (>2.0)")
        elif sharpe_ratio > 1.5:
            st.success("⭐ Very Good (1.5-2.0)")
        elif sharpe_ratio > 1:
            st.info("👍 Good (1.0-1.5)")
        elif sharpe_ratio > 0.5:
            st.warning("📊 Acceptable (0.5-1.0)")
        elif sharpe_ratio > 0:
            st.warning("⚠️ Below Average (0-0.5)")
        else:
            st.error("🔻 Poor (<0)")
    
    with col4:
        # Maximum Drawdown with enhanced severity assessment
        max_drawdown = metrics.get('max_drawdown', 0)
        max_drawdown_pct = max_drawdown * 100
        
        # Color-coded drawdown display (red for high drawdown)
        drawdown_abs = abs(max_drawdown_pct)
        if drawdown_abs < 5:
            dd_color = METRICS_STYLING['positive_color']
            bg_color = METRICS_STYLING['background_good']
        elif drawdown_abs < 15:
            dd_color = METRICS_STYLING['neutral_color']
            bg_color = METRICS_STYLING['background_neutral']
        else:
            dd_color = METRICS_STYLING['negative_color']
            bg_color = METRICS_STYLING['background_bad']
        
        st.markdown(
            f"<div style='padding: 12px; background-color: {bg_color}; "
            f"border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {dd_color};'>"
            f"<h4 style='color: {METRICS_STYLING['text_primary']}; margin: 0; font-size: 14px;'>📉 Max Drawdown</h4>"
            f"<h2 style='color: {dd_color}; margin: 5px 0; font-size: 28px; font-weight: 600;'>{max_drawdown_pct:.2f}%</h2>"
            f"</div>", unsafe_allow_html=True
        )
        
        # Enhanced drawdown severity assessment
        if drawdown_abs < 3:
            st.success("🛡️ Excellent Risk Control")
        elif drawdown_abs < 8:
            st.success("✅ Low Risk")
        elif drawdown_abs < 15:
            st.info("📊 Moderate Risk")
        elif drawdown_abs < 25:
            st.warning("⚠️ High Risk")
        elif drawdown_abs < 40:
            st.error("🚨 Very High Risk")
        else:
            st.error("💀 Extreme Risk")


def display_detailed_metrics(metrics: Dict[str, Any]) -> None:
    """Display comprehensive performance metrics in organized sections.
    
    This function provides an in-depth view of all available performance metrics,
    organized into logical categories for thorough strategy evaluation. It includes
    advanced metrics, statistical measures, and trading effectiveness indicators.
    
    METRICS ORGANIZATION:
    ====================
    1. Return Analysis: Various return calculations and time-series analysis
    2. Risk Assessment: Volatility, drawdown, and risk measures
    3. Risk-Adjusted Performance: Efficiency ratios and adjusted returns
    4. Trading Statistics: Win rates, trade analysis, and execution metrics
    
    ADVANCED FEATURES:
    =================
    - Statistical significance indicators
    - Percentile rankings and benchmarks
    - Time-period breakdowns (monthly, quarterly, yearly)
    - Distribution analysis and tail risk measures
    - Correlation and beta analysis with benchmarks
    
    Args:
        metrics (Dict[str, Any]): Comprehensive metrics dictionary with all available
                                 performance statistics, risk measures, and trading data
    """
    st.subheader("📋 Detailed Performance Analysis")
    
    # Organize metrics into logical sections
    tabs = st.tabs(["📈 Returns", "⚠️ Risk", "⚖️ Risk-Adjusted", "🎯 Trading Stats"])
    
    with tabs[0]:
        _display_return_metrics(metrics)
    
    with tabs[1]:
        _display_risk_metrics(metrics)
    
    with tabs[2]:
        _display_risk_adjusted_metrics(metrics)
        
    with tabs[3]:
        _display_trading_metrics(metrics)


def _display_return_metrics(metrics: Dict[str, Any]) -> None:
    """Display return-focused performance metrics with enhanced formatting."""
    st.markdown("#### 📈 Return Analysis")
    
    # Create columns for different return measures
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Period Returns**")
        
        # Total return with color coding
        total_return = metrics.get('total_return', 0) * 100
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric(
            "Total Return", 
            f"{total_return:.2f}%",
            delta=f"{total_return:+.2f}%" if total_return != 0 else None,
            delta_color=delta_color
        )
        
        # Annualized return
        annual_return = metrics.get('annual_return', 0) * 100
        st.metric("Annualized Return", f"{annual_return:.2f}%")
        
        # Monthly return (if available)
        monthly_return = metrics.get('monthly_return', 0) * 100
        if monthly_return:
            st.metric("Avg Monthly Return", f"{monthly_return:.2f}%")
    
    with col2:
        st.markdown("**Benchmark Comparison**")
        
        # Benchmark return
        benchmark_return = metrics.get('benchmark_return', 0) * 100
        st.metric("Benchmark Return", f"{benchmark_return:.2f}%")
        
        # Alpha (excess return) with enhanced delta display
        alpha = metrics.get('alpha', 0) * 100
        if alpha == 0 and 'annual_return' in metrics and 'benchmark_return' in metrics:
            alpha = (metrics.get('annual_return', 0) - metrics.get('benchmark_return', 0)) * 100
        
        alpha_color = "normal" if alpha >= 0 else "inverse"
        st.metric(
            "Alpha (Excess Return)", 
            f"{alpha:.2f}%",
            delta=f"{alpha:+.2f}% vs benchmark" if alpha != 0 else None,
            delta_color=alpha_color
        )
        
        # Beta (systematic risk) with interpretation
        beta = metrics.get('beta', 1.0)
        beta_interpretation = "Higher" if beta > 1.1 else "Lower" if beta < 0.9 else "Similar"
        st.metric(
            "Beta (Market Sensitivity)", 
            f"{beta:.2f}",
            help=f"{beta_interpretation} volatility than market"
        )
    
    with col3:
        st.markdown("**Return Statistics**")
        
        # Best month/period with color coding
        best_period = metrics.get('best_month', 0) * 100
        st.metric(
            "Best Month", 
            f"{best_period:.2f}%",
            delta=f"+{best_period:.2f}%" if best_period > 0 else None,
            delta_color="normal"
        )
        
        # Worst month/period with color coding
        worst_period = metrics.get('worst_month', 0) * 100
        st.metric(
            "Worst Month", 
            f"{worst_period:.2f}%",
            delta=f"{worst_period:+.2f}%" if worst_period != 0 else None,
            delta_color="inverse"
        )
        
        # Win rate (percentage of positive periods)
        win_rate = metrics.get('win_rate_periods', 0) * 100
        win_color = "normal" if win_rate >= 50 else "inverse"
        st.metric(
            "Positive Periods", 
            f"{win_rate:.1f}%",
            delta=f"{win_rate-50:+.1f}% vs 50%" if win_rate != 0 else None,
            delta_color=win_color
        )


def _display_risk_metrics(metrics: Dict[str, Any]) -> None:
    """Display risk-focused metrics including volatility and drawdown analysis with benchmark comparisons."""
    st.markdown("#### ⚠️ Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Volatility Measures**")
        
        # Annual volatility with benchmark comparison
        volatility = metrics.get('volatility', 0) * 100
        benchmark_volatility = metrics.get('benchmark_volatility', 0) * 100
        
        if benchmark_volatility > 0:
            vol_diff = volatility - benchmark_volatility
            vol_delta_color = "inverse" if vol_diff > 0 else "normal"  # Higher vol is worse
            st.metric(
                "Annual Volatility", 
                f"{volatility:.2f}%",
                delta=f"{vol_diff:+.2f}% vs benchmark",
                delta_color=vol_delta_color
            )
        else:
            st.metric("Annual Volatility", f"{volatility:.2f}%")
        
        # Downside deviation
        downside_deviation = metrics.get('downside_deviation', 0) * 100
        st.metric("Downside Deviation", f"{downside_deviation:.2f}%")
        
        # Tracking error vs benchmark
        tracking_error = metrics.get('tracking_error', 0) * 100
        if tracking_error:
            st.metric("Tracking Error", f"{tracking_error:.2f}%")
    
    with col2:
        st.markdown("**Drawdown Analysis**")
        
        # Maximum drawdown with benchmark comparison
        max_drawdown = metrics.get('max_drawdown', 0) * 100
        benchmark_max_drawdown = metrics.get('benchmark_max_drawdown', 0) * 100
        
        if benchmark_max_drawdown != 0:
            dd_improvement = benchmark_max_drawdown - max_drawdown  # Positive if strategy is better
            dd_delta_color = "normal" if dd_improvement > 0 else "inverse"
            st.metric(
                "Maximum Drawdown", 
                f"{max_drawdown:.2f}%",
                delta=f"{dd_improvement:+.2f}% vs benchmark",
                delta_color=dd_delta_color,
                help="Positive delta means lower drawdown (better)"
            )
        else:
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
        
        # Average drawdown
        avg_drawdown = metrics.get('avg_drawdown', 0) * 100
        if avg_drawdown:
            st.metric("Average Drawdown", f"{avg_drawdown:.2f}%")
        
        # Recovery time
        recovery_time = metrics.get('avg_recovery_time', 0)
        if recovery_time:
            st.metric("Avg Recovery (days)", f"{recovery_time:.0f}")
    
    with col3:
        st.markdown("**Tail Risk Measures**")
        
        # Value at Risk (95%)
        var_95 = metrics.get('var_95', 0) * 100
        if var_95:
            st.metric("VaR (95%)", f"{var_95:.2f}%")
        
        # Conditional Value at Risk
        cvar_95 = metrics.get('cvar_95', 0) * 100
        if cvar_95:
            st.metric("CVaR (95%)", f"{cvar_95:.2f}%")
        
        # Skewness
        skewness = metrics.get('skewness', 0)
        if skewness != 0:
            st.metric("Skewness", f"{skewness:.2f}")


def _display_risk_adjusted_metrics(metrics: Dict[str, Any]) -> None:
    """Display risk-adjusted performance ratios and efficiency measures."""
    st.markdown("#### ⚖️ Risk-Adjusted Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Classic Ratios**")
        
        # Sharpe ratio with benchmark comparison
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        benchmark_sharpe = metrics.get('benchmark_sharpe_ratio', 0)
        
        if benchmark_sharpe != 0:
            sharpe_diff = sharpe_ratio - benchmark_sharpe
            sharpe_delta_color = "normal" if sharpe_diff > 0 else "inverse"
            st.metric(
                "Sharpe Ratio", 
                f"{sharpe_ratio:.2f}",
                delta=f"{sharpe_diff:+.2f} vs benchmark",
                delta_color=sharpe_delta_color
            )
        else:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Sortino ratio
        sortino_ratio = metrics.get('sortino_ratio', 0)
        st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        
        # Calmar ratio
        calmar_ratio = metrics.get('calmar_ratio', 0)
        st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
    
    with col2:
        st.markdown("**Information Ratios**")
        
        # Information ratio
        information_ratio = metrics.get('information_ratio', 0)
        st.metric("Information Ratio", f"{information_ratio:.2f}")
        
        # Treynor ratio
        treynor_ratio = metrics.get('treynor_ratio', 0)
        if treynor_ratio:
            st.metric("Treynor Ratio", f"{treynor_ratio:.2f}")
        
        # Jensen's Alpha
        jensen_alpha = metrics.get('jensen_alpha', 0) * 100
        if jensen_alpha:
            st.metric("Jensen's Alpha", f"{jensen_alpha:.2f}%")
    
    with col3:
        st.markdown("**Efficiency Measures**")
        
        # Return/Risk ratio
        return_risk_ratio = metrics.get('return_risk_ratio', 0)
        if return_risk_ratio:
            st.metric("Return/Risk Ratio", f"{return_risk_ratio:.2f}")
        
        # Omega ratio
        omega_ratio = metrics.get('omega_ratio', 0)
        if omega_ratio:
            st.metric("Omega Ratio", f"{omega_ratio:.2f}")
        
        # Gain-to-Pain ratio
        gain_pain_ratio = metrics.get('gain_pain_ratio', 0)
        if gain_pain_ratio:
            st.metric("Gain-to-Pain Ratio", f"{gain_pain_ratio:.2f}")


def _display_trading_metrics(metrics: Dict[str, Any]) -> None:
    """Display trading-specific performance metrics and execution statistics."""
    st.markdown("#### 🎯 Trading Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Trade Analysis**")
        
        # Total number of trades
        total_trades = metrics.get('total_trades', 0)
        st.metric("Total Trades", f"{total_trades:,}")
        
        # Win rate
        win_rate = metrics.get('win_rate', 0) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Average trade return
        avg_trade_return = metrics.get('avg_trade_return', 0) * 100
        st.metric("Avg Trade Return", f"{avg_trade_return:.2f}%")
    
    with col2:
        st.markdown("**Profit Analysis**")
        
        # Profit factor
        profit_factor = metrics.get('profit_factor', 0)
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        # Average winning trade
        avg_win = metrics.get('avg_winning_trade', 0) * 100
        st.metric("Avg Winning Trade", f"{avg_win:.2f}%")
        
        # Average losing trade
        avg_loss = metrics.get('avg_losing_trade', 0) * 100
        st.metric("Avg Losing Trade", f"{avg_loss:.2f}%")
    
    with col3:
        st.markdown("**Execution Quality**")
        
        # Largest winning trade
        largest_win = metrics.get('largest_winning_trade', 0) * 100
        st.metric("Largest Win", f"{largest_win:.2f}%")
        
        # Largest losing trade
        largest_loss = metrics.get('largest_losing_trade', 0) * 100
        st.metric("Largest Loss", f"{largest_loss:.2f}%")
        
        # Consecutive wins/losses
        max_consecutive_wins = metrics.get('max_consecutive_wins', 0)
        if max_consecutive_wins:
            st.metric("Max Consecutive Wins", f"{max_consecutive_wins}")


def display_benchmark_comparison(strategy_metrics: Dict[str, Any], 
                               benchmark_metrics: Dict[str, Any],
                               benchmark_name: str = "Benchmark") -> None:
    """Display side-by-side comparison with benchmark."""
    
    st.subheader(f"📊 Strategy vs {benchmark_name} Comparison")
    
    comparison_data = {
        "Metric": [
            "Total Return",
            "Annual Return", 
            "Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
            "Win Rate"
        ],
        "Strategy": [
            f"{strategy_metrics.get('total_return', 0) * 100:.2f}%",
            f"{strategy_metrics.get('annual_return', 0) * 100:.2f}%",
            f"{strategy_metrics.get('annual_volatility', 0) * 100:.2f}%",
            f"{strategy_metrics.get('sharpe_ratio', 0):.2f}",
            f"{strategy_metrics.get('max_drawdown', 0) * 100:.2f}%",
            f"{strategy_metrics.get('win_rate', 0) * 100:.1f}%"
        ],
        benchmark_name: [
            f"{benchmark_metrics.get('total_return', 0) * 100:.2f}%",
            f"{benchmark_metrics.get('annual_return', 0) * 100:.2f}%",
            f"{benchmark_metrics.get('annual_volatility', 0) * 100:.2f}%",
            f"{benchmark_metrics.get('sharpe_ratio', 0):.2f}",
            f"{benchmark_metrics.get('max_drawdown', 0) * 100:.2f}%",
            f"{benchmark_metrics.get('win_rate', 0) * 100:.1f}%"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True
    )
    
    # Add delta metrics
    st.write("**Performance Difference:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        return_diff = (strategy_metrics.get('annual_return', 0) - 
                      benchmark_metrics.get('annual_return', 0)) * 100
        st.metric(
            "Excess Return (Annual)",
            f"{return_diff:.2f}%",
            delta=f"{return_diff:.2f}%" if return_diff != 0 else None
        )
    
    with col2:
        sharpe_diff = (strategy_metrics.get('sharpe_ratio', 0) - 
                      benchmark_metrics.get('sharpe_ratio', 0))
        st.metric(
            "Sharpe Difference",
            f"{sharpe_diff:.2f}",
            delta=f"{sharpe_diff:.2f}" if sharpe_diff != 0 else None
        )
    
    with col3:
        dd_diff = (benchmark_metrics.get('max_drawdown', 0) - 
                  strategy_metrics.get('max_drawdown', 0)) * 100
        st.metric(
            "Drawdown Improvement",
            f"{dd_diff:.2f}%",
            delta=f"{dd_diff:.2f}%" if dd_diff != 0 else None
        )


def create_metrics_summary_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a radar chart showing key metrics."""
    
    categories = ['Return', 'Risk-Adj Return', 'Low Risk', 'Consistency', 'Efficiency']
    
    # Normalize metrics to 0-1 scale for radar chart
    values = [
        min(max(metrics.get('annual_return', 0) * 5, 0), 1),  # Return (scaled)
        min(max(metrics.get('sharpe_ratio', 0) / 3, 0), 1),   # Sharpe (scaled)
        min(max(1 - abs(metrics.get('max_drawdown', 0)) * 2, 0), 1),  # Low risk
        min(max(metrics.get('win_rate', 0), 0), 1),           # Consistency
        min(max(metrics.get('profit_factor', 1) / 3, 0), 1)   # Efficiency
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Strategy Performance',
        line_color='rgb(255, 107, 107)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Strategy Performance Overview",
        height=400
    )
    
    return fig


def display_trade_distribution(trades_df: pd.DataFrame) -> None:
    """Display trade return distribution analysis."""
    
    if trades_df.empty:
        st.warning("No trades available for analysis.")
        return
    
    st.subheader("📈 Trade Analysis")
    
    # Trade statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
    
    with col2:
        winning_trades = len(trades_df[trades_df['return'] > 0])
        win_rate = winning_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        avg_win = trades_df[trades_df['return'] > 0]['return'].mean() * 100 if winning_trades > 0 else 0
        st.metric("Avg Win", f"{avg_win:.2f}%")
    
    with col4:
        losing_trades = trades_df[trades_df['return'] < 0]
        avg_loss = losing_trades['return'].mean() * 100 if len(losing_trades) > 0 else 0
        st.metric("Avg Loss", f"{avg_loss:.2f}%")
    
    # Trade distribution histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=trades_df['return'] * 100,
        nbinsx=30,
        name='Trade Returns',
        marker_color='rgb(255, 107, 107)',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Trade Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True) 