"""Professional Results Layout Component for Strategy Analyzer.

This module provides a sophisticated, institutional-grade results layout that organizes
complex strategy analysis into intuitive tabs for professional presentation.

The layout follows best practices for financial analysis presentation:
- Performance Overview: Key metrics and equity curve
- Detailed Metrics: Comprehensive statistics and risk analysis  
- Trade Analysis: Individual trade records and distribution analysis
- AI Report: GPT-generated insights and commentary
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import plotly.graph_objects as go

from app.components.metrics_cards import (
    display_key_metrics, 
    display_detailed_metrics,
    display_benchmark_comparison
)
from app.components.charts import (
    create_equity_curve_chart,
    create_underwater_drawdown_chart,
    create_returns_distribution_chart,
    create_price_and_signals_chart,
    create_rolling_metrics_chart,
    create_monthly_returns_heatmap,
    create_trade_distribution_chart
)


def create_professional_metrics_row(strategy_metrics: Dict[str, Any]) -> None:
    """Create a professional row of 4 key metric cards at the top."""
    
    # Extract key metrics
    total_return = strategy_metrics.get('total_return', 0) * 100
    cagr = strategy_metrics.get('annualized_return', 0) * 100
    sharpe_ratio = strategy_metrics.get('sharpe_ratio', 0)
    max_drawdown = strategy_metrics.get('max_drawdown', 0) * 100
    
    # Create 4-column layout for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric(
            label="📈 Total Return",
            value=f"{total_return:.1f}%",
            delta=None,
            help="Cumulative return over the entire backtest period"
        )
    
    with col2:
        delta_color = "normal" if cagr >= 0 else "inverse"
        st.metric(
            label="📊 CAGR", 
            value=f"{cagr:.1f}%",
            delta=None,
            help="Compound Annual Growth Rate - annualized return"
        )
    
    with col3:
        # Sharpe ratio color coding
        if sharpe_ratio >= 1.0:
            label_color = "🟢"
        elif sharpe_ratio >= 0.5:
            label_color = "🟡"
        else:
            label_color = "🔴"
            
        st.metric(
            label=f"{label_color} Sharpe Ratio",
            value=f"{sharpe_ratio:.2f}",
            delta=None,
            help="Risk-adjusted return measure (return per unit of risk)"
        )
    
    with col4:
        # Max drawdown is always negative, so show as positive with warning color
        st.metric(
            label="⚠️ Max Drawdown",
            value=f"{abs(max_drawdown):.1f}%",
            delta=None,
            help="Maximum peak-to-trough decline in portfolio value"
        )


def display_professional_results(config: Dict[str, Any], 
                                price_data: pd.DataFrame,
                                signals: pd.DataFrame,
                                equity_curve: pd.DataFrame,
                                trades: pd.DataFrame,
                                strategy_metrics: Dict[str, Any],
                                benchmark_metrics: Optional[Dict[str, Any]] = None,
                                benchmark_data: Optional[pd.DataFrame] = None,
                                gpt_insights: Optional[Dict[str, Any]] = None):
    """
    Display comprehensive analysis results in a professional tabbed layout.
    
    This function creates an institutional-grade results dashboard with organized
    tabs for different aspects of the analysis, following best practices for
    financial analysis presentation.
    
    Args:
        config: User configuration dictionary
        price_data: Historical price data for the analyzed symbol
        signals: Generated trading signals with entry/exit points
        equity_curve: Portfolio value over time
        trades: Detailed trade records with P&L
        strategy_metrics: Comprehensive performance statistics
        benchmark_metrics: Optional benchmark performance for comparison
        benchmark_data: Optional benchmark price data
        gpt_insights: Optional AI-generated insights and commentary
    """
    
    # Strategy Information Header
    st.markdown("### 📈 Strategy Analysis Results")
    
    strategy_name = config.get('strategy_type', 'Strategy')
    symbol = config.get('data_config', {}).get('ticker', 'UNKNOWN')
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Strategy:** {strategy_name}")
    with col2:
        st.markdown(f"**Symbol:** {symbol}")
    with col3:
        st.markdown(f"**Period:** {len(equity_curve)} days")
    
    st.markdown("---")
    
    # Professional 4-metric row at the top
    create_professional_metrics_row(strategy_metrics)
    
    st.markdown("---")
    
    # Create professional tabbed interface
    tabs = st.tabs([
        "📊 Performance Overview", 
        "📋 Detailed Metrics", 
        "🎯 Trade Analysis", 
        "🤖 AI Report"
    ])
    
    # Tab 1: Performance Overview
    with tabs[0]:
        render_performance_overview_tab(
            equity_curve, price_data, signals, strategy_metrics, 
            benchmark_data, benchmark_metrics
        )
    
    # Tab 2: Detailed Metrics
    with tabs[1]:
        render_detailed_metrics_tab(strategy_metrics, benchmark_metrics)
    
    # Tab 3: Trade Analysis
    with tabs[2]:
        render_trade_analysis_tab(trades, strategy_metrics)
    
    # Tab 4: AI Report
    with tabs[3]:
        render_ai_report_tab(gpt_insights, strategy_metrics, config)


def render_performance_overview_tab(equity_curve: pd.DataFrame,
                                   price_data: pd.DataFrame,
                                   signals: pd.DataFrame,
                                   strategy_metrics: Dict[str, Any],
                                   benchmark_data: Optional[pd.DataFrame] = None,
                                   benchmark_metrics: Optional[Dict[str, Any]] = None):
    """Render the Performance Overview tab with key charts and metrics."""
    
    st.subheader("🎯 Performance Summary")
    
    # Two-column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Equity Curve")
        equity_fig = create_equity_curve_chart(
            equity_curve, 
            benchmark_data=benchmark_data,
            benchmark_name="Benchmark" if benchmark_data is not None else None
        )
        st.plotly_chart(equity_fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 📉 Drawdown Analysis")
        drawdown_fig = create_underwater_drawdown_chart(equity_curve)
        st.plotly_chart(drawdown_fig, use_container_width=True)
    
    # Price chart with signals (full width)
    st.markdown("##### 💹 Price Action & Signals")
    price_fig = create_price_and_signals_chart(
        price_data, signals, 
        show_volume=True,
        title=f"Price Action with Trading Signals"
    )
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Benchmark comparison if available
    if benchmark_metrics:
        st.markdown("---")
        st.subheader("🔄 Benchmark Comparison")
        display_benchmark_comparison(strategy_metrics, benchmark_metrics)


def render_detailed_metrics_tab(strategy_metrics: Dict[str, Any],
                               benchmark_metrics: Optional[Dict[str, Any]] = None):
    """Render the Detailed Metrics tab with comprehensive statistics."""
    
    st.subheader("📊 Comprehensive Performance Analysis")
    
    # Display detailed metrics in organized sections
    display_detailed_metrics(strategy_metrics)
    
    # Additional analysis charts
    st.markdown("---")
    st.subheader("📈 Performance Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'returns' in strategy_metrics:
            st.markdown("##### 📊 Returns Distribution")
            returns_data = strategy_metrics['returns']
            if hasattr(returns_data, 'dropna'):
                returns_fig = create_returns_distribution_chart(returns_data.dropna())
                st.plotly_chart(returns_fig, use_container_width=True)
    
    with col2:
        if 'equity_curve' in strategy_metrics:
            st.markdown("##### 🗓️ Monthly Returns Heatmap")
            equity_data = strategy_metrics['equity_curve']
            if hasattr(equity_data, 'index'):
                monthly_fig = create_monthly_returns_heatmap(equity_data)
                st.plotly_chart(monthly_fig, use_container_width=True)


def render_trade_analysis_tab(trades: pd.DataFrame, strategy_metrics: Dict[str, Any]):
    """Render the Trade Analysis tab with individual trade analysis."""
    
    st.subheader("🎯 Trade-Level Analysis")
    
    if trades is not None and len(trades) > 0:
        # Trade statistics summary
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0]) if 'pnl' in trades.columns else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade = trades['pnl'].mean() if 'pnl' in trades.columns else 0
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Winning Trades", winning_trades)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Avg Trade P&L", f"${avg_trade:.2f}")
        
        st.markdown("---")
        
        # Trade distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Trade P&L Distribution")
            if 'pnl' in trades.columns:
                trade_dist_fig = create_trade_distribution_chart(trades)
                st.plotly_chart(trade_dist_fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 Trade Duration Analysis")
            if 'duration' in trades.columns:
                duration_fig = go.Figure()
                duration_fig.add_trace(go.Histogram(
                    x=trades['duration'],
                    nbinsx=20,
                    name="Trade Duration",
                    marker_color='lightblue'
                ))
                duration_fig.update_layout(
                    title="Trade Duration Distribution",
                    xaxis_title="Duration (days)",
                    yaxis_title="Number of Trades",
                    height=400
                )
                st.plotly_chart(duration_fig, use_container_width=True)
        
        # Detailed trade log
        st.markdown("---")
        st.subheader("📋 Detailed Trade Log")
        
        # Format trades table for display
        display_trades = trades.copy()
        if 'entry_date' in display_trades.columns:
            display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
        if 'exit_date' in display_trades.columns:
            display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
        if 'pnl' in display_trades.columns:
            display_trades['pnl'] = display_trades['pnl'].round(2)
        
        st.dataframe(
            display_trades, 
            use_container_width=True,
            height=400
        )
        
        # CSV download
        csv = trades.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log as CSV",
            data=csv,
            file_name=f"trade_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No individual trade data available for this analysis.")


def render_ai_report_tab(gpt_insights: Optional[Dict[str, Any]], 
                        strategy_metrics: Dict[str, Any],
                        config: Dict[str, Any]):
    """Render the AI Report tab with GPT-generated insights."""
    
    st.subheader("🤖 AI-Powered Analysis")
    
    if gpt_insights and config.get('enable_gpt', False):
        # Display AI insights if available
        if 'strategy_overview' in gpt_insights:
            st.markdown("##### 📝 Strategy Overview")
            st.markdown(gpt_insights['strategy_overview'])
            
        if 'performance_summary' in gpt_insights:
            st.markdown("##### 📊 Performance Analysis")
            st.markdown(gpt_insights['performance_summary'])
            
        if 'risk_assessment' in gpt_insights:
            st.markdown("##### ⚠️ Risk Assessment")
            st.markdown(gpt_insights['risk_assessment'])
            
        if 'recommendations' in gpt_insights:
            st.markdown("##### 💡 Recommendations")
            st.markdown(gpt_insights['recommendations'])
            
        # Trade insights if available
        if 'trade_insights' in gpt_insights:
            st.markdown("---")
            st.markdown("##### 🎯 Trade Insights")
            for insight in gpt_insights['trade_insights'][:5]:  # Show top 5
                st.markdown(f"• {insight}")
    else:
        # Provide basic analysis when AI is not available
        st.markdown("##### 📊 Basic Performance Summary")
        
        total_return = strategy_metrics.get('total_return', 0) * 100
        sharpe_ratio = strategy_metrics.get('sharpe_ratio', 0)
        max_drawdown = strategy_metrics.get('max_drawdown', 0) * 100
        
        summary_text = f"""
        **Performance Summary:**
        
        This strategy achieved a total return of {total_return:.1f}% over the backtest period.
        
        **Risk-Adjusted Performance:**
        - Sharpe Ratio: {sharpe_ratio:.2f}
        - Maximum Drawdown: {abs(max_drawdown):.1f}%
        
        **Key Observations:**
        """
        
        if sharpe_ratio > 1.0:
            summary_text += "\n- ✅ Strong risk-adjusted returns (Sharpe > 1.0)"
        elif sharpe_ratio > 0.5:
            summary_text += "\n- 🔶 Moderate risk-adjusted returns (Sharpe 0.5-1.0)"
        else:
            summary_text += "\n- ⚠️ Weak risk-adjusted returns (Sharpe < 0.5)"
            
        if abs(max_drawdown) < 10:
            summary_text += "\n- ✅ Low maximum drawdown (< 10%)"
        elif abs(max_drawdown) < 20:
            summary_text += "\n- 🔶 Moderate maximum drawdown (10-20%)"
        else:
            summary_text += "\n- ⚠️ High maximum drawdown (> 20%)"
        
        st.markdown(summary_text)
        
        st.info("💡 Enable AI features in the sidebar for detailed GPT-powered analysis and insights.") 