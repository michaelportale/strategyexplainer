"""Metric display components for performance visualization."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def display_key_metrics(metrics: Dict[str, Any]) -> None:
    """Display key performance metrics in a card layout."""
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = metrics.get('total_return', 0) * 100
        color = "normal" if total_return >= 0 else "inverse"
        st.metric(
            label="Total Return",
            value=f"{total_return:.2f}%",
            delta=None
        )
    
    with col2:
        annual_return = metrics.get('annual_return', 0) * 100
        st.metric(
            label="Annual Return",
            value=f"{annual_return:.2f}%",
            delta=None
        )
    
    with col3:
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe_ratio:.2f}",
            delta=None
        )
    
    with col4:
        max_drawdown = metrics.get('max_drawdown', 0) * 100
        st.metric(
            label="Max Drawdown",
            value=f"{max_drawdown:.2f}%",
            delta=None
        )


def display_detailed_metrics(metrics: Dict[str, Any]) -> None:
    """Display detailed performance metrics in expandable sections."""
    
    # Returns Section
    with st.expander("📈 Return Metrics", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Return Statistics**")
            return_metrics = {
                "Total Return": f"{metrics.get('total_return', 0) * 100:.2f}%",
                "Annual Return": f"{metrics.get('annual_return', 0) * 100:.2f}%",
                "Monthly Return": f"{metrics.get('monthly_return', 0) * 100:.2f}%",
                "Daily Return (Avg)": f"{metrics.get('daily_return_avg', 0) * 100:.3f}%",
            }
            
            for label, value in return_metrics.items():
                st.write(f"• **{label}:** {value}")
        
        with col2:
            st.write("**Volatility Metrics**")
            vol_metrics = {
                "Annual Volatility": f"{metrics.get('annual_volatility', 0) * 100:.2f}%",
                "Daily Volatility": f"{metrics.get('daily_volatility', 0) * 100:.2f}%",
                "Downside Deviation": f"{metrics.get('downside_deviation', 0) * 100:.2f}%",
                "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.2f}",
            }
            
            for label, value in vol_metrics.items():
                st.write(f"• **{label}:** {value}")
    
    # Risk Metrics Section
    with st.expander("⚠️ Risk Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Drawdown Analysis**")
            drawdown_metrics = {
                "Max Drawdown": f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
                "Avg Drawdown": f"{metrics.get('avg_drawdown', 0) * 100:.2f}%",
                "Max Drawdown Duration": f"{metrics.get('max_dd_duration', 0)} days",
                "Recovery Factor": f"{metrics.get('recovery_factor', 0):.2f}",
            }
            
            for label, value in drawdown_metrics.items():
                st.write(f"• **{label}:** {value}")
        
        with col2:
            st.write("**Risk-Adjusted Returns**")
            risk_metrics = {
                "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
                "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
                "Information Ratio": f"{metrics.get('information_ratio', 0):.2f}",
                "Beta": f"{metrics.get('beta', 0):.2f}",
            }
            
            for label, value in risk_metrics.items():
                st.write(f"• **{label}:** {value}")
    
    # Trading Statistics
    with st.expander("📊 Trading Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trade Analysis**")
            trade_metrics = {
                "Total Trades": f"{metrics.get('total_trades', 0)}",
                "Win Rate": f"{metrics.get('win_rate', 0) * 100:.1f}%",
                "Avg Win": f"{metrics.get('avg_win', 0) * 100:.2f}%",
                "Avg Loss": f"{metrics.get('avg_loss', 0) * 100:.2f}%",
            }
            
            for label, value in trade_metrics.items():
                st.write(f"• **{label}:** {value}")
        
        with col2:
            st.write("**Trade Efficiency**")
            efficiency_metrics = {
                "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
                "Expectancy": f"{metrics.get('expectancy', 0) * 100:.3f}%",
                "Kelly %": f"{metrics.get('kelly_criterion', 0) * 100:.1f}%",
                "MAR Ratio": f"{metrics.get('mar_ratio', 0):.2f}",
            }
            
            for label, value in efficiency_metrics.items():
                st.write(f"• **{label}:** {value}")


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