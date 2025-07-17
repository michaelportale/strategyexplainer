"""Enhanced Analytics Layout Component for Professional Strategy Analysis.

This module provides a comprehensive, professionally organized analytics layout that
presents performance metrics, visualizations, and insights in a structured format
suitable for institutional analysis and decision-making.

LAYOUT PHILOSOPHY:
=================
The layout follows a hierarchical information architecture:

1. EXECUTIVE SUMMARY: Key metrics and performance overview
2. PERFORMANCE ANALYSIS: Returns, benchmarks, and growth analysis  
3. RISK ANALYSIS: Drawdowns, volatility, and risk metrics
4. TRADING ANALYSIS: Trade statistics and execution quality
5. ADVANCED ANALYTICS: Distribution analysis and statistical insights

DESIGN PRINCIPLES:
=================
- Progressive disclosure: Most important information first
- Contextual grouping: Related metrics and charts together
- Visual hierarchy: Clear section separation and emphasis
- Interactive exploration: Expandable sections and detailed views
- Export capabilities: Professional reporting and sharing

PROFESSIONAL FEATURES:
=====================
- Institutional-grade metric categorization
- Enhanced color coding and visual cues
- Comprehensive statistical analysis
- Benchmark comparison and attribution
- Risk attribution and decomposition
- Performance attribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import plotly.graph_objects as go
from datetime import datetime, timedelta

from app.components.metrics_cards import (
    display_key_metrics, 
    display_detailed_metrics,
    display_benchmark_comparison,
    METRICS_STYLING
)
from app.components.charts import (
    create_equity_curve_chart,
    create_multi_strategy_comparison_chart,
    create_underwater_drawdown_chart,
    create_rolling_metrics_chart,
    create_monthly_returns_heatmap,
    create_returns_distribution_chart,
    create_trade_distribution_chart,
    CHART_COLORS
)


class EnhancedAnalyticsLayout:
    """Professional analytics layout manager for strategy performance analysis."""
    
    def __init__(self, theme: str = "professional"):
        """Initialize the analytics layout manager.
        
        Args:
            theme: Visual theme for the layout ("professional", "dark", "light")
        """
        self.theme = theme
        self.section_config = {
            "show_executive_summary": True,
            "show_performance_section": True,
            "show_risk_section": True,
            "show_trading_section": True,
            "show_advanced_analytics": True,
            "enable_exports": True,
            "default_expanded": ["executive_summary", "performance"]
        }
    
    def render_complete_analysis(self,
                               strategy_data: Dict[str, Any],
                               benchmark_data: Optional[Dict[str, Any]] = None,
                               trades_data: Optional[pd.DataFrame] = None,
                               strategy_name: str = "Strategy") -> None:
        """Render complete professional analytics layout.
        
        This is the main entry point for displaying comprehensive strategy analysis
        with all sections, metrics, and visualizations organized professionally.
        
        Args:
            strategy_data: Dictionary containing strategy performance data including:
                         - equity_curve: DataFrame with equity values over time
                         - metrics: Dictionary with calculated performance metrics
                         - returns: Series with period returns
            benchmark_data: Optional benchmark data for comparison
            trades_data: Optional individual trade records
            strategy_name: Display name for the strategy
        """
        # Render page header with strategy overview
        self._render_page_header(strategy_name, strategy_data)
        
        # Executive Summary Section
        if self.section_config["show_executive_summary"]:
            self._render_executive_summary(strategy_data, benchmark_data)
        
        # Main analysis sections in tabs
        self._render_main_analysis_tabs(strategy_data, benchmark_data, trades_data)
        
        # Export and sharing options
        if self.section_config["enable_exports"]:
            self._render_export_section(strategy_data, strategy_name)
    
    def _render_page_header(self, strategy_name: str, strategy_data: Dict[str, Any]) -> None:
        """Render professional page header with strategy overview."""
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, {CHART_COLORS['primary']}, {CHART_COLORS['secondary']}); 
                    padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0; font-size: 28px; font-weight: 600;'>
                📊 {strategy_name} - Performance Analysis
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 16px;'>
                Comprehensive institutional-grade performance analytics and risk assessment
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats row
        if 'metrics' in strategy_data:
            metrics = strategy_data['metrics']
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_return = metrics.get('total_return', 0) * 100
                st.metric("📈 Total Return", f"{total_return:.2f}%")
            
            with col2:
                sharpe = metrics.get('sharpe_ratio', 0)
                st.metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}")
            
            with col3:
                max_dd = metrics.get('max_drawdown', 0) * 100
                st.metric("📉 Max Drawdown", f"{max_dd:.2f}%")
            
            with col4:
                volatility = metrics.get('annual_volatility', 0) * 100
                st.metric("📊 Volatility", f"{volatility:.1f}%")
            
            with col5:
                win_rate = metrics.get('win_rate', 0) * 100
                st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    
    def _render_executive_summary(self, 
                                strategy_data: Dict[str, Any],
                                benchmark_data: Optional[Dict[str, Any]] = None) -> None:
        """Render executive summary section with key insights."""
        with st.expander("📋 Executive Summary", expanded=True):
            st.markdown("### Key Performance Highlights")
            
            # Enhanced metrics display
            if 'metrics' in strategy_data:
                display_key_metrics(strategy_data['metrics'])
            
            # Performance vs benchmark comparison
            if benchmark_data and 'metrics' in benchmark_data:
                st.markdown("---")
                display_benchmark_comparison(
                    strategy_data['metrics'], 
                    benchmark_data['metrics'],
                    benchmark_data.get('name', 'Benchmark')
                )
            
            # AI-powered insights (if available)
            self._render_ai_insights(strategy_data)
    
    def _render_main_analysis_tabs(self,
                                 strategy_data: Dict[str, Any],
                                 benchmark_data: Optional[Dict[str, Any]] = None,
                                 trades_data: Optional[pd.DataFrame] = None) -> None:
        """Render main analysis sections in organized tabs."""
        
        tab_names = ["📈 Performance", "⚠️ Risk Analysis", "🎯 Trading Analysis", "📊 Advanced Analytics"]
        tabs = st.tabs(tab_names)
        
        # Performance Analysis Tab
        with tabs[0]:
            self._render_performance_section(strategy_data, benchmark_data)
        
        # Risk Analysis Tab  
        with tabs[1]:
            self._render_risk_section(strategy_data, benchmark_data)
        
        # Trading Analysis Tab
        with tabs[2]:
            self._render_trading_section(strategy_data, trades_data)
        
        # Advanced Analytics Tab
        with tabs[3]:
            self._render_advanced_analytics_section(strategy_data, benchmark_data)
    
    def _render_performance_section(self,
                                  strategy_data: Dict[str, Any],
                                  benchmark_data: Optional[Dict[str, Any]] = None) -> None:
        """Render comprehensive performance analysis section."""
        st.markdown("## 📈 Performance Analysis")
        
        # Enhanced equity curve with benchmark comparison and insights
        if 'equity_curve' in strategy_data:
            st.markdown("### Equity Curve Analysis")
            st.markdown("Track portfolio value growth over time compared to benchmark performance.")
            
            equity_df = strategy_data['equity_curve']
            benchmark_df = benchmark_data.get('equity_curve') if benchmark_data else None
            benchmark_name = benchmark_data.get('name', 'Benchmark') if benchmark_data else None
            
            # Add performance comparison insights
            if benchmark_df is not None:
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate performance metrics for comparison
                strategy_total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1) * 100
                benchmark_total_return = (benchmark_df['equity'].iloc[-1] / benchmark_df['equity'].iloc[0] - 1) * 100
                excess_return = strategy_total_return - benchmark_total_return
                
                with col1:
                    st.metric("Strategy Return", f"{strategy_total_return:.2f}%")
                
                with col2:
                    st.metric(f"{benchmark_name} Return", f"{benchmark_total_return:.2f}%")
                
                with col3:
                    delta_color = "normal" if excess_return >= 0 else "inverse"
                    st.metric("Excess Return", f"{excess_return:+.2f}%", 
                             delta=f"{excess_return:+.2f}%", delta_color=delta_color)
                
                with col4:
                    # Calculate outperformance percentage
                    if benchmark_total_return != 0:
                        outperformance = (strategy_total_return / benchmark_total_return - 1) * 100
                        st.metric("Relative Performance", f"{outperformance:+.1f}%")
                    else:
                        st.metric("Relative Performance", "N/A")
            
            fig = create_equity_curve_chart(equity_df, benchmark_df, benchmark_name)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rolling performance metrics with enhanced options
        st.markdown("### Rolling Performance Analysis")
        st.markdown("Track how risk-adjusted performance evolves over time with rolling metrics.")
        
        if 'equity_curve' in strategy_data:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                rolling_window = st.selectbox(
                    "Rolling Window:",
                    options=[42, 63, 126, 189, 252, 504],  # 2M, 3M, 6M, 9M, 1Y, 2Y
                    index=2,  # Default to 6 months
                    format_func=lambda x: f"{x//21} Months" if x < 252 else f"{x//252} Year{'s' if x > 252 else ''}"
                )
                
                # Add explanation for selected window
                if rolling_window <= 63:
                    st.info("💡 Short-term trend analysis")
                elif rolling_window <= 189:
                    st.info("📊 Medium-term pattern analysis")
                else:
                    st.info("📈 Long-term consistency analysis")
                
                # Show rolling metrics summary if data is available
                if 'equity_curve' in strategy_data and len(strategy_data['equity_curve']) >= rolling_window:
                    returns = strategy_data['equity_curve']['equity'].pct_change().dropna()
                    
                    # Calculate current rolling metrics
                    rolling_return = returns.rolling(rolling_window).mean().iloc[-1] * 252 * 100 if len(returns) > rolling_window else 0
                    rolling_vol = returns.rolling(rolling_window).std().iloc[-1] * np.sqrt(252) * 100 if len(returns) > rolling_window else 0
                    rolling_sharpe = rolling_return / rolling_vol if rolling_vol > 0 else 0
                    
                    st.markdown("**Current Rolling Metrics:**")
                    st.metric("Rolling Return", f"{rolling_return:.2f}%")
                    st.metric("Rolling Volatility", f"{rolling_vol:.2f}%")
                    st.metric("Rolling Sharpe", f"{rolling_sharpe:.2f}")
            
            with col2:
                # Check data sufficiency for rolling analysis
                equity_length = len(strategy_data['equity_curve'])
                
                if equity_length >= rolling_window + 30:  # Need buffer for meaningful rolling analysis
                    benchmark_df = benchmark_data.get('equity_curve') if benchmark_data else None
                    fig = create_rolling_metrics_chart(
                        strategy_data['equity_curve'], 
                        window=rolling_window,
                        benchmark_data=benchmark_df
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    required_days = rolling_window + 30
                    st.warning(f"📊 Insufficient data for {rolling_window//21}-month rolling analysis")
                    st.info(f"Need at least {required_days} days of data. Current: {equity_length} days")
                    
                    # Show a simplified version if we have some data
                    if equity_length > 60:
                        st.markdown("**Showing simplified rolling analysis with available data:**")
                        simple_window = min(21, equity_length // 3)  # Use 1 month or 1/3 of available data
                        benchmark_df = benchmark_data.get('equity_curve') if benchmark_data else None
                        fig = create_rolling_metrics_chart(
                            strategy_data['equity_curve'], 
                            window=simple_window,
                            benchmark_data=benchmark_df
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap with enhanced analysis
        st.markdown("### Seasonality Analysis")
        st.markdown("Analyze strategy performance patterns across months and years to identify seasonal trends.")
        
        if 'returns' in strategy_data:
            returns = strategy_data['returns']
            
            # Check if we have sufficient data for meaningful seasonality analysis
            months_of_data = len(returns) / 21  # Approximate months of data
            
            if months_of_data >= 6:  # At least 6 months for meaningful analysis
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    benchmark_returns = benchmark_data.get('returns') if benchmark_data else None
                    fig = create_monthly_returns_heatmap(returns, benchmark_returns, show_statistics=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Show seasonality insights
                    st.markdown("**🗓️ Seasonality Insights**")
                    
                    # Calculate monthly averages
                    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                    if len(monthly_returns) > 0:
                        monthly_stats = monthly_returns.groupby(monthly_returns.index.month).agg(['mean', 'std'])
                        
                        best_month = monthly_stats['mean'].idxmax()
                        worst_month = monthly_stats['mean'].idxmin()
                        
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        st.metric("Best Month (Avg)", month_names[best_month-1], 
                                 f"{monthly_stats['mean'].iloc[best_month-1]*100:+.1f}%")
                        st.metric("Worst Month (Avg)", month_names[worst_month-1],
                                 f"{monthly_stats['mean'].iloc[worst_month-1]*100:+.1f}%")
                        
                        # Show consistency metric
                        monthly_consistency = (monthly_stats['mean'] > 0).sum() / len(monthly_stats['mean']) * 100
                        st.metric("Monthly Consistency", f"{monthly_consistency:.0f}%")
                        
                        if monthly_consistency > 70:
                            st.success("✅ High monthly consistency")
                        elif monthly_consistency > 50:
                            st.info("📊 Moderate consistency")
                        else:
                            st.warning("⚠️ Low monthly consistency")
            else:
                st.warning(f"📊 Insufficient data for seasonality analysis")
                st.info(f"Need at least 6 months of data. Current: {months_of_data:.1f} months")
                
                if months_of_data >= 2:
                    st.markdown("**Showing limited seasonality data:**")
                    benchmark_returns = benchmark_data.get('returns') if benchmark_data else None
                    fig = create_monthly_returns_heatmap(returns, benchmark_returns, show_statistics=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Detailed performance metrics
        if 'metrics' in strategy_data:
            with st.expander("📊 Detailed Performance Metrics", expanded=False):
                display_detailed_metrics(strategy_data['metrics'])
    
    def _render_risk_section(self,
                           strategy_data: Dict[str, Any],
                           benchmark_data: Optional[Dict[str, Any]] = None) -> None:
        """Render comprehensive risk analysis section."""
        st.markdown("## ⚠️ Risk Analysis")
        
        # Enhanced underwater drawdown chart with insights
        if 'equity_curve' in strategy_data:
            st.markdown("### Drawdown Analysis")
            st.markdown("Visualize peak-to-trough declines to understand risk exposure and recovery patterns.")
            
            # Add drawdown insights before the chart
            equity_curve = strategy_data['equity_curve']
            peak = equity_curve['equity'].cummax()
            drawdown = (equity_curve['equity'] - peak) / peak
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                max_dd = drawdown.min() * 100
                st.metric("Maximum Drawdown", f"{max_dd:.2f}%")
            
            with col2:
                # Calculate average drawdown
                negative_dd = drawdown[drawdown < 0]
                avg_dd = negative_dd.mean() * 100 if len(negative_dd) > 0 else 0
                st.metric("Average Drawdown", f"{avg_dd:.2f}%")
            
            with col3:
                # Time underwater (days below peak)
                underwater_days = (drawdown < -0.01).sum()  # More than 1% below peak
                st.metric("Days Underwater", f"{underwater_days}")
            
            with col4:
                # Current drawdown
                current_dd = drawdown.iloc[-1] * 100
                dd_status = "At Peak" if current_dd >= -0.01 else f"{current_dd:.2f}%"
                st.metric("Current Drawdown", dd_status)
            
            benchmark_df = benchmark_data.get('equity_curve') if benchmark_data else None
            benchmark_name = benchmark_data.get('name', 'Benchmark') if benchmark_data else None
            
            fig = create_underwater_drawdown_chart(
                strategy_data['equity_curve'],
                benchmark_df,
                benchmark_name
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics summary
        if 'metrics' in strategy_data:
            st.markdown("### Risk Metrics Summary")
            
            col1, col2, col3 = st.columns(3)
            metrics = strategy_data['metrics']
            
            with col1:
                st.markdown("**Volatility Measures**")
                vol = metrics.get('annual_volatility', 0) * 100
                downside_vol = metrics.get('downside_deviation', 0) * 100
                
                st.metric("Annual Volatility", f"{vol:.2f}%")
                st.metric("Downside Deviation", f"{downside_vol:.2f}%")
                
                if downside_vol > 0:
                    vol_ratio = vol / downside_vol
                    st.metric("Vol Ratio (Up/Down)", f"{vol_ratio:.2f}")
            
            with col2:
                st.markdown("**Drawdown Analysis**")
                max_dd = metrics.get('max_drawdown', 0) * 100
                avg_dd = metrics.get('avg_drawdown', 0) * 100
                recovery_factor = metrics.get('recovery_factor', 0)
                
                st.metric("Maximum Drawdown", f"{max_dd:.2f}%")
                st.metric("Average Drawdown", f"{avg_dd:.2f}%")
                st.metric("Recovery Factor", f"{recovery_factor:.2f}")
            
            with col3:
                st.markdown("**Tail Risk Measures**")
                var_95 = metrics.get('var_95', 0) * 100
                skewness = metrics.get('skewness', 0)
                kurtosis = metrics.get('kurtosis', 0)
                
                st.metric("VaR (95%)", f"{var_95:.2f}%")
                st.metric("Skewness", f"{skewness:.3f}")
                st.metric("Excess Kurtosis", f"{kurtosis:.3f}")
        
        # Value at Risk analysis
        if 'returns' in strategy_data:
            st.markdown("### Value at Risk Analysis")
            returns = strategy_data['returns']
            
            # Calculate different VaR levels
            var_levels = [0.01, 0.05, 0.1]
            var_values = [np.percentile(returns, level * 100) * 100 for level in var_levels]
            
            var_df = pd.DataFrame({
                'Confidence Level': ['99%', '95%', '90%'],
                'Daily VaR': [f"{val:.2f}%" for val in var_values],
                'Monthly VaR (approx)': [f"{val * np.sqrt(21):.2f}%" for val in var_values]
            })
            
            st.dataframe(var_df, use_container_width=True, hide_index=True)
    
    def _render_trading_section(self,
                              strategy_data: Dict[str, Any],
                              trades_data: Optional[pd.DataFrame] = None) -> None:
        """Render trading analysis section."""
        st.markdown("## 🎯 Trading Analysis")
        
        if trades_data is not None and not trades_data.empty:
            # Trade distribution analysis with insights
            st.markdown("### Trade Performance Distribution")
            st.markdown("Analyze individual trade outcomes and identify patterns in trading performance.")
            
            # Add trade insights before the chart
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_trades = len(trades_data)
                st.metric("Total Trades", f"{total_trades:,}")
            
            with col2:
                if 'return' in trades_data.columns:
                    avg_trade_return = trades_data['return'].mean() * 100
                    st.metric("Avg Trade Return", f"{avg_trade_return:.2f}%")
                else:
                    st.metric("Avg Trade Return", "N/A")
            
            with col3:
                if 'return' in trades_data.columns:
                    best_trade = trades_data['return'].max() * 100
                    st.metric("Best Trade", f"{best_trade:.2f}%")
                else:
                    st.metric("Best Trade", "N/A")
            
            with col4:
                if 'return' in trades_data.columns:
                    worst_trade = trades_data['return'].min() * 100
                    st.metric("Worst Trade", f"{worst_trade:.2f}%")
                else:
                    st.metric("Worst Trade", "N/A")
            
            fig = create_trade_distribution_chart(trades_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade statistics table
            st.markdown("### Trade Statistics Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Statistics**")
                total_trades = len(trades_data)
                winning_trades = len(trades_data[trades_data['return'] > 0]) if 'return' in trades_data.columns else 0
                win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                
                trade_stats = pd.DataFrame({
                    'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate'],
                    'Value': [
                        f"{total_trades:,}",
                        f"{winning_trades:,}",
                        f"{total_trades - winning_trades:,}",
                        f"{win_rate:.1f}%"
                    ]
                })
                st.dataframe(trade_stats, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Performance Statistics**")
                if 'return' in trades_data.columns:
                    returns = trades_data['return'] * 100
                    avg_return = returns.mean()
                    best_trade = returns.max()
                    worst_trade = returns.min()
                    
                    # Calculate profit factor
                    winning_returns = returns[returns > 0]
                    losing_returns = returns[returns < 0]
                    profit_factor = (winning_returns.sum() / abs(losing_returns.sum())) if len(losing_returns) > 0 else np.inf
                    
                    perf_stats = pd.DataFrame({
                        'Metric': ['Average Return', 'Best Trade', 'Worst Trade', 'Profit Factor'],
                        'Value': [
                            f"{avg_return:.2f}%",
                            f"{best_trade:.2f}%",
                            f"{worst_trade:.2f}%",
                            f"{profit_factor:.2f}" if not np.isinf(profit_factor) else "∞"
                        ]
                    })
                    st.dataframe(perf_stats, use_container_width=True, hide_index=True)
        else:
            st.warning("No individual trade data available for detailed trading analysis.")
            
            # Show aggregate trading metrics from strategy metrics if available
            if 'metrics' in strategy_data:
                metrics = strategy_data['metrics']
                st.markdown("### Strategy-Level Trading Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    win_rate = metrics.get('win_rate', 0) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col2:
                    profit_factor = metrics.get('profit_factor', 0)
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                
                with col3:
                    total_trades = metrics.get('total_trades', 0)
                    st.metric("Total Trades", f"{total_trades:,}")
    
    def _render_advanced_analytics_section(self,
                                         strategy_data: Dict[str, Any],
                                         benchmark_data: Optional[Dict[str, Any]] = None) -> None:
        """Render advanced analytics and statistical analysis."""
        st.markdown("## 📊 Advanced Analytics")
        
        # Returns distribution analysis with enhanced insights
        if 'returns' in strategy_data:
            st.markdown("### Returns Distribution Analysis")
            st.markdown("Examine the shape and characteristics of return distributions for risk assessment.")
            
            returns = strategy_data['returns']
            benchmark_returns = benchmark_data.get('returns') if benchmark_data else None
            benchmark_name = benchmark_data.get('name', 'Benchmark') if benchmark_data else None
            
            # Add distribution insights before the chart
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_days = (returns > 0).sum()
                total_days = len(returns)
                positive_pct = positive_days / total_days * 100
                st.metric("Positive Days", f"{positive_pct:.1f}%", f"{positive_days}/{total_days}")
            
            with col2:
                tail_risk_5pct = np.percentile(returns, 5) * 100
                st.metric("5% VaR (Daily)", f"{tail_risk_5pct:.2f}%", 
                         "Worst expected daily loss")
            
            with col3:
                from scipy import stats
                skewness = stats.skew(returns)
                skew_interpretation = "Right-skewed" if skewness > 0.5 else "Left-skewed" if skewness < -0.5 else "Symmetric"
                st.metric("Skewness", f"{skewness:.2f}", skew_interpretation)
            
            fig = create_returns_distribution_chart(returns, benchmark_returns, benchmark_name)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical analysis summary
        if 'returns' in strategy_data:
            st.markdown("### Statistical Analysis Summary")
            
            returns = strategy_data['returns']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribution Statistics**")
                from scipy import stats
                
                # Calculate enhanced statistics
                mean_return = returns.mean() * 252 * 100  # Annualized
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
                jb_stat, jb_pvalue = stats.jarque_bera(returns)
                
                dist_stats = pd.DataFrame({
                    'Statistic': [
                        'Mean (Annualized)', 'Volatility (Annualized)', 
                        'Skewness', 'Excess Kurtosis', 'Jarque-Bera p-value'
                    ],
                    'Value': [
                        f"{mean_return:.2f}%", f"{volatility:.2f}%",
                        f"{skewness:.3f}", f"{kurtosis:.3f}", f"{jb_pvalue:.4f}"
                    ],
                    'Interpretation': [
                        'Expected annual return',
                        'Expected annual volatility',
                        'Positive = right tail heavier' if skewness > 0 else 'Negative = left tail heavier',
                        'Positive = fat tails' if kurtosis > 0 else 'Negative = thin tails',
                        'Normal distribution' if jb_pvalue > 0.05 else 'Non-normal distribution'
                    ]
                })
                st.dataframe(dist_stats, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Risk-Adjusted Metrics**")
                
                if 'metrics' in strategy_data:
                    metrics = strategy_data['metrics']
                    
                    sharpe = metrics.get('sharpe_ratio', 0)
                    sortino = metrics.get('sortino_ratio', 0)
                    calmar = metrics.get('calmar_ratio', 0)
                    
                    # Calculate additional ratios
                    max_dd = metrics.get('max_drawdown', 0)
                    total_return = metrics.get('total_return', 0)
                    
                    risk_adj_stats = pd.DataFrame({
                        'Ratio': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
                        'Value': [f"{sharpe:.3f}", f"{sortino:.3f}", f"{calmar:.3f}"],
                        'Benchmark': ['> 1.0 Good', '> 1.5 Good', '> 0.5 Good'],
                        'Status': [
                            '✅ Excellent' if sharpe > 1.5 else '✅ Good' if sharpe > 1.0 else '⚠️ Below Average',
                            '✅ Excellent' if sortino > 1.5 else '✅ Good' if sortino > 1.0 else '⚠️ Below Average',
                            '✅ Good' if calmar > 0.5 else '⚠️ Below Average'
                        ]
                    })
                    st.dataframe(risk_adj_stats, use_container_width=True, hide_index=True)
        
        # Performance attribution (if benchmark available)
        if benchmark_data and 'metrics' in strategy_data:
            st.markdown("### Performance Attribution Analysis")
            
            strategy_metrics = strategy_data['metrics']
            benchmark_metrics = benchmark_data['metrics']
            
            # Calculate attribution metrics
            alpha = strategy_metrics.get('alpha', 0) * 100
            beta = strategy_metrics.get('beta', 1.0)
            correlation = strategy_metrics.get('correlation', 0)
            tracking_error = strategy_metrics.get('tracking_error', 0) * 100
            information_ratio = strategy_metrics.get('information_ratio', 0)
            
            attribution_df = pd.DataFrame({
                'Metric': ['Alpha (Excess Return)', 'Beta (Market Sensitivity)', 'Correlation', 'Tracking Error', 'Information Ratio'],
                'Value': [f"{alpha:.2f}%", f"{beta:.3f}", f"{correlation:.3f}", f"{tracking_error:.2f}%", f"{information_ratio:.3f}"],
                'Interpretation': [
                    'Positive = outperformance' if alpha > 0 else 'Negative = underperformance',
                    'High = more volatile than market' if beta > 1.1 else 'Low = less volatile than market' if beta < 0.9 else 'Similar to market',
                    'High = moves with market' if correlation > 0.7 else 'Low = independent of market' if correlation < 0.3 else 'Moderate correlation',
                    'High = volatile vs benchmark' if tracking_error > 5 else 'Low = similar to benchmark',
                    'High = good active management' if information_ratio > 0.5 else 'Low = poor active management'
                ]
            })
            st.dataframe(attribution_df, use_container_width=True, hide_index=True)
    
    def _render_ai_insights(self, strategy_data: Dict[str, Any]) -> None:
        """Render AI-powered insights and recommendations."""
        if 'metrics' not in strategy_data:
            return
        
        metrics = strategy_data['metrics']
        insights = []
        
        # Generate automated insights based on metrics
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Return insights
        if total_return > 0.15:
            insights.append("🎯 **Strong Performance**: Strategy generated exceptional returns above 15%")
        elif total_return > 0.05:
            insights.append("📈 **Positive Performance**: Strategy delivered solid positive returns")
        elif total_return < -0.05:
            insights.append("⚠️ **Performance Warning**: Strategy experienced negative returns requiring analysis")
        
        # Risk insights
        if sharpe_ratio > 1.5:
            insights.append("⭐ **Excellent Risk-Adjusted Returns**: Sharpe ratio indicates superior efficiency")
        elif sharpe_ratio < 0.5:
            insights.append("🔍 **Risk Efficiency Concern**: Low Sharpe ratio suggests poor risk-adjusted performance")
        
        # Drawdown insights
        if abs(max_drawdown) > 0.2:
            insights.append("🚨 **High Risk Warning**: Maximum drawdown exceeds 20%, indicating high risk exposure")
        elif abs(max_drawdown) < 0.05:
            insights.append("🛡️ **Low Risk Profile**: Maximum drawdown under 5% shows excellent risk control")
        
        # Win rate insights
        if win_rate > 0.65:
            insights.append("🎯 **High Consistency**: Win rate above 65% demonstrates reliable signal generation")
        elif win_rate < 0.4:
            insights.append("📊 **Consistency Opportunity**: Low win rate suggests potential for signal improvement")
        
        if insights:
            st.markdown("### 🤖 AI-Powered Insights")
            for insight in insights:
                st.markdown(f"- {insight}")
    
    def _render_export_section(self, strategy_data: Dict[str, Any], strategy_name: str) -> None:
        """Render export and sharing options."""
        st.markdown("---")
        st.markdown("### 📤 Export & Sharing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Metrics (CSV)", use_container_width=True):
                if 'metrics' in strategy_data:
                    metrics_df = pd.DataFrame.from_dict(strategy_data['metrics'], orient='index', columns=['Value'])
                    csv = metrics_df.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{strategy_name}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📈 Export Equity Curve", use_container_width=True):
                if 'equity_curve' in strategy_data:
                    csv = strategy_data['equity_curve'].to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{strategy_name}_equity_curve_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📋 Generate Report", use_container_width=True):
                st.info("📄 Professional PDF report generation coming soon!")


def render_enhanced_analytics(strategy_data: Dict[str, Any],
                            benchmark_data: Optional[Dict[str, Any]] = None,
                            trades_data: Optional[pd.DataFrame] = None,
                            strategy_name: str = "Strategy",
                            theme: str = "professional") -> None:
    """Main function to render enhanced analytics layout.
    
    This is the primary entry point for displaying comprehensive strategy analysis
    with professional formatting and organization.
    
    Args:
        strategy_data: Strategy performance data and metrics
        benchmark_data: Optional benchmark data for comparison
        trades_data: Optional individual trade records
        strategy_name: Display name for the strategy
        theme: Visual theme for the analysis
    """
    layout_manager = EnhancedAnalyticsLayout(theme=theme)
    layout_manager.render_complete_analysis(
        strategy_data=strategy_data,
        benchmark_data=benchmark_data,
        trades_data=trades_data,
        strategy_name=strategy_name
    )