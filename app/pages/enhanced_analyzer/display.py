"""
Enhanced Strategy Analyzer - Display Components

This module contains all display-related functions for the AI-powered strategy analyzer.
These functions handle the UI presentation of analysis results, metrics, and insights.

Functions:
- display_enhanced_welcome_screen: Welcome screen with AI features
- display_enhanced_results: Main results display orchestrator
- display_overview_tab: Overview tab with key metrics
- display_ml_analysis_tab: ML analysis results display
- display_ai_insights_tab: AI insights and trade explanations
- display_charts_tab: Interactive charts and visualizations
- display_trade_log_tab: Detailed trade log with AI commentary
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Import chart components
from components.charts import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_price_and_signals_chart
)

# Import AI display components
from components.gpt_display import (
    display_strategy_overview_gpt,
    display_performance_summary_gpt,
    display_trade_explanations,
    display_ai_metrics_comparison,
    display_gpt_trade_insights_chart
)


def display_enhanced_welcome_screen(enable_ml: bool, enable_gpt: bool):
    """
    Display AI-Enhanced Welcome Screen.
    
    This advanced welcome screen showcases the revolutionary AI capabilities
    and guides users through the enhanced features available in the AI-powered
    analyzer.
    
    AI Feature Highlights:
    - Machine Learning Integration: Traditional vs AI signal comparison
    - GPT-Powered Insights: Natural language strategy explanations
    - Advanced Analytics: Multi-modal AI analysis
    - Real-time AI Feedback: Dynamic insights during analysis
    
    Educational Components:
    - Clear explanation of AI capabilities
    - Feature status indicators
    - Getting started guidance
    - Expectation setting for AI features
    
    Args:
        enable_ml (bool): Whether ML features are enabled
        enable_gpt (bool): Whether GPT features are enabled
    """
    
    # Centered layout for focused AI feature presentation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🚀 Next-Gen Strategy Analysis
        
        **Revolutionary features:**
        
        ### 🧠 Machine Learning Integration
        - **ML vs Rule-Based**: Compare traditional signals with AI predictions
        - **Feature Importance**: See which indicators drive performance
        - **Confidence Scores**: Know when the ML model is most certain
        
        ### 🤖 GPT-Powered Insights
        - **Trade Commentary**: AI explains every trade decision
        - **Strategy Analysis**: Deep insights into your approach
        - **Performance Narrative**: Human-like analysis of results
        
        ### 📊 Advanced Analytics
        - **Real-time Backtesting**: Instant strategy evaluation
        - **Risk Assessment**: AI-powered risk analysis
        - **Benchmark Comparison**: See how you stack up
        """)
        
        # Real-time AI feature status indicators
        # Provides immediate feedback on current AI configuration
        st.markdown("---")
        st.markdown("### 🎛️ Current Settings")
        
        col_ml, col_gpt = st.columns(2)
        
        with col_ml:
            ml_status = "🟢 Enabled" if enable_ml else "🔴 Disabled"
            st.markdown(f"**ML Signals**: {ml_status}")
            if enable_ml:
                st.info("✨ AI will compare rule-based signals with ML predictions")
            
        with col_gpt:
            gpt_status = "🟢 Enabled" if enable_gpt else "🔴 Disabled"
            st.markdown(f"**GPT Analysis**: {gpt_status}")
            if enable_gpt:
                st.info("🤖 AI will provide natural language explanations")
        
        st.markdown("---")
        st.info("👈 Configure your strategy in the sidebar and click **'Run Analysis'** to start!")


def display_enhanced_results(config: Dict[str, Any],
                           price_data: pd.DataFrame,
                           rule_signals: pd.DataFrame,
                           ml_results: Optional[Dict[str, Any]],
                           equity_curve: pd.DataFrame,
                           trades: pd.DataFrame,
                           strategy_metrics: Dict[str, Any],
                           gpt_insights: Dict[str, Any],
                           symbol: str):
    """
    Display Comprehensive AI-Enhanced Results.
    
    This function creates an advanced tabbed interface for presenting
    complex AI analysis results in an organized, digestible format.
    
    Advanced Presentation Features:
    - Multi-tab organization for complex analysis
    - Progressive disclosure of AI insights
    - Interactive AI-enhanced visualizations
    - Comparative analysis presentation
    - Export capabilities for AI insights
    
    Tab Organization:
    1. Overview: High-level results with AI summary
    2. ML Analysis: Machine learning comparison and insights
    3. AI Insights: GPT-generated analysis and commentary
    4. Charts: Interactive visualizations with AI overlays
    5. Trade Log: Detailed trade records with AI explanations
    
    Args:
        config: Enhanced configuration dictionary
        price_data: Historical market data
        rule_signals: Traditional strategy signals
        ml_results: Machine learning analysis results
        equity_curve: Portfolio performance over time
        trades: Detailed trade records
        strategy_metrics: Performance statistics
        gpt_insights: AI-generated insights and commentary
        symbol: Trading symbol for context
    """
    
    # Create sophisticated tabbed interface for complex AI analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🧠 ML Analysis", 
        "🤖 AI Insights", 
        "📈 Charts", 
        "📋 Trade Log"
    ])
    
    with tab1:
        # Overview tab with AI-enhanced metrics and summary
        display_overview_tab(
            config, strategy_metrics, gpt_insights, symbol
        )
    
    with tab2:
        # Machine learning analysis and comparison
        display_ml_analysis_tab(ml_results, rule_signals)
    
    with tab3:
        # GPT-powered insights and trade explanations
        display_ai_insights_tab(gpt_insights, trades)
    
    with tab4:
        # Interactive charts with AI-enhanced visualizations
        display_charts_tab(
            price_data, rule_signals, ml_results, equity_curve, trades
        )
    
    with tab5:
        # Detailed trade log with AI commentary
        display_trade_log_tab(trades)


def display_overview_tab(config: Dict[str, Any],
                        metrics: Dict[str, Any],
                        gpt_insights: Dict[str, Any],
                        symbol: str):
    """
    Display AI-Enhanced Overview Tab.
    
    This tab provides a high-level summary of strategy performance with
    AI-generated insights and explanations prominently featured.
    
    Overview Components:
    - Strategy information header with AI context
    - Key performance metrics with AI interpretation
    - GPT-generated strategy overview
    - AI-enhanced metrics comparison
    - Performance summary with natural language analysis
    
    Args:
        config: Strategy configuration
        metrics: Performance metrics
        gpt_insights: AI-generated insights
        symbol: Trading symbol
    """
    
    # Strategy information header with AI enhancement indicators
    strategy_name = config.get('strategy', {}).get('category', 'Unknown').title()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 🎯 {strategy_name} Strategy on {symbol}")
    with col2:
        st.metric("Analysis Date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    with col3:
        total_return = metrics.get('Total Return', 0) * 100
        st.metric("Total Return", f"{total_return:.1f}%")
    
    # GPT-generated strategy overview with AI branding
    if 'strategy_overview' in gpt_insights:
        display_strategy_overview_gpt(gpt_insights['strategy_overview'], strategy_name)
    
    # AI-enhanced key metrics display with interpretation
    display_ai_metrics_comparison(metrics, config.get('enable_gpt', False))
    
    # GPT-powered performance summary with natural language analysis
    if 'performance_summary' in gpt_insights:
        display_performance_summary_gpt(gpt_insights['performance_summary'])


def display_ml_analysis_tab(ml_results: Optional[Dict[str, Any]], 
                           rule_signals: pd.DataFrame):
    """
    Display Machine Learning Analysis and Comparison.
    
    This tab provides comprehensive analysis of ML model performance,
    training results, and comparison with rule-based signals.
    
    ML Analysis Components:
    - Training performance metrics and validation
    - Feature importance analysis and interpretation
    - Signal agreement analysis between ML and rule-based
    - Confidence distribution analysis
    - Model interpretation and decision explanations
    
    Args:
        ml_results: ML training and prediction results
        rule_signals: Rule-based signals for comparison
    """
    
    if not ml_results or not ml_results.get('success', False):
        st.warning("🧠 ML Analysis not available")
        if ml_results:
            st.error(f"Error: {ml_results.get('error', 'Unknown error')}")
        return
    
    st.subheader("🧠 Machine Learning Analysis")
    
    # Extract comprehensive ML analysis results
    training = ml_results['training_results']
    comparison = ml_results['comparison']
    
    # Display training performance metrics in organized layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Accuracy", f"{training['training_accuracy']:.3f}")
    with col2:
        st.metric("Validation Accuracy", f"{training['validation_accuracy']:.3f}")
    with col3:
        st.metric("Agreement Rate", f"{comparison['agreement_rate']:.3f}")
    with col4:
        st.metric("Avg Confidence", f"{comparison['average_ml_confidence']:.3f}")
    
    # Feature importance analysis for model interpretation
    st.subheader("📊 Top Features")
    top_features = training['top_features'][:10]
    
    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    st.bar_chart(feature_df.set_index('Feature')['Importance'])
    
    # Signal comparison analysis between ML and rule-based approaches
    st.subheader("⚖️ Signal Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rule-Based Signals**")
        rule_dist = comparison['rule_signal_distribution']
        st.json(rule_dist)
    
    with col2:
        st.markdown("**ML Predicted Signals**")
        ml_dist = comparison['ml_signal_distribution']
        st.json(ml_dist)
    
    # Comprehensive agreement analysis with confidence considerations
    st.info(f"""
    **ML vs Rule-Based Summary:**
    - Total signals analyzed: {comparison['total_signals']}
    - Agreements: {comparison['agreements']} ({comparison['agreement_rate']:.1%})
    - High confidence signals: {comparison['high_confidence_signals']}
    - High confidence agreement: {comparison['high_confidence_agreement_rate']:.1%}
    """)


def display_ai_insights_tab(gpt_insights: Dict[str, Any], trades: pd.DataFrame):
    """
    Display AI-Generated Insights and Trade Explanations.
    
    This tab showcases the GPT-powered analysis capabilities, providing
    natural language explanations and insights about trading performance.
    
    AI Insights Components:
    - Individual trade explanations with AI commentary
    - Strategic analysis and recommendations
    - Risk assessment with AI interpretation
    - Market context and timing analysis
    - Performance improvement suggestions
    
    Args:
        gpt_insights: GPT-generated analysis and commentary
        trades: Trade records for explanation
    """
    
    if not gpt_insights or 'error' in gpt_insights:
        st.warning("🤖 AI insights not available")
        return
    
    # Display comprehensive trade explanations with GPT commentary
    display_trade_explanations(trades)
    
    # Interactive trade performance chart with AI insights overlay
    if not trades.empty:
        display_gpt_trade_insights_chart(trades)


def display_charts_tab(price_data: pd.DataFrame,
                      rule_signals: pd.DataFrame,
                      ml_results: Optional[Dict[str, Any]],
                      equity_curve: pd.DataFrame,
                      trades: pd.DataFrame):
    """
    Display AI-Enhanced Interactive Charts.
    
    This tab provides comprehensive visualization of strategy performance
    with AI-enhanced overlays and interpretations.
    
    Chart Features:
    - Price action with signal overlays
    - ML vs rule-based signal comparison
    - Equity curve with AI commentary points
    - Drawdown analysis with risk assessment
    - Enhanced interactive features
    
    Args:
        price_data: Historical market data
        rule_signals: Traditional strategy signals
        ml_results: ML predictions for overlay
        equity_curve: Portfolio performance
        trades: Trade records for annotation
    """
    
    # Create comprehensive price and signals chart
    fig_price = create_price_and_signals_chart(price_data, rule_signals)
    
    # TODO: Add ML signals overlay when available
    # Future enhancement: Integrate ML predictions into price chart
    if ml_results and ml_results.get('success', False):
        # This would require updating the chart function to handle ML signals
        pass
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Display equity curve with AI insights
    fig_equity = create_equity_curve_chart(equity_curve)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Show drawdown analysis for risk assessment
    fig_drawdown = create_drawdown_chart(equity_curve)
    st.plotly_chart(fig_drawdown, use_container_width=True)


def display_trade_log_tab(trades: pd.DataFrame):
    """
    Display Detailed Trade Log with AI Commentary.
    
    This tab provides comprehensive trade-level analysis with AI-generated
    explanations for individual trading decisions.
    
    Trade Log Features:
    - Complete trade records with performance metrics
    - AI explanations for trade decisions (when available)
    - Trade performance summary statistics
    - Win/loss analysis with AI interpretation
    - Export capabilities for further analysis
    
    Args:
        trades: Detailed trade records with optional AI commentary
    """
    
    if trades.empty:
        st.info("No trades executed")
        return
    
    st.subheader("📋 Complete Trade Log")
    
    # Calculate and display trade performance summary
    winning_trades = len(trades[trades['net_pnl'] > 0])
    total_trades = len(trades)
    total_pnl = trades['net_pnl'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Winning Trades", winning_trades)
    with col3:
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1%}")
    with col4:
        st.metric("Total P&L", f"${total_pnl:.2f}")
    
    # Display comprehensive trade table with all available information
    display_columns = [
        'entry_time', 'exit_time', 'side', 'entry_price', 
        'exit_price', 'net_pnl', 'return', 'duration'
    ]
    
    # Include GPT explanations if available
    if 'gpt_explanation' in trades.columns:
        display_columns.append('gpt_explanation')
    
    # Filter to only available columns
    available_columns = [col for col in display_columns if col in trades.columns]
    
    # Display interactive trade table with scrolling capability
    st.dataframe(
        trades[available_columns],
        use_container_width=True,
        height=400
    ) 