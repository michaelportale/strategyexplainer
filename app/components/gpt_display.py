"""Streamlit components for displaying GPT-generated insights."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go


def display_strategy_overview_gpt(gpt_overview: str, strategy_name: str):
    """Display AI-generated strategy overview."""
    st.subheader("🧠 AI Strategy Analysis")
    
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: white;">🎯 {strategy_name} Strategy Insights</h4>
            <p style="margin: 0; font-size: 16px; line-height: 1.5;">{gpt_overview}</p>
        </div>
        """, unsafe_allow_html=True)


def display_performance_summary_gpt(gpt_summary: str):
    """Display AI-generated performance summary."""
    st.subheader("📊 AI Performance Analysis")
    
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: white;">📈 Performance Verdict</h4>
            <p style="margin: 0; font-size: 16px; line-height: 1.5;">{gpt_summary}</p>
        </div>
        """, unsafe_allow_html=True)


def display_trade_explanations(trades_df: pd.DataFrame):
    """Display individual trade explanations from GPT."""
    if trades_df.empty or 'gpt_explanation' not in trades_df.columns:
        st.info("No trade explanations available")
        return
    
    st.subheader("🤖 AI Trade Analysis")
    
    # Filter out trades without explanations
    explained_trades = trades_df[
        (trades_df['gpt_explanation'].notna()) & 
        (trades_df['gpt_explanation'] != "No explanation generated") &
        (trades_df['gpt_explanation'] != "GPT disabled or no trades completed")
    ].copy()
    
    if explained_trades.empty:
        st.info("No AI explanations generated for trades")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏆 Best Trades", "💀 Worst Trades", "📋 All Trades"])
    
    with tab1:
        display_top_trades(explained_trades, "best")
    
    with tab2:
        display_top_trades(explained_trades, "worst")
    
    with tab3:
        display_all_trades(explained_trades)


def display_top_trades(trades_df: pd.DataFrame, trade_type: str = "best"):
    """Display top performing trades with explanations."""
    
    if trade_type == "best":
        sorted_trades = trades_df.nlargest(5, 'net_pnl')
        emoji = "🏆"
        color = "#28a745"
    else:
        sorted_trades = trades_df.nsmallest(5, 'net_pnl')
        emoji = "💀"
        color = "#dc3545"
    
    for idx, trade in sorted_trades.iterrows():
        pnl = trade['net_pnl']
        return_pct = trade.get('return', 0) * 100
        side = trade.get('side', 'unknown').title()
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        duration = trade.get('duration', 0)
        
        with st.expander(f"{emoji} {side} Trade: ${pnl:+.2f} ({return_pct:+.1f}%)", expanded=False):
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                **Trade Details:**
                - Entry: ${entry_price:.2f}
                - Exit: ${exit_price:.2f}
                - Duration: {duration} days
                - P&L: ${pnl:+.2f}
                - Return: {return_pct:+.1f}%
                """)
            
            with col2:
                explanation = trade.get('gpt_explanation', 'No explanation available')
                st.markdown(f"""
                <div style="
                    background-color: {color}15;
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                ">
                    <strong>🧠 AI Analysis:</strong><br>
                    {explanation}
                </div>
                """, unsafe_allow_html=True)


def display_all_trades(trades_df: pd.DataFrame):
    """Display all trades in a filterable table with explanations."""
    
    # Add selection controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trade_filter = st.selectbox(
            "Filter trades:",
            ["All", "Winners Only", "Losers Only"]
        )
    
    with col2:
        min_pnl = st.number_input("Min P&L", value=trades_df['net_pnl'].min())
    
    with col3:
        max_pnl = st.number_input("Max P&L", value=trades_df['net_pnl'].max())
    
    # Apply filters
    filtered_trades = trades_df.copy()
    
    if trade_filter == "Winners Only":
        filtered_trades = filtered_trades[filtered_trades['net_pnl'] > 0]
    elif trade_filter == "Losers Only":
        filtered_trades = filtered_trades[filtered_trades['net_pnl'] <= 0]
    
    filtered_trades = filtered_trades[
        (filtered_trades['net_pnl'] >= min_pnl) & 
        (filtered_trades['net_pnl'] <= max_pnl)
    ]
    
    # Display filtered trades
    st.write(f"**Showing {len(filtered_trades)} trades:**")
    
    for idx, trade in filtered_trades.iterrows():
        pnl = trade['net_pnl']
        return_pct = trade.get('return', 0) * 100
        side = trade.get('side', 'unknown').title()
        entry_time = trade.get('entry_time', 'Unknown')
        exit_time = trade.get('exit_time', 'Unknown')
        
        # Color based on performance
        color = "#28a745" if pnl > 0 else "#dc3545"
        emoji = "📈" if pnl > 0 else "📉"
        
        with st.expander(f"{emoji} Trade #{idx+1}: {side} | ${pnl:+.2f} | {entry_time}", expanded=False):
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                **Trade Summary:**
                - **Entry:** {entry_time}
                - **Exit:** {exit_time}
                - **Side:** {side}
                - **P&L:** ${pnl:+.2f}
                - **Return:** {return_pct:+.1f}%
                - **Duration:** {trade.get('duration', 0)} days
                """)
            
            with col2:
                explanation = trade.get('gpt_explanation', 'No explanation available')
                st.markdown(f"""
                <div style="
                    background-color: {color}15;
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                ">
                    <strong>🤖 AI Explanation:</strong><br>
                    {explanation}
                </div>
                """, unsafe_allow_html=True)


def display_gpt_insights_sidebar():
    """Display GPT service status and controls in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 AI Features")
    
    # Check if GPT is enabled
    gpt_enabled = st.sidebar.checkbox("Enable AI Commentary", value=True, key="gpt_enabled")
    
    if gpt_enabled:
        api_key_status = check_openai_key()
        if api_key_status == "valid":
            st.sidebar.success("✅ AI Ready")
        elif api_key_status == "missing":
            st.sidebar.error("❌ No OpenAI API Key")
            st.sidebar.info("Add OPENAI_API_KEY to environment variables or Streamlit secrets")
        else:
            st.sidebar.warning("⚠️ API Key Invalid")
    else:
        st.sidebar.info("🤖 AI Commentary Disabled")
    
    return gpt_enabled


def check_openai_key() -> str:
    """Check if OpenAI API key is available and valid."""
    import os
    
    # Check environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Check Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
    
    if not api_key:
        return "missing"
    
    # Basic validation (starts with sk-)
    if api_key.startswith('sk-'):
        return "valid"
    else:
        return "invalid"


def display_ai_metrics_comparison(strategy_metrics: Dict[str, Any], gpt_enabled: bool):
    """Display AI-enhanced metrics comparison."""
    
    st.subheader("📊 Performance Metrics")
    
    # Display standard metrics first
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = strategy_metrics.get('Total Return', 0) * 100
        st.metric("Total Return", f"{total_return:.1f}%")
    
    with col2:
        sharpe = strategy_metrics.get('Sharpe', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col3:
        max_dd = strategy_metrics.get('Max Drawdown', 0) * 100
        st.metric("Max Drawdown", f"{max_dd:.1f}%")
    
    with col4:
        win_rate = strategy_metrics.get('Win Rate', 0) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    # Add AI interpretation if enabled
    if gpt_enabled:
        st.markdown("---")
        
        # Create AI risk assessment
        risk_level = assess_risk_level(strategy_metrics)
        risk_color = get_risk_color(risk_level)
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {risk_color}20 0%, {risk_color}10 100%);
            border-left: 4px solid {risk_color};
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        ">
            <h4 style="margin: 0 0 10px 0; color: {risk_color};">🎯 AI Risk Assessment: {risk_level}</h4>
            <p style="margin: 0;">{generate_risk_interpretation(strategy_metrics)}</p>
        </div>
        """, unsafe_allow_html=True)


def assess_risk_level(metrics: Dict[str, Any]) -> str:
    """Assess risk level based on metrics."""
    sharpe = metrics.get('Sharpe', 0)
    max_dd = abs(metrics.get('Max Drawdown', 0))
    
    if sharpe > 1.5 and max_dd < 0.15:
        return "LOW RISK"
    elif sharpe > 1.0 and max_dd < 0.25:
        return "MODERATE RISK"
    elif sharpe > 0.5 and max_dd < 0.35:
        return "HIGH RISK"
    else:
        return "EXTREME RISK"


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "LOW RISK": "#28a745",
        "MODERATE RISK": "#ffc107", 
        "HIGH RISK": "#fd7e14",
        "EXTREME RISK": "#dc3545"
    }
    return colors.get(risk_level, "#6c757d")


def generate_risk_interpretation(metrics: Dict[str, Any]) -> str:
    """Generate risk interpretation without GPT (fallback)."""
    sharpe = metrics.get('Sharpe', 0)
    max_dd = abs(metrics.get('Max Drawdown', 0)) * 100
    total_return = metrics.get('Total Return', 0) * 100
    
    if sharpe > 1.5:
        risk_adj = "excellent risk-adjusted returns"
    elif sharpe > 1.0:
        risk_adj = "good risk-adjusted returns"
    elif sharpe > 0.5:
        risk_adj = "moderate risk-adjusted returns"
    else:
        risk_adj = "poor risk-adjusted returns"
    
    drawdown_assessment = "manageable" if max_dd < 20 else "significant" if max_dd < 35 else "extreme"
    
    return f"Strategy shows {risk_adj} with {drawdown_assessment} drawdown risk. Total return of {total_return:.1f}% comes with maximum drawdown of {max_dd:.1f}%."


def display_gpt_trade_insights_chart(trades_df: pd.DataFrame):
    """Display chart showing trade performance with AI insights."""
    
    if trades_df.empty:
        return
    
    st.subheader("📈 Trade Performance Timeline")
    
    # Create cumulative P&L chart
    trades_df = trades_df.copy()
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    trades_df['trade_number'] = range(1, len(trades_df) + 1)
    
    # Color trades by performance
    trades_df['color'] = trades_df['net_pnl'].apply(
        lambda x: 'green' if x > 0 else 'red'
    )
    
    fig = go.Figure()
    
    # Add cumulative P&L line
    fig.add_trace(go.Scatter(
        x=trades_df['trade_number'],
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='blue', width=2),
        marker=dict(size=8, color=trades_df['color'])
    ))
    
    # Add hover text with AI explanations if available
    if 'gpt_explanation' in trades_df.columns:
        hover_text = []
        for _, trade in trades_df.iterrows():
            explanation = trade.get('gpt_explanation', 'No explanation')
            # Truncate long explanations for hover
            explanation_short = explanation[:100] + "..." if len(explanation) > 100 else explanation
            
            hover_text.append(
                f"Trade #{trade['trade_number']}<br>" +
                f"P&L: ${trade['net_pnl']:+.2f}<br>" +
                f"Return: {trade.get('return', 0)*100:+.1f}%<br>" +
                f"AI: {explanation_short}"
            )
        
        fig.update_traces(hovertemplate='%{text}<extra></extra>', text=hover_text)
    
    fig.update_layout(
        title="Trade Performance with AI Insights",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L ($)",
        hovermode='closest',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_gpt_enabled_config() -> Dict[str, Any]:
    """Create configuration for GPT-enabled analysis."""
    
    gpt_enabled = display_gpt_insights_sidebar()
    
    return {
        'enable_gpt': gpt_enabled,
        'gpt_features': {
            'strategy_overview': gpt_enabled,
            'trade_explanations': gpt_enabled,
            'performance_summary': gpt_enabled,
            'risk_assessment': gpt_enabled
        }
    }