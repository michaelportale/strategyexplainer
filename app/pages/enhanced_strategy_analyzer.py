"""Enhanced strategy analyzer with GPT and ML integration."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing components
from components.sidebar import render_sidebar, display_strategy_info, render_run_button
from components.metrics_cards import display_key_metrics, display_detailed_metrics
from components.charts import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_price_and_signals_chart
)

# Import new GPT and ML components
from components.gpt_display import (
    display_strategy_overview_gpt,
    display_performance_summary_gpt,
    display_trade_explanations,
    display_gpt_insights_sidebar,
    display_ai_metrics_comparison,
    display_gpt_trade_insights_chart,
    create_gpt_enabled_config
)

# Import backend modules
try:
    from backend.momentum_backtest import MomentumBacktest
    from backend.data_loader import DataLoader
    from backend.metrics import PerformanceMetrics
    from backend.enhanced_simulator import GPTEnhancedSimulator, create_enhanced_simulator
    from backend.ml_classifier import MLSignalClassifier, create_ml_classifier
    from backend.gpt_service import GPTService
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.info("Please ensure all backend modules are properly installed and accessible.")


def main():
    """Enhanced strategy analyzer with AI features."""
    st.set_page_config(
        page_title="🚀 AI-Powered Strategy Analyzer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with AI branding
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🚀 AI-Powered Strategy Analyzer</h1>
        <p style="color: white; margin: 5px 0 0 0; opacity: 0.9;">Real-time backtesting with GPT commentary and ML signal prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration with AI controls
    config = render_sidebar()
    gpt_config = create_gpt_enabled_config()
    config.update(gpt_config)
    
    display_strategy_info(config)
    
    # AI feature selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Features")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        enable_ml = st.checkbox("🧠 ML Signals", value=True, help="Compare rule-based vs ML predicted signals")
    with col2:
        enable_gpt = st.checkbox("💬 GPT Analysis", value=config.get('enable_gpt', True), help="AI-powered trade explanations")
    
    config['enable_ml'] = enable_ml
    config['enable_gpt'] = enable_gpt
    
    # Main content area
    if render_run_button():
        run_enhanced_analysis(config)
    else:
        display_enhanced_welcome_screen(enable_ml, enable_gpt)


def display_enhanced_welcome_screen(enable_ml: bool, enable_gpt: bool):
    """Display enhanced welcome screen highlighting AI features."""
    
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
        
        # Feature status indicators
        st.markdown("---")
        st.markdown("### 🎛️ Current Settings")
        
        col_ml, col_gpt = st.columns(2)
        
        with col_ml:
            ml_status = "🟢 Enabled" if enable_ml else "🔴 Disabled"
            st.markdown(f"**ML Signals**: {ml_status}")
        
        with col_gpt:
            gpt_status = "🟢 Enabled" if enable_gpt else "🔴 Disabled"
            st.markdown(f"**GPT Analysis**: {gpt_status}")
        
        st.markdown("---")
        st.info("👈 Configure your strategy in the sidebar and click **'Run Analysis'** to start!")


def run_enhanced_analysis(config: Dict[str, Any]):
    """Run enhanced analysis with GPT and ML features."""
    
    try:
        # Show progress
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Step 1: Load data
        status_text.text("📊 Loading market data...")
        progress_bar.progress(10)
        
        data_loader = DataLoader()
        symbol = config['data']['symbol']
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        
        price_data = data_loader.fetch_data(symbol, start_date, end_date)
        
        # Step 2: Generate rule-based signals
        status_text.text("⚡ Generating rule-based signals...")
        progress_bar.progress(25)
        
        strategy = initialize_strategy(config)
        rule_signals = strategy.generate_signals(price_data)
        
        # Step 3: Train ML model (if enabled)
        ml_results = None
        if config.get('enable_ml', False):
            status_text.text("🧠 Training ML model...")
            progress_bar.progress(40)
            
            ml_results = train_and_predict_ml(price_data, rule_signals, config)
        
        # Step 4: Run enhanced simulation
        status_text.text("🔄 Running enhanced simulation...")
        progress_bar.progress(60)
        
        # Create enhanced simulator with GPT
        simulator = create_enhanced_simulator(config, strategy.__class__.__name__)
        
        # Run simulation
        equity_curve, trades = simulator.simulate_strategy(
            price_data, 
            rule_signals, 
            symbol
        )
        
        # Step 5: Calculate metrics
        status_text.text("📊 Calculating performance metrics...")
        progress_bar.progress(80)
        
        perf_metrics = PerformanceMetrics()
        strategy_metrics = perf_metrics.calculate_all_metrics(equity_curve, trades)
        
        # Generate GPT insights
        gpt_insights = {}
        if config.get('enable_gpt', False) and simulator.gpt_service and simulator.gpt_service.enabled:
            status_text.text("🤖 Generating AI insights...")
            progress_bar.progress(90)
            
            gpt_insights = generate_gpt_insights(
                simulator, config, strategy_metrics, symbol
            )
        
        # Step 6: Display results
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_container.empty()
        
        # Display comprehensive results
        display_enhanced_results(
            config=config,
            price_data=price_data,
            rule_signals=rule_signals,
            ml_results=ml_results,
            equity_curve=equity_curve,
            trades=trades,
            strategy_metrics=strategy_metrics,
            gpt_insights=gpt_insights,
            symbol=symbol
        )
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)


def train_and_predict_ml(price_data: pd.DataFrame, 
                        rule_signals: pd.DataFrame,
                        config: Dict[str, Any]) -> Dict[str, Any]:
    """Train ML model and generate predictions."""
    
    try:
        # Create ML classifier
        ml_classifier = create_ml_classifier(config)
        
        # Train on rule-based signals
        training_results = ml_classifier.train(price_data, rule_signals)
        
        # Generate ML predictions
        ml_predictions = ml_classifier.predict_signals(price_data)
        
        # Compare with rule-based
        comparison = ml_classifier.compare_with_rule_based(price_data, rule_signals)
        
        return {
            'classifier': ml_classifier,
            'training_results': training_results,
            'predictions': ml_predictions,
            'comparison': comparison,
            'success': True
        }
        
    except Exception as e:
        st.warning(f"ML training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def generate_gpt_insights(simulator: GPTEnhancedSimulator,
                         config: Dict[str, Any],
                         metrics: Dict[str, Any],
                         symbol: str) -> Dict[str, Any]:
    """Generate comprehensive GPT insights."""
    
    insights = {}
    
    try:
        # Strategy overview
        strategy_params = config.get('strategy', {}).get('parameters', {})
        insights['strategy_overview'] = simulator.generate_strategy_overview(strategy_params)
        
        # Performance summary
        insights['performance_summary'] = simulator.generate_performance_summary(metrics)
        
        # Trade insights already generated in simulator
        insights['trade_insights'] = simulator.get_trade_insights()
        
    except Exception as e:
        st.warning(f"GPT insight generation failed: {str(e)}")
        insights['error'] = str(e)
    
    return insights


def display_enhanced_results(config: Dict[str, Any],
                           price_data: pd.DataFrame,
                           rule_signals: pd.DataFrame,
                           ml_results: Optional[Dict[str, Any]],
                           equity_curve: pd.DataFrame,
                           trades: pd.DataFrame,
                           strategy_metrics: Dict[str, Any],
                           gpt_insights: Dict[str, Any],
                           symbol: str):
    """Display comprehensive enhanced results."""
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🧠 ML Analysis", 
        "🤖 AI Insights", 
        "📈 Charts", 
        "📋 Trade Log"
    ])
    
    with tab1:
        display_overview_tab(
            config, strategy_metrics, gpt_insights, symbol
        )
    
    with tab2:
        display_ml_analysis_tab(ml_results, rule_signals)
    
    with tab3:
        display_ai_insights_tab(gpt_insights, trades)
    
    with tab4:
        display_charts_tab(
            price_data, rule_signals, ml_results, equity_curve, trades
        )
    
    with tab5:
        display_trade_log_tab(trades)


def display_overview_tab(config: Dict[str, Any],
                        metrics: Dict[str, Any],
                        gpt_insights: Dict[str, Any],
                        symbol: str):
    """Display overview tab with key metrics and AI summary."""
    
    # Strategy info header
    strategy_name = config.get('strategy', {}).get('category', 'Unknown').title()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 🎯 {strategy_name} Strategy on {symbol}")
    with col2:
        st.metric("Analysis Date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    with col3:
        total_return = metrics.get('Total Return', 0) * 100
        st.metric("Total Return", f"{total_return:.1f}%")
    
    # GPT Strategy Overview
    if 'strategy_overview' in gpt_insights:
        display_strategy_overview_gpt(gpt_insights['strategy_overview'], strategy_name)
    
    # Key metrics with AI interpretation
    display_ai_metrics_comparison(metrics, config.get('enable_gpt', False))
    
    # GPT Performance Summary
    if 'performance_summary' in gpt_insights:
        display_performance_summary_gpt(gpt_insights['performance_summary'])


def display_ml_analysis_tab(ml_results: Optional[Dict[str, Any]], 
                           rule_signals: pd.DataFrame):
    """Display ML analysis and comparison."""
    
    if not ml_results or not ml_results.get('success', False):
        st.warning("🧠 ML Analysis not available")
        if ml_results:
            st.error(f"Error: {ml_results.get('error', 'Unknown error')}")
        return
    
    st.subheader("🧠 Machine Learning Analysis")
    
    # Training results
    training = ml_results['training_results']
    comparison = ml_results['comparison']
    
    # Training metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Accuracy", f"{training['training_accuracy']:.3f}")
    with col2:
        st.metric("Validation Accuracy", f"{training['validation_accuracy']:.3f}")
    with col3:
        st.metric("Agreement Rate", f"{comparison['agreement_rate']:.3f}")
    with col4:
        st.metric("Avg Confidence", f"{comparison['average_ml_confidence']:.3f}")
    
    # Feature importance
    st.subheader("📊 Top Features")
    top_features = training['top_features'][:10]
    
    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    st.bar_chart(feature_df.set_index('Feature')['Importance'])
    
    # Signal comparison
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
    
    # Agreement analysis
    st.info(f"""
    **ML vs Rule-Based Summary:**
    - Total signals analyzed: {comparison['total_signals']}
    - Agreements: {comparison['agreements']} ({comparison['agreement_rate']:.1%})
    - High confidence signals: {comparison['high_confidence_signals']}
    - High confidence agreement: {comparison['high_confidence_agreement_rate']:.1%}
    """)


def display_ai_insights_tab(gpt_insights: Dict[str, Any], trades: pd.DataFrame):
    """Display AI-generated insights and trade explanations."""
    
    if not gpt_insights or 'error' in gpt_insights:
        st.warning("🤖 AI insights not available")
        return
    
    # Trade explanations
    display_trade_explanations(trades)
    
    # Trade performance chart with AI insights
    if not trades.empty:
        display_gpt_trade_insights_chart(trades)


def display_charts_tab(price_data: pd.DataFrame,
                      rule_signals: pd.DataFrame,
                      ml_results: Optional[Dict[str, Any]],
                      equity_curve: pd.DataFrame,
                      trades: pd.DataFrame):
    """Display comprehensive charts."""
    
    # Price and signals chart
    fig_price = create_price_and_signals_chart(price_data, rule_signals)
    
    # Add ML signals if available
    if ml_results and ml_results.get('success', False):
        # This would require updating the chart function to handle ML signals
        # For now, just show rule-based
        pass
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Equity curve
    fig_equity = create_equity_curve_chart(equity_curve)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Drawdown chart
    fig_drawdown = create_drawdown_chart(equity_curve)
    st.plotly_chart(fig_drawdown, use_container_width=True)


def display_trade_log_tab(trades: pd.DataFrame):
    """Display detailed trade log."""
    
    if trades.empty:
        st.info("No trades executed")
        return
    
    st.subheader("📋 Complete Trade Log")
    
    # Trade summary
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
    
    # Display trade table
    display_columns = [
        'entry_time', 'exit_time', 'side', 'entry_price', 
        'exit_price', 'net_pnl', 'return', 'duration'
    ]
    
    # Add GPT explanation column if available
    if 'gpt_explanation' in trades.columns:
        display_columns.append('gpt_explanation')
    
    available_columns = [col for col in display_columns if col in trades.columns]
    
    st.dataframe(
        trades[available_columns],
        use_container_width=True,
        height=400
    )


def initialize_strategy(config: Dict[str, Any]):
    """Initialize strategy based on configuration."""
    strategy_type = config['strategy']['category']
    parameters = config['strategy']['parameters']
    
    if strategy_type == 'momentum':
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
        )
    else:
        # Default to momentum
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
        )


if __name__ == "__main__":
    main()