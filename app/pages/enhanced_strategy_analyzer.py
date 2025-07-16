"""
Enhanced Strategy Analyzer - AI-Powered Trading Analysis Platform

This module represents the next generation of strategy analysis, integrating
artificial intelligence and machine learning to provide unprecedented insights
into trading strategy performance and market behavior.

Revolutionary Features:
- GPT-powered strategy explanations and trade commentary
- Machine learning signal prediction and comparison
- Real-time AI insights during backtesting
- Advanced ML feature importance analysis
- Confidence-scored signal generation
- Interactive AI-human collaboration interface

AI Integration Architecture:
The enhanced analyzer implements a sophisticated AI pipeline:

1. ML Signal Layer: Parallel signal generation using rule-based vs ML approaches
2. GPT Commentary Layer: Real-time natural language explanations of trades
3. Confidence Scoring: ML model uncertainty quantification
4. Feature Importance: Understanding which indicators drive performance
5. AI Insights Generation: Strategic analysis and recommendations
6. Human-AI Interface: Seamless integration of AI assistance

Core AI Components:
- MLSignalClassifier: Machine learning model for signal prediction
- GPTService: Natural language processing for trade explanations
- GPTEnhancedSimulator: AI-augmented trading simulation
- Confidence Analysis: Model uncertainty and reliability assessment
- Feature Engineering: Automated indicator discovery and ranking

Educational AI Philosophy:
This platform demonstrates how AI can enhance rather than replace human
trading expertise by:
- Providing explanations for every AI decision
- Comparing AI predictions with traditional methods
- Quantifying model confidence and uncertainty
- Teaching through interactive exploration
- Maintaining transparency in AI reasoning

Advanced Workflow:
1. Configuration: Enhanced sidebar with AI feature toggles
2. Data Preparation: Automated feature engineering for ML models
3. Model Training: Real-time ML model training on historical data
4. Parallel Signal Generation: Rule-based vs ML signal comparison
5. AI-Enhanced Simulation: Trading simulation with GPT commentary
6. Comprehensive Analysis: Multi-modal result presentation
7. AI Insights: Strategic recommendations and explanations

Machine Learning Pipeline:
- Feature Engineering: Technical indicators, market patterns, sentiment
- Model Selection: Ensemble methods for robust predictions
- Training Process: Cross-validation with temporal splits
- Prediction Generation: Confidence-scored signal classification
- Performance Comparison: ML vs rule-based signal analysis
- Model Interpretation: Feature importance and decision explanations

GPT Integration Features:
- Strategy Overview: AI-generated strategy descriptions
- Trade Commentary: Explanations for individual trade decisions
- Performance Analysis: Natural language summary of results
- Risk Assessment: AI-powered risk analysis and recommendations
- Market Context: Integration of market conditions in explanations

User Experience Innovation:
- Tabbed Interface: Organized presentation of complex AI analysis
- Progressive Disclosure: Layered information from overview to details
- Interactive AI: Real-time AI assistance and explanations
- Visual AI Integration: Charts enhanced with AI insights
- Export Capabilities: AI insights included in downloadable reports

Research and Development:
This platform serves as a research environment for:
- AI-human collaboration in trading
- Explainable AI in financial applications
- Machine learning model comparison and validation
- Natural language generation for financial analysis
- Advanced visualization of AI decision processes

Usage Examples:
    # Run enhanced analyzer with full AI features
    streamlit run app/pages/enhanced_strategy_analyzer.py
    
    # Configure for specific AI features
    config = {
        'enable_ml': True,
        'enable_gpt': True,
        'gpt_model': 'gpt-4',
        'ml_confidence_threshold': 0.7
    }

Integration Requirements:
    - OpenAI API access for GPT functionality
    - Scikit-learn for machine learning models
    - Advanced visualization libraries
    - Enhanced backend simulation engine
    - ML feature engineering pipeline

Author: Strategy Explainer AI Research Team
Version: 3.0 (AI-Enhanced)
Last Updated: 2024
AI Features: GPT-4 Integration, ML Signal Prediction, Confidence Scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add the project root to the path for comprehensive module access
# Enhanced analyzer requires access to both frontend and backend components
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing UI components for consistency with base analyzer
from components.sidebar import render_sidebar, display_strategy_info, render_run_button
from components.metrics_cards import display_key_metrics, display_detailed_metrics
from components.charts import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_price_and_signals_chart
)

# Import advanced GPT and ML components for AI-powered analysis
# These components provide the revolutionary AI features of the enhanced analyzer
from components.gpt_display import (
    display_strategy_overview_gpt,     # GPT-generated strategy explanations
    display_performance_summary_gpt,   # AI performance analysis
    display_trade_explanations,        # Individual trade commentary
    display_gpt_insights_sidebar,      # Real-time AI insights
    display_ai_metrics_comparison,     # AI-enhanced metrics display
    display_gpt_trade_insights_chart,  # Trade analysis with AI overlay
    create_gpt_enabled_config          # GPT configuration interface
)

# Import enhanced backend modules with AI capabilities
try:
    from backend.momentum_backtest import MomentumBacktest
    from backend.data_loader import DataLoader
    from backend.metrics import PerformanceMetrics
    from backend.enhanced_simulator import GPTEnhancedSimulator, create_enhanced_simulator
    from backend.ml_classifier import MLSignalClassifier, create_ml_classifier
    from backend.gpt_service import GPTService
except ImportError as e:
    # Enhanced error handling with AI feature guidance
    st.error(f"Error importing backend modules: {e}")
    st.info("""
    Please ensure all backend modules are properly installed and accessible.
    
    For AI features, you'll also need:
    - OpenAI API key configured
    - Scikit-learn for ML models
    - Additional Python dependencies
    """)


def main():
    """
    Enhanced Strategy Analyzer Main Application.
    
    This is the flagship interface showcasing the full potential of AI-powered
    trading analysis. The application seamlessly integrates traditional
    backtesting with cutting-edge AI capabilities.
    
    Revolutionary Features:
    - Real-time GPT commentary on trading decisions
    - Machine learning signal prediction and comparison
    - AI-powered strategy analysis and recommendations
    - Interactive AI-human collaboration interface
    - Advanced visualization with AI insights overlay
    
    User Experience Design:
    - Premium AI branding and visual design
    - Progressive feature enablement
    - Real-time AI feedback and insights
    - Comprehensive AI configuration options
    - Educational AI explanations throughout
    
    AI Integration Philosophy:
    The interface demonstrates how AI can enhance human trading expertise
    rather than replace it, providing transparency, explanation, and
    collaborative intelligence.
    """
    # Enhanced page configuration for AI-powered experience
    st.set_page_config(
        page_title="🚀 AI-Powered Strategy Analyzer",  # Premium AI branding
        page_icon="🤖",  # AI-focused icon
        layout="wide",  # Full-width for complex AI visualizations
        initial_sidebar_state="expanded"  # AI controls need prominent display
    )
    
    # Premium AI-branded header with gradient styling
    # This immediately signals the advanced AI capabilities to users
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🚀 AI-Powered Strategy Analyzer</h1>
        <p style="color: white; margin: 5px 0 0 0; opacity: 0.9;">Real-time backtesting with GPT commentary and ML signal prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar configuration with AI-specific controls
    # Builds upon base configuration with additional AI feature settings
    config = render_sidebar()
    gpt_config = create_gpt_enabled_config()
    config.update(gpt_config)
    
    # Display strategy information with AI enhancement indicators
    display_strategy_info(config)
    
    # AI Feature Selection Interface
    # =============================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Features")
    
    # Create intuitive AI feature toggles with explanatory help text
    col1, col2 = st.sidebar.columns(2)
    with col1:
        enable_ml = st.checkbox(
            "🧠 ML Signals", 
            value=True, 
            help="Compare rule-based vs ML predicted signals"
        )
    with col2:
        enable_gpt = st.checkbox(
            "💬 GPT Analysis", 
            value=config.get('enable_gpt', True), 
            help="AI-powered trade explanations"
        )
    
    # Update configuration with AI feature selections
    config['enable_ml'] = enable_ml
    config['enable_gpt'] = enable_gpt
    
    # Main application flow control with AI feature awareness
    if render_run_button():
        run_enhanced_analysis(config)
    else:
        display_enhanced_welcome_screen(enable_ml, enable_gpt)


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


def run_enhanced_analysis(config: Dict[str, Any]):
    """
    Execute Enhanced AI-Powered Strategy Analysis.
    
    This is the core AI analysis pipeline that orchestrates machine learning
    model training, GPT insight generation, and comprehensive AI-enhanced
    backtesting. The function implements a sophisticated multi-stage process
    that seamlessly integrates traditional analysis with cutting-edge AI.
    
    Enhanced Pipeline Stages:
    1. Data Loading: Historical market data with feature engineering
    2. Rule-Based Signals: Traditional strategy signal generation
    3. ML Model Training: Real-time machine learning model training
    4. Signal Comparison: ML vs rule-based signal analysis
    5. AI-Enhanced Simulation: Trading simulation with GPT commentary
    6. Performance Analytics: Comprehensive metrics calculation
    7. GPT Insights: AI-generated strategy analysis and recommendations
    8. Multi-Modal Results: Advanced tabbed presentation with AI insights
    
    AI Integration Features:
    - Parallel signal generation (rule-based vs ML)
    - Real-time ML model training and validation
    - GPT-powered trade explanations and strategy analysis
    - Confidence scoring for ML predictions
    - Feature importance analysis for model interpretation
    - AI-enhanced visualization and reporting
    
    Args:
        config (Dict[str, Any]): Enhanced configuration including AI settings:
            - enable_ml: Machine learning features toggle
            - enable_gpt: GPT analysis features toggle
            - ml_confidence_threshold: ML prediction confidence filter
            - gpt_model: GPT model selection
            
    Error Handling:
    - Graceful AI feature degradation
    - ML training failure recovery
    - GPT API error handling
    - Partial analysis completion
    - User-friendly AI error explanations
    """
    
    try:
        # Enhanced progress tracking for complex AI pipeline
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Stage 1: Enhanced Data Loading with Feature Engineering
        # =====================================================
        status_text.text("📊 Loading market data...")
        progress_bar.progress(10)
        
        # Initialize advanced data loader with caching and preprocessing
        data_loader = DataLoader()
        symbol = config['data']['symbol']
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        
        # Fetch comprehensive market data for AI analysis
        price_data = data_loader.fetch_data(symbol, start_date, end_date)
        
        # Stage 2: Rule-Based Signal Generation
        # ===================================
        status_text.text("⚡ Generating rule-based signals...")
        progress_bar.progress(25)
        
        # Initialize traditional strategy for baseline comparison
        strategy = initialize_strategy(config)
        rule_signals = strategy.generate_signals(price_data)
        
        # Stage 3: Machine Learning Model Training (Conditional)
        # =====================================================
        ml_results = None
        if config.get('enable_ml', False):
            status_text.text("🧠 Training ML model...")
            progress_bar.progress(40)
            
            # Execute comprehensive ML training and prediction pipeline
            ml_results = train_and_predict_ml(price_data, rule_signals, config)
        
        # Stage 4: AI-Enhanced Trading Simulation
        # ======================================
        status_text.text("🔄 Running enhanced simulation...")
        progress_bar.progress(60)
        
        # Create GPT-enhanced simulator with AI commentary capabilities
        simulator = create_enhanced_simulator(config, strategy.__class__.__name__)
        
        # Execute AI-enhanced trading simulation with real-time commentary
        equity_curve, trades = simulator.simulate_strategy(
            price_data, 
            rule_signals, 
            symbol
        )
        
        # Stage 5: Comprehensive Performance Analytics
        # ==========================================
        status_text.text("📊 Calculating performance metrics...")
        progress_bar.progress(80)
        
        # Calculate detailed performance metrics for AI analysis
        perf_metrics = PerformanceMetrics()
        strategy_metrics = perf_metrics.calculate_all_metrics(equity_curve, trades)
        
        # Stage 6: GPT Insights Generation (Conditional)
        # =============================================
        gpt_insights = {}
        if config.get('enable_gpt', False) and simulator.gpt_service and simulator.gpt_service.enabled:
            status_text.text("🤖 Generating AI insights...")
            progress_bar.progress(90)
            
            # Generate comprehensive GPT-powered analysis
            gpt_insights = generate_gpt_insights(
                simulator, config, strategy_metrics, symbol
            )
        
        # Stage 7: Complete Analysis Presentation
        # ======================================
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators for results display
        progress_container.empty()
        
        # Stage 8: Multi-Modal Results Display
        # ===================================
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
        # Enhanced error handling with AI-specific guidance
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)
        
        # Provide specific guidance for AI-related issues
        st.markdown("""
        **AI Feature Troubleshooting:**
        - Check OpenAI API key configuration for GPT features
        - Verify ML dependencies (scikit-learn, pandas) are installed
        - Ensure sufficient data for ML model training (>100 points)
        - Try disabling AI features if errors persist
        """)


def train_and_predict_ml(price_data: pd.DataFrame, 
                        rule_signals: pd.DataFrame,
                        config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Comprehensive Machine Learning Pipeline.
    
    This function implements a sophisticated ML pipeline that trains models
    on historical data, generates predictions, and provides comprehensive
    comparison with rule-based signals.
    
    ML Pipeline Components:
    1. Feature Engineering: Technical indicators and market patterns
    2. Model Training: Cross-validated ensemble learning
    3. Signal Prediction: Confidence-scored signal classification
    4. Performance Comparison: ML vs rule-based signal analysis
    5. Model Interpretation: Feature importance and decision explanations
    
    Training Process:
    - Temporal split validation for realistic performance assessment
    - Ensemble methods for robust predictions
    - Hyperparameter optimization for model tuning
    - Overfitting prevention through regularization
    
    Prediction Features:
    - Confidence scoring for prediction reliability
    - Signal strength quantification
    - Uncertainty estimation for risk management
    - Decision boundary analysis
    
    Args:
        price_data (pd.DataFrame): Historical market data for training
        rule_signals (pd.DataFrame): Rule-based signals for comparison
        config (Dict[str, Any]): ML configuration parameters
        
    Returns:
        Dict[str, Any]: Comprehensive ML analysis results containing:
            - classifier: Trained ML model instance
            - training_results: Training performance metrics
            - predictions: ML signal predictions with confidence
            - comparison: Detailed ML vs rule-based comparison
            - success: Training success indicator
            
    Error Handling:
    - Insufficient data detection
    - Training failure recovery
    - Model validation error handling
    - Graceful degradation for ML failures
    """
    
    try:
        # Initialize ML classifier with enhanced configuration
        ml_classifier = create_ml_classifier(config)
        
        # Execute comprehensive training pipeline with validation
        training_results = ml_classifier.train(price_data, rule_signals)
        
        # Generate ML predictions with confidence scoring
        ml_predictions = ml_classifier.predict_signals(price_data)
        
        # Perform detailed comparison analysis with rule-based signals
        comparison = ml_classifier.compare_with_rule_based(price_data, rule_signals)
        
        return {
            'classifier': ml_classifier,
            'training_results': training_results,
            'predictions': ml_predictions,
            'comparison': comparison,
            'success': True
        }
        
    except Exception as e:
        # User-friendly error handling for ML training failures
        st.warning(f"ML training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def generate_gpt_insights(simulator: GPTEnhancedSimulator,
                         config: Dict[str, Any],
                         metrics: Dict[str, Any],
                         symbol: str) -> Dict[str, Any]:
    """
    Generate Comprehensive GPT-Powered Insights.
    
    This function orchestrates the generation of natural language insights
    using GPT models, providing human-like analysis and explanations of
    trading strategy performance.
    
    GPT Analysis Components:
    1. Strategy Overview: High-level strategy description and methodology
    2. Performance Summary: Natural language performance analysis
    3. Trade Insights: Individual trade explanations and commentary
    4. Risk Assessment: AI-powered risk analysis and recommendations
    5. Market Context: Integration of market conditions in explanations
    
    Insight Generation Process:
    - Context-aware prompt engineering
    - Multi-turn GPT conversations for depth
    - Technical analysis integration
    - Risk-adjusted performance commentary
    - Actionable recommendations generation
    
    Args:
        simulator (GPTEnhancedSimulator): AI-enhanced trading simulator
        config (Dict[str, Any]): Strategy configuration
        metrics (Dict[str, Any]): Performance metrics for analysis
        symbol (str): Trading symbol for context
        
    Returns:
        Dict[str, Any]: Comprehensive GPT insights including:
            - strategy_overview: AI-generated strategy explanation
            - performance_summary: Natural language performance analysis
            - trade_insights: Individual trade commentary
            - risk_analysis: AI-powered risk assessment
            
    Error Handling:
    - GPT API failure recovery
    - Partial insight generation
    - Fallback to traditional analysis
    - User notification of AI limitations
    """
    
    insights = {}
    
    try:
        # Generate comprehensive strategy overview with AI analysis
        strategy_params = config.get('strategy', {}).get('parameters', {})
        insights['strategy_overview'] = simulator.generate_strategy_overview(strategy_params)
        
        # Create natural language performance summary
        insights['performance_summary'] = simulator.generate_performance_summary(metrics)
        
        # Extract trade-level insights from simulator
        # These are generated during the simulation process
        insights['trade_insights'] = simulator.get_trade_insights()
        
    except Exception as e:
        # Graceful handling of GPT service failures
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


def initialize_strategy(config: Dict[str, Any]):
    """
    Initialize Enhanced Strategy Instance.
    
    Factory function for creating strategy instances with enhanced
    configuration support for AI-powered analysis.
    
    Enhanced Strategy Support:
    - Traditional momentum strategies
    - AI-enhanced parameter optimization
    - Dynamic configuration injection
    - Future support for ML-hybrid strategies
    
    Args:
        config: Enhanced configuration dictionary
        
    Returns:
        Strategy instance ready for AI-enhanced analysis
    """
    # Extract strategy configuration for enhanced initialization
    strategy_type = config['strategy']['category']
    parameters = config['strategy']['parameters']
    
    if strategy_type == 'momentum':
        # Initialize momentum strategy with enhanced parameter support
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
        )
    else:
        # Default fallback with enhanced parameter support
        # TODO: Add support for additional AI-enhanced strategy types
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
        )


# Enhanced Application Entry Point
# ===============================
if __name__ == "__main__":
    # Direct execution support for AI-powered standalone testing
    main()