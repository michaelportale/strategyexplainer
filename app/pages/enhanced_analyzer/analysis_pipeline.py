"""
Enhanced Strategy Analyzer - Analysis Pipeline Module

This module contains the core analysis workflow and pipeline functions for 
the enhanced strategy analyzer. It orchestrates the entire analysis process
from data loading to results display.

Functions:
- run_enhanced_analysis: Main analysis orchestration function
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import backend modules
from backend.data_loader import DataLoader
from backend.metrics import PerformanceMetrics
from backend.enhanced_simulator import create_enhanced_simulator

# Import analysis modules
from .ai_analysis import train_and_predict_ml, generate_gpt_insights
from .display import display_enhanced_results
from .utils import initialize_strategy


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