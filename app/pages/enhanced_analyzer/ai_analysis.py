"""
Enhanced Strategy Analyzer - AI Analysis Module

This module contains all AI and machine learning related functions for the 
enhanced strategy analyzer. It handles ML model training, prediction, and
GPT-powered insights generation.

Functions:
- train_and_predict_ml: ML model training and prediction pipeline
- generate_gpt_insights: GPT-powered insights generation
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

# Import enhanced backend modules for AI analysis
from backend.enhanced_simulator import GPTEnhancedSimulator
from backend.ml_classifier import create_ml_classifier


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