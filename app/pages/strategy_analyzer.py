"""
Strategy Analyzer - Core Streamlit Application Page

This module provides the main strategy analysis interface for the trading system,
offering comprehensive backtesting and visualization capabilities through an 
interactive Streamlit application.

Core Features:
- Interactive strategy configuration through sidebar controls
- Real-time data loading and validation with progress indicators
- Comprehensive backtesting engine integration
- Advanced financial visualizations and charts
- Performance metrics calculation and display
- Benchmark comparison and analysis
- Trade-level analysis with downloadable results
- Risk assessment and drawdown analysis

Architecture:
The analyzer follows a modular design pattern with clear separation of concerns:

1. Configuration Layer: Sidebar component handles user inputs and validation
2. Data Layer: DataLoader integration for market data retrieval
3. Strategy Layer: Pluggable strategy initialization and signal generation
4. Simulation Layer: TradingSimulator for realistic trade execution
5. Analytics Layer: PerformanceMetrics for comprehensive evaluation
6. Visualization Layer: Plotly-based interactive charts and displays

User Experience Flow:
1. Welcome Screen: Guides new users through available features
2. Configuration: Interactive sidebar for strategy and data setup
3. Analysis Execution: Progress-tracked backtesting pipeline
4. Results Display: Multi-section comprehensive analysis
5. Export Functionality: CSV downloads for further analysis

Educational Value:
The interface serves as both a practical trading tool and educational platform,
demonstrating best practices in:
- Quantitative strategy development
- Risk management principles
- Performance evaluation methodologies
- Financial data visualization
- Trading system architecture

Integration Points:
- Backend strategy implementations (momentum, mean reversion, breakout)
- Data loading and caching systems
- Metrics calculation engine
- Visualization component library
- Trading simulation framework

Usage Examples:
    # Run standalone application
    streamlit run app/pages/strategy_analyzer.py
    
    # Integration in multi-page app
    from app.pages.strategy_analyzer import main
    main()

Dependencies:
    - streamlit: Web application framework
    - pandas/numpy: Data manipulation and analysis
    - plotly: Interactive financial charting
    - Backend modules: Strategy engine and simulation

Author: Strategy Explainer Development Team
Version: 2.0
Last Updated: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add the backend directory to the path for module imports
# This allows the frontend to access backend trading components
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))

# Import UI component modules for modular interface construction
from app.components.sidebar import render_sidebar, display_strategy_info, render_run_button
from app.components.metrics_cards import (
    display_key_metrics, 
    display_detailed_metrics, 
    display_benchmark_comparison,
    create_metrics_summary_chart,
    display_trade_distribution
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
from app.components.enhanced_analytics_layout import render_enhanced_analytics

# Import backend trading system components
# These modules provide the core functionality for strategy analysis
try:
    from backend.strategies import SmaEmaRsiStrategy
    from backend.visualize import create_performance_charts
    from backend.data_loader import DataLoader
    from backend.metrics import PerformanceMetrics
    from backend.simulate import TradingSimulator
except ImportError as e:
    # Graceful error handling for missing dependencies
    st.error(f"Error importing backend modules: {e}")
    st.info("Please ensure all backend modules are properly installed and accessible.")


def main():
    """
    Main entry point for the Strategy Analyzer application.
    
    Initializes the Streamlit page configuration, renders the main interface,
    and orchestrates the user interaction flow between configuration and analysis.
    
    Page Configuration:
    - Wide layout for comprehensive data display
    - Custom page title and icon for branding
    - Expanded sidebar for easy access to controls
    
    User Interface Flow:
    1. Render sidebar configuration interface
    2. Display strategy information and validation
    3. Show run button when configuration is complete
    4. Execute analysis or display welcome screen based on user action
    
    Error Handling:
    - Graceful degradation for missing backend modules
    - User-friendly error messages and guidance
    - Input validation through sidebar components
    """
    # Configure Streamlit page settings for optimal user experience
    st.set_page_config(
        page_title="Strategy Analyzer - Strategy Explainer",  # Browser tab title
        page_icon="📈",  # Favicon for visual identification
        layout="wide",  # Full-width layout for data-heavy interface
        initial_sidebar_state="expanded"  # Show controls by default
    )
    
    # Main page header with visual hierarchy
    st.title("📈 Strategy Analyzer")
    st.markdown("---")  # Visual separator for clean design
    
    # Render interactive sidebar and capture user configuration
    # This returns a comprehensive config dictionary with all user settings
    config = render_sidebar()
    
    # Display strategy information based on current configuration
    # Provides real-time feedback on selected strategy and parameters
    display_strategy_info(config)
    
    # Main application logic flow control
    # Run analysis if user clicks the run button, otherwise show welcome screen
    if render_run_button():
        run_analysis(config)
    else:
        display_welcome_screen()


def display_welcome_screen():
    """
    Display a professional welcome screen showcasing the application's capabilities.
    
    Purpose:
    - Professional introduction to institutional-grade features
    - Clear guidance for getting started with analysis
    - Showcase advanced features and professional presentation
    - Set expectations for comprehensive analysis results
    """
    # Professional header with gradient styling
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🚀 Professional Strategy Analyzer</h1>
        <p style="color: white; margin: 10px 0 0 0; opacity: 0.9; font-size: 1.2em;">Institutional-grade backtesting with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase with professional layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Get Started
        
        **Professional Workflow:**
        1. **📊 Select Strategy & Risk Profile**
           - Choose from momentum, mean reversion, or breakout strategies
           - Pick Conservative, Moderate, or Aggressive presets
        
        2. **📈 Configure Analysis**
           - Select ticker symbol and time period
           - Choose benchmark for comparison
           - Set advanced analysis options
        
        3. **🚀 Run Professional Analysis**
           - Comprehensive performance metrics
           - Risk analysis and drawdown statistics
           - AI-powered insights and explanations
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Professional Features
        
        **Advanced Analytics:**
        - 📈 **Tabbed Results Layout**
        - 📋 **Comprehensive Metrics**
        - 🎯 **Trade-Level Analysis**
        - 🤖 **AI-Powered Insights**
        
        **Risk Management:**
        - ⚠️ **Value at Risk (VaR)**
        - 📉 **Drawdown Analysis**
        - 🔗 **Correlation Studies**
        - 🔥 **Stress Testing**
        
        **Professional Output:**
        - 💾 **Export Capabilities**
        - 📄 **PDF Reports**
        - 📊 **Interactive Charts**
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Strategy Categories
        
        **Momentum Strategies:**
        - Simple momentum with price rate of change
        - RSI momentum with overbought/oversold levels
        
        **Mean Reversion:**
        - Bollinger Bands mean reversion
        - Z-Score statistical reversion
        
        **Breakout Strategies:**
        - Channel breakout systems
        - Volume-confirmed breakouts
        
        **Risk Profiles:**
        - 🛡️ **Conservative**: Lower risk, smaller positions
        - ⚖️ **Moderate**: Balanced risk-reward
        - 🚀 **Aggressive**: Higher risk, larger positions
        """)
    
    # Professional sample visualization
    st.markdown("---")
    st.markdown("### 📈 Sample Professional Analysis")
    
    # Create 4-metric preview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Total Return",
            value="24.3%",
            delta="5.2%",
            help="Sample cumulative return"
        )
    
    with col2:
        st.metric(
            label="📊 Sharpe Ratio",
            value="1.42",
            delta="0.18",
            help="Sample risk-adjusted return"
        )
    
    with col3:
        st.metric(
            label="⚠️ Max Drawdown",
            value="8.7%",
            delta="-2.1%",
            delta_color="inverse",
            help="Sample maximum decline"
        )
    
    with col4:
        st.metric(
            label="🎯 Win Rate",
            value="67%",
            delta="12%",
            help="Sample winning trades percentage"
        )
    
    # Sample chart showing professional presentation
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # More realistic market data simulation
    strategy_returns = np.random.randn(252) * 0.012 + 0.0003
    benchmark_returns = np.random.randn(252) * 0.008 + 0.0002
    
    strategy_equity = (1 + pd.Series(strategy_returns)).cumprod() * 10000
    benchmark_equity = (1 + pd.Series(benchmark_returns)).cumprod() * 10000
    
    sample_data = pd.DataFrame({
        'Strategy Performance': strategy_equity,
        'S&P 500 Benchmark': benchmark_equity
    }, index=dates)
    
    st.line_chart(sample_data, height=400)
    
    # Professional call-to-action
    st.markdown("""
    <div style='text-align: center; padding: 25px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;'>
        <h3 style='color: #333; margin-bottom: 15px;'>Ready for Professional Analysis? 🚀</h3>
        <p style='margin: 0; color: #666; font-size: 1.1em;'>
            Configure your strategy in the sidebar and click <strong>"Run Analysis"</strong> to generate comprehensive institutional-grade results.
        </p>
    </div>
    """, unsafe_allow_html=True)


def run_analysis(config: Dict[str, Any]):
    """
    Execute the complete strategy analysis pipeline.
    
    This is the core function that orchestrates the entire backtesting process,
    from data loading through results display. It implements a robust pipeline
    with progress tracking, error handling, and comprehensive result generation.
    
    Pipeline Stages:
    1. Data Loading: Fetch historical market data for specified symbol/period
    2. Strategy Initialization: Create strategy instance with user parameters
    3. Signal Generation: Apply strategy logic to generate trading signals
    4. Trade Simulation: Execute realistic trading simulation with costs
    5. Metrics Calculation: Compute comprehensive performance statistics
    6. Benchmark Analysis: Compare strategy performance against market index
    7. Results Display: Present findings through interactive visualizations
    
    Args:
        config (Dict[str, Any]): Complete configuration dictionary containing:
            - strategy: Strategy type and parameters
            - data: Symbol, date range, and data source settings
            - trading: Commission, slippage, and execution parameters
            
    Error Handling:
    - Graceful degradation for data loading failures
    - Strategy initialization error recovery
    - Comprehensive exception logging and user feedback
    - Partial result display when possible
    
    Performance Considerations:
    - Progress tracking for long-running operations
    - Efficient data loading with caching
    - Optimized metrics calculation
    - Memory management for large datasets
    """
    try:
        # Create progress tracking interface for user feedback
        with st.spinner("Loading data and running analysis..."):
            # Initialize progress indicators for multi-stage process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Stage 1: Data Loading and Validation
            # ====================================
            status_text.text("📊 Loading market data...")
            progress_bar.progress(10)
            
            # Initialize data loader with caching for performance
            data_loader = DataLoader()
            
            # Fetch historical price data based on user configuration
            price_data = data_loader.load_stock_data(
                symbol=config['data']['symbol'],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            
            # Validate data availability and quality
            if price_data.empty:
                st.error(f"No data available for {config['data']['symbol']}")
                st.info("Please try a different symbol or date range.")
                return
            
            progress_bar.progress(25)
            
            # Stage 2: Strategy Initialization and Signal Generation
            # ====================================================
            status_text.text("🔄 Running strategy backtest...")
            
            # Create strategy instance based on user configuration
            # This supports multiple strategy types with dynamic parameter injection
            strategy = initialize_strategy(config)
            
            # Generate trading signals using strategy logic
            # Signals include entry/exit points with confidence levels
            signals = strategy.generate_signals(price_data)
            
            progress_bar.progress(50)
            
            # Stage 3: Trading Simulation with Realistic Execution
            # ===================================================
            status_text.text("💼 Simulating trades...")
            
            # Initialize trading simulator with user-specified parameters
            # Includes realistic transaction costs and market impact
            simulator = TradingSimulator(
                initial_capital=100000,  # Standard starting capital
                commission=config['trading']['commission'],
                slippage=config['trading']['slippage']
            )
            
            # Execute complete trading simulation
            # Returns equity curve and detailed trade records
            equity_curve, trades = simulator.simulate_strategy(price_data, signals)
            
            progress_bar.progress(75)
            
            # Stage 4: Performance Metrics and Benchmark Analysis
            # ==================================================
            status_text.text("📊 Calculating performance metrics...")
            
            # Calculate comprehensive performance statistics
            metrics_calculator = PerformanceMetrics()
            strategy_metrics = metrics_calculator.calculate_all_metrics(
                equity_curve, trades, risk_free_rate=0.02
            )
            
            # Load and analyze benchmark data for comparison
            benchmark_data = data_loader.load_stock_data(
                symbol=config['trading']['benchmark'],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            
            # Calculate benchmark metrics for relative performance analysis
            benchmark_metrics = None
            if not benchmark_data.empty:
                # Convert benchmark prices to returns and equity curve
                benchmark_returns = benchmark_data['close'].pct_change().dropna()
                benchmark_equity = (1 + benchmark_returns).cumprod() * 100000
                
                # Calculate benchmark performance metrics
                benchmark_metrics = metrics_calculator.calculate_all_metrics(
                    pd.DataFrame({'equity': benchmark_equity}), 
                    pd.DataFrame(),  # No trades for benchmark
                    risk_free_rate=0.02
                )
            
            # Complete progress tracking
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators to make room for results
            progress_bar.empty()
            status_text.empty()
            
        # Stage 5: Comprehensive Results Display
        # ====================================
        display_results(
            config=config,
            price_data=price_data,
            signals=signals,
            equity_curve=equity_curve,
            trades=trades,
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            benchmark_data=benchmark_data
        )
        
    except Exception as e:
        # Comprehensive error handling with user guidance
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)  # Technical details for debugging
        
        # Provide helpful suggestions for common issues
        st.markdown("""
        **Common solutions:**
        - Check if the symbol exists and has data for the selected period
        - Verify your internet connection for data loading
        - Try a different date range or symbol
        - Ensure all required backend modules are installed
        """)


def initialize_strategy(config: Dict[str, Any]):
    """
    Initialize the appropriate strategy instance based on user configuration.
    
    This factory function creates strategy objects dynamically based on user
    selections, allowing for flexible strategy composition and parameter injection.
    
    Strategy Types Supported:
    - Momentum: Trend-following strategies using price momentum
    - Mean Reversion: Contrarian strategies exploiting price reversals
    - Breakout: Volatility-based strategies capturing range breaks
    
    Args:
        config (Dict[str, Any]): Configuration containing:
            - strategy.category: Strategy type identifier
            - strategy.parameters: Dictionary of strategy-specific parameters
            
    Returns:
        Strategy instance: Configured strategy ready for signal generation
        
    Parameter Mapping:
    - lookback_period: Historical data window for calculations
    - threshold: Signal generation sensitivity
    - position_size: Risk per trade as portfolio percentage
    - stop_loss: Maximum loss tolerance per trade
    - take_profit: Target profit for trade exit
    
    Design Pattern:
    Uses factory pattern for extensible strategy creation, allowing easy
    addition of new strategy types without modifying existing code.
    """
    # Extract strategy configuration from user settings
    strategy_type = config['strategy']['category']
    parameters = config['strategy']['parameters']
    
    # Strategy factory implementation with parameter injection
    if strategy_type == 'momentum':
        # Initialize momentum-based strategy with trend-following logic
        return SmaEmaRsiStrategy(
            fast_period=parameters.get('lookback_period', 20),  # Default: 20-day lookback
            slow_period=parameters.get('slow_period', 50),  # Default: 50-day slow MA
            rsi_period=parameters.get('rsi_period', 14),  # Default: 14-day RSI
            rsi_oversold=parameters.get('rsi_oversold', 30),  # Default: 30 oversold
            rsi_overbought=parameters.get('rsi_overbought', 70)  # Default: 70 overbought
        )
    else:
        # Default fallback to momentum strategy for unsupported types
        # TODO: Add support for mean reversion and breakout strategies
        return SmaEmaRsiStrategy(
            fast_period=parameters.get('lookback_period', 20),
            slow_period=50,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70
        )


def display_results(config: Dict[str, Any], 
                   price_data: pd.DataFrame,
                   signals: pd.DataFrame,
                   equity_curve: pd.DataFrame,
                   trades: pd.DataFrame,
                   strategy_metrics: Dict[str, Any],
                   benchmark_metrics: Optional[Dict[str, Any]] = None,
                   benchmark_data: Optional[pd.DataFrame] = None):
    """
    Display comprehensive analysis results using professional tabbed layout.
    
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
    """
    from app.components.professional_results_layout import display_professional_results
    
    # Gather GPT insights if AI features are enabled
    gpt_insights = None
    if config.get('enable_gpt', False):
        # This would be populated by AI analysis if available
        gpt_insights = config.get('gpt_insights', None)
    
    # Use professional tabbed layout for results
    display_professional_results(
        config=config,
        price_data=price_data,
        signals=signals,
        equity_curve=equity_curve,
        trades=trades,
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        benchmark_data=benchmark_data,
        gpt_insights=gpt_insights
    )
    
    # Additional Export Section
    # ========================
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    st.markdown("Download analysis results for further processing.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export trading signals
        if signals is not None and len(signals) > 0:
            csv = signals.to_csv()
            st.download_button(
                label="⬇️ Download Signals CSV",
                data=csv,
                file_name=f"signals_{config['data']['symbol']}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # Export price data with indicators
        if st.button("📈 Export Price Data", use_container_width=True):
            csv = price_data.to_csv()
            st.download_button(
                label="⬇️ Download Price Data CSV",
                data=csv,
                file_name=f"price_data_{config['data']['symbol']}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        # Export trade records (if available)
        if not trades.empty and st.button("📋 Export Trade Records", use_container_width=True):
            csv = trades.to_csv()
            st.download_button(
                label="⬇️ Download Trades CSV",
                data=csv,
                file_name=f"trades_{config['data']['symbol']}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        elif trades.empty:
            st.info("No individual trades to export")
    
    # Analysis Summary Footer
    # ======================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin-top: 20px;'>
        <p style='margin: 0; color: #6c757d; font-style: italic;'>
            📈 Analysis completed successfully. Review all sections above for comprehensive strategy evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)


# Application Entry Point
# =======================
if __name__ == "__main__":
    # Direct execution support for standalone testing and development
    main() 