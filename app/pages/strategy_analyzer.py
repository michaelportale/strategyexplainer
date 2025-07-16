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
from components.sidebar import render_sidebar, display_strategy_info, render_run_button
from components.metrics_cards import (
    display_key_metrics, 
    display_detailed_metrics, 
    display_benchmark_comparison,
    create_metrics_summary_chart,
    display_trade_distribution
)
from components.charts import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_returns_distribution_chart,
    create_price_and_signals_chart,
    create_rolling_metrics_chart,
    create_monthly_returns_heatmap
)

# Import backend trading system components
# These modules provide the core functionality for strategy analysis
try:
    from momentum_backtest import MomentumBacktest
    from visualize import create_performance_charts
    from data_loader import DataLoader
    from metrics import PerformanceMetrics
    from simulate import TradingSimulator
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
    Display an engaging welcome screen for new users.
    
    Purpose:
    - Introduce users to the application's capabilities
    - Provide clear guidance on how to get started
    - Showcase key features and expected outcomes
    - Set appropriate expectations for analysis results
    
    Design Elements:
    - Centered layout for focus
    - Step-by-step instructions for clarity
    - Feature highlights to build excitement
    - Sample visualization to demonstrate value
    - Helpful tips for optimal user experience
    
    Educational Value:
    - Explains different strategy categories available
    - Sets expectations for analysis depth and quality
    - Demonstrates the type of insights users will receive
    """
    # Create centered layout for focused presentation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Main welcome content with clear value proposition
        st.markdown("""
        ## 🚀 Welcome to Strategy Analyzer
        
        **Get started by:**
        1. **Select a strategy** from the sidebar
        2. **Configure parameters** to match your preferences
        3. **Choose your data** (symbol and date range)
        4. **Click "Run Analysis"** to see results
        
        ### 📊 What you'll get:
        - **Comprehensive performance metrics**
        - **Interactive charts and visualizations**
        - **Risk analysis and drawdown statistics**
        - **AI-powered strategy explanation**
        - **Benchmark comparison**
        
        ### 🎯 Available Strategies:
        - **Momentum**: Trend-following strategies
        - **Mean Reversion**: Counter-trend strategies  
        - **Breakout**: Volatility-based strategies
        
        Start by selecting a strategy from the sidebar! 👈
        """)
        
        # Sample visualization to demonstrate expected output quality
        st.markdown("### 📈 Sample Analysis Preview:")
        
        # Create a realistic sample chart showing strategy vs benchmark performance
        # This helps users understand the type of analysis they'll receive
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic random walk data for demonstration
        # Strategy shows higher volatility but potentially better returns
        np.random.seed(42)  # Consistent sample data
        strategy_returns = np.random.randn(100) * 0.01
        benchmark_returns = np.random.randn(100) * 0.008
        
        sample_data = pd.DataFrame({
            'Strategy': np.cumsum(strategy_returns) + 1,
            'Benchmark': np.cumsum(benchmark_returns) + 1
        }, index=dates)
        
        # Display sample chart with Streamlit's built-in charting
        st.line_chart(sample_data)
        
        # Important disclaimer to set appropriate expectations
        st.info("💡 **Tip**: This is just a sample. Your actual analysis will show real market data and strategy performance!")


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
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),  # Default: 20-day lookback
            threshold=parameters.get('threshold', 0.02),  # Default: 2% threshold
            position_size=parameters.get('position_size', 0.1),  # Default: 10% position
            stop_loss=parameters.get('stop_loss', 0.05),  # Default: 5% stop loss
            take_profit=parameters.get('take_profit', 0.10)  # Default: 10% take profit
        )
    else:
        # Default fallback to momentum strategy for unsupported types
        # TODO: Add support for mean reversion and breakout strategies
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
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
    Display comprehensive analysis results in an organized, interactive format.
    
    This function creates a complete results dashboard with multiple visualization
    sections, performance analytics, and export capabilities. The layout is designed
    for both quick overview and detailed analysis.
    
    Results Sections:
    1. Strategy Information Header
    2. Key Performance Metrics Cards
    3. Interactive Financial Charts
    4. Detailed Performance Analysis
    5. Benchmark Comparison (if available)
    6. Trade-Level Analysis
    7. Risk and Return Distribution Analysis
    8. Export and Download Options
    
    Args:
        config: User configuration dictionary
        price_data: Historical price data for the analyzed symbol
        signals: Generated trading signals with entry/exit points
        equity_curve: Portfolio value over time
        trades: Detailed trade records with P&L
        strategy_metrics: Comprehensive performance statistics
        benchmark_metrics: Optional benchmark performance for comparison
        benchmark_data: Optional benchmark price data
        
    Visualization Philosophy:
    - Progressive disclosure: Key metrics first, details on demand
    - Interactive charts for exploration
    - Multiple perspectives on performance
    - Export capabilities for further analysis
    - Mobile-responsive design considerations
    
    Educational Elements:
    - Contextual explanations for metrics
    - Visual aids for understanding concepts
    - Comparative analysis for learning
    - Professional presentation standards
    """
    
    # Strategy Information Header
    # ==========================
    st.markdown("## 📊 Analysis Results")
    
    # Display key strategy and data information prominently
    st.markdown(f"**Strategy**: {config['strategy']['name']}")
    st.markdown(f"**Symbol**: {config['data']['symbol']}")
    st.markdown(f"**Period**: {config['data']['start_date']} to {config['data']['end_date']}")
    st.markdown("---")
    
    # Key Performance Metrics Section
    # ==============================
    st.markdown("### 🎯 Key Performance Metrics")
    # Display high-level metrics in card format for quick assessment
    display_key_metrics(strategy_metrics)
    st.markdown("---")
    
    # Interactive Charts Section
    # =========================
    # Split screen layout for efficient space utilization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Equity Curve")
        
        # Prepare benchmark data for comparison if available
        benchmark_equity = None
        if benchmark_data is not None and not benchmark_data.empty:
            # Convert benchmark to same scale as strategy for comparison
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            benchmark_equity = pd.DataFrame({
                'equity': (1 + benchmark_returns).cumprod() * 100000
            }, index=benchmark_returns.index)
        
        # Create interactive equity curve with benchmark overlay
        equity_chart = create_equity_curve_chart(
            equity_curve, 
            benchmark_equity, 
            config['trading']['benchmark']
        )
        st.plotly_chart(equity_chart, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Drawdown Analysis")
        
        # Display drawdown chart for risk assessment
        # Drawdowns show peak-to-trough declines in portfolio value
        drawdown_chart = create_drawdown_chart(equity_curve)
        st.plotly_chart(drawdown_chart, use_container_width=True)
    
    # Price Action and Trading Signals Visualization
    # =============================================
    st.markdown("### 📊 Price Action & Trading Signals")
    
    # Create comprehensive price chart with signal overlays
    # This helps users understand when and why trades were executed
    signals_chart = create_price_and_signals_chart(
        price_data, 
        signals, 
        f"{config['data']['symbol']} - {config['strategy']['name']}"
    )
    st.plotly_chart(signals_chart, use_container_width=True)
    
    # Detailed Performance Analysis Section
    # ===================================
    st.markdown("### 📈 Detailed Performance Analysis")
    
    # Show comprehensive metrics in organized format
    display_detailed_metrics(strategy_metrics)
    
    # Benchmark Comparison Section
    # ===========================
    if benchmark_metrics:
        # Display side-by-side comparison with benchmark
        display_benchmark_comparison(
            strategy_metrics, 
            benchmark_metrics, 
            config['trading']['benchmark']
        )
    
    # Trade Analysis Section
    # =====================
    if not trades.empty:
        # Analyze trade distribution and patterns
        display_trade_distribution(trades)
    
    # Additional Advanced Visualizations
    # =================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Performance Overview")
        
        # Radar chart showing multiple performance dimensions
        radar_chart = create_metrics_summary_chart(strategy_metrics)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Monthly Returns")
        
        # Monthly returns heatmap for seasonality analysis
        if not equity_curve.empty:
            returns = equity_curve['equity'].pct_change().dropna()
            
            # Only show monthly analysis with sufficient data
            if len(returns) > 30:
                monthly_chart = create_monthly_returns_heatmap(returns)
                st.plotly_chart(monthly_chart, use_container_width=True)
            else:
                st.info("Insufficient data for monthly returns analysis (need > 30 days)")
    
    # Rolling Performance Metrics
    # ===========================
    st.markdown("### 📈 Rolling Performance Metrics")
    
    # Display rolling metrics only with sufficient data for statistical validity
    if len(equity_curve) > 252:  # Approximately one year of trading days
        rolling_chart = create_rolling_metrics_chart(equity_curve, window=60)
        st.plotly_chart(rolling_chart, use_container_width=True)
    else:
        st.info("Insufficient data for rolling metrics analysis (need > 252 days)")
    
    # Returns Distribution Analysis
    # ============================
    st.markdown("### 📊 Returns Distribution Analysis")
    
    # Analyze return distribution characteristics
    returns = equity_curve['equity'].pct_change().dropna()
    if len(returns) > 30:
        # Create distribution chart with statistical overlays
        dist_chart = create_returns_distribution_chart(returns)
        st.plotly_chart(dist_chart, use_container_width=True)
    else:
        st.info("Insufficient data for returns distribution analysis")
    
    # Export and Download Section
    # ===========================
    st.markdown("### 💾 Download Results")
    
    # Provide multiple export formats for further analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export performance metrics as CSV
        if st.button("📊 Download Metrics CSV"):
            # Convert metrics dictionary to DataFrame for export
            metrics_df = pd.DataFrame([strategy_metrics]).T
            metrics_df.columns = ['Value']
            csv = metrics_df.to_csv()
            
            # Create download button with appropriate filename
            st.download_button(
                label="Download Metrics",
                data=csv,
                file_name=f"{config['data']['symbol']}_metrics.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export trade records as CSV
        if st.button("💼 Download Trades CSV") and not trades.empty:
            csv = trades.to_csv()
            st.download_button(
                label="Download Trades",
                data=csv,
                file_name=f"{config['data']['symbol']}_trades.csv",
                mime="text/csv"
            )
    
    with col3:
        # Export equity curve data as CSV
        if st.button("📈 Download Equity CSV"):
            csv = equity_curve.to_csv()
            st.download_button(
                label="Download Equity Curve",
                data=csv,
                file_name=f"{config['data']['symbol']}_equity.csv",
                mime="text/csv"
            )


# Application Entry Point
# =======================
if __name__ == "__main__":
    # Direct execution support for standalone testing and development
    main() 