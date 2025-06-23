"""Main strategy analysis page for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))

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

# Import backend modules
try:
    from momentum_backtest import MomentumBacktest
    from visualize import create_performance_charts
    from data_loader import DataLoader
    from metrics import PerformanceMetrics
    from simulate import TradingSimulator
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.info("Please ensure all backend modules are properly installed and accessible.")


def main():
    """Main strategy analyzer page."""
    st.set_page_config(
        page_title="Strategy Analyzer - Strategy Explainer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Strategy Analyzer")
    st.markdown("---")
    
    # Sidebar configuration
    config = render_sidebar()
    display_strategy_info(config)
    
    # Main content area
    if render_run_button():
        run_analysis(config)
    else:
        display_welcome_screen()


def display_welcome_screen():
    """Display welcome screen with instructions."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
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
        
        # Sample strategy preview
        st.markdown("### 📈 Sample Analysis Preview:")
        
        # Create a simple sample chart
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Strategy': np.cumsum(np.random.randn(100) * 0.01) + 1,
            'Benchmark': np.cumsum(np.random.randn(100) * 0.008) + 1
        }, index=dates)
        
        st.line_chart(sample_data)
        
        st.info("💡 **Tip**: This is just a sample. Your actual analysis will show real market data and strategy performance!")


def run_analysis(config: Dict[str, Any]):
    """Run the complete strategy analysis."""
    try:
        with st.spinner("Loading data and running analysis..."):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load data
            status_text.text("📊 Loading market data...")
            progress_bar.progress(10)
            
            data_loader = DataLoader()
            price_data = data_loader.load_stock_data(
                symbol=config['data']['symbol'],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            
            if price_data.empty:
                st.error(f"No data available for {config['data']['symbol']}")
                return
            
            progress_bar.progress(25)
            
            # Step 2: Run backtest
            status_text.text("🔄 Running strategy backtest...")
            
            # Initialize strategy based on configuration
            strategy = initialize_strategy(config)
            signals = strategy.generate_signals(price_data)
            
            progress_bar.progress(50)
            
            # Step 3: Simulate trading
            status_text.text("💼 Simulating trades...")
            
            simulator = TradingSimulator(
                initial_capital=100000,
                commission=config['trading']['commission'],
                slippage=config['trading']['slippage']
            )
            
            equity_curve, trades = simulator.simulate_strategy(price_data, signals)
            
            progress_bar.progress(75)
            
            # Step 4: Calculate metrics
            status_text.text("📊 Calculating performance metrics...")
            
            metrics_calculator = PerformanceMetrics()
            strategy_metrics = metrics_calculator.calculate_all_metrics(
                equity_curve, trades, risk_free_rate=0.02
            )
            
            # Load benchmark data
            benchmark_data = data_loader.load_stock_data(
                symbol=config['trading']['benchmark'],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            
            benchmark_metrics = None
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data['close'].pct_change().dropna()
                benchmark_equity = (1 + benchmark_returns).cumprod() * 100000
                benchmark_metrics = metrics_calculator.calculate_all_metrics(
                    pd.DataFrame({'equity': benchmark_equity}), 
                    pd.DataFrame(), 
                    risk_free_rate=0.02
                )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        # Display results
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
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)


def initialize_strategy(config: Dict[str, Any]):
    """Initialize the appropriate strategy based on configuration."""
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
        # Default to momentum for now
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
    """Display comprehensive analysis results."""
    
    # Strategy Information
    st.markdown("## 📊 Analysis Results")
    st.markdown(f"**Strategy**: {config['strategy']['name']}")
    st.markdown(f"**Symbol**: {config['data']['symbol']}")
    st.markdown(f"**Period**: {config['data']['start_date']} to {config['data']['end_date']}")
    st.markdown("---")
    
    # Key Metrics Cards
    st.markdown("### 🎯 Key Performance Metrics")
    display_key_metrics(strategy_metrics)
    st.markdown("---")
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Equity Curve")
        benchmark_equity = None
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            benchmark_equity = pd.DataFrame({
                'equity': (1 + benchmark_returns).cumprod() * 100000
            }, index=benchmark_returns.index)
        
        equity_chart = create_equity_curve_chart(
            equity_curve, 
            benchmark_equity, 
            config['trading']['benchmark']
        )
        st.plotly_chart(equity_chart, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Drawdown Analysis")
        drawdown_chart = create_drawdown_chart(equity_curve)
        st.plotly_chart(drawdown_chart, use_container_width=True)
    
    # Price and Signals Chart
    st.markdown("### 📊 Price Action & Trading Signals")
    signals_chart = create_price_and_signals_chart(
        price_data, 
        signals, 
        f"{config['data']['symbol']} - {config['strategy']['name']}"
    )
    st.plotly_chart(signals_chart, use_container_width=True)
    
    # Detailed Metrics
    st.markdown("### 📈 Detailed Performance Analysis")
    display_detailed_metrics(strategy_metrics)
    
    # Benchmark Comparison
    if benchmark_metrics:
        display_benchmark_comparison(
            strategy_metrics, 
            benchmark_metrics, 
            config['trading']['benchmark']
        )
    
    # Trade Analysis
    if not trades.empty:
        display_trade_distribution(trades)
    
    # Additional Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Performance Overview")
        radar_chart = create_metrics_summary_chart(strategy_metrics)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Monthly Returns")
        if not equity_curve.empty:
            returns = equity_curve['equity'].pct_change().dropna()
            if len(returns) > 30:  # Only show if we have enough data
                monthly_chart = create_monthly_returns_heatmap(returns)
                st.plotly_chart(monthly_chart, use_container_width=True)
            else:
                st.info("Insufficient data for monthly returns analysis")
    
    # Rolling Metrics
    st.markdown("### 📈 Rolling Performance Metrics")
    if len(equity_curve) > 252:  # Only show if we have enough data
        rolling_chart = create_rolling_metrics_chart(equity_curve, window=60)
        st.plotly_chart(rolling_chart, use_container_width=True)
    else:
        st.info("Insufficient data for rolling metrics analysis (need > 252 days)")
    
    # Returns Distribution
    st.markdown("### 📊 Returns Distribution Analysis")
    returns = equity_curve['equity'].pct_change().dropna()
    if len(returns) > 30:
        dist_chart = create_returns_distribution_chart(returns)
        st.plotly_chart(dist_chart, use_container_width=True)
    else:
        st.info("Insufficient data for returns distribution analysis")
    
    # Download Results
    st.markdown("### 💾 Download Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Metrics CSV"):
            metrics_df = pd.DataFrame([strategy_metrics]).T
            metrics_df.columns = ['Value']
            csv = metrics_df.to_csv()
            st.download_button(
                label="Download Metrics",
                data=csv,
                file_name=f"{config['data']['symbol']}_metrics.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("💼 Download Trades CSV") and not trades.empty:
            csv = trades.to_csv()
            st.download_button(
                label="Download Trades",
                data=csv,
                file_name=f"{config['data']['symbol']}_trades.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📈 Download Equity CSV"):
            csv = equity_curve.to_csv()
            st.download_button(
                label="Download Equity Curve",
                data=csv,
                file_name=f"{config['data']['symbol']}_equity.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main() 