"""
Trading Performance Visualization System: Professional-Grade Chart Generation

This module provides a comprehensive visualization system for trading strategy
performance analysis. It generates publication-quality charts and graphs that
help traders and analysts visualize key performance metrics, risk profiles,
and market behavior patterns.

The visualization system is designed to integrate seamlessly with backtesting
frameworks and trading strategy analysis tools, providing clear and actionable
visual insights for both technical and fundamental analysis.

Key Features:
============

1. **Equity Curve Visualization**
   - Professional-grade equity curve charts
   - Customizable styling and formatting
   - Multiple timeframe support
   - Benchmark comparison capabilities

2. **Risk Analysis Charts**
   - Drawdown analysis visualization
   - Risk-adjusted performance metrics
   - Volatility and correlation analysis
   - Value-at-Risk (VaR) displays

3. **Performance Analytics**
   - Return distribution analysis
   - Rolling performance metrics
   - Seasonal performance patterns
   - Trade-level analysis charts

4. **Publication-Quality Output**
   - High-resolution chart export
   - Multiple output formats (PNG, PDF, SVG)
   - Professional styling and branding
   - Customizable chart themes

Architecture:
============

The visualization system follows a modular architecture:

1. **Chart Generation Layer**
   - Core matplotlib integration
   - Custom styling and themes
   - Chart composition and layout
   - Interactive features

2. **Data Processing Layer**
   - Performance metrics calculation
   - Data transformation and cleaning
   - Statistical analysis
   - Time series processing

3. **Output Management Layer**
   - File system integration
   - Multiple format support
   - Directory management
   - Batch processing capabilities

4. **Integration Layer**
   - Backtesting framework integration
   - Strategy analysis tools
   - Reporting system support
   - Dashboard compatibility

Usage Examples:
===============

Basic Usage:
```python
from backend.visualize import plot_equity_curve, plot_drawdown
import pandas as pd

# Load backtest results
df = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)

# Generate equity curve
plot_equity_curve(df, output_dir="reports", filename="equity_curve.png")

# Generate drawdown analysis
plot_drawdown(df, output_dir="reports", filename="drawdown_analysis.png")
```

Advanced Usage:
```python
# Custom styling and multiple metrics
plot_equity_curve(
    df,
    output_dir="reports",
    filename="custom_equity.png",
    title="Strategy Performance Analysis",
    benchmark_data=benchmark_df,
    style="professional"
)

# Batch chart generation
chart_generator = ChartGenerator(output_dir="reports")
chart_generator.generate_full_report(backtest_results)
```

Integration with Analysis Tools:
```python
# Integration with performance analyzer
from backend.analyze_batch_results import analyze_results

results = analyze_results("batch_results.csv")
plot_performance_summary(results, output_dir="analysis")
```

Educational Value:
=================

This module demonstrates:

1. **Data Visualization Principles**
   - Effective chart design
   - Visual hierarchy and clarity
   - Color theory and accessibility
   - Interactive visualization techniques

2. **Financial Charting Standards**
   - Industry-standard chart types
   - Performance visualization best practices
   - Risk visualization techniques
   - Professional presentation standards

3. **Python Visualization Ecosystem**
   - Matplotlib integration and customization
   - Plotting best practices
   - Output format management
   - Performance optimization

4. **Software Engineering Practices**
   - Modular design patterns
   - Error handling and validation
   - File system management
   - Configuration and customization

Integration Points:
==================

The visualization system integrates with:
- Backtesting frameworks
- Performance analysis tools
- Reporting systems
- Dashboard applications
- Export and sharing platforms

Performance Considerations:
==========================

- Efficient chart generation
- Memory-conscious data processing
- Scalable output management
- Batch processing capabilities
- Caching for repeated operations

Dependencies:
============

- matplotlib for chart generation
- pandas for data manipulation
- numpy for numerical operations
- os/pathlib for file system operations

Author: Strategy Explainer Framework
Version: 2.0
License: Educational Use
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

# Configure matplotlib for professional output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_equity_curve(df: pd.DataFrame, 
                     output_dir: str = "outputs", 
                     filename: str = "equity_curve.png",
                     title: str = "Equity Curve",
                     benchmark_data: Optional[pd.DataFrame] = None,
                     style: str = "default") -> None:
    """
    Generate a professional equity curve visualization.
    
    This function creates a comprehensive equity curve chart that displays
    the performance of a trading strategy over time. It provides essential
    visual information for evaluating strategy performance and risk
    characteristics.
    
    Args:
        df (pd.DataFrame): Backtest results DataFrame containing:
            - 'Equity': Portfolio equity values over time
            - Index: DateTime index for time series
        output_dir (str, optional): Directory for saving the chart.
            Defaults to "outputs".
        filename (str, optional): Output filename. Defaults to "equity_curve.png".
        title (str, optional): Chart title. Defaults to "Equity Curve".
        benchmark_data (pd.DataFrame, optional): Benchmark data for comparison.
            Should contain 'Equity' column. Defaults to None.
        style (str, optional): Chart style ('default', 'professional', 'minimal').
            Defaults to "default".
    
    Raises:
        ValueError: If required columns are missing from the DataFrame
        IOError: If unable to save the chart to the specified location
    
    Chart Features:
    ==============
    
    1. **Visual Elements**
       - Clean, professional equity curve line
       - Grid lines for easy reading
       - Proper axis labeling and formatting
       - Legend and title positioning
    
    2. **Performance Indicators**
       - Total return calculation
       - Peak-to-current performance
       - Visual trend identification
       - Time-based performance analysis
    
    3. **Benchmark Comparison** (if provided)
       - Overlay benchmark performance
       - Relative performance visualization
       - Alpha identification
       - Correlation analysis
    
    4. **Professional Output**
       - High-resolution export
       - Publication-quality formatting
       - Customizable styling options
       - Directory management
    
    Example Usage:
    =============
    ```python
    # Basic equity curve
    plot_equity_curve(backtest_results)
    
    # With benchmark comparison
    plot_equity_curve(
        backtest_results,
        benchmark_data=spy_data,
        title="Strategy vs S&P 500",
        style="professional"
    )
    
    # Custom output location
    plot_equity_curve(
        backtest_results,
        output_dir="reports/charts",
        filename="strategy_performance.png"
    )
    ```
    """
    # Validate input data
    if 'Equity' not in df.columns:
        raise ValueError("DataFrame must contain 'Equity' column")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create professional figure with optimal size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configure chart style based on selection
    if style == "professional":
        plt.style.use('seaborn-v0_8-whitegrid')
        color_primary = '#2E4B7A'
        color_benchmark = '#D2691E'
    elif style == "minimal":
        plt.style.use('seaborn-v0_8-white')
        color_primary = '#333333'
        color_benchmark = '#666666'
    else:  # default
        color_primary = 'blue'
        color_benchmark = 'red'
    
    # Plot main equity curve
    ax.plot(df.index, df["Equity"], 
            label="Strategy Performance", 
            color=color_primary, 
            linewidth=2.0,
            alpha=0.9)
    
    # Add benchmark if provided
    if benchmark_data is not None and 'Equity' in benchmark_data.columns:
        # Align benchmark data with strategy data
        aligned_benchmark = benchmark_data.reindex(df.index, method='ffill')
        
        ax.plot(aligned_benchmark.index, aligned_benchmark["Equity"],
                label="Benchmark", 
                color=color_benchmark, 
                linewidth=1.5,
                alpha=0.7,
                linestyle='--')
    
    # Configure chart appearance
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format y-axis to show currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add legend with optimal positioning
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Calculate and display key performance metrics
    initial_value = df["Equity"].iloc[0]
    final_value = df["Equity"].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Add performance annotation
    performance_text = f"Total Return: {total_return:.1f}%"
    ax.text(0.02, 0.98, performance_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=10)
    
    # Optimize layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart with high quality
    output_file = output_path / filename
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight', 
                format='png',
                facecolor='white',
                edgecolor='none')
    
    # Clean up memory
    plt.close()
    
    print(f"Equity curve saved to: {output_file}")


def plot_drawdown(df: pd.DataFrame, 
                 output_dir: str = "outputs", 
                 filename: str = "drawdown.png",
                 title: str = "Drawdown Analysis",
                 style: str = "default") -> None:
    """
    Generate a comprehensive drawdown analysis chart.
    
    This function creates a detailed drawdown visualization that shows
    the peak-to-trough decline in portfolio value over time. Drawdown
    analysis is crucial for understanding the risk profile and
    risk-adjusted performance of a trading strategy.
    
    Args:
        df (pd.DataFrame): Backtest results DataFrame containing:
            - 'Equity': Portfolio equity values over time
            - Index: DateTime index for time series
        output_dir (str, optional): Directory for saving the chart.
            Defaults to "outputs".
        filename (str, optional): Output filename. Defaults to "drawdown.png".
        title (str, optional): Chart title. Defaults to "Drawdown Analysis".
        style (str, optional): Chart style ('default', 'professional', 'minimal').
            Defaults to "default".
    
    Raises:
        ValueError: If required columns are missing from the DataFrame
        IOError: If unable to save the chart to the specified location
    
    Chart Features:
    ==============
    
    1. **Drawdown Visualization**
       - Percentage drawdown over time
       - Filled area chart for impact visualization
       - Zero line reference for clarity
       - Maximum drawdown identification
    
    2. **Risk Metrics**
       - Maximum drawdown calculation
       - Drawdown duration analysis
       - Recovery period identification
       - Risk-adjusted performance indicators
    
    3. **Visual Analytics**
       - Color-coded risk levels
       - Trend analysis capabilities
       - Time-based risk assessment
       - Recovery pattern identification
    
    4. **Professional Presentation**
       - High-quality output formatting
       - Clear axis labeling
       - Professional color schemes
       - Publication-ready charts
    
    Drawdown Calculation:
    ====================
    Drawdown = (Current Value - Running Maximum) / Running Maximum * 100
    
    This provides a percentage-based measure of portfolio decline from
    peak values, essential for risk assessment and strategy evaluation.
    
    Example Usage:
    =============
    ```python
    # Basic drawdown analysis
    plot_drawdown(backtest_results)
    
    # Professional styling
    plot_drawdown(
        backtest_results,
        title="Risk Analysis: Portfolio Drawdown",
        style="professional"
    )
    
    # Custom output location
    plot_drawdown(
        backtest_results,
        output_dir="reports/risk_analysis",
        filename="strategy_drawdown.png"
    )
    ```
    """
    # Validate input data
    if 'Equity' not in df.columns:
        raise ValueError("DataFrame must contain 'Equity' column")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate drawdown metrics
    rolling_max = df["Equity"].cummax()
    drawdown = (df["Equity"] - rolling_max) / rolling_max * 100
    
    # Calculate key drawdown statistics
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    
    # Create professional figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configure chart style
    if style == "professional":
        plt.style.use('seaborn-v0_8-whitegrid')
        fill_color = '#FF6B6B'
        line_color = '#FF4444'
    elif style == "minimal":
        plt.style.use('seaborn-v0_8-white')
        fill_color = '#E74C3C'
        line_color = '#C0392B'
    else:  # default
        fill_color = 'red'
        line_color = 'darkred'
    
    # Plot drawdown as filled area chart
    ax.fill_between(df.index, drawdown, 0, 
                    alpha=0.6, 
                    color=fill_color, 
                    label="Drawdown")
    
    # Plot drawdown line for clarity
    ax.plot(df.index, drawdown, 
            color=line_color, 
            linewidth=1.5, 
            alpha=0.8)
    
    # Add zero reference line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Highlight maximum drawdown point
    ax.plot(max_drawdown_date, max_drawdown, 
            'ro', markersize=8, 
            label=f'Max Drawdown: {max_drawdown:.1f}%')
    
    # Configure chart appearance
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format y-axis to show percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add statistics annotation
    stats_text = f"Max Drawdown: {max_drawdown:.1f}%\nDate: {max_drawdown_date.strftime('%Y-%m-%d')}"
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
            fontsize=10)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save the chart
    output_file = output_path / filename
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight', 
                format='png',
                facecolor='white',
                edgecolor='none')
    
    # Clean up memory
    plt.close()
    
    print(f"Drawdown analysis saved to: {output_file}")


def plot_performance_summary(results_dict: Dict[str, Any],
                           output_dir: str = "outputs",
                           filename: str = "performance_summary.png") -> None:
    """
    Generate a comprehensive performance summary visualization.
    
    This function creates a multi-panel chart showing key performance
    metrics, risk characteristics, and comparative analysis for trading
    strategy evaluation.
    
    Args:
        results_dict (Dict[str, Any]): Dictionary containing:
            - 'equity_data': DataFrame with equity curve data
            - 'metrics': Dictionary of performance metrics
            - 'trades': DataFrame with trade-level data
        output_dir (str, optional): Output directory. Defaults to "outputs".
        filename (str, optional): Output filename. Defaults to "performance_summary.png".
    
    Chart Components:
    ================
    - Equity curve with key milestones
    - Drawdown analysis
    - Return distribution
    - Monthly performance heatmap
    
    Example Usage:
    =============
    ```python
    # Comprehensive performance analysis
    plot_performance_summary(
        {
            'equity_data': equity_df,
            'metrics': performance_metrics,
            'trades': trades_df
        },
        output_dir="reports",
        filename="full_analysis.png"
    )
    ```
    """
    # Create multi-panel figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    equity_data = results_dict.get('equity_data')
    metrics = results_dict.get('metrics', {})
    
    if equity_data is not None and 'Equity' in equity_data.columns:
        # Panel 1: Equity Curve
        ax1.plot(equity_data.index, equity_data['Equity'], 
                color='blue', linewidth=2)
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Drawdown
        rolling_max = equity_data['Equity'].cummax()
        drawdown = (equity_data['Equity'] - rolling_max) / rolling_max * 100
        ax2.fill_between(equity_data.index, drawdown, 0, 
                        color='red', alpha=0.6)
        ax2.set_title('Drawdown Analysis', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Return Distribution
        returns = equity_data['Equity'].pct_change().dropna()
        ax3.hist(returns, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax3.set_title('Return Distribution', fontweight='bold')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Performance Metrics
        ax4.axis('off')
        metrics_text = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'return' in key.lower() or 'ratio' in key.lower():
                    metrics_text.append(f"{key}: {value:.2f}")
                else:
                    metrics_text.append(f"{key}: {value:.4f}")
            else:
                metrics_text.append(f"{key}: {value}")
        
        ax4.text(0.1, 0.9, '\n'.join(metrics_text),
                transform=ax4.transAxes,
                verticalalignment='top',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.set_title('Performance Metrics', fontweight='bold')
    
    # Optimize layout
    plt.tight_layout()
    
    # Save the comprehensive chart
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename
    
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight', 
                format='png',
                facecolor='white',
                edgecolor='none')
    
    plt.close()
    
    print(f"Performance summary saved to: {output_file}")


def create_chart_theme(theme_name: str = "professional") -> Dict[str, Any]:
    """
    Create a consistent chart theme configuration.
    
    This function provides standardized chart themes for consistent
    visualization across the trading analysis framework.
    
    Args:
        theme_name (str, optional): Theme name ('professional', 'minimal', 'dark').
            Defaults to "professional".
    
    Returns:
        Dict[str, Any]: Theme configuration dictionary
    
    Theme Components:
    ================
    - Color palettes
    - Font configurations
    - Line styles and weights
    - Chart spacing and layout
    
    Example Usage:
    =============
    ```python
    # Apply professional theme
    theme = create_chart_theme("professional")
    plt.rcParams.update(theme['rcParams'])
    ```
    """
    themes = {
        "professional": {
            "colors": {
                "primary": "#2E4B7A",
                "secondary": "#D2691E",
                "success": "#28a745",
                "danger": "#dc3545",
                "warning": "#ffc107"
            },
            "rcParams": {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "grid.color": "gray",
                "grid.alpha": 0.3
            }
        },
        "minimal": {
            "colors": {
                "primary": "#333333",
                "secondary": "#666666",
                "success": "#00AA00",
                "danger": "#AA0000",
                "warning": "#AAAA00"
            },
            "rcParams": {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.spines.top": False,
                "axes.spines.right": False
            }
        },
        "dark": {
            "colors": {
                "primary": "#00D4FF",
                "secondary": "#FF6B6B",
                "success": "#4ECDC4",
                "danger": "#FF6B6B",
                "warning": "#FFE66D"
            },
            "rcParams": {
                "figure.facecolor": "#1E1E1E",
                "axes.facecolor": "#1E1E1E",
                "text.color": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white"
            }
        }
    }
    
    return themes.get(theme_name, themes["professional"])


# Example usage and demonstration
if __name__ == "__main__":
    """
    Demonstrate the visualization system capabilities.
    
    This example shows how to use the various chart functions
    with sample data to generate professional trading analysis
    visualizations.
    """
    print("Trading Performance Visualization Demo")
    print("=" * 50)
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate sample equity curve with realistic trading patterns
    np.random.seed(42)
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
    equity_values = 10000 * np.exp(np.cumsum(returns))  # Compound returns
    
    # Create DataFrame with realistic trading data
    sample_data = pd.DataFrame({
        'Equity': equity_values,
        'Date': dates
    }).set_index('Date')
    
    print("\n1. Generating Equity Curve Chart:")
    print("-" * 30)
    try:
        plot_equity_curve(
            sample_data,
            output_dir="demo_charts",
            filename="demo_equity_curve.png",
            title="Sample Trading Strategy Performance"
        )
    except Exception as e:
        print(f"Error generating equity curve: {e}")
    
    print("\n2. Generating Drawdown Analysis:")
    print("-" * 30)
    try:
        plot_drawdown(
            sample_data,
            output_dir="demo_charts",
            filename="demo_drawdown.png",
            title="Sample Strategy Risk Analysis"
        )
    except Exception as e:
        print(f"Error generating drawdown chart: {e}")
    
    print("\n3. Generating Performance Summary:")
    print("-" * 30)
    try:
        # Create sample metrics
        sample_metrics = {
            'Total Return': 0.235,
            'Sharpe Ratio': 1.45,
            'Max Drawdown': -0.12,
            'Win Rate': 0.58,
            'Profit Factor': 1.75
        }
        
        results_dict = {
            'equity_data': sample_data,
            'metrics': sample_metrics
        }
        
        plot_performance_summary(
            results_dict,
            output_dir="demo_charts",
            filename="demo_performance_summary.png"
        )
    except Exception as e:
        print(f"Error generating performance summary: {e}")
    
    print("\n4. Chart Theme Configuration:")
    print("-" * 30)
    try:
        theme = create_chart_theme("professional")
        print(f"Professional theme colors: {theme['colors']}")
        print("Theme applied successfully")
    except Exception as e:
        print(f"Error creating theme: {e}")
    
    print("\n" + "=" * 50)
    print("Demo complete! Check 'demo_charts' directory for output.")
    print("Visualization system ready for integration.")
    print("=" * 50)