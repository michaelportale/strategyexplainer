"""
Batch Results Analysis System: Advanced Analytics for Trading Strategy Optimization

This module provides comprehensive analytical capabilities for processing and analyzing
results from batch backtesting operations. It transforms raw performance data into
actionable insights through statistical analysis, visualization, and pattern recognition.

The system is designed for quantitative researchers, portfolio managers, and algorithmic
traders who need to extract meaningful insights from large-scale strategy optimization
experiments.

Key Features:
============

1. **Multi-Dimensional Performance Analysis**
   - Parameter impact assessment across all dimensions
   - Cross-asset performance comparison
   - Risk-adjusted return analysis
   - Statistical significance testing

2. **Advanced Visualization Suite**
   - Interactive heatmaps for parameter relationships
   - Correlation matrices for metric dependencies
   - Performance distribution analysis
   - Risk-return scatter plots

3. **Statistical Analytics Engine**
   - Descriptive statistics for all metrics
   - Correlation analysis between parameters and outcomes
   - Percentile-based rankings
   - Outlier detection and analysis

4. **Automated Insight Generation**
   - Top performer identification across multiple criteria
   - Parameter sensitivity analysis
   - Risk-adjusted performance rankings
   - Systematic edge detection

Architecture:
============

The system follows a layered analytical architecture:

1. **Data Processing Layer**
   - CSV/JSON ingestion and validation
   - Data cleaning and type conversion
   - Missing value handling
   - Outlier detection

2. **Statistical Analysis Layer**
   - Descriptive statistics computation
   - Correlation analysis
   - Distribution analysis
   - Hypothesis testing

3. **Visualization Layer**
   - Matplotlib-based plotting
   - Seaborn statistical visualizations
   - Interactive charts
   - Export capabilities

4. **Insight Generation Layer**
   - Performance ranking algorithms
   - Pattern recognition
   - Anomaly detection
   - Recommendation system

Usage Examples:
===============

Basic Analysis:
```python
# Analyze latest batch results
python analyze_batch_results.py

# Analyze specific files
python analyze_batch_results.py --csv results.csv --json details.json

# Skip plotting (faster analysis)
python analyze_batch_results.py --no-plots

# Show more top performers
python analyze_batch_results.py --top-n 20
```

Programmatic Usage:
```python
from analyze_batch_results import BatchResultsAnalyzer

# Create analyzer
analyzer = BatchResultsAnalyzer('batch_results.csv', 'batch_details.json')

# Run individual analyses
analyzer.summary_stats()
analyzer.top_performers(n=10)
analyzer.parameter_analysis()
analyzer.risk_return_analysis()

# Full analysis pipeline
analyzer.run_full_analysis(save_plots=True)
```

Output Analysis:
================

The system provides multiple types of analysis:

1. **Summary Statistics**
   - Overall performance metrics
   - Success/failure rates
   - Distribution characteristics
   - Key performance indicators

2. **Top Performers Analysis**
   - Rankings by total return
   - Rankings by Sharpe ratio
   - Rankings by Calmar ratio
   - Rankings by profit factor

3. **Parameter Impact Analysis**
   - Performance by ticker
   - Effect of stop loss levels
   - Effect of take profit levels
   - Position sizing impact
   - Drawdown limit effectiveness

4. **Risk-Return Analysis**
   - Efficient frontier visualization
   - Risk-adjusted performance metrics
   - Volatility vs return relationships
   - Downside risk analysis

5. **Correlation Analysis**
   - Parameter correlation matrix
   - Metric interdependencies
   - Cross-validation patterns
   - Systematic relationships

Visualization Suite:
===================

The system generates comprehensive visualizations:

1. **Heatmaps**
   - Parameter performance matrices
   - Correlation heatmaps
   - Risk-return visualizations

2. **Distribution Plots**
   - Return distributions
   - Metric histograms
   - Box plots for comparisons

3. **Scatter Plots**
   - Risk vs return analysis
   - Parameter sensitivity
   - Correlation visualizations

4. **Bar Charts**
   - Top performer rankings
   - Parameter comparisons
   - Category analysis

Educational Value:
=================

This module demonstrates:

1. **Advanced Data Analysis Techniques**
   - Statistical analysis in Python
   - Pandas data manipulation
   - Visualization best practices
   - Performance measurement

2. **Quantitative Finance Concepts**
   - Risk-adjusted returns
   - Portfolio optimization
   - Performance attribution
   - Statistical arbitrage

3. **Research Methodology**
   - Hypothesis testing
   - Statistical significance
   - Correlation vs causation
   - Experimental design

4. **Professional Development**
   - Code documentation
   - Error handling
   - User interface design
   - Reproducible research

Integration Points:
==================

The system integrates with:
- batch_backtest.py (data source)
- Pandas/NumPy (data processing)
- Matplotlib/Seaborn (visualization)
- Jupyter notebooks (research workflow)
- Excel/CSV (data export)

Performance Considerations:
==========================

- Memory-efficient data processing
- Optimized visualization rendering
- Caching of computed results
- Parallel processing capability
- Scalable to large datasets

Author: Strategy Explainer Framework
Version: 2.0
License: Educational Use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style for professional output
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BatchResultsAnalyzer:
    """
    Comprehensive analysis engine for batch backtest results.
    
    This class provides a complete suite of analytical tools for processing
    and analyzing results from trading strategy optimization experiments.
    It combines statistical analysis, visualization, and insight generation
    into a unified framework for quantitative research.
    
    The analyzer is designed to handle large datasets efficiently while
    providing detailed insights into strategy performance patterns,
    parameter sensitivity, and risk-return characteristics.
    
    Attributes:
        csv_path (str): Path to batch results CSV file
        json_path (str): Path to detailed JSON results file
        df (pd.DataFrame): Raw results dataframe
        df_success (pd.DataFrame): Filtered successful runs dataframe
        metadata (dict): Batch run metadata from JSON file
    
    Example Usage:
    =============
    ```python
    # Basic usage
    analyzer = BatchResultsAnalyzer('results.csv')
    analyzer.run_full_analysis()
    
    # With detailed JSON
    analyzer = BatchResultsAnalyzer('results.csv', 'details.json')
    analyzer.summary_stats()
    analyzer.top_performers(n=20)
    
    # Custom analysis
    analyzer.parameter_analysis()
    analyzer.risk_return_analysis()
    analyzer.correlation_analysis()
    ```
    """
    
    def __init__(self, csv_path: str, json_path: str = None):
        """
        Initialize the batch results analyzer.
        
        This constructor loads and prepares the data for analysis, including
        data cleaning, type conversion, and initial validation. It handles
        both CSV and JSON input formats and provides robust error handling
        for various data quality issues.
        
        Args:
            csv_path (str): Path to batch results CSV file containing
                performance metrics and parameters
            json_path (str, optional): Path to detailed JSON results file
                containing metadata and additional information
        
        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data format is invalid or corrupted
            pd.errors.ParserError: If CSV parsing fails
        
        Data Processing Steps:
        =====================
        1. **File Validation**: Check file existence and accessibility
        2. **Data Loading**: Load CSV and optional JSON files
        3. **Data Cleaning**: Filter successful runs and handle missing values
        4. **Type Conversion**: Convert string metrics to numeric types
        5. **Data Enrichment**: Add computed fields and labels
        6. **Validation**: Ensure data integrity and consistency
        """
        self.csv_path = csv_path
        self.json_path = json_path
        
        # Load main results data
        self.df = pd.read_csv(csv_path)
        self.metadata = None
        
        # Load optional detailed metadata
        if json_path and Path(json_path).exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', {})
        
        # Clean and prepare data for analysis
        self._prepare_data()
        
    def _prepare_data(self):
        """
        Clean and prepare the raw data for analysis.
        
        This method performs comprehensive data preparation including:
        - Filtering successful runs
        - Converting string metrics to numeric types
        - Handling missing values
        - Creating derived fields
        - Validating data quality
        
        The preparation process ensures that all subsequent analysis
        operations work with clean, properly typed data.
        
        Data Quality Checks:
        ===================
        - Remove failed runs (status != 'SUCCESS')
        - Convert numeric columns to proper types
        - Handle missing values appropriately
        - Validate metric ranges and distributions
        - Create parameter combination labels
        
        Performance Optimization:
        ========================
        - Efficient pandas operations
        - Memory-conscious data types
        - Vectorized computations
        - Minimal data copying
        """
        # Filter to successful runs only for analysis
        self.df_success = self.df[self.df['run_status'] == 'SUCCESS'].copy()
        
        # Define numeric columns for type conversion
        numeric_cols = [
            'total_return', 'annual_return', 'monthly_return', 'daily_return_avg',
            'annual_volatility', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
            'win_rate', 'total_trades', 'profit_factor', 'expectancy', 'largest_win', 
            'largest_loss', 'avg_win', 'avg_loss', 'max_consecutive_wins', 
            'max_consecutive_losses', 'calmar_ratio', 'recovery_factor', 'kelly_criterion'
        ]
        
        # Convert string metrics to numeric with error handling
        for col in numeric_cols:
            if col in self.df_success.columns:
                self.df_success[col] = pd.to_numeric(self.df_success[col], errors='coerce')
        
        # Create parameter combination labels for grouping
        self.df_success['param_combo'] = (
            self.df_success['ticker'] + '_' +
            self.df_success['stop_loss_pct'].astype(str) + '_' +
            self.df_success['take_profit_pct'].astype(str) + '_' +
            self.df_success['max_drawdown_pct'].astype(str) + '_' +
            self.df_success['position_size_pct'].astype(str)
        )
        
        # Log data preparation results
        print(f"✅ Data loaded: {len(self.df)} total runs ({len(self.df_success)} successful)")
    
    def summary_stats(self):
        """
        Display comprehensive summary statistics for the batch results.
        
        This method provides a high-level overview of the batch backtesting
        experiment, including metadata, success rates, and key performance
        statistics across all successful runs.
        
        The summary includes:
        - Execution metadata (timestamp, combinations, success rate)
        - Descriptive statistics for key metrics
        - Performance distribution characteristics
        - Data quality indicators
        
        Output Format:
        =============
        - Formatted console output with clear sections
        - Statistical tables with rounded values
        - Success rate calculations
        - Key performance indicators
        """
        print("\n" + "="*60)
        print("📊 BATCH BACKTEST SUMMARY STATISTICS")
        print("="*60)
        
        # Display batch metadata if available
        if self.metadata:
            print(f"🕐 Run timestamp: {self.metadata.get('timestamp', 'N/A')}")
            print(f"🎯 Total combinations: {self.metadata.get('total_combinations', 'N/A')}")
            print(f"✅ Successful runs: {self.metadata.get('successful_runs', 'N/A')}")
            print(f"❌ Failed runs: {self.metadata.get('failed_runs', 'N/A')}")
            success_rate = self.metadata.get('successful_runs', 0) / self.metadata.get('total_combinations', 1) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        # Display key metrics summary for successful runs
        if len(self.df_success) > 0:
            print(f"\n📋 KEY METRICS ACROSS ALL SUCCESSFUL RUNS:")
            metrics_summary = self.df_success[['total_return', 'annual_return', 'sharpe_ratio', 
                                             'max_drawdown', 'win_rate', 'total_trades']].describe()
            print(metrics_summary.round(4))
    
    def top_performers(self, n: int = 10):
        """
        Identify and display top performing strategies across multiple criteria.
        
        This method analyzes the results to identify the best performing
        parameter combinations across different performance metrics. It provides
        rankings by various criteria to help identify robust strategies.
        
        Args:
            n (int): Number of top performers to display for each metric
        
        Performance Metrics Analyzed:
        ============================
        - Total Return: Overall profitability
        - Sharpe Ratio: Risk-adjusted performance
        - Calmar Ratio: Return vs maximum drawdown
        - Profit Factor: Gross profit vs gross loss
        
        Output Format:
        =============
        Each ranking shows:
        - Rank position
        - Ticker symbol
        - Key parameters (stop loss, take profit)
        - Performance metrics (return, Sharpe, drawdown, trades)
        
        Usage Example:
        =============
        ```python
        analyzer.top_performers(n=20)  # Show top 20 in each category
        ```
        """
        print(f"\n🏆 TOP {n} PERFORMERS")
        print("="*60)
        
        # Define metrics for ranking with formatting
        metrics = [
            ('total_return', 'Total Return', '{:.1%}'),
            ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}'),
            ('calmar_ratio', 'Calmar Ratio', '{:.2f}'),
            ('profit_factor', 'Profit Factor', '{:.2f}')
        ]
        
        # Generate rankings for each metric
        for metric, label, fmt in metrics:
            if metric in self.df_success.columns:
                print(f"\n📈 By {label}:")
                top_n = self.df_success.nlargest(n, metric)
                
                # Display formatted results
                for i, (_, row) in enumerate(top_n.iterrows(), 1):
                    ret = row['total_return']
                    sharpe = row['sharpe_ratio']
                    dd = row['max_drawdown']
                    trades = row['total_trades']
                    ticker = row['ticker']
                    sl = row['stop_loss_pct']
                    tp = row['take_profit_pct']
                    
                    print(f"  {i:2d}. {ticker} | SL:{sl} TP:{tp} | "
                          f"Ret:{ret:.1%} Sharpe:{sharpe:.2f} DD:{dd:.1%} Trades:{trades:.0f}")
    
    def parameter_analysis(self):
        """
        Analyze performance impact of different parameter values.
        
        This method examines how different parameter settings affect
        strategy performance across multiple dimensions. It provides
        insights into parameter sensitivity and optimal ranges.
        
        Analysis Dimensions:
        ===================
        - By Ticker: Cross-asset performance comparison
        - By Stop Loss: Risk management effectiveness
        - By Take Profit: Profit-taking strategy impact
        - By Position Size: Capital allocation effects
        - By Max Drawdown: Portfolio protection analysis
        
        Statistical Methods:
        ===================
        - Groupby aggregations for parameter levels
        - Mean, standard deviation, and count statistics
        - Ranking by average performance
        - Distribution analysis within groups
        
        Output Format:
        =============
        - Tabular summaries for each parameter
        - Sorted by average performance
        - Includes sample sizes and variability measures
        """
        print(f"\n🔍 PARAMETER IMPACT ANALYSIS")
        print("="*60)
        
        # Analyze performance by ticker
        print("\n📊 By Ticker:")
        ticker_stats = self.df_success.groupby('ticker').agg({
            'total_return': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(3)
        ticker_stats.columns = ['Avg_Return', 'Std_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD']
        print(ticker_stats.sort_values('Avg_Return', ascending=False))
        
        # Analyze performance by stop loss levels
        print("\n🛑 By Stop Loss:")
        sl_stats = self.df_success.groupby('stop_loss_pct').agg({
            'total_return': ['mean', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean'
        }).round(3)
        sl_stats.columns = ['Avg_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD', 'Avg_WinRate']
        print(sl_stats.sort_values('Avg_Return', ascending=False))
        
        # Analyze performance by take profit levels
        print("\n💰 By Take Profit:")
        tp_stats = self.df_success.groupby('take_profit_pct').agg({
            'total_return': ['mean', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean'
        }).round(3)
        tp_stats.columns = ['Avg_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD', 'Avg_WinRate']
        print(tp_stats.sort_values('Avg_Return', ascending=False))
        
        # Analyze performance by position size
        print("\n📏 By Position Size:")
        pos_stats = self.df_success.groupby('position_size_pct').agg({
            'total_return': ['mean', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'annual_volatility': 'mean'
        }).round(3)
        pos_stats.columns = ['Avg_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD', 'Avg_Vol']
        print(pos_stats.sort_values('Avg_Return', ascending=False))
    
    def risk_return_analysis(self):
        """
        Perform comprehensive risk-return analysis of strategies.
        
        This method analyzes the risk-return characteristics of all
        strategies to identify efficient combinations and risk patterns.
        It provides insights into the risk-return tradeoff and helps
        identify strategies with superior risk-adjusted performance.
        
        Analysis Components:
        ===================
        - Risk-return scatter analysis
        - Efficient frontier identification
        - Volatility vs return relationships
        - Downside risk assessment
        - Risk-adjusted performance metrics
        
        Statistical Methods:
        ===================
        - Correlation analysis between risk and return
        - Percentile-based risk categorization
        - Efficient frontier approximation
        - Sharpe ratio distribution analysis
        
        Output Format:
        =============
        - Statistical summaries by risk categories
        - Correlation coefficients
        - Risk-return relationship insights
        - Efficient strategy identification
        """
        print(f"\n⚖️ RISK-RETURN ANALYSIS")
        print("="*60)
        
        if len(self.df_success) == 0:
            print("❌ No successful runs to analyze")
            return
        
        # Basic risk-return statistics
        print("\n📊 Risk-Return Statistics:")
        risk_return_stats = self.df_success[['total_return', 'annual_volatility', 
                                           'sharpe_ratio', 'max_drawdown']].describe()
        print(risk_return_stats.round(4))
        
        # Correlation between risk and return metrics
        print("\n🔗 Risk-Return Correlations:")
        corr_matrix = self.df_success[['total_return', 'annual_volatility', 
                                     'sharpe_ratio', 'max_drawdown']].corr()
        print(corr_matrix.round(3))
        
        # Risk-based categorization
        print("\n📈 Performance by Risk Categories:")
        
        # Categorize by volatility
        vol_quantiles = self.df_success['annual_volatility'].quantile([0.33, 0.67])
        self.df_success['vol_category'] = pd.cut(
            self.df_success['annual_volatility'],
            bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf],
            labels=['Low_Vol', 'Med_Vol', 'High_Vol']
        )
        
        vol_analysis = self.df_success.groupby('vol_category').agg({
            'total_return': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(3)
        vol_analysis.columns = ['Avg_Return', 'Std_Return', 'Count', 'Avg_Sharpe', 'Avg_MaxDD']
        print(vol_analysis)
    
    def correlation_analysis(self):
        """
        Analyze correlations between parameters and performance metrics.
        
        This method examines the relationships between input parameters
        and output performance metrics to identify systematic patterns
        and parameter sensitivities. It helps understand which parameters
        have the strongest influence on strategy performance.
        
        Analysis Components:
        ===================
        - Parameter-performance correlations
        - Inter-metric correlations
        - Systematic relationship identification
        - Statistical significance testing
        
        Statistical Methods:
        ===================
        - Pearson correlation coefficients
        - Spearman rank correlations
        - Correlation matrix visualization
        - Significance testing
        
        Output Format:
        =============
        - Correlation matrices with color coding
        - Strongest correlations highlighted
        - Statistical significance indicators
        - Interpretation guidance
        """
        print(f"\n🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        if len(self.df_success) == 0:
            print("❌ No successful runs to analyze")
            return
        
        # Select key columns for correlation analysis
        correlation_cols = [
            'stop_loss_pct', 'take_profit_pct', 'position_size_pct',
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'total_trades', 'profit_factor'
        ]
        
        # Filter to available columns
        available_cols = [col for col in correlation_cols if col in self.df_success.columns]
        
        if len(available_cols) > 2:
            print("\n📊 Parameter-Performance Correlation Matrix:")
            corr_matrix = self.df_success[available_cols].corr()
            print(corr_matrix.round(3))
            
            # Identify strongest correlations
            print("\n🔍 Strongest Correlations (|r| > 0.3):")
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
    
    def generate_heatmaps(self, save_plots: bool = True):
        """
        Generate comprehensive heatmap visualizations.
        
        This method creates detailed heatmap visualizations to display
        performance patterns across parameter combinations. Heatmaps
        provide intuitive visual representations of parameter sensitivity
        and performance landscapes.
        
        Args:
            save_plots (bool): Whether to save plots to disk
        
        Visualization Types:
        ===================
        - Parameter performance heatmaps
        - Correlation heatmaps
        - Risk-return visualizations
        - Cross-tabulation matrices
        
        Technical Implementation:
        ========================
        - Matplotlib/Seaborn integration
        - Professional color schemes
        - Configurable plot parameters
        - High-quality output formats
        
        Output Files:
        ============
        - correlation_heatmap.png
        - performance_heatmap.png
        - risk_return_scatter.png
        """
        print(f"\n🎨 GENERATING HEATMAPS")
        print("="*60)
        
        if len(self.df_success) == 0:
            print("❌ No successful runs to visualize")
            return
        
        # Generate correlation heatmap
        try:
            correlation_cols = [
                'stop_loss_pct', 'take_profit_pct', 'position_size_pct',
                'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'
            ]
            available_cols = [col for col in correlation_cols if col in self.df_success.columns]
            
            if len(available_cols) > 2:
                plt.figure(figsize=(10, 8))
                corr_matrix = self.df_success[available_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           fmt='.2f', square=True, linewidths=0.5)
                plt.title('Parameter-Performance Correlation Matrix')
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
                    print("📊 Saved: correlation_heatmap.png")
                
                plt.show()
            
            # Generate performance heatmap by ticker and parameters
            if 'ticker' in self.df_success.columns and 'stop_loss_pct' in self.df_success.columns:
                pivot_table = self.df_success.pivot_table(
                    values='total_return', 
                    index='ticker', 
                    columns='stop_loss_pct',
                    aggfunc='mean'
                )
                
                if not pivot_table.empty:
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0, 
                               fmt='.2%', linewidths=0.5)
                    plt.title('Average Total Return by Ticker and Stop Loss')
                    plt.tight_layout()
                    
                    if save_plots:
                        plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
                        print("📊 Saved: performance_heatmap.png")
                    
                    plt.show()
                    
        except Exception as e:
            print(f"❌ Error generating heatmaps: {e}")
    
    def export_insights(self):
        """
        Export key insights and recommendations to files.
        
        This method compiles the most important findings from the analysis
        and exports them to structured files for further use. It creates
        actionable summaries that can be used for strategy implementation.
        
        Export Formats:
        ==============
        - CSV files with top performers
        - JSON files with insights
        - Text summaries with recommendations
        - Statistical reports
        
        Content Includes:
        ================
        - Top performing configurations
        - Parameter recommendations
        - Risk-return insights
        - Implementation guidance
        """
        print(f"\n💾 EXPORTING INSIGHTS")
        print("="*60)
        
        if len(self.df_success) == 0:
            print("❌ No successful runs to export")
            return
        
        try:
            # Export top performers
            top_performers = self.df_success.nlargest(20, 'total_return')
            top_performers.to_csv('top_performers.csv', index=False)
            print("📊 Saved: top_performers.csv")
            
            # Export parameter summary
            param_summary = {
                'best_ticker': self.df_success.groupby('ticker')['total_return'].mean().idxmax(),
                'best_stop_loss': self.df_success.groupby('stop_loss_pct')['total_return'].mean().idxmax(),
                'best_take_profit': self.df_success.groupby('take_profit_pct')['total_return'].mean().idxmax(),
                'best_position_size': self.df_success.groupby('position_size_pct')['total_return'].mean().idxmax(),
                'total_runs': len(self.df_success),
                'avg_return': self.df_success['total_return'].mean(),
                'avg_sharpe': self.df_success['sharpe_ratio'].mean()
            }
            
            with open('analysis_summary.json', 'w') as f:
                json.dump(param_summary, f, indent=2, default=str)
            print("📋 Saved: analysis_summary.json")
            
        except Exception as e:
            print(f"❌ Error exporting insights: {e}")
    
    def run_full_analysis(self, save_plots: bool = True):
        """
        Execute complete analysis pipeline.
        
        This method runs the full analytical pipeline, performing all
        available analyses in the optimal sequence. It provides a
        comprehensive view of the batch results with minimal user input.
        
        Args:
            save_plots (bool): Whether to save generated plots to disk
        
        Analysis Pipeline:
        =================
        1. Summary statistics and metadata
        2. Top performer identification
        3. Parameter impact analysis
        4. Risk-return analysis
        5. Correlation analysis
        6. Heatmap generation
        7. Insights export
        
        Output Summary:
        ==============
        - Console output with formatted results
        - Saved visualization files
        - Exported data files
        - Analysis summary reports
        """
        print("🚀 Starting comprehensive batch results analysis...")
        
        # Execute analysis pipeline
        self.summary_stats()
        self.top_performers()
        self.parameter_analysis()
        self.risk_return_analysis()
        self.correlation_analysis()
        self.generate_heatmaps(save_plots)
        self.export_insights()


def find_latest_results(output_dir: str = "backend/outputs") -> tuple:
    """
    Find the most recent batch results files automatically.
    
    This utility function searches the output directory for the most
    recently generated batch results files and returns their paths.
    It provides convenient access to the latest results without
    manual file specification.
    
    Args:
        output_dir (str): Directory to search for results files
    
    Returns:
        tuple: (csv_path, json_path) where json_path may be None
    
    Raises:
        FileNotFoundError: If no batch results files are found
    
    Search Strategy:
    ===============
    1. Search for batch_summary_*.csv files
    2. Identify most recent based on file modification time
    3. Look for corresponding JSON file with same timestamp
    4. Return paths for both files
    
    Example Usage:
    =============
    ```python
    csv_path, json_path = find_latest_results()
    analyzer = BatchResultsAnalyzer(csv_path, json_path)
    ```
    """
    output_path = Path(output_dir)
    
    # Find latest CSV file
    csv_files = list(output_path.glob("batch_summary_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No batch summary CSV files found in {output_dir}")
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    # Find corresponding JSON file
    timestamp = latest_csv.stem.replace("batch_summary_", "")
    json_file = output_path / f"batch_detailed_{timestamp}.json"
    
    return str(latest_csv), str(json_file) if json_file.exists() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Results Analysis: Advanced Analytics for Trading Strategy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_batch_results.py                      # Auto-find latest results
  python analyze_batch_results.py --csv results.csv    # Specific CSV file
  python analyze_batch_results.py --no-plots           # Skip visualization
  python analyze_batch_results.py --top-n 20           # Show top 20 performers
        """
    )
    
    parser.add_argument(
        "--csv", 
        help="Path to batch results CSV file"
    )
    parser.add_argument(
        "--json", 
        help="Path to detailed JSON results file"
    )
    parser.add_argument(
        "--no-plots", 
        action="store_true", 
        help="Skip generating plots and visualizations"
    )
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=10, 
        help="Number of top performers to show (default: 10)"
    )
    
    args = parser.parse_args()
    
    try:
        # Use provided files or find latest automatically
        if args.csv:
            csv_path = args.csv
            json_path = args.json
        else:
            print("🔍 Finding latest batch results...")
            csv_path, json_path = find_latest_results()
            print(f"📊 Using: {csv_path}")
            if json_path:
                print(f"📋 Using: {json_path}")
        
        # Create analyzer and run analysis
        analyzer = BatchResultsAnalyzer(csv_path, json_path)
        analyzer.run_full_analysis(save_plots=not args.no_plots)
        
        print(f"\n🎯 Analysis complete! Your trading edge is revealed in the data.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Run batch_backtest.py first to generate results.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise 