"""Financial Visualization and Interactive Charting Component for Strategy Explainer.

This module provides comprehensive charting and visualization capabilities for financial
data analysis and trading strategy evaluation. It creates interactive, professional-grade
charts using Plotly that help users understand strategy performance, market behavior,
and trading signal effectiveness.

CHARTING PHILOSOPHY:
===================
The charting system follows these design principles:

1. CLARITY: Charts should clearly communicate insights without overwhelming users
2. INTERACTIVITY: Users can zoom, pan, and explore data to gain deeper understanding  
3. CONTEXT: Multiple related views provide comprehensive analysis perspective
4. CONSISTENCY: Uniform styling and color schemes across all visualizations
5. PERFORMANCE: Efficient rendering even with large datasets

VISUALIZATION CATEGORIES:
========================

1. PRICE CHARTS: Core financial data visualization
   - Candlestick and OHLC price charts with volume
   - Technical indicators overlay (SMA, EMA, Bollinger Bands, etc.)
   - Trading signals marked with buy/sell indicators
   - Support and resistance level annotations

2. PERFORMANCE ANALYSIS: Strategy evaluation visualization
   - Equity curve progression over time
   - Benchmark comparison and relative performance
   - Rolling performance metrics (Sharpe ratio, volatility)
   - Drawdown analysis and underwater curves

3. RISK ANALYSIS: Risk management visualization
   - Value at Risk (VaR) distributions
   - Return distribution histograms and statistics
   - Correlation analysis heatmaps
   - Risk-adjusted return scatter plots

4. SIGNAL ANALYSIS: Trading signal effectiveness
   - Signal timing and frequency analysis
   - Signal performance attribution
   - Entry/exit timing optimization
   - Signal correlation and overlap analysis

TECHNICAL IMPLEMENTATION:
========================
The module leverages Plotly for interactive visualizations with these features:

- Responsive design that adapts to different screen sizes
- Real-time data updates and animation capabilities
- Export functionality for reports and presentations
- Mobile-friendly touch interactions
- Accessibility compliance for screen readers

COLOR SCHEMES AND STYLING:
==========================
Consistent color palette for financial data:
- Bullish/Positive: Green (#00FF88, #26A69A)
- Bearish/Negative: Red (#FF6B6B, #EF5350)  
- Neutral/Secondary: Blue (#99B3FF, #42A5F5)
- Background/Grid: Light gray (#F5F5F5, #E0E0E0)
- Text/Labels: Dark gray (#333333, #666666)

USAGE PATTERNS:
==============
```python
# Create equity curve chart with benchmark
fig = create_equity_curve_chart(
    equity_data=strategy_results,
    benchmark_data=benchmark_results,
    benchmark_name="S&P 500"
)
st.plotly_chart(fig, use_container_width=True)

# Create price chart with signals
fig = create_price_chart_with_signals(
    data=price_data,
    signals=trading_signals,
    indicators=['SMA_20', 'SMA_50'],
    show_volume=True
)
st.plotly_chart(fig, use_container_width=True)
```
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from config.config_manager import get_config_manager


# Load chart colors from configuration
config = get_config_manager()
chart_colors_config = config.get('ui.charts.colors', {})

# Define consistent color palette for financial visualizations
CHART_COLORS = {
    'bullish': chart_colors_config.get('success', '#26A69A'),      # Teal green for positive/bullish
    'bearish': chart_colors_config.get('danger', '#EF5350'),       # Coral red for negative/bearish
    'primary': chart_colors_config.get('primary', '#42A5F5'),      # Blue for primary data series
    'secondary': chart_colors_config.get('secondary', '#66BB6A'),   # Light green for secondary series
    'neutral': chart_colors_config.get('neutral_color', '#9E9E9E'), # Gray for neutral/benchmark data
    'background': chart_colors_config.get('background', '#FAFAFA'), # Light background
    'grid': chart_colors_config.get('grid', '#E0E0E0'),            # Grid lines
    'text': chart_colors_config.get('text_color', '#333333'),      # Primary text
    'accent': chart_colors_config.get('warning', '#FF9800')        # Orange for highlights/accents
}


def create_multi_strategy_comparison_chart(strategy_data: Dict[str, pd.DataFrame],
                                          benchmark_data: Optional[pd.DataFrame] = None,
                                          benchmark_name: str = "Benchmark") -> go.Figure:
    """Create comprehensive multi-strategy equity curve comparison.
    
    This function generates professional equity curve comparisons supporting
    multiple strategies, benchmark overlays, and performance attribution analysis.
    
    ENHANCED FEATURES:
    =================
    - Multiple strategy overlay with distinct styling
    - Relative performance analysis panel
    - Performance attribution and correlation analysis
    - Risk-adjusted comparison metrics
    - Interactive legend and data exploration
    
    VISUAL DESIGN:
    =============
    - Color-coded strategy lines with unique patterns
    - Performance milestone annotations
    - Drawdown period highlighting
    - Statistical summary overlays
    - Professional legend and hover information
    
    Args:
        strategy_data: Dictionary mapping strategy names to equity DataFrames
        benchmark_data: Optional benchmark equity data
        benchmark_name: Display name for benchmark
    """
    # Create subplot structure for multi-panel analysis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'Equity Curve Comparison',
            'Relative Performance vs Benchmark (%)',
            'Rolling Correlation Matrix'
        ),
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Color palette for multiple strategies
    strategy_colors = [
        CHART_COLORS['primary'], CHART_COLORS['bullish'], CHART_COLORS['accent'],
        CHART_COLORS['secondary'], '#9C27B0', '#795548', '#607D8B'
    ]
    
    line_styles = ['solid', 'dash', 'dot', 'dashdot']
    
    # Track performance metrics for summary
    performance_summary = {}
    
    # Add strategy equity curves
    for i, (strategy_name, equity_df) in enumerate(strategy_data.items()):
        color = strategy_colors[i % len(strategy_colors)]
        line_style = line_styles[i % len(line_styles)]
        
        # Main equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name=strategy_name,
                line=dict(color=color, width=2.5, dash=line_style),
                hovertemplate=f'<b>{strategy_name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Portfolio Value: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Calculate performance metrics
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        performance_summary[strategy_name] = {
            'total_return': total_return,
            'final_value': equity_df['equity'].iloc[-1],
            'color': color
        }
        
        # Relative performance vs benchmark (if available)
        if benchmark_data is not None and not benchmark_data.empty:
            # Align dates and calculate relative performance
            aligned_strategy = equity_df['equity'].reindex(benchmark_data.index, method='ffill')
            aligned_benchmark = benchmark_data['equity']
            
            # Calculate relative performance
            strategy_returns = aligned_strategy.pct_change().fillna(0)
            benchmark_returns = aligned_benchmark.pct_change().fillna(0)
            relative_perf = (strategy_returns - benchmark_returns).cumsum() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=relative_perf.index,
                    y=relative_perf.values,
                    mode='lines',
                    name=f'{strategy_name} vs {benchmark_name}',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate=f'<b>{strategy_name} Relative Performance</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Excess Return: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Add benchmark if provided
    if benchmark_data is not None and not benchmark_data.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data['equity'],
                mode='lines',
                name=benchmark_name,
                line=dict(color=CHART_COLORS['neutral'], width=2.5, dash='dash'),
                hovertemplate=f'<b>{benchmark_name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Portfolio Value: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add zero line for relative performance
        fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Rolling correlation analysis (if multiple strategies)
    if len(strategy_data) > 1:
        strategy_names = list(strategy_data.keys())
        returns_data = {}
        
        for name, equity_df in strategy_data.items():
            returns_data[name] = equity_df['equity'].pct_change().fillna(0)
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate rolling 30-day correlation for first two strategies
        if len(returns_df.columns) >= 2:
            rolling_corr = returns_df.iloc[:, 0].rolling(30).corr(returns_df.iloc[:, 1])
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    name=f'Correlation: {strategy_names[0]} vs {strategy_names[1]}',
                    line=dict(color=CHART_COLORS['accent'], width=2),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Correlation: %{y:.3f}<br><extra></extra>'
                ),
                row=3, col=1
            )
            
            # Add correlation reference lines
            fig.add_hline(y=0.5, row=3, col=1, line_dash="dot", line_color="green", opacity=0.3)
            fig.add_hline(y=-0.5, row=3, col=1, line_dash="dot", line_color="red", opacity=0.3)
    
    # Enhanced layout with professional styling
    fig.update_layout(
        title={
            'text': "Multi-Strategy Performance Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': CHART_COLORS['text']}
        },
        height=800,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    # Update axes styling
    for row in range(1, 4):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=CHART_COLORS['grid'],
            row=row, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=CHART_COLORS['grid'],
            row=row, col=1
        )
    
    # Axis labels
    fig.update_yaxes(title_text="Portfolio Value ($)", tickformat='$,.0f', row=1, col=1)
    fig.update_yaxes(title_text="Excess Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Add performance summary annotation
    if performance_summary:
        summary_lines = ["<b>Performance Summary:</b>"]
        for name, metrics in performance_summary.items():
            summary_lines.append(
                f"<span style='color:{metrics['color']}'>{name}: {metrics['total_return']:.1%}</span>"
            )
        
        fig.add_annotation(
            text="<br>".join(summary_lines),
            xref="paper", yref="paper",
            x=0.02, y=0.95,
            xanchor="left", yanchor="top",
            showarrow=False,
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11)
        )
    
    return fig


def create_equity_curve_chart(equity_data: pd.DataFrame, 
                            benchmark_data: Optional[pd.DataFrame] = None,
                            benchmark_name: str = "Benchmark") -> go.Figure:
    """Create an interactive equity curve chart with optional benchmark comparison.
    
    This function generates a professional equity curve visualization that shows
    portfolio value progression over time. It supports benchmark comparison,
    interactive features, and provides key performance insights.
    
    CHART FEATURES:
    ==============
    - Time-series line chart showing portfolio value growth
    - Optional benchmark overlay for relative performance analysis
    - Interactive zoom and pan capabilities
    - Hover tooltips with detailed information
    - Performance statistics annotations
    - Responsive design for different screen sizes
    
    VISUAL ELEMENTS:
    ===============
    - Primary equity line in distinctive color
    - Benchmark line in contrasting color (if provided)
    - Shaded areas for significant drawdown periods
    - Performance milestone markers
    - Grid lines for easy value reading
    
    Args:
        equity_data (pd.DataFrame): Strategy equity curve data
                                   Must contain 'equity' column with portfolio values
                                   Index should be datetime for proper time-series display
        benchmark_data (Optional[pd.DataFrame]): Benchmark comparison data
                                                Contains 'equity' column for benchmark values
                                                Index must align with equity_data timeframe
        benchmark_name (str): Display name for benchmark series in legend and tooltips
                             Default: "Benchmark"
    
    Returns:
        go.Figure: Interactive Plotly figure object ready for display
                  Configured with professional styling and interactive features
                  
    Example:
        # Create equity curve with S&P 500 benchmark
        strategy_equity = pd.DataFrame({
            'equity': [10000, 10150, 10080, 10220, 10180],
        }, index=pd.date_range('2023-01-01', periods=5))
        
        benchmark_equity = pd.DataFrame({
            'equity': [10000, 10100, 10090, 10150, 10140],
        }, index=pd.date_range('2023-01-01', periods=5))
        
        fig = create_equity_curve_chart(strategy_equity, benchmark_equity, "S&P 500")
    """
    # Create figure with professional layout
    fig = go.Figure()
    
    # Add strategy equity curve line
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=equity_data['equity'],
        mode='lines',
        name='Strategy Portfolio',
        line=dict(color=CHART_COLORS['primary'], width=2.5),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Date: %{x}<br>' +
                     'Portfolio Value: $%{y:,.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add benchmark comparison if provided
    if benchmark_data is not None and not benchmark_data.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data['equity'],
            mode='lines',
            name=benchmark_name,
            line=dict(color=CHART_COLORS['neutral'], width=2, dash='dash'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Portfolio Value: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Configure professional chart layout
    fig.update_layout(
        title={
            'text': "Portfolio Equity Curve",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': CHART_COLORS['text']}
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=CHART_COLORS['grid']
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            tickformat='$,.0f',
            showline=True,
            linecolor=CHART_COLORS['grid']
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    return fig


def create_underwater_drawdown_chart(equity_data: pd.DataFrame,
                                   benchmark_data: Optional[pd.DataFrame] = None,
                                   benchmark_name: str = "Benchmark") -> go.Figure:
    """Create comprehensive underwater drawdown chart with enhanced analysis.
    
    This function creates a professional "underwater" chart showing when and how deeply
    the strategy fell from its peaks. This visualization is crucial for understanding
    risk exposure, recovery periods, and comparing risk profiles.
    
    ENHANCED FEATURES:
    =================
    - Color-coded drawdown severity zones
    - Benchmark comparison overlay
    - Recovery period annotations
    - Maximum drawdown highlighting
    - Statistical summary annotations
    
    RISK ZONES:
    ==========
    - Green: 0% to -5% (Low risk)
    - Yellow: -5% to -15% (Moderate risk)
    - Orange: -15% to -25% (High risk)
    - Red: Below -25% (Very high risk)
    
    Args:
        equity_data: Strategy equity curve data
        benchmark_data: Optional benchmark for risk comparison
        benchmark_name: Name for benchmark series
    """
    # Calculate strategy drawdown
    peak = equity_data['equity'].cummax()
    drawdown = (equity_data['equity'] - peak) / peak * 100
    
    # Calculate benchmark drawdown if provided
    bench_drawdown = None
    if benchmark_data is not None and not benchmark_data.empty:
        bench_peak = benchmark_data['equity'].cummax()
        bench_drawdown = (benchmark_data['equity'] - bench_peak) / bench_peak * 100
    
    fig = go.Figure()
    
    # Strategy underwater chart with enhanced styling
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=drawdown,
        mode='lines',
        fill='tonexty',
        name='Strategy Drawdown',
        line=dict(color=CHART_COLORS['bearish'], width=2.5),
        fillcolor='rgba(239, 83, 80, 0.3)',
        hovertemplate='<b>Strategy Drawdown</b><br>' +
                     'Date: %{x}<br>' +
                     'Drawdown: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Add benchmark drawdown if available
    if bench_drawdown is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_data.index,
            y=bench_drawdown,
            mode='lines',
            name=f'{benchmark_name} Drawdown',
            line=dict(color=CHART_COLORS['neutral'], width=2, dash='dash'),
            hovertemplate=f'<b>{benchmark_name} Drawdown</b><br>' +
                         'Date: %{x}<br>' +
                         'Drawdown: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Add risk zone reference lines with annotations
    risk_zones = [
        {'level': -5, 'color': 'yellow', 'label': 'Moderate Risk Zone'},
        {'level': -15, 'color': 'orange', 'label': 'High Risk Zone'},
        {'level': -25, 'color': 'red', 'label': 'Very High Risk Zone'}
    ]
    
    for zone in risk_zones:
        fig.add_hline(
            y=zone['level'], 
            line_dash="dot", 
            line_color=zone['color'], 
            opacity=0.6,
            annotation_text=f"{zone['level']}% - {zone['label']}",
            annotation_position="right"
        )
    
    # Highlight maximum drawdown point
    max_dd_value = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    fig.add_trace(go.Scatter(
        x=[max_dd_date],
        y=[max_dd_value],
        mode='markers',
        name=f'Max Drawdown ({max_dd_value:.2f}%)',
        marker=dict(
            symbol='circle',
            size=15,
            color='darkred',
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>Maximum Drawdown</b><br>' +
                     f'Date: {max_dd_date.strftime("%Y-%m-%d")}<br>' +
                     f'Drawdown: {max_dd_value:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Enhanced layout with professional styling
    fig.update_layout(
        title={
            'text': "Underwater Equity Chart - Drawdown Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': CHART_COLORS['text']}
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=CHART_COLORS['grid']
        ),
        yaxis=dict(
            title="Drawdown (%)",
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            tickformat='.1f',
            showline=True,
            linecolor=CHART_COLORS['grid']
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    return fig


def create_returns_distribution_chart(returns: pd.Series, 
                                     benchmark_returns: Optional[pd.Series] = None,
                                     benchmark_name: str = "Benchmark") -> go.Figure:
    """Create comprehensive returns distribution analysis with normality testing.
    
    This function provides detailed statistical analysis of return distributions,
    including comparison with normal distribution, skewness/kurtosis analysis,
    and optional benchmark comparison.
    
    STATISTICAL ANALYSIS:
    ====================
    - Histogram with normal curve overlay
    - Q-Q plot for normality assessment  
    - Skewness and kurtosis calculations
    - Jarque-Bera normality test
    - Percentile analysis and tail risk
    - Benchmark distribution comparison
    
    VISUAL FEATURES:
    ===============
    - Color-coded distribution (green=positive, red=negative)
    - Statistical annotations and interpretations
    - Side-by-side benchmark comparison
    - Tail risk highlighting
    
    Args:
        returns: Strategy return series
        benchmark_returns: Optional benchmark returns for comparison
        benchmark_name: Name for benchmark series
    """
    from scipy import stats
    
    # Calculate enhanced statistics
    returns_pct = returns * 100
    mean_ret = returns_pct.mean()
    std_ret = returns_pct.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    # Percentile analysis
    percentiles = [1, 5, 25, 75, 95, 99]
    percentile_values = [np.percentile(returns_pct, p) for p in percentiles]
    
    # Create subplot structure
    subplot_specs = [[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]]
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Returns Distribution vs Normal Distribution',
            'Statistical Summary', 'Q-Q Plot vs Normal',
            'Tail Risk Analysis'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25],
        specs=subplot_specs
    )
    
    # 1. Enhanced histogram with normal overlay
    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name='Strategy Returns',
            marker_color=CHART_COLORS['primary'],
            opacity=0.7,
            histnorm='probability density',
            hovertemplate='Range: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Normal distribution overlay
    x_range = np.linspace(returns_pct.min(), returns_pct.max(), 200)
    normal_curve = stats.norm.pdf(x_range, mean_ret, std_ret)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='black', dash='dash', width=3),
            hovertemplate='Return: %{x:.2f}%<br>Normal Density: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add benchmark distribution if provided
    if benchmark_returns is not None:
        bench_returns_pct = benchmark_returns * 100
        fig.add_trace(
            go.Histogram(
                x=bench_returns_pct,
                nbinsx=50,
                name=f'{benchmark_name} Returns',
                marker_color=CHART_COLORS['neutral'],
                opacity=0.5,
                histnorm='probability density'
            ),
            row=1, col=1
        )
    
    # 2. Q-Q Plot for normality assessment
    (theoretical_q, sample_q), (slope, intercept, r_value) = stats.probplot(returns, dist="norm", plot=None)
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=sample_q,
            mode='markers',
            name='Q-Q Points',
            marker=dict(color=CHART_COLORS['primary'], size=6),
            showlegend=False,
            hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Perfect normal reference line
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=slope * theoretical_q + intercept,
            mode='lines',
            name='Perfect Normal',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. Box plot for tail analysis
    fig.add_trace(
        go.Box(
            y=returns_pct,
            name='Return Distribution',
            marker_color=CHART_COLORS['primary'],
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add percentile reference lines
    for i, (p, val) in enumerate(zip([5, 95], [percentile_values[1], percentile_values[4]])):
        fig.add_hline(
            y=val,
            line_dash="dot",
            line_color="red" if p == 5 else "green",
            annotation_text=f"{p}th percentile: {val:.2f}%",
            annotation_position="right",
            row=2, col=2
        )
    
    # Enhanced layout with comprehensive styling
    fig.update_layout(
        title={
            'text': "Comprehensive Returns Distribution Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': CHART_COLORS['text']}
        },
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    # Update axes labels and formatting
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)
    
    # Add statistical summary as annotation
    stats_text = (
        f"<b>Statistical Summary:</b><br>"
        f"Mean: {mean_ret:.2f}%<br>"
        f"Std Dev: {std_ret:.2f}%<br>"
        f"Skewness: {skewness:.3f}<br>"
        f"Kurtosis: {kurtosis:.3f}<br>"
        f"Jarque-Bera p-value: {jb_pvalue:.4f}<br>"
        f"Normal Distribution: {'Yes' if jb_pvalue > 0.05 else 'No'}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.65,
        xanchor="right", yanchor="top",
        showarrow=False,
        bordercolor=CHART_COLORS['grid'],
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=11)
    )
    
    return fig


def create_price_chart_with_signals(data: pd.DataFrame,
                                   signals: Optional[pd.DataFrame] = None,
                                   indicators: Optional[List[str]] = None,
                                   show_volume: bool = True,
                                   chart_type: str = 'candlestick') -> go.Figure:
    """Create comprehensive price chart with trading signals and technical indicators.
    
    This function generates a multi-panel financial chart that combines price action,
    trading signals, technical indicators, and volume analysis. It provides a complete
    view of market behavior and strategy execution.
    
    CHART STRUCTURE:
    ===============
    - Main panel: Price chart (candlestick or line) with indicators
    - Signal overlay: Buy/sell markers and signal annotations  
    - Volume panel: Trading volume bars (optional)
    - Indicator panels: Additional technical indicators (optional)
    
    SUPPORTED INDICATORS:
    ====================
    - Moving averages (SMA, EMA)
    - Bollinger Bands with upper/lower bounds
    - RSI with overbought/oversold levels
    - MACD with signal line and histogram
    - Custom indicators from data columns
    
    SIGNAL VISUALIZATION:
    ====================
    - Buy signals: Green upward triangles
    - Sell signals: Red downward triangles  
    - Signal strength: Marker size variation
    - Entry/exit pairs: Connecting lines (optional)
    
    Args:
        data (pd.DataFrame): OHLCV price data with datetime index
                           Required columns: ['open', 'high', 'low', 'close', 'volume']
                           Optional: Technical indicator columns
        signals (Optional[pd.DataFrame]): Trading signal data
                                        Contains 'signal' column with 1=buy, -1=sell, 0=hold
                                        Index must align with price data
        indicators (Optional[List[str]]): List of indicator column names to overlay
                                        Column names should exist in data DataFrame
        show_volume (bool): Whether to include volume panel below price chart
        chart_type (str): Price chart type - 'candlestick' or 'line'
    
    Returns:
        go.Figure: Multi-panel interactive chart with subplots for different data types
        
    Example:
        # Create comprehensive price chart
        fig = create_price_chart_with_signals(
            data=price_data,
            signals=strategy_signals,
            indicators=['SMA_20', 'SMA_50', 'BB_upper', 'BB_lower'],
            show_volume=True,
            chart_type='candlestick'
        )
    """
    # Determine subplot configuration based on options
    subplot_titles = ['Price & Indicators']
    rows = 1
    row_heights = [0.7]
    
    if show_volume:
        subplot_titles.append('Volume')
        rows += 1
        row_heights = [0.7, 0.3]
    
    # Create subplot structure
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}]] * rows
    )
    
    # Add price chart based on specified type
    if chart_type.lower() == 'candlestick':
        # Candlestick chart for detailed OHLC visualization
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=CHART_COLORS['bullish'],
                decreasing_line_color=CHART_COLORS['bearish'],
                showlegend=False
            ),
            row=1, col=1
        )
    else:
        # Line chart for simplified price visualization
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=CHART_COLORS['primary'], width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add technical indicators if specified
    if indicators:
        for indicator in indicators:
            if indicator in data.columns:
                # Handle different indicator types with appropriate styling
                if 'BB' in indicator or 'bollinger' in indicator.lower():
                    # Bollinger Bands - use lighter colors and fills
                    color = CHART_COLORS['secondary'] if 'upper' in indicator else CHART_COLORS['accent']
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(color=color, width=1, dash='dot'),
                            opacity=0.7,
                            hovertemplate=f'{indicator}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                else:
                    # Standard indicators - solid lines
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(width=2),
                            hovertemplate=f'{indicator}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
    
    # Add trading signals if provided
    if signals is not None and not signals.empty:
        # Extract buy and sell signals
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        # Add buy signal markers
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=data.loc[buy_signals.index, 'close'],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=CHART_COLORS['bullish'],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add sell signal markers
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=data.loc[sell_signals.index, 'close'],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=CHART_COLORS['bearish'],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>SELL SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add volume panel if requested
    if show_volume and 'volume' in data.columns:
        # Color volume bars based on price movement
        volume_colors = []
        for i in range(len(data)):
            if i == 0:
                volume_colors.append(CHART_COLORS['neutral'])
            else:
                if data['close'].iloc[i] >= data['close'].iloc[i-1]:
                    volume_colors.append(CHART_COLORS['bullish'])
                else:
                    volume_colors.append(CHART_COLORS['bearish'])
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>',
                showlegend=False
            ),
            row=2 if show_volume else 1, col=1
        )
    
    # Configure chart layout and styling
    fig.update_layout(
        title={
            'text': "Price Chart with Trading Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': CHART_COLORS['text']}
        },
        height=600 if show_volume else 500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor=CHART_COLORS['grid'],
        gridwidth=1,
        showline=True,
        linecolor=CHART_COLORS['grid']
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=CHART_COLORS['grid'],
        gridwidth=1,
        tickformat='$,.2f',
        showline=True,
        linecolor=CHART_COLORS['grid'],
        row=1, col=1
    )
    
    if show_volume:
        fig.update_yaxes(
            title_text="Volume",
            tickformat=',',
            row=2, col=1
        )
    
    return fig


def create_rolling_metrics_chart(equity_data: pd.DataFrame, 
                                window: int = 252,
                                benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """Create comprehensive rolling performance metrics charts.
    
    This function generates multi-panel charts showing how key performance metrics
    evolve over time, helping identify periods of strong/weak performance and
    strategy consistency.
    
    METRICS DISPLAYED:
    =================
    - Rolling Annualized Return: Shows return consistency over time
    - Rolling Volatility: Risk level changes and market regime impacts
    - Rolling Sharpe Ratio: Risk-adjusted performance evolution
    - Rolling Maximum Drawdown: Peak risk exposure over time
    
    VISUAL FEATURES:
    ===============
    - Color-coded performance levels (green=good, red=poor)
    - Reference lines for key thresholds
    - Benchmark comparison overlays (if provided)
    - Performance regime highlighting
    
    Args:
        equity_data: Strategy equity curve data
        window: Rolling window size in periods (default 252 for 1 year)
        benchmark_data: Optional benchmark for comparison
    """
    # Calculate rolling returns and metrics
    returns = equity_data['equity'].pct_change().dropna()
    
    # Rolling metrics calculation
    rolling_return = returns.rolling(window=window).mean() * 252 * 100  # Annualized %
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized %
    rolling_sharpe = rolling_return / rolling_vol  # Risk-adjusted return
    
    # Rolling maximum drawdown calculation
    rolling_drawdown = []
    for i in range(window, len(returns)):
        period_returns = returns.iloc[i-window:i]
        cumulative = (1 + period_returns).cumprod()
        peak = cumulative.cummax()
        dd = ((cumulative - peak) / peak).min() * 100
        rolling_drawdown.append(dd)
    
    rolling_dd_series = pd.Series(rolling_drawdown, index=returns.index[window:])
    
    # Benchmark rolling metrics if provided
    bench_rolling_return = None
    bench_rolling_sharpe = None
    if benchmark_data is not None and not benchmark_data.empty:
        bench_returns = benchmark_data['equity'].pct_change().dropna()
        bench_rolling_return = bench_returns.rolling(window=window).mean() * 252 * 100
        bench_rolling_vol = bench_returns.rolling(window=window).std() * np.sqrt(252) * 100
        bench_rolling_sharpe = bench_rolling_return / bench_rolling_vol
    
    # Create subplot structure
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'Rolling {window//21}-Month Annualized Return (%)',
            f'Rolling {window//21}-Month Volatility (%)',
            f'Rolling {window//21}-Month Sharpe Ratio',
            f'Rolling {window//21}-Month Maximum Drawdown (%)'
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # 1. Rolling Return with performance zones
    fig.add_trace(
        go.Scatter(
            x=rolling_return.index,
            y=rolling_return,
            mode='lines',
            name='Strategy Return',
            line=dict(color=CHART_COLORS['primary'], width=2.5),
            fill='tonexty' if bench_rolling_return is None else None,
            fillcolor='rgba(66, 165, 245, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add benchmark return if available
    if bench_rolling_return is not None:
        fig.add_trace(
            go.Scatter(
                x=bench_rolling_return.index,
                y=bench_rolling_return,
                mode='lines',
                name='Benchmark Return',
                line=dict(color=CHART_COLORS['neutral'], width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add performance reference lines
    fig.add_hline(y=0, row=1, col=1, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(y=10, row=1, col=1, line_dash="dot", line_color="green", opacity=0.3)
    
    # 2. Rolling Volatility with risk zones
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='Strategy Volatility',
            line=dict(color=CHART_COLORS['accent'], width=2.5),
            showlegend=False,
            fill='tonexty',
            fillcolor='rgba(255, 152, 0, 0.1)'
        ),
        row=2, col=1
    )
    
    # Add volatility reference lines
    fig.add_hline(y=15, row=2, col=1, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_hline(y=25, row=2, col=1, line_dash="dot", line_color="red", opacity=0.5)
    
    # 3. Rolling Sharpe Ratio with quality zones
    # Color-code Sharpe ratio based on values
    sharpe_colors = [CHART_COLORS['bullish'] if x > 1 else 
                    CHART_COLORS['neutral'] if x > 0 else 
                    CHART_COLORS['bearish'] for x in rolling_sharpe.fillna(0)]
    
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name='Strategy Sharpe',
            line=dict(color=CHART_COLORS['bullish'], width=2.5),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add benchmark Sharpe if available
    if bench_rolling_sharpe is not None:
        fig.add_trace(
            go.Scatter(
                x=bench_rolling_sharpe.index,
                y=bench_rolling_sharpe,
                mode='lines',
                name='Benchmark Sharpe',
                line=dict(color=CHART_COLORS['neutral'], width=2, dash='dash'),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Sharpe ratio reference lines
    fig.add_hline(y=0, row=3, col=1, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(y=1, row=3, col=1, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=2, row=3, col=1, line_dash="dot", line_color="darkgreen", opacity=0.3)
    
    # 4. Rolling Maximum Drawdown
    fig.add_trace(
        go.Scatter(
            x=rolling_dd_series.index,
            y=rolling_dd_series,
            mode='lines',
            name='Rolling Max DD',
            line=dict(color=CHART_COLORS['bearish'], width=2.5),
            fill='tonexty',
            fillcolor='rgba(239, 83, 80, 0.1)',
            showlegend=False
        ),
        row=4, col=1
    )
    
    # Drawdown reference lines
    fig.add_hline(y=-10, row=4, col=1, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_hline(y=-20, row=4, col=1, line_dash="dot", line_color="red", opacity=0.5)
    
    # Enhanced layout with professional styling
    fig.update_layout(
        height=900,
        title={
            'text': f"Rolling Performance Analysis ({window//21}-Month Window)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': CHART_COLORS['text']}
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    # Update axes styling
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=CHART_COLORS['grid'],
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=CHART_COLORS['grid'],
            gridwidth=1,
            showline=True,
            linecolor=CHART_COLORS['grid'],
            row=i, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    return fig


def create_monthly_returns_heatmap(returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  show_statistics: bool = True) -> go.Figure:
    """Create comprehensive monthly returns heatmap with seasonality analysis.
    
    This function creates a professional calendar-style heatmap showing monthly
    returns with enhanced statistical analysis, seasonality patterns, and
    optional benchmark comparison.
    
    ENHANCED FEATURES:
    =================
    - Color-coded performance (green=positive, red=negative)
    - Monthly statistics (best/worst months, consistency)
    - Seasonality analysis and pattern identification
    - Year-over-year comparison capabilities
    - Optional benchmark overlay
    
    VISUAL DESIGN:
    =============
    - Professional color scale with intuitive mapping
    - Readable percentage formatting in cells
    - Summary statistics panel
    - Interactive hover information
    - Responsive layout for different screen sizes
    
    Args:
        returns: Strategy return series
        benchmark_returns: Optional benchmark returns for comparison
        show_statistics: Whether to display summary statistics
    """
    # Resample to monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Handle benchmark if provided
    bench_monthly = None
    if benchmark_returns is not None:
        bench_monthly = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create pivot table for heatmap
    monthly_data = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot_table = monthly_data.pivot(index='Year', columns='Month', values='Return')
    
    # Calculate monthly statistics
    monthly_stats = {
        'avg_returns': monthly_returns.groupby(monthly_returns.index.month).mean(),
        'volatility': monthly_returns.groupby(monthly_returns.index.month).std(),
        'win_rate': (monthly_returns > 0).groupby(monthly_returns.index.month).mean() * 100,
        'best_month': monthly_returns.groupby(monthly_returns.index.month).max(),
        'worst_month': monthly_returns.groupby(monthly_returns.index.month).min()
    }
    
    # Enhanced month names with statistics
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create main heatmap figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        subplot_titles=('Monthly Returns Calendar', 'Monthly Performance Statistics'),
        vertical_spacing=0.1
    )
    
    # Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_table.values,
            x=month_names,
            y=pivot_table.index,
            colorscale=[
                [0.0, '#8B0000'],    # Dark red for large losses
                [0.2, '#FF4444'],    # Red for losses
                [0.4, '#FFCCCC'],    # Light red for small losses
                [0.5, '#FFFFFF'],    # White for neutral
                [0.6, '#CCFFCC'],    # Light green for small gains
                [0.8, '#44FF44'],    # Green for gains
                [1.0, '#006400']     # Dark green for large gains
            ],
            zmid=0,
            zmin=pivot_table.values.min() if not np.isnan(pivot_table.values).all() else -10,
            zmax=pivot_table.values.max() if not np.isnan(pivot_table.values).all() else 10,
            text=np.where(
                np.isnan(pivot_table.values), 
                '', 
                np.round(pivot_table.values, 1).astype(str) + '%'
            ),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "black"},
            hovertemplate='<b>%{y} %{x}</b><br>' +
                         'Return: %{z:.2f}%<br>' +
                         '<extra></extra>',
            showscale=True,
            colorbar=dict(
                title="Monthly Return (%)",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=5
            )
        ),
        row=1, col=1
    )
    
    # Monthly statistics bar chart
    if show_statistics:
        fig.add_trace(
            go.Bar(
                x=month_names,
                y=[monthly_stats['avg_returns'].get(i+1, 0) for i in range(12)],
                name='Avg Monthly Return',
                marker_color=[
                    CHART_COLORS['bullish'] if val > 0 else CHART_COLORS['bearish']
                    for val in [monthly_stats['avg_returns'].get(i+1, 0) for i in range(12)]
                ],
                opacity=0.8,
                hovertemplate='Month: %{x}<br>Avg Return: %{y:.2f}%<br><extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="gray")
    
    # Enhanced layout with professional styling
    fig.update_layout(
        title={
            'text': "Monthly Returns Calendar - Seasonality Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': CHART_COLORS['text']}
        },
        height=700 if show_statistics else 500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    # Update axes
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="Year", row=1, col=1)
    
    if show_statistics:
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Average Return (%)", row=2, col=1)
    
    # Add summary statistics as annotation
    if show_statistics:
        best_month_idx = monthly_stats['avg_returns'].idxmax()
        worst_month_idx = monthly_stats['avg_returns'].idxmin()
        
        summary_text = (
            f"<b>Seasonality Summary:</b><br>"
            f"Best Month: {month_names[best_month_idx-1]} "
            f"({monthly_stats['avg_returns'][best_month_idx]:.2f}%)<br>"
            f"Worst Month: {month_names[worst_month_idx-1]} "
            f"({monthly_stats['avg_returns'][worst_month_idx]:.2f}%)<br>"
            f"Most Consistent: {month_names[monthly_stats['volatility'].idxmin()-1]}<br>"
            f"Highest Win Rate: {month_names[monthly_stats['win_rate'].idxmax()-1]} "
            f"({monthly_stats['win_rate'].max():.0f}%)"
        )
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.98, y=0.95,
            xanchor="right", yanchor="top",
            showarrow=False,
            bordercolor=CHART_COLORS['grid'],
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=10)
        )
    
    return fig


def create_trade_distribution_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Create comprehensive trade distribution analysis with P&L insights.
    
    This function provides detailed analysis of individual trade performance,
    including distribution patterns, win/loss analysis, and trade size insights.
    
    ANALYSIS FEATURES:
    =================
    - Trade P&L histogram with normal curve overlay
    - Win/loss distribution analysis
    - Trade duration vs P&L scatter plot
    - Cumulative P&L progression
    - Statistical summary and outlier identification
    
    Args:
        trades_df: DataFrame with trade records including 'pnl', 'duration', 'return' columns
    """
    if trades_df.empty:
        return go.Figure().add_annotation(
            text="No trade data available for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create comprehensive trade analysis layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Trade P&L Distribution',
            'Trade Duration vs Return',
            'Cumulative Trade P&L',
            'Win/Loss Analysis'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "domain"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Trade P&L Distribution
    trade_returns = trades_df['return'] * 100 if 'return' in trades_df.columns else trades_df['pnl']
    
    fig.add_trace(
        go.Histogram(
            x=trade_returns,
            nbinsx=30,
            name='Trade Returns',
            marker_color=CHART_COLORS['primary'],
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=1
    )
    
    # Add normal distribution overlay
    from scipy import stats
    mean_return = trade_returns.mean()
    std_return = trade_returns.std()
    x_range = np.linspace(trade_returns.min(), trade_returns.max(), 100)
    normal_curve = stats.norm.pdf(x_range, mean_return, std_return)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Trade Duration vs Return Scatter
    if 'duration' in trades_df.columns:
        colors = [CHART_COLORS['bullish'] if x > 0 else CHART_COLORS['bearish'] for x in trade_returns]
        
        fig.add_trace(
            go.Scatter(
                x=trades_df['duration'],
                y=trade_returns,
                mode='markers',
                name='Individual Trades',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                hovertemplate='Duration: %{x} days<br>Return: %{y:.2f}%<br><extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, row=1, col=2, line_dash="dot", line_color="gray")
    
    # 3. Cumulative Trade P&L
    cumulative_pnl = trade_returns.cumsum()
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_pnl) + 1)),
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color=CHART_COLORS['primary'], width=2.5),
            fill='tonexty',
            fillcolor=f'rgba(66, 165, 245, 0.1)',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Win/Loss Pie Chart
    winning_trades = (trade_returns > 0).sum()
    losing_trades = (trade_returns < 0).sum()
    breakeven_trades = (trade_returns == 0).sum()
    
    fig.add_trace(
        go.Pie(
            labels=['Winning Trades', 'Losing Trades', 'Breakeven'],
            values=[winning_trades, losing_trades, breakeven_trades],
            marker_colors=[CHART_COLORS['bullish'], CHART_COLORS['bearish'], CHART_COLORS['neutral']],
            hovertemplate='%{label}: %{value} trades<br>Percentage: %{percent}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': "Comprehensive Trade Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': CHART_COLORS['text']}
        },
        height=700,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    
    if 'duration' in trades_df.columns:
        fig.update_xaxes(title_text="Trade Duration (days)", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Trade Number", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)
    
    # Add statistical summary
    win_rate = winning_trades / len(trade_returns) * 100
    avg_win = trade_returns[trade_returns > 0].mean() if winning_trades > 0 else 0
    avg_loss = trade_returns[trade_returns < 0].mean() if losing_trades > 0 else 0
    profit_factor = abs(trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum()) if losing_trades > 0 else np.inf
    
    stats_text = (
        f"<b>Trade Statistics:</b><br>"
        f"Total Trades: {len(trade_returns)}<br>"
        f"Win Rate: {win_rate:.1f}%<br>"
        f"Avg Win: {avg_win:.2f}%<br>"
        f"Avg Loss: {avg_loss:.2f}%<br>"
        f"Profit Factor: {profit_factor:.2f}<br>"
        f"Best Trade: {trade_returns.max():.2f}%<br>"
        f"Worst Trade: {trade_returns.min():.2f}%"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False,
        bordercolor=CHART_COLORS['grid'],
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=10)
    )
    
    return fig


def create_correlation_matrix(data: pd.DataFrame) -> go.Figure:
    """Create enhanced correlation matrix heatmap with clustering analysis."""
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create hierarchical clustering for better organization
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    # Convert correlation to distance matrix
    distance_matrix = 1 - corr_matrix.abs()
    condensed_distances = squareform(distance_matrix)
    
    # Perform clustering
    linkage_matrix = linkage(condensed_distances, method='ward')
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
    
    # Reorder correlation matrix based on clustering
    clustered_order = dendro['leaves']
    clustered_corr = corr_matrix.iloc[clustered_order, clustered_order]
    
    fig = go.Figure(data=go.Heatmap(
        z=clustered_corr.values,
        x=clustered_corr.columns,
        y=clustered_corr.columns,
        colorscale=[
            [0.0, '#8B0000'],    # Dark red for strong negative correlation
            [0.25, '#FF4444'],   # Red for negative correlation
            [0.5, '#FFFFFF'],    # White for no correlation
            [0.75, '#4444FF'],   # Blue for positive correlation
            [1.0, '#000080']     # Dark blue for strong positive correlation
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(clustered_corr.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>' +
                     'Correlation: %{z:.3f}<br>' +
                     '<extra></extra>',
        colorbar=dict(
            title="Correlation Coefficient",
            titleside="right",
            tickmode="linear",
            tick0=-1,
            dtick=0.5
        )
    ))
    
    fig.update_layout(
        title={
            'text': "Asset Correlation Matrix (Clustered)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': CHART_COLORS['text']}
        },
        width=700,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=CHART_COLORS['text'])
    )
    
    return fig 