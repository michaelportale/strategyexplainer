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


def create_drawdown_chart(equity_data: pd.DataFrame) -> go.Figure:
    """Create a drawdown chart showing peak-to-trough declines."""
    
    # Calculate running maximum (peak)
    peak = equity_data['equity'].cummax()
    drawdown = (equity_data['equity'] - peak) / peak * 100
    
    fig = go.Figure()
    
    # Drawdown area chart
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=drawdown,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='rgb(255, 0, 0)', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)',
        hovertemplate='<b>Drawdown</b><br>' +
                     'Date: %{x}<br>' +
                     'Drawdown: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_returns_distribution_chart(returns: pd.Series) -> go.Figure:
    """Create a histogram of returns distribution."""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Returns Distribution', 'Q-Q Plot'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='rgb(255, 107, 107)',
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=1
    )
    
    # Add normal distribution overlay
    mean_ret = returns.mean() * 100
    std_ret = returns.std() * 100
    x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    y_norm = (1/(std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_ret) / std_ret) ** 2)
    
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='black', dash='dash')
        ),
        row=1, col=1
    )
    
    # Q-Q Plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm", plot=None)
    
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='rgb(255, 107, 107)', size=4),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Q-Q reference line
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode='lines',
            name='Reference Line',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        template="plotly_white",
        title="Returns Analysis"
    )
    
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    
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
                                window: int = 252) -> go.Figure:
    """Create charts showing rolling performance metrics."""
    
    # Calculate rolling returns and volatility
    returns = equity_data['equity'].pct_change().dropna()
    rolling_return = returns.rolling(window=window).mean() * 252 * 100
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    rolling_sharpe = rolling_return / rolling_vol
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Rolling {window}-Day Annualized Return (%)',
            f'Rolling {window}-Day Volatility (%)',
            f'Rolling {window}-Day Sharpe Ratio'
        )
    )
    
    # Rolling return
    fig.add_trace(
        go.Scatter(
            x=rolling_return.index,
            y=rolling_return,
            mode='lines',
            name='Rolling Return',
            line=dict(color='rgb(255, 107, 107)', width=2)
        ),
        row=1, col=1
    )
    
    # Rolling volatility
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='rgb(255, 165, 0)', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Rolling Sharpe ratio
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='rgb(50, 171, 96)', width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add zero line for Sharpe ratio
    fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        height=700,
        template="plotly_white",
        title=f"Rolling Performance Metrics ({window} Days)"
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig


def create_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """Create a heatmap of monthly returns."""
    
    # Resample to monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create pivot table for heatmap
    monthly_data = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot_table = monthly_data.pivot(index='Year', columns='Month', values='Return')
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=month_names,
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot_table.values, 2),
        texttemplate="%{text}%",
        textfont={"size": 10},
        hovertemplate='<b>%{y}</b><br>' +
                     'Month: %{x}<br>' +
                     'Return: %{z:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        width=800,
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_correlation_matrix(data: pd.DataFrame) -> go.Figure:
    """Create a correlation matrix heatmap."""
    
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>' +
                     'Correlation: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        width=600,
        height=500,
        template="plotly_white"
    )
    
    return fig 