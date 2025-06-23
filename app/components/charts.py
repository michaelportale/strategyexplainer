"""Chart generation helpers for financial visualizations."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


def create_equity_curve_chart(equity_data: pd.DataFrame, 
                            benchmark_data: Optional[pd.DataFrame] = None,
                            benchmark_name: str = "Benchmark") -> go.Figure:
    """Create an equity curve chart with optional benchmark comparison."""
    
    fig = go.Figure()
    
    # Strategy equity curve
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=equity_data['equity'],
        mode='lines',
        name='Strategy',
        line=dict(color='rgb(255, 107, 107)', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Date: %{x}<br>' +
                     'Equity: $%{y:,.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Benchmark comparison if provided
    if benchmark_data is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data['equity'],
            mode='lines',
            name=benchmark_name,
            line=dict(color='rgb(99, 110, 250)', width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Equity: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template="plotly_white"
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


def create_price_and_signals_chart(price_data: pd.DataFrame, 
                                 signals: pd.DataFrame,
                                 title: str = "Price Action with Trading Signals") -> go.Figure:
    """Create a price chart with buy/sell signals overlaid."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Position'),
        row_heights=[0.8, 0.2]
    )
    
    # Price candlestick chart
    if 'open' in price_data.columns:
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Buy signals
    buy_signals = signals[signals['signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green'
                )
            ),
            row=1, col=1
        )
    
    # Sell signals
    sell_signals = signals[signals['signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red'
                )
            ),
            row=1, col=1
        )
    
    # Position chart
    if 'position' in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['position'],
                mode='lines',
                name='Position',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.3)'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
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