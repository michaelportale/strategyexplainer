"""Professional performance metrics dashboard for institutional-grade strategy analysis.

This module provides a comprehensive dashboard for presenting performance metrics
in a format suitable for portfolio managers, institutional investors, and
quantitative analysts.

Key Features:
- Professional metric categorization and presentation
- Interactive visualizations with Plotly
- Detailed metric explanations and interpretations
- Export capabilities for reports and presentations
- Benchmark comparison analysis
- Risk attribution and decomposition
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .enhanced_metrics import EnhancedPerformanceMetrics, MetricsFormatter


class MetricsDashboard:
    """Professional performance metrics dashboard with institutional-grade presentation."""
    
    def __init__(self, theme: str = "professional"):
        """Initialize metrics dashboard.
        
        Args:
            theme: Dashboard theme ("professional", "dark", "light")
        """
        self.theme = theme
        self.metrics_calculator = EnhancedPerformanceMetrics()
        self.formatter = MetricsFormatter()
        self.logger = logging.getLogger(__name__)
        
        # Color schemes for different themes
        self.color_schemes = {
            "professional": {
                "primary": "#2E86AB",
                "secondary": "#A23B72", 
                "success": "#00C851",
                "warning": "#FF8800",
                "danger": "#FF4444",
                "info": "#33B5E5",
                "background": "#FFFFFF",
                "text": "#333333"
            },
            "dark": {
                "primary": "#4FC3F7",
                "secondary": "#FF6EC7",
                "success": "#4CAF50", 
                "warning": "#FF9800",
                "danger": "#F44336",
                "info": "#2196F3",
                "background": "#1E1E1E",
                "text": "#FFFFFF"
            }
        }
    
    def create_comprehensive_dashboard(self,
                                     equity_curve: pd.DataFrame,
                                     trades: pd.DataFrame,
                                     signals: Optional[pd.DataFrame] = None,
                                     benchmark_returns: Optional[pd.Series] = None,
                                     strategy_name: str = "Strategy") -> Dict[str, Any]:
        """Create a comprehensive metrics dashboard.
        
        Args:
            equity_curve: DataFrame with equity values over time
            trades: DataFrame with individual trade records
            signals: Optional DataFrame with strategy signals
            benchmark_returns: Optional benchmark returns for comparison
            strategy_name: Name of the strategy for display
            
        Returns:
            Dictionary containing formatted metrics, charts, and analysis
        """
        # Calculate enhanced metrics
        raw_metrics = self.metrics_calculator.calculate_all_metrics(
            equity_curve, trades, signals, benchmark_returns
        )
        
        # Format metrics for display
        formatted_metrics = self.formatter.format_all_metrics(raw_metrics)
        
        # Create visualizations
        charts = self._create_dashboard_charts(equity_curve, trades, raw_metrics, benchmark_returns)
        
        # Generate insights and analysis
        insights = self._generate_insights(raw_metrics)
        
        # Create summary cards
        summary_cards = self._create_summary_cards(raw_metrics)
        
        # Generate detailed analysis
        detailed_analysis = self._create_detailed_analysis(raw_metrics)
        
        return {
            "strategy_name": strategy_name,
            "generated_at": datetime.now().isoformat(),
            "metrics": {
                "raw": raw_metrics,
                "formatted": formatted_metrics
            },
            "summary_cards": summary_cards,
            "charts": charts,
            "insights": insights,
            "detailed_analysis": detailed_analysis,
            "dashboard_metadata": {
                "theme": self.theme,
                "metric_categories": len(raw_metrics),
                "total_metrics": sum(len(v) for v in raw_metrics.values() if isinstance(v, dict)),
                "has_benchmark": benchmark_returns is not None,
                "has_signals": signals is not None
            }
        }
    
    def _create_summary_cards(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create high-level summary cards for dashboard overview."""
        basic = metrics.get('basic_metrics', {})
        exposure = metrics.get('exposure_metrics', {})
        tail_risk = metrics.get('tail_risk_metrics', {})
        summary = metrics.get('summary_metrics', {})
        
        cards = [
            {
                "title": "Total Return",
                "value": basic.get('total_return', 0),
                "format": "percentage",
                "icon": "trending-up",
                "color": "success" if basic.get('total_return', 0) > 0 else "danger",
                "subtitle": f"Annualized: {basic.get('annual_return', 0):.1%}"
            },
            {
                "title": "Sharpe Ratio",
                "value": basic.get('sharpe_ratio', 0),
                "format": "decimal2",
                "icon": "award",
                "color": "primary" if basic.get('sharpe_ratio', 0) > 1 else "warning",
                "subtitle": "Risk-adjusted return"
            },
            {
                "title": "Maximum Drawdown",
                "value": basic.get('max_drawdown', 0),
                "format": "percentage",
                "icon": "trending-down", 
                "color": "danger" if basic.get('max_drawdown', 0) < -0.1 else "warning",
                "subtitle": f"Recovery: {basic.get('recovery_factor', 0):.2f}"
            },
            {
                "title": "Time in Market",
                "value": exposure.get('time_in_market', 0),
                "format": "percentage",
                "icon": "clock",
                "color": "info",
                "subtitle": "Capital efficiency"
            },
            {
                "title": "Win Rate",
                "value": basic.get('win_rate', 0),
                "format": "percentage", 
                "icon": "target",
                "color": "success" if basic.get('win_rate', 0) > 0.5 else "warning",
                "subtitle": f"Profit Factor: {basic.get('profit_factor', 0):.2f}"
            },
            {
                "title": "Strategy Score",
                "value": summary.get('strategy_efficiency_score', 0) / 100,
                "format": "percentage",
                "icon": "star",
                "color": "primary" if summary.get('strategy_efficiency_score', 0) > 60 else "warning",
                "subtitle": "Overall efficiency"
            }
        ]
        
        return cards
    
    def _create_dashboard_charts(self,
                               equity_curve: pd.DataFrame,
                               trades: pd.DataFrame,
                               metrics: Dict[str, Any],
                               benchmark_returns: Optional[pd.Series] = None) -> Dict[str, go.Figure]:
        """Create comprehensive dashboard charts."""
        charts = {}
        
        # 1. Equity Curve Chart
        charts['equity_curve'] = self._create_equity_curve_chart(equity_curve, benchmark_returns)
        
        # 2. Drawdown Chart
        charts['drawdown'] = self._create_drawdown_chart(equity_curve)
        
        # 3. Returns Distribution
        charts['returns_distribution'] = self._create_returns_distribution_chart(equity_curve)
        
        # 4. Risk-Return Scatter (if benchmark available)
        if benchmark_returns is not None:
            charts['risk_return_scatter'] = self._create_risk_return_scatter(equity_curve, benchmark_returns)
        
        # 5. Monthly Returns Heatmap
        charts['monthly_returns'] = self._create_monthly_returns_heatmap(equity_curve)
        
        # 6. Metrics Comparison Chart
        charts['metrics_comparison'] = self._create_metrics_comparison_chart(metrics)
        
        # 7. Exposure Analysis
        if 'exposure_metrics' in metrics:
            charts['exposure_analysis'] = self._create_exposure_chart(metrics['exposure_metrics'])
        
        # 8. Tail Risk Analysis
        if 'tail_risk_metrics' in metrics:
            charts['tail_risk'] = self._create_tail_risk_chart(equity_curve, metrics['tail_risk_metrics'])
        
        return charts
    
    def _create_equity_curve_chart(self, equity_curve: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> go.Figure:
        """Create professional equity curve chart with benchmark comparison."""
        fig = go.Figure()
        
        # Strategy equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='Strategy',
            line=dict(color=self.color_schemes[self.theme]["primary"], width=2)
        ))
        
        # Benchmark comparison if available
        if benchmark_returns is not None:
            # Calculate benchmark equity curve
            benchmark_equity = (1 + benchmark_returns).cumprod() * equity_curve['equity'].iloc[0]
            benchmark_aligned = benchmark_equity.reindex(equity_curve.index, method='ffill')
            
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=benchmark_aligned,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.color_schemes[self.theme]["secondary"], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Equity Curve Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            hovermode="x unified",
            showlegend=True
        )
        
        return fig
    
    def _create_drawdown_chart(self, equity_curve: pd.DataFrame) -> go.Figure:
        """Create underwater equity chart showing drawdown periods."""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return go.Figure()
        
        equity = equity_curve['equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        fig = go.Figure()
        
        # Drawdown area chart
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color=self.color_schemes[self.theme]["danger"]),
            fillcolor=f"rgba(255, 68, 68, 0.3)"
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Drawdown Analysis (Underwater Equity)",
            xaxis_title="Date", 
            yaxis_title="Drawdown",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            yaxis=dict(tickformat=".1%"),
            showlegend=False
        )
        
        return fig
    
    def _create_returns_distribution_chart(self, equity_curve: pd.DataFrame) -> go.Figure:
        """Create returns distribution analysis chart."""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return go.Figure()
        
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Create subplots for histogram and Q-Q plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Returns Distribution", "Normal Q-Q Plot"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name="Returns",
                marker_color=self.color_schemes[self.theme]["primary"],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = np.exp(-0.5 * ((x_range - returns.mean()) / returns.std()) ** 2) / (returns.std() * np.sqrt(2 * np.pi))
        normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 30  # Scale to histogram
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name="Normal Distribution",
                line=dict(color=self.color_schemes[self.theme]["warning"], dash='dash')
            ),
            row=1, col=1
        )
        
        # Q-Q plot
        from scipy import stats
        qq_theoretical, qq_actual = stats.probplot(returns, dist="norm")
        
        fig.add_trace(
            go.Scatter(
                x=qq_theoretical[0],
                y=qq_theoretical[1],
                mode='markers',
                name="Q-Q Plot",
                marker=dict(color=self.color_schemes[self.theme]["info"])
            ),
            row=1, col=2
        )
        
        # Perfect normal line
        fig.add_trace(
            go.Scatter(
                x=qq_theoretical[0], 
                y=qq_theoretical[0] * returns.std() + returns.mean(),
                mode='lines',
                name="Perfect Normal",
                line=dict(color=self.color_schemes[self.theme]["warning"], dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Returns Distribution Analysis",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            showlegend=True
        )
        
        return fig
    
    def _create_risk_return_scatter(self, equity_curve: pd.DataFrame, benchmark_returns: pd.Series) -> go.Figure:
        """Create risk-return scatter plot with benchmark comparison."""
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Calculate metrics
        strategy_return = returns.mean() * 252  # Annualized
        strategy_vol = returns.std() * np.sqrt(252)  # Annualized
        
        benchmark_return = benchmark_returns.mean() * 252
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        fig = go.Figure()
        
        # Benchmark point
        fig.add_trace(go.Scatter(
            x=[benchmark_vol],
            y=[benchmark_return],
            mode='markers',
            name='Benchmark',
            marker=dict(
                size=15,
                color=self.color_schemes[self.theme]["secondary"],
                symbol='square'
            )
        ))
        
        # Strategy point
        fig.add_trace(go.Scatter(
            x=[strategy_vol],
            y=[strategy_return],
            mode='markers',
            name='Strategy',
            marker=dict(
                size=15,
                color=self.color_schemes[self.theme]["primary"],
                symbol='circle'
            )
        ))
        
        # Efficient frontier reference lines
        vol_range = np.linspace(0, max(strategy_vol, benchmark_vol) * 1.2, 100)
        
        # Risk-free rate line (Sharpe ratio = 1)
        rf_rate = 0.02  # 2% risk-free rate
        sharpe_1_line = rf_rate + vol_range
        
        fig.add_trace(go.Scatter(
            x=vol_range,
            y=sharpe_1_line,
            mode='lines',
            name='Sharpe = 1.0',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Volatility (Annualized)",
            yaxis_title="Return (Annualized)",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%")
        )
        
        return fig
    
    def _create_monthly_returns_heatmap(self, equity_curve: pd.DataFrame) -> go.Figure:
        """Create monthly returns heatmap."""
        if equity_curve.empty:
            return go.Figure()
        
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_data = monthly_returns.to_frame('returns')
        monthly_data['year'] = monthly_data.index.year
        monthly_data['month'] = monthly_data.index.month
        
        heatmap_data = monthly_data.pivot(index='year', columns='month', values='returns')
        
        # Month names for better display
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=month_names,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            colorbar=dict(title="Monthly Return"),
            text=np.round(heatmap_data.values * 100, 1),
            texttemplate="%{text}%",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white" if self.theme != "dark" else "plotly_dark"
        )
        
        return fig
    
    def _create_metrics_comparison_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create comprehensive metrics comparison chart."""
        # Key metrics for comparison
        key_metrics = {
            'Return Metrics': {
                'Total Return': metrics.get('basic_metrics', {}).get('total_return', 0),
                'Annual Return': metrics.get('basic_metrics', {}).get('annual_return', 0),
                'Monthly Return': metrics.get('basic_metrics', {}).get('monthly_return', 0)
            },
            'Risk Metrics': {
                'Volatility': metrics.get('basic_metrics', {}).get('annual_volatility', 0),
                'Max Drawdown': abs(metrics.get('basic_metrics', {}).get('max_drawdown', 0)),
                'VaR 95%': abs(metrics.get('basic_metrics', {}).get('var_95', 0))
            },
            'Risk-Adjusted': {
                'Sharpe Ratio': metrics.get('basic_metrics', {}).get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('basic_metrics', {}).get('sortino_ratio', 0),
                'Calmar Ratio': metrics.get('basic_metrics', {}).get('calmar_ratio', 0)
            }
        }
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=list(key_metrics.keys()),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = [self.color_schemes[self.theme]["primary"], 
                 self.color_schemes[self.theme]["warning"],
                 self.color_schemes[self.theme]["success"]]
        
        for i, (category, category_metrics) in enumerate(key_metrics.items()):
            fig.add_trace(
                go.Bar(
                    x=list(category_metrics.keys()),
                    y=list(category_metrics.values()),
                    name=category,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Key Metrics Comparison",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            height=400
        )
        
        return fig
    
    def _create_exposure_chart(self, exposure_metrics: Dict[str, float]) -> go.Figure:
        """Create exposure analysis chart."""
        # Pie chart showing time allocation
        labels = ['Time in Market', 'Time in Cash']
        values = [
            exposure_metrics.get('time_in_market', 0),
            exposure_metrics.get('time_in_cash', 1)
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=[self.color_schemes[self.theme]["primary"], 
                          self.color_schemes[self.theme]["secondary"]]
        )])
        
        fig.update_layout(
            title="Capital Allocation Analysis",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            annotations=[dict(text=f"{values[0]:.1%}<br>In Market", 
                             x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        
        return fig
    
    def _create_tail_risk_chart(self, equity_curve: pd.DataFrame, tail_metrics: Dict[str, float]) -> go.Figure:
        """Create tail risk analysis chart."""
        if equity_curve.empty:
            return go.Figure()
        
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [np.percentile(returns, p) for p in percentiles]
        
        fig = go.Figure()
        
        # Box plot of returns
        fig.add_trace(go.Box(
            y=returns,
            name="Returns Distribution",
            marker_color=self.color_schemes[self.theme]["primary"]
        ))
        
        # Add percentile markers
        for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
            if p in [1, 5, 95, 99]:  # Highlight extreme percentiles
                fig.add_hline(
                    y=val,
                    line_dash="dash",
                    annotation_text=f"{p}th percentile: {val:.2%}",
                    annotation_position="right"
                )
        
        fig.update_layout(
            title="Tail Risk Analysis",
            yaxis_title="Returns",
            template="plotly_white" if self.theme != "dark" else "plotly_dark",
            yaxis=dict(tickformat=".1%")
        )
        
        return fig
    
    def _generate_insights(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate automated insights from metrics analysis."""
        insights = []
        
        basic = metrics.get('basic_metrics', {})
        exposure = metrics.get('exposure_metrics', {})
        tail_risk = metrics.get('tail_risk_metrics', {})
        summary = metrics.get('summary_metrics', {})
        
        # Return insights
        total_return = basic.get('total_return', 0)
        if total_return > 0.1:
            insights.append({
                "type": "positive",
                "category": "Returns",
                "title": "Strong Total Return",
                "description": f"Strategy generated {total_return:.1%} total return, indicating strong performance."
            })
        elif total_return < -0.05:
            insights.append({
                "type": "negative", 
                "category": "Returns",
                "title": "Negative Returns",
                "description": f"Strategy had negative returns of {total_return:.1%}, requiring analysis of risk management."
            })
        
        # Risk insights
        sharpe = basic.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            insights.append({
                "type": "positive",
                "category": "Risk-Adjusted Performance",
                "title": "Excellent Sharpe Ratio",
                "description": f"Sharpe ratio of {sharpe:.2f} indicates superior risk-adjusted returns."
            })
        elif sharpe < 0.5:
            insights.append({
                "type": "warning",
                "category": "Risk-Adjusted Performance", 
                "title": "Low Sharpe Ratio",
                "description": f"Sharpe ratio of {sharpe:.2f} suggests poor risk-adjusted performance."
            })
        
        # Exposure insights
        time_in_market = exposure.get('time_in_market', 0)
        if time_in_market < 0.5:
            insights.append({
                "type": "info",
                "category": "Capital Efficiency",
                "title": "Low Market Exposure",
                "description": f"Strategy was only invested {time_in_market:.1%} of the time, suggesting selective approach."
            })
        
        # Tail risk insights
        tail_ratio = tail_risk.get('tail_ratio_95_5', 0)
        if tail_ratio > 1.2:
            insights.append({
                "type": "positive",
                "category": "Tail Risk",
                "title": "Favorable Tail Risk Profile",
                "description": f"Tail ratio of {tail_ratio:.2f} indicates better upside than downside extremes."
            })
        
        # Skewness insights
        skewness = tail_risk.get('skewness', 0)
        if skewness > 0.5:
            insights.append({
                "type": "positive",
                "category": "Return Distribution",
                "title": "Positive Skewness",
                "description": f"Positive skewness of {skewness:.2f} indicates more frequent small losses and occasional large gains."
            })
        elif skewness < -0.5:
            insights.append({
                "type": "warning",
                "category": "Return Distribution",
                "title": "Negative Skewness", 
                "description": f"Negative skewness of {skewness:.2f} indicates frequent small gains but occasional large losses."
            })
        
        return insights
    
    def _create_detailed_analysis(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create detailed written analysis of strategy performance."""
        basic = metrics.get('basic_metrics', {})
        exposure = metrics.get('exposure_metrics', {})
        benchmark = metrics.get('enhanced_benchmark_metrics', {})
        
        analysis = {}
        
        # Executive Summary
        analysis['executive_summary'] = f"""
        The strategy generated a total return of {basic.get('total_return', 0):.1%} with a Sharpe ratio of {basic.get('sharpe_ratio', 0):.2f}.
        Maximum drawdown was {basic.get('max_drawdown', 0):.1%}, indicating {'moderate' if abs(basic.get('max_drawdown', 0)) < 0.15 else 'high'} risk levels.
        The strategy was invested {exposure.get('time_in_market', 0):.1%} of the time, suggesting {'active' if exposure.get('time_in_market', 0) > 0.7 else 'selective'} market participation.
        """
        
        # Risk Analysis
        analysis['risk_analysis'] = f"""
        Risk metrics show annual volatility of {basic.get('annual_volatility', 0):.1%} with downside deviation of {basic.get('downside_deviation', 0):.1%}.
        VaR at 95% confidence level is {basic.get('var_95', 0):.1%}, while expected shortfall is {basic.get('expected_shortfall_95', 0):.1%}.
        The strategy's risk profile is {'conservative' if basic.get('annual_volatility', 0) < 0.15 else 'aggressive'} based on volatility levels.
        """
        
        # Benchmark Comparison (if available)
        if benchmark:
            analysis['benchmark_comparison'] = f"""
            Relative to benchmark, the strategy achieved alpha of {benchmark.get('alpha', 0):.1%} with beta of {benchmark.get('beta', 1):.2f}.
            Information ratio of {benchmark.get('information_ratio', 0):.2f} indicates {'strong' if benchmark.get('information_ratio', 0) > 0.5 else 'weak'} active management value.
            Correlation with benchmark is {benchmark.get('correlation', 0):.2f}, suggesting {'high' if benchmark.get('correlation', 0) > 0.7 else 'low'} market dependence.
            """
        
        return analysis
    
    def export_dashboard_report(self, dashboard_data: Dict[str, Any], format: str = "html") -> str:
        """Export dashboard as comprehensive report.
        
        Args:
            dashboard_data: Dashboard data from create_comprehensive_dashboard
            format: Export format ("html", "pdf", "json")
            
        Returns:
            File path or content string
        """
        if format == "html":
            return self._export_html_report(dashboard_data)
        elif format == "json":
            return self._export_json_report(dashboard_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_html_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Export dashboard as HTML report."""
        # This would generate a comprehensive HTML report
        # Implementation would include charts, tables, and analysis
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Performance Report - {dashboard_data['strategy_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Strategy Performance Report</h1>
            <h2>{dashboard_data['strategy_name']}</h2>
            <p>Generated: {dashboard_data['generated_at']}</p>
            
            <h3>Summary Metrics</h3>
            <!-- Metrics cards would be rendered here -->
            
            <h3>Detailed Analysis</h3>
            <!-- Detailed analysis would be rendered here -->
            
        </body>
        </html>
        """
        return html_content
    
    def _export_json_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Export dashboard as JSON report."""
        import json
        return json.dumps(dashboard_data, indent=2, default=str)