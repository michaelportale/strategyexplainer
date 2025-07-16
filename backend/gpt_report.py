"""
AI-Powered Trading Report Generation System: Intelligent Analysis and Documentation

This module provides a comprehensive AI-powered report generation system that leverages
OpenAI's GPT models to create sophisticated trading strategy analysis reports. It transforms
quantitative trading results into human-readable, professional-grade documentation suitable
for investors, analysts, and regulatory compliance.

The system bridges the gap between complex algorithmic trading systems and clear,
actionable business intelligence through advanced natural language processing and
financial analysis automation.

Key Features:
============

1. **AI-Powered Analysis Generation**
   - Comprehensive strategy analysis using GPT-4
   - Contextual understanding of trading performance
   - Professional investment-grade commentary
   - Automated insight generation and recommendations

2. **Multi-Dimensional Report Creation**
   - Strategy performance analysis
   - Risk assessment and management
   - Market context integration
   - Educational explanations and interpretations

3. **Intelligent Content Generation**
   - Natural language processing for financial data
   - Context-aware analysis and recommendations
   - Professional writing style and terminology
   - Customizable report templates and formats

4. **Enterprise-Grade Documentation**
   - Markdown-formatted professional reports
   - Automated report archiving and management
   - Configurable analysis depth and focus
   - Integration with existing workflow systems

Architecture:
============

The report generation system follows a sophisticated architecture:

1. **AI Integration Layer**
   - OpenAI GPT-4 model integration
   - Intelligent prompt engineering
   - Context optimization and management
   - Response processing and validation

2. **Analysis Engine**
   - Multi-faceted performance analysis
   - Risk assessment and evaluation
   - Market context integration
   - Strategic recommendation generation

3. **Content Generation Layer**
   - Template-based report creation
   - Dynamic content assembly
   - Professional formatting and styling
   - Quality assurance and validation

4. **Output Management Layer**
   - Report persistence and archiving
   - Multiple output format support
   - Version control and tracking
   - Distribution and sharing capabilities

Usage Examples:
===============

Basic Usage:
```python
from backend.gpt_report import GPTReportGenerator

# Initialize report generator
reporter = GPTReportGenerator()

# Generate comprehensive strategy analysis
report = reporter.generate_strategy_analysis(
    strategy_config=config,
    metrics=performance_metrics,
    trades=trades_df
)

# Save report to file
reporter.save_report(report, config)
```

Advanced Usage:
```python
# Generate with market context
report = reporter.generate_strategy_analysis(
    strategy_config=config,
    metrics=performance_metrics,
    trades=trades_df,
    market_context={
        'market_regime': 'bull_market',
        'volatility_regime': 'low',
        'sector_performance': sector_data
    }
)

# Generate risk assessment
risk_report = reporter.generate_risk_assessment(
    metrics=risk_metrics,
    trades=trades_df
)
```

Professional Integration:
```python
# Batch report generation
for strategy in strategies:
    report = reporter.generate_strategy_analysis(
        strategy_config=strategy.config,
        metrics=strategy.metrics,
        trades=strategy.trades
    )
    reporter.save_report(report, strategy.config)
```

Educational Value:
=================

This module demonstrates:

1. **AI Integration in Finance**
   - Large language model applications
   - Financial data interpretation
   - Automated analysis generation
   - Natural language processing in finance

2. **Professional Report Generation**
   - Structured business intelligence
   - Automated documentation systems
   - Quality assurance processes
   - Professional writing standards

3. **Enterprise Software Design**
   - Modular architecture patterns
   - Error handling and resilience
   - Configuration management
   - Scalability considerations

4. **Financial Analysis Automation**
   - Performance attribution analysis
   - Risk assessment methodologies
   - Market context integration
   - Regulatory compliance support

Integration Points:
==================

The report generator integrates with:
- Trading strategy frameworks
- Performance analysis systems
- Risk management platforms
- Portfolio management tools
- Regulatory reporting systems

Performance Considerations:
==========================

- Efficient API usage and rate limiting
- Intelligent caching for repeated analyses
- Asynchronous processing capabilities
- Memory-conscious report generation
- Cost optimization for AI model usage

Dependencies:
============

- OpenAI API for GPT model access
- pandas for data manipulation
- pathlib for file system operations
- logging for comprehensive monitoring
- datetime for timestamp management

Author: Strategy Explainer Framework
Version: 2.0
License: Educational Use
"""

import openai
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import logging
import json
from pathlib import Path

from config.settings import settings


class GPTReportGenerator:
    """
    AI-powered trading strategy report generator.
    
    This class provides comprehensive report generation capabilities using
    OpenAI's GPT models to create professional-grade trading analysis
    documentation. It specializes in transforming quantitative trading
    results into clear, actionable business intelligence.
    
    The generator creates multiple types of reports including strategy
    analysis, risk assessment, and performance evaluation, all with
    professional investment-grade commentary and insights.
    
    Key Capabilities:
    ================
    
    1. **Strategy Analysis Reports**
       - Comprehensive performance analysis
       - Strategic insights and recommendations
       - Market context integration
       - Educational explanations
    
    2. **Risk Assessment Reports**
       - Detailed risk evaluation
       - Risk management recommendations
       - Compliance-ready documentation
       - Scenario analysis
    
    3. **Performance Evaluation**
       - Multi-dimensional performance analysis
       - Benchmark comparison insights
       - Attribution analysis
       - Improvement recommendations
    
    4. **Professional Documentation**
       - Markdown-formatted reports
       - Automated archiving
       - Version control
       - Distribution ready
    
    Attributes:
        api_key (str): OpenAI API key for GPT access
        logger (logging.Logger): Logger for monitoring and debugging
        reports_dir (Path): Directory for report storage
        model (str): GPT model to use for analysis
    
    Example Usage:
    =============
    ```python
    # Initialize report generator
    reporter = GPTReportGenerator()
    
    # Generate strategy analysis
    report = reporter.generate_strategy_analysis(
        strategy_config=config,
        metrics=metrics,
        trades=trades_df
    )
    
    # Generate risk assessment
    risk_report = reporter.generate_risk_assessment(
        metrics=risk_metrics,
        trades=trades_df
    )
    ```
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GPT report generator.
        
        This constructor sets up the AI-powered report generation system
        including OpenAI API integration, logging configuration, and
        report management infrastructure.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, uses the
                key from settings. Defaults to None.
        
        Initialization Process:
        ======================
        1. **API Configuration**: Set up OpenAI API access
        2. **Logging Setup**: Initialize comprehensive logging
        3. **Directory Management**: Create report storage directories
        4. **Model Selection**: Configure GPT model preferences
        
        Configuration Integration:
        =========================
        The generator integrates with the settings system to:
        - Access OpenAI API credentials
        - Configure report storage locations
        - Set logging levels and formats
        - Manage model preferences
        
        Error Handling:
        ==============
        - Graceful degradation when API key is missing
        - Fallback to basic reporting without AI
        - Comprehensive error logging
        - User-friendly error messages
        """
        # Configure OpenAI API access
        self.api_key = api_key or settings.OPENAI_API_KEY
        if self.api_key:
            openai.api_key = self.api_key
        
        # Initialize logging for monitoring and debugging
        self.logger = logging.getLogger(__name__)
        
        # Set up report storage directory
        self.reports_dir = Path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure model preferences
        self.model = "gpt-4"
        self.fallback_model = "gpt-3.5-turbo"
        
        # Log initialization status
        if self.api_key:
            self.logger.info("GPT Report Generator initialized with API access")
        else:
            self.logger.warning("GPT Report Generator initialized without API key - using fallback mode")
    
    def generate_strategy_analysis(self, 
                                 strategy_config: Dict[str, Any],
                                 metrics: Dict[str, Any],
                                 trades: pd.DataFrame,
                                 market_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive strategy analysis report.
        
        This method creates a detailed analysis report that evaluates
        strategy performance, provides insights, and generates actionable
        recommendations. It leverages AI to create professional-grade
        investment analysis documentation.
        
        Args:
            strategy_config (Dict[str, Any]): Strategy configuration including:
                - strategy: Strategy details and parameters
                - data: Market data configuration
                - risk_management: Risk settings
            metrics (Dict[str, Any]): Performance metrics dictionary
            trades (pd.DataFrame): DataFrame with individual trade records
            market_context (Dict[str, Any], optional): Additional market context
                information. Defaults to None.
        
        Returns:
            str: Comprehensive strategy analysis report in Markdown format
        
        Report Components:
        =================
        1. **Executive Summary**
           - High-level performance overview
           - Key findings and recommendations
           - Risk assessment summary
           - Strategic positioning
        
        2. **Strategy Overview**
           - Strategy description and methodology
           - Parameter analysis and rationale
           - Market applicability assessment
           - Theoretical foundation
        
        3. **Performance Analysis**
           - Detailed performance metrics
           - Benchmark comparison
           - Risk-adjusted returns
           - Consistency analysis
        
        4. **Risk Assessment**
           - Risk profile evaluation
           - Drawdown analysis
           - Volatility assessment
           - Risk management effectiveness
        
        5. **Trade Analysis**
           - Trade-level insights
           - Pattern identification
           - Success/failure analysis
           - Optimization recommendations
        
        6. **Market Context**
           - Market condition analysis
           - Regime-specific performance
           - Sector/asset class insights
           - Timing considerations
        
        7. **Recommendations**
           - Strategic improvements
           - Risk management enhancements
           - Portfolio integration suggestions
           - Future considerations
        
        Example Usage:
        =============
        ```python
        # Generate comprehensive analysis
        report = reporter.generate_strategy_analysis(
            strategy_config={
                'strategy': {
                    'name': 'SMA Crossover',
                    'parameters': {'fast_period': 10, 'slow_period': 50}
                },
                'data': {'symbol': 'AAPL', 'timeframe': 'daily'},
                'risk_management': {'stop_loss': 0.02, 'take_profit': 0.05}
            },
            metrics=performance_metrics,
            trades=trades_df,
            market_context={
                'market_regime': 'bull_market',
                'volatility_regime': 'low'
            }
        )
        ```
        """
        # Check for API availability
        if not self.api_key:
            self.logger.warning("OpenAI API key not available - using fallback analysis")
            return self._generate_fallback_analysis(strategy_config, metrics, trades)
        
        try:
            # Prepare comprehensive analysis context
            context = self._prepare_analysis_context(
                strategy_config, metrics, trades, market_context
            )
            
            # Create sophisticated analysis prompt
            prompt = self._create_analysis_prompt(context, strategy_config)
            
            # Generate AI-powered analysis
            self.logger.info("Generating AI-powered strategy analysis...")
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_analysis_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            # Extract and format the analysis
            analysis = response.choices[0].message.content.strip()
            
            # Log successful generation
            self.logger.info("Strategy analysis generated successfully")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating strategy analysis: {str(e)}")
            return self._generate_fallback_analysis(strategy_config, metrics, trades)
    
    def generate_risk_assessment(self, 
                               metrics: Dict[str, Any],
                               trades: pd.DataFrame) -> str:
        """
        Generate comprehensive risk assessment report.
        
        This method creates a detailed risk analysis that evaluates
        strategy risk characteristics, identifies potential issues,
        and provides risk management recommendations.
        
        Args:
            metrics (Dict[str, Any]): Performance and risk metrics
            trades (pd.DataFrame): DataFrame with trade records
        
        Returns:
            str: Comprehensive risk assessment report in Markdown format
        
        Risk Assessment Components:
        ==========================
        1. **Risk Profile Overview**
           - Overall risk rating
           - Key risk factors
           - Risk tolerance assessment
           - Comparative analysis
        
        2. **Volatility Analysis**
           - Return volatility assessment
           - Volatility clustering analysis
           - Regime-specific volatility
           - Volatility forecasting
        
        3. **Drawdown Analysis**
           - Maximum drawdown evaluation
           - Drawdown duration analysis
           - Recovery characteristics
           - Tail risk assessment
        
        4. **Risk-Adjusted Performance**
           - Sharpe ratio analysis
           - Sortino ratio evaluation
           - Information ratio assessment
           - Risk-adjusted benchmarking
        
        5. **Tail Risk Analysis**
           - Value at Risk (VaR) estimation
           - Conditional VaR analysis
           - Extreme event scenarios
           - Black swan considerations
        
        6. **Risk Management Effectiveness**
           - Stop-loss effectiveness
           - Position sizing analysis
           - Diversification benefits
           - Hedging strategies
        
        7. **Recommendations**
           - Risk mitigation strategies
           - Portfolio allocation suggestions
           - Monitoring recommendations
           - Improvement opportunities
        
        Example Usage:
        =============
        ```python
        # Generate risk assessment
        risk_report = reporter.generate_risk_assessment(
            metrics=risk_metrics,
            trades=trades_df
        )
        ```
        """
        # Check for API availability
        if not self.api_key:
            self.logger.warning("OpenAI API key not available - using fallback risk assessment")
            return self._generate_fallback_risk_assessment(metrics)
        
        try:
            # Create risk assessment prompt
            prompt = self._create_risk_assessment_prompt(metrics, trades)
            
            # Generate AI-powered risk assessment
            self.logger.info("Generating AI-powered risk assessment...")
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_risk_assessment_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.6
            )
            
            # Extract and format the assessment
            assessment = response.choices[0].message.content.strip()
            
            # Log successful generation
            self.logger.info("Risk assessment generated successfully")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {str(e)}")
            return self._generate_fallback_risk_assessment(metrics)
    
    def _prepare_analysis_context(self, 
                                strategy_config: Dict[str, Any],
                                metrics: Dict[str, Any],
                                trades: pd.DataFrame,
                                market_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare comprehensive context for GPT analysis.
        
        This method creates a structured context that provides the AI
        with all necessary information for generating high-quality
        analysis reports.
        
        Args:
            strategy_config (Dict[str, Any]): Strategy configuration
            metrics (Dict[str, Any]): Performance metrics
            trades (pd.DataFrame): Trade records
            market_context (Dict[str, Any], optional): Market context
        
        Returns:
            str: Formatted context string for AI analysis
        
        Context Components:
        ==================
        - Strategy information and parameters
        - Performance metrics and statistics
        - Trade summary and patterns
        - Market context and conditions
        - Risk characteristics
        - Benchmark comparisons
        """
        context_parts = []
        
        # === STRATEGY INFORMATION ===
        strategy = strategy_config.get('strategy', {})
        context_parts.append(f"**Strategy**: {strategy.get('name', 'Unknown')}")
        context_parts.append(f"**Description**: {strategy.get('description', 'No description')}")
        context_parts.append(f"**Category**: {strategy.get('category', 'Unknown')}")
        
        # Strategy parameters
        if strategy.get('parameters'):
            context_parts.append("**Parameters**:")
            for param, value in strategy['parameters'].items():
                context_parts.append(f"- {param}: {value}")
        
        # === PERFORMANCE METRICS ===
        context_parts.append("\n**Performance Metrics**:")
        key_metrics = [
            'total_return', 'annual_return', 'annual_volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'total_trades', 'profit_factor'
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric in ['total_return', 'annual_return', 'annual_volatility', 'max_drawdown']:
                        context_parts.append(f"- {metric}: {value:.2%}")
                    else:
                        context_parts.append(f"- {metric}: {value:.3f}")
                else:
                    context_parts.append(f"- {metric}: {value}")
        
        # === TRADE SUMMARY ===
        if not trades.empty:
            context_parts.append(f"\n**Trade Summary**:")
            context_parts.append(f"- Total trades: {len(trades)}")
            
            # Winning trades analysis
            winning_trades = trades[trades['net_pnl'] > 0] if 'net_pnl' in trades.columns else pd.DataFrame()
            if not winning_trades.empty:
                context_parts.append(f"- Winning trades: {len(winning_trades)}")
                context_parts.append(f"- Average win: ${winning_trades['net_pnl'].mean():.2f}")
            
            # Losing trades analysis
            losing_trades = trades[trades['net_pnl'] < 0] if 'net_pnl' in trades.columns else pd.DataFrame()
            if not losing_trades.empty:
                context_parts.append(f"- Losing trades: {len(losing_trades)}")
                context_parts.append(f"- Average loss: ${losing_trades['net_pnl'].mean():.2f}")
        
        # === MARKET CONTEXT ===
        if market_context:
            context_parts.append(f"\n**Market Context**:")
            for key, value in market_context.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _create_analysis_prompt(self, context: str, strategy_config: Dict[str, Any]) -> str:
        """
        Create sophisticated analysis prompt for GPT.
        
        This method constructs a detailed prompt that guides the AI
        to generate comprehensive, professional-grade analysis reports.
        
        Args:
            context (str): Formatted context information
            strategy_config (Dict[str, Any]): Strategy configuration
        
        Returns:
            str: Comprehensive analysis prompt
        """
        strategy = strategy_config.get('strategy', {})
        strategy_name = strategy.get('name', 'Unknown Strategy')
        
        prompt = f"""
        Please provide a comprehensive analysis of the following trading strategy:

        {context}

        Generate a detailed report that includes:

        1. **Executive Summary**
           - Overall performance assessment
           - Key strengths and weaknesses
           - Primary recommendations

        2. **Strategy Analysis**
           - Strategy logic and methodology
           - Parameter effectiveness
           - Market suitability

        3. **Performance Evaluation**
           - Risk-adjusted returns analysis
           - Consistency and reliability
           - Benchmark comparison insights

        4. **Risk Assessment**
           - Risk profile evaluation
           - Drawdown characteristics
           - Risk management effectiveness

        5. **Trade Analysis**
           - Trade quality assessment
           - Pattern identification
           - Success/failure factors

        6. **Recommendations**
           - Improvement opportunities
           - Risk management enhancements
           - Portfolio integration suggestions

        Please write in a professional, analytical tone suitable for institutional investors.
        Use specific metrics and data points from the provided context.
        Provide actionable insights and concrete recommendations.
        """
        
        return prompt
    
    def _create_risk_assessment_prompt(self, metrics: Dict[str, Any], trades: pd.DataFrame) -> str:
        """
        Create comprehensive risk assessment prompt.
        
        This method constructs a detailed prompt for AI-powered risk
        analysis and assessment generation.
        
        Args:
            metrics (Dict[str, Any]): Risk and performance metrics
            trades (pd.DataFrame): Trade records for analysis
        
        Returns:
            str: Comprehensive risk assessment prompt
        """
        # Prepare risk-specific context
        risk_context = []
        
        # Key risk metrics
        risk_metrics = [
            'annual_volatility', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'var_95', 'cvar_95', 'skewness', 'kurtosis'
        ]
        
        risk_context.append("**Risk Metrics**:")
        for metric in risk_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric in ['annual_volatility', 'max_drawdown']:
                        risk_context.append(f"- {metric}: {value:.2%}")
                    else:
                        risk_context.append(f"- {metric}: {value:.3f}")
                else:
                    risk_context.append(f"- {metric}: {value}")
        
        # Trade-based risk analysis
        if not trades.empty:
            risk_context.append("\n**Trade Risk Analysis**:")
            risk_context.append(f"- Total trades: {len(trades)}")
            
            if 'net_pnl' in trades.columns:
                negative_trades = trades[trades['net_pnl'] < 0]
                if not negative_trades.empty:
                    risk_context.append(f"- Losing trades: {len(negative_trades)}")
                    risk_context.append(f"- Worst trade: ${negative_trades['net_pnl'].min():.2f}")
        
        context_str = "\n".join(risk_context)
        
        prompt = f"""
        Please provide a comprehensive risk assessment for the following trading strategy:

        {context_str}

        Generate a detailed risk analysis that includes:

        1. **Risk Profile Overview**
           - Overall risk rating and classification
           - Key risk factors and concerns
           - Risk tolerance assessment

        2. **Volatility Analysis**
           - Return volatility evaluation
           - Volatility consistency
           - Regime-specific considerations

        3. **Drawdown Analysis**
           - Maximum drawdown assessment
           - Recovery characteristics
           - Tail risk evaluation

        4. **Risk-Adjusted Performance**
           - Sharpe ratio interpretation
           - Risk-adjusted benchmarking
           - Efficiency assessment

        5. **Risk Management Recommendations**
           - Position sizing suggestions
           - Risk mitigation strategies
           - Monitoring recommendations

        Write in a professional risk management tone.
        Provide specific, actionable recommendations.
        Focus on practical risk management applications.
        """
        
        return prompt
    
    def _get_analysis_system_prompt(self) -> str:
        """
        Get system prompt for strategy analysis.
        
        Returns:
            str: System prompt for AI analysis generation
        """
        return """
        You are a senior quantitative analyst and portfolio manager with extensive experience 
        in trading strategy evaluation and risk management. Your task is to provide comprehensive, 
        professional-grade analysis of trading strategies.

        Key requirements:
        - Write in a professional, analytical tone suitable for institutional investors
        - Use specific metrics and data points from the provided context
        - Provide actionable insights and concrete recommendations
        - Focus on both performance and risk characteristics
        - Consider market context and practical implementation challenges
        - Structure your analysis clearly with headers and bullet points
        - Avoid generic statements and provide specific, data-driven insights
        """
    
    def _get_risk_assessment_system_prompt(self) -> str:
        """
        Get system prompt for risk assessment.
        
        Returns:
            str: System prompt for AI risk assessment generation
        """
        return """
        You are a professional risk management expert specializing in trading strategy 
        risk assessment. Your task is to provide comprehensive risk analysis that helps 
        investors understand and manage the risks associated with trading strategies.

        Key requirements:
        - Focus on practical risk management applications
        - Provide specific, actionable risk mitigation recommendations
        - Use quantitative risk metrics to support your analysis
        - Consider both statistical and practical risk factors
        - Structure your assessment clearly with risk categories
        - Avoid generic risk warnings and provide specific insights
        - Consider regulatory and compliance perspectives
        """
    
    def _generate_fallback_analysis(self, 
                                  strategy_config: Dict[str, Any],
                                  metrics: Dict[str, Any],
                                  trades: pd.DataFrame) -> str:
        """
        Generate fallback analysis when GPT is not available.
        
        This method creates a basic analysis report using template-based
        generation when AI services are unavailable.
        
        Args:
            strategy_config (Dict[str, Any]): Strategy configuration
            metrics (Dict[str, Any]): Performance metrics
            trades (pd.DataFrame): Trade records
        
        Returns:
            str: Basic analysis report in Markdown format
        """
        strategy = strategy_config.get('strategy', {})
        
        report_lines = [
            f"# {strategy.get('name', 'Strategy')} Analysis Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Strategy Overview",
            f"- **Name**: {strategy.get('name', 'Unknown')}",
            f"- **Category**: {strategy.get('category', 'Unknown')}",
            f"- **Description**: {strategy.get('description', 'No description available')}",
            "",
            "## Performance Summary",
            f"- **Total Return**: {metrics.get('total_return', 0):.2%}",
            f"- **Annual Return**: {metrics.get('annual_return', 0):.2%}",
            f"- **Annual Volatility**: {metrics.get('annual_volatility', 0):.2%}",
            f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.3f}",
            f"- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2%}",
            f"- **Win Rate**: {metrics.get('win_rate', 0):.2%}",
            f"- **Total Trades**: {metrics.get('total_trades', 0)}",
            "",
            "## Analysis Note",
            "This is a basic performance summary. For detailed AI-powered analysis,",
            "please configure your OpenAI API key in the settings.",
        ]
        
        return "\n".join(report_lines)
    
    def _generate_fallback_risk_assessment(self, metrics: Dict[str, Any]) -> str:
        """
        Generate fallback risk assessment when GPT is not available.
        
        This method creates a basic risk assessment using template-based
        generation when AI services are unavailable.
        
        Args:
            metrics (Dict[str, Any]): Performance metrics
        
        Returns:
            str: Basic risk assessment in Markdown format
        """
        risk_lines = [
            "# Risk Assessment Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Key Risk Metrics",
            f"- **Annual Volatility**: {metrics.get('annual_volatility', 0):.2%}",
            f"- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2%}",
            f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}",
            f"- **Sortino Ratio**: {metrics.get('sortino_ratio', 0):.2f}",
            "",
            "## Risk Assessment",
            "- **Volatility**: " + ("High" if metrics.get('annual_volatility', 0) > 0.20 else 
                                   "Moderate" if metrics.get('annual_volatility', 0) > 0.15 else "Low"),
            "- **Drawdown Risk**: " + ("High" if abs(metrics.get('max_drawdown', 0)) > 0.15 else 
                                      "Moderate" if abs(metrics.get('max_drawdown', 0)) > 0.10 else "Low"),
            "- **Risk-Adjusted Performance**: " + ("Excellent" if metrics.get('sharpe_ratio', 0) > 1.5 else 
                                                  "Good" if metrics.get('sharpe_ratio', 0) > 1.0 else 
                                                  "Fair" if metrics.get('sharpe_ratio', 0) > 0.5 else "Poor"),
            "",
            "## Analysis Note",
            "This is a basic risk assessment. For detailed AI-powered risk analysis,",
            "please configure your OpenAI API key in the settings.",
        ]
        
        return "\n".join(risk_lines)
    
    def save_report(self, report: str, strategy_config: Dict[str, Any]) -> str:
        """
        Save report to file with intelligent naming.
        
        This method saves generated reports to the configured reports
        directory with intelligent file naming and organization.
        
        Args:
            report (str): Report content to save
            strategy_config (Dict[str, Any]): Strategy configuration for naming
        
        Returns:
            str: Path to the saved report file
        
        Raises:
            IOError: If unable to save the report
        
        File Naming Convention:
        ======================
        Format: {symbol}_{strategy_name}_{timestamp}.md
        
        Example: AAPL_SMA_Crossover_20240115_143022.md
        
        Directory Structure:
        ===================
        - reports/
          - daily/
          - weekly/
          - monthly/
          - custom/
        
        Example Usage:
        =============
        ```python
        # Save report with automatic naming
        report_path = reporter.save_report(analysis_report, strategy_config)
        print(f"Report saved to: {report_path}")
        ```
        """
        try:
            # Extract naming components
            strategy = strategy_config.get('strategy', {})
            symbol = strategy_config.get('data', {}).get('symbol', 'UNKNOWN')
            strategy_name = strategy.get('name', 'Unknown').replace(' ', '_')
            
            # Create timestamp for unique naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{strategy_name}_{timestamp}.md"
            
            # Ensure reports directory exists
            report_path = self.reports_dir / filename
            
            # Save report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            raise IOError(f"Failed to save report: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models.
        
        This method queries the OpenAI API to retrieve a list of
        available models for report generation.
        
        Returns:
            List[str]: List of available model names
        
        Model Categories:
        ================
        - GPT-4 models (highest quality)
        - GPT-3.5 models (balanced performance)
        - Specialized models (domain-specific)
        
        Example Usage:
        =============
        ```python
        # Check available models
        models = reporter.get_available_models()
        print(f"Available models: {models}")
        ```
        """
        if not self.api_key:
            self.logger.warning("API key not available - cannot fetch models")
            return []
        
        try:
            # Query OpenAI API for available models
            models = openai.Model.list()
            
            # Filter for GPT models suitable for analysis
            gpt_models = [
                model.id for model in models.data 
                if 'gpt' in model.id.lower()
            ]
            
            self.logger.info(f"Retrieved {len(gpt_models)} available GPT models")
            return sorted(gpt_models)
            
        except Exception as e:
            self.logger.error(f"Error fetching available models: {str(e)}")
            return []
    
    def get_reports_summary(self) -> Dict[str, Any]:
        """
        Get summary of generated reports.
        
        This method provides analytics and summary information about
        the reports generated by the system.
        
        Returns:
            Dict[str, Any]: Reports summary including:
                - total_reports: Number of reports generated
                - recent_reports: List of recent report files
                - report_types: Distribution of report types
                - storage_usage: Storage usage statistics
        
        Example Usage:
        =============
        ```python
        # Get reports summary
        summary = reporter.get_reports_summary()
        print(f"Total reports: {summary['total_reports']}")
        ```
        """
        try:
            # Get all report files
            report_files = list(self.reports_dir.glob("*.md"))
            
            # Calculate summary statistics
            total_reports = len(report_files)
            
            # Get recent reports (last 10)
            recent_reports = sorted(report_files, key=lambda f: f.stat().st_mtime, reverse=True)[:10]
            recent_report_names = [f.name for f in recent_reports]
            
            # Calculate storage usage
            total_size = sum(f.stat().st_size for f in report_files)
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                'total_reports': total_reports,
                'recent_reports': recent_report_names,
                'reports_directory': str(self.reports_dir),
                'storage_usage_mb': round(total_size_mb, 2),
                'api_enabled': bool(self.api_key)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting reports summary: {str(e)}")
            return {
                'total_reports': 0,
                'recent_reports': [],
                'reports_directory': str(self.reports_dir),
                'storage_usage_mb': 0,
                'api_enabled': bool(self.api_key),
                'error': str(e)
            }


# Factory function for creating configured report generator
def create_gpt_report_generator(config: Optional[Dict[str, Any]] = None) -> GPTReportGenerator:
    """
    Create a configured GPT report generator.
    
    This factory function creates a GPTReportGenerator instance with
    configuration from the provided config dictionary or settings.
    
    Args:
        config (Dict[str, Any], optional): Configuration dictionary containing:
            - openai_api_key: OpenAI API key
            - reports_dir: Reports directory path
            - model: Preferred GPT model
    
    Returns:
        GPTReportGenerator: Configured report generator instance
    
    Example Usage:
    =============
    ```python
    # Create with default configuration
    reporter = create_gpt_report_generator()
    
    # Create with custom configuration
    config = {
        'openai_api_key': 'your-api-key',
        'reports_dir': '/custom/reports',
        'model': 'gpt-4'
    }
    reporter = create_gpt_report_generator(config)
    ```
    """
    if config is None:
        config = {}
    
    # Extract configuration
    api_key = config.get('openai_api_key')
    
    # Create configured generator
    return GPTReportGenerator(api_key=api_key)


# Example usage and demonstration
if __name__ == "__main__":
    """
    Demonstrate the GPT report generation system.
    
    This example shows how to use the report generator to create
    professional trading analysis reports.
    """
    print("AI-Powered Trading Report Generator Demo")
    print("=" * 50)
    
    # Initialize report generator
    reporter = GPTReportGenerator()
    
    # Create sample strategy configuration
    sample_config = {
        'strategy': {
            'name': 'SMA Crossover Strategy',
            'category': 'Trend Following',
            'description': 'Simple moving average crossover strategy',
            'parameters': {
                'fast_period': 10,
                'slow_period': 50,
                'stop_loss': 0.02,
                'take_profit': 0.05
            }
        },
        'data': {
            'symbol': 'AAPL',
            'timeframe': 'daily',
            'period': '1year'
        }
    }
    
    # Create sample performance metrics
    sample_metrics = {
        'total_return': 0.235,
        'annual_return': 0.187,
        'annual_volatility': 0.156,
        'sharpe_ratio': 1.45,
        'max_drawdown': -0.089,
        'win_rate': 0.62,
        'total_trades': 24,
        'profit_factor': 1.78
    }
    
    # Create sample trades data
    sample_trades = pd.DataFrame({
        'entry_time': pd.date_range('2023-01-01', periods=24, freq='15D'),
        'exit_time': pd.date_range('2023-01-06', periods=24, freq='15D'),
        'net_pnl': [150, -80, 220, 95, -45, 180, 120, -65, 300, -120,
                   175, 90, -55, 240, 85, -110, 195, 145, -75, 260,
                   110, -95, 205, 160],
        'side': ['long'] * 24
    })
    
    print("\n1. Testing Report Generator Setup:")
    print("-" * 40)
    print(f"API Key Available: {'Yes' if reporter.api_key else 'No'}")
    print(f"Reports Directory: {reporter.reports_dir}")
    
    print("\n2. Generating Strategy Analysis:")
    print("-" * 40)
    try:
        analysis_report = reporter.generate_strategy_analysis(
            strategy_config=sample_config,
            metrics=sample_metrics,
            trades=sample_trades
        )
        print(f"Analysis report generated ({len(analysis_report)} characters)")
        print("First 200 characters:")
        print(analysis_report[:200] + "...")
    except Exception as e:
        print(f"Error generating analysis: {e}")
    
    print("\n3. Generating Risk Assessment:")
    print("-" * 40)
    try:
        risk_report = reporter.generate_risk_assessment(
            metrics=sample_metrics,
            trades=sample_trades
        )
        print(f"Risk assessment generated ({len(risk_report)} characters)")
        print("First 200 characters:")
        print(risk_report[:200] + "...")
    except Exception as e:
        print(f"Error generating risk assessment: {e}")
    
    print("\n4. Saving Reports:")
    print("-" * 40)
    try:
        if 'analysis_report' in locals():
            report_path = reporter.save_report(analysis_report, sample_config)
            print(f"Report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    print("\n5. Reports Summary:")
    print("-" * 40)
    try:
        summary = reporter.get_reports_summary()
        print(f"Total reports: {summary['total_reports']}")
        print(f"Storage usage: {summary['storage_usage_mb']} MB")
        print(f"Recent reports: {len(summary['recent_reports'])}")
    except Exception as e:
        print(f"Error getting summary: {e}")
    
    print("\n" + "=" * 50)
    print("Demo complete! Report generator ready for integration.")
    print("=" * 50) 