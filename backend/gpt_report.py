"""GPT-powered analysis and report generation."""

import openai
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import logging
import json
from pathlib import Path

from config.settings import settings


class GPTReportGenerator:
    """Generate AI-powered analysis reports using GPT."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT report generator.
        
        Args:
            api_key: OpenAI API key. Uses settings if not provided.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if self.api_key:
            openai.api_key = self.api_key
        
        self.logger = logging.getLogger(__name__)
        self.reports_dir = Path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_strategy_analysis(self, 
                                 strategy_config: Dict[str, Any],
                                 metrics: Dict[str, Any],
                                 trades: pd.DataFrame,
                                 market_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive strategy analysis report.
        
        Args:
            strategy_config: Strategy configuration and parameters
            metrics: Performance metrics dictionary
            trades: DataFrame with trade records
            market_context: Optional market context information
            
        Returns:
            Generated analysis report as markdown string
        """
        if not self.api_key:
            return self._generate_fallback_report(strategy_config, metrics, trades)
        
        try:
            # Prepare analysis context
            context = self._prepare_analysis_context(
                strategy_config, metrics, trades, market_context
            )
            
            # Generate report using GPT
            report = self._call_gpt_analysis(context)
            
            # Save report
            if settings.DEBUG:
                self._save_report(report, strategy_config)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating GPT analysis: {str(e)}")
            return self._generate_fallback_report(strategy_config, metrics, trades)
    
    def generate_strategy_explanation(self, strategy_config: Dict[str, Any]) -> str:
        """Generate educational explanation of the strategy.
        
        Args:
            strategy_config: Strategy configuration and parameters
            
        Returns:
            Strategy explanation as markdown string
        """
        if not self.api_key:
            return self._generate_fallback_explanation(strategy_config)
        
        try:
            prompt = self._create_explanation_prompt(strategy_config)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial education expert who explains trading strategies in clear, accessible language."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating strategy explanation: {str(e)}")
            return self._generate_fallback_explanation(strategy_config)
    
    def generate_risk_assessment(self, 
                               metrics: Dict[str, Any],
                               trades: pd.DataFrame) -> str:
        """Generate risk assessment and recommendations.
        
        Args:
            metrics: Performance metrics dictionary
            trades: DataFrame with trade records
            
        Returns:
            Risk assessment as markdown string
        """
        if not self.api_key:
            return self._generate_fallback_risk_assessment(metrics)
        
        try:
            prompt = self._create_risk_assessment_prompt(metrics, trades)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a risk management expert who provides clear, actionable risk assessments for trading strategies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {str(e)}")
            return self._generate_fallback_risk_assessment(metrics)
    
    def _prepare_analysis_context(self, 
                                strategy_config: Dict[str, Any],
                                metrics: Dict[str, Any],
                                trades: pd.DataFrame,
                                market_context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare context for GPT analysis.
        
        Args:
            strategy_config: Strategy configuration
            metrics: Performance metrics
            trades: Trade records
            market_context: Market context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Strategy information
        strategy = strategy_config.get('strategy', {})
        context_parts.append(f"**Strategy**: {strategy.get('name', 'Unknown')}")
        context_parts.append(f"**Description**: {strategy.get('description', 'No description')}")
        context_parts.append(f"**Category**: {strategy.get('category', 'Unknown')}")
        
        # Parameters
        if strategy.get('parameters'):
            context_parts.append("**Parameters**:")
            for param, value in strategy['parameters'].items():
                context_parts.append(f"- {param}: {value}")
        
        # Key metrics
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
        
        # Trade summary
        if not trades.empty:
            context_parts.append(f"\n**Trade Summary**:")
            context_parts.append(f"- Total trades: {len(trades)}")
            winning_trades = trades[trades['net_pnl'] > 0] if 'net_pnl' in trades.columns else pd.DataFrame()
            if not winning_trades.empty:
                context_parts.append(f"- Winning trades: {len(winning_trades)}")
                context_parts.append(f"- Average win: ${winning_trades['net_pnl'].mean():.2f}")
            
            losing_trades = trades[trades['net_pnl'] < 0] if 'net_pnl' in trades.columns else pd.DataFrame()
            if not losing_trades.empty:
                context_parts.append(f"- Losing trades: {len(losing_trades)}")
                context_parts.append(f"- Average loss: ${losing_trades['net_pnl'].mean():.2f}")
        
        # Market context
        if market_context:
            context_parts.append(f"\n**Market Context**:")
            for key, value in market_context.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _call_gpt_analysis(self, context: str) -> str:
        """Call GPT API for strategy analysis.
        
        Args:
            context: Analysis context
            
        Returns:
            Generated analysis report
        """
        prompt = f"""
        Analyze the following trading strategy performance and provide a comprehensive report:

        {context}

        Please provide a detailed analysis covering:
        1. **Executive Summary** - Brief overview of performance
        2. **Strategy Performance** - Detailed performance analysis
        3. **Risk Analysis** - Risk characteristics and concerns
        4. **Strengths** - What the strategy does well
        5. **Weaknesses** - Areas of concern or limitation
        6. **Market Conditions** - How it might perform in different markets
        7. **Recommendations** - Suggestions for improvement or usage

        Format the response in clear markdown with appropriate headers and bullet points.
        Be specific, insightful, and practical in your analysis.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert quantitative analyst and trading strategist with deep experience in strategy evaluation and risk management."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def _create_explanation_prompt(self, strategy_config: Dict[str, Any]) -> str:
        """Create prompt for strategy explanation.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            Formatted prompt string
        """
        strategy = strategy_config.get('strategy', {})
        
        return f"""
        Explain the following trading strategy in educational terms:

        **Strategy**: {strategy.get('name', 'Unknown')}
        **Type**: {strategy.get('category', 'Unknown')}
        **Description**: {strategy.get('description', 'No description')}

        Please explain:
        1. **What it is** - Basic concept and approach
        2. **How it works** - The mechanics and logic
        3. **When to use it** - Market conditions where it's effective
        4. **Key parameters** - Important settings and their impact
        5. **Pros and cons** - Advantages and limitations
        6. **Examples** - Simple examples of how it would trade

        Make it accessible to someone learning about trading strategies.
        Use clear language and avoid excessive jargon.
        """
    
    def _create_risk_assessment_prompt(self, 
                                     metrics: Dict[str, Any],
                                     trades: pd.DataFrame) -> str:
        """Create prompt for risk assessment.
        
        Args:
            metrics: Performance metrics
            trades: Trade records
            
        Returns:
            Formatted prompt string
        """
        context = []
        
        # Key risk metrics
        risk_metrics = [
            'annual_volatility', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
            'var_95', 'downside_deviation', 'win_rate'
        ]
        
        for metric in risk_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric in ['annual_volatility', 'max_drawdown', 'downside_deviation']:
                        context.append(f"- {metric}: {value:.2%}")
                    else:
                        context.append(f"- {metric}: {value:.3f}")
        
        context_str = "\n".join(context)
        
        return f"""
        Assess the risk characteristics of this trading strategy:

        **Risk Metrics**:
        {context_str}

        **Total Trades**: {len(trades) if not trades.empty else 0}

        Please provide:
        1. **Risk Level** - Overall risk assessment (Low/Medium/High)
        2. **Key Risks** - Primary risk factors and concerns
        3. **Risk-Adjusted Performance** - How well it compensates for risk
        4. **Suitability** - What type of investor this suits
        5. **Risk Management** - Suggestions for managing risks
        6. **Red Flags** - Any concerning patterns or metrics

        Be specific and actionable in your assessment.
        """
    
    def _generate_fallback_report(self, 
                                strategy_config: Dict[str, Any],
                                metrics: Dict[str, Any],
                                trades: pd.DataFrame) -> str:
        """Generate fallback report when GPT is not available.
        
        Args:
            strategy_config: Strategy configuration
            metrics: Performance metrics
            trades: Trade records
            
        Returns:
            Basic analysis report
        """
        strategy = strategy_config.get('strategy', {})
        
        report_lines = [
            f"# Strategy Analysis: {strategy.get('name', 'Unknown')}",
            "",
            "## Executive Summary",
            f"Strategy performance analysis for {strategy.get('name', 'Unknown Strategy')}.",
            "",
            "## Key Metrics",
            f"- Total Return: {metrics.get('total_return', 0):.2%}",
            f"- Annual Return: {metrics.get('annual_return', 0):.2%}",
            f"- Volatility: {metrics.get('annual_volatility', 0):.2%}",
            f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"- Win Rate: {metrics.get('win_rate', 0):.2%}",
            f"- Total Trades: {metrics.get('total_trades', 0)}",
            "",
            "## Analysis Note",
            "This is a basic performance summary. For detailed AI-powered analysis,",
            "please configure your OpenAI API key in the settings.",
        ]
        
        return "\n".join(report_lines)
    
    def _generate_fallback_explanation(self, strategy_config: Dict[str, Any]) -> str:
        """Generate fallback explanation when GPT is not available.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            Basic strategy explanation
        """
        strategy = strategy_config.get('strategy', {})
        
        explanation_lines = [
            f"# {strategy.get('name', 'Strategy')} Explanation",
            "",
            f"**Category**: {strategy.get('category', 'Unknown')}",
            "",
            f"**Description**: {strategy.get('description', 'No description available')}",
            "",
            "## Note",
            "For detailed strategy explanations powered by AI,",
            "please configure your OpenAI API key in the settings.",
        ]
        
        return "\n".join(explanation_lines)
    
    def _generate_fallback_risk_assessment(self, metrics: Dict[str, Any]) -> str:
        """Generate fallback risk assessment when GPT is not available.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Basic risk assessment
        """
        risk_lines = [
            "# Risk Assessment",
            "",
            "## Key Risk Metrics",
            f"- Volatility: {metrics.get('annual_volatility', 0):.2%}",
            f"- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            "",
            "## Note",
            "For detailed AI-powered risk analysis,",
            "please configure your OpenAI API key in the settings.",
        ]
        
        return "\n".join(risk_lines)
    
    def _save_report(self, report: str, strategy_config: Dict[str, Any]) -> None:
        """Save report to file.
        
        Args:
            report: Report content
            strategy_config: Strategy configuration
        """
        try:
            strategy = strategy_config.get('strategy', {})
            symbol = strategy_config.get('data', {}).get('symbol', 'UNKNOWN')
            strategy_name = strategy.get('name', 'Unknown').replace(' ', '_')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{strategy_name}_{timestamp}.md"
            
            report_path = self.reports_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models.
        
        Returns:
            List of model names
        """
        if not self.api_key:
            return []
        
        try:
            models = openai.Model.list()
            return [model.id for model in models.data if 'gpt' in model.id]
        except Exception as e:
            self.logger.error(f"Error fetching models: {str(e)}")
            return [] 