"""GPT-powered trade analysis and strategy explanation service."""

import openai
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeContext:
    """Context for a single trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    shares: int
    net_pnl: float
    return_pct: float
    duration_days: int
    exit_reason: str
    strategy_name: str


class GPTService:
    """AI-powered trading analysis and commentary service."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """Initialize GPT service.
        
        Args:
            api_key: OpenAI API key (uses env OPENAI_API_KEY if None)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            logger.warning("No OpenAI API key found. GPT features will be disabled.")
            self.enabled = False
        else:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            self.enabled = True
            logger.info(f"GPT service initialized with model: {model}")
    
    def _call_gpt(self, prompt: str, max_tokens: int = 200) -> str:
        """Make API call to GPT with error handling."""
        if not self.enabled:
            return "GPT service not available (no API key)"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional quantitative analyst explaining trading strategies and decisions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            return f"Analysis unavailable: {str(e)}"
    
    def generate_strategy_overview(self, 
                                 strategy_name: str,
                                 parameters: Dict[str, Any],
                                 symbol: str = "the market") -> str:
        """Generate AI explanation of strategy approach."""
        
        # Format parameters nicely
        param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
        
        prompt = f"""
        Explain this trading strategy in 2-3 sentences for a sophisticated investor:
        
        Strategy: {strategy_name}
        Parameters: {param_str}
        Target: {symbol}
        
        Focus on:
        1. What market conditions this strategy exploits
        2. How the parameters affect risk/reward
        3. Why this approach might generate alpha
        
        Write in a professional, confident tone. No generic disclaimers.
        """
        
        return self._call_gpt(prompt, max_tokens=150)
    
    def explain_trade_decision(self, trade: TradeContext, market_data: Dict = None) -> str:
        """Generate AI explanation for why a specific trade was taken."""
        
        # Calculate trade quality metrics
        is_winner = trade.net_pnl > 0
        trade_quality = "winning" if is_winner else "losing"
        magnitude = abs(trade.return_pct * 100)
        
        # Add market context if available
        market_context = ""
        if market_data:
            market_context = f"Market context: {market_data.get('context', '')}"
        
        prompt = f"""
        Analyze this {trade_quality} {trade.strategy_name} trade:
        
        {trade.symbol} {trade.side} position:
        • Entry: ${trade.entry_price:.2f} on {trade.entry_time.strftime('%Y-%m-%d')}
        • Exit: ${trade.exit_price:.2f} on {trade.exit_time.strftime('%Y-%m-%d')} ({trade.exit_reason})
        • P&L: ${trade.net_pnl:.2f} ({trade.return_pct*100:.1f}%)
        • Duration: {trade.duration_days} days
        
        {market_context}
        
        Explain in 2-3 sentences:
        1. Why the strategy triggered this trade entry
        2. What happened during the trade
        3. Key lesson from this outcome
        
        Be specific and analytical, not generic.
        """
        
        return self._call_gpt(prompt, max_tokens=200)
    
    def summarize_performance(self, 
                            metrics: Dict[str, Any],
                            strategy_name: str,
                            symbol: str,
                            benchmark_comparison: Dict = None) -> str:
        """Generate AI summary of overall strategy performance."""
        
        # Extract key metrics
        total_return = metrics.get('Total Return', 0) * 100
        sharpe = metrics.get('Sharpe', 0)
        max_dd = metrics.get('Max Drawdown', 0) * 100
        win_rate = metrics.get('Win Rate', 0) * 100
        total_trades = metrics.get('Total Trades', 0)
        
        # Format benchmark comparison
        vs_benchmark = ""
        if benchmark_comparison:
            benchmark_return = benchmark_comparison.get('benchmark_return', 0) * 100
            alpha = total_return - benchmark_return
            vs_benchmark = f"vs benchmark return of {benchmark_return:.1f}% (alpha: {alpha:+.1f}%)"
        
        prompt = f"""
        Write a professional performance summary for this trading strategy:
        
        Strategy: {strategy_name} on {symbol}
        Results:
        • Total Return: {total_return:.1f}% {vs_benchmark}
        • Sharpe Ratio: {sharpe:.2f}
        • Max Drawdown: {max_dd:.1f}%
        • Win Rate: {win_rate:.1f}% ({int(total_trades)} total trades)
        
        Provide a 3-4 sentence analysis covering:
        1. Overall assessment of risk-adjusted performance
        2. Key strengths and weaknesses
        3. Whether this strategy shows genuine edge
        
        Write like a hedge fund PM reviewing results. Be honest about limitations.
        """
        
        return self._call_gpt(prompt, max_tokens=250)
    
    def generate_market_regime_insight(self, 
                                     regime_data: Dict[str, Any],
                                     strategy_performance: Dict[str, float]) -> str:
        """Generate insights about strategy performance in different market regimes."""
        
        prompt = f"""
        Analyze how this strategy performed across market regimes:
        
        Regime Performance:
        {regime_data}
        
        Strategy Returns by Regime:
        {strategy_performance}
        
        Explain in 2-3 sentences:
        1. Which market conditions favor this strategy
        2. When the strategy struggles and why
        3. Risk management implications
        
        Focus on actionable insights for position sizing and timing.
        """
        
        return self._call_gpt(prompt, max_tokens=200)
    
    def explain_signal_logic(self, 
                           signal_data: Dict[str, Any],
                           strategy_name: str) -> str:
        """Explain why a signal was generated at a specific time."""
        
        prompt = f"""
        Explain why the {strategy_name} strategy generated a signal:
        
        Signal Data: {signal_data}
        
        Break down:
        1. Which indicators triggered the signal
        2. What market pattern the strategy detected
        3. Why this setup has edge potential
        
        Keep it concise and technical. 2-3 sentences max.
        """
        
        return self._call_gpt(prompt, max_tokens=150)
    
    def generate_risk_assessment(self, 
                               portfolio_metrics: Dict[str, Any],
                               recent_trades: List[TradeContext]) -> str:
        """Generate AI-powered risk assessment of current strategy performance."""
        
        # Analyze recent trade pattern
        recent_pnl = [trade.net_pnl for trade in recent_trades[-10:]]  # Last 10 trades
        recent_winners = sum(1 for pnl in recent_pnl if pnl > 0)
        recent_win_rate = recent_winners / len(recent_pnl) if recent_pnl else 0
        
        drawdown = portfolio_metrics.get('Current Drawdown', 0) * 100
        
        prompt = f"""
        Assess the current risk profile of this trading strategy:
        
        Portfolio Metrics:
        • Current Drawdown: {drawdown:.1f}%
        • Recent Win Rate: {recent_win_rate*100:.1f}% (last {len(recent_pnl)} trades)
        • Recent P&L: {sum(recent_pnl):.2f}
        
        Risk Assessment Focus:
        1. Is the strategy in a normal drawdown or showing degradation?
        2. Are recent trades following expected patterns?
        3. Recommended position sizing adjustments
        
        Provide actionable risk management guidance in 3-4 sentences.
        """
        
        return self._call_gpt(prompt, max_tokens=250)


def create_trade_context_from_log(trade_log_row: Dict, strategy_name: str) -> TradeContext:
    """Convert trade log dictionary to TradeContext object."""
    return TradeContext(
        entry_time=trade_log_row.get('entry_time'),
        exit_time=trade_log_row.get('exit_time'),
        symbol=trade_log_row.get('symbol', 'UNKNOWN'),
        side=trade_log_row.get('side', 'long'),
        entry_price=trade_log_row.get('entry_price', 0),
        exit_price=trade_log_row.get('exit_price', 0),
        shares=int(trade_log_row.get('shares', 0)),
        net_pnl=trade_log_row.get('net_pnl', 0),
        return_pct=trade_log_row.get('return', 0),
        duration_days=trade_log_row.get('duration', 0),
        exit_reason=trade_log_row.get('exit_reason', 'signal'),
        strategy_name=strategy_name
    )


# Example usage
if __name__ == "__main__":
    # Demo the GPT service
    gpt = GPTService()
    
    # Test strategy overview
    overview = gpt.generate_strategy_overview(
        "SMA Crossover",
        {"fast_period": 10, "slow_period": 50},
        "AAPL"
    )
    print("Strategy Overview:")
    print(overview)
    print("\n" + "="*50 + "\n")
    
    # Test trade explanation
    sample_trade = TradeContext(
        entry_time=datetime(2024, 1, 15),
        exit_time=datetime(2024, 1, 22),
        symbol="AAPL",
        side="long",
        entry_price=185.50,
        exit_price=192.30,
        shares=100,
        net_pnl=680.0,
        return_pct=0.037,
        duration_days=7,
        exit_reason="take_profit",
        strategy_name="SMA Crossover"
    )
    
    trade_explanation = gpt.explain_trade_decision(sample_trade)
    print("Trade Explanation:")
    print(trade_explanation)