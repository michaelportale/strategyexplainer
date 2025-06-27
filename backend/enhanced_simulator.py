"""Enhanced trading simulator with GPT commentary integration."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import sys
import os

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulate import TradingSimulator
from backend.gpt_service import GPTService, TradeContext, create_trade_context_from_log

logger = logging.getLogger(__name__)


class GPTEnhancedSimulator(TradingSimulator):
    """Trading simulator with AI-powered trade explanations."""
    
    def __init__(self, 
                 initial_capital,
                 commission=0.0,
                 slippage=0.0,
                 margin_requirement=1.0,
                 stop_loss_pct: float = None,
                 take_profit_pct: float = None,
                 max_drawdown_pct: float = None,
                 position_size_pct: float = 1.0,
                 enable_gpt: bool = True,
                 strategy_name: str = "Unknown Strategy"):
        """Initialize enhanced simulator with GPT capabilities."""
        
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            margin_requirement=margin_requirement,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_drawdown_pct=max_drawdown_pct,
            position_size_pct=position_size_pct
        )
        
        self.strategy_name = strategy_name
        self.enable_gpt = enable_gpt
        self.gpt_service = GPTService() if enable_gpt else None
        self.trade_explanations = []  # Store GPT explanations
        
        if enable_gpt and not self.gpt_service.enabled:
            logger.warning("GPT service disabled - no API key found")
            self.enable_gpt = False
    
    def _close_position_with_gpt(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Close position and generate GPT explanation."""
        if self.position == 0 or self.current_trade is None:
            return
        
        # Store original trade data before closing
        original_trade = self.current_trade.copy()
        original_position = self.position
        original_shares = self.shares
        
        # Close position using parent method
        super()._close_position(timestamp, price, reason)
        
        # Generate GPT explanation for the completed trade
        if self.enable_gpt and self.trades:
            try:
                # Get the just-completed trade
                completed_trade = self.trades[-1]
                
                # Create trade context
                trade_context = TradeContext(
                    entry_time=completed_trade['entry_time'],
                    exit_time=completed_trade['exit_time'],
                    symbol=getattr(self, 'current_symbol', 'UNKNOWN'),
                    side=completed_trade['side'],
                    entry_price=completed_trade['entry_price'],
                    exit_price=completed_trade['exit_price'],
                    shares=completed_trade['shares'],
                    net_pnl=completed_trade['net_pnl'],
                    return_pct=completed_trade['return'],
                    duration_days=completed_trade['duration'],
                    exit_reason=reason,
                    strategy_name=self.strategy_name
                )
                
                # Generate AI explanation
                explanation = self.gpt_service.explain_trade_decision(trade_context)
                
                # Store explanation
                self.trade_explanations.append({
                    'trade_index': len(self.trades) - 1,
                    'entry_time': completed_trade['entry_time'],
                    'exit_time': completed_trade['exit_time'],
                    'explanation': explanation,
                    'trade_id': f"{completed_trade['side']}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                })
                
                logger.info(f"Generated GPT explanation for {completed_trade['side']} trade")
                
            except Exception as e:
                logger.error(f"Failed to generate trade explanation: {e}")
                self.trade_explanations.append({
                    'trade_index': len(self.trades) - 1,
                    'explanation': f"Explanation generation failed: {str(e)}",
                    'trade_id': f"error_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                })
    
    def simulate_strategy(self, 
                         price_data: pd.DataFrame, 
                         signals: pd.DataFrame,
                         symbol: str = "UNKNOWN") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate strategy with GPT-enhanced trade explanations."""
        
        # Store symbol for GPT context
        self.current_symbol = symbol
        
        # Override the _close_position method temporarily
        original_close = self._close_position
        self._close_position = self._close_position_with_gpt
        
        try:
            # Run simulation using parent method
            equity_curve, trades_df = super().simulate_strategy(price_data, signals)
            
            # Add GPT explanations to trades dataframe
            if self.trade_explanations:
                explanations_df = pd.DataFrame(self.trade_explanations)
                
                # Merge explanations with trades
                trades_df = trades_df.reset_index(drop=True)
                trades_df['trade_index'] = trades_df.index
                
                # Add explanation column
                trades_df['gpt_explanation'] = trades_df['trade_index'].map(
                    {exp['trade_index']: exp['explanation'] for exp in self.trade_explanations}
                )
                
                # Fill missing explanations
                trades_df['gpt_explanation'] = trades_df['gpt_explanation'].fillna(
                    "No explanation generated"
                )
            else:
                trades_df['gpt_explanation'] = "GPT disabled or no trades completed"
            
            return equity_curve, trades_df
            
        finally:
            # Restore original method
            self._close_position = original_close
    
    def generate_strategy_overview(self, parameters: Dict[str, Any]) -> str:
        """Generate AI overview of the strategy being used."""
        if not self.enable_gpt:
            return f"Strategy: {self.strategy_name} with parameters: {parameters}"
        
        try:
            return self.gpt_service.generate_strategy_overview(
                self.strategy_name, 
                parameters, 
                getattr(self, 'current_symbol', 'the market')
            )
        except Exception as e:
            logger.error(f"Failed to generate strategy overview: {e}")
            return f"Strategy overview generation failed: {str(e)}"
    
    def generate_performance_summary(self, 
                                   metrics: Dict[str, Any],
                                   benchmark_comparison: Dict = None) -> str:
        """Generate AI summary of strategy performance."""
        if not self.enable_gpt:
            return self._basic_performance_summary(metrics)
        
        try:
            return self.gpt_service.summarize_performance(
                metrics,
                self.strategy_name,
                getattr(self, 'current_symbol', 'UNKNOWN'),
                benchmark_comparison
            )
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return self._basic_performance_summary(metrics)
    
    def _basic_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Fallback performance summary without AI."""
        total_return = metrics.get('Total Return', 0) * 100
        sharpe = metrics.get('Sharpe', 0)
        max_dd = metrics.get('Max Drawdown', 0) * 100
        
        return f"""Strategy Performance Summary:
        • Total Return: {total_return:.1f}%
        • Sharpe Ratio: {sharpe:.2f}
        • Max Drawdown: {max_dd:.1f}%
        
        Basic analysis: {"Strong performance" if sharpe > 1 else "Moderate performance" if sharpe > 0.5 else "Weak performance"} with {"acceptable" if max_dd < 20 else "high"} drawdown risk."""
    
    def get_trade_insights(self) -> Dict[str, Any]:
        """Get comprehensive trade insights including AI analysis."""
        insights = {
            'total_trades': len(self.trades),
            'total_explanations': len(self.trade_explanations),
            'gpt_enabled': self.enable_gpt,
            'strategy_name': self.strategy_name
        }
        
        if self.trades:
            winning_trades = [t for t in self.trades if t['net_pnl'] > 0]
            losing_trades = [t for t in self.trades if t['net_pnl'] <= 0]
            
            insights.update({
                'win_rate': len(winning_trades) / len(self.trades),
                'avg_winner': np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0,
                'avg_loser': np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0,
                'profit_factor': (sum(t['net_pnl'] for t in winning_trades) / 
                                abs(sum(t['net_pnl'] for t in losing_trades))) if losing_trades else float('inf')
            })
        
        return insights
    
    def export_detailed_results(self) -> Dict[str, Any]:
        """Export comprehensive results including GPT insights."""
        return {
            'trades': self.trades,
            'trade_explanations': self.trade_explanations,
            'equity_curve': self.equity_curve,
            'insights': self.get_trade_insights(),
            'strategy_config': {
                'name': self.strategy_name,
                'gpt_enabled': self.enable_gpt,
                'initial_capital': self.initial_capital,
                'risk_management': {
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'max_drawdown_pct': self.max_drawdown_pct,
                    'position_size_pct': self.position_size_pct
                }
            }
        }


def create_enhanced_simulator(config: Dict[str, Any], strategy_name: str = "Custom Strategy") -> GPTEnhancedSimulator:
    """Factory function to create GPT-enhanced simulator from config."""
    
    sim_config = config.get('simulation', {})
    risk_config = config.get('risk_management', {})
    
    return GPTEnhancedSimulator(
        initial_capital=sim_config.get('initial_capital', 10000),
        commission=sim_config.get('commission', 0.001),
        slippage=sim_config.get('slippage', 0.0005),
        stop_loss_pct=risk_config.get('stop_loss_pct'),
        take_profit_pct=risk_config.get('take_profit_pct'), 
        max_drawdown_pct=risk_config.get('max_drawdown_pct'),
        position_size_pct=risk_config.get('position_size_pct', 1.0),
        enable_gpt=config.get('enable_gpt', True),
        strategy_name=strategy_name
    )


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Create sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download("AAPL", start=start_date, end=end_date)
    data = data.reset_index()
    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    
    # Create simple signals (SMA crossover)
    data['sma_fast'] = data['close'].rolling(10).mean()
    data['sma_slow'] = data['close'].rolling(20).mean()
    data['signal'] = 0
    data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
    data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
    
    # Create enhanced simulator
    simulator = GPTEnhancedSimulator(
        initial_capital=10000,
        enable_gpt=True,
        strategy_name="SMA Crossover Demo"
    )
    
    # Run simulation
    equity_curve, trades = simulator.simulate_strategy(data, data[['signal']], "AAPL")
    
    print(f"Simulation complete: {len(trades)} trades")
    if not trades.empty:
        print("\nTrade explanations:")
        for idx, trade in trades.iterrows():
            print(f"\nTrade {idx + 1}:")
            print(f"P&L: ${trade['net_pnl']:.2f}")
            print(f"Explanation: {trade['gpt_explanation']}")