"""Enhanced trading simulator with AI-powered trade analysis and explanation.

Extends the base TradingSimulator with GPT-4 integration to provide human-readable
explanations for trading decisions and comprehensive trade analysis.

Classes:
    GPTEnhancedSimulator: Enhanced simulator with AI trade commentary
    
Functions:
    create_gpt_enhanced_simulator(): Factory function using config
"""

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
    """Trading simulator extended with GPT-4 powered trade analysis and explanations.
    
    Attributes:
        strategy_name (str): Name of the trading strategy being simulated
        enable_gpt (bool): Whether GPT analysis is enabled
        gpt_service (GPTService): AI service for generating explanations
        trade_explanations (list): Storage for AI-generated trade explanations
        current_symbol (str): Current trading symbol for context
    """
    
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
        """Initialize enhanced trading simulator with GPT-4 trade analysis capabilities.
        
        Args:
            initial_capital (float): Starting capital for simulation
            enable_gpt (bool): Enable GPT-4 trade explanations (default: True)
            strategy_name (str): Name of the trading strategy for context
            **kwargs: Additional parameters passed to base TradingSimulator
        """
        # Initialize base simulator with all standard parameters
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
        
        # Enhanced simulator specific configuration
        self.strategy_name = strategy_name
        self.enable_gpt = enable_gpt
        self.gpt_service = GPTService() if enable_gpt else None
        self.trade_explanations = []  # Storage for AI-generated explanations
        
        # Validate GPT service availability
        if enable_gpt and not self.gpt_service.enabled:
            logger.warning("GPT service disabled - no API key found")
            self.enable_gpt = False
    
    def _close_position_with_gpt(self, timestamp: pd.Timestamp, price: float, reason: str):
        """
        Close a position and generate AI-powered trade explanation.
        
        This method extends the standard position closing functionality with
        AI-powered analysis of the completed trade. It generates contextual
        explanations that help users understand the trading decision and outcome.
        
        The method maintains the complete trade execution flow while adding
        sophisticated AI analysis capabilities that provide educational value
        and strategic insights.
        
        Args:
            timestamp (pd.Timestamp): Time when the position is closed
            price (float): Price at which the position is closed
            reason (str): Reason for closing the position (e.g., 'signal', 'stop_loss')
        
        Processing Flow:
        ===============
        1. **Trade Data Capture**: Store original trade information before closing
        2. **Position Closure**: Execute standard position closing logic
        3. **Context Creation**: Build comprehensive trade context for AI analysis
        4. **AI Analysis**: Generate explanation using GPT-4 language model
        5. **Result Storage**: Store explanation with trade metadata
        
        AI Analysis Components:
        ======================
        - Entry and exit price analysis
        - Trade duration and timing assessment
        - Profit/loss attribution and reasoning
        - Market context and strategy effectiveness
        - Educational insights and lessons learned
        
        Error Handling:
        ==============
        - Graceful handling of API failures
        - Fallback to standard closing when AI unavailable
        - Comprehensive error logging
        - Preservation of core simulation functionality
        
        Performance Optimization:
        ========================
        - Efficient trade context creation
        - Optimized API call patterns
        - Memory-conscious explanation storage
        - Lazy evaluation of AI analysis
        """
        # Exit early if no position to close
        if self.position == 0 or self.current_trade is None:
            return
        
        # Capture original trade data before closing for AI analysis
        original_trade = self.current_trade.copy()
        original_position = self.position
        original_shares = self.shares
        
        # Execute standard position closing using parent method
        super()._close_position(timestamp, price, reason)
        
        # Generate AI-powered trade explanation if enabled and trade completed
        if self.enable_gpt and self.trades:
            try:
                # Retrieve the just-completed trade from the trade log
                completed_trade = self.trades[-1]
                
                # Create comprehensive trade context for AI analysis
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
                
                # Generate AI-powered explanation using GPT-4
                explanation = self.gpt_service.explain_trade_decision(trade_context)
                
                # Store the explanation with comprehensive metadata
                self.trade_explanations.append({
                    'trade_index': len(self.trades) - 1,
                    'entry_time': completed_trade['entry_time'],
                    'exit_time': completed_trade['exit_time'],
                    'explanation': explanation,
                    'trade_id': f"{completed_trade['side']}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                })
                
                logger.info(f"Generated GPT explanation for {completed_trade['side']} trade")
                
            except Exception as e:
                # Handle AI service failures gracefully
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
        """
        Execute strategy simulation with AI-enhanced trade analysis.
        
        This method provides the main interface for running trading strategy
        backtests with AI-powered explanations. It extends the base simulation
        functionality with sophisticated trade analysis capabilities.
        
        The method maintains full compatibility with the base simulator while
        adding comprehensive AI-driven insights that make the results more
        interpretable and educational.
        
        Args:
            price_data (pd.DataFrame): Historical price data with OHLCV columns
            signals (pd.DataFrame): Trading signals with signal column
            symbol (str, optional): Trading symbol for context. Defaults to "UNKNOWN".
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - equity_curve: Time series of portfolio equity values
                - trades_df: Enhanced trade log with AI explanations
        
        Simulation Process:
        ==================
        1. **Setup Phase**: Configure AI context and override position closing
        2. **Execution Phase**: Run standard backtesting with AI integration
        3. **Analysis Phase**: Generate AI explanations for all trades
        4. **Enhancement Phase**: Merge explanations with trade data
        5. **Output Phase**: Return enhanced results with AI insights
        
        AI Enhancement Features:
        =======================
        - Real-time trade explanation generation
        - Context-aware market analysis
        - Professional investment commentary
        - Educational insights and lessons
        - Performance attribution with reasoning
        
        Data Enhancement:
        ================
        The returned trade dataframe includes:
        - All standard trade metrics
        - AI-generated explanations for each trade
        - Trade-specific insights and analysis
        - Educational commentary and lessons
        - Professional investment reasoning
        
        Error Handling:
        ==============
        - Graceful degradation when AI service fails
        - Fallback to standard simulation results
        - Comprehensive error logging and reporting
        - Preservation of core functionality
        
        Performance Considerations:
        ==========================
        - Efficient AI service integration
        - Optimized explanation generation
        - Memory-conscious result storage
        - Scalable to large datasets
        
        Example Usage:
        =============
        ```python
        # Run enhanced simulation
        equity_curve, trades = simulator.simulate_strategy(
            price_data, signals, "AAPL"
        )
        
        # Access AI explanations
        for idx, trade in trades.iterrows():
            print(f"Trade {idx}: {trade['gpt_explanation']}")
        ```
        """
        # Store trading symbol for AI context generation
        self.current_symbol = symbol
        
        # Override position closing method to include AI analysis
        original_close = self._close_position
        self._close_position = self._close_position_with_gpt
        
        try:
            # Execute standard backtesting with AI-enhanced position closing
            equity_curve, trades_df = super().simulate_strategy(price_data, signals)
            
            # Enhance trade dataframe with AI explanations
            if self.trade_explanations:
                # Convert explanations to DataFrame for efficient merging
                explanations_df = pd.DataFrame(self.trade_explanations)
                
                # Prepare trades dataframe for merging
                trades_df = trades_df.reset_index(drop=True)
                trades_df['trade_index'] = trades_df.index
                
                # Merge AI explanations with trade data
                trades_df['gpt_explanation'] = trades_df['trade_index'].map(
                    {exp['trade_index']: exp['explanation'] for exp in self.trade_explanations}
                )
                
                # Fill missing explanations with appropriate messages
                trades_df['gpt_explanation'] = trades_df['gpt_explanation'].fillna(
                    "No explanation generated"
                )
            else:
                # Add explanation column even when no explanations generated
                trades_df['gpt_explanation'] = "GPT disabled or no trades completed"
            
            return equity_curve, trades_df
            
        finally:
            # Restore original position closing method
            self._close_position = original_close
    
    def get_strategy_summary(self) -> str:
        """
        Generate AI-powered strategy performance summary.
        
        This method provides a comprehensive, AI-generated summary of the
        strategy's performance using natural language processing. It creates
        professional-grade investment commentary that's accessible to both
        technical and non-technical audiences.
        
        Returns:
            str: AI-generated strategy performance summary
        
        Summary Components:
        ==================
        - Overall strategy performance assessment
        - Key strengths and weaknesses
        - Risk-adjusted performance analysis
        - Market condition effectiveness
        - Recommendations for improvement
        
        Example Output:
        ==============
        "This SMA Crossover strategy delivered a 12.5% total return with a
        Sharpe ratio of 1.3, demonstrating strong risk-adjusted performance
        in trending markets. The strategy's 68% win rate indicates effective
        signal generation, though the average holding period of 8 days may
        limit its effectiveness in high-volatility environments."
        """
        if not self.enable_gpt or not self.gpt_service.enabled:
            return "AI strategy summary not available (GPT service disabled)"
        
        try:
            # Calculate performance metrics for AI analysis
            metrics = self.get_performance_metrics()
            
            # Generate AI-powered performance summary
            summary = self.gpt_service.summarize_performance(
                metrics, 
                self.strategy_name, 
                getattr(self, 'current_symbol', 'the market')
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate strategy summary: {e}")
            return f"Strategy summary generation failed: {str(e)}"
    
    def get_trade_insights(self, trade_index: int = None) -> str:
        """
        Get AI insights for a specific trade or all trades.
        
        This method provides detailed AI analysis for individual trades or
        comprehensive insights across all trades. It's useful for understanding
        specific trading decisions and their outcomes.
        
        Args:
            trade_index (int, optional): Index of specific trade to analyze.
                If None, returns insights for all trades.
        
        Returns:
            str: AI-generated trade insights and analysis
        
        Insight Categories:
        ==================
        - Trade-specific analysis and reasoning
        - Market context and conditions
        - Strategy effectiveness assessment
        - Risk management evaluation
        - Educational lessons and takeaways
        """
        if not self.trade_explanations:
            return "No trade explanations available"
        
        if trade_index is not None:
            # Return specific trade explanation
            for exp in self.trade_explanations:
                if exp['trade_index'] == trade_index:
                    return exp['explanation']
            return f"No explanation found for trade index {trade_index}"
        else:
            # Return summary of all trade insights
            total_explanations = len(self.trade_explanations)
            successful_explanations = sum(1 for exp in self.trade_explanations 
                                        if not exp['explanation'].startswith('Explanation generation failed'))
            
            return (f"Generated {successful_explanations}/{total_explanations} "
                   f"trade explanations for {self.strategy_name}")


def create_gpt_enhanced_simulator(config: Dict[str, Any], strategy_name: str) -> GPTEnhancedSimulator:
    """
    Create GPT-enhanced simulator from configuration.
    
    This factory function creates a properly configured GPTEnhancedSimulator
    instance from a configuration dictionary. It provides a convenient way
    to initialize the simulator with standard configuration patterns.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - simulation: Basic simulation parameters
            - risk_management: Risk management settings
            - enable_gpt: Whether to enable GPT analysis
        strategy_name (str): Name of the trading strategy
    
    Returns:
        GPTEnhancedSimulator: Configured simulator instance
    
    Configuration Structure:
    =======================
    ```yaml
    simulation:
      initial_capital: 10000
      commission: 0.001
      slippage: 0.0005
    
    risk_management:
      stop_loss_pct: 0.02
      take_profit_pct: 0.05
      max_drawdown_pct: 0.15
      position_size_pct: 0.8
    
    enable_gpt: true
    ```
    
    Example Usage:
    =============
    ```python
    # Create from configuration
    simulator = create_gpt_enhanced_simulator(config, "MyStrategy")
    
    # Run simulation
    equity_curve, trades = simulator.simulate_strategy(data, signals, "AAPL")
    ```
    """
    # Extract configuration sections with defaults
    sim_config = config.get('simulation', {})
    risk_config = config.get('risk_management', {})
    
    # Create and return configured simulator
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


# Example usage and demonstration
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Create sample market data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Download real market data for demonstration
    data = yf.download("AAPL", start=start_date, end=end_date)
    data = data.reset_index()
    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    
    # Create simple SMA crossover signals for demonstration
    data['sma_fast'] = data['close'].rolling(10).mean()
    data['sma_slow'] = data['close'].rolling(20).mean()
    data['signal'] = 0
    data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
    data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
    
    # Create enhanced simulator with AI capabilities
    simulator = GPTEnhancedSimulator(
        initial_capital=10000,
        enable_gpt=True,
        strategy_name="SMA Crossover Demo"
    )
    
    # Run simulation with AI explanations
    equity_curve, trades = simulator.simulate_strategy(data, data[['signal']], "AAPL")
    
    # Display results
    print(f"Simulation complete: {len(trades)} trades executed")
    print(f"Final equity: ${equity_curve.iloc[-1]['equity']:.2f}")
    
    # Show AI-generated trade explanations
    if not trades.empty:
        print("\nAI-Generated Trade Explanations:")
        print("=" * 60)
        for idx, trade in trades.iterrows():
            print(f"\nTrade {idx + 1}:")
            print(f"Entry: {trade['entry_time']} at ${trade['entry_price']:.2f}")
            print(f"Exit: {trade['exit_time']} at ${trade['exit_price']:.2f}")
            print(f"P&L: ${trade['net_pnl']:.2f} ({trade['return']:.2%})")
            print(f"Duration: {trade['duration']} days")
            print(f"AI Analysis: {trade['gpt_explanation']}")
    
    # Generate strategy summary
    summary = simulator.get_strategy_summary()
    print(f"\nStrategy Summary:")
    print("=" * 60)
    print(summary)