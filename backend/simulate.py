"""Trading simulation engine for strategy backtesting."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import sys
import os

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


class TradingSimulator:
    """Simulate trading based on strategy signals."""
    
    def __init__(
        self,
        initial_capital,
        commission=0.0,
        slippage=0.0,
        margin_requirement=1.0,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        max_drawdown_pct: float = None,
        position_size_pct: float = 1.0,
    ):
        """Initialize the trading simulator.
        
        Args:
            initial_capital: Starting capital amount
            commission: Commission rate (as decimal, e.g., 0.001 = 0.1%)
            slippage: Slippage rate (as decimal)
            margin_requirement: Margin requirement (1.0 = no margin)
            stop_loss_pct: Stop loss percentage (e.g., 0.1 for 10%), None to disable
            take_profit_pct: Take profit percentage (e.g., 0.2 for 20%), None to disable
            max_drawdown_pct: Max drawdown percentage before halting (e.g., 0.3 for 30%), None to disable
            position_size_pct: Fraction of capital to use per trade (default 1.0)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.margin_requirement = margin_requirement
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.position_size_pct = position_size_pct
        self.peak_equity = self.initial_capital
        self.halted = False
        self.logger = logging.getLogger(__name__)
        
        # Track state
        self.reset()
    
    def reset(self):
        """Reset simulator state."""
        self.capital = self.initial_capital
        self.position = 0.0
        self.shares = 0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.current_trade = None
        self.peak_equity = self.initial_capital
        self.halted = False
        
    def simulate_strategy(self, 
                         price_data: pd.DataFrame, 
                         signals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate trading based on price data and signals.
        
        Args:
            price_data: DataFrame with OHLCV data
            signals: DataFrame with trading signals
            
        Returns:
            Tuple of (equity_curve, trades) DataFrames
        """
        self.reset()
        
        # Align data
        aligned_data = self._align_data(price_data, signals)
        
        if aligned_data.empty:
            self.logger.warning("No aligned data available for simulation")
            return pd.DataFrame(), pd.DataFrame()
        
        # Process each time step
        for idx, row in aligned_data.iterrows():
            self._process_timestep(idx, row)
        
        # Close any open position at the end
        if self.position != 0:
            last_row = aligned_data.iloc[-1]
            self._close_position(aligned_data.index[-1], last_row['close'], "End of period")
        
        # Convert to DataFrames
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        self.logger.info(f"Simulation complete: {len(trades_df)} trades, "
                        f"Final equity: ${equity_df['equity'].iloc[-1]:,.2f}")
        
        return equity_df, trades_df
    
    def _align_data(self, price_data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Align price data and signals by index.
        
        Args:
            price_data: Price data DataFrame
            signals: Signals DataFrame
            
        Returns:
            Aligned DataFrame with both price and signal data
        """
        try:
            # Merge on index
            aligned = price_data.join(signals, how='inner', rsuffix='_signal')
            
            # Ensure we have required columns
            required_price_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_price_cols if col not in aligned.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing price columns: {missing_cols}")
                # Use close price for missing OHLC data
                for col in missing_cols:
                    if col != 'volume':
                        aligned[col] = aligned['close']
                    else:
                        aligned[col] = 0
            
            # Ensure signal column exists
            if 'signal' not in aligned.columns:
                aligned['signal'] = 0
                
            return aligned.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error aligning data: {str(e)}")
            return pd.DataFrame()
    
    def _process_timestep(self, timestamp: pd.Timestamp, row: pd.Series):
        """Process a single timestep in the simulation.
        
        Args:
            timestamp: Current timestamp
            row: Row data with price and signal information
        """
        signal = row.get('signal', 0)
        price = row['close']
        
        # Calculate current portfolio value
        portfolio_value = self._calculate_portfolio_value(price)

        # CRITICAL: Enforce no negative equity - halt if portfolio goes to zero or below
        if portfolio_value <= 0:
            if not self.halted:
                self.logger.warning(f"💀 RISK OF RUIN: Portfolio value went to zero or negative ({portfolio_value:.2f}) at {timestamp}")
                self.logger.warning(f"   Cash: {self.capital:.2f}, Position: {self.position}, Shares: {self.shares}")
                # Force close all positions
                if self.position != 0:
                    self._close_position(timestamp, price, "Risk of ruin - forced liquidation")
                self.halted = True
            return

        # Log warnings for dangerous equity levels
        equity_ratio = portfolio_value / self.initial_capital
        if equity_ratio < 0.5 and not getattr(self, '_warned_low_equity', False):
            self.logger.warning(f"⚠️  Portfolio equity down to {equity_ratio:.1%} of initial capital at {timestamp}")
            self._warned_low_equity = True

        # Max drawdown halt logic
        self.peak_equity = max(self.peak_equity, portfolio_value)
        if (
            self.max_drawdown_pct is not None
            and self.peak_equity > 0
            and (self.peak_equity - portfolio_value) / self.peak_equity >= self.max_drawdown_pct
        ):
            self.halted = True
            self.logger.warning(f"🛑 Halting simulation at {timestamp} due to max drawdown ({self.max_drawdown_pct:.1%}).")
            # Force close all positions when max drawdown is hit
            if self.position != 0:
                self._close_position(timestamp, price, "Max drawdown limit reached")
            return

        # Record equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio_value,
            'position': self.position,
            'cash': self.capital,
            'price': price
        })

        # If simulation is halted, do not process further
        if self.halted:
            return

        # Stop-loss / take-profit logic for open positions
        if self.position != 0 and self.current_trade is not None:
            entry = self.current_trade['entry_price']
            side = self.current_trade['side']
            stop_loss_triggered = False
            take_profit_triggered = False
            stop_price = None
            reason = None
            # Only check if pct is not None
            if self.stop_loss_pct is not None:
                if side == 'long' and price <= entry * (1 - self.stop_loss_pct):
                    stop_loss_triggered = True
                    stop_price = entry * (1 - self.stop_loss_pct)
                elif side == 'short' and price >= entry * (1 + self.stop_loss_pct):
                    stop_loss_triggered = True
                    stop_price = entry * (1 + self.stop_loss_pct)
            if self.take_profit_pct is not None:
                if side == 'long' and price >= entry * (1 + self.take_profit_pct):
                    take_profit_triggered = True
                    stop_price = entry * (1 + self.take_profit_pct)
                elif side == 'short' and price <= entry * (1 - self.take_profit_pct):
                    take_profit_triggered = True
                    stop_price = entry * (1 - self.take_profit_pct)
            # If both triggered, use the one hit first (conservative: choose worst for us)
            if stop_loss_triggered:
                self._close_position(timestamp, stop_price, "Stop loss triggered")
                self.logger.info(f"Stop loss triggered at {timestamp} at price {stop_price:.2f}")
                return
            elif take_profit_triggered:
                self._close_position(timestamp, stop_price, "Take profit triggered")
                self.logger.info(f"Take profit triggered at {timestamp} at price {stop_price:.2f}")
                return

        # Process signals
        if signal != 0 and signal != self.position:
            self._execute_signal(timestamp, row, signal)
    
    def _execute_signal(self, timestamp: pd.Timestamp, row: pd.Series, signal: float):
        """Execute a trading signal.
        
        Args:
            timestamp: Current timestamp
            row: Row data with price information
            signal: Trading signal (-1, 0, 1)
        """
        price = row['close']
        
        # Close existing position if we have one
        if self.position != 0:
            self._close_position(timestamp, price, f"Signal change to {signal}")
        
        # Open new position if signal is not flat
        if signal != 0:
            self._open_position(timestamp, row, signal)
    
    def _open_position(self, timestamp: pd.Timestamp, row: pd.Series, signal: float):
        """Open a new position.
        
        Args:
            timestamp: Current timestamp
            row: Row data with price information
            signal: Trading signal (1 for long, -1 for short)
        """
        price = row['close']
        
        # Apply slippage
        if signal > 0:  # Long position
            execution_price = price * (1 + self.slippage)
        else:  # Short position
            execution_price = price * (1 - self.slippage)
        
        # Calculate position size based on position_size_pct and available capital
        position_value = self.capital * self.position_size_pct * abs(signal)
        
        # Account for margin
        required_capital = position_value * self.margin_requirement
        
        if required_capital > self.capital:
            # Adjust position size to available capital
            position_value = self.capital / self.margin_requirement
        
        # Calculate shares
        shares = int(position_value / execution_price)
        
        if shares > 0:
            # Calculate actual cost including commission
            total_cost = shares * execution_price
            commission_cost = total_cost * self.commission
            total_with_commission = total_cost + commission_cost
            
            # Final safety check: ensure we don't go negative on capital
            if total_with_commission > self.capital:
                self.logger.warning(f"⚠️  Position would exceed available capital. Adjusting size.")
                # Recalculate with maximum available capital
                max_cost = self.capital * 0.99  # Leave 1% buffer
                shares = int(max_cost / execution_price)
                if shares <= 0:
                    self.logger.warning(f"❌ Insufficient capital to open position. Skipping.")
                    return
                total_cost = shares * execution_price
                commission_cost = total_cost * self.commission
                total_with_commission = total_cost + commission_cost
            
            # Update state
            self.position = signal
            self.shares = shares if signal > 0 else -shares
            self.entry_price = execution_price
            self.capital -= total_with_commission
            
            # Start tracking the trade
            self.current_trade = {
                'entry_time': timestamp,
                'entry_price': execution_price,
                'shares': self.shares,
                'side': 'long' if signal > 0 else 'short',
                'entry_commission': commission_cost
            }
            
            self.logger.debug(f"Opened {self.current_trade['side']} position: "
                            f"{abs(self.shares)} shares at ${execution_price:.2f}")
    
    def _close_position(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Close the current position.
        
        Args:
            timestamp: Current timestamp
            price: Current price
            reason: Reason for closing position
        """
        if self.position == 0 or self.current_trade is None:
            return
        
        # Apply slippage
        if self.position > 0:  # Closing long position
            execution_price = price * (1 - self.slippage)
        else:  # Closing short position
            execution_price = price * (1 + self.slippage)
        
        # Calculate trade results
        shares = abs(self.shares)
        total_proceeds = shares * execution_price
        commission_cost = total_proceeds * self.commission
        net_proceeds = total_proceeds - commission_cost
        
        # Update capital
        if self.position > 0:  # Was long
            self.capital += net_proceeds
            trade_return = (execution_price - self.entry_price) / self.entry_price
        else:  # Was short
            self.capital += net_proceeds
            trade_return = (self.entry_price - execution_price) / self.entry_price
        
        # Calculate trade P&L
        entry_value = shares * self.entry_price
        exit_value = shares * execution_price
        
        if self.position > 0:
            gross_pnl = exit_value - entry_value
        else:
            gross_pnl = entry_value - exit_value
        
        total_commission = self.current_trade['entry_commission'] + commission_cost
        net_pnl = gross_pnl - total_commission
        
        # Record the completed trade
        trade_record = {
            'entry_time': self.current_trade['entry_time'],
            'exit_time': timestamp,
            'side': self.current_trade['side'],
            'shares': shares,
            'entry_price': self.current_trade['entry_price'],
            'exit_price': execution_price,
            'gross_pnl': gross_pnl,
            'commission': total_commission,
            'net_pnl': net_pnl,
            'return': trade_return,
            'duration': (timestamp - self.current_trade['entry_time']).days,
            'exit_reason': reason
        }
        
        self.trades.append(trade_record)
        
        # Reset position state
        self.position = 0.0
        self.shares = 0
        self.entry_price = 0.0
        self.current_trade = None
        
        self.logger.debug(f"Closed position: {trade_record['side']} "
                         f"P&L: ${net_pnl:.2f} ({trade_return*100:.2f}%)")
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value.

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value
        """
        cash_value = self.capital

        if self.position != 0 and self.shares != 0:
            # Portfolio value = cash + market value of shares
            position_value = self.shares * current_price
            return cash_value + position_value

        return cash_value
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the simulation.
        
        Returns:
            Dictionary with simulation summary
        """
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        stats = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'commission_paid': trades_df['commission'].sum() if not trades_df.empty else 0,
            'gross_profit': trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() if not trades_df.empty else 0,
            'gross_loss': trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum() if not trades_df.empty else 0,
        }
        
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] < 0]
            
            stats.update({
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                'avg_win': winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0,
                'largest_win': winning_trades['net_pnl'].max() if len(winning_trades) > 0 else 0,
                'largest_loss': losing_trades['net_pnl'].min() if len(losing_trades) > 0 else 0,
                'avg_trade_duration': trades_df['duration'].mean(),
                'profit_factor': abs(stats['gross_profit'] / stats['gross_loss']) if stats['gross_loss'] != 0 else np.inf
            })
        
        return stats 