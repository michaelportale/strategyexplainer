"""Batch backtest script for hyperparameter sweeps across multiple strategies and parameters."""

import subprocess
import itertools
import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directory exists
OUTPUT_DIR = "backend/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== HYPERPARAMETER GRID =====
TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"]
STOP_LOSSES = [0.02, 0.05, 0.10, None]  # 2%, 5%, 10%, or None
TAKE_PROFITS = [0.05, 0.10, 0.20, None]  # 5%, 10%, 20%, or None
POSITION_SIZES = [1.0, 0.75, 0.5]  # 100%, 75%, 50% of capital
MAX_DRAWDOWNS = [None, 0.15, 0.20, 0.25]  # None, 15%, 20%, 25% max portfolio drawdown
CAPITAL = 10000
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

# Output files
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SUMMARY_CSV = f"{OUTPUT_DIR}/batch_summary_{TIMESTAMP}.csv"
DETAILED_JSON = f"{OUTPUT_DIR}/batch_detailed_{TIMESTAMP}.json"

# CSV Header
CSV_HEADER = [
    "ticker", "stop_loss_pct", "take_profit_pct", "max_drawdown_pct", "position_size_pct",
    "total_return", "annual_return", "monthly_return", "daily_return_avg",
    "annual_volatility", "max_drawdown", "sharpe_ratio", "sortino_ratio",
    "win_rate", "total_trades", "profit_factor", "expectancy", "largest_win", "largest_loss",
    "avg_win", "avg_loss", "max_consecutive_wins", "max_consecutive_losses",
    "calmar_ratio", "recovery_factor", "kelly_criterion", "run_status", "error_message"
]


def parse_metrics_from_output(output: str) -> Dict[str, Any]:
    """Parse performance metrics from the backtest output.
    
    Args:
        output: String output from the backtest run
        
    Returns:
        Dictionary with parsed metrics
    """
    metrics = {}
    
    try:
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for metric lines with format "key: value"
            if ':' in line and not line.startswith('INFO') and not line.startswith('DEBUG'):
                # Handle log format lines like "2024-01-01 12:00:00 INFO: total_return: 0.1234"
                if 'INFO:' in line or 'DEBUG:' in line:
                    # Extract the part after INFO: or DEBUG:
                    metric_part = line.split('INFO:')[-1].split('DEBUG:')[-1].strip()
                else:
                    metric_part = line
                
                if ':' in metric_part:
                    key, value = metric_part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert to float
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
                        
    except Exception as e:
        logger.error(f"Error parsing metrics: {e}")
        
    return metrics


def run_single_backtest(ticker: str, 
                       stop_loss_pct: Optional[float],
                       take_profit_pct: Optional[float], 
                       max_drawdown_pct: Optional[float],
                       position_size_pct: float) -> Dict[str, Any]:
    """Run a single backtest with specified parameters.
    
    Args:
        ticker: Stock ticker symbol
        stop_loss_pct: Stop loss percentage (or None)
        take_profit_pct: Take profit percentage (or None)
        max_drawdown_pct: Max drawdown percentage (or None)
        position_size_pct: Position size percentage
        
    Returns:
        Dictionary with run results and metrics
    """
    # Build command arguments
    args = [
        "python", "backend/momentum_backtest.py",
        "--ticker", str(ticker),
        "--capital", str(CAPITAL),
        "--start", START_DATE,
        "--end", END_DATE,
        "--position_size_pct", str(position_size_pct),
    ]
    
    # Add optional parameters
    if stop_loss_pct is not None:
        args.extend(["--stop_loss_pct", str(stop_loss_pct)])
    if take_profit_pct is not None:
        args.extend(["--take_profit_pct", str(take_profit_pct)])
    if max_drawdown_pct is not None:
        args.extend(["--max_drawdown_pct", str(max_drawdown_pct)])
    
    result = {
        "ticker": ticker,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "position_size_pct": position_size_pct,
        "run_status": "SUCCESS",
        "error_message": "",
        "command": " ".join(args)
    }
    
    try:
        logger.info(f"Running: {ticker} | SL: {stop_loss_pct} | TP: {take_profit_pct} | MDD: {max_drawdown_pct} | POS: {position_size_pct}")
        
        # Run the backtest
        output = subprocess.check_output(
            args, 
            stderr=subprocess.STDOUT, 
            universal_newlines=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse metrics from output
        metrics = parse_metrics_from_output(output)
        result.update(metrics)
        
        # Log key results
        total_return = metrics.get('total_return', 'N/A')
        sharpe = metrics.get('sharpe_ratio', 'N/A')
        max_dd = metrics.get('max_drawdown', 'N/A')
        trades = metrics.get('total_trades', 'N/A')
        
        logger.info(f"  ✓ Return: {total_return:.1%} | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.1%} | Trades: {trades}" 
                   if isinstance(total_return, (int, float)) else f"  ✓ Completed: {ticker}")
        
    except subprocess.TimeoutExpired:
        result["run_status"] = "TIMEOUT"
        result["error_message"] = "Process timed out after 5 minutes"
        logger.error(f"  ✗ TIMEOUT: {ticker}")
        
    except subprocess.CalledProcessError as e:
        result["run_status"] = "ERROR"
        result["error_message"] = f"Process failed: {e.output}"
        logger.error(f"  ✗ ERROR: {ticker} - {e}")
        
    except Exception as e:
        result["run_status"] = "ERROR"
        result["error_message"] = str(e)
        logger.error(f"  ✗ EXCEPTION: {ticker} - {e}")
    
    return result


def run_batch_backtest():
    """Run the full batch backtest across all parameter combinations."""
    
    logger.info("=" * 60)
    logger.info("STARTING BATCH HYPERPARAMETER SWEEP")
    logger.info("=" * 60)
    
    # Generate all parameter combinations
    combos = list(itertools.product(
        TICKERS, STOP_LOSSES, TAKE_PROFITS, MAX_DRAWDOWNS, POSITION_SIZES
    ))
    
    total_combos = len(combos)
    logger.info(f"Total combinations to test: {total_combos}")
    logger.info(f"Estimated runtime: {total_combos * 30 / 60:.1f} minutes")
    logger.info(f"Results will be saved to: {SUMMARY_CSV}")
    
    # Initialize results storage
    all_results = []
    successful_runs = 0
    failed_runs = 0
    
    # Open CSV file for writing
    with open(SUMMARY_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
        writer.writeheader()
        
        # Run each combination
        for i, combo in enumerate(combos, 1):
            ticker, stop_loss, take_profit, max_drawdown, position_size = combo
            
            logger.info(f"\n[{i:3d}/{total_combos}] " + "="*40)
            
            # Run single backtest
            result = run_single_backtest(
                ticker=ticker,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                max_drawdown_pct=max_drawdown,
                position_size_pct=position_size
            )
            
            # Track success/failure
            if result["run_status"] == "SUCCESS":
                successful_runs += 1
            else:
                failed_runs += 1
            
            # Store result
            all_results.append(result)
            
            # Write to CSV (create row with all CSV headers)
            csv_row = {}
            for header in CSV_HEADER:
                csv_row[header] = result.get(header, "")
            writer.writerow(csv_row)
            
            # Flush CSV file to disk
            csv_file.flush()
    
    # Save detailed results to JSON
    with open(DETAILED_JSON, "w") as json_file:
        json.dump({
            "metadata": {
                "timestamp": TIMESTAMP,
                "total_combinations": total_combos,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "parameters": {
                    "tickers": TICKERS,
                    "stop_losses": STOP_LOSSES,
                    "take_profits": TAKE_PROFITS,
                    "max_drawdowns": MAX_DRAWDOWNS,
                    "position_sizes": POSITION_SIZES,
                    "capital": CAPITAL,
                    "start_date": START_DATE,
                    "end_date": END_DATE
                }
            },
            "results": all_results
        }, json_file, indent=2, default=str)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH BACKTEST COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total combinations: {total_combos}")
    logger.info(f"Successful runs: {successful_runs}")
    logger.info(f"Failed runs: {failed_runs}")
    logger.info(f"Success rate: {successful_runs/total_combos*100:.1f}%")
    logger.info(f"\nResults saved to:")
    logger.info(f"  📊 Summary CSV: {SUMMARY_CSV}")
    logger.info(f"  📋 Detailed JSON: {DETAILED_JSON}")
    
    if successful_runs > 0:
        logger.info(f"\n🎯 Ready for analysis! Load {SUMMARY_CSV} into pandas/Excel to find your edge.")
        
        # Show top performers if we have successful runs
        try:
            import pandas as pd
            df = pd.read_csv(SUMMARY_CSV)
            successful_df = df[df['run_status'] == 'SUCCESS'].copy()
            
            if len(successful_df) > 0:
                # Convert string numbers to float for sorting
                for col in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                    if col in successful_df.columns:
                        successful_df[col] = pd.to_numeric(successful_df[col], errors='coerce')
                
                logger.info("\n🏆 TOP 5 PERFORMERS BY TOTAL RETURN:")
                top_by_return = successful_df.nlargest(5, 'total_return')
                for _, row in top_by_return.iterrows():
                    logger.info(f"  {row['ticker']} | Return: {row['total_return']:.1%} | Sharpe: {row.get('sharpe_ratio', 'N/A'):.2f} | MaxDD: {row.get('max_drawdown', 'N/A'):.1%}")
                
                logger.info("\n🛡️  TOP 5 PERFORMERS BY SHARPE RATIO:")
                top_by_sharpe = successful_df.nlargest(5, 'sharpe_ratio')
                for _, row in top_by_sharpe.iterrows():
                    logger.info(f"  {row['ticker']} | Sharpe: {row.get('sharpe_ratio', 'N/A'):.2f} | Return: {row['total_return']:.1%} | MaxDD: {row.get('max_drawdown', 'N/A'):.1%}")
                    
        except ImportError:
            logger.info("Install pandas to see top performers: pip install pandas")
        except Exception as e:
            logger.warning(f"Could not show top performers: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch backtest with hyperparameter sweep")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer combinations")
    parser.add_argument("--tickers", nargs="+", help="Override tickers list", default=None)
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing (future)")
    
    args = parser.parse_args()
    
    # Quick mode for testing
    if args.quick:
        TICKERS = ["AAPL", "SPY"]
        STOP_LOSSES = [0.05, None]
        TAKE_PROFITS = [0.10, None]
        POSITION_SIZES = [1.0, 0.5]
        MAX_DRAWDOWNS = [None, 0.20]  # Add real max drawdown testing
        logger.info("🚀 QUICK MODE: Running reduced parameter set for testing (including max drawdown limits)")
    
    # Override tickers if provided
    if args.tickers:
        TICKERS = args.tickers
        logger.info(f"Using custom tickers: {TICKERS}")
    
    # Run the batch backtest
    run_batch_backtest() 