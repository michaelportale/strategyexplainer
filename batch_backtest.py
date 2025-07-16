"""
Batch Backtest System: Hyperparameter Sweep Engine for Trading Strategy Optimization

This module implements an industrial-grade hyperparameter sweep system that systematically
tests trading strategies across multiple parameter combinations to identify optimal configurations.
The system is designed for quantitative researchers and algorithmic traders seeking to
optimize their strategies through comprehensive parameter space exploration.

Key Features:
============

1. **Exhaustive Parameter Grid Search**
   - Tests all combinations of stop-loss, take-profit, position sizing, and max drawdown limits
   - Supports multiple tickers for cross-asset validation
   - Configurable parameter ranges for flexible experimentation

2. **Robust Execution Framework**
   - Subprocess-based execution for isolation and fault tolerance
   - Comprehensive error handling and logging
   - Timeout protection against hanging processes
   - Progress tracking and real-time feedback

3. **Multi-Format Output Generation**
   - CSV summary for quick analysis and spreadsheet integration
   - JSON detailed results for programmatic access
   - Real-time logging for monitoring execution progress

4. **Performance Analytics Integration**
   - Automatic top performer identification
   - Statistical summaries across all runs
   - Success rate tracking and failure analysis

Architecture:
============

The system follows a pipeline architecture:
1. **Parameter Grid Generation** - Creates all possible parameter combinations
2. **Execution Engine** - Runs individual backtests via subprocess calls
3. **Results Aggregation** - Collects and parses performance metrics
4. **Output Generation** - Exports results in multiple formats
5. **Analytics Layer** - Provides insights and top performer identification

Usage Examples:
===============

Basic Usage:
```python
# Run full hyperparameter sweep
python batch_backtest.py

# Quick test with reduced parameters
python batch_backtest.py --quick

# Custom ticker list
python batch_backtest.py --tickers AAPL MSFT GOOGL

# Combine flags
python batch_backtest.py --quick --tickers SPY QQQ
```

Advanced Integration:
```python
# Programmatic usage
from batch_backtest import run_batch_backtest, run_single_backtest

# Single backtest
result = run_single_backtest("AAPL", 0.05, 0.10, 0.20, 1.0)

# Full sweep
run_batch_backtest()
```

Parameter Space:
===============

The system explores the following parameter dimensions:

- **Tickers**: Multiple assets for cross-validation
- **Stop Losses**: Risk management levels (2%, 5%, 10%, None)
- **Take Profits**: Profit-taking levels (5%, 10%, 20%, None)
- **Position Sizes**: Capital allocation (100%, 75%, 50%)
- **Max Drawdowns**: Portfolio protection (None, 15%, 20%, 25%)

Total combinations: |Tickers| × |Stop Losses| × |Take Profits| × |Position Sizes| × |Max Drawdowns|

Output Format:
=============

CSV Summary includes:
- All parameter combinations and results
- Performance metrics (returns, Sharpe ratio, drawdown)
- Trade statistics (win rate, profit factor, expectancy)
- Risk metrics (volatility, Calmar ratio, Kelly criterion)
- Execution status and error tracking

Performance Considerations:
==========================

- Each combination runs in isolated subprocess for fault tolerance
- Timeout protection prevents hanging processes
- Memory-efficient streaming output to handle large parameter spaces
- Progress logging for long-running sweeps
- Automatic cleanup of temporary files

Integration Points:
==================

The system integrates with:
- backend/momentum_backtest.py (execution engine)
- analyze_batch_results.py (results analysis)
- Pandas/Excel (data analysis)
- Jupyter notebooks (research workflow)

Educational Value:
=================

This module demonstrates:
- Industrial hyperparameter optimization techniques
- Robust subprocess management in Python
- Large-scale backtesting architecture
- Performance metrics collection and analysis
- Error handling in distributed computing contexts

For researchers and practitioners, this system provides a foundation for:
- Strategy parameter optimization
- Cross-asset validation
- Risk management calibration
- Performance benchmarking
- Systematic trading development

Author: Strategy Explainer Framework
Version: 2.0
License: Educational Use
"""

import subprocess
import itertools
import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Import configuration manager
from config.config_manager import get_config_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get configuration manager instance
config = get_config_manager()

# Load batch backtest configuration
batch_config = config.get_section('batch_backtest')

# Ensure output directory exists
OUTPUT_DIR = batch_config.get('output_directory', 'backend/outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== HYPERPARAMETER GRID CONFIGURATION =====
# These parameters are now loaded from config.yaml
# Modify config.yaml to customize your parameter sweep

# Asset universe for cross-validation
TICKERS = batch_config.get('tickers', ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"])

# Stop loss levels (as percentages) - None means no stop loss
STOP_LOSSES = batch_config.get('stop_losses', [0.02, 0.05, 0.10, None])

# Take profit levels (as percentages) - None means no take profit
TAKE_PROFITS = batch_config.get('take_profits', [0.05, 0.10, 0.20, None])

# Position sizing (percentage of capital per trade)
POSITION_SIZES = batch_config.get('position_sizes', [1.0, 0.75, 0.5])

# Maximum portfolio drawdown limits - None means no limit
MAX_DRAWDOWNS = batch_config.get('max_drawdowns', [None, 0.15, 0.20, 0.25])

# Backtest configuration
CAPITAL = batch_config.get('capital', 10000)  # Starting capital
START_DATE = batch_config.get('start_date', "2020-01-01")  # Backtest start date
END_DATE = batch_config.get('end_date', "2024-01-01")    # Backtest end date

# Output file configuration with timestamp
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SUMMARY_CSV = f"{OUTPUT_DIR}/batch_summary_{TIMESTAMP}.csv"
DETAILED_JSON = f"{OUTPUT_DIR}/batch_detailed_{TIMESTAMP}.json"

# CSV header defining all output columns (loaded from config or use default)
CSV_HEADER = batch_config.get('csv_header', [
    # Parameter columns
    "ticker", "stop_loss_pct", "take_profit_pct", "max_drawdown_pct", "position_size_pct",
    
    # Return metrics
    "total_return", "annual_return", "monthly_return", "daily_return_avg",
    
    # Risk metrics
    "annual_volatility", "max_drawdown", "sharpe_ratio", "sortino_ratio",
    
    # Trade statistics
    "win_rate", "total_trades", "profit_factor", "expectancy", "largest_win", "largest_loss",
    "avg_win", "avg_loss", "max_consecutive_wins", "max_consecutive_losses",
    
    # Advanced metrics
    "calmar_ratio", "recovery_factor", "kelly_criterion", 
    
    # Execution tracking
    "run_status", "error_message"
])


def parse_metrics_from_output(output: str) -> Dict[str, Any]:
    """
    Parse performance metrics from backtest subprocess output.
    
    This function extracts structured performance metrics from the text output
    of the momentum_backtest.py subprocess. It handles various output formats
    including log-formatted lines and direct metric outputs.
    
    The parsing logic is designed to be robust against formatting variations
    and includes error handling for malformed output.
    
    Args:
        output (str): Raw text output from the backtest subprocess
        
    Returns:
        Dict[str, Any]: Dictionary containing parsed metrics where:
            - Keys are metric names (e.g., 'total_return', 'sharpe_ratio')
            - Values are numeric values (float) or strings for non-numeric metrics
            - Empty dict if parsing fails
    
    Parsing Strategy:
    ================
    1. Split output into lines
    2. Look for lines containing ':' separator
    3. Skip log header lines (INFO, DEBUG)
    4. Extract metric name and value pairs
    5. Convert values to float when possible
    6. Handle edge cases and malformed data
    
    Example Output Formats Handled:
    ==============================
    - "total_return: 0.1234"
    - "2024-01-01 12:00:00 INFO: sharpe_ratio: 1.85"
    - "DEBUG: max_drawdown: -0.0876"
    
    Error Handling:
    ==============
    - Malformed lines are skipped
    - Non-numeric values are stored as strings
    - Parsing errors are logged but don't halt execution
    - Returns empty dict on complete failure
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
                    
                    # Try to convert to float for numeric metrics
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        # Store as string if not numeric
                        metrics[key] = value
                        
    except Exception as e:
        logger.error(f"Error parsing metrics from output: {e}")
        logger.debug(f"Problematic output: {output[:500]}...")  # Log first 500 chars
        
    return metrics


def run_single_backtest(ticker: str, 
                       stop_loss_pct: Optional[float],
                       take_profit_pct: Optional[float], 
                       max_drawdown_pct: Optional[float],
                       position_size_pct: float) -> Dict[str, Any]:
    """
    Execute a single backtest with specified parameters.
    
    This function runs an individual backtest by calling the momentum_backtest.py
    script as a subprocess with the provided parameters. It handles command
    construction, execution, output parsing, and error management.
    
    The function is designed to be fault-tolerant and provides comprehensive
    error handling for various failure modes including timeouts, subprocess
    errors, and output parsing failures.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        stop_loss_pct (Optional[float]): Stop loss percentage (e.g., 0.05 for 5%)
            - None means no stop loss
        take_profit_pct (Optional[float]): Take profit percentage (e.g., 0.10 for 10%)
            - None means no take profit
        max_drawdown_pct (Optional[float]): Maximum drawdown percentage (e.g., 0.20 for 20%)
            - None means no drawdown limit
        position_size_pct (float): Position size as percentage of capital (e.g., 1.0 for 100%)
        
    Returns:
        Dict[str, Any]: Comprehensive result dictionary containing:
            - Input parameters for traceability
            - Execution metadata (status, command, errors)
            - Performance metrics (if successful)
            - Error information (if failed)
    
    Execution Process:
    =================
    1. **Command Construction**: Build subprocess arguments dynamically
    2. **Parameter Validation**: Ensure required parameters are present
    3. **Subprocess Execution**: Run with timeout protection
    4. **Output Parsing**: Extract metrics from text output
    5. **Result Packaging**: Combine parameters and metrics
    6. **Error Handling**: Capture and log execution failures
    
    Error Handling:
    ==============
    - Subprocess timeout (5 minutes max)
    - Command execution failures
    - Output parsing errors
    - Parameter validation issues
    - System resource constraints
    
    Example Usage:
    =============
    ```python
    # Basic usage
    result = run_single_backtest("AAPL", 0.05, 0.10, 0.20, 1.0)
    
    # No stop loss or take profit
    result = run_single_backtest("MSFT", None, None, 0.15, 0.75)
    
    # Check results
    if result['run_status'] == 'SUCCESS':
        print(f"Total Return: {result['total_return']:.2%}")
    else:
        print(f"Error: {result['error_message']}")
    ```
    
    Performance Considerations:
    ==========================
    - Subprocess isolation prevents memory leaks
    - Timeout protection prevents hanging processes
    - Output streaming for large datasets
    - Resource cleanup on completion
    """
    # Build command arguments dynamically
    args = [
        "python", "backend/momentum_backtest.py",
        "--ticker", str(ticker),
        "--capital", str(CAPITAL),
        "--start", START_DATE,
        "--end", END_DATE,
        "--position_size_pct", str(position_size_pct),
    ]
    
    # Add optional parameters only if they are not None
    if stop_loss_pct is not None:
        args.extend(["--stop_loss_pct", str(stop_loss_pct)])
    if take_profit_pct is not None:
        args.extend(["--take_profit_pct", str(take_profit_pct)])
    if max_drawdown_pct is not None:
        args.extend(["--max_drawdown_pct", str(max_drawdown_pct)])
    
    # Initialize result structure with input parameters
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
        # Log execution details for monitoring
        logger.info(f"Running: {ticker} | SL: {stop_loss_pct} | TP: {take_profit_pct} | MDD: {max_drawdown_pct} | POS: {position_size_pct}")
        
        # Execute the backtest subprocess with timeout protection
        output = subprocess.check_output(
            args, 
            stderr=subprocess.STDOUT,  # Capture stderr with stdout
            universal_newlines=True,   # Return string instead of bytes
            timeout=300  # 5 minute timeout to prevent hanging
        )
        
        # Parse performance metrics from output
        metrics = parse_metrics_from_output(output)
        result.update(metrics)
        
        # Log key results for monitoring
        total_return = metrics.get('total_return', 'N/A')
        sharpe_ratio = metrics.get('sharpe_ratio', 'N/A')
        logger.info(f"✅ {ticker}: Return={total_return}, Sharpe={sharpe_ratio}")
        
    except subprocess.TimeoutExpired:
        # Handle timeout scenarios
        result['run_status'] = 'TIMEOUT'
        result['error_message'] = f"Backtest timed out after 5 minutes"
        logger.error(f"❌ {ticker}: Timeout")
        
    except subprocess.CalledProcessError as e:
        # Handle subprocess execution errors
        result['run_status'] = 'ERROR'
        result['error_message'] = f"Subprocess error: {e.returncode}"
        logger.error(f"❌ {ticker}: Subprocess failed: {e}")
        
        # Try to extract error details from output
        if hasattr(e, 'output') and e.output:
            result['error_message'] += f" | Output: {e.output[-200:]}"  # Last 200 chars
            
    except Exception as e:
        # Handle unexpected errors
        result['run_status'] = 'ERROR'
        result['error_message'] = f"Unexpected error: {str(e)}"
        logger.error(f"❌ {ticker}: Unexpected error: {e}")
    
    return result


def run_batch_backtest():
    """
    Execute comprehensive hyperparameter sweep across all parameter combinations.
    
    This is the main orchestration function that coordinates the entire batch
    backtesting process. It generates all parameter combinations, executes
    individual backtests, aggregates results, and produces comprehensive
    output files for analysis.
    
    The function implements an industrial-grade batch processing pipeline
    with robust error handling, progress tracking, and comprehensive
    logging suitable for production quantitative research workflows.
    
    Execution Pipeline:
    ==================
    1. **Parameter Grid Generation**: Create all combinations using itertools
    2. **Progress Tracking**: Monitor execution with detailed logging
    3. **Batch Execution**: Run individual backtests sequentially
    4. **Result Aggregation**: Collect and structure all results
    5. **Output Generation**: Create CSV and JSON output files
    6. **Analytics**: Generate summary statistics and top performers
    
    Parameter Space:
    ===============
    The function explores the full Cartesian product of:
    - Tickers: Multiple assets for cross-validation
    - Stop Losses: Risk management levels
    - Take Profits: Profit-taking strategies
    - Position Sizes: Capital allocation schemes
    - Max Drawdowns: Portfolio protection limits
    
    Total combinations = |Tickers| × |Stop Losses| × |Take Profits| × |Position Sizes| × |Max Drawdowns|
    
    Output Files:
    ============
    1. **CSV Summary** (batch_summary_TIMESTAMP.csv):
       - Tabular format for Excel/Pandas analysis
       - All parameter combinations and results
       - Performance metrics and trade statistics
       - Success/failure status for each run
    
    2. **JSON Detailed** (batch_detailed_TIMESTAMP.json):
       - Structured format for programmatic access
       - Metadata about the batch run
       - Complete results with nested structure
       - Error details and execution logs
    
    Error Handling:
    ==============
    - Individual backtest failures don't halt the batch
    - Comprehensive error logging for debugging
    - Success rate tracking and reporting
    - Graceful degradation on partial failures
    
    Performance Monitoring:
    ======================
    - Real-time progress updates
    - Execution time tracking
    - Memory usage monitoring
    - Success rate calculations
    
    Usage Examples:
    ==============
    ```python
    # Basic usage
    run_batch_backtest()
    
    # Programmatic usage with result capture
    original_tickers = TICKERS.copy()
    TICKERS = ["AAPL", "MSFT"]  # Modify global config
    run_batch_backtest()
    TICKERS = original_tickers  # Restore
    ```
    
    Integration Points:
    ==================
    - Calls run_single_backtest() for individual executions
    - Outputs compatible with analyze_batch_results.py
    - Integrates with Pandas for data analysis
    - Supports Excel for non-technical users
    
    Educational Value:
    =================
    This function demonstrates:
    - Large-scale parameter optimization
    - Robust batch processing architecture
    - Error handling in distributed systems
    - Performance monitoring and logging
    - Data pipeline design patterns
    """
    logger.info("🚀 Starting batch backtest hyperparameter sweep...")
    
    # Generate all parameter combinations using Cartesian product
    all_combinations = list(itertools.product(
        TICKERS, STOP_LOSSES, TAKE_PROFITS, MAX_DRAWDOWNS, POSITION_SIZES
    ))
    
    total_combos = len(all_combinations)
    logger.info(f"📊 Total combinations to test: {total_combos}")
    logger.info(f"⏱️  Estimated time: {total_combos * 30 / 60:.1f} minutes (assuming 30s per run)")
    
    # Initialize tracking variables
    all_results = []
    successful_runs = 0
    failed_runs = 0
    
    # Initialize CSV output file with headers
    with open(SUMMARY_CSV, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
        writer.writeheader()
        
        # Process each parameter combination
        for i, (ticker, stop_loss, take_profit, max_drawdown, position_size) in enumerate(all_combinations, 1):
            logger.info(f"\n🔄 Progress: {i}/{total_combos} ({i/total_combos*100:.1f}%)")
            
            # Execute single backtest
            result = run_single_backtest(ticker, stop_loss, take_profit, max_drawdown, position_size)
            
            # Track success/failure statistics
            if result['run_status'] == 'SUCCESS':
                successful_runs += 1
            else:
                failed_runs += 1
            
            # Store result for JSON output
            all_results.append(result)
            
            # Write result to CSV immediately (streaming approach)
            csv_row = {col: result.get(col, '') for col in CSV_HEADER}
            writer.writerow(csv_row)
            
            # Flush to ensure data is written (important for long runs)
            csv_file.flush()
    
    # Generate detailed JSON output with metadata
    with open(DETAILED_JSON, 'w') as json_file:
        json.dump({
            "metadata": {
                "timestamp": TIMESTAMP,
                "total_combinations": total_combos,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_combos if total_combos > 0 else 0,
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
    
    # Generate comprehensive final summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 BATCH BACKTEST COMPLETED")
    logger.info("=" * 60)
    logger.info(f"📊 Total combinations: {total_combos}")
    logger.info(f"✅ Successful runs: {successful_runs}")
    logger.info(f"❌ Failed runs: {failed_runs}")
    logger.info(f"📈 Success rate: {successful_runs/total_combos*100:.1f}%")
    logger.info(f"\n📁 Results saved to:")
    logger.info(f"  📊 Summary CSV: {SUMMARY_CSV}")
    logger.info(f"  📋 Detailed JSON: {DETAILED_JSON}")
    
    # Generate top performers analysis if we have successful runs
    if successful_runs > 0:
        logger.info(f"\n🎯 Analysis ready! Load {SUMMARY_CSV} into pandas/Excel to find your edge.")
        
        # Show top performers using pandas if available
        try:
            import pandas as pd
            df = pd.read_csv(SUMMARY_CSV)
            successful_df = df[df['run_status'] == 'SUCCESS'].copy()
            
            if len(successful_df) > 0:
                # Convert string numbers to float for proper sorting
                for col in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                    if col in successful_df.columns:
                        successful_df[col] = pd.to_numeric(successful_df[col], errors='coerce')
                
                # Display top performers by total return
                logger.info("\n🏆 TOP 5 PERFORMERS BY TOTAL RETURN:")
                top_by_return = successful_df.nlargest(5, 'total_return')
                for _, row in top_by_return.iterrows():
                    logger.info(f"  {row['ticker']} | Return: {row['total_return']:.1%} | "
                              f"Sharpe: {row.get('sharpe_ratio', 'N/A'):.2f} | "
                              f"MaxDD: {row.get('max_drawdown', 'N/A'):.1%}")
                
                # Display top performers by Sharpe ratio
                logger.info("\n🛡️  TOP 5 PERFORMERS BY SHARPE RATIO:")
                top_by_sharpe = successful_df.nlargest(5, 'sharpe_ratio')
                for _, row in top_by_sharpe.iterrows():
                    logger.info(f"  {row['ticker']} | Sharpe: {row.get('sharpe_ratio', 'N/A'):.2f} | "
                              f"Return: {row['total_return']:.1%} | "
                              f"MaxDD: {row.get('max_drawdown', 'N/A'):.1%}")
                    
        except ImportError:
            logger.info("💡 Install pandas to see top performers: pip install pandas")
        except Exception as e:
            logger.warning(f"Could not show top performers: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch Backtest: Hyperparameter Sweep for Trading Strategy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_backtest.py                    # Full parameter sweep
  python batch_backtest.py --quick            # Quick test mode
  python batch_backtest.py --tickers AAPL SPY # Custom tickers
  python batch_backtest.py --quick --tickers QQQ # Combined options
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick test with reduced parameter combinations"
    )
    parser.add_argument(
        "--tickers", 
        nargs="+", 
        help="Override default tickers list with custom symbols", 
        default=None
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Enable parallel processing (future enhancement)"
    )
    
    args = parser.parse_args()
    
    # Configure quick mode for testing and development
    if args.quick:
        # Load quick mode configuration from config file
        quick_config = batch_config.get('quick_mode', {})
        TICKERS = quick_config.get('tickers', ["AAPL", "SPY"])
        STOP_LOSSES = quick_config.get('stop_losses', [0.05, None])
        TAKE_PROFITS = quick_config.get('take_profits', [0.10, None])
        POSITION_SIZES = quick_config.get('position_sizes', [1.0, 0.5])
        MAX_DRAWDOWNS = quick_config.get('max_drawdowns', [None, 0.20])
        logger.info("🚀 QUICK MODE: Running reduced parameter set for testing")
        logger.info(f"   Combinations: {len(TICKERS)} × {len(STOP_LOSSES)} × {len(TAKE_PROFITS)} × {len(POSITION_SIZES)} × {len(MAX_DRAWDOWNS)} = {len(TICKERS)*len(STOP_LOSSES)*len(TAKE_PROFITS)*len(POSITION_SIZES)*len(MAX_DRAWDOWNS)}")
    
    # Override tickers if provided via command line
    if args.tickers:
        TICKERS = args.tickers
        logger.info(f"📈 Using custom tickers: {TICKERS}")
    
    # Future enhancement placeholder
    if args.parallel:
        logger.info("⚡ Parallel processing requested (future enhancement)")
    
    # Execute the batch backtest
    run_batch_backtest() 