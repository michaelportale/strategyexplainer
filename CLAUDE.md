# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular trading strategy framework called "Strategy Explainer" designed for systematic alpha generation. The architecture provides plug-and-play strategy components that can be combined, filtered, and orchestrated for maximum edge extraction.

## Common Development Commands

### Running Strategies
```bash
# Run demo strategy suite
python demo_strategy_suite.py

# Test individual strategy
python backend/strategy_engine.py --strategy sma_ema_rsi --ticker AAPL

# Run batch backtest
python backend/strategy_engine.py --batch AAPL MSFT GOOGL --combinations

# List available strategies
python backend/strategy_engine.py --list-strategies

# Legacy momentum backtest
python backend/momentum_backtest.py --strategy crossover --fast-period 5 --slow-period 20
```

### Batch Testing
```bash
# Quick batch test (8 combinations)
python batch_backtest.py --quick

# Full parameter sweep (720+ combinations)
python batch_backtest.py

# Analyze batch results
python analyze_batch_results.py
```

### Streamlit Applications
```bash
# Run enhanced strategy analyzer
streamlit run app/pages/enhanced_strategy_analyzer.py

# Run basic strategy analyzer
streamlit run app/pages/strategy_analyzer.py
```

### Testing
```bash
# Run tests using pytest
pytest tests/

# Run specific test
pytest tests/test_momentum_backtest.py
```

### Package Management
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Architecture

### Core Components

- **Strategy Engine** (`backend/strategy_engine.py`): Central orchestration layer for strategy execution, backtesting, and results aggregation
- **Strategy Base** (`backend/strategies/base.py`): Abstract base class and strategy composer
- **Data Loader** (`backend/data_loader.py`): Handles data fetching from yfinance
- **Simulator** (`backend/simulate.py`): Trading simulation engine
- **Metrics** (`backend/metrics.py`): Performance analytics and calculations

### Strategy Categories

1. **Momentum/Trend Strategies** (`backend/strategies/momentum/`):
   - SMA/EMA crossover with RSI filter
   - Pure moving average crossover
   - RSI momentum strategy

2. **Breakout Strategies** (`backend/strategies/breakout/`):
   - Volume-confirmed price breakouts
   - Donchian channel breakouts
   - Volume spike breakouts

3. **Mean Reversion Strategies** (`backend/strategies/mean_reversion/`):
   - Bollinger Band reversals
   - Z-score based reversals
   - RSI divergence reversals

4. **Meta Strategies** (`backend/strategies/meta/`):
   - Regime detection and switching
   - Sentiment overlay filtering

### Frontend Applications

- **Streamlit Apps** (`app/`): Interactive web interfaces for strategy analysis
- **Enhanced Analyzer** (`app/pages/enhanced_analyzer/`): Advanced analysis pipeline with AI-powered insights

### Configuration

- **Config Manager** (`config/config_manager.py`): Centralized configuration handling
- **YAML Config** (`config/config.yaml`): Main configuration file for strategy parameters, risk management, and batch testing settings

## Key Development Patterns

### Strategy Creation
All strategies inherit from `BaseStrategy` and implement the `generate_signals()` method. Strategies are registered in the `STRATEGY_REGISTRY` in `strategy_engine.py`.

### Strategy Combinations
Strategies can be combined using different voting methods (majority, unanimous, any) through the strategy composer system.

### Risk Management
Integrated position sizing, stop losses, take profits, and maximum drawdown controls are configured through the config system.

### Batch Operations
The system supports extensive parameter sweeps across multiple assets and strategy combinations for robust edge discovery.

## Output Files

Results are automatically saved to `backend/outputs/`:
- CSV files: Performance summaries and equity curves
- JSON files: Detailed metadata and parameters
- PNG files: Charts and visualizations
- Trade logs: Detailed transaction records

## Dependencies

Key dependencies include:
- `yfinance`: Market data fetching
- `pandas`: Data manipulation
- `streamlit`: Web interface
- `plotly`: Interactive charts
- `quantstats`: Performance analytics
- `openai`: AI-powered analysis
- `scipy`: Scientific computing

## Logging and Error Handling

### Centralized Logging Framework

The application uses a centralized logging framework located in `backend/utils/logging_config.py`:

```python
from backend.utils.logging_config import LoggerManager, StrategyError, DataError

# Get a logger for your module
logger = LoggerManager.get_logger('module_name')

# Use custom exceptions with context
raise StrategyError("Invalid parameters", context={'param': 'value'})
```

### Logging Standards

- **DEBUG**: Detailed technical information for debugging
- **INFO**: General application flow and important events  
- **WARNING**: Potentially harmful situations that don't stop execution
- **ERROR**: Error events that allow application to continue
- **CRITICAL**: Serious errors that may cause application to abort

### Error Handling Patterns

1. **Graceful Degradation**: Use fallback values and safe defaults
2. **Context-Rich Errors**: Include relevant context in exception messages
3. **User-Friendly Messages**: Display helpful error messages in frontend
4. **Comprehensive Logging**: Log all errors with full context for debugging

### Common Decorators

```python
from backend.utils.logging_config import with_error_handling, with_performance_logging

@with_error_handling(fallback_value=None, log_errors=True)
@with_performance_logging(threshold_ms=1000.0)
def my_function():
    # Function implementation
    pass
```

### Custom Exception Classes

- `StrategyError`: Strategy-related errors
- `DataError`: Data loading/validation errors
- `ConfigurationError`: Configuration-related errors
- `SimulationError`: Trading simulation errors  
- `ValidationError`: Input validation errors

### Best Practices

- Always validate inputs and provide meaningful error messages
- Use the centralized logging framework for consistent formatting
- Implement graceful fallbacks to prevent application crashes
- Add context to exceptions to aid debugging
- Log performance metrics for slow operations