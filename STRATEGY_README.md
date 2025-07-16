## Overview

This is a modular trading strategy framework designed for systematic alpha generation. The architecture provides plug-and-play strategy components that can be combined, filtered, and orchestrated for maximum edge extraction.

### Core Philosophy
- **Modular**: Every strategy is a self-contained, swappable component
- **Composable**: Strategies can be combined with different logic (majority, unanimous, any)
- **Filterable**: Regime detection and sentiment overlays act as signal gates
- **Scalable**: Batch testing across multiple assets and time periods
- **Edge-Focused**: Built-in performance analytics and leaderboards

## Architecture

```
backend/
├── strategies/
│   ├── base.py                    # Abstract base class + composer
│   ├── sma_ema_rsi.py            # Trend/momentum strategies
│   ├── vol_breakout.py           # Breakout strategies  
│   ├── mean_reversion.py         # Mean reversion strategies
│   ├── regime_switch.py          # Regime detection & gating
│   ├── sentiment_overlay.py      # Sentiment filtering
│   └── __init__.py               # Module exports
├── strategy_engine.py            # Core orchestration engine
├── simulate.py                   # Trading simulator
├── metrics.py                    # Performance analytics
└── data_loader.py               # Data fetching

config/
└── config.yaml                  # Unified configuration file
```

##Quick Start

### 1. Run the Demo
```bash
python demo_strategy_suite.py
```

### 2. Test Individual Strategy
```bash
python backend/strategy_engine.py --strategy sma_ema_rsi --ticker AAPL
```

### 3. Run Batch Backtest
```bash
python backend/strategy_engine.py --batch AAPL MSFT GOOGL --combinations
```

### 4. List Available Strategies
```bash
python backend/strategy_engine.py --list-strategies
```

## Strategy Arsenal

### Trend/Momentum Strategies
- **`sma_ema_rsi`**: SMA/EMA crossover with RSI filter
- **`crossover`**: Pure moving average crossover
- **`rsi`**: RSI momentum strategy

### Breakout Strategies  
- **`volatility_breakout`**: Volume-confirmed price breakouts
- **`channel_breakout`**: Donchian channel breakouts
- **`volume_breakout`**: Volume spike breakouts

### Mean Reversion Strategies
- **`bollinger_mean_reversion`**: Bollinger Band reversals
- **`zscore_mean_reversion`**: Z-score based reversals
- **`rsi_mean_reversion`**: RSI divergence reversals
- **`mean_reversion_combo`**: Multi-indicator consensus

### Meta Strategies
- **Regime Gating**: Only trade in favorable market regimes
- **Sentiment Overlay**: Filter signals by market sentiment
- **Strategy Combinations**: Combine multiple strategies with voting logic

## Usage Examples

### Basic Strategy Usage
```python
from backend.strategies import SmaEmaRsiStrategy
from backend.strategy_engine import StrategyEngine

# Create strategy
strategy = SmaEmaRsiStrategy({
    'fast_period': 10,
    'slow_period': 50,
    'rsi_period': 14,
    'use_rsi_filter': True
})

# Generate signals
signals_df = strategy.generate_signals(price_data)

# Run full backtest
engine = StrategyEngine()
result = engine.run_single_strategy(strategy, 'AAPL')
```

### Strategy Combinations
```python
from backend.strategies import create_strategy_combo

# Combine multiple strategies
combo = create_strategy_combo(
    strategy_names=['sma_ema_rsi', 'volatility_breakout', 'bollinger_mean_reversion'],
    combination_method='majority'  # Require 2+ strategies to agree
)

signals_df = combo.generate_combined_signals(price_data)
```

### Regime-Gated Strategies
```python
from backend.strategies import create_regime_gated_strategy

# Only trade in favorable regimes
gated_strategy = create_regime_gated_strategy(
    base_strategy=my_strategy,
    regime_method='sma_slope',  # Trend-following friendly regime
    regime_params={'sma_period': 200, 'slope_threshold': 0.001}
)
```

### Sentiment-Filtered Strategies
```python
from backend.strategies import create_sentiment_overlay

# Filter signals by market sentiment
sentiment_strategy = create_sentiment_overlay(
    base_strategy=my_strategy,
    provider_type='mock',  # or 'newsapi', 'finnhub'
    sentiment_params={'sentiment_threshold': 0.2}
)
```

## Configuration

Edit `config/config.yaml` to customize:

### Strategy Parameters
```json
{
  "strategies": {
    "sma_ema_rsi": {
      "enabled": true,
      "parameters": {
        "fast_period": 10,
        "slow_period": 50,
        "rsi_period": 14,
        "use_rsi_filter": true
      }
    }
  }
}
```

### Risk Management
```json
{
  "risk_management": {
    "stop_loss_pct": null,
    "take_profit_pct": null,
    "max_drawdown_pct": 0.20,
    "position_size_pct": 1.0
  }
}
```

### Batch Testing
```json
{
  "batch_testing": {
    "default_tickers": ["AAPL", "MSFT", "GOOGL"],
    "test_combinations": true,
    "combination_methods": ["majority", "unanimous"]
  }
}
```

## Advanced Architectures

### Multi-Layer Strategy Stack
```python
# 1. Base strategies
trend_strategy = engine.create_strategy('sma_ema_rsi')
breakout_strategy = engine.create_strategy('volatility_breakout')
mean_revert_strategy = engine.create_strategy('bollinger_mean_reversion')

# 2. Add regime gating
gated_trend = create_regime_gated_strategy(trend_strategy, 'combined')
gated_breakout = create_regime_gated_strategy(breakout_strategy, 'atr_volatility')

# 3. Add sentiment overlay
sentiment_mean_revert = create_sentiment_overlay(mean_revert_strategy, 'mock')

# 4. Combine into ultimate weapon
ultimate_strategies = [gated_trend, gated_breakout, sentiment_mean_revert]
result = engine.run_strategy_combo(ultimate_strategies, 'AAPL', 'majority')
```

### Regime Switching Strategy
```python
from backend.strategies.regime_switch import RegimeSwitchStrategy, RegimeDetector

# Use different strategies for different market regimes
trend_strategy = engine.create_strategy('sma_ema_rsi')
mean_revert_strategy = engine.create_strategy('bollinger_mean_reversion')

regime_detector = RegimeDetector('sma_slope')
switching_strategy = RegimeSwitchStrategy(
    trend_strategy=trend_strategy,      # For trending markets
    mean_revert_strategy=mean_revert_strategy,  # For ranging markets
    regime_detector=regime_detector
)
```

## Performance Analytics

### Built-in Metrics
- **Returns**: Total return, CAGR, period returns
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown, volatility
- **Trading**: Win rate, profit factor, average trade, total trades
- **Risk-Adjusted**: Calmar ratio, recovery factor, expectancy

### Leaderboard System
```python
# Get strategy rankings
leaderboard = engine.get_strategy_leaderboard(results, metric='sharpe_ratio')
print(leaderboard.head(10))
```

### Output Files
Results are automatically saved to `backend/outputs/`:
- `strategy_results_summary_TIMESTAMP.csv`: Performance summary
- `equity_STRATEGY_TICKER_TIMESTAMP.csv`: Equity curves  
- `trades_STRATEGY_TICKER_TIMESTAMP.csv`: Trade logs

## Regime Detection Methods

### Available Regime Types
- **`sma_slope`**: Trend regime based on SMA slope
- **`atr_volatility`**: Volatility regime based on ATR percentiles
- **`vix`**: Fear regime based on VIX levels
- **`combined`**: Multi-factor regime detection

### Custom Regime Detection
```python
from backend.strategies.regime_switch import RegimeDetector

detector = RegimeDetector('sma_slope', {
    'sma_period': 200,
    'slope_threshold': 0.001
})

regime_signals = detector.detect_regime(price_data)
```

## Sentiment Integration

### Supported Providers
- **Mock**: Deterministic sentiment for testing
- **NewsAPI**: Real news sentiment via newsapi.org
- **Finnhub**: Financial news sentiment via finnhub.io

### Adding New Sentiment Providers
```python
from backend.strategies.sentiment_overlay import SentimentProvider

class MyCustomSentimentProvider(SentimentProvider):
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        # Return sentiment score between -1 and 1
        return my_sentiment_logic(ticker, date)
```

## Batch Operations

### Batch Backtest
```python
results = engine.run_batch_backtest(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    strategies=['sma_ema_rsi', 'volatility_breakout'],
    include_combinations=True
)
```

### Strategy Tournament
```python
# Test all enabled strategies on multiple assets
results = engine.run_batch_backtest(
    tickers=config['batch_testing']['default_tickers'],
    strategies=None,  # Use all enabled strategies
    include_combinations=True
)

# Get winners
winners = engine.get_strategy_leaderboard(results, 'sharpe_ratio')
```

## Extending the Framework

### Adding New Strategies
```python
from backend.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, parameters=None):
        super().__init__("My Custom Strategy", parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Your signal logic here
        df['signal'] = 0  # 1=buy, -1=sell, 0=hold
        
        return df
```

### Register New Strategy
```python
# Add to backend/strategy_engine.py
STRATEGY_REGISTRY['my_custom'] = MyCustomStrategy
```

## CLI Reference

### Strategy Engine Commands
```bash
# Test single strategy
python backend/strategy_engine.py --strategy sma_ema_rsi --ticker AAPL

# Batch test multiple tickers
python backend/strategy_engine.py --batch AAPL MSFT GOOGL --combinations

# Use custom config
python backend/strategy_engine.py --config my_config.json --strategy volatility_breakout

# List available strategies
python backend/strategy_engine.py --list-strategies
```

### Legacy Momentum Backtest
```bash
# Use new modular system (default)
python backend/momentum_backtest.py --strategy crossover --fast-period 5 --slow-period 20

# Use legacy system
python backend/momentum_backtest.py --legacy --ticker AAPL
```

## Best Practices

### Strategy Development
1. **Start Simple**: Begin with basic strategies, add complexity gradually
2. **Parameter Optimization**: Use batch testing to find optimal parameters
3. **Regime Awareness**: Consider regime gating for trend-following strategies
4. **Risk Management**: Always implement proper position sizing and risk controls
5. **Walk-Forward Testing**: Test strategies on out-of-sample data

### Portfolio Construction
1. **Diversification**: Combine strategies with different alpha sources
2. **Correlation Analysis**: Avoid highly correlated strategies
3. **Regime Diversification**: Mix trend-following and mean-reverting strategies
4. **Sentiment Integration**: Use sentiment for timing and risk management

### Production Deployment
1. **Monitoring**: Set up alerts for strategy performance degradation
2. **Position Limits**: Implement maximum position and concentration limits
3. **Risk Budgets**: Allocate risk budgets across strategies
4. **Regular Rebalancing**: Periodically rebalance strategy allocations

## Risk Warnings

- **Backtest Overfitting**: Extensive parameter optimization can lead to overfitting
- **Regime Changes**: Strategies may fail during unprecedented market conditions
- **Sentiment Lag**: News sentiment often lags price action
- **Transaction Costs**: Real trading costs may significantly impact returns
- **Liquidity**: Strategy capacity depends on underlying asset liquidity

## API Integrations

### News Sentiment APIs
- **NewsAPI**: Get API key from newsapi.org
- **Finnhub**: Get API key from finnhub.io

### Data Sources
- **yfinance**: Default price data source
- **Custom**: Extend DataLoader for alternative data sources

## System Requirements

- Python 3.8+
- pandas, numpy, requests
- yfinance for market data
- Optional: newsapi, finnhub for sentiment

## Getting Started Checklist

- [ ] Clone repository
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run demo: `python demo_strategy_suite.py`
- [ ] Review config: `config/config.yaml`
- [ ] Test individual strategy: `python backend/strategy_engine.py --strategy sma_ema_rsi`
- [ ] Run batch backtest: `python backend/strategy_engine.py --batch AAPL MSFT`
- [ ] Analyze results in `backend/outputs/`

---