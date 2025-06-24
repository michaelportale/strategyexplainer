# 🚀 Batch Backtest & Hyperparameter Sweep Guide

## Overview

This batch backtesting system allows you to run **hundreds of parameter combinations** systematically, discovering which settings actually work across different assets and market conditions. No more single "lucky" backtests—this gives you statistical confidence in your edge.

## 📁 Files

- **`batch_backtest.py`** - Main batch runner that executes parameter sweeps
- **`analyze_batch_results.py`** - Comprehensive analysis and visualization of results
- **`backend/momentum_backtest.py`** - Your core backtester (enhanced with new parameters)

## 🎯 Quick Start

### 1. Run Quick Test (8 combinations)
```bash
python batch_backtest.py --quick
```

### 2. Run Full Sweep (180+ combinations)
```bash
python batch_backtest.py
```

### 3. Analyze Results
```bash
python analyze_batch_results.py
```

## ⚙️ Parameter Grid

The system tests every combination of:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **Tickers** | AAPL, MSFT, SPY, QQQ, TSLA | Different assets |
| **Stop Loss** | 2%, 5%, 10%, None | Risk management levels |
| **Take Profit** | 5%, 10%, 20%, None | Profit taking levels |
| **Position Size** | 100%, 75%, 50% | Capital allocation per trade |
| **Max Drawdown** | None, 15%, 20% | Portfolio halt triggers |

**Total Combinations:** 5 tickers × 4 stop losses × 4 take profits × 3 position sizes × 3 max drawdowns = **720 combinations**

## 📊 What You Get

### CSV Output (`batch_summary_TIMESTAMP.csv`)
- All parameter combinations tested
- Key performance metrics for each
- Easy to import into Excel/pandas for analysis

### JSON Output (`batch_detailed_TIMESTAMP.json`)
- Complete metadata about the run
- Detailed parameters used
- Full result history

### Analysis Report
- Top performers by multiple metrics
- Parameter impact analysis
- Risk-return distributions
- Correlation insights
- Visual heatmaps

## 🔍 Key Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall strategy performance |
| **Sharpe Ratio** | Risk-adjusted returns |
| **Max Drawdown** | Worst peak-to-trough loss |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Calmar Ratio** | Annual return / Max drawdown |
| **Kelly Criterion** | Optimal position sizing |

## 🎨 Analysis Features

### Top Performers
```
🏆 TOP 10 PERFORMERS BY TOTAL RETURN:
  1. AAPL | SL:None TP:None | Ret:183.2% Sharpe:-2.52 DD:-202.7%
  2. AAPL | SL:0.05 TP:None | Ret:167.8% Sharpe:-3.00 DD:-202.7%
  ...
```

### Parameter Impact
```
📊 By Ticker:
        Avg_Return  Std_Return  Count  Avg_Sharpe
ticker                                          
AAPL         1.385       0.445      4      -3.203
SPY          0.763       0.135      4      -2.690
```

### Risk-Return Analysis
- Categorizes strategies by risk/return buckets
- Identifies best risk-adjusted performers
- Shows distribution across risk profiles

### Correlation Analysis
- Which parameters drive performance
- Asset-specific insights
- Risk factor relationships

### Visual Heatmaps
- Parameter performance matrices
- Stop loss vs Take profit effectiveness
- Asset vs Position size analysis

## 🛠️ Customization

### Custom Parameter Grid
Edit the top of `batch_backtest.py`:

```python
TICKERS = ["YOUR", "CUSTOM", "TICKERS"]
STOP_LOSSES = [0.03, 0.07, None]  # Your preferred levels
TAKE_PROFITS = [0.08, 0.15, None]
# ... etc
```

### Custom Assets
```bash
python batch_backtest.py --tickers NVDA AMZN GOOGL
```

### Quick Testing
```bash
python batch_backtest.py --quick  # Reduced parameter set
```

## 📈 Edge Discovery Process

### 1. Identify Patterns
Look for parameter combinations that consistently perform well across:
- Multiple assets
- Different market conditions
- Various risk profiles

### 2. Avoid Overfitting
- Focus on parameters that work across assets, not just one
- Prefer consistent moderate performers over extreme outliers
- Validate on out-of-sample data

### 3. Risk Management First
- Prioritize strategies with acceptable max drawdown
- Look for positive Sharpe ratios (risk-adjusted returns)
- Consider correlation with market conditions

### 4. Portfolio Construction
- Combine multiple parameter sets for diversification
- Size positions based on Kelly criterion
- Regular rebalancing based on performance

## 🚨 Important Notes

### Computing Resources
- Full sweep = 720 combinations × ~30 seconds = ~6 hours
- Use `--quick` for testing (8 combinations, ~2 minutes)
- Consider running overnight for full analysis

### Data Requirements
- Ensure yfinance can fetch all ticker data
- Check date ranges have sufficient history
- Handle missing data gracefully

### Statistical Significance
- More combinations = better statistical confidence
- Focus on patterns that appear across multiple assets
- Don't overfit to historical data

## 🎯 Next Steps

### After Your First Sweep:
1. **Analyze Results:** Look for clear winners and patterns
2. **Refine Parameters:** Narrow grid around promising areas
3. **Out-of-Sample Test:** Validate on different time periods
4. **Live Paper Trading:** Test with real market conditions
5. **Scale Gradually:** Start small, increase size with confidence

### Advanced Usage:
- **Multi-Strategy Sweeps:** Test different signal generation methods
- **Market Regime Analysis:** Separate bull/bear/sideways periods  
- **Sector Analysis:** Group by industry/sector performance
- **Correlation Analysis:** Build uncorrelated strategy portfolios

## 💡 Pro Tips

1. **Start Small:** Run `--quick` first to verify everything works
2. **Save Everything:** All results are timestamped and preserved
3. **Focus on Robustness:** Prefer strategies that work across assets
4. **Mind the Drawdown:** High returns with huge drawdowns aren't sustainable
5. **Validate Forward:** Always test on out-of-sample data
6. **Combine Strategies:** Diversify across multiple parameter sets

---

**🎯 Remember:** The goal isn't to find the single "best" backtest, but to discover robust parameter combinations that provide edge across different market conditions. Your batch results are a statistical foundation for systematic trading decisions.

**Ready to unlock your edge? Start with `python batch_backtest.py --quick` and let the data guide your strategy evolution.** 