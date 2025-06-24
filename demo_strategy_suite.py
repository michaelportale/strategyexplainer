#!/usr/bin/env python3
"""Demo Script for Modular Strategy Suite.

Demonstrates the capabilities of the weaponized strategy lab:
- Individual strategies
- Strategy combinations  
- Regime-gated strategies
- Sentiment-filtered strategies
- Batch backtesting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import the strategy engine and strategies
from backend.strategy_engine import StrategyEngine
from backend.strategies import (
    SmaEmaRsiStrategy, VolatilityBreakoutStrategy, BollingerBandMeanReversionStrategy,
    create_regime_gated_strategy, create_sentiment_overlay, MockSentimentProvider,
    create_strategy_combo, RegimeDetector
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_individual_strategies():
    """Demo individual strategy performance."""
    print("\n" + "="*60)
    print("🎯 DEMO 1: INDIVIDUAL STRATEGY PERFORMANCE")
    print("="*60)
    
    # Initialize engine
    engine = StrategyEngine("config/strategy_config.json")
    
    # Test individual strategies
    strategies_to_test = [
        ('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50}),
        ('volatility_breakout', {'breakout_period': 20, 'volume_multiplier': 1.5}),
        ('bollinger_mean_reversion', {'bb_period': 20, 'bb_std_dev': 2.0})
    ]
    
    results = []
    
    for strategy_name, params in strategies_to_test:
        print(f"\n📊 Testing {strategy_name.upper().replace('_', ' ')}...")
        
        strategy = engine.create_strategy(strategy_name, params)
        result = engine.run_single_strategy(strategy, "AAPL", "2022-01-01", "2024-01-01")
        
        if 'error' not in result:
            metrics = result['metrics']
            print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
            print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
            print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
            print(f"   🎯 Win Rate: {metrics.get('Win Rate', 0):.2%}")
            print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
            results.append(result)
        else:
            print(f"   ❌ Error: {result['error']}")
    
    return results


def demo_strategy_combinations():
    """Demo strategy combination performance."""
    print("\n" + "="*60)
    print("🔗 DEMO 2: STRATEGY COMBINATIONS")
    print("="*60)
    
    engine = StrategyEngine("config/strategy_config.json")
    
    # Create individual strategies
    trend_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    breakout_strategy = engine.create_strategy('volatility_breakout', {'breakout_period': 20})
    mean_revert_strategy = engine.create_strategy('bollinger_mean_reversion', {'bb_period': 20})
    
    strategies = [trend_strategy, breakout_strategy, mean_revert_strategy]
    combination_methods = ['majority', 'unanimous', 'any']
    
    for method in combination_methods:
        print(f"\n🎲 Testing {method.upper()} combination...")
        
        result = engine.run_strategy_combo(strategies, "AAPL", method, "2022-01-01", "2024-01-01")
        
        if 'error' not in result:
            metrics = result['metrics']
            print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
            print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
            print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
            print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
        else:
            print(f"   ❌ Error: {result['error']}")


def demo_regime_gated_strategies():
    """Demo regime-gated strategy performance."""
    print("\n" + "="*60)
    print("🌐 DEMO 3: REGIME-GATED STRATEGIES")
    print("="*60)
    
    engine = StrategyEngine("config/strategy_config.json")
    
    # Create base strategy
    base_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    
    # Test different regime detection methods
    regime_methods = [
        ('sma_slope', {'sma_period': 200, 'slope_threshold': 0.001}),
        ('atr_volatility', {'atr_period': 14, 'low_vol_percentile': 20, 'high_vol_percentile': 80})
    ]
    
    for method, params in regime_methods:
        print(f"\n🎯 Testing {method.upper().replace('_', ' ')} regime filter...")
        
        # Create regime-gated strategy
        gated_strategy = create_regime_gated_strategy(base_strategy, method, params)
        result = engine.run_single_strategy(gated_strategy, "AAPL", "2022-01-01", "2024-01-01")
        
        if 'error' not in result:
            metrics = result['metrics']
            print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
            print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
            print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
            print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
        else:
            print(f"   ❌ Error: {result['error']}")


def demo_sentiment_overlay():
    """Demo sentiment-filtered strategy performance."""
    print("\n" + "="*60)
    print("💭 DEMO 4: SENTIMENT-FILTERED STRATEGIES")
    print("="*60)
    
    engine = StrategyEngine("config/strategy_config.json")
    
    # Create base strategy
    base_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    
    # Create sentiment overlay (using mock provider for demo)
    sentiment_params = {
        'sentiment_threshold': 0.2,
        'sentiment_period': 5,
        'bearish_sentiment_threshold': -0.3,
        'bullish_sentiment_threshold': 0.3
    }
    
    print("\n💭 Testing MOCK SENTIMENT overlay...")
    
    sentiment_strategy = create_sentiment_overlay(
        base_strategy, 
        provider_type='mock',
        sentiment_params=sentiment_params
    )
    
    result = engine.run_single_strategy(sentiment_strategy, "AAPL", "2022-01-01", "2024-01-01")
    
    if 'error' not in result:
        metrics = result['metrics']
        print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
        print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
        print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
    else:
        print(f"   ❌ Error: {result['error']}")


def demo_batch_backtest():
    """Demo batch backtesting across multiple tickers."""
    print("\n" + "="*60)
    print("⚡ DEMO 5: BATCH BACKTESTING")
    print("="*60)
    
    engine = StrategyEngine("config/strategy_config.json")
    
    # Test strategies on multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    strategies = ['sma_ema_rsi', 'volatility_breakout']
    
    print(f"\n🚀 Running batch backtest on {len(tickers)} tickers with {len(strategies)} strategies...")
    
    results = engine.run_batch_backtest(
        tickers=tickers,
        strategies=strategies,
        include_combinations=True
    )
    
    # Get leaderboard
    leaderboard = engine.get_strategy_leaderboard(results, 'sharpe_ratio')
    
    if not leaderboard.empty:
        print("\n🏆 STRATEGY LEADERBOARD (Top 10 by Sharpe Ratio):")
        print("-" * 80)
        top_10 = leaderboard.head(10)
        
        for _, row in top_10.iterrows():
            print(f"{row['rank']:2d}. {row['strategy']:<30} ({row['ticker']}) - "
                  f"Sharpe: {row['sharpe_ratio']:6.2f} | "
                  f"Return: {row['total_return']:7.2%} | "
                  f"Trades: {row['total_trades']:3.0f}")
    
    # Save results
    engine.save_results(results)
    print(f"\n💾 Saved {len(results)} backtest results to backend/outputs/")


def demo_advanced_combinations():
    """Demo advanced strategy architectures."""
    print("\n" + "="*60)
    print("🎪 DEMO 6: ADVANCED STRATEGY ARCHITECTURES")
    print("="*60)
    
    engine = StrategyEngine("config/strategy_config.json")
    
    # Create sophisticated strategy combo
    print("\n🎯 Building WEAPON OF MASS ALPHA...")
    
    # 1. Base strategies
    trend_strategy = engine.create_strategy('sma_ema_rsi', {
        'fast_period': 10, 'slow_period': 50, 'use_rsi_filter': True
    })
    
    breakout_strategy = engine.create_strategy('volatility_breakout', {
        'breakout_period': 20, 'volume_multiplier': 1.8, 'use_atr_filter': True
    })
    
    mean_revert_strategy = engine.create_strategy('bollinger_mean_reversion', {
        'bb_period': 20, 'use_rsi_filter': True, 'exit_at_middle': True
    })
    
    # 2. Create regime-gated versions
    regime_detector = RegimeDetector('combined', {
        'sma_period': 200, 'slope_threshold': 0.001,
        'atr_period': 14, 'vix_threshold': 25.0
    })
    
    from backend.strategies.regime_switch import RegimeGatedStrategy
    gated_trend = RegimeGatedStrategy(trend_strategy, regime_detector)
    gated_breakout = RegimeGatedStrategy(breakout_strategy, regime_detector)
    
    # 3. Add sentiment overlay to mean reversion
    sentiment_mean_revert = create_sentiment_overlay(
        mean_revert_strategy, 
        provider_type='mock',
        sentiment_params={'sentiment_threshold': 0.1}
    )
    
    # 4. Combine all into ultimate strategy
    ultimate_strategies = [gated_trend, gated_breakout, sentiment_mean_revert]
    
    print("   🧬 Component strategies:")
    print("      - Regime-gated SMA/EMA+RSI trend following")
    print("      - Regime-gated volatility breakouts")
    print("      - Sentiment-filtered Bollinger Band mean reversion")
    
    result = engine.run_strategy_combo(
        ultimate_strategies, 
        "AAPL",
        combination_method='majority',
        start_date="2022-01-01",
        end_date="2024-01-01"
    )
    
    if 'error' not in result:
        metrics = result['metrics']
        print(f"\n🚀 ULTIMATE STRATEGY RESULTS:")
        print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
        print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
        print(f"   🎯 Win Rate: {metrics.get('Win Rate', 0):.2%}")
        print(f"   💰 Profit Factor: {metrics.get('Profit Factor', 0):.2f}")
        print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
    else:
        print(f"   ❌ Error: {result['error']}")


def main():
    """Run all demos."""
    print("🏛️  QUANT SUITE WAR ROOM: STRATEGY ARSENAL DEMO")
    print("=" * 80)
    print("Demonstrating the modular, weaponized strategy framework...")
    
    try:
        # Run all demos
        demo_individual_strategies()
        demo_strategy_combinations()
        demo_regime_gated_strategies()
        demo_sentiment_overlay()
        demo_batch_backtest()
        demo_advanced_combinations()
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETE! Your strategy arsenal is ready for deployment.")
        print("📁 Check backend/outputs/ for detailed results and trade logs.")
        print("⚡ Use strategy_engine.py CLI for production runs.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 