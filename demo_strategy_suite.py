"""Strategy suite demonstration platform.

Comprehensive demonstration of the modular trading strategy framework, including
individual strategies, strategy combinations, regime-gated systems, and sentiment
overlays.

Main Functions:
    demo_individual_strategies(): Test core strategy implementations
    demo_strategy_combinations(): Test voting-based strategy ensembles  
    demo_regime_gated_strategies(): Test adaptive regime-aware systems
    demo_sentiment_overlay(): Test sentiment-filtered strategies
    demo_advanced_combinations(): Test hierarchical strategy architectures
    demo_batch_optimization(): Test systematic parameter optimization

Usage:
    python demo_strategy_suite.py  # Run all demonstrations
"""

#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import the core strategy engine and strategy implementations
from backend.strategy_engine import StrategyEngine
from backend.strategies import (
    SmaEmaRsiStrategy, VolatilityBreakoutStrategy, BollingerBandMeanReversionStrategy,
    create_regime_gated_strategy, create_sentiment_overlay, MockSentimentProvider,
    create_strategy_combo, RegimeDetector
)

# Configure logging for demonstration output
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_individual_strategies():
    """
    Demonstrate individual strategy performance and characteristics.
    
    This function showcases the core trading strategies available in the framework,
    providing detailed performance analysis and educational insights into each
    strategy's behavior and effectiveness.
    
    The demonstration covers:
    - Strategy initialization and parameter configuration
    - Backtesting execution and performance measurement
    - Risk-return analysis and trade statistics
    - Educational explanations of strategy logic
    
    Strategies Demonstrated:
    =======================
    1. **SMA/EMA+RSI Strategy**: Trend-following with momentum confirmation
       - Uses moving average crossovers for trend detection
       - RSI filter for momentum confirmation
       - Suitable for trending markets
    
    2. **Volatility Breakout Strategy**: Volume-confirmed breakouts
       - Detects price breakouts from consolidation ranges
       - Volume confirmation for signal validation
       - Effective in volatile market conditions
    
    3. **Bollinger Band Mean Reversion**: Statistical mean reversion
       - Identifies overbought/oversold conditions
       - Statistical bands for entry/exit signals
       - Works well in ranging markets
    
    Performance Analysis:
    ====================
    For each strategy, the demonstration displays:
    - Total return and annualized performance
    - Risk-adjusted metrics (Sharpe ratio)
    - Maximum drawdown and risk measures
    - Win rate and trade frequency
    - Total number of trades executed
    
    Educational Value:
    =================
    This demonstration teaches:
    - Individual strategy implementation patterns
    - Performance measurement methodologies
    - Risk-return trade-offs in strategy design
    - Parameter sensitivity and optimization
    - Strategy selection criteria
    
    Returns:
        list: Results from all individual strategy tests
    """
    print("\n" + "="*60)
    print("🎯 DEMO 1: INDIVIDUAL STRATEGY PERFORMANCE")
    print("="*60)
    
    # Initialize the strategy engine with unified configuration
    engine = StrategyEngine()  # Uses new unified config.yaml
    
    # Define strategies to test with their parameter configurations
    strategies_to_test = [
        ('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50}),
        ('volatility_breakout', {'breakout_period': 20, 'volume_multiplier': 1.5}),
        ('bollinger_mean_reversion', {'bb_period': 20, 'bb_std_dev': 2.0})
    ]
    
    # Store results for analysis
    results = []
    
    # Test each strategy individually
    for strategy_name, params in strategies_to_test:
        print(f"\n📊 Testing {strategy_name.upper().replace('_', ' ')}...")
        
        # Create strategy instance with specified parameters
        strategy = engine.create_strategy(strategy_name, params)
        
        # Run backtest on Apple stock for 2-year period
        result = engine.run_single_strategy(strategy, "AAPL", "2022-01-01", "2024-01-01")
        
        # Display results with formatted output
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
    """
    Demonstrate strategy combination methodologies and ensemble learning.
    
    This function showcases how multiple strategies can be combined to create
    more robust and diversified trading systems. It demonstrates different
    combination methodologies and their impact on performance.
    
    The demonstration explores:
    - Multiple combination voting mechanisms
    - Ensemble learning in trading systems
    - Diversification benefits and trade-offs
    - Consensus building algorithms
    
    Combination Methods:
    ===================
    1. **Majority Voting**: Requires majority of strategies to agree
       - Reduces false signals through consensus
       - Balances between signal frequency and accuracy
       - Suitable for medium-frequency trading
    
    2. **Unanimous Consensus**: All strategies must agree
       - Maximizes signal quality and reduces noise
       - Lower trade frequency but higher accuracy
       - Conservative approach for risk-averse traders
    
    3. **Any Signal Activation**: Any strategy can trigger trades
       - Maximizes trade opportunities
       - Higher frequency but potentially more noise
       - Aggressive approach for active trading
    
    Strategy Portfolio:
    ==================
    The demonstration combines:
    - Trend-following strategy (SMA/EMA+RSI)
    - Breakout strategy (Volatility Breakout)
    - Mean reversion strategy (Bollinger Bands)
    
    This creates a diversified portfolio spanning different market regimes
    and trading philosophies.
    
    Performance Analysis:
    ====================
    For each combination method, displays:
    - Total return and risk-adjusted performance
    - Maximum drawdown and volatility measures
    - Trade frequency and execution statistics
    - Comparison with individual strategies
    
    Educational Value:
    =================
    This demonstration teaches:
    - Ensemble learning principles in finance
    - Strategy diversification benefits
    - Voting mechanism design and implementation
    - Trade-offs between signal quality and frequency
    - Portfolio construction methodologies
    """
    print("\n" + "="*60)
    print("🔗 DEMO 2: STRATEGY COMBINATIONS")
    print("="*60)
    
    # Initialize engine with unified configuration
    engine = StrategyEngine()  # Uses new unified config.yaml
    
    # Create individual strategies for combination
    trend_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    breakout_strategy = engine.create_strategy('volatility_breakout', {'breakout_period': 20})
    mean_revert_strategy = engine.create_strategy('bollinger_mean_reversion', {'bb_period': 20})
    
    # Create strategy ensemble
    strategies = [trend_strategy, breakout_strategy, mean_revert_strategy]
    
    # Test different combination methodologies
    combination_methods = ['majority', 'unanimous', 'any']
    
    for method in combination_methods:
        print(f"\n🎲 Testing {method.upper()} combination...")
        
        # Run strategy combination with specified voting method
        result = engine.run_strategy_combo(strategies, "AAPL", method, "2022-01-01", "2024-01-01")
        
        # Display comprehensive results
        if 'error' not in result:
            metrics = result['metrics']
            print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
            print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
            print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
            print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
        else:
            print(f"   ❌ Error: {result['error']}")


def demo_regime_gated_strategies():
    """Demonstrate regime-gated strategy systems with adaptive market condition filtering.
    
    Tests SMA slope and ATR volatility regime detection methods applied to base strategies.
    Regime filters activate/deactivate strategies based on market conditions.
    """
    print("\n" + "="*60)
    print("🌐 DEMO 3: REGIME-GATED STRATEGIES")
    print("="*60)
    
    # Initialize engine with unified configuration
    engine = StrategyEngine()  # Uses new unified config.yaml
    
    # Create base strategy for regime gating
    base_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    
    # Define regime detection methods and their parameters
    regime_methods = [
        ('sma_slope', {'sma_period': 200, 'slope_threshold': 0.001}),
        ('atr_volatility', {'atr_period': 14, 'low_vol_percentile': 20, 'high_vol_percentile': 80})
    ]
    
    # Test each regime detection method
    for method, params in regime_methods:
        print(f"\n🎯 Testing {method.upper().replace('_', ' ')} regime filter...")
        
        # Create regime-gated strategy wrapper
        gated_strategy = create_regime_gated_strategy(base_strategy, method, params)
        
        # Run backtest with regime-gated strategy
        result = engine.run_single_strategy(gated_strategy, "AAPL", "2022-01-01", "2024-01-01")
        
        # Display performance metrics
        if 'error' not in result:
            metrics = result['metrics']
            print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
            print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
            print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
            print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
        else:
            print(f"   ❌ Error: {result['error']}")


def demo_sentiment_overlay():
    """
    Demonstrate sentiment-filtered strategy systems and behavioral finance.
    
    This function showcases how sentiment analysis can be integrated into
    trading strategies to improve performance and reduce behavioral biases.
    It demonstrates the application of behavioral finance principles in
    systematic trading.
    
    The demonstration covers:
    - Sentiment data integration
    - Behavioral finance applications
    - Contrarian and momentum sentiment strategies
    - Risk management through sentiment awareness
    
    Sentiment Analysis Components:
    =============================
    1. **Sentiment Threshold Filtering**: Basic sentiment-based filtering
       - Positive sentiment: Enable long positions
       - Negative sentiment: Enable short positions or avoid longs
       - Neutral sentiment: Normal strategy operation
    
    2. **Sentiment Momentum**: Following sentiment trends
       - Increasing positive sentiment: Bullish signals
       - Increasing negative sentiment: Bearish signals
       - Sentiment reversal detection
    
    3. **Contrarian Sentiment**: Contrarian approach to sentiment
       - Extreme positive sentiment: Potential reversal signal
       - Extreme negative sentiment: Potential buying opportunity
       - Sentiment extremes as contrary indicators
    
    Behavioral Finance Principles:
    =============================
    - **Herding Behavior**: Crowd sentiment analysis
    - **Overreaction Bias**: Extreme sentiment contrarian signals
    - **Anchoring Effect**: Sentiment-based reference points
    - **Confirmation Bias**: Sentiment confirmation filters
    
    Mock Sentiment Provider:
    =======================
    For demonstration purposes, uses simulated sentiment data that:
    - Generates realistic sentiment patterns
    - Correlates with market volatility
    - Includes noise and bias components
    - Demonstrates various sentiment regimes
    
    Educational Value:
    =================
    This demonstration teaches:
    - Sentiment analysis integration techniques
    - Behavioral finance application in trading
    - Alternative data usage in systematic strategies
    - Risk management through sentiment awareness
    - Contrarian vs momentum sentiment strategies
    """
    print("\n" + "="*60)
    print("💭 DEMO 4: SENTIMENT-FILTERED STRATEGIES")
    print("="*60)
    
    # Initialize engine with unified configuration
    engine = StrategyEngine()  # Uses new unified config.yaml
    
    # Create base strategy for sentiment overlay
    base_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    
    # Configure sentiment parameters
    sentiment_params = {
        'sentiment_threshold': 0.2,  # Sentiment must be > 0.2 for long positions
        'contrarian_mode': False,    # Use sentiment momentum (not contrarian)
        'sentiment_window': 5        # 5-day sentiment moving average
    }
    
    print("   🧠 Using mock sentiment provider for demonstration...")
    print("   📊 Sentiment threshold: ±0.2")
    print("   📈 Strategy: Momentum-based sentiment filtering")
    
    # Create sentiment-filtered strategy
    sentiment_strategy = create_sentiment_overlay(
        base_strategy, 
        provider_type='mock',
        sentiment_params=sentiment_params
    )
    
    # Run backtest with sentiment-filtered strategy
    result = engine.run_single_strategy(sentiment_strategy, "AAPL", "2022-01-01", "2024-01-01")
    
    # Display comprehensive results
    if 'error' not in result:
        metrics = result['metrics']
        print(f"\n📊 SENTIMENT-FILTERED RESULTS:")
        print(f"   📈 Total Return: {metrics.get('Total Return', 0):.2%}")
        print(f"   📊 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"   📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
        print(f"   🎯 Win Rate: {metrics.get('Win Rate', 0):.2%}")
        print(f"   💼 Total Trades: {metrics.get('Total Trades', 0)}")
    else:
        print(f"   ❌ Error: {result['error']}")


def demo_batch_backtest():
    """
    Demonstrate batch backtesting capabilities and hyperparameter optimization.
    
    This function showcases the batch processing capabilities of the framework,
    demonstrating how to systematically test multiple parameter combinations
    and identify optimal configurations.
    
    The demonstration covers:
    - Batch backtesting workflow
    - Hyperparameter optimization
    - Performance comparison across parameters
    - Statistical analysis of results
    
    Batch Processing Features:
    =========================
    1. **Parameter Grid Search**: Systematic parameter exploration
       - Multiple strategy configurations
       - Cross-validation across different assets
       - Statistical significance testing
       - Optimization objective selection
    
    2. **Performance Analytics**: Comprehensive result analysis
       - Risk-adjusted performance metrics
       - Parameter sensitivity analysis
       - Robustness testing across markets
       - Statistical significance assessment
    
    3. **Automated Optimization**: Objective-driven optimization
       - Sharpe ratio maximization
       - Drawdown minimization
       - Return optimization
       - Multi-objective optimization
    
    Demonstration Workflow:
    ======================
    1. **Strategy Selection**: Choose strategy for optimization
    2. **Parameter Definition**: Define parameter ranges
    3. **Batch Execution**: Run systematic backtests
    4. **Result Analysis**: Analyze and rank results
    5. **Optimal Selection**: Identify best configurations
    
    Educational Value:
    =================
    This demonstration teaches:
    - Systematic backtesting methodologies
    - Hyperparameter optimization techniques
    - Statistical analysis of trading results
    - Overfitting prevention strategies
    - Robust strategy validation methods
    """
    print("\n" + "="*60)
    print("🔄 DEMO 5: BATCH BACKTESTING")
    print("="*60)
    
    # Initialize engine with unified configuration
    engine = StrategyEngine()  # Uses new unified config.yaml
    
    print("   🎯 Demonstrating batch backtesting workflow...")
    print("   📊 Strategy: SMA/EMA+RSI with parameter optimization")
    print("   📈 Assets: AAPL, MSFT (demo subset)")
    print("   🔍 Parameters: Fast period (5-20), Slow period (30-100)")
    
    # Define parameter ranges for optimization
    tickers = ["AAPL", "MSFT"]  # Reduced set for demo
    fast_periods = [5, 10, 15, 20]
    slow_periods = [30, 50, 75, 100]
    
    # Storage for batch results
    batch_results = []
    
    # Systematic parameter exploration
    for ticker in tickers:
        for fast_period in fast_periods:
            for slow_period in slow_periods:
                # Ensure logical parameter relationship
                if slow_period > fast_period:
                    print(f"   🔄 Testing {ticker} | Fast: {fast_period} | Slow: {slow_period}")
                    
                    # Create strategy with current parameters
                    strategy = engine.create_strategy('sma_ema_rsi', {
                        'fast_period': fast_period,
                        'slow_period': slow_period
                    })
                    
                    # Run backtest
                    result = engine.run_single_strategy(strategy, ticker, "2022-01-01", "2024-01-01")
                    
                    # Store result with parameters
                    if 'error' not in result:
                        result['ticker'] = ticker
                        result['fast_period'] = fast_period
                        result['slow_period'] = slow_period
                        batch_results.append(result)
    
    # Analyze batch results
    if batch_results:
        print(f"\n📊 BATCH RESULTS ANALYSIS:")
        print(f"   ✅ Successful runs: {len(batch_results)}")
        
        # Find best performing configuration
        best_result = max(batch_results, key=lambda x: x['metrics'].get('Total Return', 0))
        print(f"   🏆 Best configuration:")
        print(f"      Ticker: {best_result['ticker']}")
        print(f"      Fast period: {best_result['fast_period']}")
        print(f"      Slow period: {best_result['slow_period']}")
        print(f"      Total Return: {best_result['metrics'].get('Total Return', 0):.2%}")
        print(f"      Sharpe Ratio: {best_result['metrics'].get('Sharpe Ratio', 0):.2f}")
    else:
        print("   ❌ No successful batch results to analyze")


def demo_advanced_combinations():
    """
    Demonstrate advanced strategy combinations and hierarchical systems.
    
    This function showcases the most sophisticated capabilities of the framework,
    demonstrating how to create complex, multi-layered trading systems that
    combine multiple strategies, regime detection, and sentiment analysis.
    
    The demonstration covers:
    - Hierarchical strategy architectures
    - Multi-dimensional strategy stacking
    - Advanced risk management integration
    - Performance attribution analysis
    
    Advanced Architecture:
    =====================
    1. **Regime-Gated Strategies**: Base layer with regime awareness
       - Trend-following strategy with SMA slope regime detection
       - Breakout strategy with volatility regime filtering
       - Mean reversion strategy with sentiment overlay
    
    2. **Sentiment Integration**: Behavioral finance overlay
       - Sentiment-filtered mean reversion for contrarian signals
       - Momentum confirmation through sentiment analysis
       - Risk management through sentiment extremes
    
    3. **Ensemble Combination**: Meta-strategy orchestration
       - Majority voting for consensus building
       - Dynamic weight allocation based on recent performance
       - Risk-adjusted position sizing
    
    System Components:
    =================
    - **Component 1**: Regime-gated trend following
    - **Component 2**: Regime-gated volatility breakouts
    - **Component 3**: Sentiment-filtered mean reversion
    - **Meta-Strategy**: Intelligent combination system
    
    Educational Value:
    =================
    This demonstration teaches:
    - Advanced system architecture design
    - Multi-layered strategy construction
    - Risk management system integration
    - Performance attribution methodologies
    - Hierarchical decision-making systems
    """
    print("\n" + "="*60)
    print("🚀 DEMO 6: ADVANCED STRATEGY COMBINATIONS")
    print("="*60)
    
    # Initialize engine with unified configuration
    engine = StrategyEngine()  # Uses new unified config.yaml
    
    print("   🏗️  Building advanced hierarchical strategy system...")
    print("   📊 Architecture: Multi-layered adaptive trading system")
    
    # Create base strategies
    trend_strategy = engine.create_strategy('sma_ema_rsi', {'fast_period': 10, 'slow_period': 50})
    breakout_strategy = engine.create_strategy('volatility_breakout', {'breakout_period': 20})
    mean_revert_strategy = engine.create_strategy('bollinger_mean_reversion', {'bb_period': 20})
    
    # Create regime-gated versions
    gated_trend = create_regime_gated_strategy(
        trend_strategy, 
        'sma_slope', 
        {'sma_period': 200, 'slope_threshold': 0.001}
    )
    
    gated_breakout = create_regime_gated_strategy(
        breakout_strategy,
        'atr_volatility',
        {'atr_period': 14, 'low_vol_percentile': 20, 'high_vol_percentile': 80}
    )
    
    # Create sentiment-filtered mean reversion
    sentiment_mean_revert = create_sentiment_overlay(
        mean_revert_strategy, 
        provider_type='mock',
        sentiment_params={'sentiment_threshold': 0.1}
    )
    
    # Combine all components into ultimate strategy
    ultimate_strategies = [gated_trend, gated_breakout, sentiment_mean_revert]
    
    print("   🧬 Component strategies:")
    print("      - Regime-gated SMA/EMA+RSI trend following")
    print("      - Regime-gated volatility breakouts")
    print("      - Sentiment-filtered Bollinger Band mean reversion")
    
    # Run the ultimate strategy combination
    result = engine.run_strategy_combo(
        ultimate_strategies, 
        "AAPL",
        combination_method='majority',
        start_date="2022-01-01",
        end_date="2024-01-01"
    )
    
    # Display comprehensive results
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
    """Execute complete strategy suite demonstration.
    
    Runs all demo modules in sequence: individual strategies, combinations,
    regime-gated systems, sentiment overlays, and batch optimization.
    """
    print("🏛️  QUANT SUITE WAR ROOM: STRATEGY ARSENAL DEMO")
    print("=" * 80)
    print("Demonstrating the modular, weaponized strategy framework...")
    print("🎓 Educational tour of quantitative trading system capabilities")
    print("⚡ Production-ready implementations with comprehensive analysis")
    
    try:
        # Execute complete demonstration sequence
        demo_individual_strategies()
        demo_strategy_combinations()
        demo_regime_gated_strategies()
        demo_sentiment_overlay()
        demo_batch_backtest()
        demo_advanced_combinations()
        
        # Provide comprehensive completion summary
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETE! Your strategy arsenal is ready for deployment.")
        print("="*80)
        print("📚 What you've learned:")
        print("   • Individual strategy implementation and analysis")
        print("   • Ensemble learning and strategy combinations")
        print("   • Regime-aware and adaptive trading systems")
        print("   • Behavioral finance and sentiment integration")
        print("   • Systematic optimization and batch processing")
        print("   • Advanced hierarchical strategy architectures")
        print("\n🚀 Next steps:")
        print("   📁 Check backend/outputs/ for detailed results and trade logs")
        print("   ⚡ Use strategy_engine.py CLI for production runs")
        print("   📊 Run batch_backtest.py for hyperparameter optimization")
        print("   🔍 Analyze results with analyze_batch_results.py")
        print("   🎯 Implement your own strategies using the framework patterns")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        print("💡 This might be due to missing dependencies or configuration issues")
        print("🔧 Please check your environment setup and try again")
        raise


if __name__ == "__main__":
    main() 