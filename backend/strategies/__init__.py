"""Modular Trading Strategies Package.

This package provides a modular framework for implementing and combining
trading strategies. All strategies inherit from BaseStrategy and implement
the generate_signals() method.

Core Components:
- BaseStrategy: Abstract base class for all strategies
- StrategyComposer: Combine multiple strategies
- RegimeDetector: Market regime detection
- SentimentProvider: Sentiment analysis integration

Available Strategies:
- Trend/Momentum: SmaEmaRsiStrategy, CrossoverStrategy, RsiStrategy
- Breakout: VolatilityBreakoutStrategy, ChannelBreakoutStrategy, VolumeBreakoutStrategy  
- Mean Reversion: BollingerBandMeanReversionStrategy, ZScoreMeanReversionStrategy, RSIMeanReversionStrategy
- Meta Strategies: RegimeGatedStrategy, RegimeSwitchStrategy, SentimentOverlayStrategy
"""

# Core base classes
from .base import BaseStrategy, StrategyComposer

# Trend/Momentum strategies
from .sma_ema_rsi import (
    SmaEmaRsiStrategy,
    CrossoverStrategy, 
    RsiStrategy
)

# Breakout strategies
from .vol_breakout import (
    VolatilityBreakoutStrategy,
    ChannelBreakoutStrategy,
    VolumeBreakoutStrategy
)

# Mean reversion strategies
from .mean_reversion import (
    BollingerBandMeanReversionStrategy,
    ZScoreMeanReversionStrategy,
    RSIMeanReversionStrategy,
    MeanReversionComboStrategy
)

# Regime switching
from .regime_switch import (
    RegimeDetector,
    RegimeGatedStrategy,
    RegimeSwitchStrategy,
    regime_gate_decorator,
    create_regime_gated_strategy,
    create_regime_switch_strategy
)

# Sentiment analysis
from .sentiment_overlay import (
    SentimentProvider,
    MockSentimentProvider,
    NewsAPISentimentProvider,
    FinnhubSentimentProvider,
    SentimentOverlayStrategy,
    SentimentMeanReversionStrategy,
    create_sentiment_overlay
)

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    # Trend/Momentum
    'sma_ema_rsi': SmaEmaRsiStrategy,
    'crossover': CrossoverStrategy,
    'rsi': RsiStrategy,
    
    # Breakout
    'volatility_breakout': VolatilityBreakoutStrategy,
    'channel_breakout': ChannelBreakoutStrategy,
    'volume_breakout': VolumeBreakoutStrategy,
    
    # Mean Reversion
    'bollinger_mean_reversion': BollingerBandMeanReversionStrategy,
    'zscore_mean_reversion': ZScoreMeanReversionStrategy,
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'mean_reversion_combo': MeanReversionComboStrategy,
    
    # Sentiment
    'sentiment_mean_reversion': SentimentMeanReversionStrategy,
}

# Convenience functions
def get_strategy(name: str, parameters: dict = None) -> BaseStrategy:
    """Get strategy instance by name.
    
    Args:
        name: Strategy name
        parameters: Strategy parameters
        
    Returns:
        Strategy instance
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[name]
    return strategy_class(parameters=parameters)


def list_strategies() -> list:
    """List all available strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def create_strategy_combo(strategy_names: list, 
                         parameters_list: list = None,
                         combination_method: str = 'majority') -> StrategyComposer:
    """Create combination of strategies.
    
    Args:
        strategy_names: List of strategy names
        parameters_list: List of parameter dicts (same length as strategy_names)
        combination_method: How to combine signals ('majority', 'unanimous', 'any')
        
    Returns:
        StrategyComposer instance
    """
    if parameters_list is None:
        parameters_list = [None] * len(strategy_names)
    
    if len(parameters_list) != len(strategy_names):
        raise ValueError("parameters_list must have same length as strategy_names")
    
    strategies = []
    for name, params in zip(strategy_names, parameters_list):
        strategy = get_strategy(name, params)
        strategies.append(strategy)
    
    return StrategyComposer(strategies, combination_method)


__all__ = [
    # Base classes
    'BaseStrategy',
    'StrategyComposer',
    
    # Trend/Momentum
    'SmaEmaRsiStrategy',
    'CrossoverStrategy', 
    'RsiStrategy',
    
    # Breakout
    'VolatilityBreakoutStrategy',
    'ChannelBreakoutStrategy',
    'VolumeBreakoutStrategy',
    
    # Mean Reversion
    'BollingerBandMeanReversionStrategy',
    'ZScoreMeanReversionStrategy',
    'RSIMeanReversionStrategy',
    'MeanReversionComboStrategy',
    
    # Regime Switching
    'RegimeDetector',
    'RegimeGatedStrategy',
    'RegimeSwitchStrategy',
    'regime_gate_decorator',
    'create_regime_gated_strategy',
    'create_regime_switch_strategy',
    
    # Sentiment
    'SentimentProvider',
    'MockSentimentProvider',
    'NewsAPISentimentProvider', 
    'FinnhubSentimentProvider',
    'SentimentOverlayStrategy',
    'SentimentMeanReversionStrategy',
    'create_sentiment_overlay',
    
    # Utilities
    'STRATEGY_REGISTRY',
    'get_strategy',
    'list_strategies',
    'create_strategy_combo'
] 