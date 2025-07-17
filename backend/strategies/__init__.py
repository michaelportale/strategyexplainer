"""Modular Trading Strategies Package.

This package provides a modular framework for implementing and combining
trading strategies. All strategies inherit from BaseStrategy and implement
the generate_signals() method.

The strategies are now organized by category for better maintainability:
- momentum/: Trend-following and momentum strategies
- breakout/: Breakout and volatility strategies  
- mean_reversion/: Mean reversion and contrarian strategies
- meta/: Advanced strategy composition and regime detection

Core Components:
- BaseStrategy: Abstract base class for all strategies
- StrategyComposer: Combine multiple strategies
- RegimeDetector: Market regime detection
- SentimentProvider: Sentiment analysis integration

Available Strategy Categories:
- Momentum: SmaEmaRsiStrategy, CrossoverStrategy, RsiStrategy
- Breakout: VolatilityBreakoutStrategy, ChannelBreakoutStrategy, VolumeBreakoutStrategy  
- Mean Reversion: BollingerBandMeanReversionStrategy, ZScoreMeanReversionStrategy, 
                  RSIMeanReversionStrategy, MeanReversionComboStrategy
- Meta Strategies: RegimeGatedStrategy, RegimeSwitchStrategy, SentimentOverlayStrategy
"""

# Core base classes
from .base import BaseStrategy, StrategyComposer

# Momentum strategies
from .momentum import (
    SmaEmaRsiStrategy,
    CrossoverStrategy, 
    RsiStrategy
)

# Breakout strategies
from .breakout import (
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

# Meta strategies (regime switching and sentiment)
from .meta import (
    RegimeDetector,
    RegimeGatedStrategy,
    RegimeSwitchStrategy,
    regime_gate_decorator,
    create_regime_gated_strategy,
    create_regime_switch_strategy,
    SentimentProvider,
    MockSentimentProvider,
    NewsAPISentimentProvider,
    FinnhubSentimentProvider,
    SentimentOverlayStrategy,
    SentimentMeanReversionStrategy,
    create_sentiment_overlay
)

# Strategy registry is now handled dynamically via StrategyRegistry singleton
# All strategies are automatically registered when their classes are defined

# Convenience functions
def get_strategy(name: str, parameters: dict = None) -> BaseStrategy:
    """Get strategy instance by name.
    
    Args:
        name: Strategy name from StrategyRegistry
        parameters: Strategy parameters
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name not found
    """
    from .base import StrategyRegistry
    
    registry = StrategyRegistry.get_instance()
    strategy_class = registry.get_strategy_class(name)
    return strategy_class(parameters=parameters)


def list_strategies() -> list:
    """List all available strategy names."""
    from .base import StrategyRegistry
    
    registry = StrategyRegistry.get_instance()
    return registry.list_strategies()


def list_strategies_by_category() -> dict:
    """List strategies grouped by category."""
    from .base import StrategyRegistry
    
    registry = StrategyRegistry.get_instance()
    return registry.list_strategies_by_category()





def create_strategy_combo(strategy_names: list, 
                         parameters_list: list = None,
                         combination_method: str = 'majority') -> StrategyComposer:
    """Create combination of strategies.
    
    Args:
        strategy_names: List of strategy names from StrategyRegistry
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
    
    # Momentum strategies
    'SmaEmaRsiStrategy',
    'CrossoverStrategy', 
    'RsiStrategy',
    
    # Breakout strategies
    'VolatilityBreakoutStrategy',
    'ChannelBreakoutStrategy',
    'VolumeBreakoutStrategy',
    
    # Mean reversion strategies
    'BollingerBandMeanReversionStrategy',
    'ZScoreMeanReversionStrategy',
    'RSIMeanReversionStrategy',
    'MeanReversionComboStrategy',
    
    # Meta strategies
    'RegimeDetector',
    'RegimeGatedStrategy',
    'RegimeSwitchStrategy',
    'regime_gate_decorator',
    'create_regime_gated_strategy',
    'create_regime_switch_strategy',
    'SentimentProvider',
    'MockSentimentProvider',
    'NewsAPISentimentProvider', 
    'FinnhubSentimentProvider',
    'SentimentOverlayStrategy',
    'SentimentMeanReversionStrategy',
    'create_sentiment_overlay',
    
    # Utilities
    'get_strategy',
    'list_strategies',
    'list_strategies_by_category',
    'create_strategy_combo'
] 