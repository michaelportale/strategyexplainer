"""Meta strategies for advanced strategy composition and market regime adaptation.

This package contains advanced strategies that enhance or modify the behavior of
base strategies through regime detection, sentiment analysis, and strategy combination.
These meta strategies act as wrappers or filters for core trading strategies.

Available Strategies:
    RegimeDetector: Market regime detection system
    RegimeGatedStrategy: Strategy execution gated by market regime
    RegimeSwitchStrategy: Switches between strategies based on market regime
    SentimentProvider: Sentiment analysis interface
    SentimentOverlayStrategy: Sentiment-filtered strategy execution
    SentimentMeanReversionStrategy: Sentiment-enhanced mean reversion

Example:
    >>> from .regime_switch import RegimeDetector, RegimeGatedStrategy, RegimeSwitchStrategy
    >>> from .sentiment_overlay import SentimentProvider, SentimentOverlayStrategy
    
    >>> # Create regime-aware strategy
    >>> base_strategy = SomeBaseStrategy()
    >>> regime_detector = RegimeDetector('sma_slope')
    >>> gated_strategy = RegimeGatedStrategy(base_strategy, regime_detector)
    
    >>> # Create sentiment-filtered strategy  
    >>> sentiment_provider = MockSentimentProvider()
    >>> sentiment_strategy = SentimentOverlayStrategy(base_strategy, sentiment_provider)
"""

from .regime_switch import (
    RegimeDetector,
    RegimeGatedStrategy,
    RegimeSwitchStrategy,
    regime_gate_decorator,
    create_regime_gated_strategy,
    create_regime_switch_strategy
)

from .sentiment_overlay import (
    SentimentProvider,
    MockSentimentProvider,
    NewsAPISentimentProvider,
    FinnhubSentimentProvider,
    SentimentOverlayStrategy,
    SentimentMeanReversionStrategy,
    create_sentiment_overlay
)

__all__ = [
    # Regime Switching
    'RegimeDetector',
    'RegimeGatedStrategy',
    'RegimeSwitchStrategy',
    'regime_gate_decorator',
    'create_regime_gated_strategy',
    'create_regime_switch_strategy',
    
    # Sentiment Analysis
    'SentimentProvider',
    'MockSentimentProvider',
    'NewsAPISentimentProvider', 
    'FinnhubSentimentProvider',
    'SentimentOverlayStrategy',
    'SentimentMeanReversionStrategy',
    'create_sentiment_overlay'
] 