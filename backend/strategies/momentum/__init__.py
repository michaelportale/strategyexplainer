"""Momentum and trend-following strategies.

This package contains strategies that identify and follow market momentum and trends.
These strategies work best in trending market conditions and can be combined with
mean reversion strategies for balanced portfolio construction.

Available Strategies:
    SmaEmaRsiStrategy: Advanced trend following with RSI momentum filter
    CrossoverStrategy: Classic moving average crossover system
    RsiStrategy: RSI-based momentum strategy
    MacdCrossoverStrategy: MACD signal line crossover strategy
    MacdZeroCrossStrategy: MACD zero line crossover strategy
    AdxTrendFilterStrategy: ADX trend strength filter with moving averages
    AdxDirectionalStrategy: Pure ADX directional indicator strategy
    AdxComboStrategy: Combined ADX strategy with multiple confirmations

Example:
    >>> from .sma_ema_crossover import SmaEmaRsiStrategy, CrossoverStrategy
    >>> from .rsi_momentum import RsiStrategy
    >>> from .macd_crossover import MacdCrossoverStrategy, MacdZeroCrossStrategy
    >>> from .adx_trend_filter import AdxTrendFilterStrategy, AdxDirectionalStrategy, AdxComboStrategy
    
    >>> # Create momentum ensemble
    >>> strategies = [
    ...     SmaEmaRsiStrategy({'fast_period': 10, 'slow_period': 50}),
    ...     CrossoverStrategy({'fast_period': 5, 'slow_period': 20}),
    ...     RsiStrategy({'rsi_period': 14}),
    ...     MacdCrossoverStrategy({'fast_period': 12, 'slow_period': 26}),
    ...     AdxTrendFilterStrategy({'adx_threshold': 25})
    ... ]
"""

from .sma_ema_crossover import SmaEmaRsiStrategy, CrossoverStrategy
from .rsi_momentum import RsiStrategy
from .macd_crossover import MacdCrossoverStrategy, MacdZeroCrossStrategy
from .adx_trend_filter import AdxTrendFilterStrategy, AdxDirectionalStrategy, AdxComboStrategy

__all__ = [
    'SmaEmaRsiStrategy',
    'CrossoverStrategy', 
    'RsiStrategy',
    'MacdCrossoverStrategy',
    'MacdZeroCrossStrategy',
    'AdxTrendFilterStrategy',
    'AdxDirectionalStrategy',
    'AdxComboStrategy'
] 