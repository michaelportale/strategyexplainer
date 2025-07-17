"""Mean reversion strategies for contrarian trading and market normalization capture.

This package implements sophisticated mean reversion strategies based on the principle
that prices tend to revert to their statistical mean over time. These strategies
excel in range-bound markets and during periods of market overreaction.

Strategy Philosophy:
"Buy blood, sell euphoria" - Mean reversion strategies profit from temporary price
dislocations by betting on the return to normalcy. They work on the assumption that
extreme price movements are often temporary and will be corrected by market forces.

Available Strategies:
    BollingerBandMeanReversionStrategy: Volatility band-based reversion
    ZScoreMeanReversionStrategy: Pure statistical mean reversion
    RSIMeanReversionStrategy: RSI-based momentum strategy with divergence detection
    MeanReversionComboStrategy: Multi-indicator consensus approach
    Rsi2Strategy: RSI-2 ultra-short-term mean reversion strategy
    Rsi2TrendFilterStrategy: RSI-2 with mandatory trend filter

Example:
    >>> from .bollinger_bands import BollingerBandMeanReversionStrategy
    >>> from .zscore_reversion import ZScoreMeanReversionStrategy
    >>> from .rsi_divergence import RSIMeanReversionStrategy
    >>> from .combo_reversion import MeanReversionComboStrategy
    >>> from .rsi2_strategy import Rsi2Strategy, Rsi2TrendFilterStrategy
    
    >>> # Create mean reversion ensemble
    >>> strategies = [
    ...     BollingerBandMeanReversionStrategy({'bb_period': 20, 'bb_std_dev': 2.5}),
    ...     ZScoreMeanReversionStrategy({'z_threshold': 2.0, 'lookback_period': 50}),
    ...     RSIMeanReversionStrategy({'use_divergence': True}),
    ...     MeanReversionComboStrategy({'require_consensus': True}),
    ...     Rsi2Strategy({'oversold_threshold': 15, 'overbought_threshold': 85})
    ... ]
"""

from .bollinger_bands import BollingerBandMeanReversionStrategy
from .zscore_reversion import ZScoreMeanReversionStrategy
from .rsi_divergence import RSIMeanReversionStrategy
from .combo_reversion import MeanReversionComboStrategy
from .rsi2_strategy import Rsi2Strategy, Rsi2TrendFilterStrategy

__all__ = [
    'BollingerBandMeanReversionStrategy',
    'ZScoreMeanReversionStrategy',
    'RSIMeanReversionStrategy',
    'MeanReversionComboStrategy',
    'Rsi2Strategy',
    'Rsi2TrendFilterStrategy'
] 