"""Breakout strategies for capturing momentum and trend changes.

This package contains strategies that identify and capture significant price
movements and trend reversals through breakout detection. These strategies
work best in liquid markets with clear support/resistance levels.

Available Strategies:
    VolatilityBreakoutStrategy: Multi-factor breakout with volume and volatility confirmation
    ChannelBreakoutStrategy: Classic Donchian channel breakout system
    VolumeBreakoutStrategy: Volume spike-driven momentum capture

Example:
    >>> from .volatility_breakout import VolatilityBreakoutStrategy
    >>> from .channel_breakout import ChannelBreakoutStrategy  
    >>> from .volume_breakout import VolumeBreakoutStrategy
    
    >>> # Create breakout ensemble
    >>> strategies = [
    ...     VolatilityBreakoutStrategy({'breakout_period': 20, 'volume_multiplier': 2.0}),
    ...     ChannelBreakoutStrategy({'channel_period': 20, 'breakout_threshold': 0.01}),
    ...     VolumeBreakoutStrategy({'volume_spike_multiplier': 3.0})
    ... ]
"""

from .volatility_breakout import VolatilityBreakoutStrategy
from .channel_breakout import ChannelBreakoutStrategy
from .volume_breakout import VolumeBreakoutStrategy

__all__ = [
    'VolatilityBreakoutStrategy',
    'ChannelBreakoutStrategy',
    'VolumeBreakoutStrategy'
] 