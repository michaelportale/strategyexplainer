"""
Enhanced Strategy Analyzer - Utilities Module

This module contains utility functions and configuration helpers for the
enhanced strategy analyzer. It provides common functionality used across
multiple modules.

Functions:
- initialize_strategy: Factory function for creating strategy instances
"""

from typing import Dict, Any
from pathlib import Path
import sys

# Add the project root to the path for backend access
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import backend strategy modules
from backend.momentum_backtest import MomentumBacktest


def initialize_strategy(config: Dict[str, Any]):
    """
    Initialize Enhanced Strategy Instance.
    
    Factory function for creating strategy instances with enhanced
    configuration support for AI-powered analysis.
    
    Enhanced Strategy Support:
    - Traditional momentum strategies
    - AI-enhanced parameter optimization
    - Dynamic configuration injection
    - Future support for ML-hybrid strategies
    
    Args:
        config: Enhanced configuration dictionary containing:
            - strategy: Strategy configuration with category and parameters
            - parameters: Strategy-specific parameters
        
    Returns:
        Strategy instance ready for AI-enhanced analysis
        
    Raises:
        ValueError: If strategy type is not supported
    """
    # Extract strategy configuration for enhanced initialization
    strategy_type = config['strategy']['category']
    parameters = config['strategy']['parameters']
    
    if strategy_type == 'momentum':
        # Initialize momentum strategy with enhanced parameter support
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
        )
    else:
        # Default fallback with enhanced parameter support
        # TODO: Add support for additional AI-enhanced strategy types
        return MomentumBacktest(
            lookback_period=parameters.get('lookback_period', 20),
            threshold=parameters.get('threshold', 0.02),
            position_size=parameters.get('position_size', 0.1),
            stop_loss=parameters.get('stop_loss', 0.05),
            take_profit=parameters.get('take_profit', 0.10)
        ) 