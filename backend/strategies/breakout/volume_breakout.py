"""Volume-driven breakout strategy focusing on unusual trading activity.

This module implements a unique approach to breakout detection by prioritizing
volume analysis over price levels. It identifies periods of unusually high
trading activity combined with significant price moves, capturing momentum
driven by institutional activity or news events.

Strategy Philosophy:
- Volume precedes price movement
- Unusual volume indicates informed trading
- Price moves with volume confirmation are more sustainable
- Institutional activity creates detectable volume patterns

Volume Analysis Components:
1. Volume Spike Detection: Identifying unusually high volume
2. Price Move Validation: Ensuring volume accompanies significant price change
3. Direction Determination: Matching volume with price direction

Classes:
    VolumeBreakoutStrategy: Volume spike-driven momentum capture

Example:
    >>> # Sensitive volume breakout for active stocks
    >>> strategy = VolumeBreakoutStrategy({
    ...     'volume_period': 10,            # Short baseline
    ...     'volume_spike_multiplier': 2.5, # Moderate spike requirement
    ...     'price_move_threshold': 0.015   # 1.5% price move minimum
    ... })
    >>> signals = strategy.generate_signals(intraday_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseStrategy


class VolumeBreakoutStrategy(BaseStrategy):
    """Volume-driven breakout strategy focusing on unusual trading activity.
    
    This strategy takes a unique approach to breakout detection by prioritizing
    volume analysis over price levels. It identifies periods of unusually high
    trading activity combined with significant price moves, capturing momentum
    driven by institutional activity or news events.
    
    Strategy Philosophy:
    - Volume precedes price movement
    - Unusual volume indicates informed trading
    - Price moves with volume confirmation are more sustainable
    - Institutional activity creates detectable volume patterns
    
    Volume Analysis Components:
    1. Volume Spike Detection: Identifying unusually high volume
    2. Price Move Validation: Ensuring volume accompanies significant price change
    3. Direction Determination: Matching volume with price direction
    
    Signal Logic:
    - BUY: Volume spike + significant positive price move
    - SELL: Volume spike + significant negative price move
    - HOLD: Normal volume or insufficient price movement
    
    Key Features:
    - Adapts to changing volume patterns
    - Captures news-driven moves early
    - Works well in liquid markets
    - Complementary to price-based strategies
    
    Best Applications:
    - High-volume liquid stocks
    - News and earnings event trading
    - Intraday momentum capture
    - Institutional flow detection
    
    Limitations:
    - Requires reliable volume data
    - Less effective in thin markets
    - Can generate false signals on settlement days
    - Sensitive to unusual market conditions
    
    Example:
        >>> # Sensitive volume breakout for active stocks
        >>> strategy = VolumeBreakoutStrategy({
        ...     'volume_period': 10,            # Short baseline
        ...     'volume_spike_multiplier': 2.5, # Moderate spike requirement
        ...     'price_move_threshold': 0.015   # 1.5% price move minimum
        ... })
        >>> signals = strategy.generate_signals(intraday_data)
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'breakout'
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize volume-driven breakout strategy with spike detection.
        
        Configures the volume breakout strategy to identify unusual trading
        activity combined with significant price movements. Parameters can
        be adjusted based on the typical volume patterns of target assets.
        
        Args:
            parameters: Strategy configuration dictionary:
                - volume_period (int): Period for volume average baseline (default: 20)
                    Shorter periods = more sensitive to recent volume changes
                    Longer periods = more stable baseline, fewer false signals
                - volume_spike_multiplier (float): Volume spike threshold (default: 3.0)
                    Multiple of average volume required for spike detection
                    Higher values = require more extreme volume for signals
                    Lower values = more sensitive to volume increases
                - price_move_threshold (float): Minimum price move % (default: 0.02)
                    Percentage price change required to confirm breakout
                    Higher values = reduce noise but may miss smaller moves
                    Lower values = more signals but potentially more false positives
                    
        Parameter Guidelines:
        - High-volume stocks: Use higher volume_spike_multiplier (4.0+)
        - Volatile stocks: Use higher price_move_threshold (0.03+)
        - Intraday trading: Use shorter volume_period (10-15)
        - Daily trading: Use longer volume_period (20-30)
        
        Example:
            >>> # Conservative setup for large-cap stocks
            >>> params = {
            ...     'volume_period': 30,        # Stable baseline
            ...     'volume_spike_multiplier': 4.0,  # High threshold
            ...     'price_move_threshold': 0.025     # 2.5% minimum move
            ... }
            >>> strategy = VolumeBreakoutStrategy(params)
        """
        # Volume-focused default parameters for momentum detection
        default_params = {
            'volume_period': 20,            # 20-day volume baseline
            'volume_spike_multiplier': 3.0, # 3x average volume spike
            'price_move_threshold': 0.02    # 2% minimum price move
        }
        
        # Update with user-specified parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base strategy with volume focus
        super().__init__("Volume Breakout", default_params)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data contains volume column for volume analysis.
        
        Volume breakout strategies specifically require volume data for analysis.
        This method ensures volume information is available before processing.
        
        Args:
            data: DataFrame to validate for volume requirements
            
        Returns:
            bool: True if volume data is available, False otherwise
        """
        if 'volume' not in data.columns:
            self.logger.error("Volume column required for volume breakout strategy")
            return False
        
        # Call parent validation for additional standard checks
        return super().validate_data(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-driven breakout signals based on trading activity spikes.
        
        Implements volume-centric breakout detection by identifying periods of
        unusual trading activity combined with significant price movements.
        This approach captures momentum driven by institutional activity or
        news events that may not be apparent in price-only analysis.
        
        Implementation Process:
        1. Validate input data contains required volume information
        2. Calculate rolling volume baseline and spike thresholds
        3. Identify volume spikes exceeding normal activity
        4. Calculate price movement magnitudes
        5. Generate signals combining volume spikes with price moves
        6. Add volume analysis columns for evaluation
        7. Log volume breakout statistics and patterns
        
        Args:
            data: DataFrame containing market data with 'close' and 'volume' columns
            
        Returns:
            DataFrame with original data plus volume analysis:
                - 'signal': Trading signal (1=buy, -1=sell, 0=hold)
                - 'volume_avg': Rolling average volume baseline
                - 'volume_spike': Boolean indicator of volume spike
                - 'price_change': Period-over-period price change percentage
                - 'volume_ratio': Current volume / average volume ratio
                
        Signal Conditions:
            BUY Signal:
            - Volume > (average volume × spike multiplier)
            - Price change > +price_move_threshold
            
            SELL Signal:
            - Volume > (average volume × spike multiplier)
            - Price change < -price_move_threshold
            
        Volume Spike Analysis:
        The strategy identifies volume spikes as periods where trading volume
        significantly exceeds recent averages, indicating increased interest
        from institutions or informed traders.
        
        Example:
            >>> signals = strategy.generate_signals(stock_data)
            >>> volume_breakouts = signals[signals['volume_spike']]
            >>> avg_volume_ratio = signals['volume_ratio'].mean()
            >>> print(f"Found {len(volume_breakouts)} volume spikes")
        """
        # Step 1: Validate input data contains volume information
        if not self.validate_data(data):
            self.logger.error("Data validation failed for volume breakout strategy")
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Start with clean copy of market data
        df = data.copy()
        
        # Step 2: Extract strategy parameters for volume analysis
        period = self.parameters['volume_period']
        multiplier = self.parameters['volume_spike_multiplier']
        
        # Step 3: Calculate volume baseline and spike detection
        
        # Establish normal volume baseline using rolling average
        df['volume_avg'] = df['volume'].rolling(window=period).mean()
        
        # Identify volume spikes: current volume exceeds threshold
        df['volume_spike'] = df['volume'] > (df['volume_avg'] * multiplier)
        
        # Step 4: Calculate price movement analysis
        
        # Period-over-period price change percentage
        df['price_change'] = df['close'].pct_change()
        
        # Extract minimum price move threshold
        threshold = self.parameters['price_move_threshold']
        
        # Step 5: Initialize signals with hold position
        df['signal'] = 0
        
        # Step 6: Generate volume-confirmed breakout signals
        
        # BUY: Volume spike with significant positive price movement
        buy_condition = df['volume_spike'] & (df['price_change'] > threshold)
        
        # SELL: Volume spike with significant negative price movement
        sell_condition = df['volume_spike'] & (df['price_change'] < -threshold)
        
        # Apply volume breakout signals
        df.loc[buy_condition, 'signal'] = 1   # Buy on volume-confirmed upside move
        df.loc[sell_condition, 'signal'] = -1 # Sell on volume-confirmed downside move
        
        # Step 7: Add comprehensive volume analysis columns
        
        # Volume ratio: current volume relative to average (strength indicator)
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        # Step 8: Log detailed volume breakout statistics
        buy_signals = buy_condition.sum()
        sell_signals = sell_condition.sum()
        total_signals = buy_signals + sell_signals
        total_volume_spikes = df['volume_spike'].sum()
        
        # Volume pattern analysis
        avg_volume_ratio = df['volume_ratio'].mean()
        max_volume_ratio = df['volume_ratio'].max()
        spike_rate = (total_volume_spikes / len(df)) * 100
        
        self.logger.info(f"Volume breakout analysis: {total_signals} total signals "
                        f"({buy_signals} volume buy spikes, {sell_signals} volume sell spikes)")
        self.logger.info(f"Volume patterns: {total_volume_spikes} total spikes "
                        f"({spike_rate:.1f}% spike rate)")
        self.logger.debug(f"Volume statistics: avg_ratio={avg_volume_ratio:.2f}, "
                         f"max_ratio={max_volume_ratio:.2f}, multiplier={multiplier}")
        
        return df 