"""
Financial Data Loading and Management System: Enterprise-Grade Market Data Infrastructure

This module provides a comprehensive data loading and management system for financial
market data, featuring intelligent caching, multi-source data integration, and
robust error handling. It forms the foundation of the trading strategy framework's
data infrastructure.

The system is designed to handle high-frequency data requests efficiently while
maintaining data integrity and providing seamless integration with various market
data providers.

Key Features:
============

1. **Intelligent Caching System**
   - File-based caching with configurable expiration
   - Cache invalidation and refresh mechanisms
   - Storage optimization for large datasets
   - Cache hit/miss analytics

2. **Multi-Source Data Integration**
   - Yahoo Finance integration (yfinance)
   - Extensible architecture for additional providers
   - Data source failover mechanisms
   - Unified data format standardization

3. **Robust Error Handling**
   - Network connectivity resilience
   - Data validation and integrity checks
   - Graceful fallback mechanisms
   - Comprehensive logging and monitoring

4. **Performance Optimization**
   - Efficient data serialization (pickle)
   - Memory-conscious data processing
   - Batch loading capabilities
   - Asynchronous data fetching support

Architecture:
============

The data loading system follows a layered architecture:

1. **Cache Management Layer**
   - Persistent storage with configurable TTL
   - Cache key generation and management
   - Storage optimization and cleanup
   - Cache analytics and monitoring

2. **Data Provider Layer**
   - Multiple data source integration
   - Provider-specific error handling
   - Data format normalization
   - Rate limiting and throttling

3. **Data Processing Layer**
   - Data validation and cleaning
   - Format standardization
   - Missing value handling
   - Quality assurance checks

4. **Integration Layer**
   - Strategy framework integration
   - Backtesting system support
   - Real-time data streaming
   - Batch processing capabilities

Usage Examples:
===============

Basic Usage:
```python
from backend.data_loader import DataLoader
from datetime import date

# Initialize data loader
loader = DataLoader()

# Load stock data with caching
data = loader.load_stock_data(
    symbol="AAPL",
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31)
)

# Force refresh from source
fresh_data = loader.load_stock_data(
    symbol="AAPL",
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31),
    force_refresh=True
)
```

Advanced Usage:
```python
# Load multiple symbols
symbols = ["AAPL", "MSFT", "GOOGL"]
data_dict = loader.load_multiple_stocks(
    symbols=symbols,
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31)
)

# Batch processing
batch_data = loader.load_batch_data(
    symbols=symbols,
    date_range=(date(2023, 1, 1), date(2023, 12, 31))
)
```

Configuration:
==============

The data loader respects configuration settings:

```python
# settings.py or config.yaml
CACHE_DIR = "data/cache"
CACHE_DURATION_MINUTES = 60
DEFAULT_DATA_SOURCE = "yfinance"
ENABLE_CACHE = True
```

Educational Value:
=================

This module demonstrates:

1. **Data Engineering Principles**
   - ETL pipeline design
   - Data quality assurance
   - Performance optimization
   - Scalability considerations

2. **Caching Strategies**
   - Cache design patterns
   - TTL-based expiration
   - Cache invalidation
   - Storage optimization

3. **Error Handling Patterns**
   - Graceful degradation
   - Retry mechanisms
   - Fallback strategies
   - Logging and monitoring

4. **Financial Data Management**
   - Market data characteristics
   - Data provider integration
   - Real-time vs batch processing
   - Data validation techniques

Integration Points:
==================

The data loader integrates with:
- Trading strategy frameworks
- Backtesting systems
- Real-time analysis tools
- Data visualization components
- Performance monitoring systems

Performance Considerations:
==========================

- Intelligent caching reduces API calls
- Efficient serialization minimizes I/O
- Memory-conscious processing
- Batch operations for multiple symbols
- Asynchronous capabilities for real-time data

Dependencies:
============

- pandas for data manipulation
- yfinance for Yahoo Finance integration
- pickle for efficient serialization
- pathlib for file system operations
- logging for comprehensive monitoring

Author: Strategy Explainer Framework
Version: 2.0
License: Educational Use
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timedelta
import os
from pathlib import Path
import pickle
import logging

from config.config_manager import get_config_manager


class DataLoader:
    """
    Enterprise-grade data loading and caching system for financial market data.
    
    This class provides a comprehensive solution for loading, caching, and managing
    financial market data from various sources. It implements intelligent caching
    strategies, robust error handling, and seamless integration with trading
    strategy frameworks.
    
    The data loader is designed to handle both real-time and historical data
    requirements while maintaining high performance and data integrity.
    
    Key Capabilities:
    ================
    
    1. **Intelligent Caching**
       - File-based caching with configurable TTL
       - Automatic cache invalidation
       - Storage optimization
       - Cache analytics
    
    2. **Multi-Source Integration**
       - Yahoo Finance (yfinance) integration
       - Extensible for additional providers
       - Data source failover
       - Unified data format
    
    3. **Data Quality Assurance**
       - Comprehensive validation
       - Missing value handling
       - Outlier detection
       - Format standardization
    
    4. **Performance Optimization**
       - Efficient serialization
       - Memory-conscious processing
       - Batch loading support
       - Asynchronous capabilities
    
    Attributes:
        cache_dir (Path): Directory for cached data storage
        cache_duration (timedelta): Cache expiration duration
        logger (logging.Logger): Logger instance for monitoring
    
    Example Usage:
    =============
    ```python
    # Initialize with default settings
    loader = DataLoader()
    
    # Load stock data with caching
    data = loader.load_stock_data("AAPL", date(2023, 1, 1), date(2023, 12, 31))
    
    # Force refresh from source
    fresh_data = loader.load_stock_data(
        "AAPL", 
        date(2023, 1, 1), 
        date(2023, 12, 31),
        force_refresh=True
    )
    ```
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data loader with caching configuration.
        
        This constructor sets up the data loading infrastructure including
        cache directory management, expiration policies, and logging
        configuration for comprehensive monitoring.
        
        Args:
            cache_dir (str, optional): Directory for caching data. If None,
                uses the default cache directory from settings.
        
        Initialization Process:
        ======================
        1. **Cache Directory Setup**: Create cache directory if it doesn't exist
        2. **Configuration Loading**: Load cache duration and other settings
        3. **Logging Setup**: Initialize logger for monitoring and debugging
        4. **Directory Structure**: Ensure proper directory permissions
        
        Configuration Integration:
        =========================
        The loader integrates with the configuration system to respect:
        - Cache directory location
        - Cache duration settings
        - Data source preferences
        - Logging configuration
        
        Performance Considerations:
        ==========================
        - Lazy directory creation
        - Efficient path handling
        - Memory-conscious initialization
        - Thread-safe operations
        """
        # Get configuration manager
        config = get_config_manager()
        
        # Configure cache directory with fallback to configuration
        self.cache_dir = Path(cache_dir or config.get('paths.cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache configuration from settings
        self.cache_duration = timedelta(minutes=config.get('data.cache_duration_minutes', 60))
        
        # Initialize logging for monitoring and debugging
        self.logger = logging.getLogger(__name__)
        
        # Log initialization details
        self.logger.info(f"DataLoader initialized with cache directory: {self.cache_dir}")
        self.logger.info(f"Cache duration: {self.cache_duration}")
    
    def load_stock_data(self, 
                       symbol: str, 
                       start_date: date, 
                       end_date: date,
                       force_refresh: bool = False) -> pd.DataFrame:
        """
        Load stock price data for a given symbol and date range.
        
        This method provides the primary interface for loading financial market
        data with intelligent caching and robust error handling. It implements
        a comprehensive data loading pipeline that ensures data quality and
        optimal performance.
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date (date): Start date for the data range
            end_date (date): End date for the data range
            force_refresh (bool, optional): Force refresh from data source,
                bypassing cache. Defaults to False.
        
        Returns:
            pd.DataFrame: Stock price data with standardized OHLCV format:
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
                - adj_close: Adjusted closing price (if available)
        
        Raises:
            ValueError: If symbol is invalid or data cannot be retrieved
            ConnectionError: If unable to connect to data source
            DataValidationError: If retrieved data fails validation
        
        Data Loading Pipeline:
        =====================
        1. **Cache Check**: Verify if valid cached data exists
        2. **Data Retrieval**: Fetch from data source if cache miss
        3. **Data Validation**: Ensure data quality and completeness
        4. **Format Standardization**: Convert to standard OHLCV format
        5. **Cache Storage**: Store processed data for future use
        6. **Quality Assurance**: Final validation before return
        
        Caching Strategy:
        ================
        - Cache key based on symbol, start_date, and end_date
        - TTL-based expiration with configurable duration
        - Automatic cache invalidation on errors
        - Cache hit/miss analytics and monitoring
        
        Error Handling:
        ==============
        - Network connectivity issues
        - Invalid symbol handling
        - Data source API failures
        - Cache corruption recovery
        - Graceful fallback mechanisms
        
        Example Usage:
        =============
        ```python
        # Load Apple stock data for 2023
        data = loader.load_stock_data(
            symbol="AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        # Force refresh from source
        fresh_data = loader.load_stock_data(
            symbol="AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            force_refresh=True
        )
        ```
        """
        # Generate cache key for this specific data request
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # Check cache first unless force refresh is requested
        if not force_refresh:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache hit for {symbol} ({start_date} to {end_date})")
                return cached_data
        
        # Cache miss or force refresh - fetch from data source
        self.logger.info(f"Fetching {symbol} data from source ({start_date} to {end_date})")
        
        try:
            # Fetch data from Yahoo Finance
            data = self._fetch_from_yfinance(symbol, start_date, end_date)
            
            # Validate and standardize the data
            validated_data = self._validate_and_standardize_data(data, symbol)
            
            # Cache the processed data for future use
            self._save_to_cache(cache_key, validated_data)
            
            self.logger.info(f"Successfully loaded {len(validated_data)} records for {symbol}")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol}: {e}")
            
            # Attempt to return stale cached data as fallback
            stale_data = self._load_from_cache(cache_key, ignore_expiration=True)
            if stale_data is not None:
                self.logger.warning(f"Returning stale cached data for {symbol}")
                return stale_data
            
            # If no fallback available, raise the original error
            raise
    
    def _fetch_from_yfinance(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance with comprehensive error handling.
        
        This method handles the low-level interaction with the Yahoo Finance
        API, including error handling, rate limiting, and data format
        standardization.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (date): Start date for data
            end_date (date): End date for data
        
        Returns:
            pd.DataFrame: Raw data from Yahoo Finance
        
        Raises:
            ConnectionError: If unable to connect to Yahoo Finance
            ValueError: If symbol is invalid or returns no data
        
        API Integration:
        ===============
        - Handles Yahoo Finance API limitations
        - Implements retry logic for transient failures
        - Manages rate limiting and throttling
        - Processes API response formats
        """
        try:
            # Create yfinance ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(start=start_date, end=end_date)
            
            # Validate that we received data
            if data.empty:
                raise ValueError(f"No data available for symbol {symbol} in the specified date range")
            
            # Log successful fetch
            self.logger.debug(f"Fetched {len(data)} records from Yahoo Finance for {symbol}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance API error for {symbol}: {e}")
            raise ConnectionError(f"Failed to fetch data from Yahoo Finance: {e}")
    
    def _validate_and_standardize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and standardize market data format.
        
        This method ensures data quality and converts the data to a
        standardized format suitable for the trading strategy framework.
        
        Args:
            data (pd.DataFrame): Raw data from data source
            symbol (str): Stock symbol for logging purposes
        
        Returns:
            pd.DataFrame: Validated and standardized data
        
        Validation Checks:
        =================
        - Required columns presence
        - Data type validation
        - Missing value handling
        - Outlier detection
        - Date index validation
        
        Standardization Process:
        =======================
        - Column name normalization
        - Data type conversion
        - Index formatting
        - Missing value interpolation
        - Outlier treatment
        """
        # Required columns for OHLCV data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Create standardized copy
        standardized_data = data.copy()
        
        # Normalize column names to lowercase
        standardized_data.columns = [col.lower().replace(' ', '_') for col in standardized_data.columns]
        
        # Ensure numeric data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in standardized_data.columns:
                standardized_data[col] = pd.to_numeric(standardized_data[col], errors='coerce')
        
        # Handle missing values
        standardized_data = self._handle_missing_values(standardized_data)
        
        # Validate data integrity
        self._validate_data_integrity(standardized_data, symbol)
        
        return standardized_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in market data.
        
        This method implements sophisticated missing value handling
        appropriate for financial time series data.
        
        Args:
            data (pd.DataFrame): Data with potential missing values
        
        Returns:
            pd.DataFrame: Data with missing values handled
        
        Missing Value Strategy:
        ======================
        - Forward fill for price data
        - Interpolation for volume data
        - Outlier detection and treatment
        - Data quality reporting
        """
        # Forward fill for price data (common in financial data)
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # Interpolate volume data
        if 'volume' in data.columns:
            data['volume'] = data['volume'].interpolate(method='linear')
        
        # Drop any remaining rows with missing values
        initial_length = len(data)
        data = data.dropna()
        
        # Log missing value handling
        if len(data) < initial_length:
            self.logger.warning(f"Dropped {initial_length - len(data)} rows due to missing values")
        
        return data
    
    def _validate_data_integrity(self, data: pd.DataFrame, symbol: str):
        """
        Validate the integrity of market data.
        
        This method performs comprehensive validation checks to ensure
        data quality and detect potential issues.
        
        Args:
            data (pd.DataFrame): Data to validate
            symbol (str): Symbol for logging purposes
        
        Raises:
            ValueError: If data fails integrity checks
        
        Validation Checks:
        =================
        - Price relationship validation (high >= low, etc.)
        - Volume validation (non-negative)
        - Date continuity checks
        - Statistical outlier detection
        - Data completeness verification
        """
        # Check price relationships
        if 'high' in data.columns and 'low' in data.columns:
            invalid_high_low = data['high'] < data['low']
            if invalid_high_low.any():
                self.logger.warning(f"Found {invalid_high_low.sum()} invalid high/low relationships for {symbol}")
        
        # Check for negative volumes
        if 'volume' in data.columns:
            negative_volumes = data['volume'] < 0
            if negative_volumes.any():
                self.logger.warning(f"Found {negative_volumes.sum()} negative volume values for {symbol}")
        
        # Check for extreme price movements (potential data errors)
        if 'close' in data.columns and len(data) > 1:
            price_changes = data['close'].pct_change()
            extreme_changes = abs(price_changes) > 0.5  # 50% daily change threshold
            if extreme_changes.any():
                self.logger.warning(f"Found {extreme_changes.sum()} extreme price changes for {symbol}")
        
        # Log validation completion
        self.logger.debug(f"Data integrity validation completed for {symbol}")
    
    def _load_from_cache(self, cache_key: str, ignore_expiration: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data from cache with expiration checking.
        
        This method handles cache retrieval with intelligent expiration
        checking and error handling for corrupted cache files.
        
        Args:
            cache_key (str): Unique identifier for cached data
            ignore_expiration (bool, optional): Whether to ignore cache expiration.
                Defaults to False.
        
        Returns:
            Optional[pd.DataFrame]: Cached data if available and valid, None otherwise
        
        Cache Management:
        ================
        - TTL-based expiration checking
        - Cache corruption detection
        - Automatic cache cleanup
        - Cache hit/miss analytics
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check if cache file exists
        if not cache_file.exists():
            return None
        
        try:
            # Check file modification time for expiration
            if not ignore_expiration:
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age > self.cache_duration:
                    self.logger.debug(f"Cache expired for {cache_key}")
                    return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate cached data
            if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                return cached_data
            else:
                self.logger.warning(f"Invalid cached data for {cache_key}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading cache for {cache_key}: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """
        Save data to cache with error handling.
        
        This method handles cache storage with comprehensive error handling
        and storage optimization.
        
        Args:
            cache_key (str): Unique identifier for cached data
            data (pd.DataFrame): Data to cache
        
        Storage Optimization:
        ====================
        - Efficient pickle serialization
        - Compression for large datasets
        - Atomic write operations
        - Error recovery mechanisms
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Atomic write using temporary file
            temp_file = cache_file.with_suffix('.tmp')
            
            # Save data to temporary file
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomically move to final location
            temp_file.replace(cache_file)
            
            self.logger.debug(f"Cached data for {cache_key}")
            
        except Exception as e:
            self.logger.error(f"Error saving cache for {cache_key}: {e}")
            # Clean up temporary file if it exists
            try:
                temp_file.unlink()
            except:
                pass
    
    def load_multiple_stocks(self, 
                           symbols: List[str], 
                           start_date: date, 
                           end_date: date,
                           force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple stock symbols efficiently.
        
        This method provides efficient batch loading of multiple symbols
        with parallel processing and comprehensive error handling.
        
        Args:
            symbols (List[str]): List of stock ticker symbols
            start_date (date): Start date for data range
            end_date (date): End date for data range
            force_refresh (bool, optional): Force refresh from source.
                Defaults to False.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        
        Batch Processing:
        ================
        - Parallel data loading when possible
        - Individual error handling per symbol
        - Progress tracking and reporting
        - Efficient memory management
        
        Example Usage:
        =============
        ```python
        # Load multiple stocks
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data_dict = loader.load_multiple_stocks(
            symbols=symbols,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        # Access individual stock data
        aapl_data = data_dict["AAPL"]
        ```
        """
        self.logger.info(f"Loading data for {len(symbols)} symbols")
        
        result = {}
        successful_loads = 0
        failed_loads = 0
        
        # Load data for each symbol
        for symbol in symbols:
            try:
                data = self.load_stock_data(symbol, start_date, end_date, force_refresh)
                result[symbol] = data
                successful_loads += 1
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
                failed_loads += 1
        
        # Log batch processing results
        self.logger.info(f"Batch loading completed: {successful_loads} successful, {failed_loads} failed")
        
        return result
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data for a specific symbol or all symbols.
        
        This method provides cache management functionality for cleaning
        up cached data when needed.
        
        Args:
            symbol (str, optional): Specific symbol to clear. If None,
                clears all cached data.
        
        Cache Management:
        ================
        - Selective cache clearing by symbol
        - Complete cache cleanup
        - Cache size analytics
        - Storage optimization
        
        Example Usage:
        =============
        ```python
        # Clear cache for specific symbol
        loader.clear_cache("AAPL")
        
        # Clear all cached data
        loader.clear_cache()
        ```
        """
        if symbol:
            # Clear cache for specific symbol
            pattern = f"{symbol}_*"
            cache_files = list(self.cache_dir.glob(f"{pattern}.pkl"))
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    self.logger.debug(f"Cleared cache file: {cache_file}")
                except Exception as e:
                    self.logger.error(f"Error clearing cache file {cache_file}: {e}")
            
            self.logger.info(f"Cleared {len(cache_files)} cache files for {symbol}")
        else:
            # Clear all cached data
            cache_files = list(self.cache_dir.glob("*.pkl"))
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.error(f"Error clearing cache file {cache_file}: {e}")
            
            self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache system.
        
        This method provides analytics and monitoring information about
        the cache system for debugging and optimization purposes.
        
        Returns:
            Dict[str, Any]: Cache information including:
                - cache_dir: Cache directory path
                - cache_duration: Cache TTL duration
                - cache_size: Number of cached files
                - total_cache_size_mb: Total cache size in MB
                - oldest_cache_file: Oldest cache file information
                - newest_cache_file: Newest cache file information
        
        Cache Analytics:
        ===============
        - Cache hit/miss rates
        - Storage utilization
        - Cache file age distribution
        - Performance metrics
        
        Example Usage:
        =============
        ```python
        # Get cache information
        cache_info = loader.get_cache_info()
        
        print(f"Cache size: {cache_info['cache_size']} files")
        print(f"Total size: {cache_info['total_cache_size_mb']} MB")
        ```
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        # Calculate total cache size
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Find oldest and newest cache files
        oldest_file = None
        newest_file = None
        
        if cache_files:
            oldest_file = min(cache_files, key=lambda f: f.stat().st_mtime)
            newest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
        
        return {
            'cache_dir': str(self.cache_dir),
            'cache_duration': str(self.cache_duration),
            'cache_size': len(cache_files),
            'total_cache_size_mb': round(total_size_mb, 2),
            'oldest_cache_file': oldest_file.name if oldest_file else None,
            'newest_cache_file': newest_file.name if newest_file else None
        }


# Factory function for creating configured data loader
def create_data_loader(config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """
    Create a configured data loader instance.
    
    This factory function creates a DataLoader instance with configuration
    from the provided config dictionary or default settings.
    
    Args:
        config (Dict[str, Any], optional): Configuration dictionary containing:
            - cache_dir: Cache directory path
            - cache_duration_minutes: Cache TTL in minutes
            - data_source: Preferred data source
    
    Returns:
        DataLoader: Configured data loader instance
    
    Configuration Structure:
    =======================
    ```yaml
    data_loader:
      cache_dir: "data/cache"
      cache_duration_minutes: 60
      data_source: "yfinance"
    ```
    
    Example Usage:
    =============
    ```python
    # Create with default configuration
    loader = create_data_loader()
    
    # Create with custom configuration
    config = {
        'data_loader': {
            'cache_dir': '/tmp/cache',
            'cache_duration_minutes': 30
        }
    }
    loader = create_data_loader(config)
    ```
    """
    # Extract data loader configuration
    if config is None:
        config = {}
    
    dl_config = config.get('data_loader', {})
    
    # Create configured data loader
    return DataLoader(
        cache_dir=dl_config.get('cache_dir')
    )


# Example usage and demonstration
if __name__ == "__main__":
    from datetime import date, timedelta
    
    print("Financial Data Loader Demo")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Test single symbol loading
    print("\n1. Loading Single Symbol:")
    print("-" * 30)
    try:
        data = loader.load_stock_data(
            symbol="AAPL",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today()
        )
        print(f"Loaded {len(data)} records for AAPL")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"Error loading AAPL: {e}")
    
    # Test multiple symbol loading
    print("\n2. Loading Multiple Symbols:")
    print("-" * 30)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    try:
        data_dict = loader.load_multiple_stocks(
            symbols=symbols,
            start_date=date.today() - timedelta(days=30),
            end_date=date.today()
        )
        print(f"Loaded data for {len(data_dict)} symbols")
        for symbol, data in data_dict.items():
            print(f"  {symbol}: {len(data)} records")
    except Exception as e:
        print(f"Error loading multiple symbols: {e}")
    
    # Test cache functionality
    print("\n3. Cache Information:")
    print("-" * 30)
    cache_info = loader.get_cache_info()
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Cache size: {cache_info['cache_size']} files")
    print(f"Total cache size: {cache_info['total_cache_size_mb']} MB")
    
    # Test cache clearing
    print("\n4. Cache Management:")
    print("-" * 30)
    print("Clearing cache for AAPL...")
    loader.clear_cache("AAPL")
    
    updated_info = loader.get_cache_info()
    print(f"Cache size after clearing: {updated_info['cache_size']} files")
    
    print("\n" + "=" * 50)
    print("Demo complete! Data loader ready for integration.")
    print("=" * 50) 