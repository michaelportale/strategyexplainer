"""Data loading and management utilities."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timedelta
import os
from pathlib import Path
import pickle
import logging

from config.settings import settings


class DataLoader:
    """Data loading and caching utility for market data."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching data. Defaults to settings cache directory.
        """
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(minutes=settings.CACHE_DURATION_MINUTES)
        self.logger = logging.getLogger(__name__)
    
    def load_stock_data(self, 
                       symbol: str, 
                       start_date: date, 
                       end_date: date,
                       force_refresh: bool = False) -> pd.DataFrame:
        """Load stock price data for a given symbol and date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            force_refresh: Force refresh from API ignoring cache
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if not force_refresh:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info(f"Loading {symbol} data from cache")
                return cached_data
        
        try:
            self.logger.info(f"Fetching {symbol} data from Yahoo Finance")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date, 
                end=end_date,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            
            # Cache the data
            self._save_to_cache(cache_key, data)
            
            self.logger.info(f"Successfully loaded {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def load_multiple_symbols(self, 
                            symbols: List[str],
                            start_date: date,
                            end_date: date) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            data = self.load_stock_data(symbol, start_date, end_date)
            if not data.empty:
                data_dict[symbol] = data
        
        return data_dict
    
    def get_benchmark_data(self, 
                          benchmark: str = "SPY",
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> pd.DataFrame:
        """Load benchmark data (default SPY).
        
        Args:
            benchmark: Benchmark symbol (default SPY)
            start_date: Start date (defaults to 1 year ago)
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with benchmark data
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()
        
        return self.load_stock_data(benchmark, start_date, end_date)
    
    def get_market_data(self, 
                       symbol: str,
                       period: str = "1y",
                       interval: str = "1d") -> pd.DataFrame:
        """Get market data using yfinance period notation.
        
        Args:
            symbol: Stock ticker symbol
            period: Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if data.empty:
                return pd.DataFrame()
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            company_info = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'description': info.get('longBusinessSummary', ''),
            }
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"Error loading company info for {symbol}: {str(e)}")
            return {'name': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and has data.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get recent data
            data = ticker.history(period="5d")
            return not data.empty
        except:
            return False
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh.
        
        Args:
            cache_key: Cache key for the data
            
        Returns:
            Cached DataFrame or None if not available/stale
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is still fresh
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > self.cache_duration:
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache.
        
        Args:
            cache_key: Cache key for the data
            data: DataFrame to cache
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60
        } 