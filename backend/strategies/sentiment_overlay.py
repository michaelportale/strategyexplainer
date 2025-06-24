"""Sentiment Overlay Strategy.

Phase 6: Filter signals by "market mood" using sentiment analysis.
Supports real APIs (newsapi, finnhub) and mock sentiment for now.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base import BaseStrategy
import logging
import requests
import time
import random
from datetime import datetime, timedelta


class SentimentProvider:
    """Base class for sentiment data providers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Get sentiment score for ticker on given date.
        
        Args:
            ticker: Stock ticker symbol
            date: Date for sentiment
            
        Returns:
            Sentiment score between -1 (very negative) and 1 (very positive)
        """
        raise NotImplementedError


class MockSentimentProvider(SentimentProvider):
    """Mock sentiment provider for testing."""
    
    def __init__(self):
        super().__init__("Mock")
        # Generate some deterministic but realistic sentiment patterns
        np.random.seed(42)  # For reproducibility
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Generate mock sentiment based on date and ticker.
        
        Creates realistic sentiment patterns with some persistence.
        """
        # Use date and ticker to create semi-realistic sentiment
        date_seed = int(date.strftime("%Y%m%d"))
        ticker_seed = hash(ticker) % 1000
        
        # Create some persistence in sentiment
        base_sentiment = np.sin(date_seed / 100) * 0.5
        noise = (np.random.RandomState(date_seed + ticker_seed).random() - 0.5) * 0.6
        
        sentiment = np.clip(base_sentiment + noise, -1, 1)
        
        return sentiment


class NewsAPISentimentProvider(SentimentProvider):
    """Sentiment provider using NewsAPI."""
    
    def __init__(self, api_key: str):
        super().__init__("NewsAPI")
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}  # Simple caching
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Get sentiment from NewsAPI."""
        # Check cache first
        cache_key = f"{ticker}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Query NewsAPI for ticker-related news
            params = {
                'q': f'"{ticker}" OR "{self._get_company_name(ticker)}"',
                'from': date.strftime('%Y-%m-%d'),
                'to': date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                sentiment = 0.0  # Neutral if no news
            else:
                # Simple sentiment scoring based on keywords
                sentiment = self._analyze_articles(articles)
            
            # Cache result
            self.cache[cache_key] = sentiment
            
            # Rate limiting
            time.sleep(0.1)
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error fetching sentiment for {ticker}: {e}")
            return 0.0  # Neutral on error
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for ticker (simplified mapping)."""
        # This would normally be a proper ticker->company mapping
        ticker_to_name = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'NVDA': 'NVIDIA'
        }
        return ticker_to_name.get(ticker.upper(), ticker)
    
    def _analyze_articles(self, articles: List[Dict]) -> float:
        """Simple sentiment analysis of articles."""
        positive_words = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'strong',
            'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'optimistic'
        ]
        negative_words = [
            'bearish', 'negative', 'loss', 'decline', 'fall', 'weak', 'miss',
            'underperform', 'downgrade', 'sell', 'pessimistic', 'concern'
        ]
        
        total_score = 0
        total_articles = len(articles)
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            text = f"{title} {description}"
            
            pos_count = sum(word in text for word in positive_words)
            neg_count = sum(word in text for word in negative_words)
            
            if pos_count > neg_count:
                total_score += 1
            elif neg_count > pos_count:
                total_score -= 1
        
        # Normalize to -1 to 1 range
        if total_articles > 0:
            sentiment = total_score / total_articles
        else:
            sentiment = 0.0
            
        return np.clip(sentiment, -1, 1)


class FinnhubSentimentProvider(SentimentProvider):
    """Sentiment provider using Finnhub API."""
    
    def __init__(self, api_key: str):
        super().__init__("Finnhub")
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.cache = {}
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Get sentiment from Finnhub."""
        cache_key = f"{ticker}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Get news sentiment from Finnhub
            params = {
                'symbol': ticker,
                'from': date.strftime('%Y-%m-%d'),
                'to': date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/news-sentiment", 
                                  params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Use buzz and sentiment scores from Finnhub
            buzz = data.get('buzz', {})
            sentiment_data = data.get('sentiment', {})
            
            if sentiment_data:
                # Finnhub provides sentiment scores, use them directly
                sentiment = sentiment_data.get('bearerSentiment', 0)
                sentiment = np.clip(sentiment, -1, 1)
            else:
                sentiment = 0.0
            
            self.cache[cache_key] = sentiment
            time.sleep(0.1)  # Rate limiting
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error fetching Finnhub sentiment for {ticker}: {e}")
            return 0.0


class SentimentOverlayStrategy(BaseStrategy):
    """Strategy that filters base strategy signals using sentiment analysis."""
    
    def __init__(self, 
                 base_strategy: BaseStrategy,
                 sentiment_provider: SentimentProvider,
                 parameters: Dict[str, Any] = None):
        """Initialize sentiment overlay strategy.
        
        Args:
            base_strategy: The underlying strategy to wrap
            sentiment_provider: Sentiment data provider
            parameters: Sentiment filtering parameters
        """
        default_params = {
            'sentiment_threshold': 0.2,  # Minimum sentiment to allow signals
            'sentiment_period': 5,       # Days to smooth sentiment
            'use_sentiment_strength': True,  # Scale signals by sentiment strength
            'bearish_sentiment_threshold': -0.3,  # Block signals below this
            'bullish_sentiment_threshold': 0.3    # Enhance signals above this
        }
        
        if parameters:
            default_params.update(parameters)
        
        self.base_strategy = base_strategy
        self.sentiment_provider = sentiment_provider
        
        name = f"Sentiment-Filtered {base_strategy.name}"
        combined_params = {
            'base_strategy': base_strategy.get_info(),
            'sentiment_provider': sentiment_provider.name,
            'sentiment_params': default_params
        }
        
        super().__init__(name, combined_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment-filtered signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with sentiment-filtered signals
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Get base strategy signals
        df = self.base_strategy.generate_signals(data)
        
        # Get sentiment data for each date
        # Note: In real implementation, ticker would be passed as parameter
        ticker = "AAPL"  # Default ticker for demo
        sentiment_scores = []
        
        for date in df.index:
            if isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            else:
                date_obj = pd.to_datetime(date).to_pydatetime()
            
            sentiment = self.sentiment_provider.get_sentiment(ticker, date_obj)
            sentiment_scores.append(sentiment)
        
        df['sentiment_raw'] = sentiment_scores
        
        # Smooth sentiment over specified period
        period = self.parameters['sentiment_period']
        df['sentiment'] = df['sentiment_raw'].rolling(window=period).mean()
        
        # Store original signals
        df['base_signal'] = df['signal'].copy()
        
        # Apply sentiment filters
        sentiment_threshold = self.parameters['sentiment_threshold']
        bearish_threshold = self.parameters['bearish_sentiment_threshold']
        bullish_threshold = self.parameters['bullish_sentiment_threshold']
        
        # Block signals in very negative sentiment
        very_bearish = df['sentiment'] < bearish_threshold
        df.loc[very_bearish, 'signal'] = 0
        
        # For buy signals, require positive sentiment
        buy_signals = df['base_signal'] == 1
        insufficient_sentiment = df['sentiment'] < sentiment_threshold
        df.loc[buy_signals & insufficient_sentiment, 'signal'] = 0
        
        # For sell signals, allow in negative sentiment (might be justified)
        # But block in very positive sentiment
        sell_signals = df['base_signal'] == -1
        very_bullish = df['sentiment'] > bullish_threshold
        df.loc[sell_signals & very_bullish, 'signal'] = 0
        
        # Optionally scale signal strength by sentiment
        if self.parameters['use_sentiment_strength']:
            # Scale buy signals by positive sentiment
            buy_mask = (df['signal'] == 1) & (df['sentiment'] > 0)
            # Scale sell signals by negative sentiment magnitude
            sell_mask = (df['signal'] == -1) & (df['sentiment'] < 0)
            
            # Add sentiment strength column for analysis
            df['sentiment_strength'] = 1.0
            df.loc[buy_mask, 'sentiment_strength'] = 1 + df.loc[buy_mask, 'sentiment']
            df.loc[sell_mask, 'sentiment_strength'] = 1 + abs(df.loc[sell_mask, 'sentiment'])
        
        # Add sentiment analysis columns
        df['sentiment_bullish'] = df['sentiment'] > bullish_threshold
        df['sentiment_bearish'] = df['sentiment'] < bearish_threshold
        df['sentiment_blocked'] = very_bearish | (buy_signals & insufficient_sentiment)
        
        # Count filtered signals
        original_buys = (df['base_signal'] == 1).sum()
        original_sells = (df['base_signal'] == -1).sum()
        filtered_buys = (df['signal'] == 1).sum()
        filtered_sells = (df['signal'] == -1).sum()
        
        avg_sentiment = df['sentiment'].mean()
        
        self.logger.info(f"Sentiment filter (avg: {avg_sentiment:.2f}): "
                        f"{original_buys} -> {filtered_buys} buys, "
                        f"{original_sells} -> {filtered_sells} sells")
        
        return df


class SentimentMeanReversionStrategy(BaseStrategy):
    """Pure sentiment-based mean reversion strategy.
    
    Buy when sentiment is very negative (contrarian).
    Sell when sentiment is very positive.
    """
    
    def __init__(self, 
                 sentiment_provider: SentimentProvider,
                 parameters: Dict[str, Any] = None):
        """Initialize sentiment mean reversion strategy."""
        default_params = {
            'bearish_threshold': -0.5,   # Buy when sentiment very negative
            'bullish_threshold': 0.5,    # Sell when sentiment very positive
            'sentiment_period': 3,       # Smoothing period
            'require_price_confirmation': True,  # Require price to support signal
            'price_lookback': 5          # Days to check price momentum
        }
        
        if parameters:
            default_params.update(parameters)
        
        self.sentiment_provider = sentiment_provider
        
        name = f"Sentiment Mean Reversion ({sentiment_provider.name})"
        super().__init__(name, default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment-based mean reversion signals."""
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Get sentiment data
        ticker = "AAPL"  # Default for demo
        sentiment_scores = []
        
        for date in df.index:
            if isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            else:
                date_obj = pd.to_datetime(date).to_pydatetime()
            
            sentiment = self.sentiment_provider.get_sentiment(ticker, date_obj)
            sentiment_scores.append(sentiment)
        
        df['sentiment_raw'] = sentiment_scores
        
        # Smooth sentiment
        period = self.parameters['sentiment_period']
        df['sentiment'] = df['sentiment_raw'].rolling(window=period).mean()
        
        # Initialize signals
        df['signal'] = 0
        
        # Mean reversion signals based on sentiment extremes
        bearish_threshold = self.parameters['bearish_threshold']
        bullish_threshold = self.parameters['bullish_threshold']
        
        very_bearish_sentiment = df['sentiment'] < bearish_threshold
        very_bullish_sentiment = df['sentiment'] > bullish_threshold
        
        # Price confirmation if required
        if self.parameters['require_price_confirmation']:
            lookback = self.parameters['price_lookback']
            price_declining = df['close'] < df['close'].shift(lookback)
            price_rising = df['close'] > df['close'].shift(lookback)
            
            # Buy when sentiment very negative AND price has been declining
            buy_condition = very_bearish_sentiment & price_declining
            
            # Sell when sentiment very positive AND price has been rising
            sell_condition = very_bullish_sentiment & price_rising
        else:
            buy_condition = very_bearish_sentiment
            sell_condition = very_bullish_sentiment
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Add analysis columns
        df['sentiment_extreme_bearish'] = very_bearish_sentiment
        df['sentiment_extreme_bullish'] = very_bullish_sentiment
        
        self.logger.info(f"Sentiment mean reversion: {buy_condition.sum()} contrarian buys, "
                        f"{sell_condition.sum()} contrarian sells")
        
        return df


def create_sentiment_overlay(base_strategy: BaseStrategy,
                           provider_type: str = 'mock',
                           api_key: str = None,
                           sentiment_params: Dict[str, Any] = None) -> SentimentOverlayStrategy:
    """Helper function to create sentiment overlay strategy.
    
    Args:
        base_strategy: Strategy to wrap with sentiment
        provider_type: 'mock', 'newsapi', or 'finnhub'
        api_key: API key for real providers
        sentiment_params: Sentiment filtering parameters
        
    Returns:
        Sentiment overlay strategy instance
    """
    # Create sentiment provider
    if provider_type.lower() == 'mock':
        provider = MockSentimentProvider()
    elif provider_type.lower() == 'newsapi':
        if not api_key:
            raise ValueError("API key required for NewsAPI")
        provider = NewsAPISentimentProvider(api_key)
    elif provider_type.lower() == 'finnhub':
        if not api_key:
            raise ValueError("API key required for Finnhub")
        provider = FinnhubSentimentProvider(api_key)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return SentimentOverlayStrategy(base_strategy, provider, sentiment_params)
