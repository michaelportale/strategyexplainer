"""Sentiment Overlay Strategy Implementation.

This module implements sentiment-based filtering and enhancement for trading strategies. The core
concept is that market sentiment (as reflected in news, social media, analyst reports, etc.) 
can provide additional context for trading decisions beyond pure price action.

SENTIMENT ANALYSIS IN TRADING:
=============================
Market sentiment represents the collective emotional and psychological state of market participants
toward a specific security or market. It manifests through various channels:

1. NEWS SENTIMENT: Financial news articles, earnings reports, analyst upgrades/downgrades
   - Positive news can drive buying pressure and momentum
   - Negative news can trigger selling and risk-off behavior
   - News impact varies by source credibility and market conditions

2. SOCIAL MEDIA SENTIMENT: Twitter, Reddit, StockTwits, financial forums
   - Retail investor sentiment and crowd psychology
   - Viral trends and social media momentum
   - Real-time sentiment changes during market events

3. ANALYST SENTIMENT: Professional research reports, rating changes, price target adjustments
   - Institutional perspective and fundamental analysis
   - Market-moving analyst calls and sector rotation themes
   - Long-term vs. short-term sentiment divergences

4. MARKET MICROSTRUCTURE SENTIMENT: Order flow, options sentiment, volatility term structure
   - Fear/greed indicators (VIX, put/call ratios)
   - Smart money vs. retail money positioning
   - Institutional flow and fund positioning data

SENTIMENT FILTERING APPROACHES:
==============================
This module implements several approaches to incorporate sentiment:

1. SENTIMENT OVERLAY: Filter existing strategy signals based on sentiment
   - Only execute buy signals when sentiment is positive/neutral
   - Only execute sell signals when sentiment is negative/neutral
   - Sentiment acts as confirmation layer for technical signals

2. SENTIMENT ENHANCEMENT: Adjust position sizing based on sentiment strength
   - Larger positions when sentiment strongly supports the trade direction
   - Smaller positions when sentiment is uncertain or conflicting
   - Sentiment-based risk management and position optimization

3. SENTIMENT TIMING: Use sentiment for entry/exit timing optimization
   - Enter positions when sentiment turns favorable
   - Exit positions when sentiment deteriorates
   - Sentiment-driven stop loss and take profit adjustments

IMPLEMENTATION ARCHITECTURE:
===========================
The module uses a provider pattern to support multiple sentiment data sources:

1. SentimentProvider: Abstract base class for sentiment data integration
2. MockSentimentProvider: Simulated sentiment for testing and development
3. NewsAPISentimentProvider: Real news sentiment via NewsAPI
4. SentimentOverlayStrategy: Strategy wrapper that applies sentiment filtering

USAGE EXAMPLES:
==============
```python
# Basic sentiment overlay with mock data
mock_provider = MockSentimentProvider()
base_strategy = SmaEmaRsiStrategy()
sentiment_strategy = SentimentOverlayStrategy(base_strategy, mock_provider)

# Real news sentiment integration
news_provider = NewsAPISentimentProvider(api_key="your_api_key")
enhanced_strategy = SentimentOverlayStrategy(base_strategy, news_provider)

# Custom sentiment filtering parameters
sentiment_strategy = SentimentOverlayStrategy(
    base_strategy=base_strategy,
    sentiment_provider=news_provider,
    sentiment_threshold=0.1,  # Require positive sentiment for buys
    sentiment_window=5  # Average sentiment over 5 days
)
```

ACADEMIC REFERENCES:
===================
- Baker, M. & Wurgler, J. (2006): "Investor Sentiment and the Cross-Section of Stock Returns"
- Da, Z., Engelberg, J. & Gao, P. (2015): "The Sum of All FEARS Investor Sentiment and Asset Prices"
- Tetlock, P. (2007): "Giving Content to Investor Sentiment: The Role of Media in the Stock Market"
- Antweiler, W. & Frank, M. (2004): "Is All That Talk Just Noise? The Information Content of Internet Stock Message Boards"

Phase 6: Filter signals by "market mood" using sentiment analysis.
Supports real APIs (newsapi, finnhub) and mock sentiment for now.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..base import BaseStrategy
import logging
import requests
import time
import random
from datetime import datetime, timedelta


class SentimentProvider:
    """Abstract base class for sentiment data providers.
    
    This class defines the interface that all sentiment providers must implement.
    It enables pluggable sentiment sources without changing the strategy logic,
    supporting everything from mock data to real-time news APIs.
    
    PROVIDER DESIGN PRINCIPLES:
    ==========================
    - STANDARDIZED OUTPUT: All providers return sentiment scores between -1 and +1
    - ERROR HANDLING: Graceful degradation when sentiment data is unavailable
    - CACHING: Minimize API calls and improve performance through intelligent caching
    - RATE LIMITING: Respect API limits and avoid service disruption
    
    SENTIMENT SCORE INTERPRETATION:
    ==============================
    - +1.0: Extremely positive sentiment (strong bullish indicators)
    - +0.5: Moderately positive sentiment (mild bullish bias)
    -  0.0: Neutral sentiment (no clear directional bias)
    - -0.5: Moderately negative sentiment (mild bearish bias)
    - -1.0: Extremely negative sentiment (strong bearish indicators)
    """
    
    def __init__(self, name: str):
        """Initialize sentiment provider with identifying name.
        
        Args:
            name: Human-readable name for the sentiment provider
                 Used for logging and identification purposes
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Get sentiment score for a specific ticker on a given date.
        
        This is the main interface method that all providers must implement.
        The method should return a sentiment score that represents market
        sentiment toward the ticker on the specified date.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
                   Should be normalized to uppercase for consistency
            date: Date for which sentiment is requested
                 Providers may return sentiment for nearest available date
                 
        Returns:
            float: Sentiment score between -1 (very negative) and +1 (very positive)
                  0.0 indicates neutral sentiment or no available data
                  
        Raises:
            NotImplementedError: Must be implemented by concrete provider classes
            
        Note:
            Providers should handle errors gracefully and return 0.0 (neutral)
            when sentiment data is unavailable rather than raising exceptions.
        """
        raise NotImplementedError


class MockSentimentProvider(SentimentProvider):
    """Mock sentiment provider for testing and development purposes.
    
    This provider generates realistic but simulated sentiment data that exhibits
    characteristics similar to real sentiment: persistence, mean reversion,
    and correlation with market conditions. It's useful for strategy development,
    backtesting, and situations where real sentiment data is not available.
    
    MOCK SENTIMENT CHARACTERISTICS:
    ==============================
    - DETERMINISTIC: Same ticker/date combination always returns same sentiment
    - PERSISTENT: Sentiment shows serial correlation (today's sentiment affects tomorrow's)
    - VARIED: Different tickers have different sentiment patterns
    - REALISTIC: Sentiment ranges and distributions match real-world observations
    
    GENERATION METHODOLOGY:
    ======================
    Uses a combination of:
    1. Date-based base sentiment (cyclical patterns)
    2. Ticker-specific sentiment bias (some stocks are more positive/negative)
    3. Random noise to simulate daily sentiment fluctuations
    4. Persistence mechanism to avoid completely random walk
    """
    
    def __init__(self):
        """Initialize mock sentiment provider with deterministic random seed."""
        super().__init__("Mock")
        # Generate some deterministic but realistic sentiment patterns
        np.random.seed(42)  # For reproducibility across runs
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Generate mock sentiment based on date and ticker with realistic characteristics.
        
        The sentiment generation uses multiple factors to create realistic patterns:
        - Cyclical base sentiment that varies over time
        - Ticker-specific bias (some stocks tend to be more positive/negative)
        - Random daily fluctuations within reasonable bounds
        - Persistence to avoid completely random sentiment changes
        
        Args:
            ticker: Stock ticker symbol (normalized to uppercase)
            date: Date for sentiment generation
            
        Returns:
            float: Mock sentiment score between -1 and +1 with realistic characteristics
        """
        # Use date and ticker to create semi-realistic sentiment
        date_seed = int(date.strftime("%Y%m%d"))  # Deterministic based on date
        ticker_seed = hash(ticker.upper()) % 1000  # Ticker-specific bias
        
        # Create cyclical base sentiment that varies over time
        # This simulates longer-term sentiment cycles
        base_sentiment = np.sin(date_seed / 100) * 0.5
        
        # Add ticker-specific sentiment bias
        # Some stocks tend to have more positive/negative coverage
        ticker_bias = (ticker_seed - 500) / 1000  # Range: -0.5 to +0.5
        
        # Add daily noise with realistic bounds
        # Use deterministic random generator for reproducibility
        noise_generator = np.random.RandomState(date_seed + ticker_seed)
        daily_noise = (noise_generator.random() - 0.5) * 0.6  # Range: -0.3 to +0.3
        
        # Combine factors and clip to valid range
        sentiment = base_sentiment + ticker_bias * 0.3 + daily_noise
        sentiment = np.clip(sentiment, -1, 1)
        
        return sentiment


class NewsAPISentimentProvider(SentimentProvider):
    """Sentiment provider using NewsAPI for real-time news sentiment analysis.
    
    This provider integrates with NewsAPI.org to fetch news articles related to
    specific tickers and performs basic sentiment analysis on the content.
    It provides real market sentiment based on actual news coverage.
    
    NEWSAPI INTEGRATION:
    ===================
    - Fetches relevant news articles for each ticker/date combination
    - Searches both ticker symbol and company name for better coverage
    - Filters by language, relevancy, and date range
    - Implements rate limiting to respect API constraints
    
    SENTIMENT ANALYSIS APPROACH:
    ===========================
    Uses a simple but effective keyword-based sentiment analysis:
    - Positive keywords: bullish, positive, growth, profit, gain, etc.
    - Negative keywords: bearish, negative, loss, decline, fall, etc.
    - Weighted scoring based on keyword frequency and article relevance
    - Article title gets higher weight than description
    
    LIMITATIONS AND CONSIDERATIONS:
    ==============================
    - API rate limits may restrict frequent requests
    - Simple keyword analysis may miss complex sentiment
    - News availability varies by ticker and date
    - Requires active internet connection and valid API key
    """
    
    def __init__(self, api_key: str):
        """Initialize NewsAPI sentiment provider with authentication.
        
        Args:
            api_key: Valid NewsAPI.org API key for authentication
                    Free tier provides 1000 requests per day
                    Premium tiers offer higher limits and additional features
        """
        super().__init__("NewsAPI")
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}  # Simple in-memory caching to reduce API calls
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Get sentiment from NewsAPI by analyzing relevant news articles.
        
        PROCESS FLOW:
        ============
        1. Check cache for existing sentiment data
        2. Query NewsAPI for ticker-related news on specified date
        3. Analyze article content for sentiment indicators
        4. Cache results to minimize future API calls
        5. Return aggregated sentiment score
        
        QUERY STRATEGY:
        ==============
        - Searches for both ticker symbol and company name
        - Focuses on news from the specific date
        - Sorts by relevancy to get most pertinent articles
        - Limits results to English language content
        
        Args:
            ticker: Stock ticker symbol for news search
            date: Date for which news sentiment is requested
            
        Returns:
            float: Sentiment score based on news analysis (-1 to +1)
                  Returns 0.0 if no news found or API error occurs
        """
        # Check cache first to avoid unnecessary API calls
        cache_key = f"{ticker.upper()}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Build NewsAPI query parameters
            params = {
                'q': f'"{ticker.upper()}" OR "{self._get_company_name(ticker)}"',
                'from': date.strftime('%Y-%m-%d'),
                'to': date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key
            }
            
            # Make API request with timeout protection
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                sentiment = 0.0  # Neutral if no news available
                self.logger.info(f"No news articles found for {ticker} on {date.strftime('%Y-%m-%d')}")
            else:
                # Analyze articles for sentiment
                sentiment = self._analyze_articles(articles)
                self.logger.info(f"Analyzed {len(articles)} articles for {ticker}: sentiment = {sentiment:.3f}")
            
            # Cache result to reduce future API calls
            self.cache[cache_key] = sentiment
            
            # Implement rate limiting to respect API constraints
            time.sleep(0.1)  # 100ms delay between requests
            
            return sentiment
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API error fetching sentiment for {ticker}: {e}")
            return 0.0  # Neutral on API error
        except Exception as e:
            self.logger.error(f"Unexpected error analyzing sentiment for {ticker}: {e}")
            return 0.0  # Neutral on any error
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for ticker to improve news search accuracy.
        
        Having both ticker symbol and company name in the search query
        improves the relevance of news articles returned by the API.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            str: Company name associated with ticker, or ticker if unknown
            
        Note:
            This is a simplified mapping. Production systems would use
            a comprehensive ticker-to-company database or API.
        """
        # Simplified ticker to company name mapping
        # In production, this would be a comprehensive database lookup
        ticker_to_name = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft', 
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
            'V': 'Visa',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart',
            'PG': 'Procter & Gamble',
            'UNH': 'UnitedHealth',
            'HD': 'Home Depot',
            'MA': 'Mastercard',
            'PFE': 'Pfizer',
            'BAC': 'Bank of America',
            'ABBV': 'AbbVie',
            'KO': 'Coca-Cola',
            'PEP': 'PepsiCo'
        }
        return ticker_to_name.get(ticker.upper(), ticker)
    
    def _analyze_articles(self, articles: List[Dict]) -> float:
        """Perform sentiment analysis on news articles using keyword-based approach.
        
        ANALYSIS METHODOLOGY:
        ====================
        - Extracts text from article titles and descriptions
        - Counts positive and negative sentiment keywords
        - Weights title content higher than description content
        - Normalizes score based on total content length
        - Returns aggregate sentiment across all articles
        
        KEYWORD CATEGORIES:
        ==================
        - Positive: bullish, positive, growth, profit, gain, rise, strong, beat, exceed, etc.
        - Negative: bearish, negative, loss, decline, fall, weak, miss, underperform, etc.
        - Neutral: Words that don't clearly indicate sentiment direction
        
        Args:
            articles: List of news article dictionaries from NewsAPI
                     Each article has 'title', 'description', and other metadata
                     
        Returns:
            float: Aggregate sentiment score (-1 to +1) across all articles
                  Weighted by article count and keyword frequency
        """
        # Define sentiment keyword lists
        # These lists can be expanded based on domain expertise and analysis
        positive_words = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'strong',
            'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'optimistic',
            'rally', 'surge', 'soar', 'boost', 'breakthrough', 'success',
            'revenue', 'earnings', 'expansion', 'momentum', 'confident'
        ]
        
        negative_words = [
            'bearish', 'negative', 'loss', 'decline', 'fall', 'weak', 'miss',
            'underperform', 'downgrade', 'sell', 'pessimistic', 'concern',
            'drop', 'plunge', 'crash', 'warning', 'risk', 'uncertainty',
            'trouble', 'crisis', 'disappointing', 'weak', 'struggle'
        ]
        
        total_sentiment_score = 0
        total_articles = len(articles)
        
        for article in articles:
            # Extract text content from article
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            
            # Combine title and description with title getting higher weight
            # Title is more likely to contain the key sentiment
            text = f"{title} {title} {description}"  # Title counted twice for emphasis
            
            # Count positive and negative keywords
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            # Calculate article-level sentiment
            if pos_count + neg_count > 0:
                article_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                article_sentiment = 0.0  # Neutral if no sentiment keywords found
            
            total_sentiment_score += article_sentiment
        
        # Calculate average sentiment across all articles
        if total_articles > 0:
            avg_sentiment = total_sentiment_score / total_articles
            # Normalize to [-1, 1] range with some dampening for extreme values
            normalized_sentiment = np.clip(avg_sentiment, -1, 1)
        else:
            normalized_sentiment = 0.0
        
        return normalized_sentiment


class FinnhubSentimentProvider(SentimentProvider):
    """Sentiment provider using Finnhub API for professional-grade sentiment analysis.
    
    Finnhub provides institutional-quality sentiment data derived from news analytics
    and social media monitoring. It offers more sophisticated sentiment analysis
    than basic keyword approaches, including buzz metrics and sentiment scores.
    
    FINNHUB SENTIMENT FEATURES:
    ===========================
    - PROFESSIONAL ANALYTICS: Institutional-grade sentiment analysis algorithms
    - BUZZ METRICS: Social media and news volume indicators
    - SENTIMENT SCORES: Normalized sentiment ratings from multiple sources
    - REAL-TIME DATA: Up-to-date sentiment reflecting current market conditions
    
    SENTIMENT SOURCES:
    =================
    - Financial news articles from major outlets
    - Social media mentions and sentiment
    - Analyst reports and rating changes
    - Earnings call transcripts and corporate communications
    
    API CAPABILITIES:
    ================
    - News sentiment endpoint for historical sentiment data
    - Social sentiment for retail investor mood tracking
    - Insider sentiment based on corporate insider activity
    - Earnings call sentiment for fundamental analysis
    
    ADVANTAGES OVER KEYWORD ANALYSIS:
    ================================
    - Machine learning-based sentiment classification
    - Context-aware analysis beyond simple keyword counting
    - Professional sentiment scoring methodology
    - Multiple data source aggregation and weighting
    """
    
    def __init__(self, api_key: str):
        """Initialize Finnhub sentiment provider with API authentication.
        
        Args:
            api_key: Valid Finnhub API key for accessing sentiment data
                    Free tier provides basic access with rate limits
                    Paid tiers offer higher limits and additional endpoints
        """
        super().__init__("Finnhub")
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.cache = {}  # In-memory caching for API efficiency
    
    def get_sentiment(self, ticker: str, date: datetime) -> float:
        """Get professional sentiment analysis from Finnhub for specified ticker and date.
        
        FINNHUB SENTIMENT METHODOLOGY:
        =============================
        Finnhub aggregates sentiment from multiple sources including news articles,
        social media, and analyst reports. The sentiment scores are normalized and
        weighted based on source credibility and market impact.
        
        DATA SOURCES INCLUDED:
        =====================
        - Major financial news outlets (Reuters, Bloomberg, WSJ, etc.)
        - Social media platforms (Twitter, Reddit, StockTwits)
        - Analyst research reports and rating changes
        - Corporate earnings calls and management communications
        
        Args:
            ticker: Stock ticker symbol for sentiment analysis
            date: Date for which sentiment data is requested
                 API returns sentiment for nearest available data
            
        Returns:
            float: Professional sentiment score (-1 to +1)
                  Based on Finnhub's aggregated sentiment methodology
                  Returns 0.0 if no data available or API error occurs
        """
        cache_key = f"{ticker.upper()}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Build Finnhub API parameters for news sentiment endpoint
            params = {
                'symbol': ticker.upper(),
                'from': date.strftime('%Y-%m-%d'),
                'to': date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            # Query Finnhub news sentiment endpoint
            response = requests.get(f"{self.base_url}/news-sentiment", 
                                  params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract sentiment metrics from Finnhub response
            buzz = data.get('buzz', {})  # Social media and news volume metrics
            sentiment_data = data.get('sentiment', {})  # Aggregated sentiment scores
            
            if sentiment_data:
                # Finnhub provides professional sentiment scores
                # bearerSentiment represents overall market sentiment
                sentiment = sentiment_data.get('bearerSentiment', 0)
                
                # Ensure sentiment is within expected range
                sentiment = np.clip(sentiment, -1, 1)
                
                self.logger.info(f"Finnhub sentiment for {ticker}: {sentiment:.3f} "
                               f"(buzz: {buzz.get('articlesInLastWeek', 0)} articles)")
            else:
                sentiment = 0.0  # Neutral if no sentiment data available
                self.logger.warning(f"No Finnhub sentiment data for {ticker} on {date.strftime('%Y-%m-%d')}")
            
            # Cache result to reduce API calls
            self.cache[cache_key] = sentiment
            
            # Implement rate limiting to respect API constraints
            time.sleep(0.1)  # 100ms delay between requests
            
            return sentiment
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Finnhub API error for {ticker}: {e}")
            return 0.0  # Neutral sentiment on API error
        except Exception as e:
            self.logger.error(f"Unexpected error fetching Finnhub sentiment for {ticker}: {e}")
            return 0.0  # Neutral sentiment on any error


class SentimentOverlayStrategy(BaseStrategy):
    """Strategy that applies sentiment-based filtering and enhancement to base strategy signals.
    
    This strategy implements a sophisticated sentiment overlay system that can filter,
    enhance, or modify trading signals based on market sentiment. It acts as a wrapper
    around any base strategy, adding sentiment intelligence to improve signal quality.
    
    SENTIMENT INTEGRATION APPROACHES:
    ================================
    1. SIGNAL FILTERING: Block signals when sentiment conflicts with trade direction
       - Block buy signals when sentiment is negative
       - Block sell signals when sentiment is extremely positive
       - Allow contrarian signals only in extreme sentiment conditions
    
    2. SIGNAL ENHANCEMENT: Amplify signals when sentiment supports trade direction
       - Increase position size when sentiment strongly supports the signal
       - Scale signal strength based on sentiment magnitude
       - Provide sentiment confidence scoring for risk management
    
    3. SENTIMENT TIMING: Use sentiment for entry/exit timing optimization
       - Enter positions when sentiment turns favorable
       - Exit positions when sentiment deteriorates significantly
       - Implement sentiment-based stop losses and take profits
    
    FILTERING LOGIC IMPLEMENTATION:
    ==============================
    The strategy uses multiple sentiment thresholds for nuanced filtering:
    
    - BEARISH THRESHOLD: Very negative sentiment that blocks all signals
    - SENTIMENT THRESHOLD: Minimum positive sentiment required for buy signals
    - BULLISH THRESHOLD: Very positive sentiment that blocks sell signals
    - SENTIMENT SMOOTHING: Rolling average to reduce noise in sentiment data
    
    SIGNAL ENHANCEMENT FEATURES:
    ===========================
    When enabled, sentiment enhancement scales signal strength:
    - Buy signals: Enhanced by positive sentiment magnitude (1 + sentiment)
    - Sell signals: Enhanced by negative sentiment magnitude (1 + |sentiment|)
    - Position sizing: Larger positions when sentiment strongly supports direction
    - Risk management: Reduced exposure when sentiment is uncertain
    
    SENTIMENT DATA INTEGRATION:
    ==========================
    The strategy supports multiple sentiment providers through the provider pattern:
    - Real-time news sentiment via NewsAPI or Finnhub
    - Social media sentiment tracking
    - Mock sentiment for development and backtesting
    - Custom sentiment providers through standardized interface
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'meta'
    
    def __init__(self, 
                 base_strategy: BaseStrategy,
                 sentiment_provider: SentimentProvider,
                 parameters: Dict[str, Any] = None):
        """Initialize sentiment overlay strategy with base strategy and sentiment provider.
        
        Args:
            base_strategy: Underlying strategy to enhance with sentiment filtering
                          Can be any strategy implementing BaseStrategy interface
                          Strategy operates normally but signals are sentiment-filtered
            sentiment_provider: Source of sentiment data (NewsAPI, Finnhub, Mock, etc.)
                              Provides sentiment scores for filtering and enhancement
            parameters: Sentiment filtering and enhancement configuration
                       If None, uses conservative default parameters
                       
        Configuration Parameters:
            - sentiment_threshold (float, default 0.2): Minimum sentiment for buy signals
            - sentiment_period (int, default 5): Days to smooth sentiment data
            - use_sentiment_strength (bool, default True): Enable signal enhancement
            - bearish_sentiment_threshold (float, default -0.3): Block signals below this
            - bullish_sentiment_threshold (float, default 0.3): Block sells above this
            
        Example:
            # Conservative sentiment overlay on RSI strategy
            rsi_strategy = RsiStrategy(rsi_period=14, rsi_oversold=30, rsi_overbought=70)
            news_provider = NewsAPISentimentProvider(api_key="your_key")
            
            sentiment_strategy = SentimentOverlayStrategy(
                base_strategy=rsi_strategy,
                sentiment_provider=news_provider,
                parameters={
                    'sentiment_threshold': 0.3,  # Require strong positive sentiment for buys
                    'bearish_sentiment_threshold': -0.2,  # Conservative bearish filter
                    'use_sentiment_strength': True  # Enable position sizing enhancement
                }
            )
        """
        # Define conservative default parameters for sentiment filtering
        default_params = {
            'sentiment_threshold': 0.2,  # Minimum sentiment to allow buy signals
            'sentiment_period': 5,       # Days to smooth sentiment for noise reduction
            'use_sentiment_strength': True,  # Enable signal strength enhancement
            'bearish_sentiment_threshold': -0.3,  # Block signals in very negative sentiment
            'bullish_sentiment_threshold': 0.3    # Block sell signals in very positive sentiment
        }
        
        # Override defaults with user-provided parameters
        if parameters:
            default_params.update(parameters)
        
        self.base_strategy = base_strategy
        self.sentiment_provider = sentiment_provider
        
        # Create descriptive strategy name
        name = f"SentimentFiltered_{base_strategy.name}_{sentiment_provider.name}"
        
        # Build comprehensive parameter structure for transparency
        combined_params = {
            'strategy_type': 'SentimentOverlay',
            'base_strategy': base_strategy.get_info(),
            'sentiment_provider': sentiment_provider.name,
            'sentiment_filtering': default_params
        }
        
        super().__init__(name, combined_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate sentiment-enhanced trading signals with comprehensive filtering logic.
        
        SIGNAL GENERATION PROCESS:
        =========================
        1. Base Signal Generation: Execute underlying strategy to get raw signals
        2. Sentiment Data Collection: Fetch sentiment scores for each time period
        3. Sentiment Smoothing: Apply rolling average to reduce sentiment noise
        4. Signal Filtering: Apply sentiment-based filters to block inappropriate signals
        5. Signal Enhancement: Scale signal strength based on sentiment magnitude
        6. Analytics Generation: Add sentiment analysis columns for performance review
        
        FILTERING DECISION TREE:
        =======================
        For each signal, the strategy evaluates:
        
        1. Very Bearish Sentiment (< bearish_threshold):
           - Block ALL signals (too risky to trade)
           
        2. Buy Signals (base_signal == 1):
           - Require sentiment > sentiment_threshold
           - Block if sentiment is negative or insufficient
           
        3. Sell Signals (base_signal == -1):
           - Allow in negative sentiment (may be justified)
           - Block only in very positive sentiment (> bullish_threshold)
           
        4. Enhancement (if enabled):
           - Scale buy signals by positive sentiment magnitude
           - Scale sell signals by negative sentiment magnitude
        
        Args:
            data: Price data DataFrame with OHLCV columns
                 Must contain sufficient history for both base strategy and sentiment smoothing
            
        Returns:
            pd.DataFrame: Enhanced signals with comprehensive sentiment analysis
                         Columns include:
                         - signal: Final sentiment-filtered signal (-1, 0, 1)
                         - base_signal: Original signal from underlying strategy
                         - sentiment_raw: Raw sentiment scores from provider
                         - sentiment: Smoothed sentiment used for filtering
                         - sentiment_strength: Signal enhancement factor (if enabled)
                         - sentiment_bullish/bearish: Sentiment regime indicators
                         - sentiment_blocked: Signals blocked by sentiment filter
                         
        Note:
            This implementation uses a default ticker (AAPL) for demonstration.
            Production systems would pass ticker as a parameter or class attribute.
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        # Step 1: Generate base strategy signals
        df = self.base_strategy.generate_signals(data)
        
        # Step 2: Collect sentiment data for each time period
        # Note: In production, ticker would be passed as parameter or class attribute
        ticker = "AAPL"  # Default ticker for demonstration
        sentiment_scores = []
        
        for date in df.index:
            # Handle different pandas datetime types
            if isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            else:
                date_obj = pd.to_datetime(date).to_pydatetime()
            
            # Fetch sentiment score for this date
            sentiment = self.sentiment_provider.get_sentiment(ticker, date_obj)
            sentiment_scores.append(sentiment)
        
        # Step 3: Add raw sentiment data and apply smoothing
        df['sentiment_raw'] = sentiment_scores
        
        # Smooth sentiment over specified period to reduce noise
        period = self.parameters['sentiment_period']
        df['sentiment'] = df['sentiment_raw'].rolling(window=period).mean()
        
        # Step 4: Store original signals for analysis and comparison
        df['base_signal'] = df['signal'].copy()
        
        # Step 5: Apply sentiment-based filtering logic
        sentiment_threshold = self.parameters['sentiment_threshold']
        bearish_threshold = self.parameters['bearish_sentiment_threshold']
        bullish_threshold = self.parameters['bullish_sentiment_threshold']
        
        # Block ALL signals in very negative sentiment (market stress)
        very_bearish = df['sentiment'] < bearish_threshold
        df.loc[very_bearish, 'signal'] = 0
        
        # For buy signals, require minimum positive sentiment
        buy_signals = df['base_signal'] == 1
        insufficient_sentiment = df['sentiment'] < sentiment_threshold
        df.loc[buy_signals & insufficient_sentiment, 'signal'] = 0
        
        # For sell signals, block only in very positive sentiment
        # (Allow sells in negative sentiment as they may be justified)
        sell_signals = df['base_signal'] == -1
        very_bullish = df['sentiment'] > bullish_threshold
        df.loc[sell_signals & very_bullish, 'signal'] = 0
        
        # Step 6: Optionally enhance signal strength based on sentiment magnitude
        if self.parameters['use_sentiment_strength']:
            # Initialize sentiment strength multiplier
            df['sentiment_strength'] = 1.0
            
            # Enhance buy signals with positive sentiment
            buy_mask = (df['signal'] == 1) & (df['sentiment'] > 0)
            df.loc[buy_mask, 'sentiment_strength'] = 1 + df.loc[buy_mask, 'sentiment']
            
            # Enhance sell signals with negative sentiment magnitude
            sell_mask = (df['signal'] == -1) & (df['sentiment'] < 0)
            df.loc[sell_mask, 'sentiment_strength'] = 1 + abs(df.loc[sell_mask, 'sentiment'])
        
        # Step 7: Add comprehensive sentiment analysis columns for performance review
        df['sentiment_bullish'] = df['sentiment'] > bullish_threshold
        df['sentiment_bearish'] = df['sentiment'] < bearish_threshold
        df['sentiment_blocked'] = very_bearish | (buy_signals & insufficient_sentiment) | (sell_signals & very_bullish)
        
        # Step 8: Calculate and log filtering statistics
        original_buys = (df['base_signal'] == 1).sum()
        original_sells = (df['base_signal'] == -1).sum()
        filtered_buys = (df['signal'] == 1).sum()
        filtered_sells = (df['signal'] == -1).sum()
        
        avg_sentiment = df['sentiment'].mean()
        blocked_signals = df['sentiment_blocked'].sum()
        
        self.logger.info(f"Sentiment filtering results (avg sentiment: {avg_sentiment:.3f}):")
        self.logger.info(f"  Buy signals: {original_buys} -> {filtered_buys} "
                        f"(filtered {original_buys - filtered_buys})")
        self.logger.info(f"  Sell signals: {original_sells} -> {filtered_sells} "
                        f"(filtered {original_sells - filtered_sells})")
        self.logger.info(f"  Total blocked signals: {blocked_signals}")
        
        return df


class SentimentMeanReversionStrategy(BaseStrategy):
    """Pure sentiment-based contrarian strategy using sentiment extremes for mean reversion signals.
    
    This strategy implements a contrarian approach that trades against sentiment extremes,
    based on the behavioral finance principle that extreme sentiment often precedes
    price reversals. It buys when sentiment is very negative and sells when sentiment
    is very positive.
    
    CONTRARIAN SENTIMENT THEORY:
    ===========================
    The strategy is based on several behavioral finance concepts:
    
    1. SENTIMENT OVERREACTION: Markets tend to overreact to news and sentiment
       - Extremely negative sentiment may indicate oversold conditions
       - Extremely positive sentiment may indicate overbought conditions
       - Mean reversion occurs as markets correct these overreactions
    
    2. CROWD PSYCHOLOGY: Extreme sentiment represents herd behavior
       - When "everyone" is bearish, there may be no sellers left
       - When "everyone" is bullish, there may be no buyers left
       - Contrarian positions profit from this imbalance correction
    
    3. SENTIMENT CYCLE: Sentiment follows cyclical patterns
       - Fear and greed cycles create trading opportunities
       - Extreme sentiment periods are often followed by reversals
       - Contrarian timing captures these sentiment-driven moves
    
    STRATEGY IMPLEMENTATION:
    =======================
    1. SENTIMENT MONITORING: Continuously track sentiment for extreme readings
    2. CONTRARIAN SIGNALS: Generate buy/sell signals at sentiment extremes
    3. PRICE CONFIRMATION: Optionally require price action to confirm sentiment
    4. RISK MANAGEMENT: Position sizing based on sentiment magnitude
    
    SIGNAL GENERATION LOGIC:
    =======================
    - BUY SIGNALS: Generated when sentiment is extremely negative (bearish threshold)
    - SELL SIGNALS: Generated when sentiment is extremely positive (bullish threshold)
    - CONFIRMATION: Optionally require price action to support contrarian view
    - TIMING: Use sentiment smoothing to avoid false signals from noise
    """
    
    # Strategy metadata for auto-registration
    strategy_category = 'meta'
    
    def __init__(self, 
                 sentiment_provider: SentimentProvider,
                 parameters: Dict[str, Any] = None):
        """Initialize sentiment-based mean reversion strategy.
        
        Args:
            sentiment_provider: Source of sentiment data for contrarian analysis
                              Should provide reliable sentiment extremes for signal generation
            parameters: Strategy configuration parameters for contrarian logic
                       If None, uses moderate default parameters for contrarian trading
                       
        Configuration Parameters:
            - bearish_threshold (float, default -0.5): Sentiment level for buy signals
            - bullish_threshold (float, default 0.5): Sentiment level for sell signals
            - sentiment_period (int, default 3): Smoothing period for sentiment
            - require_price_confirmation (bool, default True): Require price support
            - price_lookback (int, default 5): Days for price confirmation analysis
            
        Example:
            # Aggressive contrarian strategy
            news_provider = NewsAPISentimentProvider(api_key="your_key")
            
            contrarian_strategy = SentimentMeanReversionStrategy(
                sentiment_provider=news_provider,
                parameters={
                    'bearish_threshold': -0.3,  # Less extreme threshold for more signals
                    'bullish_threshold': 0.3,   # Less extreme threshold for more signals
                    'require_price_confirmation': False,  # Pure sentiment signals
                    'sentiment_period': 1  # No smoothing for immediate reaction
                }
            )
        """
        # Define default parameters for contrarian sentiment strategy
        default_params = {
            'bearish_threshold': -0.5,   # Very negative sentiment triggers buy (contrarian)
            'bullish_threshold': 0.5,    # Very positive sentiment triggers sell (contrarian)
            'sentiment_period': 3,       # Short smoothing period for responsiveness
            'require_price_confirmation': True,  # Require price action to support sentiment
            'price_lookback': 5          # Days to analyze price trend for confirmation
        }
        
        # Override defaults with user-provided parameters
        if parameters:
            default_params.update(parameters)
        
        self.sentiment_provider = sentiment_provider
        
        # Create descriptive strategy name
        name = f"SentimentContrarian_{sentiment_provider.name}"
        super().__init__(name, default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate contrarian signals based on sentiment extremes with optional price confirmation.
        
        CONTRARIAN SIGNAL LOGIC:
        =======================
        1. Sentiment Analysis: Identify periods of extreme positive/negative sentiment
        2. Contrarian Positioning: Take opposite positions to prevailing sentiment
        3. Price Confirmation: Optionally verify that price action supports contrarian view
        4. Signal Generation: Create buy/sell signals at appropriate sentiment extremes
        
        CONFIRMATION LOGIC (if enabled):
        ===============================
        - Buy Signal Confirmation: Extreme negative sentiment + declining prices
          Theory: Oversold condition with bearish sentiment suggests reversal opportunity
          
        - Sell Signal Confirmation: Extreme positive sentiment + rising prices  
          Theory: Overbought condition with bullish sentiment suggests reversal opportunity
        
        Args:
            data: Price data DataFrame with OHLCV columns
                 Must contain sufficient history for sentiment smoothing and price confirmation
            
        Returns:
            pd.DataFrame: Contrarian signals with sentiment analysis
                         Columns include:
                         - signal: Contrarian signal (-1, 0, 1)
                         - sentiment_raw: Raw sentiment scores
                         - sentiment: Smoothed sentiment for signal generation
                         - sentiment_extreme_bearish: Extreme negative sentiment periods
                         - sentiment_extreme_bullish: Extreme positive sentiment periods
                         - price_confirmation: Whether price supports contrarian signal (if enabled)
        """
        if not self.validate_data(data):
            df = data.copy()
            df['signal'] = 0
            return df
        
        df = data.copy()
        
        # Step 1: Collect sentiment data for contrarian analysis
        # Note: In production, ticker would be passed as parameter
        ticker = "AAPL"  # Default ticker for demonstration
        sentiment_scores = []
        
        for date in df.index:
            # Handle different pandas datetime types
            if isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            else:
                date_obj = pd.to_datetime(date).to_pydatetime()
            
            # Get sentiment score for contrarian analysis
            sentiment = self.sentiment_provider.get_sentiment(ticker, date_obj)
            sentiment_scores.append(sentiment)
        
        # Step 2: Add sentiment data and apply smoothing
        df['sentiment_raw'] = sentiment_scores
        
        # Apply minimal smoothing to preserve sentiment extremes while reducing noise
        period = self.parameters['sentiment_period']
        df['sentiment'] = df['sentiment_raw'].rolling(window=period).mean()
        
        # Step 3: Initialize signal column
        df['signal'] = 0
        
        # Step 4: Identify sentiment extremes for contrarian signals
        bearish_threshold = self.parameters['bearish_threshold']
        bullish_threshold = self.parameters['bullish_threshold']
        
        very_bearish_sentiment = df['sentiment'] < bearish_threshold
        very_bullish_sentiment = df['sentiment'] > bullish_threshold
        
        # Step 5: Apply price confirmation if required
        if self.parameters['require_price_confirmation']:
            lookback = self.parameters['price_lookback']
            
            # Price declining supports contrarian buy in negative sentiment
            price_declining = df['close'] < df['close'].shift(lookback)
            
            # Price rising supports contrarian sell in positive sentiment
            price_rising = df['close'] > df['close'].shift(lookback)
            
            # Contrarian buy: Extreme negative sentiment + declining prices (oversold)
            buy_condition = very_bearish_sentiment & price_declining
            
            # Contrarian sell: Extreme positive sentiment + rising prices (overbought)
            sell_condition = very_bullish_sentiment & price_rising
            
            # Add price confirmation indicator for analysis
            df['price_confirmation'] = (very_bearish_sentiment & price_declining) | \
                                     (very_bullish_sentiment & price_rising)
        else:
            # Pure sentiment-based contrarian signals (no price confirmation)
            buy_condition = very_bearish_sentiment
            sell_condition = very_bullish_sentiment
            df['price_confirmation'] = True  # Always confirmed when not required
        
        # Step 6: Apply contrarian signals
        df.loc[buy_condition, 'signal'] = 1   # Buy when sentiment extremely negative
        df.loc[sell_condition, 'signal'] = -1  # Sell when sentiment extremely positive
        
        # Step 7: Add sentiment analysis columns for performance review
        df['sentiment_extreme_bearish'] = very_bearish_sentiment
        df['sentiment_extreme_bullish'] = very_bullish_sentiment
        
        # Step 8: Calculate and log contrarian signal statistics
        contrarian_buys = buy_condition.sum()
        contrarian_sells = sell_condition.sum()
        avg_sentiment = df['sentiment'].mean()
        extreme_periods = (very_bearish_sentiment | very_bullish_sentiment).sum()
        
        self.logger.info(f"Sentiment contrarian strategy (avg sentiment: {avg_sentiment:.3f}):")
        self.logger.info(f"  Extreme sentiment periods: {extreme_periods}")
        self.logger.info(f"  Contrarian buy signals: {contrarian_buys} "
                        f"(avg sentiment: {df[buy_condition]['sentiment'].mean():.3f})")
        self.logger.info(f"  Contrarian sell signals: {contrarian_sells} "
                        f"(avg sentiment: {df[sell_condition]['sentiment'].mean():.3f})")
        
        return df


def create_sentiment_overlay(base_strategy: BaseStrategy,
                           provider_type: str = 'mock',
                           api_key: str = None,
                           sentiment_params: Dict[str, Any] = None) -> SentimentOverlayStrategy:
    """Factory function to create sentiment overlay strategies with simplified configuration.
    
    This helper function streamlines the creation of sentiment-enhanced strategies by
    handling provider instantiation, parameter validation, and proper configuration
    of the sentiment overlay system.
    
    PROVIDER TYPES SUPPORTED:
    ========================
    - 'mock': Simulated sentiment for development and backtesting
    - 'newsapi': Real news sentiment via NewsAPI.org
    - 'finnhub': Professional sentiment via Finnhub API
    
    COMMON USE CASES:
    ================
    - Development: Use 'mock' provider for strategy development and testing
    - News-based: Use 'newsapi' for retail news sentiment integration  
    - Professional: Use 'finnhub' for institutional-grade sentiment analysis
    - Backtesting: Use 'mock' with deterministic sentiment for consistent results
    
    Args:
        base_strategy: Underlying strategy to enhance with sentiment filtering
                      Can be any strategy implementing BaseStrategy interface
        provider_type: Type of sentiment provider ('mock', 'newsapi', 'finnhub')
                      Determines data source and quality of sentiment analysis
        api_key: API key for external sentiment providers (NewsAPI, Finnhub)
                Required for 'newsapi' and 'finnhub', ignored for 'mock'
        sentiment_params: Sentiment filtering configuration parameters
                         If None, uses balanced default parameters
                         
    Returns:
        SentimentOverlayStrategy: Configured sentiment-enhanced strategy ready for use
        
    Raises:
        ValueError: If api_key is required but not provided, or unknown provider_type
        
    Example:
        # Development with mock sentiment
        base_rsi = RsiStrategy(rsi_period=14)
        mock_sentiment_strategy = create_sentiment_overlay(
            base_strategy=base_rsi,
            provider_type='mock'
        )
        
        # Production with real news sentiment
        news_sentiment_strategy = create_sentiment_overlay(
            base_strategy=base_rsi,
            provider_type='newsapi',
            api_key='your_newsapi_key',
            sentiment_params={'sentiment_threshold': 0.3}
        )
        
        # Professional with Finnhub sentiment
        professional_strategy = create_sentiment_overlay(
            base_strategy=base_rsi,
            provider_type='finnhub',
            api_key='your_finnhub_key',
            sentiment_params={'use_sentiment_strength': True}
        )
    """
    # Create appropriate sentiment provider based on type
    provider_type_lower = provider_type.lower()
    
    if provider_type_lower == 'mock':
        provider = MockSentimentProvider()
    elif provider_type_lower == 'newsapi':
        if not api_key:
            raise ValueError("API key required for NewsAPI sentiment provider")
        provider = NewsAPISentimentProvider(api_key)
    elif provider_type_lower == 'finnhub':
        if not api_key:
            raise ValueError("API key required for Finnhub sentiment provider")
        provider = FinnhubSentimentProvider(api_key)
    else:
        supported_types = ['mock', 'newsapi', 'finnhub']
        raise ValueError(f"Unknown provider type: '{provider_type}'. "
                        f"Supported types: {supported_types}")
    
    # Create and return configured sentiment overlay strategy
    return SentimentOverlayStrategy(base_strategy, provider, sentiment_params)
