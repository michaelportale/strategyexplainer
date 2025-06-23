"""Application settings and configuration management."""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings and constants."""
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
    # Data Configuration
    DATA_SOURCE: str = os.getenv('DATA_SOURCE', 'yfinance')
    DEFAULT_BENCHMARK: str = os.getenv('DEFAULT_BENCHMARK', 'SPY')
    CACHE_DURATION_MINUTES: int = int(os.getenv('CACHE_DURATION_MINUTES', '60'))
    
    # Application Configuration
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Streamlit Configuration
    STREAMLIT_THEME: str = os.getenv('STREAMLIT_THEME', 'light')
    STREAMLIT_PORT: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    
    # File Paths
    DATA_DIR: str = 'data'
    CACHE_DIR: str = 'data/cache'
    OUTPUTS_DIR: str = 'backend/outputs'
    REPORTS_DIR: str = 'backend/outputs/reports'
    CHARTS_DIR: str = 'backend/outputs/charts'
    
    # Default Strategy Parameters
    DEFAULT_LOOKBACK_PERIOD: int = 20
    DEFAULT_THRESHOLD: float = 0.02
    DEFAULT_POSITION_SIZE: float = 0.1
    DEFAULT_STOP_LOSS: float = 0.05
    DEFAULT_TAKE_PROFIT: float = 0.10
    
    # Performance Metrics
    RISK_FREE_RATE: float = 0.02  # 2% annual risk-free rate
    TRADING_DAYS_PER_YEAR: int = 252


# Global settings instance
settings = Settings()


def get_config() -> Dict[str, Any]:
    """Get application configuration as dictionary."""
    return {
        'api_keys': {
            'openai': settings.OPENAI_API_KEY,
            'alpha_vantage': settings.ALPHA_VANTAGE_API_KEY,
        },
        'data': {
            'source': settings.DATA_SOURCE,
            'benchmark': settings.DEFAULT_BENCHMARK,
            'cache_duration': settings.CACHE_DURATION_MINUTES,
        },
        'app': {
            'debug': settings.DEBUG,
            'log_level': settings.LOG_LEVEL,
        },
        'paths': {
            'data_dir': settings.DATA_DIR,
            'cache_dir': settings.CACHE_DIR,
            'outputs_dir': settings.OUTPUTS_DIR,
            'reports_dir': settings.REPORTS_DIR,
            'charts_dir': settings.CHARTS_DIR,
        },
        'defaults': {
            'lookback_period': settings.DEFAULT_LOOKBACK_PERIOD,
            'threshold': settings.DEFAULT_THRESHOLD,
            'position_size': settings.DEFAULT_POSITION_SIZE,
            'stop_loss': settings.DEFAULT_STOP_LOSS,
            'take_profit': settings.DEFAULT_TAKE_PROFIT,
        },
        'metrics': {
            'risk_free_rate': settings.RISK_FREE_RATE,
            'trading_days_per_year': settings.TRADING_DAYS_PER_YEAR,
        }
    } 