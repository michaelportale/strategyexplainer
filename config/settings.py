"""
Legacy Application Settings and Configuration Management

DEPRECATED: This module is maintained for backward compatibility only.
New code should use config.config_manager.ConfigManager instead.

Historical Context:
This settings module represents the original configuration system for the
Strategy Explainer application. It follows a simple, environment variable-based
approach to configuration management that was sufficient for early development
but has been superseded by the more sophisticated YAML-based configuration
system in config_manager.py.

Legacy Architecture:
The original settings system was designed around these principles:
1. Environment Variables: All configuration through environment variables
2. Dataclass Structure: Type-safe configuration with @dataclass decorator
3. Simple Defaults: Hardcoded fallback values for all settings
4. Direct Access: Global settings instance for immediate access
5. Minimal Dependencies: No external configuration file requirements

Migration Path:
Applications using this legacy settings system should migrate to the new
configuration manager following this pattern:

```python
# Legacy approach (deprecated)
from config.settings import settings, get_config
debug_mode = settings.DEBUG
api_key = settings.OPENAI_API_KEY
config_dict = get_config()

# Modern approach (recommended)
from config.config_manager import get_config_manager, get_config
config = get_config_manager()
debug_mode = config.get('app.debug', False)
api_key = config.get('api.openai_api_key', '')
config_dict = get_config()
```

Key Differences Between Legacy and Modern Systems:
- Configuration Source: Environment variables only vs YAML with env substitution
- Structure: Flat namespace vs hierarchical organization
- Extensibility: Fixed fields vs dynamic configuration sections
- Validation: Basic type coercion vs comprehensive validation
- Documentation: Inline comments vs self-documenting YAML structure
- Environment Support: Manual env handling vs automatic substitution

Backward Compatibility Features:
This module continues to work alongside the new configuration system by:
1. Maintaining identical API surface for existing code
2. Providing the same configuration data structure
3. Supporting all legacy environment variable names
4. Offering seamless migration path for gradual updates

Configuration Categories in Legacy System:
- API Configuration: Service authentication keys
- Data Configuration: Market data sources and settings
- Application Configuration: Debug modes and logging
- Streamlit Configuration: UI behavior and appearance
- File Paths: Data storage and output directories
- Strategy Defaults: Common strategy parameter defaults
- Performance Settings: Financial calculation parameters

Deprecation Strategy:
1. Current Phase: Full backward compatibility with deprecation warnings
2. Warning Phase: Visible deprecation notices in logs and documentation
3. Migration Phase: Helper tools and automated migration scripts
4. Sunset Phase: Removal of legacy settings system

Educational Value:
This legacy module demonstrates the evolution of configuration management:
- Simple Environment Variable Patterns: Basic configuration approaches
- Type Safety with Dataclasses: Python typing for configuration
- Global Configuration State: Singleton pattern implementation
- Configuration Dictionary Export: API compatibility layers
- Backward Compatibility: Maintaining API contracts during transitions

Performance Characteristics:
- Initialization: O(1) environment variable lookups at startup
- Access: Direct attribute access (fastest possible)
- Memory: Minimal overhead with dataclass storage
- Threading: Thread-safe after initialization
- Migration: Zero performance impact during transition period

Security Considerations:
- Environment Variable Exposure: All secrets through environment variables
- Default Value Safety: Secure defaults for production deployment
- API Key Protection: No hardcoded credentials in source code
- Debug Mode Control: Environment-controlled debug information exposure

Author: Strategy Explainer Development Team
Version: 1.0 (Legacy)
Status: DEPRECATED - Use config.config_manager.ConfigManager
Last Updated: 2024
Migration: See config_manager.py for modern configuration system
"""

import os
import warnings
from typing import Dict, Any
from dataclasses import dataclass

# Issue deprecation warning to encourage migration to new configuration system
# This warning alerts developers to the deprecated status while maintaining functionality
warnings.warn(
    "config.settings is deprecated. Use config.config_manager.ConfigManager instead.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class Settings:
    """
    Legacy Application Settings and Constants (DEPRECATED).
    
    This dataclass provides type-safe access to application configuration
    through environment variables. It represents the original configuration
    system design before migration to the modern YAML-based approach.
    
    Design Philosophy (Legacy):
    - Environment Variable Driven: All configuration through env vars
    - Type Safety: Dataclass fields provide type checking
    - Simple Defaults: Hardcoded fallback values for reliability
    - Direct Access: Attribute-based configuration access
    - Production Ready: Secure defaults for production deployment
    
    Configuration Categories:
    1. API Configuration: External service authentication
    2. Data Configuration: Market data sources and caching
    3. Application Configuration: Debug modes and logging levels
    4. UI Configuration: Streamlit-specific settings
    5. File Paths: Storage directories and organization
    6. Strategy Defaults: Common trading strategy parameters
    7. Performance Settings: Financial calculation constants
    
    Environment Variable Mapping:
    Each field corresponds to an environment variable with the same name.
    If the environment variable is not set, the default value is used.
    
    Migration Notes:
    When migrating to the new configuration system, map these settings as follows:
    - API keys → api section in YAML
    - Data settings → data section in YAML
    - App settings → app section in YAML
    - Paths → paths section in YAML
    - Defaults → strategy defaults in YAML
    - Performance → performance_metrics section in YAML
    
    Thread Safety:
    This dataclass is thread-safe after initialization since all fields
    are set during class creation and never modified afterward.
    
    Security Features:
    - No hardcoded secrets (all through environment variables)
    - Secure production defaults
    - Debug mode controlled via environment
    - Sensitive data isolation through environment variables
    """
    
    # API Configuration - External Service Authentication
    # =================================================
    # These fields store API keys and authentication credentials for external
    # services. All values should come from environment variables to maintain
    # security and avoid hardcoding sensitive information.
    
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    """OpenAI API key for GPT-powered analysis and natural language generation.
    
    Required for AI features including:
    - Strategy explanations and commentary
    - Trade decision explanations
    - Performance analysis summaries
    - Risk assessment narratives
    
    Security: Must be set via environment variable only.
    Default: Empty string (disables AI features if not configured)
    """
    
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    """Alpha Vantage API key for alternative market data source.
    
    Used as backup or alternative to Yahoo Finance for:
    - Real-time market data
    - Historical price data
    - Fundamental data
    - Economic indicators
    
    Security: Must be set via environment variable only.
    Default: Empty string (uses primary data source if not configured)
    """
    
    # Data Configuration - Market Data and Caching
    # ===========================================
    # Settings that control market data retrieval, caching, and processing
    # behavior throughout the application.
    
    DATA_SOURCE: str = os.getenv('DATA_SOURCE', 'yfinance')
    """Primary data source for market data retrieval.
    
    Supported values:
    - 'yfinance': Yahoo Finance (free, reliable, good coverage)
    - 'alpha_vantage': Alpha Vantage (requires API key)
    - 'iex': IEX Cloud (requires API key, real-time data)
    
    Default: 'yfinance' for broad compatibility and no API key requirement
    """
    
    DEFAULT_BENCHMARK: str = os.getenv('DEFAULT_BENCHMARK', 'SPY')
    """Default benchmark symbol for performance comparison.
    
    Used for:
    - Relative performance analysis
    - Risk-adjusted return calculations
    - Beta calculations
    - Market correlation analysis
    
    Common benchmarks: SPY (S&P 500), QQQ (NASDAQ), VTI (Total Market)
    Default: 'SPY' as the most widely used equity benchmark
    """
    
    CACHE_DURATION_MINUTES: int = int(os.getenv('CACHE_DURATION_MINUTES', '60'))
    """Duration in minutes to cache market data for performance optimization.
    
    Balances data freshness with performance:
    - Higher values: Better performance, less fresh data
    - Lower values: More fresh data, more API calls
    - 0: No caching (always fetch fresh data)
    
    Default: 60 minutes for reasonable balance in development
    Production: Consider shorter durations for real-time applications
    """
    
    # Application Configuration - Core Behavior
    # ========================================
    # Settings that control fundamental application behavior, logging,
    # and development vs production modes.
    
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    """Debug mode flag for development and troubleshooting.
    
    When enabled:
    - Verbose logging and error details
    - Additional diagnostic information
    - Development-friendly error messages
    - Performance profiling information
    
    Security: Should be False in production to avoid information disclosure
    Default: False for production-safe behavior
    """
    
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    """Logging level for application-wide log output control.
    
    Supported levels (in order of verbosity):
    - 'DEBUG': Detailed diagnostic information
    - 'INFO': General informational messages
    - 'WARNING': Warning messages for potential issues
    - 'ERROR': Error messages for failures
    - 'CRITICAL': Critical system failures only
    
    Default: 'INFO' for balanced logging in development and production
    """
    
    # Streamlit Configuration - UI Behavior
    # ===================================
    # Settings specific to the Streamlit web application interface,
    # controlling appearance and server behavior.
    
    STREAMLIT_THEME: str = os.getenv('STREAMLIT_THEME', 'light')
    """Streamlit UI theme preference for consistent visual design.
    
    Supported themes:
    - 'light': Light theme with dark text on light background
    - 'dark': Dark theme with light text on dark background
    - 'auto': Automatic theme based on user's system preference
    
    Default: 'light' for broad compatibility and readability
    """
    
    STREAMLIT_PORT: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    """Port number for Streamlit web server.
    
    Considerations:
    - 8501: Streamlit default port
    - 8080: Common alternative for web applications
    - Custom ports: For multi-app deployments or port conflicts
    
    Default: 8501 (Streamlit standard)
    """
    
    # File Paths - Storage Organization
    # ===============================
    # Directory paths for organized data storage, outputs, and temporary files.
    # All paths are relative to the project root for portability.
    
    DATA_DIR: str = 'data'
    """Root directory for all data storage and organization.
    
    Contains:
    - Raw market data downloads
    - Processed datasets
    - Benchmark data
    - User-uploaded data files
    
    Organization: Subdirectories by data type and date
    """
    
    CACHE_DIR: str = 'data/cache'
    """Directory for cached data and temporary performance optimization files.
    
    Contains:
    - Market data cache files
    - Processed indicator calculations
    - Temporary computation results
    - Performance optimization data
    
    Maintenance: Can be safely cleared for troubleshooting
    """
    
    OUTPUTS_DIR: str = 'backend/outputs'
    """Root directory for all analysis outputs and generated content.
    
    Contains:
    - Backtest results
    - Performance reports
    - Analysis summaries
    - Generated charts and visualizations
    
    Organization: Subdirectories by analysis type and timestamp
    """
    
    REPORTS_DIR: str = 'backend/outputs/reports'
    """Directory specifically for generated reports and analysis documents.
    
    Contains:
    - PDF performance reports
    - CSV analysis summaries
    - Strategy comparison reports
    - Risk analysis documents
    
    Format: Human-readable reports for sharing and documentation
    """
    
    CHARTS_DIR: str = 'backend/outputs/charts'
    """Directory for generated charts, graphs, and visualizations.
    
    Contains:
    - PNG/SVG chart files
    - Interactive HTML charts
    - Custom visualization outputs
    - Chart templates and styles
    
    Usage: Visual content for reports and presentations
    """
    
    # Default Strategy Parameters - Trading Logic Defaults
    # ==================================================
    # Common default values for trading strategy parameters, providing
    # reasonable starting points for strategy development and testing.
    
    DEFAULT_LOOKBACK_PERIOD: int = 20
    """Default lookback period in days for technical indicators and signals.
    
    Usage:
    - Moving averages calculation window
    - Momentum calculation period
    - Volatility measurement window
    - Trend analysis timeframe
    
    Rationale: 20 days represents approximately one trading month
    """
    
    DEFAULT_THRESHOLD: float = 0.02
    """Default threshold for signal generation as decimal percentage.
    
    Usage:
    - Minimum price movement for signal triggering
    - Momentum threshold for trend following
    - Breakout detection sensitivity
    - Reversion signal confirmation
    
    Value: 0.02 = 2% movement threshold (common for equity strategies)
    """
    
    DEFAULT_POSITION_SIZE: float = 0.1
    """Default position size as fraction of total portfolio.
    
    Risk Management:
    - 0.1 = 10% maximum position size per trade
    - Diversification through multiple smaller positions
    - Capital preservation through position limits
    - Risk-adjusted portfolio construction
    
    Rationale: Conservative default suitable for most risk tolerances
    """
    
    DEFAULT_STOP_LOSS: float = 0.05
    """Default stop-loss percentage for risk management.
    
    Risk Control:
    - 0.05 = 5% maximum loss per position
    - Automatic position exit on adverse moves
    - Capital preservation mechanism
    - Emotional decision elimination
    
    Consideration: Balance between risk control and strategy flexibility
    """
    
    DEFAULT_TAKE_PROFIT: float = 0.10
    """Default take-profit percentage for profit realization.
    
    Profit Management:
    - 0.10 = 10% target profit per position
    - Systematic profit taking approach
    - Risk-reward ratio optimization
    - Prevents profit erosion in reversals
    
    Strategy: 2:1 reward-to-risk ratio with default settings
    """
    
    # Performance Metrics - Financial Calculation Constants
    # ===================================================
    # Standard financial market assumptions and constants used throughout
    # performance calculations and risk analysis.
    
    RISK_FREE_RATE: float = 0.02
    """Annual risk-free rate for Sharpe ratio and risk-adjusted return calculations.
    
    Financial Usage:
    - Sharpe ratio denominator
    - Risk premium calculations
    - Capital Asset Pricing Model (CAPM)
    - Risk-adjusted performance metrics
    
    Value: 0.02 = 2% annual rate (approximate US Treasury rate)
    Update: Should be adjusted based on current market conditions
    """
    
    TRADING_DAYS_PER_YEAR: int = 252
    """Number of trading days per year for annualized calculations.
    
    Financial Calendar:
    - Standard assumption in quantitative finance
    - Excludes weekends and major holidays
    - Used for volatility annualization
    - Basis for return annualization
    
    Alternative: 365 for calendar day calculations, 250 for conservative estimates
    """


# Global Legacy Settings Instance
# ==============================
# Singleton instance providing backward compatibility for existing code
# that expects direct access to configuration settings.

settings = Settings()
"""Global settings instance for backward compatibility.

This instance provides direct access to all configuration settings
using the legacy attribute-based approach. Maintained for existing
code that has not yet migrated to the new configuration system.

Usage Examples:
    # Legacy access patterns (still supported)
    debug_mode = settings.DEBUG
    api_key = settings.OPENAI_API_KEY
    data_dir = settings.DATA_DIR
    
    # Modern equivalent (recommended for new code)
    from config.config_manager import get_config
    debug_mode = get_config('app.debug', False)
    api_key = get_config('api.openai_api_key', '')
    data_dir = get_config('paths.data_dir', 'data')

Thread Safety:
Safe for concurrent access after module initialization since
settings values are immutable after startup.
"""


def get_config() -> Dict[str, Any]:
    """
    Get Application Configuration as Dictionary (Legacy Format).
    
    Provides configuration data in the dictionary format expected by
    existing code written for the original settings system. This function
    maintains API compatibility while the codebase transitions to the
    new configuration management system.
    
    Dictionary Structure:
    The returned configuration dictionary is organized into logical sections
    that group related settings together for easier access and management.
    
    Returns:
        Dict[str, Any]: Configuration dictionary with the following structure:
        
        ```python
        {
            'api_keys': {
                'openai': str,           # OpenAI API key
                'alpha_vantage': str,    # Alpha Vantage API key
            },
            'data': {
                'source': str,           # Primary data source
                'benchmark': str,        # Default benchmark symbol
                'cache_duration': int,   # Cache duration in minutes
            },
            'app': {
                'debug': bool,           # Debug mode flag
                'log_level': str,        # Logging level
            },
            'paths': {
                'data_dir': str,         # Data storage directory
                'cache_dir': str,        # Cache directory
                'outputs_dir': str,      # Analysis outputs directory
                'reports_dir': str,      # Reports directory
                'charts_dir': str,       # Charts directory
            },
            'defaults': {
                'lookback_period': int,  # Default lookback period
                'threshold': float,      # Default signal threshold
                'position_size': float,  # Default position size
                'stop_loss': float,      # Default stop loss
                'take_profit': float,    # Default take profit
            },
            'metrics': {
                'risk_free_rate': float, # Risk-free rate for calculations
                'trading_days_per_year': int, # Trading calendar assumption
            }
        }
        ```
    
    Usage Examples:
        # Get complete configuration
        config = get_config()
        
        # Access API keys
        openai_key = config['api_keys']['openai']
        
        # Access data settings
        data_source = config['data']['source']
        
        # Access application settings
        debug_mode = config['app']['debug']
        
        # Access file paths
        data_directory = config['paths']['data_dir']
        
        # Access default parameters
        default_lookback = config['defaults']['lookback_period']
        
        # Access performance metrics settings
        risk_free_rate = config['metrics']['risk_free_rate']
    
    Migration Strategy:
    Code using this function can be gradually migrated to the new
    configuration system:
    
    ```python
    # Current legacy approach
    from config.settings import get_config
    config = get_config()
    debug_mode = config['app']['debug']
    
    # Transitional approach
    from config.config_manager import get_settings
    config = get_settings()  # Same format as legacy
    debug_mode = config['app']['debug']
    
    # Modern approach (target)
    from config.config_manager import get_config_manager
    config = get_config_manager()
    debug_mode = config.get('app.debug', False)
    ```
    
    Backward Compatibility:
    This function maintains exact compatibility with the original settings
    format, ensuring existing code continues to work without modification
    during the migration period.
    
    Performance Characteristics:
    - Dictionary construction: O(1) with fixed number of settings
    - Memory usage: Minimal overhead for dictionary creation
    - Access patterns: Direct dictionary key access (O(1))
    - Thread safety: Safe after initialization
    
    Deprecation Timeline:
    - Phase 1: Full backward compatibility (current)
    - Phase 2: Deprecation warnings in development
    - Phase 3: Migration tools and automated updates
    - Phase 4: Legacy system removal
    """
    return {
        # API Keys Section - External Service Authentication
        'api_keys': {
            'openai': settings.OPENAI_API_KEY,
            'alpha_vantage': settings.ALPHA_VANTAGE_API_KEY,
        },
        
        # Data Configuration Section - Market Data and Sources
        'data': {
            'source': settings.DATA_SOURCE,
            'benchmark': settings.DEFAULT_BENCHMARK,
            'cache_duration': settings.CACHE_DURATION_MINUTES,
        },
        
        # Application Configuration Section - Core Behavior
        'app': {
            'debug': settings.DEBUG,
            'log_level': settings.LOG_LEVEL,
        },
        
        # File Paths Section - Storage and Organization
        'paths': {
            'data_dir': settings.DATA_DIR,
            'cache_dir': settings.CACHE_DIR,
            'outputs_dir': settings.OUTPUTS_DIR,
            'reports_dir': settings.REPORTS_DIR,
            'charts_dir': settings.CHARTS_DIR,
        },
        
        # Default Parameters Section - Strategy Defaults
        'defaults': {
            'lookback_period': settings.DEFAULT_LOOKBACK_PERIOD,
            'threshold': settings.DEFAULT_THRESHOLD,
            'position_size': settings.DEFAULT_POSITION_SIZE,
            'stop_loss': settings.DEFAULT_STOP_LOSS,
            'take_profit': settings.DEFAULT_TAKE_PROFIT,
        },
        
        # Performance Metrics Section - Financial Constants
        'metrics': {
            'risk_free_rate': settings.RISK_FREE_RATE,
            'trading_days_per_year': settings.TRADING_DAYS_PER_YEAR,
        }
    } 