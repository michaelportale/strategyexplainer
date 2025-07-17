"""Sidebar Configuration Component for Strategy Explainer Streamlit Application.

This module implements the main navigation and configuration sidebar for the Strategy Explainer
application. The sidebar provides an intuitive interface for users to select trading strategies,
configure parameters, set data ranges, and control various analysis options.

SIDEBAR FUNCTIONALITY:
=====================
The sidebar serves as the primary control panel for the application, enabling:

1. STRATEGY SELECTION: Choose from categorized trading strategies
   - Momentum/Trend strategies (SMA/EMA crossovers, RSI, breakouts)
   - Mean reversion strategies (Bollinger Bands, Z-score, RSI contrarian)
   - Combined strategies (regime switching, sentiment overlay)

2. PARAMETER CONFIGURATION: Dynamic parameter inputs for selected strategies
   - Strategy-specific parameters loaded from configuration files
   - Input validation and sensible defaults
   - Real-time parameter updates affecting analysis

3. DATA MANAGEMENT: Control data sources and time ranges
   - Date range selection for backtesting
   - Ticker symbol input and validation
   - Data source configuration (local, API, etc.)

4. ANALYSIS OPTIONS: Configure analysis and visualization settings
   - Benchmark comparison selection
   - Risk metrics calculation options
   - Visualization preferences and chart types

CONFIGURATION SYSTEM INTEGRATION:
================================
The sidebar integrates with the unified configuration management system:
- Loads strategy definitions from config.yaml
- Supports both new unified config and legacy configurations
- Provides parameter validation and default value management
- Enables real-time configuration updates

STREAMLIT UI DESIGN PRINCIPLES:
==============================
- INTUITIVE LAYOUT: Logical grouping of related controls
- RESPONSIVE DESIGN: Adapts to different screen sizes and content
- USER FEEDBACK: Clear labels, help text, and validation messages
- CONSISTENT STYLING: Follows application design patterns

USAGE PATTERNS:
==============
The sidebar is typically called from main application pages:

```python
# In main app page
from app.components.sidebar import render_sidebar

# Get user configuration from sidebar
config = render_sidebar()

# Use configuration for strategy execution
strategy = create_strategy(config['strategy'], config['parameters'])
results = strategy.backtest(config['data_range'], config['ticker'])
```
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import yaml
import logging
import sys
import os
from pathlib import Path
from config.config_manager import get_config_manager
from datetime import datetime, timedelta
import pandas as pd

# Initialize logging for frontend component
logger = logging.getLogger(__name__)


def load_strategy_configs() -> Dict[str, Any]:
    """Load strategy configurations from the unified configuration management system.
    
    This function interfaces with the config manager to retrieve all available strategy
    definitions, parameters, and metadata. It handles both the new unified configuration
    system and legacy configuration files for backward compatibility.
    
    CONFIGURATION STRUCTURE:
    =======================
    The returned configuration follows this hierarchy:
    ```
    {
        'momentum': {
            'SmaEmaRsiStrategy': {
                'parameters': {...},
                'description': '...',
                'category': 'momentum'
            }
        },
        'mean_reversion': {
            'BollingerBandMeanReversionStrategy': {...}
        }
    }
    ```
    
    ERROR HANDLING:
    ==============
    - Gracefully handles missing configuration files
    - Provides fallback to default configurations
    - Displays user-friendly error messages in Streamlit
    - Logs detailed error information for debugging
    
    Returns:
        Dict[str, Any]: Nested dictionary of strategy configurations by category
                       Empty dict if configuration loading fails
                       
    Raises:
        None: All exceptions are caught and handled gracefully with error display
    """
    try:
        # Access the unified configuration manager
        config_manager = get_config_manager()
        strategy_configs = config_manager.get_strategy_definitions()
        
        if not strategy_configs:
            logger.warning("No strategy configurations found in config manager")
            st.warning("⚠️ No strategy configurations found. Using default settings.")
            return _get_default_strategy_configs()
        
        logger.info(f"Successfully loaded {len(strategy_configs)} strategy configurations")
        return strategy_configs
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        st.error(f"❌ Configuration file not found: {e}")
        st.info("💡 Using default strategy configurations.")
        return _get_default_strategy_configs()
        
    except yaml.YAMLError as e:
        logger.error(f"YAML configuration error: {e}")
        st.error(f"❌ Configuration file format error: {e}")
        st.info("💡 Please check your config.yaml file format.")
        return _get_default_strategy_configs()
        
    except Exception as e:
        logger.error(f"Unexpected error loading strategy configurations: {e}", exc_info=True)
        st.error(f"❌ Failed to load strategy configurations: {e}")
        st.info("💡 Contact administrator if this problem persists.")
        return _get_default_strategy_configs()


def _get_default_strategy_configs() -> Dict[str, Any]:
    """Provide fallback strategy configurations when config loading fails.
    
    This function returns a minimal set of default strategy configurations
    to ensure the application remains functional even when configuration
    files are missing or corrupted.
    
    Returns:
        Dict[str, Any]: Basic strategy configurations for core strategies
    """
    return {
        'momentum': {
            'SmaEmaRsiStrategy': {
                'parameters': {
                    'sma_short': {'default': 20, 'min': 5, 'max': 50},
                    'sma_long': {'default': 50, 'min': 20, 'max': 200},
                    'rsi_period': {'default': 14, 'min': 5, 'max': 30}
                },
                'description': 'Combined SMA/EMA and RSI momentum strategy'
            }
        },
        'mean_reversion': {
            'BollingerBandMeanReversionStrategy': {
                'parameters': {
                    'bb_period': {'default': 20, 'min': 10, 'max': 50},
                    'bb_std': {'default': 2.0, 'min': 1.0, 'max': 3.0}
                },
                'description': 'Mean reversion using Bollinger Bands'
            }
        }
    }


def render_strategy_selection() -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """Render strategy category and type selection interface in the sidebar.
    
    This function creates a two-level selection interface where users first choose
    a strategy category (momentum, mean_reversion, etc.) and then select a specific
    strategy within that category. The interface dynamically updates available
    strategies based on the selected category.
    
    SELECTION WORKFLOW:
    ==================
    1. Load available strategy configurations
    2. Present category selection dropdown
    3. Filter strategies by selected category
    4. Present strategy type selection dropdown
    5. Return selected strategy information
    
    UI COMPONENTS:
    =============
    - Category selectbox: Groups strategies by trading approach
    - Strategy selectbox: Shows available strategies in selected category
    - Help text: Provides strategy descriptions and guidance
    - Error handling: Graceful degradation when strategies unavailable
    
    Returns:
        Tuple containing:
        - selected_category (str or None): Chosen strategy category
        - selected_strategy (str or None): Chosen strategy type
        - strategy_config (dict or None): Configuration for selected strategy
        
    Note:
        Returns (None, None, None) if no strategies are available or configured
    """
    st.sidebar.header("🎯 Strategy Selection")
    
    # Load strategy configurations
    strategy_configs = load_strategy_configs()
    
    if not strategy_configs:
        st.sidebar.error("No strategies available")
        return None, None, None
    
    # Filter out global configuration if present
    categories = [cat for cat in strategy_configs.keys() if cat != 'global']
    
    if not categories:
        st.sidebar.error("No strategy categories found")
        return None, None, None
    
    # Strategy Category Selection
    selected_category = st.sidebar.selectbox(
        "📊 Strategy Category",
        categories,
        index=0,
        help="Choose the type of trading approach. Momentum strategies follow trends, "
             "while mean reversion strategies trade against extremes."
    )
    
    if not selected_category or selected_category not in strategy_configs:
        return None, None, None
    
    # Strategy Type Selection within category
    category_strategies = strategy_configs[selected_category]
    
    if not category_strategies:
        st.sidebar.warning(f"No strategies found in {selected_category} category")
        return selected_category, None, None
    
    strategy_types = list(category_strategies.keys())
    
    selected_strategy = st.sidebar.selectbox(
        "⚙️ Strategy Type",
        strategy_types,
        index=0,
        help="Select the specific strategy implementation to analyze."
    )
    
    if not selected_strategy:
        return selected_category, None, None
    
    # Get strategy configuration
    strategy_config = category_strategies.get(selected_strategy, {})
    
    # Display strategy description if available
    description = strategy_config.get('description', '')
    if description:
        st.sidebar.info(f"ℹ️ {description}")
    
    return selected_category, selected_strategy, strategy_config


def render_parameter_inputs(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Render dynamic parameter input controls with preset selection for the selected strategy.
    
    This function creates appropriate Streamlit input widgets based on the strategy's
    parameter configuration, now enhanced with preset functionality allowing users to
    choose from Conservative, Moderate, or Aggressive parameter profiles.
    
    ENHANCED FEATURES:
    =================
    - Preset Selection: Choose from pre-configured parameter sets
    - Custom Parameters: Manually configure individual parameters
    - Real-time Updates: Preset selection auto-populates parameters
    - Parameter Override: Ability to customize preset parameters
    
    PARAMETER TYPES SUPPORTED:
    =========================
    - Integer parameters: Using number_input with min/max validation
    - Float parameters: Using number_input with step and precision control
    - Boolean parameters: Using checkbox for true/false options
    - Selection parameters: Using selectbox for enumerated choices
    - Range parameters: Using slider for bounded numeric inputs
    
    PRESET SYSTEM:
    =============
    - Conservative: Lower risk, smaller positions, tighter stops
    - Moderate: Balanced risk-reward, standard parameters
    - Aggressive: Higher risk, larger positions, wider stops
    - Custom: User-defined parameter configuration
    
    Args:
        strategy_config (Dict[str, Any]): Strategy configuration containing parameter definitions
                                        Expected format: {'parameters': {...}, 'presets': {...}}
    
    Returns:
        Dict[str, Any]: User-configured parameter values for strategy initialization
                       Includes preset selection and individual parameter overrides
    """
    st.sidebar.header("⚙️ Strategy Parameters")
    
    parameters = strategy_config.get('parameters', {})
    presets = strategy_config.get('presets', {})
    
    if not parameters:
        st.sidebar.info("ℹ️ This strategy uses default parameters")
        return {}
    
    # Enhanced Preset Selection Interface
    # ====================================
    selected_preset = None
    preset_params = {}
    
    if presets:
        st.sidebar.subheader("🎯 Risk Profile")
        
        # Create preset options with descriptions
        preset_options = ["Custom"] + list(presets.keys())
        preset_labels = {
            "Custom": "🔧 Custom Parameters",
            "conservative": "🛡️ Conservative (Lower Risk)",
            "moderate": "⚖️ Moderate (Balanced)",
            "aggressive": "🚀 Aggressive (Higher Risk)"
        }
        
        # Preset selection with enhanced UI
        selected_preset = st.sidebar.selectbox(
            "Select Risk Profile",
            options=preset_options,
            format_func=lambda x: preset_labels.get(x, x.title()),
            help="Choose a pre-configured risk profile or customize parameters manually"
        )
        
        # Display preset description
        if selected_preset != "Custom" and selected_preset in presets:
            preset_info = presets[selected_preset]
            st.sidebar.info(f"ℹ️ {preset_info.get('description', 'Pre-configured parameter set')}")
            preset_params = preset_info.get('parameters', {})
            
            # Show preset values in an expandable section
            with st.sidebar.expander("📋 Preset Values", expanded=False):
                for param_name, value in preset_params.items():
                    st.markdown(f"• **{_format_parameter_label(param_name)}**: {value}")
        
        st.sidebar.markdown("---")
        
        # Allow parameter customization for presets
        if selected_preset != "Custom":
            show_custom = st.sidebar.checkbox(
                "🔧 Customize Parameters", 
                value=False,
                help="Fine-tune the selected preset parameters"
            )
        else:
            show_custom = True
    else:
        show_custom = True
    
    user_params = {}
    
    # Create input widgets for each parameter
    if show_custom:
        for param_name, param_config in parameters.items():
            if not isinstance(param_config, dict):
                continue
            
            # Use preset value as default if available, otherwise use config default
            if selected_preset and selected_preset != "Custom" and param_name in preset_params:
                default_value = preset_params[param_name]
            else:
                default_value = param_config.get('default', 0)
                
            # Extract parameter configuration
            param_type = param_config.get('type', 'number')
            min_value = param_config.get('min', None)
            max_value = param_config.get('max', None)
            step = param_config.get('step', None)
            help_text = param_config.get('help', f"Configure {param_name.replace('_', ' ')}")
            
            # Create appropriate input widget based on parameter type
            if param_type == 'boolean' or isinstance(default_value, bool):
                # Boolean checkbox input
                user_params[param_name] = st.sidebar.checkbox(
                    _format_parameter_label(param_name),
                    value=bool(default_value),
                    help=help_text
                )
                
            elif param_type == 'select' and 'options' in param_config:
                # Selection dropdown input
                options = param_config['options']
                default_index = 0
                if default_value in options:
                    default_index = options.index(default_value)
                    
                user_params[param_name] = st.sidebar.selectbox(
                    _format_parameter_label(param_name),
                    options=options,
                    index=default_index,
                    help=help_text
                )
                
            elif param_type == 'slider' or (min_value is not None and max_value is not None):
                # Slider input for bounded ranges
                if isinstance(default_value, float):
                    step = step or 0.01
                    user_params[param_name] = st.sidebar.slider(
                        _format_parameter_label(param_name),
                        min_value=float(min_value) if min_value is not None else 0.0,
                        max_value=float(max_value) if max_value is not None else 1.0,
                        value=float(default_value),
                        step=step,
                        help=help_text
                    )
                else:
                    user_params[param_name] = st.sidebar.slider(
                        _format_parameter_label(param_name),
                        min_value=int(min_value) if min_value is not None else 1,
                        max_value=int(max_value) if max_value is not None else 100,
                        value=int(default_value),
                        step=int(step) if step else 1,
                        help=help_text
                    )
                    
            else:
                # Numeric input (integer or float)
                if isinstance(default_value, float):
                    step = step or 0.01
                    user_params[param_name] = st.sidebar.number_input(
                        _format_parameter_label(param_name),
                        min_value=float(min_value) if min_value is not None else None,
                        max_value=float(max_value) if max_value is not None else None,
                        value=float(default_value),
                        step=step,
                        help=help_text
                    )
                else:
                    user_params[param_name] = st.sidebar.number_input(
                        _format_parameter_label(param_name),
                        min_value=int(min_value) if min_value is not None else None,
                        max_value=int(max_value) if max_value is not None else None,
                        value=int(default_value),
                        step=int(step) if step else 1,
                        help=help_text
                    )
    else:
        # Use preset parameters directly without showing individual inputs
        user_params = preset_params.copy()
    
    # Add preset information to the returned parameters
    if presets:
        user_params['_preset_selected'] = selected_preset or "Custom"
    
    return user_params


def _format_parameter_label(param_name: str) -> str:
    """Format parameter names for user-friendly display in the sidebar.
    
    Converts technical parameter names to readable labels with proper capitalization
    and spacing for better user experience.
    
    Args:
        param_name (str): Technical parameter name (e.g., 'sma_short', 'rsi_period')
        
    Returns:
        str: Formatted label for display (e.g., 'SMA Short', 'RSI Period')
    """
    # Replace underscores with spaces and title case
    formatted = param_name.replace('_', ' ').title()
    
    # Handle common abbreviations for better readability
    abbreviations = {
        'Sma': 'SMA',
        'Ema': 'EMA', 
        'Rsi': 'RSI',
        'Atr': 'ATR',
        'Bb': 'Bollinger Band',
        'Std': 'Standard Deviation'
    }
    
    for abbrev, full_form in abbreviations.items():
        formatted = formatted.replace(abbrev, full_form)
    
    return formatted


def render_data_configuration() -> Dict[str, Any]:
    """Render enhanced data configuration controls with advanced options.
    
    This function provides comprehensive data configuration options including
    ticker selection, date ranges, benchmark comparison, and data sampling options
    with enhanced validation and user guidance.
    
    ENHANCED FEATURES:
    =================
    - Smart ticker symbol input with suggestions
    - Benchmark selection for comparison analysis
    - Date range presets (1Y, 2Y, 5Y) for quick selection
    - Data sampling options for performance optimization
    - Real-time validation with helpful feedback
    
    ADVANCED OPTIONS:
    ================
    - Multiple benchmark indices (SPY, QQQ, IWM, etc.)
    - Custom date range with validation
    - Data quality settings and caching options
    - Performance optimization controls
    
    Returns:
        Dict[str, Any]: Comprehensive data configuration including:
                       - ticker: Selected stock symbol
                       - start_date: Analysis start date
                       - end_date: Analysis end date
                       - benchmark: Selected benchmark for comparison
                       - sampling: Data sampling options
                       - valid: Configuration validation status
    """
    st.sidebar.header("📈 Data Configuration")
    
    # Enhanced ticker symbol input with suggestions
    ticker = st.sidebar.text_input(
        "🎯 Ticker Symbol",
        value="AAPL",
        help="💡 Popular symbols: AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ"
    ).upper().strip()
    
    # Real-time ticker validation with helpful feedback
    if ticker and not _is_valid_ticker(ticker):
        st.sidebar.warning("⚠️ Please enter a valid ticker symbol (3-5 characters, letters only)")
        st.sidebar.info("Examples: AAPL, MSFT, GOOGL")
    elif ticker:
        st.sidebar.success(f"✅ {ticker} - Ready for analysis")
    
    # Enhanced benchmark selection
    st.sidebar.subheader("📊 Benchmark Comparison")
    
    benchmark_options = {
        "SPY": "S&P 500 ETF - Large Cap US Stocks",
        "QQQ": "NASDAQ-100 ETF - Tech Heavy Index", 
        "IWM": "Russell 2000 ETF - Small Cap US Stocks",
        "VTI": "Total Stock Market ETF - Broad US Market",
        "EFA": "EAFE ETF - International Developed Markets",
        "VWO": "Emerging Markets ETF - EM Stocks",
        "AGG": "Aggregate Bond ETF - US Bond Market",
        "GLD": "Gold ETF - Precious Metals",
        "None": "No Benchmark Comparison"
    }
    
    selected_benchmark = st.sidebar.selectbox(
        "Select Benchmark",
        options=list(benchmark_options.keys()),
        index=0,  # Default to SPY
        format_func=lambda x: benchmark_options[x],
        help="Choose a benchmark index to compare your strategy performance against"
    )
    
    # Date range selection with presets
    st.sidebar.subheader("📅 Analysis Period")
    
    # Quick date range presets
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("1Y", help="Last 1 year", use_container_width=True):
            st.session_state.date_preset = "1Y"
    with col2:
        if st.button("2Y", help="Last 2 years", use_container_width=True):
            st.session_state.date_preset = "2Y"
    with col3:
        if st.button("5Y", help="Last 5 years", use_container_width=True):
            st.session_state.date_preset = "5Y"
    
    # Default dates based on preset or manual selection
    if hasattr(st.session_state, 'date_preset'):
        if st.session_state.date_preset == "1Y":
            default_start = datetime.now() - timedelta(days=365)
        elif st.session_state.date_preset == "2Y":
            default_start = datetime.now() - timedelta(days=730)
        elif st.session_state.date_preset == "5Y":
            default_start = datetime.now() - timedelta(days=1825)
        else:
            default_start = datetime.now() - timedelta(days=730)
    else:
        default_start = datetime.now() - timedelta(days=730)
    
    default_end = datetime.now()
    
    # Custom date range input
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            max_value=datetime.now(),
            help="📅 Beginning of backtest period. Longer periods provide more data but may take longer to process."
        )
    
    with col2:
        end_date = st.sidebar.date_input(
            "End Date", 
            value=default_end,
            max_value=datetime.now(),
            help="📅 End of backtest period. Use recent date to include latest market conditions."
        )
    
    # Enhanced date validation with detailed feedback
    if start_date >= end_date:
        st.sidebar.error("❌ Start date must be before end date")
        date_valid = False
    elif start_date < datetime(1990, 1, 1).date():
        st.sidebar.warning("⚠️ Very old start dates may have limited data availability")
        date_valid = True
    else:
        date_valid = True
    
    # Calculate and display analysis period with insights
    if start_date < end_date:
        period_days = (end_date - start_date).days
        period_years = period_days / 365.25
        
        if period_days < 30:
            st.sidebar.warning("⚠️ Short analysis period (< 1 month) may not provide reliable results")
        elif period_days < 252:  # Less than 1 trading year
            st.sidebar.info(f"📊 Analysis period: {period_days} days ({period_years:.1f} years)")
        elif period_days > 1825:  # More than 5 years
            st.sidebar.success(f"📊 Analysis period: {period_days} days ({period_years:.1f} years) - Excellent for robust analysis")
        else:
            st.sidebar.success(f"📊 Analysis period: {period_days} days ({period_years:.1f} years) - Good for analysis")
    
    # Advanced data options
    st.sidebar.subheader("⚙️ Advanced Options")
    
    # Data sampling for performance optimization
    with st.sidebar.expander("🔧 Performance Settings", expanded=False):
        data_sampling = st.selectbox(
            "Data Sampling",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            help="📈 Daily: Most accurate but slower. Weekly/Monthly: Faster but less precise."
        )
        
        cache_data = st.checkbox(
            "Enable Data Caching",
            value=True,
            help="💾 Cache downloaded data to improve performance on repeated analysis"
        )
        
        include_volume = st.checkbox(
            "Include Volume Data",
            value=True,
            help="📊 Include trading volume in analysis (some strategies use volume signals)"
        )
    
    # Comprehensive validation
    ticker_valid = ticker and _is_valid_ticker(ticker)
    config_valid = ticker_valid and date_valid
    
    return {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'benchmark': selected_benchmark if selected_benchmark != "None" else None,
        'data_sampling': data_sampling,
        'cache_enabled': cache_data,
        'include_volume': include_volume,
        'period_days': (end_date - start_date).days if date_valid else 0,
        'valid': config_valid
    }


def _is_valid_ticker(ticker: str) -> bool:
    """Validate ticker symbol format and basic requirements.
    
    Args:
        ticker (str): Ticker symbol to validate
        
    Returns:
        bool: True if ticker appears valid, False otherwise
    """
    if not ticker or len(ticker) < 1 or len(ticker) > 5:
        return False
    
    # Basic format validation - letters and occasionally dots for some tickers
    return ticker.replace('.', '').isalpha()


def render_analysis_options() -> Dict[str, Any]:
    """Render enhanced analysis and visualization options with detailed controls.
    
    This function provides comprehensive analysis customization options with
    detailed tooltips, performance considerations, and professional settings
    for institutional-grade analysis.
    
    ENHANCED ANALYSIS OPTIONS:
    =========================
    - Performance metrics with complexity levels
    - Advanced risk analysis with confidence settings
    - Professional visualization customization
    - AI-powered insights and explanations
    - Export and reporting preferences
    
    Returns:
        Dict[str, Any]: Comprehensive analysis configuration with professional options
    """
    st.sidebar.header("📊 Analysis Options")
    
    # Performance Analysis Section
    with st.sidebar.expander("📈 Performance Analysis", expanded=True):
        show_drawdown = st.checkbox(
            "📉 Drawdown Analysis",
            value=True,
            help="📊 Display maximum drawdown curves and underwater periods. Essential for understanding risk periods."
        )
        
        show_rolling_metrics = st.checkbox(
            "🔄 Rolling Metrics", 
            value=True,
            help="📈 Calculate time-varying performance metrics (rolling Sharpe, volatility). Shows performance consistency over time."
        )
        
        show_monthly_returns = st.checkbox(
            "📅 Monthly Returns Heatmap",
            value=True,
            help="🗓️ Visualize monthly performance patterns and seasonality effects in strategy returns."
        )
        
        performance_confidence = st.slider(
            "Confidence Level",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help="📊 Statistical confidence level for performance intervals and risk metrics (higher = more conservative)"
        )
    
    # Risk Analysis Section
    with st.sidebar.expander("⚠️ Risk Analysis", expanded=True):
        calculate_var = st.checkbox(
            "📊 Value at Risk (VaR)",
            value=True,
            help="⚠️ Calculate potential losses at specified confidence levels. Important for risk management."
        )
        
        calculate_cvar = st.checkbox(
            "📈 Conditional VaR",
            value=False,
            help="📊 Calculate expected loss beyond VaR threshold. Advanced risk metric for tail risk assessment."
        )
        
        stress_testing = st.checkbox(
            "🔥 Stress Testing",
            value=False,
            help="🧪 Test strategy performance under extreme market conditions (may increase processing time)."
        )
        
        correlation_analysis = st.checkbox(
            "🔗 Correlation Analysis",
            value=True,
            help="📊 Analyze correlation with market factors and other assets for diversification insights."
        )
    
    # Visualization and UI Section
    with st.sidebar.expander("🎨 Visualization", expanded=True):
        chart_style = st.radio(
            "Chart Theme",
            options=["Professional", "Modern", "Classic"],
            index=0,
            help="🎨 Professional: Clean institutional style. Modern: Colorful and engaging. Classic: Traditional financial charts."
        )
        
        chart_resolution = st.selectbox(
            "Chart Quality",
            options=["Standard", "High", "Ultra"],
            index=1,
            help="📊 Higher resolution provides better quality but may load slower. Ultra recommended for presentations."
        )
        
        show_signals = st.checkbox(
            "🎯 Trading Signals",
            value=True,
            help="📍 Display buy/sell signal markers on price charts with entry/exit points."
        )
        
        show_volume = st.checkbox(
            "📊 Volume Bars",
            value=True,
            help="📈 Include trading volume information in price charts for better market context."
        )
        
        interactive_charts = st.checkbox(
            "🖱️ Interactive Charts",
            value=True,
            help="🔍 Enable zoom, pan, and hover features on charts for detailed exploration."
        )
    
    # AI and Insights Section
    with st.sidebar.expander("🤖 AI Features", expanded=False):
        enable_ai_insights = st.checkbox(
            "💡 AI Strategy Insights",
            value=False,
            help="🤖 Generate AI-powered explanations and insights about strategy performance (requires OpenAI API key)."
        )
        
        ai_detail_level = st.selectbox(
            "AI Detail Level",
            options=["Summary", "Detailed", "Comprehensive"],
            index=1,
            help="📝 Summary: Key points only. Detailed: Analysis with explanations. Comprehensive: Full professional report.",
            disabled=not enable_ai_insights
        )
        
        trade_explanations = st.checkbox(
            "📝 Trade Explanations",
            value=False,
            help="💬 Generate AI explanations for individual trading decisions and market context.",
            disabled=not enable_ai_insights
        )
    
    # Export and Reporting Section
    with st.sidebar.expander("📄 Export Options", expanded=False):
        include_raw_data = st.checkbox(
            "📊 Include Raw Data",
            value=True,
            help="💾 Include raw price and signal data in exports for further analysis."
        )
        
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "Excel", "PDF Report"],
            index=0,
            help="📋 CSV: Data only. Excel: Formatted data. PDF: Complete professional report with charts."
        )
        
        auto_save_results = st.checkbox(
            "💾 Auto-Save Results",
            value=False,
            help="🔄 Automatically save analysis results to outputs directory with timestamp."
        )
    
    return {
        # Performance options
        'show_drawdown': show_drawdown,
        'show_rolling_metrics': show_rolling_metrics,
        'show_monthly_returns': show_monthly_returns,
        'performance_confidence': performance_confidence / 100,
        
        # Risk options
        'calculate_var': calculate_var,
        'calculate_cvar': calculate_cvar,
        'stress_testing': stress_testing,
        'correlation_analysis': correlation_analysis,
        
        # Visualization options
        'chart_style': chart_style.lower(),
        'chart_resolution': chart_resolution.lower(),
        'show_signals': show_signals,
        'show_volume': show_volume,
        'interactive_charts': interactive_charts,
        
        # AI options
        'enable_ai_insights': enable_ai_insights,
        'ai_detail_level': ai_detail_level.lower(),
        'trade_explanations': trade_explanations,
        
        # Export options
        'include_raw_data': include_raw_data,
        'export_format': export_format.lower(),
        'auto_save_results': auto_save_results
    }


def render_sidebar() -> Dict[str, Any]:
    """Main function to render the complete sidebar configuration interface.
    
    This is the primary entry point for the sidebar component. It orchestrates
    all sidebar sections and returns a comprehensive configuration dictionary
    that can be used by the main application pages.
    
    SIDEBAR SECTIONS:
    ================
    1. Strategy Selection: Choose and configure trading strategies
    2. Data Configuration: Set up data sources and time ranges  
    3. Analysis Options: Configure analysis and visualization settings
    4. Quick Actions: Provide shortcuts for common operations
    
    CONFIGURATION VALIDATION:
    ========================
    The function performs comprehensive validation of all user inputs and
    provides clear feedback about any configuration issues that need to be
    resolved before analysis can proceed.
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary containing:
                       - strategy_category: Selected strategy category
                       - strategy_type: Selected strategy implementation
                       - strategy_config: Strategy configuration and metadata
                       - parameters: User-configured strategy parameters
                       - data_config: Data source and range configuration
                       - analysis_options: Analysis and visualization settings
                       - valid: Boolean indicating if configuration is complete and valid
                       
    Usage:
        ```python
        # In main application page
        config = render_sidebar()
        
        if config['valid']:
            # Proceed with analysis using configuration
            strategy = create_strategy(config)
            results = run_backtest(strategy, config)
        else:
            st.warning("Please complete the configuration in the sidebar")
        ```
    """
    # Application title and branding
    st.sidebar.title("🎯 Strategy Explainer")
    st.sidebar.markdown("---")
    
    # Strategy selection section
    selected_category, selected_strategy, strategy_config = render_strategy_selection()
    
    # Strategy parameters section (only if strategy selected)
    parameters = {}
    if strategy_config:
        parameters = render_parameter_inputs(strategy_config)
    
    st.sidebar.markdown("---")
    
    # Data configuration section
    data_config = render_data_configuration()
    
    st.sidebar.markdown("---")
    
    # Analysis options section
    analysis_options = render_analysis_options()
    
    st.sidebar.markdown("---")
    
    # Quick actions section
    st.sidebar.subheader("🚀 Quick Actions")
    
    if st.sidebar.button("🔄 Reset to Defaults", help="Reset all parameters to default values"):
        st.experimental_rerun()
    
    # Validate overall configuration
    config_valid = all([
        selected_category is not None,
        selected_strategy is not None,
        data_config.get('valid', False)
    ])
    
    # Configuration status indicator
    if config_valid:
        st.sidebar.success("✅ Configuration Complete")
    else:
        st.sidebar.warning("⚠️ Please complete configuration")
    
    # Compile and return complete configuration
    return {
        'strategy_category': selected_category,
        'strategy_type': selected_strategy,
        'strategy_config': strategy_config,
        'parameters': parameters,
        'data_config': data_config,
        'analysis_options': analysis_options,
        'valid': config_valid
    }


def display_strategy_info(config: Dict[str, Any]) -> None:
    """Display information about the selected strategy."""
    strategy = config.get('strategy', {})
    
    if strategy.get('name'):
        st.sidebar.info(f"**{strategy['name']}**\n\n{strategy.get('description', '')}")
    
    # Display parameter summary
    if strategy.get('parameters'):
        st.sidebar.write("**Current Parameters:**")
        for param, value in strategy['parameters'].items():
            if isinstance(value, float):
                st.sidebar.write(f"• {param.replace('_', ' ').title()}: {value:.3f}")
            else:
                st.sidebar.write(f"• {param.replace('_', ' ').title()}: {value}")


def render_run_button() -> bool:
    """Render the run analysis button."""
    return st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        help="Click to run the backtest and generate analysis"
    ) 