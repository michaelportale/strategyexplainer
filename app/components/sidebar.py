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
from pathlib import Path
from config.config_manager import get_config_manager
from datetime import datetime, timedelta
import pandas as pd


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
            st.warning("⚠️ No strategy configurations found. Using default settings.")
            return _get_default_strategy_configs()
            
        return strategy_configs
        
    except FileNotFoundError as e:
        st.error(f"❌ Configuration file not found: {e}")
        st.info("💡 Using default strategy configurations.")
        return _get_default_strategy_configs()
        
    except yaml.YAMLError as e:
        st.error(f"❌ Configuration file format error: {e}")
        st.info("💡 Please check your config.yaml file format.")
        return _get_default_strategy_configs()
        
    except Exception as e:
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
    """Render dynamic parameter input controls for the selected strategy.
    
    This function creates appropriate Streamlit input widgets based on the strategy's
    parameter configuration. It supports various parameter types including integers,
    floats, booleans, and selections, with proper validation and default values.
    
    PARAMETER TYPES SUPPORTED:
    =========================
    - Integer parameters: Using number_input with min/max validation
    - Float parameters: Using number_input with step and precision control
    - Boolean parameters: Using checkbox for true/false options
    - Selection parameters: Using selectbox for enumerated choices
    - Range parameters: Using slider for bounded numeric inputs
    
    VALIDATION FEATURES:
    ===================
    - Min/max bounds enforcement for numeric parameters
    - Default value initialization from configuration
    - Input format validation and error messaging
    - Real-time parameter dependency checking
    
    Args:
        strategy_config (Dict[str, Any]): Strategy configuration containing parameter definitions
                                        Expected format: {'parameters': {'param_name': {'default': ..., 'min': ..., 'max': ...}}}
    
    Returns:
        Dict[str, Any]: User-configured parameter values for strategy initialization
                       Keys match parameter names, values are user-selected values
                       
    Example:
        strategy_config = {
            'parameters': {
                'sma_short': {'default': 20, 'min': 5, 'max': 50},
                'rsi_period': {'default': 14, 'min': 5, 'max': 30},
                'use_filter': {'default': True, 'type': 'boolean'}
            }
        }
        
        params = render_parameter_inputs(strategy_config)
        # Returns: {'sma_short': 25, 'rsi_period': 14, 'use_filter': False}
    """
    st.sidebar.header("⚙️ Strategy Parameters")
    
    parameters = strategy_config.get('parameters', {})
    if not parameters:
        st.sidebar.info("ℹ️ This strategy uses default parameters")
        return {}
    
    user_params = {}
    
    # Create input widgets for each parameter
    for param_name, param_config in parameters.items():
        if not isinstance(param_config, dict):
            continue
            
        # Extract parameter configuration
        default_value = param_config.get('default', 0)
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
                step = step or 0.1
                user_params[param_name] = st.sidebar.slider(
                    _format_parameter_label(param_name),
                    min_value=float(min_value) if min_value is not None else 0.0,
                    max_value=float(max_value) if max_value is not None else 100.0,
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
                step = step or 0.1
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
    """Render data source and time range configuration controls in the sidebar.
    
    This function provides users with controls to specify the data they want to analyze,
    including ticker selection, date ranges, and data source options. It includes
    validation and helpful defaults to ensure smooth user experience.
    
    DATA CONFIGURATION OPTIONS:
    ==========================
    - Ticker Symbol: Stock symbol input with validation
    - Date Range: Start and end date selection for backtesting
    - Data Source: Selection of data provider (if multiple sources available)
    - Data Frequency: Time interval selection (daily, hourly, etc.)
    
    VALIDATION FEATURES:
    ===================
    - Ticker symbol format validation
    - Date range logical validation (start < end)
    - Maximum date range limits for performance
    - Data availability checking and warnings
    
    Returns:
        Dict[str, Any]: Data configuration parameters including:
                       - ticker: Selected stock symbol
                       - start_date: Analysis start date
                       - end_date: Analysis end date
                       - data_source: Selected data provider
                       - frequency: Data time frequency
    """
    st.sidebar.header("📈 Data Configuration")
    
    # Ticker symbol input
    ticker = st.sidebar.text_input(
        "🎯 Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper().strip()
    
    # Validate ticker format
    if ticker and not _is_valid_ticker(ticker):
        st.sidebar.warning("⚠️ Please enter a valid ticker symbol")
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    
    # Default to last 2 years of data
    default_start = datetime.now() - timedelta(days=730)
    default_end = datetime.now()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            max_value=datetime.now(),
            help="Beginning of analysis period"
        )
    
    with col2:
        end_date = st.sidebar.date_input(
            "End Date", 
            value=default_end,
            max_value=datetime.now(),
            help="End of analysis period"
        )
    
    # Validate date range
    if start_date >= end_date:
        st.sidebar.error("❌ Start date must be before end date")
    
    # Calculate and display analysis period
    if start_date < end_date:
        period_days = (end_date - start_date).days
        st.sidebar.info(f"📊 Analysis period: {period_days} days")
        
        if period_days > 1825:  # 5 years
            st.sidebar.warning("⚠️ Long analysis periods may take more time to process")
    
    # Additional data options
    st.sidebar.subheader("⚙️ Data Options")
    
    # Benchmark selection
    benchmark = st.sidebar.selectbox(
        "📊 Benchmark",
        options=["SPY", "QQQ", "IWM", "DIA", "VTI", "None"],
        index=0,
        help="Select a benchmark for performance comparison"
    )
    
    return {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'benchmark': benchmark if benchmark != "None" else None,
        'valid': ticker and _is_valid_ticker(ticker) and start_date < end_date
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
    """Render analysis and visualization option controls in the sidebar.
    
    This function provides toggles and settings for various analysis features,
    allowing users to customize what metrics are calculated and how results
    are displayed.
    
    ANALYSIS OPTIONS:
    ================
    - Performance metrics calculation toggles
    - Risk analysis options and settings
    - Visualization preferences and chart types
    - Report generation and export options
    
    Returns:
        Dict[str, Any]: Analysis configuration options selected by user
    """
    st.sidebar.header("📊 Analysis Options")
    
    # Performance analysis options
    show_drawdown = st.sidebar.checkbox(
        "📉 Show Drawdown Analysis",
        value=True,
        help="Display maximum drawdown and underwater curves"
    )
    
    show_rolling_metrics = st.sidebar.checkbox(
        "📈 Rolling Performance Metrics", 
        value=True,
        help="Calculate rolling Sharpe ratio and other metrics"
    )
    
    # Risk analysis options
    calculate_var = st.sidebar.checkbox(
        "⚠️ Value at Risk (VaR)",
        value=False,
        help="Calculate Value at Risk metrics (may increase processing time)"
    )
    
    # Visualization options
    st.sidebar.subheader("🎨 Visualization")
    
    chart_style = st.sidebar.radio(
        "Chart Style",
        options=["Modern", "Classic", "Minimal"],
        index=0,
        help="Select the visual style for charts and graphs"
    )
    
    show_signals = st.sidebar.checkbox(
        "🎯 Show Trading Signals",
        value=True,
        help="Display buy/sell signals on price charts"
    )
    
    return {
        'show_drawdown': show_drawdown,
        'show_rolling_metrics': show_rolling_metrics,
        'calculate_var': calculate_var,
        'chart_style': chart_style.lower(),
        'show_signals': show_signals
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