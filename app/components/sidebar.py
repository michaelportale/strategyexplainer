"""Sidebar configuration component for Streamlit app."""

import streamlit as st
from typing import Dict, Any, List
import yaml
from pathlib import Path


def load_strategy_configs() -> Dict[str, Any]:
    """Load strategy configurations from YAML file."""
    config_path = Path("config/strategies.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar with strategy and parameter selection."""
    st.sidebar.title("🎯 Strategy Explainer")
    
    # Load strategy configurations
    strategy_configs = load_strategy_configs()
    
    # Strategy Category Selection
    st.sidebar.header("Strategy Selection")
    
    categories = list(strategy_configs.keys()) if strategy_configs else ['momentum', 'mean_reversion', 'breakout']
    if 'global' in categories:
        categories.remove('global')
    
    selected_category = st.sidebar.selectbox(
        "Strategy Category",
        categories,
        index=0 if categories else None
    )
    
    # Strategy Type Selection
    if selected_category and strategy_configs.get(selected_category):
        strategy_types = list(strategy_configs[selected_category].keys())
        selected_strategy = st.sidebar.selectbox(
            "Strategy Type",
            strategy_types,
            index=0
        )
        
        # Get strategy parameters
        strategy_config = strategy_configs[selected_category][selected_strategy]
        strategy_params = strategy_config.get('parameters', {})
    else:
        selected_strategy = None
        strategy_params = {}
    
    # Data Selection
    st.sidebar.header("Data Configuration")
    
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)"
    )
    
    # Date range selection
    import datetime
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Parameter Configuration
    st.sidebar.header("Strategy Parameters")
    
    # Dynamic parameter inputs based on selected strategy
    params = {}
    if strategy_params:
        for param_name, default_value in strategy_params.items():
            if isinstance(default_value, float):
                params[param_name] = st.sidebar.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=0.0,
                    max_value=1.0 if param_name in ['position_size', 'stop_loss', 'take_profit', 'threshold'] else 5.0,
                    value=default_value,
                    step=0.01,
                    help=f"Default: {default_value}"
                )
            elif isinstance(default_value, int):
                params[param_name] = st.sidebar.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=1,
                    max_value=200 if 'period' in param_name else 100,
                    value=default_value,
                    help=f"Default: {default_value}"
                )
    else:
        # Default parameters if no config available
        params = {
            'lookback_period': st.sidebar.slider("Lookback Period", 5, 50, 20),
            'threshold': st.sidebar.slider("Threshold", 0.01, 0.10, 0.02, 0.01),
            'position_size': st.sidebar.slider("Position Size", 0.05, 0.50, 0.10, 0.05),
            'stop_loss': st.sidebar.slider("Stop Loss", 0.01, 0.20, 0.05, 0.01),
            'take_profit': st.sidebar.slider("Take Profit", 0.02, 0.50, 0.10, 0.01),
        }
    
    # Advanced Options
    with st.sidebar.expander("Advanced Options"):
        commission = st.number_input("Commission Rate", 0.0, 0.01, 0.001, 0.0001, format="%.4f")
        slippage = st.number_input("Slippage Rate", 0.0, 0.01, 0.0005, 0.0001, format="%.4f")
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)
    
    # Analysis Options
    st.sidebar.header("Analysis Options")
    
    generate_report = st.sidebar.checkbox("Generate GPT Analysis", value=True)
    save_results = st.sidebar.checkbox("Save Results", value=True)
    
    # Return configuration dictionary
    return {
        'strategy': {
            'category': selected_category,
            'type': selected_strategy,
            'name': strategy_config.get('name', selected_strategy) if selected_strategy else None,
            'description': strategy_config.get('description', '') if selected_strategy else '',
            'parameters': params
        },
        'data': {
            'symbol': symbol.upper(),
            'start_date': date_range[0] if len(date_range) == 2 else start_date,
            'end_date': date_range[1] if len(date_range) == 2 else end_date,
        },
        'trading': {
            'commission': commission,
            'slippage': slippage,
            'benchmark': benchmark,
        },
        'analysis': {
            'generate_report': generate_report,
            'save_results': save_results,
        }
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