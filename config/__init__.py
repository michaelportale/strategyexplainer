"""Configuration package for strategy explainer application."""

from .settings import Settings
from .config_manager import ConfigManager, get_config_manager, get_config, get_settings

# Export the main configuration interface
__all__ = ['Settings', 'ConfigManager', 'get_config_manager', 'get_config', 'get_settings'] 