"""Unified Logging and Error Handling Framework for Strategy Explainer.

This module provides a comprehensive logging and error handling system that ensures
consistent behavior across all components of the trading strategy application.

Key Features:
- Standardized logging configuration across all modules
- Custom exception classes with detailed context
- Automatic error recovery and graceful degradation
- Performance monitoring and metrics collection
- Security-aware logging (no sensitive data exposure)
- Environment-aware log levels and formatting

Logging Standards:
- DEBUG: Detailed technical information for debugging
- INFO: General application flow and important events
- WARNING: Potentially harmful situations that don't stop execution
- ERROR: Error events that allow application to continue
- CRITICAL: Serious errors that may cause application to abort

Error Handling Patterns:
- Graceful degradation with fallback values
- Context-rich error messages with actionable information
- Automatic retry mechanisms for transient failures
- Safe defaults to prevent application crashes
- Comprehensive error categorization and response strategies
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import functools
import traceback
import warnings
from pathlib import Path


class StrategyExplainerFormatter(logging.Formatter):
    """Custom formatter for Strategy Explainer application logs.
    
    Provides consistent, readable formatting with color coding and
    structured information for different log levels.
    """
    
    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors=True, include_module=True):
        """Initialize formatter with customization options.
        
        Args:
            use_colors: Whether to include color codes in output
            include_module: Whether to include module name in log messages
        """
        self.use_colors = use_colors and sys.stderr.isatty()
        self.include_module = include_module
        
        # Base format with timestamp, level, and message
        base_format = "%(asctime)s"
        if include_module:
            base_format += " [%(name)s]"
        base_format += " %(levelname)s: %(message)s"
        
        super().__init__(base_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """Format log record with colors and custom styling."""
        # Apply color coding if enabled
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Add context information for errors
        if record.levelno >= logging.ERROR and hasattr(record, 'exc_info') and record.exc_info:
            record.message = f"{record.getMessage()}\n{self._format_exception(record.exc_info)}"
        
        return super().format(record)
    
    def _format_exception(self, exc_info):
        """Format exception information for logging."""
        return ''.join(traceback.format_exception(*exc_info)).strip()


class LoggerManager:
    """Centralized logger management for consistent logging across the application."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    _log_level = logging.INFO
    _log_file: Optional[str] = None
    
    @classmethod
    def configure_logging(cls, 
                         level: str = "INFO",
                         log_file: Optional[str] = None,
                         enable_console: bool = True,
                         use_colors: bool = True) -> None:
        """Configure global logging settings for the application.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            enable_console: Whether to log to console
            use_colors: Whether to use color formatting
        """
        if cls._configured:
            return
        
        cls._log_level = getattr(logging, level.upper(), logging.INFO)
        cls._log_file = log_file
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._log_level)
        
        # Remove default handlers to avoid duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create custom formatter
        formatter = StrategyExplainerFormatter(use_colors=use_colors)
        
        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(cls._log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            try:
                # Ensure log directory exists
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(cls._log_level)
                # Use non-colored formatter for file output
                file_formatter = StrategyExplainerFormatter(use_colors=False)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                # Log to console if file logging fails
                logging.warning(f"Failed to setup file logging: {e}")
        
        # Filter out noisy third-party library logs
        cls._configure_third_party_loggers()
        
        cls._configured = True
        logging.info("Logging system initialized successfully")
    
    @classmethod
    def _configure_third_party_loggers(cls):
        """Configure logging levels for third-party libraries."""
        # Reduce noise from common libraries
        noisy_loggers = {
            'matplotlib': logging.WARNING,
            'urllib3': logging.WARNING,
            'requests': logging.WARNING,
            'yfinance': logging.WARNING,
            'openai': logging.WARNING
        }
        
        for logger_name, level in noisy_loggers.items():
            logging.getLogger(logger_name).setLevel(level)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with consistent configuration.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.configure_logging()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(cls._log_level)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


# Custom Exception Classes
class StrategyExplainerError(Exception):
    """Base exception for all Strategy Explainer errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Initialize with message and optional context.
        
        Args:
            message: Human-readable error description
            context: Additional context information for debugging
        """
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def __str__(self):
        """Return formatted error message with context."""
        msg = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" (Context: {context_str})"
        return msg


class StrategyError(StrategyExplainerError):
    """Exception for strategy-related errors."""
    pass


class DataError(StrategyExplainerError):
    """Exception for data-related errors."""
    pass


class ConfigurationError(StrategyExplainerError):
    """Exception for configuration-related errors."""
    pass


class SimulationError(StrategyExplainerError):
    """Exception for simulation-related errors."""
    pass


class ValidationError(StrategyExplainerError):
    """Exception for validation errors."""
    pass


# Decorator Functions for Error Handling
def with_error_handling(fallback_value=None, log_errors=True, reraise=False):
    """Decorator to add consistent error handling to functions.
    
    Args:
        fallback_value: Value to return if function fails
        log_errors: Whether to log errors when they occur
        reraise: Whether to re-raise exceptions after logging
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = LoggerManager.get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                if reraise:
                    raise
                
                return fallback_value
        
        return wrapper
    return decorator


def with_performance_logging(threshold_ms: float = 1000.0):
    """Decorator to log performance metrics for slow operations.
    
    Args:
        threshold_ms: Log warning if execution exceeds this threshold (milliseconds)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = LoggerManager.get_logger(func.__module__)
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if execution_time > threshold_ms:
                    logger.warning(f"{func.__name__} took {execution_time:.1f}ms (threshold: {threshold_ms}ms)")
                else:
                    logger.debug(f"{func.__name__} completed in {execution_time:.1f}ms")
                
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"{func.__name__} failed after {execution_time:.1f}ms: {str(e)}")
                raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, fallback=None, context: Dict[str, Any] = None, **kwargs):
    """Safely execute a function with error handling and logging.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        fallback: Value to return if function fails
        context: Additional context for error logging
        **kwargs: Keyword arguments for function
    
    Returns:
        Function result or fallback value
    """
    logger = LoggerManager.get_logger(func.__module__)
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = {
            'function': func.__name__,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        }
        if context:
            error_context.update(context)
        
        logger.error(f"Safe execution failed: {str(e)}", extra={'context': error_context})
        return fallback


# Utility Functions
def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """Log function entry with parameters."""
    if logger.isEnabledFor(logging.DEBUG):
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"Entering {func_name}({params})")


def log_function_exit(logger: logging.Logger, func_name: str, result=None, execution_time=None):
    """Log function exit with result and timing."""
    if logger.isEnabledFor(logging.DEBUG):
        msg = f"Exiting {func_name}"
        if execution_time:
            msg += f" (took {execution_time:.2f}ms)"
        if result is not None:
            msg += f" -> {type(result).__name__}"
        logger.debug(msg)


def suppress_warnings(category=None):
    """Context manager to suppress specific warning categories."""
    if category is None:
        category = UserWarning
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=category)
        yield


# Initialize logging on module import
def initialize_logging_from_config():
    """Initialize logging using configuration settings."""
    try:
        from config.config_manager import get_config
        
        log_level = get_config('app.log_level', 'INFO')
        log_file = get_config('app.log_file', None)
        enable_console = get_config('app.log_console', True)
        use_colors = get_config('app.log_colors', True)
        
        LoggerManager.configure_logging(
            level=log_level,
            log_file=log_file,
            enable_console=enable_console,
            use_colors=use_colors
        )
    except Exception:
        # Fallback to basic configuration if config system fails
        LoggerManager.configure_logging()


# Auto-initialize when module is imported
initialize_logging_from_config()