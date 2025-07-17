"""Enhanced base strategy class with improved parameter management and metadata interface.

This module extends the original BaseStrategy with a sophisticated parameter management
system, uniform metadata interface, and enhanced introspection capabilities. It provides
the foundation for a more robust and maintainable strategy architecture.

Key Enhancements:
- Centralized parameter definitions with validation
- Rich metadata interface for UI integration
- Type-safe parameter handling
- Automatic documentation generation
- Configuration synchronization
- Enhanced error handling and logging

Classes:
    EnhancedBaseStrategy: Enhanced base class with advanced parameter management
    StrategyDecorator: Abstract base for composable strategy overlays
    ParameterizedStrategy: Mixin for parameter schema integration

Example:
    >>> class MyStrategy(EnhancedBaseStrategy):
    ...     PARAMETER_SCHEMA = ParameterSchema({
    ...         'period': ParameterDefinition(default=14, type=int, range=(1, 100))
    ...     })
    ...     
    ...     def generate_signals(self, data):
    ...         period = self.get_parameter('period')
    ...         # Strategy logic here
    ...         return data
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union, ClassVar
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime

# Import existing base classes and utilities
from .base import StrategyRegistry, StrategyMeta, BaseStrategy
from .parameters import ParameterDefinition, ParameterSchema, ParameterManager

# Import logging utilities with absolute path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_config import (
    LoggerManager, 
    StrategyError, 
    DataError, 
    ValidationError,
    with_error_handling,
    with_performance_logging
)


class EnhancedBaseStrategy(BaseStrategy):
    """Enhanced base strategy class with advanced parameter management.
    
    This class extends the original BaseStrategy with a sophisticated parameter
    management system that provides type safety, validation, and rich metadata
    for UI integration. It serves as the foundation for all modern strategy
    implementations.
    
    Key Features:
    - Declarative parameter schemas with validation
    - Automatic parameter documentation generation
    - Type-safe parameter access
    - Configuration file synchronization
    - Rich metadata interface for UIs
    - Enhanced error handling and logging
    
    Class Attributes:
        PARAMETER_SCHEMA: ParameterSchema defining strategy parameters
        STRATEGY_DESCRIPTION: Human-readable strategy description
        STRATEGY_VERSION: Strategy version for compatibility tracking
        TAGS: List of tags for strategy categorization
        
    Example:
        >>> class RSIStrategy(EnhancedBaseStrategy):
        ...     PARAMETER_SCHEMA = ParameterSchema({
        ...         'rsi_period': ParameterDefinition(
        ...             default=14, type=int, range=(2, 50),
        ...             description="RSI calculation period"
        ...         ),
        ...         'oversold': ParameterDefinition(
        ...             default=30, type=float, range=(10, 40),
        ...             description="Oversold threshold"
        ...         )
        ...     }, strategy_name="RSI Strategy")
        ...     
        ...     STRATEGY_DESCRIPTION = "RSI-based momentum strategy"
        ...     TAGS = ["momentum", "oscillator"]
    """
    
    # Class-level parameter schema (to be overridden in subclasses)
    PARAMETER_SCHEMA: Optional[ParameterSchema] = None
    STRATEGY_DESCRIPTION: str = ""
    STRATEGY_VERSION: str = "1.0"
    TAGS: List[str] = []
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        """Initialize enhanced strategy with parameter validation.
        
        Args:
            parameters: User-provided parameter values
            name: Optional custom strategy name (defaults to class-derived name)
        """
        # Determine strategy name
        if name is None:
            name = self._get_default_name()
        
        # Initialize parameter schema if not defined
        if self.PARAMETER_SCHEMA is None:
            self.PARAMETER_SCHEMA = self._create_default_schema()
        
        # Validate and set parameters
        user_params = parameters or {}
        self.validated_parameters = self.PARAMETER_SCHEMA.validate(user_params)
        
        # Initialize base strategy with validated parameters
        super().__init__(name, self.validated_parameters)
        
        # Store original user parameters for reference
        self.user_parameters = user_params.copy()
        
        # Enhanced logging with parameter info
        param_count = len(self.validated_parameters)
        self.logger.info(f"Initialized enhanced strategy '{name}' with {param_count} parameters")
        self.logger.debug(f"Validated parameters: {self.validated_parameters}")
        
        # Log any parameter warnings
        self._log_parameter_warnings()
    
    def _get_default_name(self) -> str:
        """Generate default strategy name from class name."""
        class_name = self.__class__.__name__
        # Remove 'Strategy' suffix if present
        if class_name.endswith('Strategy'):
            class_name = class_name[:-8]
        
        # Convert CamelCase to Title Case
        name = ''.join([' ' + c if c.isupper() and i > 0 else c for i, c in enumerate(class_name)])
        return name.strip()
    
    def _create_default_schema(self) -> ParameterSchema:
        """Create a basic parameter schema if none is defined."""
        return ParameterSchema(
            parameters={},
            strategy_name=self._get_default_name(),
            description=self.STRATEGY_DESCRIPTION or "Strategy with no parameter schema defined"
        )
    
    def _log_parameter_warnings(self) -> None:
        """Log warnings for common parameter issues."""
        if not self.PARAMETER_SCHEMA.parameters:
            self.logger.warning("Strategy has no parameter schema defined")
        
        # Check for unused user parameters
        schema_params = set(self.PARAMETER_SCHEMA.parameters.keys())
        user_params = set(self.user_parameters.keys())
        unused_params = user_params - schema_params
        
        if unused_params:
            self.logger.warning(f"Unused parameters provided: {unused_params}")
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a validated parameter value with type safety.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value with proper type
            
        Raises:
            KeyError: If parameter not found and no default provided
        """
        if name in self.validated_parameters:
            return self.validated_parameters[name]
        elif default is not None:
            return default
        else:
            raise KeyError(f"Parameter '{name}' not found in strategy '{self.name}'")
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value with validation.
        
        Args:
            name: Parameter name
            value: New parameter value
            
        Raises:
            ValueError: If parameter validation fails
        """
        if name not in self.PARAMETER_SCHEMA.parameters:
            self.logger.warning(f"Setting unknown parameter '{name}'")
            self.validated_parameters[name] = value
        else:
            param_def = self.PARAMETER_SCHEMA.parameters[name]
            validated_value = param_def.validate(value, name)
            self.validated_parameters[name] = validated_value
            
        self.logger.debug(f"Parameter '{name}' set to {value}")
    
    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter information dictionary or None if not found
        """
        return self.PARAMETER_SCHEMA.get_parameter_info(name)
    
    def list_parameters(self) -> List[str]:
        """List all available parameter names."""
        return list(self.PARAMETER_SCHEMA.parameters.keys())
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return self.PARAMETER_SCHEMA.get_defaults()
    
    @classmethod
    def get_schema(cls) -> ParameterSchema:
        """Get the parameter schema for this strategy class.
        
        Returns:
            ParameterSchema instance for this strategy
        """
        if cls.PARAMETER_SCHEMA is None:
            # Create a basic schema if none defined
            return ParameterSchema(
                parameters={},
                strategy_name=cls.__name__,
                description=cls.STRATEGY_DESCRIPTION
            )
        return cls.PARAMETER_SCHEMA
    
    @classmethod
    def get_ui_schema(cls) -> Dict[str, Any]:
        """Get UI schema for frontend parameter forms.
        
        Returns:
            Dictionary containing complete UI schema
        """
        schema = cls.get_schema()
        ui_schema = schema.get_ui_schema()
        
        # Add class-level metadata
        ui_schema.update({
            'class_name': cls.__name__,
            'description': cls.STRATEGY_DESCRIPTION,
            'version': cls.STRATEGY_VERSION,
            'tags': cls.TAGS,
            'category': getattr(cls, 'strategy_category', 'general')
        })
        
        return ui_schema
    
    @classmethod
    def create_with_defaults(cls, **overrides) -> 'EnhancedBaseStrategy':
        """Create strategy instance with default parameters plus overrides.
        
        Args:
            **overrides: Parameter overrides
            
        Returns:
            Strategy instance with validated parameters
        """
        schema = cls.get_schema()
        defaults = schema.get_defaults()
        defaults.update(overrides)
        
        return cls(parameters=defaults)
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information.
        
        Extends the base get_info() method with enhanced metadata including
        parameter schema information, validation status, and configuration.
        
        Returns:
            Dictionary containing complete strategy information
        """
        base_info = super().get_info()
        
        enhanced_info = {
            **base_info,
            'description': self.STRATEGY_DESCRIPTION,
            'version': self.STRATEGY_VERSION,
            'tags': self.TAGS,
            'parameter_schema': {
                'total_parameters': len(self.PARAMETER_SCHEMA.parameters),
                'required_parameters': [
                    name for name, param_def in self.PARAMETER_SCHEMA.parameters.items()
                    if param_def.required
                ],
                'categories': list(set(
                    param_def.category for param_def in self.PARAMETER_SCHEMA.parameters.values()
                ))
            },
            'validated_parameters': self.validated_parameters,
            'user_parameters': self.user_parameters,
            'parameter_validation_status': 'valid',
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'has_schema': self.PARAMETER_SCHEMA is not None,
                'schema_version': self.PARAMETER_SCHEMA.version if self.PARAMETER_SCHEMA else None,
                'is_enhanced': True
            }
        }
        
        return enhanced_info
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return diagnostic information.
        
        Returns:
            Dictionary containing validation results and recommendations
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Validate parameter schema
        if self.PARAMETER_SCHEMA:
            schema_issues = self.PARAMETER_SCHEMA.validate_ranges()
            if schema_issues:
                results['warnings'].extend(schema_issues)
        
        # Check for missing required parameters
        if self.PARAMETER_SCHEMA:
            for name, param_def in self.PARAMETER_SCHEMA.parameters.items():
                if param_def.required and name not in self.validated_parameters:
                    results['errors'].append(f"Required parameter '{name}' is missing")
                    results['is_valid'] = False
        
        # Check for parameter type consistency
        for name, value in self.validated_parameters.items():
            if name in self.PARAMETER_SCHEMA.parameters:
                param_def = self.PARAMETER_SCHEMA.parameters[name]
                if not isinstance(value, param_def.type):
                    results['warnings'].append(
                        f"Parameter '{name}' type mismatch: expected {param_def.type.__name__}, got {type(value).__name__}"
                    )
        
        # Generate recommendations
        if not self.PARAMETER_SCHEMA.parameters:
            results['recommendations'].append("Consider defining a parameter schema for better validation")
        
        if self.user_parameters != self.validated_parameters:
            results['recommendations'].append("Some parameters were modified during validation")
        
        return results
    
    def clone(self, **parameter_overrides) -> 'EnhancedBaseStrategy':
        """Create a copy of this strategy with optional parameter changes.
        
        Args:
            **parameter_overrides: Parameters to override in the clone
            
        Returns:
            New strategy instance with modified parameters
        """
        new_params = self.validated_parameters.copy()
        new_params.update(parameter_overrides)
        
        return self.__class__(parameters=new_params, name=f"{self.name} (Clone)")
    
    def compare_parameters(self, other: 'EnhancedBaseStrategy') -> Dict[str, Any]:
        """Compare parameters with another strategy instance.
        
        Args:
            other: Other strategy instance to compare with
            
        Returns:
            Dictionary containing parameter comparison results
        """
        comparison = {
            'same_class': self.__class__ == other.__class__,
            'same_parameters': self.validated_parameters == other.validated_parameters,
            'parameter_differences': {},
            'unique_to_self': {},
            'unique_to_other': {}
        }
        
        self_params = set(self.validated_parameters.keys())
        other_params = set(other.validated_parameters.keys())
        
        # Find common parameters with different values
        common_params = self_params & other_params
        for param in common_params:
            if self.validated_parameters[param] != other.validated_parameters[param]:
                comparison['parameter_differences'][param] = {
                    'self': self.validated_parameters[param],
                    'other': other.validated_parameters[param]
                }
        
        # Find unique parameters
        unique_self = self_params - other_params
        unique_other = other_params - self_params
        
        if unique_self:
            comparison['unique_to_self'] = {
                param: self.validated_parameters[param] for param in unique_self
            }
        
        if unique_other:
            comparison['unique_to_other'] = {
                param: other.validated_parameters[param] for param in unique_other
            }
        
        return comparison
    
    def __repr__(self) -> str:
        """Enhanced string representation with parameter information."""
        param_summary = f"{len(self.validated_parameters)} params"
        schema_info = f"schema={self.PARAMETER_SCHEMA.version}" if self.PARAMETER_SCHEMA else "no schema"
        return f"{self.__class__.__name__}(name='{self.name}', {param_summary}, {schema_info})"


class StrategyDecorator(ABC):
    """Abstract base class for composable strategy overlays.
    
    This class implements the Decorator pattern for strategies, allowing
    functionality to be added to strategies without modifying their core
    implementation. Examples include regime filters, position sizing overlays,
    and risk management layers.
    
    Key Principles:
    - Wraps an existing strategy instance
    - Can modify signals, add filters, or enhance functionality
    - Maintains the same interface as the base strategy
    - Can be chained together for complex behaviors
    
    Example:
        >>> base_strategy = RSIStrategy()
        >>> regime_filtered = RegimeFilterDecorator(base_strategy, regime_detector)
        >>> risk_managed = RiskManagementDecorator(regime_filtered, risk_params)
    """
    
    def __init__(self, wrapped_strategy: EnhancedBaseStrategy):
        """Initialize decorator with a strategy to wrap.
        
        Args:
            wrapped_strategy: Strategy instance to wrap and enhance
        """
        self.wrapped_strategy = wrapped_strategy
        self.logger = LoggerManager.get_logger(f"decorator.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals by enhancing the wrapped strategy's signals.
        
        This method should call the wrapped strategy's generate_signals method
        and then apply additional processing, filtering, or enhancement.
        
        Args:
            data: Market data for signal generation
            
        Returns:
            Enhanced signal data
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the decorated strategy."""
        wrapped_info = self.wrapped_strategy.get_info()
        
        decorator_info = {
            'decorator_class': self.__class__.__name__,
            'wrapped_strategy': wrapped_info,
            'is_decorated': True,
            'decoration_chain': self._get_decoration_chain()
        }
        
        return decorator_info
    
    def _get_decoration_chain(self) -> List[str]:
        """Get the chain of decorators applied to the base strategy."""
        chain = [self.__class__.__name__]
        
        current = self.wrapped_strategy
        while isinstance(current, StrategyDecorator):
            chain.append(current.__class__.__name__)
            current = current.wrapped_strategy
        
        chain.append(current.__class__.__name__)
        return chain
    
    def get_base_strategy(self) -> EnhancedBaseStrategy:
        """Get the base strategy at the bottom of the decoration chain."""
        current = self.wrapped_strategy
        while isinstance(current, StrategyDecorator):
            current = current.wrapped_strategy
        return current
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped strategy."""
        return getattr(self.wrapped_strategy, name)
    
    def __repr__(self) -> str:
        """String representation showing decoration chain."""
        chain = " -> ".join(self._get_decoration_chain())
        return f"DecoratedStrategy({chain})"


@dataclass
class StrategyMetadata:
    """Rich metadata container for strategy information.
    
    This class provides a structured way to store and access comprehensive
    strategy metadata that can be used for documentation generation,
    UI display, and strategy comparison.
    """
    name: str
    description: str
    version: str
    category: str
    tags: List[str]
    author: str = ""
    created_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    risk_profile: str = "medium"  # low, medium, high
    time_horizon: str = "medium"  # short, medium, long
    market_conditions: List[str] = None  # trending, ranging, volatile, etc.
    asset_classes: List[str] = None  # stocks, forex, crypto, etc.
    complexity: str = "medium"  # simple, medium, complex
    documentation_url: str = ""
    research_papers: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.market_conditions is None:
            self.market_conditions = []
        if self.asset_classes is None:
            self.asset_classes = []
        if self.research_papers is None:
            self.research_papers = []
        if self.created_date is None:
            self.created_date = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'category': self.category,
            'tags': self.tags,
            'author': self.author,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'performance_metrics': self.performance_metrics,
            'risk_profile': self.risk_profile,
            'time_horizon': self.time_horizon,
            'market_conditions': self.market_conditions,
            'asset_classes': self.asset_classes,
            'complexity': self.complexity,
            'documentation_url': self.documentation_url,
            'research_papers': self.research_papers
        }


class ParameterizedStrategy:
    """Mixin class for enhanced parameter management integration.
    
    This mixin provides additional utilities for strategies that use the
    enhanced parameter system. It can be combined with strategy classes
    to add extra parameter management capabilities.
    """
    
    def export_parameters(self, format: str = "dict") -> Union[Dict[str, Any], str]:
        """Export parameters in various formats.
        
        Args:
            format: Export format ('dict', 'json', 'yaml')
            
        Returns:
            Parameters in requested format
        """
        if format == "dict":
            return self.validated_parameters.copy()
        elif format == "json":
            import json
            return json.dumps(self.validated_parameters, indent=2)
        elif format == "yaml":
            import yaml
            return yaml.dump(self.validated_parameters, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_parameters(self, data: Union[Dict[str, Any], str], format: str = "dict") -> None:
        """Import parameters from various formats.
        
        Args:
            data: Parameter data to import
            format: Data format ('dict', 'json', 'yaml')
        """
        if format == "dict":
            params = data
        elif format == "json":
            import json
            params = json.loads(data)
        elif format == "yaml":
            import yaml
            params = yaml.safe_load(data)
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        # Validate and update parameters
        validated = self.PARAMETER_SCHEMA.validate(params)
        self.validated_parameters.update(validated)
        
        self.logger.info(f"Imported {len(params)} parameters from {format} format")
    
    def parameter_sensitivity_analysis(self, param_name: str, values: List[Any]) -> Dict[str, Any]:
        """Analyze sensitivity to parameter changes.
        
        Args:
            param_name: Parameter to analyze
            values: List of values to test
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if param_name not in self.PARAMETER_SCHEMA.parameters:
            raise ValueError(f"Parameter '{param_name}' not found in schema")
        
        original_value = self.get_parameter(param_name)
        results = {
            'parameter': param_name,
            'original_value': original_value,
            'test_values': values,
            'validation_results': []
        }
        
        for value in values:
            try:
                param_def = self.PARAMETER_SCHEMA.parameters[param_name]
                validated_value = param_def.validate(value, param_name)
                results['validation_results'].append({
                    'value': value,
                    'validated_value': validated_value,
                    'valid': True,
                    'error': None
                })
            except (ValueError, TypeError) as e:
                results['validation_results'].append({
                    'value': value,
                    'validated_value': None,
                    'valid': False,
                    'error': str(e)
                })
        
        # Restore original value
        self.set_parameter(param_name, original_value)
        
        return results