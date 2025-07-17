"""Enhanced parameter management system for trading strategies.

This module provides a comprehensive parameter management framework that centralizes
parameter definitions, validation, and introspection capabilities. It ensures
consistency between code and configuration while providing rich metadata for UIs.

Key Features:
- Type-safe parameter definitions with validation
- Automatic parameter documentation generation
- Range validation and constraint checking
- Default value management
- UI-friendly parameter introspection
- Configuration file synchronization

Classes:
    ParameterDefinition: Single parameter specification with validation
    ParameterSchema: Complete parameter schema for a strategy
    ParameterManager: Centralized parameter management utilities

Example:
    >>> schema = ParameterSchema({
    ...     'period': ParameterDefinition(
    ...         default=14,
    ...         type=int,
    ...         range=(1, 200),
    ...         description="Moving average period"
    ...     )
    ... })
    >>> params = schema.validate({'period': 20})
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Type, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum


class ParameterType(Enum):
    """Supported parameter types for strategy configuration."""
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    STRING = "str"
    LIST = "list"
    ENUM = "enum"


@dataclass
class ParameterDefinition:
    """Definition of a single strategy parameter with validation rules.
    
    This class encapsulates all metadata and validation rules for a single
    parameter, providing a rich interface for both code validation and UI
    generation.
    
    Attributes:
        default: Default value for the parameter
        type: Expected parameter type (int, float, bool, str, list)
        range: Valid range for numeric parameters (min, max)
        choices: Valid choices for enum-like parameters
        description: Human-readable parameter description
        display_name: UI-friendly parameter name
        category: Parameter grouping for UI organization
        required: Whether parameter is required (cannot be None)
        validator: Custom validation function
        ui_widget: Suggested UI widget type for frontends
        tooltip: Extended help text for UI tooltips
        
    Example:
        >>> param = ParameterDefinition(
        ...     default=14,
        ...     type=int,
        ...     range=(1, 200),
        ...     description="RSI calculation period",
        ...     category="Technical Indicators"
        ... )
    """
    default: Any
    type: Type = field(default=int)
    range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    display_name: Optional[str] = None
    category: str = "General"
    required: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    ui_widget: str = "auto"  # auto, slider, dropdown, checkbox, text
    tooltip: str = ""
    
    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        if self.display_name is None and hasattr(self, 'name'):
            # Convert snake_case to Title Case
            self.display_name = self.name.replace('_', ' ').title()
        elif self.display_name is None:
            self.display_name = "Parameter"
    
    def validate(self, value: Any, name: str = "parameter") -> Any:
        """Validate a parameter value against this definition.
        
        Performs comprehensive validation including type checking, range
        validation, choice validation, and custom validation functions.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            
        Returns:
            Validated and potentially converted value
            
        Raises:
            ValueError: If validation fails
            TypeError: If type conversion fails
        """
        # Handle None values
        if value is None:
            if self.required:
                raise ValueError(f"Parameter '{name}' is required but got None")
            return self.default
        
        # Type validation and conversion
        try:
            if self.type == bool:
                # Handle string boolean conversion
                if isinstance(value, str):
                    if value.lower() in ('true', '1', 'yes', 'on'):
                        value = True
                    elif value.lower() in ('false', '0', 'no', 'off'):
                        value = False
                    else:
                        raise ValueError(f"Cannot convert '{value}' to boolean")
                else:
                    value = bool(value)
            elif self.type in (int, float):
                value = self.type(value)
            elif self.type == str:
                value = str(value)
            elif self.type == list:
                if not isinstance(value, list):
                    raise TypeError(f"Expected list, got {type(value).__name__}")
        except (ValueError, TypeError) as e:
            raise TypeError(f"Parameter '{name}' type error: {e}")
        
        # Range validation for numeric types
        if self.range is not None and self.type in (int, float):
            min_val, max_val = self.range
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter '{name}' value {value} outside valid range [{min_val}, {max_val}]"
                )
        
        # Choice validation
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(
                    f"Parameter '{name}' value '{value}' not in valid choices: {self.choices}"
                )
        
        # Custom validation
        if self.validator is not None:
            try:
                if not self.validator(value):
                    raise ValueError(f"Parameter '{name}' failed custom validation")
            except Exception as e:
                raise ValueError(f"Parameter '{name}' validation error: {e}")
        
        return value
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Generate UI configuration for this parameter.
        
        Returns a dictionary containing all information needed by UI frameworks
        to render an appropriate input widget for this parameter.
        
        Returns:
            Dictionary with UI configuration including widget type, constraints,
            display information, and validation rules.
        """
        config = {
            'type': self.type.__name__,
            'default': self.default,
            'display_name': self.display_name,
            'description': self.description,
            'tooltip': self.tooltip or self.description,
            'category': self.category,
            'required': self.required,
            'widget': self._determine_widget_type()
        }
        
        # Add type-specific configurations
        if self.range is not None:
            config['min'] = self.range[0]
            config['max'] = self.range[1]
            
        if self.choices is not None:
            config['choices'] = self.choices
            
        return config
    
    def _determine_widget_type(self) -> str:
        """Automatically determine the best UI widget type."""
        if self.ui_widget != "auto":
            return self.ui_widget
            
        # Auto-determine widget based on parameter characteristics
        if self.type == bool:
            return "checkbox"
        elif self.choices is not None:
            return "dropdown"
        elif self.type in (int, float) and self.range is not None:
            return "slider"
        elif self.type in (int, float):
            return "number"
        else:
            return "text"


class ParameterSchema:
    """Complete parameter schema for a trading strategy.
    
    This class manages the complete set of parameters for a strategy,
    providing validation, default value management, and metadata extraction
    capabilities.
    
    Attributes:
        parameters: Dictionary mapping parameter names to definitions
        strategy_name: Name of the strategy this schema belongs to
        description: Overall strategy description
        version: Schema version for compatibility tracking
    """
    
    def __init__(self, 
                 parameters: Dict[str, ParameterDefinition],
                 strategy_name: str = "",
                 description: str = "",
                 version: str = "1.0"):
        """Initialize parameter schema.
        
        Args:
            parameters: Dictionary of parameter name to ParameterDefinition
            strategy_name: Name of the associated strategy
            description: Strategy description for documentation
            version: Schema version for compatibility
        """
        self.parameters = parameters
        self.strategy_name = strategy_name
        self.description = description
        self.version = version
        
        # Set parameter names in definitions for validation messages
        for name, param_def in self.parameters.items():
            param_def.name = name
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters.
        
        Returns:
            Dictionary mapping parameter names to their default values
        """
        return {name: param_def.default for name, param_def in self.parameters.items()}
    
    def validate(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize user-provided parameters.
        
        Takes user parameters, validates them against the schema, fills in
        defaults for missing parameters, and returns a complete, validated
        parameter dictionary.
        
        Args:
            user_params: User-provided parameter values
            
        Returns:
            Complete, validated parameter dictionary
            
        Raises:
            ValueError: If any parameter validation fails
        """
        validated = {}
        
        # Start with defaults
        validated.update(self.get_defaults())
        
        # Validate and update with user parameters
        for name, value in user_params.items():
            if name not in self.parameters:
                logging.warning(f"Unknown parameter '{name}' for strategy '{self.strategy_name}'")
                continue
                
            param_def = self.parameters[name]
            validated[name] = param_def.validate(value, name)
        
        # Check for missing required parameters
        for name, param_def in self.parameters.items():
            if param_def.required and name not in user_params:
                raise ValueError(f"Required parameter '{name}' missing for strategy '{self.strategy_name}'")
        
        return validated
    
    def get_ui_schema(self) -> Dict[str, Any]:
        """Generate complete UI schema for frontend frameworks.
        
        Returns a comprehensive schema that UI frameworks can use to
        automatically generate parameter input forms.
        
        Returns:
            Dictionary containing complete UI schema with parameter groups,
            validation rules, and display metadata.
        """
        # Group parameters by category
        categories = {}
        for name, param_def in self.parameters.items():
            category = param_def.category
            if category not in categories:
                categories[category] = {}
            categories[category][name] = param_def.get_ui_config()
        
        return {
            'strategy_name': self.strategy_name,
            'description': self.description,
            'version': self.version,
            'categories': categories,
            'parameter_count': len(self.parameters),
            'required_parameters': [
                name for name, param_def in self.parameters.items() 
                if param_def.required
            ]
        }
    
    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter information dictionary or None if not found
        """
        if name not in self.parameters:
            return None
            
        param_def = self.parameters[name]
        return {
            'name': name,
            'type': param_def.type.__name__,
            'default': param_def.default,
            'description': param_def.description,
            'range': param_def.range,
            'choices': param_def.choices,
            'required': param_def.required,
            'category': param_def.category
        }
    
    def validate_ranges(self) -> List[str]:
        """Validate parameter schema for consistency issues.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        for name, param_def in self.parameters.items():
            # Check range consistency
            if param_def.range is not None:
                min_val, max_val = param_def.range
                if min_val >= max_val:
                    issues.append(f"Parameter '{name}': invalid range [{min_val}, {max_val}]")
                
                # Check if default is in range
                if param_def.type in (int, float):
                    if not (min_val <= param_def.default <= max_val):
                        issues.append(f"Parameter '{name}': default {param_def.default} outside range")
            
            # Check choices consistency
            if param_def.choices is not None:
                if param_def.default not in param_def.choices:
                    issues.append(f"Parameter '{name}': default not in choices")
        
        return issues


class ParameterManager:
    """Centralized parameter management utilities.
    
    This class provides utility functions for working with parameter schemas,
    including configuration file synchronization, parameter comparison, and
    schema merging capabilities.
    """
    
    @staticmethod
    def merge_schemas(*schemas: ParameterSchema) -> ParameterSchema:
        """Merge multiple parameter schemas into one.
        
        Useful for combining base strategy parameters with overlay parameters.
        
        Args:
            *schemas: Variable number of ParameterSchema instances
            
        Returns:
            Merged ParameterSchema
            
        Raises:
            ValueError: If parameter names conflict between schemas
        """
        merged_params = {}
        merged_name = "Merged Strategy"
        merged_description = "Combined parameter schema"
        
        for schema in schemas:
            for name, param_def in schema.parameters.items():
                if name in merged_params:
                    raise ValueError(f"Parameter name conflict: '{name}' appears in multiple schemas")
                merged_params[name] = param_def
        
        return ParameterSchema(
            parameters=merged_params,
            strategy_name=merged_name,
            description=merged_description
        )
    
    @staticmethod
    def compare_with_config(schema: ParameterSchema, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare schema defaults with configuration file values.
        
        Identifies mismatches between code-defined defaults and configuration
        file values to help maintain consistency.
        
        Args:
            schema: Parameter schema from code
            config_params: Parameters from configuration file
            
        Returns:
            Dictionary containing comparison results and recommendations
        """
        results = {
            'mismatched_defaults': {},
            'missing_in_config': [],
            'extra_in_config': [],
            'recommendations': []
        }
        
        schema_defaults = schema.get_defaults()
        
        # Find mismatched defaults
        for name, schema_default in schema_defaults.items():
            if name in config_params and config_params[name] != schema_default:
                results['mismatched_defaults'][name] = {
                    'schema_default': schema_default,
                    'config_value': config_params[name]
                }
        
        # Find missing parameters
        for name in schema_defaults:
            if name not in config_params:
                results['missing_in_config'].append(name)
        
        # Find extra parameters
        for name in config_params:
            if name not in schema_defaults:
                results['extra_in_config'].append(name)
        
        # Generate recommendations
        if results['mismatched_defaults']:
            results['recommendations'].append(
                "Consider updating config file or code defaults to match"
            )
        if results['missing_in_config']:
            results['recommendations'].append(
                "Add missing parameters to config file"
            )
        if results['extra_in_config']:
            results['recommendations'].append(
                "Remove unused parameters from config file or add to schema"
            )
        
        return results