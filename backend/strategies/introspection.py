"""Strategy introspection utilities for UI integration and documentation generation.

This module provides comprehensive introspection capabilities for the enhanced
strategy framework, enabling automatic documentation generation, UI form creation,
and configuration management.

Key Features:
- Automatic parameter documentation generation
- UI schema generation for frontend frameworks
- Configuration file synchronization utilities
- Strategy comparison and analysis tools
- Parameter validation and recommendation systems

Classes:
    StrategyIntrospector: Main introspection interface
    UISchemaGenerator: UI-specific schema generation
    ConfigurationManager: Configuration file management
    DocumentationGenerator: Automatic documentation creation

Example:
    >>> introspector = StrategyIntrospector()
    >>> schema = introspector.get_strategy_ui_schema(MacdCrossoverStrategy)
    >>> docs = introspector.generate_strategy_documentation(MacdCrossoverStrategy)
"""

import inspect
import json
import yaml
from typing import Dict, Any, List, Optional, Type, Union, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

from .enhanced_base import EnhancedBaseStrategy, StrategyDecorator
from .parameters import ParameterSchema, ParameterDefinition, ParameterManager
from .base import StrategyRegistry

# Import logging utilities with absolute path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_config import LoggerManager


class StrategyIntrospector:
    """Main introspection interface for strategy analysis and metadata extraction.
    
    This class provides comprehensive introspection capabilities for strategies,
    including parameter analysis, documentation generation, and UI integration
    support.
    
    Features:
    - Strategy metadata extraction
    - Parameter schema analysis
    - UI form generation
    - Documentation creation
    - Configuration validation
    - Strategy comparison utilities
    """
    
    def __init__(self):
        self.logger = LoggerManager.get_logger("introspection.StrategyIntrospector")
        self.registry = StrategyRegistry.get_instance()
    
    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all registered strategies.
        
        Returns:
            Dictionary mapping strategy names to their complete metadata
        """
        all_strategies = {}
        
        for strategy_name in self.registry.list_strategies():
            try:
                strategy_class = self.registry.get_strategy_class(strategy_name)
                all_strategies[strategy_name] = self.get_strategy_info(strategy_class)
            except Exception as e:
                self.logger.error(f"Error getting info for strategy '{strategy_name}': {e}")
                all_strategies[strategy_name] = {
                    'error': str(e),
                    'available': False
                }
        
        return all_strategies
    
    def get_strategy_info(self, strategy_class: Type[EnhancedBaseStrategy]) -> Dict[str, Any]:
        """Get comprehensive information about a specific strategy class.
        
        Args:
            strategy_class: Strategy class to analyze
            
        Returns:
            Dictionary containing complete strategy metadata
        """
        info = {
            'class_name': strategy_class.__name__,
            'module': strategy_class.__module__,
            'category': getattr(strategy_class, 'strategy_category', 'general'),
            'description': getattr(strategy_class, 'STRATEGY_DESCRIPTION', ''),
            'version': getattr(strategy_class, 'STRATEGY_VERSION', '1.0'),
            'tags': getattr(strategy_class, 'TAGS', []),
            'is_enhanced': issubclass(strategy_class, EnhancedBaseStrategy),
            'is_decorator': issubclass(strategy_class, StrategyDecorator),
            'docstring': inspect.getdoc(strategy_class) or '',
        }
        
        # Parameter schema information
        if hasattr(strategy_class, 'get_schema'):
            try:
                schema = strategy_class.get_schema()
                info['parameter_schema'] = {
                    'total_parameters': len(schema.parameters),
                    'required_parameters': [
                        name for name, param_def in schema.parameters.items()
                        if param_def.required
                    ],
                    'parameter_categories': list(set(
                        param_def.category for param_def in schema.parameters.values()
                    )),
                    'schema_version': schema.version,
                    'schema_description': schema.description
                }
                
                # Detailed parameter information
                info['parameters'] = {}
                for name, param_def in schema.parameters.items():
                    info['parameters'][name] = {
                        'type': param_def.type.__name__,
                        'default': param_def.default,
                        'description': param_def.description,
                        'category': param_def.category,
                        'required': param_def.required,
                        'range': param_def.range,
                        'choices': param_def.choices,
                        'ui_widget': param_def._determine_widget_type()
                    }
            except Exception as e:
                self.logger.warning(f"Error extracting parameter schema for {strategy_class.__name__}: {e}")
                info['parameter_schema'] = {'error': str(e)}
        
        # Method analysis
        info['methods'] = self._analyze_class_methods(strategy_class)
        
        # Inheritance analysis
        info['inheritance'] = {
            'base_classes': [cls.__name__ for cls in strategy_class.__bases__],
            'mro': [cls.__name__ for cls in strategy_class.__mro__]
        }
        
        return info
    
    def _analyze_class_methods(self, strategy_class: Type) -> Dict[str, Dict[str, Any]]:
        """Analyze methods of a strategy class."""
        methods = {}
        
        for name, method in inspect.getmembers(strategy_class, inspect.isfunction):
            if not name.startswith('_'):  # Skip private methods
                signature = inspect.signature(method)
                methods[name] = {
                    'signature': str(signature),
                    'docstring': inspect.getdoc(method) or '',
                    'parameters': [
                        {
                            'name': param.name,
                            'default': param.default if param.default != inspect.Parameter.empty else None,
                            'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                        }
                        for param in signature.parameters.values()
                    ]
                }
        
        return methods
    
    def get_strategy_ui_schema(self, strategy_class: Type[EnhancedBaseStrategy]) -> Dict[str, Any]:
        """Generate UI schema for a strategy class.
        
        Args:
            strategy_class: Strategy class to generate UI schema for
            
        Returns:
            Complete UI schema suitable for frontend form generation
        """
        if not hasattr(strategy_class, 'get_ui_schema'):
            return {
                'error': 'Strategy does not support UI schema generation',
                'strategy_name': strategy_class.__name__
            }
        
        try:
            return strategy_class.get_ui_schema()
        except Exception as e:
            self.logger.error(f"Error generating UI schema for {strategy_class.__name__}: {e}")
            return {
                'error': str(e),
                'strategy_name': strategy_class.__name__
            }
    
    def compare_strategies(self, 
                          strategy_classes: List[Type[EnhancedBaseStrategy]]) -> Dict[str, Any]:
        """Compare multiple strategies and identify similarities/differences.
        
        Args:
            strategy_classes: List of strategy classes to compare
            
        Returns:
            Detailed comparison analysis
        """
        if len(strategy_classes) < 2:
            raise ValueError("Need at least 2 strategies to compare")
        
        comparison = {
            'strategies': [cls.__name__ for cls in strategy_classes],
            'parameter_comparison': {},
            'category_analysis': {},
            'complexity_analysis': {},
            'compatibility_analysis': {}
        }
        
        # Parameter comparison
        all_parameters = {}
        for strategy_class in strategy_classes:
            if hasattr(strategy_class, 'get_schema'):
                schema = strategy_class.get_schema()
                all_parameters[strategy_class.__name__] = set(schema.parameters.keys())
        
        if all_parameters:
            # Find common and unique parameters
            all_param_names = set().union(*all_parameters.values())
            common_params = set.intersection(*all_parameters.values()) if all_parameters else set()
            
            comparison['parameter_comparison'] = {
                'total_unique_parameters': len(all_param_names),
                'common_parameters': list(common_params),
                'strategy_specific_parameters': {
                    name: list(params - common_params)
                    for name, params in all_parameters.items()
                }
            }
        
        # Category analysis
        categories = [getattr(cls, 'strategy_category', 'general') for cls in strategy_classes]
        comparison['category_analysis'] = {
            'categories': categories,
            'same_category': len(set(categories)) == 1,
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)}
        }
        
        # Complexity analysis (based on parameter count and method count)
        complexity_scores = []
        for strategy_class in strategy_classes:
            param_count = 0
            if hasattr(strategy_class, 'get_schema'):
                param_count = len(strategy_class.get_schema().parameters)
            
            method_count = len([m for m in dir(strategy_class) if not m.startswith('_')])
            complexity_score = param_count + method_count * 0.5
            complexity_scores.append({
                'strategy': strategy_class.__name__,
                'parameter_count': param_count,
                'method_count': method_count,
                'complexity_score': complexity_score
            })
        
        comparison['complexity_analysis'] = {
            'scores': complexity_scores,
            'most_complex': max(complexity_scores, key=lambda x: x['complexity_score'])['strategy'],
            'least_complex': min(complexity_scores, key=lambda x: x['complexity_score'])['strategy']
        }
        
        return comparison
    
    def validate_strategy_configuration(self, 
                                      strategy_class: Type[EnhancedBaseStrategy],
                                      config_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters against strategy schema.
        
        Args:
            strategy_class: Strategy class to validate against
            config_params: Configuration parameters to validate
            
        Returns:
            Validation results with errors, warnings, and recommendations
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'validated_parameters': {},
            'unused_parameters': [],
            'missing_required': []
        }
        
        if not hasattr(strategy_class, 'get_schema'):
            validation_results['warnings'].append(
                "Strategy does not have parameter schema - cannot validate"
            )
            return validation_results
        
        try:
            schema = strategy_class.get_schema()
            
            # Validate parameters
            validated_params = schema.validate(config_params)
            validation_results['validated_parameters'] = validated_params
            
            # Check for unused parameters
            schema_params = set(schema.parameters.keys())
            config_param_names = set(config_params.keys())
            unused = config_param_names - schema_params
            validation_results['unused_parameters'] = list(unused)
            
            # Check for missing required parameters
            required_params = {
                name for name, param_def in schema.parameters.items()
                if param_def.required
            }
            missing_required = required_params - config_param_names
            validation_results['missing_required'] = list(missing_required)
            
            if missing_required:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Missing required parameters: {list(missing_required)}"
                )
            
            if unused:
                validation_results['warnings'].append(
                    f"Unused parameters provided: {list(unused)}"
                )
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results


class UISchemaGenerator:
    """Specialized generator for UI schemas and form configurations.
    
    This class creates comprehensive UI schemas that frontend frameworks
    can use to automatically generate parameter input forms with proper
    validation, grouping, and user experience enhancements.
    """
    
    def __init__(self):
        self.logger = LoggerManager.get_logger("introspection.UISchemaGenerator")
    
    def generate_form_schema(self, 
                           strategy_class: Type[EnhancedBaseStrategy],
                           theme: str = "default") -> Dict[str, Any]:
        """Generate a complete form schema for a strategy.
        
        Args:
            strategy_class: Strategy class to generate form for
            theme: UI theme configuration ("default", "compact", "detailed")
            
        Returns:
            Complete form schema with layout, validation, and styling information
        """
        if not hasattr(strategy_class, 'get_ui_schema'):
            raise ValueError(f"Strategy {strategy_class.__name__} does not support UI schema generation")
        
        base_schema = strategy_class.get_ui_schema()
        
        form_schema = {
            'form_id': f"{strategy_class.__name__.lower()}_form",
            'title': base_schema.get('strategy_name', strategy_class.__name__),
            'description': base_schema.get('description', ''),
            'version': base_schema.get('version', '1.0'),
            'theme': theme,
            'layout': self._generate_layout_config(base_schema, theme),
            'validation': self._generate_validation_config(base_schema),
            'groups': self._organize_parameter_groups(base_schema),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0',
                'strategy_class': strategy_class.__name__
            }
        }
        
        return form_schema
    
    def _generate_layout_config(self, base_schema: Dict[str, Any], theme: str) -> Dict[str, Any]:
        """Generate layout configuration for the form."""
        layouts = {
            "default": {
                "columns": 2,
                "group_spacing": "medium",
                "field_spacing": "small",
                "label_width": "30%",
                "input_width": "70%"
            },
            "compact": {
                "columns": 3,
                "group_spacing": "small",
                "field_spacing": "tight",
                "label_width": "25%",
                "input_width": "75%"
            },
            "detailed": {
                "columns": 1,
                "group_spacing": "large",
                "field_spacing": "medium",
                "label_width": "40%",
                "input_width": "60%",
                "show_descriptions": True,
                "show_tooltips": True
            }
        }
        
        return layouts.get(theme, layouts["default"])
    
    def _generate_validation_config(self, base_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation configuration for the form."""
        validation_config = {
            'validate_on_change': True,
            'validate_on_blur': True,
            'show_validation_summary': True,
            'highlight_invalid_fields': True,
            'validation_rules': {}
        }
        
        # Extract validation rules from parameter categories
        categories = base_schema.get('categories', {})
        for category_name, parameters in categories.items():
            for param_name, param_config in parameters.items():
                rules = []
                
                # Required validation
                if param_config.get('required', False):
                    rules.append({'type': 'required', 'message': f'{param_name} is required'})
                
                # Range validation
                if 'min' in param_config and 'max' in param_config:
                    rules.append({
                        'type': 'range',
                        'min': param_config['min'],
                        'max': param_config['max'],
                        'message': f'{param_name} must be between {param_config["min"]} and {param_config["max"]}'
                    })
                
                # Type validation
                param_type = param_config.get('type', 'string')
                if param_type in ['int', 'float']:
                    rules.append({
                        'type': 'numeric',
                        'message': f'{param_name} must be a valid number'
                    })
                
                if rules:
                    validation_config['validation_rules'][param_name] = rules
        
        return validation_config
    
    def _organize_parameter_groups(self, base_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Organize parameters into logical groups for the UI."""
        categories = base_schema.get('categories', {})
        groups = []
        
        # Define group ordering and styling
        group_configs = {
            'General': {'order': 1, 'icon': 'settings', 'color': 'blue'},
            'Technical Indicators': {'order': 2, 'icon': 'chart', 'color': 'green'},
            'Risk Management': {'order': 3, 'icon': 'shield', 'color': 'red'},
            'Position Sizing': {'order': 4, 'icon': 'resize', 'color': 'orange'},
            'Regime Filter': {'order': 5, 'icon': 'filter', 'color': 'purple'},
            'Volume Filter': {'order': 6, 'icon': 'volume', 'color': 'teal'},
            'Time Filter': {'order': 7, 'icon': 'clock', 'color': 'gray'}
        }
        
        for category_name, parameters in categories.items():
            config = group_configs.get(category_name, {'order': 999, 'icon': 'folder', 'color': 'gray'})
            
            group = {
                'name': category_name,
                'title': category_name,
                'order': config['order'],
                'icon': config['icon'],
                'color': config['color'],
                'collapsible': True,
                'collapsed': config['order'] > 3,  # Collapse advanced groups by default
                'parameters': []
            }
            
            # Sort parameters within group
            sorted_params = sorted(parameters.items(), key=lambda x: x[1].get('display_name', x[0]))
            
            for param_name, param_config in sorted_params:
                group['parameters'].append({
                    'name': param_name,
                    'config': param_config
                })
            
            groups.append(group)
        
        # Sort groups by order
        groups.sort(key=lambda x: x['order'])
        
        return groups
    
    def generate_parameter_documentation(self, 
                                       strategy_class: Type[EnhancedBaseStrategy],
                                       format: str = "markdown") -> str:
        """Generate parameter documentation in various formats.
        
        Args:
            strategy_class: Strategy class to document
            format: Output format ("markdown", "html", "rst")
            
        Returns:
            Formatted documentation string
        """
        if not hasattr(strategy_class, 'get_schema'):
            return f"# {strategy_class.__name__}\n\nNo parameter schema available."
        
        schema = strategy_class.get_schema()
        
        if format == "markdown":
            return self._generate_markdown_docs(strategy_class, schema)
        elif format == "html":
            return self._generate_html_docs(strategy_class, schema)
        elif format == "rst":
            return self._generate_rst_docs(strategy_class, schema)
        else:
            raise ValueError(f"Unsupported documentation format: {format}")
    
    def _generate_markdown_docs(self, strategy_class: Type, schema: ParameterSchema) -> str:
        """Generate Markdown documentation."""
        docs = [f"# {strategy_class.__name__}"]
        
        # Strategy description
        description = getattr(strategy_class, 'STRATEGY_DESCRIPTION', '')
        if description:
            docs.append(f"\n{description}")
        
        # Strategy metadata
        version = getattr(strategy_class, 'STRATEGY_VERSION', '1.0')
        category = getattr(strategy_class, 'strategy_category', 'general')
        tags = getattr(strategy_class, 'TAGS', [])
        
        docs.append(f"\n**Version:** {version}")
        docs.append(f"**Category:** {category}")
        if tags:
            docs.append(f"**Tags:** {', '.join(tags)}")
        
        # Parameters documentation
        docs.append("\n## Parameters")
        
        if not schema.parameters:
            docs.append("\nNo parameters defined for this strategy.")
        else:
            # Group parameters by category
            categories = {}
            for name, param_def in schema.parameters.items():
                category = param_def.category
                if category not in categories:
                    categories[category] = []
                categories[category].append((name, param_def))
            
            for category, params in categories.items():
                docs.append(f"\n### {category}")
                
                for name, param_def in params:
                    docs.append(f"\n#### `{name}`")
                    docs.append(f"- **Type:** {param_def.type.__name__}")
                    docs.append(f"- **Default:** `{param_def.default}`")
                    docs.append(f"- **Required:** {'Yes' if param_def.required else 'No'}")
                    
                    if param_def.range:
                        docs.append(f"- **Range:** {param_def.range[0]} to {param_def.range[1]}")
                    
                    if param_def.choices:
                        docs.append(f"- **Choices:** {', '.join(map(str, param_def.choices))}")
                    
                    if param_def.description:
                        docs.append(f"- **Description:** {param_def.description}")
        
        return '\n'.join(docs)
    
    def _generate_html_docs(self, strategy_class: Type, schema: ParameterSchema) -> str:
        """Generate HTML documentation."""
        # Simplified HTML generation - could be enhanced with proper templates
        html = f"<h1>{strategy_class.__name__}</h1>"
        
        description = getattr(strategy_class, 'STRATEGY_DESCRIPTION', '')
        if description:
            html += f"<p>{description}</p>"
        
        html += "<h2>Parameters</h2>"
        
        if schema.parameters:
            html += "<table border='1'>"
            html += "<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>"
            
            for name, param_def in schema.parameters.items():
                html += f"<tr>"
                html += f"<td><code>{name}</code></td>"
                html += f"<td>{param_def.type.__name__}</td>"
                html += f"<td><code>{param_def.default}</code></td>"
                html += f"<td>{param_def.description}</td>"
                html += f"</tr>"
            
            html += "</table>"
        else:
            html += "<p>No parameters defined.</p>"
        
        return html
    
    def _generate_rst_docs(self, strategy_class: Type, schema: ParameterSchema) -> str:
        """Generate reStructuredText documentation."""
        docs = [strategy_class.__name__]
        docs.append("=" * len(strategy_class.__name__))
        
        description = getattr(strategy_class, 'STRATEGY_DESCRIPTION', '')
        if description:
            docs.append(f"\n{description}")
        
        docs.append("\nParameters")
        docs.append("----------")
        
        if schema.parameters:
            for name, param_def in schema.parameters.items():
                docs.append(f"\n{name}")
                docs.append("~" * len(name))
                docs.append(f"Type: {param_def.type.__name__}")
                docs.append(f"Default: ``{param_def.default}``")
                if param_def.description:
                    docs.append(f"\n{param_def.description}")
        else:
            docs.append("\nNo parameters defined for this strategy.")
        
        return '\n'.join(docs)


class ConfigurationManager:
    """Manager for strategy configuration synchronization and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = LoggerManager.get_logger("introspection.ConfigurationManager")
        self.config_path = config_path or Path("config/config.yaml")
        self.introspector = StrategyIntrospector()
    
    def sync_strategies_with_config(self) -> Dict[str, Any]:
        """Synchronize strategy schemas with configuration file.
        
        Returns:
            Report of synchronization results and recommendations
        """
        sync_report = {
            'timestamp': datetime.now().isoformat(),
            'config_file': str(self.config_path),
            'strategies_analyzed': 0,
            'recommendations': [],
            'errors': [],
            'strategy_reports': {}
        }
        
        # Load existing configuration
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                strategies_config = config_data.get('strategies', {})
            else:
                strategies_config = {}
                sync_report['errors'].append(f"Configuration file not found: {self.config_path}")
        except Exception as e:
            sync_report['errors'].append(f"Error loading configuration: {e}")
            return sync_report
        
        # Analyze all registered strategies
        registry = StrategyRegistry.get_instance()
        for strategy_name in registry.list_strategies():
            try:
                strategy_class = registry.get_strategy_class(strategy_name)
                
                # Skip if not enhanced strategy
                if not issubclass(strategy_class, EnhancedBaseStrategy):
                    continue
                
                sync_report['strategies_analyzed'] += 1
                
                # Compare with configuration
                config_params = strategies_config.get(strategy_name, {}).get('parameters', {})
                schema = strategy_class.get_schema()
                
                comparison = ParameterManager.compare_with_config(schema, config_params)
                sync_report['strategy_reports'][strategy_name] = comparison
                
                # Add to general recommendations
                if comparison['recommendations']:
                    sync_report['recommendations'].extend([
                        f"{strategy_name}: {rec}" for rec in comparison['recommendations']
                    ])
                
            except Exception as e:
                sync_report['errors'].append(f"Error analyzing {strategy_name}: {e}")
        
        return sync_report
    
    def generate_config_template(self, strategy_classes: Optional[List[Type]] = None) -> str:
        """Generate configuration template for strategies.
        
        Args:
            strategy_classes: Optional list of specific strategies to include
            
        Returns:
            YAML configuration template
        """
        if strategy_classes is None:
            # Get all enhanced strategies from registry
            registry = StrategyRegistry.get_instance()
            strategy_classes = []
            for strategy_name in registry.list_strategies():
                try:
                    strategy_class = registry.get_strategy_class(strategy_name)
                    if issubclass(strategy_class, EnhancedBaseStrategy):
                        strategy_classes.append(strategy_class)
                except Exception:
                    continue
        
        config_template = {
            '# Auto-generated strategy configuration template': None,
            f'# Generated on: {datetime.now().isoformat()}': None,
            'strategies': {}
        }
        
        for strategy_class in strategy_classes:
            try:
                schema = strategy_class.get_schema()
                strategy_name = self._get_strategy_config_name(strategy_class)
                
                strategy_config = {
                    'enabled': True,
                    'parameters': schema.get_defaults()
                }
                
                # Add parameter comments
                if hasattr(schema, 'parameters'):
                    for param_name, param_def in schema.parameters.items():
                        if param_def.description:
                            # YAML comments would need special handling
                            pass
                
                config_template['strategies'][strategy_name] = strategy_config
                
            except Exception as e:
                self.logger.error(f"Error generating config for {strategy_class.__name__}: {e}")
        
        # Convert to YAML
        yaml_output = yaml.dump(config_template, default_flow_style=False, sort_keys=False)
        
        # Clean up the None entries (comments)
        yaml_output = yaml_output.replace(': null', '')
        
        return yaml_output
    
    def _get_strategy_config_name(self, strategy_class: Type) -> str:
        """Get the configuration name for a strategy class."""
        # Convert class name to config-friendly format
        class_name = strategy_class.__name__
        if class_name.endswith('Strategy'):
            class_name = class_name[:-8]
        
        # Convert CamelCase to snake_case
        config_name = ''.join(['_' + c.lower() if c.isupper() and i > 0 else c.lower() 
                              for i, c in enumerate(class_name)])
        
        return config_name