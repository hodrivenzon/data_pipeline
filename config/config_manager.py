"""
Configuration Management

Configuration management system for the data pipeline.
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for loading and accessing YAML configuration.
    
    Provides methods to load configuration from YAML files and access
    configuration values using dot notation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        self.config_path = Path(config_path)
        self.project_root = self._get_project_root()
        self.config = self._load_config()
        self.config = self._resolve_dynamic_paths()
        logger.info(f"✅ Loaded configuration from {self.config_path}")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            logger.error(f"❌ Invalid YAML in {self.config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load config from {self.config_path}: {e}")
            raise
    
    def _get_project_root(self) -> Path:
        """
        Get the project root directory by finding the directory containing config.yaml.
        
        Returns:
            Path to the project root directory
        """
        # Start from the config file directory and work up to find project root
        current_dir = self.config_path.parent
        
        # Look for project root indicators (like .git, README.md, or specific structure)
        while current_dir != current_dir.parent:
            # Check for common project root indicators
            if (current_dir / ".git").exists() or \
               (current_dir / "README.md").exists() or \
               (current_dir / "requirements.txt").exists() or \
               (current_dir / "main.py").exists():
                return current_dir
            current_dir = current_dir.parent
        
        # Fallback: use the config file's parent directory
        return self.config_path.parent
    
    def _resolve_dynamic_paths(self) -> Dict[str, Any]:
        """
        Resolve dynamic paths in configuration by converting relative paths to absolute paths.
        
        Returns:
            Configuration dictionary with resolved absolute paths
        """
        config = self.config.copy()
        
        # Define path keys that should be resolved dynamically
        path_keys = [
            'files.projects_file',
            'files.subjects_file', 
            'files.results_file',
            'files.output_parquet',
            'files.summary_csv',
            'output.validation_report_path',
            'logging.log_file_path'
        ]
        
        for key in path_keys:
            current_value = self._get_nested_value(config, key)
            if current_value and isinstance(current_value, str):
                # Check if it's already an absolute path
                if not os.path.isabs(current_value):
                    # Convert relative path to absolute path
                    resolved_path = (self.project_root / current_value).resolve()
                    self._set_nested_value(config, key, str(resolved_path))
                    logger.debug(f"🔄 Resolved path: {key} -> {resolved_path}")
                else:
                    # Path is already absolute, but ensure it exists relative to project
                    if not current_value.startswith(str(self.project_root)):
                        # If it's absolute but outside project, make it relative to project
                        path_name = Path(current_value).name
                        # Try to find the file in project structure
                        for root, dirs, files in os.walk(self.project_root):
                            if path_name in files:
                                resolved_path = Path(root) / path_name
                                self._set_nested_value(config, key, str(resolved_path))
                                logger.debug(f"🔄 Found and resolved path: {key} -> {resolved_path}")
                                break
        
        return config
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """
        Get nested value from configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot notation key (e.g., 'files.projects_file')
            
        Returns:
            Value at the specified key or None if not found
        """
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """
        Set nested value in configuration using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot notation key (e.g., 'files.projects_file')
            value: Value to set
        """
        keys = key.split('.')
        current = config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'files.projects_file')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get('files.projects_file')
            '/path/to/projects.csv'
            >>> config.get('logging.log_level', 'INFO') 
            'DEBUG'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Examples:
            >>> config.set('files.output_dir', '/new/path')
            >>> config.set('logging.log_level', 'DEBUG')
        """
        keys = key.split('.')
        current = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    def has(self, key: str) -> bool:
        """
        Check if configuration key exists.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
        
        return True
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that all required keys exist in configuration.
        
        Args:
            required_keys: List of required configuration keys
            
        Returns:
            True if all keys exist, False otherwise
        """
        missing_keys = []
        
        for key in required_keys:
            if not self.has(key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"❌ Missing required configuration keys: {missing_keys}")
            return False
        
        return True
    
    def reload(self) -> None:
        """
        Reload configuration from file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        self.config = self._load_config()
        logger.info(f"🔄 Reloaded configuration from {self.config_path}")
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Optional path to save to. If None, saves to original path.
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"💾 Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise
