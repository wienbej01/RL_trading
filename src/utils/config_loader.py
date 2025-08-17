"""
Configuration loader for the RL trading system.

This module provides utilities for loading and managing configuration files
following 12-factor app principles with no secrets in code.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class Settings:
    """
    Configuration settings for the RL trading system.
    
    This class provides type-safe access to configuration values with
    sensible defaults and validation.
    """
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_paths(cls, settings_path: Union[str, Path]) -> 'Settings':
        """
        Load settings from YAML configuration file.
        
        Args:
            settings_path: Path to the YAML configuration file
            
        Returns:
            Settings instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        settings_path = Path(settings_path)
        if not settings_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {settings_path}")
        
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {settings_path}")
            return cls(raw=config)
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Configuration path components (e.g., 'data', 'instrument')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> settings.get('data', 'instrument')
            'MES'
            >>> settings.get('risk', 'max_daily_loss_r', 3.0)
            3.0
        """
        current = self.raw
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                logger.debug(f"Configuration key not found: {'.'.join(keys)}, returning default: {default}")
                return default
        return current
    
    def get_instrument_config(self, instrument: str) -> Dict[str, Any]:
        """
        Get instrument-specific configuration.
        
        Args:
            instrument: Instrument symbol (e.g., 'MES', 'MICRO10Y')
            
        Returns:
            Instrument configuration dictionary
            
        Raises:
            ValueError: If instrument configuration not found
        """
        instruments_config = self.get('instruments')
        if not instruments_config or instrument not in instruments_config:
            raise ValueError(f"Instrument configuration not found: {instrument}")
        
        return instruments_config[instrument]
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required sections
        required_sections = ['data', 'risk', 'execution', 'env', 'train']
        for section in required_sections:
            if not self.get(section):
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate risk parameters
        risk_per_trade = self.get('risk', 'risk_per_trade_frac')
        if not (0 < risk_per_trade <= 1):
            raise ValueError(f"risk_per_trade_frac must be between 0 and 1, got: {risk_per_trade}")
        
        # Validate reward parameters
        reward_kind = self.get('env', 'reward', 'kind')
        if reward_kind not in ['dsr', 'sharpe', 'profit']:
            raise ValueError(f"Invalid reward kind: {reward_kind}")
        
        logger.info("Configuration validation passed")


def load_config(config_path: Union[str, Path]) -> Settings:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded settings instance
    """
    settings = Settings.from_paths(config_path)
    settings.validate()
    return settings


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent.parent


def get_config_path(config_name: str = "settings.yaml") -> Path:
    """
    Get path to configuration file.
    
    Args:
        config_name: Name of configuration file
        
    Returns:
        Full path to configuration file
    """
    return get_project_root() / "configs" / config_name


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading YAML file {file_path}: {e}")
        raise