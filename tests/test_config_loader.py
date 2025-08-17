"""
Tests for configuration loader module.
"""
import pytest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.utils.config_loader import Settings, ConfigError, load_yaml


class TestSettings:
    """Test Settings configuration class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            "project": "rl-intraday",
            "seed": 42,
            "data": {
                "instrument": "MES",
                "minute_file": "data/raw/mes_1min.parquet",
                "session": {
                    "tz": "America/New_York",
                    "rth_start": "09:30",
                    "rth_end": "16:00"
                }
            },
            "features": {
                "technical": {
                    "returns_horizons": [1, 3, 5, 15, 60],
                    "atr_window": 14
                }
            }
        }
    
    def test_settings_creation(self):
        """Test Settings object creation."""
        settings = Settings(self.test_config)
        assert settings._config == self.test_config
    
    def test_get_simple_value(self):
        """Test getting simple configuration value."""
        settings = Settings(self.test_config)
        assert settings.get("project") == "rl-intraday"
        assert settings.get("seed") == 42
    
    def test_get_nested_value(self):
        """Test getting nested configuration value."""
        settings = Settings(self.test_config)
        assert settings.get("data", "instrument") == "MES"
        assert settings.get("data", "session", "tz") == "America/New_York"
        assert settings.get("features", "technical", "atr_window") == 14
    
    def test_get_with_default(self):
        """Test getting value with default fallback."""
        settings = Settings(self.test_config)
        assert settings.get("nonexistent", "default_value") == "default_value"
        assert settings.get("data", "nonexistent", "default") == "default"
    
    def test_get_list_value(self):
        """Test getting list configuration value."""
        settings = Settings(self.test_config)
        horizons = settings.get("features", "technical", "returns_horizons")
        assert horizons == [1, 3, 5, 15, 60]
    
    def test_get_nonexistent_raises_error(self):
        """Test that getting nonexistent key without default raises ConfigError."""
        settings = Settings(self.test_config)
        with pytest.raises(ConfigError):
            settings.get("nonexistent")
    
    def test_get_nested_nonexistent_raises_error(self):
        """Test that getting nonexistent nested key without default raises ConfigError."""
        settings = Settings(self.test_config)
        with pytest.raises(ConfigError):
            settings.get("data", "nonexistent")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_from_paths_single_file(self, mock_yaml_load, mock_file_open):
        """Test loading settings from single YAML file."""
        mock_yaml_load.return_value = self.test_config
        
        settings = Settings.from_paths("config.yaml")
        
        mock_file_open.assert_called_once_with("config.yaml", 'r')
        mock_yaml_load.assert_called_once()
        assert settings.get("project") == "rl-intraday"
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_from_paths_multiple_files(self, mock_yaml_load, mock_file_open):
        """Test loading and merging settings from multiple YAML files."""
        config1 = {"project": "rl-intraday", "seed": 42}
        config2 = {"data": {"instrument": "MES"}}
        mock_yaml_load.side_effect = [config1, config2]
        
        settings = Settings.from_paths("config1.yaml", "config2.yaml")
        
        assert mock_file_open.call_count == 2
        assert mock_yaml_load.call_count == 2
        assert settings.get("project") == "rl-intraday"
        assert settings.get("data", "instrument") == "MES"
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_from_paths_file_not_found(self, mock_file_open):
        """Test handling of missing configuration file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            Settings.from_paths("nonexistent.yaml")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML"))
    def test_from_paths_invalid_yaml(self, mock_yaml_load, mock_file_open):
        """Test handling of invalid YAML content."""
        with pytest.raises(ConfigError, match="Error parsing YAML"):
            Settings.from_paths("invalid.yaml")


class TestLoadYaml:
    """Test load_yaml utility function."""
    
    def test_load_yaml_success(self):
        """Test successful YAML loading."""
        yaml_content = """
        project: rl-intraday
        seed: 42
        data:
          instrument: MES
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(yaml_content)
            tmp_file_path = tmp_file.name
        
        try:
            config = load_yaml(tmp_file_path)
            assert config["project"] == "rl-intraday"
            assert config["seed"] == 42
            assert config["data"]["instrument"] == "MES"
        finally:
            os.unlink(tmp_file_path)
    
    def test_load_yaml_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_yaml("nonexistent.yaml")
    
    def test_load_yaml_invalid_content(self):
        """Test handling of invalid YAML content."""
        invalid_yaml = "invalid: yaml: content:"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(invalid_yaml)
            tmp_file_path = tmp_file.name
        
        try:
            with pytest.raises(ConfigError, match="Error parsing YAML"):
                load_yaml(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)