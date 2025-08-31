"""
Tests for UnifiedDataLoader functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.data.data_loader import UnifiedDataLoader, DataLoaderError, SchemaValidationError, DataQualityError


class TestUnifiedDataLoader:
    """Test UnifiedDataLoader functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_settings = Mock()
        self.mock_settings.get.return_value = {
            'data': {
                'cache_enabled': True,
                'validation_enabled': True,
                'quality_checks_enabled': True,
                'max_gap_minutes': 60,
                'max_price_change_pct': 0.20,
                'min_volume_threshold': 0,
                'max_workers': 2,
                'chunk_size': 100000
            }
        }
        self.loader = UnifiedDataLoader(self.mock_settings)

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.exists')
    def test_load_data_from_cache(self, mock_exists, mock_read_parquet):
        """Test loading data from cache."""
        mock_exists.return_value = True

        # Mock cached data
        cached_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        mock_read_parquet.return_value = cached_data

        result = self.loader.load_data(
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_type='ohlcv'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        mock_read_parquet.assert_called_once()

    def test_schema_validation_success(self):
        """Test successful schema validation."""
        # Create valid OHLCV data
        valid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        # Should not raise exception
        self.loader._validate_schema(valid_data, 'ohlcv')

    def test_schema_validation_missing_column(self):
        """Test schema validation with missing column."""
        # Create data missing 'volume' column
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5]
            # Missing 'volume'
        })

        with pytest.raises(SchemaValidationError, match="Missing required columns"):
            self.loader._validate_schema(invalid_data, 'ohlcv')

    def test_data_quality_checks(self):
        """Test data quality checks."""
        # Create data with potential issues
        data_with_gaps = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2023-01-01 09:30:00',
                '2023-01-01 09:31:00',
                '2023-01-01 09:32:00',
                '2023-01-01 09:45:00'  # 13-minute gap
            ]),
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [101.0, 102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0, 102.0],
            'close': [100.5, 101.5, 102.5, 103.5],
            'volume': [1000, 1100, 1200, 1300]
        })
        data_with_gaps.set_index('timestamp', inplace=True)

        # Should not raise exception but should log warnings
        result = self.loader._perform_quality_checks(data_with_gaps, 'ohlcv')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_resample_data(self):
        """Test data resampling."""
        # Create 1-minute data
        minute_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 09:30:00', periods=10, freq='1min'),
            'open': np.random.uniform(100, 105, 10),
            'high': np.random.uniform(105, 110, 10),
            'low': np.random.uniform(95, 100, 10),
            'close': np.random.uniform(100, 105, 10),
            'volume': np.random.randint(1000, 2000, 10)
        })
        minute_data.set_index('timestamp', inplace=True)

        # Resample to 5 minutes
        result = self.loader._resample_data(minute_data, '5min')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 10 minutes / 5 minutes = 2 bars

    def test_error_handling(self):
        """Test error handling in data loading."""
        # Test with invalid data type
        with pytest.raises(DataLoaderError):
            self.loader.load_data(
                symbol='TEST',
                start_date='2023-01-01',
                end_date='2023-01-02',
                data_type='invalid_type'
            )


class TestDataLoaderIntegration:
    """Test integration between DataLoader and other components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_settings = Mock()
        self.mock_settings.get.return_value = {
            'data': {
                'cache_enabled': False,
                'validation_enabled': True,
                'quality_checks_enabled': True
            }
        }
        self.loader = UnifiedDataLoader(self.mock_settings)

    def test_backward_compatibility(self):
        """Test backward compatibility with existing data formats."""
        # Create data in the format expected by existing systems
        legacy_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 09:30:00', periods=50, freq='1min'),
            'open': np.random.uniform(100, 105, 50),
            'high': np.random.uniform(105, 110, 50),
            'low': np.random.uniform(95, 100, 50),
            'close': np.random.uniform(100, 105, 50),
            'volume': np.random.randint(1000, 2000, 50)
        })
        legacy_data.set_index('timestamp', inplace=True)

        # Test that the loader can handle this format
        validated_data = self.loader._perform_quality_checks(legacy_data, 'ohlcv')

        assert isinstance(validated_data, pd.DataFrame)
        assert len(validated_data) == len(legacy_data)

        # Ensure all expected columns are present
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in validated_data.columns

    def test_polygon_databento_format_compatibility(self):
        """Test compatibility between Polygon and Databento data formats."""
        # Create sample data in Polygon format
        polygon_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 09:30:00', periods=10, freq='1min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'vwap': [100.3, 101.3, 102.3, 103.3, 104.3, 105.3, 106.3, 107.3, 108.3, 109.3],
            'transactions': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        polygon_data.set_index('timestamp', inplace=True)

        # Create equivalent data in Databento format
        databento_data = pd.DataFrame({
            'ts_event': pd.date_range('2023-01-01 09:30:00', periods=10, freq='1min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        databento_data.set_index('ts_event', inplace=True)

        # Both should pass validation
        self.loader._validate_schema(polygon_data, 'ohlcv')
        self.loader._validate_schema(databento_data, 'ohlcv')

        # Both should pass quality checks
        polygon_cleaned = self.loader._perform_quality_checks(polygon_data, 'ohlcv')
        databento_cleaned = self.loader._perform_quality_checks(databento_data, 'ohlcv')

        assert len(polygon_cleaned) == len(polygon_data)
        assert len(databento_cleaned) == len(databento_data)