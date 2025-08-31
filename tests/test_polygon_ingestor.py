"""
Tests for Polygon Data Ingestor

This module contains unit tests for the PolygonDataIngestor class,
focusing on core functionality, error handling, and integration.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd

from src.data.polygon_ingestor import PolygonDataIngestor, IngestionMetadata
from src.utils.config_loader import Settings


class TestPolygonDataIngestor:
    """Test cases for PolygonDataIngestor."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        config_data = {
            'data': {
                'polygon': {
                    'api_key': 'test_key',
                    'data_dir': 'data/polygon/historical',
                    'retry': {
                        'attempts': 1,
                        'backoff_factor': 1.0
                    }
                },
                'max_workers': 2,
                'batch_size': 100,
                'chunk_size': 1000,
                'compression': 'snappy',
                'row_group_size': 10000,
                'validation': {
                    'enabled': True
                }
            }
        }
        return config_data

    @pytest.fixture
    def mock_settings(self, temp_config):
        """Mock settings object."""
        settings = Mock(spec=Settings)
        settings.get = lambda key, default=None: temp_config.get(key, default)
        return settings

    @pytest.fixture
    def ingestor(self, mock_settings):
        """Create ingestor instance for testing."""
        with patch('src.data.polygon_ingestor.PolygonClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            ingestor = PolygonDataIngestor(mock_settings)

            # Mock the async close method
            ingestor.aclose = AsyncMock()

            return ingestor

    def test_initialization(self, ingestor):
        """Test ingestor initialization."""
        assert ingestor.data_dir.name == "historical"
        assert ingestor.metadata_dir.name == "metadata"
        assert ingestor.max_workers == 2
        assert ingestor.retry_attempts == 1
        assert ingestor.compression == 'snappy'

    def test_storage_path_generation(self, ingestor):
        """Test partitioned storage path generation."""
        path = ingestor._get_storage_path('AAPL', '2023-01-01')
        expected_parts = ['data', 'polygon', 'historical', 'symbol=AAPL',
                         'year=2023', 'month=01', 'day=01', 'data.parquet']

        path_str = str(path)
        for part in expected_parts:
            assert part in path_str

    def test_incremental_date_range_calculation(self, ingestor):
        """Test incremental date range calculation."""
        # Mock metadata
        metadata = IngestionMetadata(
            symbol='AAPL',
            data_type='ohlcv',
            last_successful_fetch=datetime(2023, 1, 3)
        )
        ingestor.metadata_cache['AAPL_ohlcv'] = metadata

        # Test incremental range
        start, end = ingestor._get_incremental_date_range(
            'AAPL', 'ohlcv', '2023-01-01', '2023-01-05'
        )

        assert start == '2023-01-04'  # Day after last successful fetch
        assert end == '2023-01-05'

    def test_data_validation_ohlcv(self, ingestor):
        """Test OHLCV data validation."""
        # Create valid OHLCV data
        data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000, 2000]
        })
        data.index = pd.date_range('2023-01-01', periods=2, freq='1min')

        validated_data = ingestor._validate_data(data, 'ohlcv')

        assert len(validated_data) == 2
        assert list(validated_data.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_data_validation_quotes(self, ingestor):
        """Test quotes data validation."""
        # Create valid quotes data
        data = pd.DataFrame({
            'bid_price': [100.0, 101.0],
            'bid_size': [100, 200],
            'ask_price': [101.0, 102.0],
            'ask_size': [150, 250]
        })
        data.index = pd.date_range('2023-01-01', periods=2, freq='1s')

        validated_data = ingestor._validate_data(data, 'quotes')

        assert len(validated_data) == 2
        assert 'bid_price' in validated_data.columns
        assert 'ask_price' in validated_data.columns

    def test_data_validation_invalid_ohlcv(self, ingestor):
        """Test validation with invalid OHLCV data."""
        # Create data with invalid OHLC values
        data = pd.DataFrame({
            'open': [100.0, -1.0],  # Negative price
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000, 2000]
        })
        data.index = pd.date_range('2023-01-01', periods=2, freq='1min')

        validated_data = ingestor._validate_data(data, 'ohlcv')

        # Should remove the invalid row
        assert len(validated_data) == 1

    def test_metadata_tracking(self, ingestor):
        """Test metadata tracking functionality."""
        # Create test metadata
        metadata = IngestionMetadata(
            symbol='AAPL',
            data_type='ohlcv',
            total_records=1000,
            last_update=datetime.now()
        )

        # Save metadata
        ingestor._save_metadata(metadata)

        # Verify it's in cache
        assert 'AAPL_ohlcv' in ingestor.metadata_cache
        cached_metadata = ingestor.metadata_cache['AAPL_ohlcv']
        assert cached_metadata.total_records == 1000
        assert cached_metadata.symbol == 'AAPL'

    def test_status_reporting(self, ingestor):
        """Test ingestion status reporting."""
        # Add some test metadata
        metadata1 = IngestionMetadata('AAPL', 'ohlcv', total_records=1000)
        metadata2 = IngestionMetadata('MSFT', 'ohlcv', total_records=2000)
        ingestor.metadata_cache['AAPL_ohlcv'] = metadata1
        ingestor.metadata_cache['MSFT_ohlcv'] = metadata2

        # Test overall status
        status = ingestor.get_ingestion_status()
        assert status['total_symbols'] == 2
        assert status['total_records'] == 3000
        assert 'AAPL' in status['symbols']
        assert 'MSFT' in status['symbols']

        # Test symbol-specific status
        aapl_status = ingestor.get_ingestion_status('AAPL')
        assert aapl_status['symbol'] == 'AAPL'
        assert 'AAPL_ohlcv' in aapl_status['metadata']

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, ingestor):
        """Test dry run functionality."""
        # Mock the fetch method
        with patch.object(ingestor, '_fetch_single_symbol_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({'test': [1, 2, 3]})

            results = await ingestor.fetch_historical_data(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-03',
                data_types=['ohlcv'],
                dry_run=True
            )

            # In dry run mode, should not call actual fetch
            mock_fetch.assert_not_called()

            # But should still return results structure
            assert 'total_symbols' in results
            assert 'successful_fetches' in results

    def test_error_handling_invalid_data_type(self, ingestor):
        """Test error handling for invalid data types."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            asyncio.run(ingestor.fetch_historical_data(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-03',
                data_types=['invalid_type']
            ))

    def test_metadata_clearing(self, ingestor):
        """Test metadata clearing functionality."""
        # Add test metadata
        metadata = IngestionMetadata('AAPL', 'ohlcv')
        ingestor.metadata_cache['AAPL_ohlcv'] = metadata

        # Clear metadata
        ingestor.clear_metadata(symbol='AAPL')

        # Verify it's cleared
        assert 'AAPL_ohlcv' not in ingestor.metadata_cache

    def test_partition_directory_creation(self, ingestor):
        """Test automatic partition directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override data_dir for testing
            ingestor.data_dir = Path(temp_dir) / "test_data"

            # Ensure partition directory is created
            data_file = ingestor._ensure_partition_directory('AAPL', '2023-01-01')

            assert data_file.parent.exists()
            assert data_file.name == "data.parquet"

            # Check directory structure
            parts = data_file.parent.parts
            assert 'symbol=AAPL' in parts
            assert 'year=2023' in parts
            assert 'month=01' in parts
            assert 'day=01' in parts


class TestIngestionMetadata:
    """Test cases for IngestionMetadata dataclass."""

    def test_metadata_creation(self):
        """Test metadata object creation."""
        metadata = IngestionMetadata(
            symbol='AAPL',
            data_type='ohlcv',
            total_records=1000,
            last_update=datetime(2023, 1, 1, 12, 0, 0)
        )

        assert metadata.symbol == 'AAPL'
        assert metadata.data_type == 'ohlcv'
        assert metadata.total_records == 1000
        assert metadata.error_count == 0
        assert metadata.last_error is None

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = IngestionMetadata('AAPL', 'ohlcv')

        assert metadata.last_update is None
        assert metadata.date_range_start is None
        assert metadata.date_range_end is None
        assert metadata.total_records == 0
        assert metadata.last_successful_fetch is None
        assert metadata.error_count == 0
        assert metadata.last_error is None


if __name__ == '__main__':
    pytest.main([__file__])