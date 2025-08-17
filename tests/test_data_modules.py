"""
Tests for data modules (IBKR client, Databento client, VIX loader, economic calendar).
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.data.ibkr_client import IBKRClient
from src.data.databento_client import DatabentoClient
from src.data.vix_loader import VIXLoader
from src.data.econ_calendar import EconCalendar


class TestIBKRClient:
    """Test IBKR client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = IBKRClient()
    
    @patch('src.data.ibkr_client.IB')
    def test_connect_success(self, mock_ib):
        """Test successful connection to IBKR."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.connect.return_value = True
        mock_ib_instance.isConnected.return_value = True
        
        result = self.client.connect()
        
        assert result is True
        mock_ib_instance.connect.assert_called_once()
    
    @patch('src.data.ibkr_client.IB')
    def test_connect_failure(self, mock_ib):
        """Test failed connection to IBKR."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        mock_ib_instance.connect.return_value = False
        mock_ib_instance.isConnected.return_value = False
        
        result = self.client.connect()
        
        assert result is False
    
    @patch('src.data.ibkr_client.IB')
    def test_disconnect(self, mock_ib):
        """Test disconnection from IBKR."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        
        self.client.ib = mock_ib_instance
        result = self.client.disconnect()
        
        mock_ib_instance.disconnect.assert_called_once()
        assert result is True
    
    @patch('src.data.ibkr_client.IB')
    def test_fetch_historical_data(self, mock_ib):
        """Test fetching historical data."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        
        # Create mock historical bars
        mock_bars = []
        base_time = datetime.now()
        for i in range(5):
            mock_bar = Mock()
            mock_bar.date = base_time + timedelta(minutes=i)
            mock_bar.open = 100 + i
            mock_bar.high = 105 + i
            mock_bar.low = 95 + i
            mock_bar.close = 102 + i
            mock_bar.volume = 1000 + i * 100
            mock_bars.append(mock_bar)
        
        mock_ib_instance.reqHistoricalData.return_value = mock_bars
        self.client.ib = mock_ib_instance
        
        # Create mock contract
        mock_contract = Mock()
        mock_contract.symbol = "MES"
        
        data = self.client.fetch_historical_data(
            contract=mock_contract,
            duration="1 D",
            bar_size="1 min"
        )
        
        # Verify returned data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in data.columns
        
        # Check data values
        assert data['open'].iloc[0] == 100
        assert data['volume'].iloc[-1] == 1400
    
    @patch('src.data.ibkr_client.IB')
    def test_get_current_price(self, mock_ib):
        """Test getting current market price."""
        mock_ib_instance = Mock()
        mock_ib.return_value = mock_ib_instance
        
        mock_ticker = Mock()
        mock_ticker.last = 4500.0
        mock_ticker.bid = 4499.5
        mock_ticker.ask = 4500.5
        
        mock_ib_instance.reqMktData.return_value = mock_ticker
        self.client.ib = mock_ib_instance
        
        mock_contract = Mock()
        price_data = self.client.get_current_price(mock_contract)
        
        assert isinstance(price_data, dict)
        assert 'last' in price_data
        assert 'bid' in price_data
        assert 'ask' in price_data
        assert price_data['last'] == 4500.0


class TestDatabentoClient:
    """Test Databento client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = DatabentoClient(api_key="test_key")
    
    @patch('src.data.databento_client.Historical')
    def test_fetch_historical_data(self, mock_historical):
        """Test fetching historical data from Databento."""
        mock_client = Mock()
        mock_historical.return_value = mock_client
        
        # Create mock response
        mock_response = Mock()
        
        # Create sample DataFrame
        sample_data = pd.DataFrame({
            'ts_event': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        mock_response.to_df.return_value = sample_data
        mock_client.timeseries.get_range.return_value = mock_response
        
        # Test data fetching
        result = self.client.fetch_historical_data(
            dataset="GLBX.MDP3",
            symbols="MES",
            schema="ohlcv-1m",
            start="2023-01-01",
            end="2023-01-02"
        )
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in result.columns
        
        # Verify API was called correctly
        mock_client.timeseries.get_range.assert_called_once_with(
            dataset="GLBX.MDP3",
            symbols="MES",
            schema="ohlcv-1m",
            start="2023-01-01",
            end="2023-01-02"
        )
    
    def test_save_to_parquet(self):
        """Test saving data to Parquet format."""
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='1min'),
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save data
            self.client.save_to_parquet(test_data, tmp_path)
            
            # Verify file exists
            assert os.path.exists(tmp_path)
            
            # Load and verify data
            loaded_data = pd.read_parquet(tmp_path)
            assert len(loaded_data) == len(test_data)
            assert list(loaded_data.columns) == list(test_data.columns)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('src.data.databento_client.Historical')
    def test_fetch_with_error_handling(self, mock_historical):
        """Test error handling in data fetching."""
        mock_client = Mock()
        mock_historical.return_value = mock_client
        mock_client.timeseries.get_range.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            self.client.fetch_historical_data(
                dataset="GLBX.MDP3",
                symbols="MES",
                schema="ohlcv-1m",
                start="2023-01-01",
                end="2023-01-02"
            )


class TestVIXLoader:
    """Test VIX data loader functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = VIXLoader()
    
    @patch('pandas.read_csv')
    def test_load_vix_data(self, mock_read_csv):
        """Test loading VIX historical data."""
        # Create mock VIX data
        mock_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Open': [20.0, 20.5, 21.0],
            'High': [20.5, 21.0, 21.5],
            'Low': [19.5, 20.0, 20.5],
            'Close': [20.2, 20.8, 21.2],
            'Volume': [100000, 110000, 120000]
        })
        mock_read_csv.return_value = mock_data
        
        result = self.loader.load_vix_data(
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in result.columns
        
        # Verify date filtering was applied
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_csv')
    def test_load_vix_term_structure(self, mock_read_csv):
        """Test loading VIX term structure data."""
        mock_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'VIX': [20.0, 20.5],
            'VIX9D': [19.5, 20.0],
            'VIX3M': [21.0, 21.5],
            'VIX6M': [21.5, 22.0]
        })
        mock_read_csv.return_value = mock_data
        
        result = self.loader.load_vix_term_structure(
            start_date="2023-01-01",
            end_date="2023-01-02"
        )
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        
        expected_columns = ['Date', 'VIX', 'VIX9D', 'VIX3M', 'VIX6M']
        for col in expected_columns:
            assert col in result.columns
    
    def test_calculate_vix_percentile(self):
        """Test VIX percentile calculation."""
        # Create sample VIX data
        vix_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.uniform(15, 35, 100)
        })
        
        current_vix = 25.0
        percentile = self.loader.calculate_vix_percentile(vix_data, current_vix)
        
        # Percentile should be between 0 and 100
        assert 0 <= percentile <= 100
        assert isinstance(percentile, float)
    
    @patch('pandas.read_csv', side_effect=FileNotFoundError())
    def test_load_vix_data_file_not_found(self, mock_read_csv):
        """Test handling of missing VIX data file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_vix_data()


class TestEconCalendar:
    """Test economic calendar functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calendar = EconCalendar(api_key="test_key")
    
    @patch('requests.get')
    def test_fetch_events(self, mock_get):
        """Test fetching economic events."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'events': [
                {
                    'date': '2023-01-01',
                    'time': '10:00:00',
                    'country': 'United States',
                    'event': 'Non-Farm Payrolls',
                    'impact': 'High',
                    'actual': '200K',
                    'forecast': '180K',
                    'previous': '190K'
                },
                {
                    'date': '2023-01-02',
                    'time': '14:00:00',
                    'country': 'United States',
                    'event': 'FOMC Interest Rate Decision',
                    'impact': 'High',
                    'actual': '5.25%',
                    'forecast': '5.25%',
                    'previous': '5.00%'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        events = self.calendar.fetch_events(
            start_date="2023-01-01",
            end_date="2023-01-31",
            countries=["United States"]
        )
        
        # Verify result
        assert isinstance(events, list)
        assert len(events) == 2
        
        # Check event structure
        event = events[0]
        expected_keys = ['date', 'time', 'country', 'event', 'impact', 'actual', 'forecast', 'previous']
        for key in expected_keys:
            assert key in event
        
        # Verify API call
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_get_high_impact_events(self, mock_get):
        """Test filtering high impact events."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'events': [
                {
                    'date': '2023-01-01',
                    'event': 'High Impact Event',
                    'impact': 'High'
                },
                {
                    'date': '2023-01-01',
                    'event': 'Medium Impact Event',
                    'impact': 'Medium'
                },
                {
                    'date': '2023-01-01',
                    'event': 'Low Impact Event',
                    'impact': 'Low'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        events = self.calendar.get_high_impact_events(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        # Should only return high impact events
        assert len(events) == 1
        assert events[0]['impact'] == 'High'
    
    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):
            self.calendar.fetch_events(
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
    
    def test_is_market_moving_event(self):
        """Test identification of market-moving events."""
        # High impact events should be market moving
        assert self.calendar.is_market_moving_event("Non-Farm Payrolls")
        assert self.calendar.is_market_moving_event("FOMC Interest Rate Decision")
        
        # Low impact events should not be market moving
        assert not self.calendar.is_market_moving_event("Building Permits")
    
    def test_get_events_for_day(self):
        """Test getting events for specific day."""
        events = [
            {'date': '2023-01-01', 'event': 'Event 1'},
            {'date': '2023-01-02', 'event': 'Event 2'},
            {'date': '2023-01-01', 'event': 'Event 3'}
        ]
        
        target_date = "2023-01-01"
        filtered_events = self.calendar.get_events_for_day(events, target_date)
        
        assert len(filtered_events) == 2
        for event in filtered_events:
            assert event['date'] == target_date


class TestDataIntegration:
    """Test integration between data modules."""
    
    def test_data_pipeline_flow(self):
        """Test end-to-end data pipeline flow."""
        # This would test the flow from data fetching to storage
        # Mock the entire pipeline
        with patch('src.data.ibkr_client.IBKRClient') as mock_ibkr, \
             patch('src.data.databento_client.DatabentoClient') as mock_databento:
            
            # Setup mocks
            mock_ibkr_instance = Mock()
            mock_databento_instance = Mock()
            mock_ibkr.return_value = mock_ibkr_instance
            mock_databento.return_value = mock_databento_instance
            
            # Mock data
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
                'open': np.random.uniform(100, 105, 10),
                'high': np.random.uniform(105, 110, 10),
                'low': np.random.uniform(95, 100, 10),
                'close': np.random.uniform(100, 105, 10),
                'volume': np.random.randint(1000, 2000, 10)
            })
            
            mock_ibkr_instance.fetch_historical_data.return_value = sample_data
            mock_databento_instance.fetch_historical_data.return_value = sample_data
            
            # Test the pipeline
            ibkr_data = mock_ibkr_instance.fetch_historical_data()
            databento_data = mock_databento_instance.fetch_historical_data()
            
            # Verify data consistency
            assert len(ibkr_data) == len(databento_data)
            assert list(ibkr_data.columns) == list(databento_data.columns)