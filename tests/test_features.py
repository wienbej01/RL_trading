"""
Tests for feature engineering modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from src.features.technical_indicators import (
    calculate_returns,
    calculate_log_returns,
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_ema,
    calculate_stochastic_oscillator,
    calculate_williams_r
)

from src.features.microstructure_features import (
    calculate_order_flow_imbalance,
    calculate_microprice,
    calculate_queue_imbalance,
    calculate_spread,
    calculate_price_impact,
    calculate_vwap,
    calculate_twap
)

from src.features.time_features import (
    extract_time_of_day_features,
    extract_day_of_week_features,
    extract_session_features,
    is_market_hours,
    get_time_to_close,
    get_time_from_open
)

from src.features.pipeline import FeaturePipeline


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        
        # Create realistic OHLCV data with some trend
        base_price = 4500
        returns = np.random.normal(0, 0.001, 100)
        prices = base_price + np.cumsum(returns * base_price)
        
        self.data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, 100),
            'high': prices + np.abs(np.random.normal(0, 1.0, 100)),
            'low': prices - np.abs(np.random.normal(0, 1.0, 100)),
            'close': prices,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        self.data['high'] = np.maximum(self.data['high'], 
                                       np.maximum(self.data['open'], self.data['close']))
        self.data['low'] = np.minimum(self.data['low'], 
                                      np.minimum(self.data['open'], self.data['close']))
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        returns = calculate_returns(self.data['close'])
        
        # Check basic properties
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(self.data)
        assert pd.isna(returns.iloc[0])  # First value should be NaN
        
        # Check calculation is correct
        expected_return = (self.data['close'].iloc[1] / self.data['close'].iloc[0]) - 1
        assert abs(returns.iloc[1] - expected_return) < 1e-10
    
    def test_calculate_log_returns(self):
        """Test log returns calculation."""
        log_returns = calculate_log_returns(self.data['close'])
        
        assert isinstance(log_returns, pd.Series)
        assert len(log_returns) == len(self.data)
        assert pd.isna(log_returns.iloc[0])
        
        # Log returns should be smaller than simple returns for small changes
        simple_returns = calculate_returns(self.data['close'])
        # For small returns, log returns ≈ simple returns
        diff = abs(log_returns.iloc[1:10] - simple_returns.iloc[1:10])
        assert diff.max() < 0.01  # Should be very close for small returns
    
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation."""
        sma_10 = calculate_sma(self.data['close'], window=10)
        
        assert isinstance(sma_10, pd.Series)
        assert len(sma_10) == len(self.data)
        
        # First 9 values should be NaN
        assert pd.isna(sma_10.iloc[:9]).all()
        
        # 10th value should equal mean of first 10 values
        expected_sma = self.data['close'].iloc[:10].mean()
        assert abs(sma_10.iloc[9] - expected_sma) < 1e-10
    
    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation."""
        ema_10 = calculate_ema(self.data['close'], window=10)
        
        assert isinstance(ema_10, pd.Series)
        assert len(ema_10) == len(self.data)
        
        # EMA should not have NaN values after the first value
        assert not pd.isna(ema_10.iloc[1:]).any()
        
        # EMA should be more responsive than SMA
        sma_10 = calculate_sma(self.data['close'], window=10)
        # This is a general property but not always true
    
    def test_calculate_atr(self):
        """Test Average True Range calculation."""
        atr = calculate_atr(self.data['high'], self.data['low'], self.data['close'], window=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(self.data)
        
        # First 13 values should be NaN (window=14)
        assert pd.isna(atr.iloc[:13]).all()
        assert not pd.isna(atr.iloc[13])
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
    
    def test_calculate_rsi(self):
        """Test Relative Strength Index calculation."""
        rsi = calculate_rsi(self.data['close'], window=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(self.data)
        
        # RSI values should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
        
        # Should have some variability
        assert valid_rsi.std() > 0
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = calculate_macd(self.data['close'])
        
        for series in [macd_line, signal_line, histogram]:
            assert isinstance(series, pd.Series)
            assert len(series) == len(self.data)
        
        # Histogram should equal MACD - Signal
        valid_indices = ~(pd.isna(macd_line) | pd.isna(signal_line))
        macd_valid = macd_line[valid_indices]
        signal_valid = signal_line[valid_indices]
        histogram_valid = histogram[valid_indices]
        
        diff = abs(histogram_valid - (macd_valid - signal_valid))
        assert diff.max() < 1e-10
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = calculate_bollinger_bands(
            self.data['close'], window=20, num_std=2
        )
        
        for band in [upper, middle, lower]:
            assert isinstance(band, pd.Series)
            assert len(band) == len(self.data)
        
        # Middle band should equal SMA
        sma_20 = calculate_sma(self.data['close'], window=20)
        valid_indices = ~pd.isna(middle)
        
        diff = abs(middle[valid_indices] - sma_20[valid_indices])
        assert diff.max() < 1e-10
        
        # Upper should be above middle, middle above lower
        valid_data = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert (upper[valid_data] >= middle[valid_data]).all()
        assert (middle[valid_data] >= lower[valid_data]).all()
    
    def test_calculate_stochastic_oscillator(self):
        """Test Stochastic Oscillator calculation."""
        k_percent, d_percent = calculate_stochastic_oscillator(
            self.data['high'], self.data['low'], self.data['close'],
            k_window=14, d_window=3
        )
        
        for series in [k_percent, d_percent]:
            assert isinstance(series, pd.Series)
            assert len(series) == len(self.data)
        
        # Values should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_calculate_williams_r(self):
        """Test Williams %R calculation."""
        williams_r = calculate_williams_r(
            self.data['high'], self.data['low'], self.data['close'], window=14
        )
        
        assert isinstance(williams_r, pd.Series)
        assert len(williams_r) == len(self.data)
        
        # Values should be between -100 and 0
        valid_wr = williams_r.dropna()
        assert (valid_wr >= -100).all()
        assert (valid_wr <= 0).all()


class TestMicrostructureFeatures:
    """Test microstructure feature calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_obs = 100
        
        # Create realistic order book data
        mid_price = 4500
        spread = 0.25
        
        self.order_book = pd.DataFrame({
            'bid_price': mid_price - spread/2 + np.random.normal(0, 0.1, n_obs),
            'ask_price': mid_price + spread/2 + np.random.normal(0, 0.1, n_obs),
            'bid_size': np.random.randint(100, 1000, n_obs),
            'ask_size': np.random.randint(100, 1000, n_obs),
            'trade_price': mid_price + np.random.normal(0, 0.2, n_obs),
            'trade_size': np.random.randint(1, 100, n_obs),
            'trade_direction': np.random.choice([-1, 1], n_obs)
        }, index=pd.date_range('2023-01-01', periods=n_obs, freq='1s'))
        
        # Ensure ask > bid
        self.order_book['ask_price'] = np.maximum(
            self.order_book['ask_price'],
            self.order_book['bid_price'] + 0.01
        )
    
    def test_calculate_spread(self):
        """Test bid-ask spread calculation."""
        spread = calculate_spread(
            self.order_book['bid_price'],
            self.order_book['ask_price']
        )
        
        assert isinstance(spread, pd.Series)
        assert len(spread) == len(self.order_book)
        
        # Spread should be positive
        assert (spread > 0).all()
        
        # Spread should equal ask - bid
        expected_spread = self.order_book['ask_price'] - self.order_book['bid_price']
        diff = abs(spread - expected_spread)
        assert diff.max() < 1e-10
    
    def test_calculate_microprice(self):
        """Test microprice calculation."""
        microprice = calculate_microprice(
            self.order_book['bid_price'],
            self.order_book['bid_size'],
            self.order_book['ask_price'],
            self.order_book['ask_size']
        )
        
        assert isinstance(microprice, pd.Series)
        assert len(microprice) == len(self.order_book)
        
        # Microprice should be between bid and ask
        assert (microprice >= self.order_book['bid_price']).all()
        assert (microprice <= self.order_book['ask_price']).all()
    
    def test_calculate_queue_imbalance(self):
        """Test queue imbalance calculation."""
        queue_imbalance = calculate_queue_imbalance(
            self.order_book['bid_size'],
            self.order_book['ask_size']
        )
        
        assert isinstance(queue_imbalance, pd.Series)
        assert len(queue_imbalance) == len(self.order_book)
        
        # Queue imbalance should be between -1 and 1
        assert (queue_imbalance >= -1).all()
        assert (queue_imbalance <= 1).all()
    
    def test_calculate_order_flow_imbalance(self):
        """Test order flow imbalance calculation."""
        ofi = calculate_order_flow_imbalance(
            self.order_book['bid_price'],
            self.order_book['bid_size'],
            self.order_book['ask_price'],
            self.order_book['ask_size']
        )
        
        assert isinstance(ofi, pd.Series)
        assert len(ofi) == len(self.order_book)
        
        # First value should be NaN (requires previous values)
        assert pd.isna(ofi.iloc[0])
    
    def test_calculate_vwap(self):
        """Test Volume Weighted Average Price calculation."""
        vwap = calculate_vwap(
            self.order_book['trade_price'],
            self.order_book['trade_size']
        )
        
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(self.order_book)
        
        # VWAP should be finite and positive
        valid_vwap = vwap.dropna()
        assert np.isfinite(valid_vwap).all()
        assert (valid_vwap > 0).all()
    
    def test_calculate_twap(self):
        """Test Time Weighted Average Price calculation."""
        twap = calculate_twap(self.order_book['trade_price'], window=10)
        
        assert isinstance(twap, pd.Series)
        assert len(twap) == len(self.order_book)
        
        # TWAP should be simple moving average of prices
        sma = calculate_sma(self.order_book['trade_price'], window=10)
        valid_indices = ~(pd.isna(twap) | pd.isna(sma))
        
        diff = abs(twap[valid_indices] - sma[valid_indices])
        assert diff.max() < 1e-10
    
    def test_calculate_price_impact(self):
        """Test price impact calculation."""
        price_impact = calculate_price_impact(
            self.order_book['trade_price'],
            self.order_book['trade_size'],
            self.order_book['bid_price'],
            self.order_book['ask_price']
        )
        
        assert isinstance(price_impact, pd.Series)
        assert len(price_impact) == len(self.order_book)
        
        # Price impact should be finite
        valid_impact = price_impact.dropna()
        assert np.isfinite(valid_impact).all()


class TestTimeFeatures:
    """Test time-based feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create timestamps covering different times of day and days of week
        self.timestamps = pd.date_range(
            start='2023-01-01 09:30:00',
            end='2023-01-07 16:00:00',
            freq='1min',
            tz='America/New_York'
        )
        
        self.data = pd.DataFrame({
            'price': np.random.uniform(4500, 4600, len(self.timestamps))
        }, index=self.timestamps)
    
    def test_extract_time_of_day_features(self):
        """Test time of day feature extraction."""
        features = extract_time_of_day_features(self.timestamps)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.timestamps)
        
        # Should contain hour and minute features
        assert 'hour' in features.columns
        assert 'minute' in features.columns
        
        # Hour should be 0-23, minute should be 0-59
        assert features['hour'].min() >= 0
        assert features['hour'].max() <= 23
        assert features['minute'].min() >= 0
        assert features['minute'].max() <= 59
        
        # Should contain sinusoidal encodings
        assert 'hour_sin' in features.columns
        assert 'hour_cos' in features.columns
        assert 'minute_sin' in features.columns
        assert 'minute_cos' in features.columns
        
        # Sin and cos values should be between -1 and 1
        for col in ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']:
            assert features[col].min() >= -1
            assert features[col].max() <= 1
    
    def test_extract_day_of_week_features(self):
        """Test day of week feature extraction."""
        features = extract_day_of_week_features(self.timestamps)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.timestamps)
        
        # Should contain day of week
        assert 'day_of_week' in features.columns
        
        # Day of week should be 0-6
        assert features['day_of_week'].min() >= 0
        assert features['day_of_week'].max() <= 6
        
        # Should contain sinusoidal encoding
        assert 'day_of_week_sin' in features.columns
        assert 'day_of_week_cos' in features.columns
    
    def test_extract_session_features(self):
        """Test trading session feature extraction."""
        features = extract_session_features(self.timestamps)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.timestamps)
        
        # Should contain session-related features
        expected_features = [
            'is_market_open',
            'time_from_open',
            'time_to_close',
            'session_progress'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
        
        # is_market_open should be boolean
        assert features['is_market_open'].dtype == bool
        
        # Session progress should be between 0 and 1
        session_progress = features['session_progress'].dropna()
        assert (session_progress >= 0).all()
        assert (session_progress <= 1).all()
    
    def test_is_market_hours(self):
        """Test market hours detection."""
        # Test during market hours
        market_time = pd.Timestamp('2023-01-03 10:00:00', tz='America/New_York')
        assert is_market_hours(market_time)
        
        # Test before market open
        before_market = pd.Timestamp('2023-01-03 08:00:00', tz='America/New_York')
        assert not is_market_hours(before_market)
        
        # Test after market close
        after_market = pd.Timestamp('2023-01-03 17:00:00', tz='America/New_York')
        assert not is_market_hours(after_market)
        
        # Test weekend
        weekend = pd.Timestamp('2023-01-07 10:00:00', tz='America/New_York')
        assert not is_market_hours(weekend)
    
    def test_get_time_from_open(self):
        """Test time from market open calculation."""
        # Test at market open
        open_time = pd.Timestamp('2023-01-03 09:30:00', tz='America/New_York')
        assert get_time_from_open(open_time) == 0
        
        # Test 1 hour after open
        one_hour_after = pd.Timestamp('2023-01-03 10:30:00', tz='America/New_York')
        assert get_time_from_open(one_hour_after) == 60  # minutes
        
        # Test outside market hours
        after_close = pd.Timestamp('2023-01-03 17:00:00', tz='America/New_York')
        assert get_time_from_open(after_close) is None
    
    def test_get_time_to_close(self):
        """Test time to market close calculation."""
        # Test at market close
        close_time = pd.Timestamp('2023-01-03 16:00:00', tz='America/New_York')
        assert get_time_to_close(close_time) == 0
        
        # Test 1 hour before close
        one_hour_before = pd.Timestamp('2023-01-03 15:00:00', tz='America/New_York')
        assert get_time_to_close(one_hour_before) == 60  # minutes


class TestFeaturePipeline:
    """Test feature engineering pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create sample market data
        dates = pd.date_range(
            start='2023-01-01 09:30:00',
            end='2023-01-01 16:00:00',
            freq='1min',
            tz='America/New_York'
        )
        
        base_price = 4500
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price + np.cumsum(returns * base_price)
        
        self.market_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(0, 1.0, len(dates))),
            'low': prices - np.abs(np.random.normal(0, 1.0, len(dates))),
            'close': prices,
            'volume': np.random.randint(100, 1000, len(dates)),
            'bid_price': prices - 0.125,
            'ask_price': prices + 0.125,
            'bid_size': np.random.randint(100, 1000, len(dates)),
            'ask_size': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure OHLC constraints
        self.market_data['high'] = np.maximum(
            self.market_data['high'],
            np.maximum(self.market_data['open'], self.market_data['close'])
        )
        self.market_data['low'] = np.minimum(
            self.market_data['low'],
            np.minimum(self.market_data['open'], self.market_data['close'])
        )
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = {
            'technical': {
                'sma_windows': [10, 20],
                'ema_windows': [12, 26],
                'rsi_window': 14,
                'atr_window': 14
            },
            'microstructure': {
                'calculate_spread': True,
                'calculate_imbalance': True,
                'vwap_window': 20
            },
            'time': {
                'time_of_day': True,
                'day_of_week': True,
                'session_features': True
            }
        }
        
        pipeline = FeaturePipeline(config)
        
        assert pipeline.config == config
        assert hasattr(pipeline, 'technical_config')
        assert hasattr(pipeline, 'microstructure_config')
        assert hasattr(pipeline, 'time_config')
    
    def test_pipeline_transform(self):
        """Test full pipeline transformation."""
        config = {
            'technical': {
                'sma_windows': [10, 20],
                'rsi_window': 14
            },
            'microstructure': {
                'calculate_spread': True,
                'calculate_imbalance': True
            },
            'time': {
                'time_of_day': True,
                'session_features': True
            }
        }
        
        pipeline = FeaturePipeline(config)
        features = pipeline.transform(self.market_data)
        
        # Should return a DataFrame
        assert isinstance(features, pd.DataFrame)
        
        # Should have same index as input data
        assert len(features) == len(self.market_data)
        pd.testing.assert_index_equal(features.index, self.market_data.index)
        
        # Should contain expected technical features
        assert 'sma_10' in features.columns
        assert 'sma_20' in features.columns
        assert 'rsi_14' in features.columns
        
        # Should contain expected microstructure features
        assert 'spread' in features.columns
        assert 'queue_imbalance' in features.columns
        
        # Should contain expected time features
        assert 'hour' in features.columns
        assert 'is_market_open' in features.columns
    
    def test_pipeline_fit_transform(self):
        """Test pipeline fit and transform."""
        config = {
            'technical': {'sma_windows': [10]},
            'microstructure': {'calculate_spread': True},
            'time': {'time_of_day': True}
        }
        
        pipeline = FeaturePipeline(config)
        
        # Split data for fit/transform test
        split_idx = len(self.market_data) // 2
        train_data = self.market_data.iloc[:split_idx]
        test_data = self.market_data.iloc[split_idx:]
        
        # Fit on training data
        pipeline.fit(train_data)
        
        # Transform test data
        features = pipeline.transform(test_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(test_data)
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline handling of missing data."""
        # Introduce some missing values
        data_with_nans = self.market_data.copy()
        data_with_nans.loc[data_with_nans.index[10:15], 'close'] = np.nan
        data_with_nans.loc[data_with_nans.index[20:25], 'volume'] = np.nan
        
        config = {
            'technical': {'sma_windows': [10]},
            'microstructure': {'calculate_spread': True}
        }
        
        pipeline = FeaturePipeline(config)
        features = pipeline.transform(data_with_nans)
        
        # Should still return a DataFrame
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data_with_nans)
    
    def test_pipeline_feature_selection(self):
        """Test pipeline feature selection."""
        config = {
            'technical': {
                'sma_windows': [10, 20, 50],
                'ema_windows': [12, 26],
                'rsi_window': 14,
                'atr_window': 14
            },
            'feature_selection': {
                'max_features': 10,
                'selection_method': 'variance'
            }
        }
        
        pipeline = FeaturePipeline(config)
        features = pipeline.transform(self.market_data)
        
        # Should limit number of features if selection is enabled
        if 'feature_selection' in config:
            assert features.shape[1] <= config['feature_selection']['max_features']
    
    def test_pipeline_normalization(self):
        """Test pipeline feature normalization."""
        config = {
            'technical': {'sma_windows': [10]},
            'normalization': {
                'method': 'standardize',
                'fit_on_train': True
            }
        }
        
        pipeline = FeaturePipeline(config)
        
        # Fit and transform
        pipeline.fit(self.market_data)
        features = pipeline.transform(self.market_data)
        
        # If normalization is applied, features should have specific properties
        if 'normalization' in config and config['normalization']['method'] == 'standardize':
            # Features should have approximately zero mean and unit variance
            # (allowing for some numerical precision and missing values)
            numeric_features = features.select_dtypes(include=[np.number])
            for col in numeric_features.columns:
                valid_values = numeric_features[col].dropna()
                if len(valid_values) > 1:
                    assert abs(valid_values.mean()) < 0.1  # Close to zero
                    assert abs(valid_values.std() - 1) < 0.1  # Close to one


class TestPolygonCompatibility:
    """Test feature pipeline compatibility with Polygon data format."""

    def setup_method(self):
        """Set up test fixtures with Polygon-style data."""
        np.random.seed(42)

        # Create sample Polygon OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        base_price = 4500
        returns = np.random.normal(0, 0.001, 100)
        prices = base_price + np.cumsum(returns * base_price)

        self.polygon_ohlcv = pd.DataFrame({
            'timestamp': dates,  # Polygon uses timestamp column
            'open': prices + np.random.normal(0, 0.5, 100),
            'high': prices + np.abs(np.random.normal(0, 1.0, 100)),
            'low': prices - np.abs(np.random.normal(0, 1.0, 100)),
            'close': prices,
            'volume': np.random.randint(100, 1000, 100),
            'vwap': prices + np.random.normal(0, 0.1, 100),  # Polygon VWAP
            'transactions': np.random.randint(10, 100, 100)  # Polygon transactions
        })

        # Create sample Polygon quote data
        self.polygon_quotes = pd.DataFrame({
            'timestamp': dates,
            'bid_price': prices - 0.125,
            'bid_size': np.random.randint(100, 1000, 100),
            'ask_price': prices + 0.125,
            'ask_size': np.random.randint(100, 1000, 100),
            'bid_exchange': np.random.randint(1, 20, 100),
            'ask_exchange': np.random.randint(1, 20, 100)
        })

        # Create sample Polygon trade data
        self.polygon_trades = pd.DataFrame({
            'timestamp': dates,
            'price': prices + np.random.normal(0, 0.2, 100),
            'size': np.random.randint(1, 100, 100),
            'exchange': np.random.randint(1, 20, 100),
            'conditions': [None] * 100,
            'trade_id': [f'trade_{i}' for i in range(100)]
        })

    def test_polygon_data_detection(self):
        """Test automatic detection of Polygon data format."""
        config = {'data_source': 'auto'}
        pipeline = FeaturePipeline(config)

        # Test OHLCV detection
        data_source = pipeline._detect_data_source(self.polygon_ohlcv)
        assert data_source == 'polygon'

        # Test quote detection
        data_source = pipeline._detect_data_source(self.polygon_quotes)
        assert data_source == 'polygon'

        # Test trade detection
        data_source = pipeline._detect_data_source(self.polygon_trades)
        assert data_source == 'polygon'

    def test_polygon_column_mapping(self):
        """Test column mapping for Polygon data."""
        config = {'data_source': 'auto'}
        pipeline = FeaturePipeline(config)

        # Test OHLCV mapping
        mapped_data = pipeline._map_columns(self.polygon_ohlcv, 'ohlcv')
        assert 'close' in mapped_data.columns
        assert 'volume' in mapped_data.columns
        assert 'vwap' in mapped_data.columns
        assert isinstance(mapped_data.index, pd.DatetimeIndex)

        # Test quote mapping
        mapped_quotes = pipeline._map_columns(self.polygon_quotes, 'quotes')
        assert 'bid_price' in mapped_quotes.columns
        assert 'ask_price' in mapped_quotes.columns
        assert 'bid_size' in mapped_quotes.columns
        assert 'ask_size' in mapped_quotes.columns

    def test_polygon_vwap_usage(self):
        """Test that Polygon VWAP is used when available."""
        config = {
            'data_source': 'auto',
            'microstructure': {
                'calculate_vwap': True
            },
            'polygon': {
                'features': {
                    'use_vwap_column': True
                }
            }
        }
        pipeline = FeaturePipeline(config)

        # Transform data
        features = pipeline.transform(self.polygon_ohlcv)

        # VWAP feature should be present
        assert 'vwap' in features.columns

        # Check that VWAP values match the input VWAP (since we're using it directly)
        original_vwap = self.polygon_ohlcv['vwap']
        feature_vwap = features['vwap']

        # They should be very close (allowing for any processing)
        diff = abs(original_vwap - feature_vwap).dropna()
        assert diff.max() < 1e-10

    def test_polygon_quality_checks(self):
        """Test Polygon-specific data quality checks."""
        config = {
            'data_source': 'auto',
            'polygon': {
                'quality_checks': {
                    'enabled': True
                }
            }
        }
        pipeline = FeaturePipeline(config)

        # Test with clean data
        features = pipeline.transform(self.polygon_ohlcv)
        assert isinstance(features, pd.DataFrame)

        # Test with problematic VWAP data
        bad_data = self.polygon_ohlcv.copy()
        bad_data.loc[10, 'vwap'] = bad_data.loc[10, 'low'] - 1  # VWAP below low

        # Should still process but log warnings
        features = pipeline.transform(bad_data)
        assert isinstance(features, pd.DataFrame)

    def test_polygon_timestamp_handling(self):
        """Test timestamp handling for Polygon data."""
        config = {'data_source': 'auto'}
        pipeline = FeaturePipeline(config)

        # Test millisecond timestamps (Polygon aggregates)
        ms_timestamps = pd.DataFrame({
            'timestamp': [1640995200000, 1640995260000, 1640995320000],  # milliseconds
            'close': [4500, 4501, 4502],
            'volume': [100, 101, 102]
        })

        mapped_data = pipeline._map_columns(ms_timestamps, 'ohlcv')
        assert isinstance(mapped_data.index, pd.DatetimeIndex)

        # Test nanosecond timestamps (Polygon quotes/trades)
        ns_timestamps = pd.DataFrame({
            'timestamp': [1640995200000000000, 1640995260000000000, 1640995320000000000],  # nanoseconds
            'bid_price': [4499, 4500, 4501],
            'ask_price': [4501, 4502, 4503],
            'bid_size': [100, 101, 102],
            'ask_size': [100, 101, 102]
        })

        mapped_quotes = pipeline._map_columns(ns_timestamps, 'quotes')
        assert isinstance(mapped_quotes.index, pd.DatetimeIndex)

    def test_backward_compatibility_databento(self):
        """Test backward compatibility with Databento data format."""
        # Create Databento-style data
        dates = pd.date_range('2023-01-01', periods=50, freq='1min')
        base_price = 4500

        databento_data = pd.DataFrame({
            'timestamp': dates,
            'open': base_price + np.random.normal(0, 1, 50),
            'high': base_price + np.abs(np.random.normal(0, 2, 50)),
            'low': base_price - np.abs(np.random.normal(0, 2, 50)),
            'close': base_price + np.random.normal(0, 1, 50),
            'volume': np.random.randint(100, 1000, 50),
            'bid': base_price - 0.125 + np.random.normal(0, 0.1, 50),  # Databento uses 'bid'/'ask'
            'ask': base_price + 0.125 + np.random.normal(0, 0.1, 50),
            'bid_size': np.random.randint(100, 1000, 50),
            'ask_size': np.random.randint(100, 1000, 50)
        })

        config = {'data_source': 'auto'}
        pipeline = FeaturePipeline(config)

        # Should detect as Databento
        data_source = pipeline._detect_data_source(databento_data)
        assert data_source == 'databento'

        # Should still work
        features = pipeline.transform(databento_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(databento_data)

    def test_polygon_feature_extraction(self):
        """Test complete feature extraction pipeline with Polygon data."""
        config = {
            'data_source': 'auto',
            'technical': {
                'sma_windows': [10, 20],
                'rsi_window': 14,
                'calculate_atr': True
            },
            'microstructure': {
                'calculate_spread': True,
                'calculate_vwap': True,
                'calculate_queue_imbalance': True
            },
            'time': {
                'extract_time_of_day': True,
                'extract_session_features': True
            },
            'polygon': {
                'features': {
                    'use_vwap_column': True
                },
                'quality_checks': {
                    'enabled': True
                }
            }
        }

        pipeline = FeaturePipeline(config)
        features = pipeline.transform(self.polygon_ohlcv)

        # Check technical features
        assert 'sma_10' in features.columns
        assert 'sma_20' in features.columns
        assert 'rsi_14' in features.columns
        assert 'atr' in features.columns

        # Check microstructure features
        assert 'spread' in features.columns
        assert 'vwap' in features.columns

        # Check time features
        assert 'hour' in features.columns
        assert 'is_market_open' in features.columns

        # Should have reasonable number of features
        assert features.shape[1] > 10