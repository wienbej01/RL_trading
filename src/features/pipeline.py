"""
Feature engineering pipeline for the RL trading system.

This module provides a comprehensive feature engineering pipeline
that combines technical indicators, microstructure features, and time-based features.
Supports both Polygon and Databento data formats with automatic column mapping.
"""
import json
import re
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import logging
import os
from pathlib import Path

# Import technical indicator functions
from .technical_indicators import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_stochastic_oscillator,
    calculate_williams_r, calculate_returns, calculate_log_returns,
    calculate_vol_of_vol, calculate_sma_slope, calculate_obv
)
from ta.trend import ADXIndicator
from .microstructure_features import (
    calculate_spread, calculate_microprice, calculate_queue_imbalance,
    calculate_order_flow_imbalance, calculate_vwap, calculate_twap,
    calculate_price_impact, calculate_fvg,
    calculate_ofi_best, compute_bar_imbalance, compute_signed_vol_delta,
    compute_spread_bps, compute_quote_intensity,
)
from .time_features import (
    extract_time_of_day_features, extract_day_of_week_features,
    extract_session_features, is_market_hours, get_time_from_open,
    get_time_to_close
)
try:
    from .microstructure_ohlcv import OHLCV_MICRO  # preferred shim path
except Exception:
    from .microstructure import OHLCV_MICRO  # fallback

from ..utils.logging import get_logger
from ..utils.feat_cache import feature_cache_path, save_features, load_features, augment_cfg_hash
from .packs import (
    resolve_feature_pack,
    LAST_CURATED_CACHE_TOKEN,
    resolve_curated_pack,
    MICROSTRUCTURE_OHLCV,
)


class FeaturePipeline:
    """
    Feature engineering pipeline for the RL trading system.

    This class provides a unified interface for extracting and transforming
    features from market data, including technical indicators, microstructure
    features, and time-based features. Supports both Polygon and Databento data formats.
    """

    # Column mapping for different data sources
    COLUMN_MAPPINGS = {
        'polygon_ohlcv': {
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'vwap': 'vwap',
            'transactions': 'transactions'
        },
        'polygon_quotes': {
            'timestamp': 'timestamp',
            'bid_price': 'bid_price',
            'bid_size': 'bid_size',
            'ask_price': 'ask_price',
            'ask_size': 'ask_size',
            'bid_exchange': 'bid_exchange',
            'ask_exchange': 'ask_exchange'
        },
        'polygon_trades': {
            'timestamp': 'timestamp',
            'price': 'price',
            'size': 'size',
            'exchange': 'exchange',
            'conditions': 'conditions',
            'trade_id': 'trade_id'
        },
        'databento_ohlcv': {
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        },
        'databento_quotes': {
            'timestamp': 'timestamp',
            'bid_price': 'bid',
            'bid_size': 'bid_size',
            'ask_price': 'ask',
            'ask_size': 'ask_size'
        },
        'databento_trades': {
            'timestamp': 'timestamp',
            'price': 'price',
            'size': 'size'
        }
    }
    LEAKAGE_PATTERN = re.compile(r'\b(lead|future|forward|next|target|t\+?\d+)\b', re.IGNORECASE)

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature pipeline.

        Args:
            config: Configuration dictionary specifying which features to extract
        """
        self.config = config
        self.technical_config = config.get('technical', {})
        self.microstructure_config = config.get('microstructure', {})
        self.time_config = config.get('time', {})
        self.normalization_config = config.get('normalization', {})
        self.feature_selection_config = config.get('feature_selection', {})
        # New feature groups/toggles
        self.vpa_config = config.get('vpa', {})
        self.ict_config = config.get('ict', {})
        self.vol_config = config.get('volatility', {})
        self.smt_config = config.get('smt', {})
        # Optional: regime tags (rolling, leak‑safe)
        self.regime_config = config.get('regime', {})

        # Data source detection and column mapping
        self.data_source = config.get('data_source', 'auto')
        self.column_mapping = {}

        # Polygon-specific configuration
        self.polygon_config = config.get('polygon', {})
        self.use_polygon_vwap = self.polygon_config.get('features', {}).get('use_vwap_column', True)
        self.polygon_quality_checks = self.polygon_config.get('quality_checks', {}).get('enabled', True)

        self.is_fitted = False
        self.scaler = None
        self.feature_selector: Optional[SelectKBest] = None
        self.selected_features = None
        self.feature_scores_: Optional[pd.DataFrame] = None
        self.normalization_stats: Optional[Dict[str, Any]] = None
        self._audit_reports: List[pd.DataFrame] = []

        # Get logger
        self.logger = get_logger(__name__)

    def _check_for_leakage(self, ohlcv: pd.DataFrame, features: pd.DataFrame):
        """Check for data leakage by comparing timestamps."""

        def _validate_index(frame: pd.DataFrame, label: str) -> None:
            idx = frame.index
            if isinstance(idx, pd.MultiIndex) and 'timestamp' in idx.names:
                idx = idx.get_level_values('timestamp')
            if not isinstance(idx, pd.DatetimeIndex):
                return
            if not idx.is_monotonic_increasing:
                raise ValueError(f"{label} index is not monotonic increasing.")

        has_ticker_col = 'ticker' in ohlcv.columns
        has_ticker_index = isinstance(ohlcv.index, pd.MultiIndex) and 'ticker' in ohlcv.index.names

        if has_ticker_col or has_ticker_index:
            if has_ticker_col:
                ohlcv_groups = ohlcv.groupby('ticker')
            else:
                tickers = ohlcv.index.get_level_values('ticker').unique()
                ohlcv_groups = ((t, ohlcv.xs(t, level='ticker')) for t in tickers)
            for ticker, group in ohlcv_groups:
                _validate_index(group, f"OHLCV ({ticker})")
        else:
            _validate_index(ohlcv, "OHLCV")

        has_feat_ticker_col = 'ticker' in features.columns
        has_feat_ticker_index = isinstance(features.index, pd.MultiIndex) and 'ticker' in features.index.names

        if has_feat_ticker_col or has_feat_ticker_index:
            if has_feat_ticker_col:
                feature_groups = features.groupby('ticker')
            else:
                tickers = features.index.get_level_values('ticker').unique()
                feature_groups = ((t, features.xs(t, level='ticker')) for t in tickers)
            for ticker, group in feature_groups:
                _validate_index(group, f"Features ({ticker})")
        else:
            _validate_index(features, "Features")

        # Compare max timestamps on aligned axes (use overall maxima).
        ohlcv_max = None
        feature_max = None
        for frame, store in ((ohlcv, 'ohlcv_max'), (features, 'feature_max')):
            idx = frame.index
            if isinstance(idx, pd.MultiIndex) and 'timestamp' in idx.names:
                idx = idx.get_level_values('timestamp')
            if isinstance(idx, pd.DatetimeIndex):
                if store == 'ohlcv_max':
                    ohlcv_max = idx.max()
                else:
                    feature_max = idx.max()
        if ohlcv_max is not None and feature_max is not None and ohlcv_max < feature_max:
            raise ValueError("Feature leakage detected: features have future timestamps.")
    def _detect_data_source(self, data: pd.DataFrame) -> str:
        """
        Detect the data source based on column names and data characteristics.

        Args:
            data: Input DataFrame

        Returns:
            Data source identifier ('polygon' or 'databento')
        """
        if self.data_source != 'auto':
            return self.data_source

        # Check for Polygon-specific columns
        polygon_indicators = ['vwap', 'transactions', 'bid_exchange', 'ask_exchange', 'conditions', 'trade_id']

        # Check for Databento-specific column patterns
        databento_indicators = ['bid', 'ask']  # Databento uses 'bid'/'ask' instead of 'bid_price'/'ask_price'

        polygon_score = sum(1 for col in polygon_indicators if col in data.columns)
        databento_score = sum(1 for col in databento_indicators if col in data.columns)

        if polygon_score > databento_score:
            return 'polygon'
        elif databento_score > polygon_score:
            return 'databento'
        else:
            # Default to polygon if unclear
            self.logger.info("Data source unclear, defaulting to polygon")
            return 'polygon'

    def _map_columns(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Map columns to standard format based on detected data source.

        Args:
            data: Input DataFrame
            data_type: Type of data ('ohlcv', 'quotes', 'trades')

        Returns:
            DataFrame with standardized column names
        """
        data_source = self._detect_data_source(data)
        mapping_key = f"{data_source}_{data_type}"

        if mapping_key not in self.COLUMN_MAPPINGS:
            self.logger.warning(f"No column mapping found for {mapping_key}, using data as-is")
            return data

        mapping = self.COLUMN_MAPPINGS[mapping_key]
        self.column_mapping = mapping

        # Create a copy to avoid modifying original data
        mapped_data = data.copy()

        # Apply column mapping
        for standard_col, source_col in mapping.items():
            if source_col in mapped_data.columns and standard_col != source_col:
                mapped_data[standard_col] = mapped_data[source_col]
                self.logger.debug(f"Mapped {source_col} to {standard_col}")

        # Ensure timestamp is the index if it's a column
        if 'timestamp' in mapped_data.columns:
            if not isinstance(mapped_data.index, pd.DatetimeIndex):
                # Convert timestamp to datetime index (Polygon aggregates: ms; quotes/trades: ns)
                if data_source == 'polygon':
                    unit = 'ns' if data_type in ('quotes', 'trades') else 'ms'
                else:
                    unit = 'ns'
                mapped_data['timestamp'] = pd.to_datetime(mapped_data['timestamp'], unit=unit)
                mapped_data.set_index('timestamp', inplace=True)
                self.logger.debug("Set timestamp as DatetimeIndex")

        return mapped_data

    def _merge_external_vix(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally merge an external VIX series as a feature if configured.

        Expects a parquet/csv at path with a DatetimeIndex and a column named one of
        ['vix', 'close', 'VIX']. Produces columns 'vix' and 'vix_z' aligned to
        features' index (America/New_York), forward-filled and lagged by 1 bar to
        avoid lookahead.
        """
        try:
            ext = self.vol_config.get('external_vix_path') if isinstance(self.vol_config, dict) else None
            if not ext:
                return features
            import pandas as _pd
            p = str(ext)
            if p.lower().endswith('.parquet'):
                v = _pd.read_parquet(p)
            else:
                v = _pd.read_csv(p)
            # Normalize index
            if 'timestamp' in v.columns:
                v['timestamp'] = _pd.to_datetime(v['timestamp'], utc=True, errors='coerce')
                v = v.loc[v['timestamp'].notna()].set_index('timestamp')
            if not isinstance(v.index, _pd.DatetimeIndex):
                v.index = _pd.to_datetime(v.index, utc=True, errors='coerce')
            if v.index.tz is None:
                v.index = v.index.tz_localize('UTC')
            v = v.sort_index().tz_convert('America/New_York')
            # Choose a usable column
            col = None
            for c in ['vix', 'VIX', 'close', 'Close']:
                if c in v.columns:
                    col = c
                    break
            if col is None and v.shape[1] >= 1:
                col = v.columns[0]
            if col is None:
                return features
            ser = _pd.to_numeric(v[col], errors='coerce').rename('vix')
            # Align to feature index, ffill, and lag by 1 bar
            aligned = ser.reindex(features.index).ffill().shift(1)
            out = features.copy()
            out['vix'] = aligned.astype(float)
            try:
                z = (aligned - aligned.rolling(60, min_periods=10).mean()) / (aligned.rolling(60, min_periods=10).std() + 1e-6)
                out['vix_z'] = z.astype(float)
            except Exception:
                pass
            return out
        except Exception:
            # Never break pipeline on optional external data
            return features

    def _validate_polygon_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Polygon-specific data quality checks and cleaning.

        Args:
            data: Input DataFrame with Polygon data

        Returns:
            Cleaned DataFrame
        """
        original_len = len(data)
        issues_found = []

        # Check for Polygon-specific data quality issues
        if 'vwap' in data.columns:
            # VWAP should be reasonable relative to OHLC prices
            vwap_anomalies = (
                (data['vwap'] < data['low'] * 0.95) |  # VWAP too low
                (data['vwap'] > data['high'] * 1.05)   # VWAP too high
            )
            if vwap_anomalies.any():
                issues_found.append(f"Found {vwap_anomalies.sum()} VWAP anomalies")
                # Flag but don't remove - VWAP can be outside OHLC range in some cases

        if 'transactions' in data.columns:
            # Transactions should be non-negative
            invalid_transactions = data['transactions'] < 0
            if invalid_transactions.any():
                issues_found.append(f"Found {invalid_transactions.sum()} negative transaction counts")
                data = data[~invalid_transactions]

        if 'bid_exchange' in data.columns and 'ask_exchange' in data.columns:
            # Check for cross-market quotes (same exchange on both sides)
            cross_market = data['bid_exchange'] == data['ask_exchange']
            if cross_market.any():
                issues_found.append(f"Found {cross_market.sum()} potential cross-market quotes")

        # Check for timestamp consistency
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for duplicate timestamps
            duplicates = data.index.duplicated()
            if duplicates.any():
                issues_found.append(f"Found {duplicates.sum()} duplicate timestamps")
                # Keep first occurrence
                data = data[~duplicates]

            # Check for gaps in millisecond data (for high-frequency data)
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dt.total_seconds() * 1000
                large_gaps = time_diffs > 60000  # 1 minute gap
                if large_gaps.sum() > 0:
                    issues_found.append(f"Found {large_gaps.sum()} gaps > 1 minute")

        cleaned_len = len(data)
        if cleaned_len < original_len:
            issues_found.append(f"Removed {original_len - cleaned_len} rows during cleaning")

        if issues_found:
            self.logger.info(f"Data quality issues found: {', '.join(issues_found)}")

        return data

    def fit(self, data: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit the feature pipeline on data.

        Args:
            data: Data to fit on

        Returns:
            Self
        """
        self.feature_scores_ = None
        self.selected_features = None
        self.normalization_stats = None
        self._audit_reports = []
        # Multi-ticker aware fitting: if a 'ticker' column is present, compute features per ticker
        features: pd.DataFrame
        if 'ticker' in data.columns or (
            isinstance(data.index, pd.MultiIndex) and 'ticker' in data.index.names
        ):
            parts = []
            if 'ticker' in data.columns:
                grouped = data.groupby('ticker')
            else:
                grouped = data.groupby(level='ticker')
            for t, df in grouped:
                df_local = df.copy()
                # Map columns and validate per group
                df_local = self._map_columns(df_local, 'ohlcv')
                if self._detect_data_source(df_local) == 'polygon' and self.polygon_quality_checks:
                    df_local = self._validate_polygon_data_quality(df_local)
                f_local = self._extract_features(df_local)
                try:
                    f_local = f_local.drop(columns=['ticker'], errors='ignore')
                except Exception:
                    pass
                f_local['ticker'] = t
                parts.append(f_local)
            if parts:
                features = pd.concat(parts, axis=0)
                try:
                    features = features.sort_index()
                except Exception:
                    pass
            else:
                features = pd.DataFrame(index=data.index)
        else:
            # Single-ticker path (original behavior)
            data = self._map_columns(data, 'ohlcv')  # Assume OHLCV for fitting
            data_source = self._detect_data_source(data)
            if data_source == 'polygon' and self.polygon_quality_checks:
                data = self._validate_polygon_data_quality(data)
            features = self._extract_features(data)

        # Apply normalization if configured
        if self.normalization_config:
            # Do not normalize non-numeric/ticker label
            features = self._normalize_features(features)

        # Apply feature selection if configured (support both 'method' and 'selection_method')
        if self.feature_selection_config and (
            'method' in self.feature_selection_config or 'selection_method' in self.feature_selection_config
        ):
            features = self._select_features(features)

        # If input carried a 'timestamp' column (Polygon-style) and did not use it as index,
        # restore a simple RangeIndex so external comparisons by position align.
        if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            try:
                features = features.reset_index(drop=True)
            except Exception:
                pass

        self.is_fitted = True
        # Optionally merge external VIX features
        try:
            features = self._merge_external_vix(features)
        except Exception:
            pass
        if self.selected_features is None:
            self.selected_features = [col for col in features.columns if col != 'ticker']
        try:
            self.audit_features(features, stage='fit')
        except ValueError as exc:
            self.logger.error("Feature audit failed during fit: %s", exc)
            raise
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            data: Data to transform

        Returns:
            Transformed features
        """
        # For test compatibility, fit if not fitted
        if not self.is_fitted:
            self.logger.warning("Pipeline not fitted, fitting on transform data")
            self.fit(data)

        # Track original timestamp-column presence before mapping
        _orig_ts_col_present = ('timestamp' in data.columns)
        _orig_index_is_dt = isinstance(data.index, pd.DatetimeIndex)
        # Multi-ticker aware transform
        if 'ticker' in data.columns or (
            isinstance(data.index, pd.MultiIndex) and 'ticker' in data.index.names
        ):
            parts = []
            if 'ticker' in data.columns:
                grouped = data.groupby('ticker')
            else:
                grouped = data.groupby(level='ticker')
            for t, df in grouped:
                df_local = df.copy()
                df_local = self._map_columns(df_local, 'ohlcv')
                if self._detect_data_source(df_local) == 'polygon' and self.polygon_quality_checks:
                    df_local = self._validate_polygon_data_quality(df_local)
                # Optional cache: attempt HIT before computing
                f_local = None
                try:
                    if bool(self.config.get('enable_caching', False)):
                        run_name = str(self.config.get('cache_run', 'default'))
                        # Window keys
                        idx = df_local.index if isinstance(df_local.index, pd.DatetimeIndex) else pd.to_datetime(df_local.index, errors='coerce')
                        start = pd.Timestamp(idx.min()) if len(idx) else pd.Timestamp(0, unit='s')
                        end = pd.Timestamp(idx.max()) if len(idx) else pd.Timestamp(0, unit='s')
                        # Approximate pack identifiers
                        feature_pack = ','.join(self.config.get('packs', [])) if bool(self.config.get('use_pack', False)) else 'manual'
                        feature_screen_run = str(self.config.get('feature_screen_run', self.config.get('override_list_path', 'NA')))
                        base_hash = f"{feature_pack}|{feature_screen_run}|{t}|{start}|{end}"
                        # If user provided a manual selected_features list, we can use it for hash
                        final_hint = None
                        try:
                            sel = self.feature_selection_config.get('selected_features', None)
                            if isinstance(sel, (list, tuple)):
                                final_hint = [str(x) for x in sel]
                        except Exception:
                            pass
                        cfg_hash = augment_cfg_hash(
                            base_hash,
                            run_name=run_name,
                            features=final_hint,
                            micro_module_path=Path('src/features/microstructure_ohlcv.py'),
                            curated_token=LAST_CURATED_CACHE_TOKEN,
                        )
                        cache_path = feature_cache_path(
                            run_name=run_name,
                            ticker=str(t),
                            split=f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                            cfg_hash=cfg_hash,
                        )
                        if cache_path.exists():
                            try:
                                self.logger.info(f"Feature cache HIT -> {cache_path}")
                            except Exception:
                                pass
                            f_local = load_features(cache_path)
                except Exception:
                    f_local = None
                if f_local is None:
                    f_local = self._extract_features(df_local)
                    # Save to cache if enabled
                    try:
                        if bool(self.config.get('enable_caching', False)):
                            # Build hash with the actual resolved features list to tie data to key
                            run_name = str(self.config.get('cache_run', 'default'))
                            idx = df_local.index if isinstance(df_local.index, pd.DatetimeIndex) else pd.to_datetime(df_local.index, errors='coerce')
                            start = pd.Timestamp(idx.min()) if len(idx) else pd.Timestamp(0, unit='s')
                            end = pd.Timestamp(idx.max()) if len(idx) else pd.Timestamp(0, unit='s')
                            feature_pack = ','.join(self.config.get('packs', [])) if bool(self.config.get('use_pack', False)) else 'manual'
                            feature_screen_run = str(self.config.get('feature_screen_run', self.config.get('override_list_path', 'NA')))
                            base_hash = f"{feature_pack}|{feature_screen_run}|{t}|{start}|{end}"
                            final_cols = [str(c) for c in f_local.columns]
                            cfg_hash = augment_cfg_hash(
                                base_hash,
                                run_name=run_name,
                                features=final_cols,
                                micro_module_path=Path('src/features/microstructure_ohlcv.py'),
                                curated_token=LAST_CURATED_CACHE_TOKEN,
                            )
                            cache_path = feature_cache_path(
                                run_name=run_name,
                                ticker=str(t),
                                split=f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                                cfg_hash=cfg_hash,
                            )
                            save_features(f_local, cache_path)
                    except Exception:
                        pass
                try:
                    f_local = f_local.drop(columns=['ticker'], errors='ignore')
                except Exception:
                    pass
                f_local['ticker'] = t
                parts.append(f_local)
            if parts:
                features = pd.concat(parts, axis=0)
                try:
                    features = features.sort_index()
                except Exception:
                    pass
            else:
                features = pd.DataFrame(index=data.index)
        else:
            data = self._map_columns(data, 'ohlcv')
            data_source = self._detect_data_source(data)
            if data_source == 'polygon' and self.polygon_quality_checks:
                data = self._validate_polygon_data_quality(data)
            # Optional cache HIT before compute (single-ticker or unlabeled)
            features = None
            try:
                if bool(self.config.get('enable_caching', False)):
                    run_name = str(self.config.get('cache_run', 'default'))
                    idx = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index, errors='coerce')
                    start = pd.Timestamp(idx.min()) if len(idx) else pd.Timestamp(0, unit='s')
                    end = pd.Timestamp(idx.max()) if len(idx) else pd.Timestamp(0, unit='s')
                    t = str(self.config.get('single_ticker', 'UNKNOWN'))
                    feature_pack = ','.join(self.config.get('packs', [])) if bool(self.config.get('use_pack', False)) else 'manual'
                    feature_screen_run = str(self.config.get('feature_screen_run', self.config.get('override_list_path', 'NA')))
                    base_hash = f"{feature_pack}|{feature_screen_run}|{t}|{start}|{end}"
                    final_hint = None
                    try:
                        sel = self.feature_selection_config.get('selected_features', None)
                        if isinstance(sel, (list, tuple)):
                            final_hint = [str(x) for x in sel]
                    except Exception:
                        pass
                    cfg_hash = augment_cfg_hash(
                        base_hash,
                        run_name=run_name,
                        features=final_hint,
                        micro_module_path=Path('src/features/microstructure_ohlcv.py'),
                        curated_token=LAST_CURATED_CACHE_TOKEN,
                    )
                    cache_path = feature_cache_path(
                        run_name=run_name,
                        ticker=str(t),
                        split=f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                        cfg_hash=cfg_hash,
                    )
                    if cache_path.exists():
                        try:
                            self.logger.info(f"Feature cache HIT -> {cache_path}")
                        except Exception:
                            pass
                        features = load_features(cache_path)
            except Exception:
                features = None
            if features is None:
                features = self._extract_features(data)
                # Save to cache if enabled
                try:
                    if bool(self.config.get('enable_caching', False)):
                        run_name = str(self.config.get('cache_run', 'default'))
                        idx = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index, errors='coerce')
                        start = pd.Timestamp(idx.min()) if len(idx) else pd.Timestamp(0, unit='s')
                        end = pd.Timestamp(idx.max()) if len(idx) else pd.Timestamp(0, unit='s')
                        t = str(self.config.get('single_ticker', 'UNKNOWN'))
                        feature_pack = ','.join(self.config.get('packs', [])) if bool(self.config.get('use_pack', False)) else 'manual'
                        feature_screen_run = str(self.config.get('feature_screen_run', self.config.get('override_list_path', 'NA')))
                        base_hash = f"{feature_pack}|{feature_screen_run}|{t}|{start}|{end}"
                        final_cols = [str(c) for c in features.columns]
                        cfg_hash = augment_cfg_hash(
                            base_hash,
                            run_name=run_name,
                            features=final_cols,
                            micro_module_path=Path('src/features/microstructure_ohlcv.py'),
                            curated_token=LAST_CURATED_CACHE_TOKEN,
                        )
                        cache_path = feature_cache_path(
                            run_name=run_name,
                            ticker=str(t),
                            split=f"{start:%Y-%m-%d}_{end:%Y-%m-%d}",
                            cfg_hash=cfg_hash,
                        )
                        save_features(features, cache_path)
                except Exception:
                    pass

        # Apply normalization if configured
        if self.normalization_config:
            features = self._normalize_features(features)

        # Apply feature selection if configured (also recognize 'selection_method')
        if self.feature_selection_config and (
            'method' in self.feature_selection_config or 'selection_method' in self.feature_selection_config
        ):
            features = self._select_features(features)
        # If original input had a 'timestamp' column and was not indexed by it, restore RangeIndex
        if _orig_ts_col_present and not _orig_index_is_dt:
            try:
                features = features.reset_index(drop=True)
            except Exception:
                pass
        # Optionally merge external VIX features after selection to ensure retention
        try:
            features = self._merge_external_vix(features)
        except Exception:
            pass
        if self.selected_features is None:
            self.selected_features = [col for col in features.columns if col != 'ticker']

        if self.selected_features:
            missing = [col for col in self.selected_features if col not in features.columns]
            for col in missing:
                features[col] = np.nan
            ordered = [c for c in self.selected_features if c in features.columns and c != 'ticker']
            if 'ticker' in features.columns:
                features = features.loc[:, ['ticker'] + ordered]
            else:
                features = features.loc[:, ordered]

        try:
            self.audit_features(features, stage='transform')
        except ValueError as exc:
            self.logger.error("Feature audit failed during transform: %s", exc)
            raise

        return features
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit pipeline and transform data.
        
        Args:
            data: Training data
            
        Returns:
            Transformed features
        """
        transformed_data = self.fit(data).transform(data)
        self._check_for_leakage(data, transformed_data)
        return transformed_data
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on configuration."""
        features = pd.DataFrame(index=data.index)

        # Optional: resolve requested features list for this extraction
        resolved: List[str] | None = None
        try:
            # Use curated resolved list if present
            from .packs import LAST_CURATED_RESOLVED  # local import
            if LAST_CURATED_RESOLVED:
                resolved = list(LAST_CURATED_RESOLVED)
        except Exception:
            pass
        # Manual selected_features override
        try:
            sel = self.feature_selection_config.get('selected_features', None)
            if isinstance(sel, (list, tuple)) and sel:
                resolved = list(sel)
        except Exception:
            pass
        # microstructure.names fallback for proxies
        try:
            names = (self.config.get('microstructure', {}) or {}).get('names', None)
            if isinstance(names, (list, tuple)) and names:
                resolved = list(dict.fromkeys(list(resolved or []) + list(names)))
        except Exception:
            pass
        # If L1 quotes are absent, remap L1 names in resolved to OHLCV proxies
        try:
            HAS_L1 = all(c in data.columns for c in ("bid_price", "ask_price", "bid_size", "ask_size"))
            if resolved is not None and not HAS_L1:
                L1_TO_OHLCV = {
                    "ofi_best": "ofi_proxy",
                    "spread_bps": "spread_bps_hl",
                    "quote_intensity": "quote_intensity_proxy",
                    "queue_imbalance": "queue_imbalance_proxy",
                }
                resolved = [L1_TO_OHLCV.get(str(f), str(f)) for f in resolved]
                # Validate that no L1 names remain in resolved list
                L1_NAMES = {"ofi_best", "spread_bps", "quote_intensity", "queue_imbalance"}
                remaining_l1 = [f for f in resolved if f in L1_NAMES]
                if remaining_l1:
                    raise ValueError(f"L1 feature names still present after remap when HAS_L1=False: {remaining_l1}")
        except Exception:
            pass
        
        # Extract technical indicators
        if 'technical' in self.config:
            tech_config = self.config['technical']
            
            # Calculate returns
            if 'calculate_returns' in tech_config and tech_config['calculate_returns']:
                features['returns'] = calculate_returns(data['close'])
            
            # Calculate log returns
            if 'calculate_log_returns' in tech_config and tech_config['calculate_log_returns']:
                features['log_returns'] = calculate_log_returns(data['close'])
            
            # Calculate SMAs
            if 'sma_windows' in tech_config:
                for window in tech_config['sma_windows']:
                    features[f'sma_{window}'] = calculate_sma(data['close'], window)
            
            # Calculate EMAs
            if 'ema_windows' in tech_config:
                for window in tech_config['ema_windows']:
                    features[f'ema_{window}'] = calculate_ema(data['close'], window)
            
            # Calculate ATR
            if 'calculate_atr' in tech_config and tech_config['calculate_atr']:
                features['atr'] = calculate_atr(data['high'], data['low'], data['close'])
            
            # Calculate RSI
            if 'calculate_rsi' in tech_config and tech_config['calculate_rsi']:
                rsi_window = tech_config.get('rsi_window', 14)
                features[f'rsi_{rsi_window}'] = calculate_rsi(data['close'], window=rsi_window)
            elif 'rsi_window' in tech_config:
                # Handle test case where only rsi_window is specified
                rsi_window = tech_config['rsi_window']
                features[f'rsi_{rsi_window}'] = calculate_rsi(data['close'], window=rsi_window)
            
            # Calculate MACD
            if 'calculate_macd' in tech_config and tech_config['calculate_macd']:
                macd_config = tech_config.get('macd_config', {})
                macd_res = calculate_macd(
                    data['close'],
                    fast_period=macd_config.get('fast_period', 12),
                    slow_period=macd_config.get('slow_period', 26),
                    signal_period=macd_config.get('signal_period', 9)
                )
                try:
                    macd_line, signal_line, histogram = macd_res  # tuple form
                except Exception:
                    # dict form fallback
                    macd_line = macd_res['macd']
                    signal_line = macd_res['signal']
                    histogram = macd_res['histogram']
                features['macd'] = macd_line
                features['macd_signal'] = signal_line
                features['macd_histogram'] = histogram
                features['macd_line'] = macd_line - signal_line
            
            # Calculate Bollinger Bands
            if 'calculate_bollinger_bands' in tech_config and tech_config['calculate_bollinger_bands']:
                bb_config = tech_config.get('bollinger_config', {})
                bb_res = calculate_bollinger_bands(
                    data['close'],
                    window=bb_config.get('window', 20),
                    num_std=bb_config.get('num_std', 2)
                )
                try:
                    upper_band, middle_band, lower_band = bb_res  # tuple form
                except Exception:
                    upper_band = bb_res['upper']
                    middle_band = bb_res['middle']
                    lower_band = bb_res['lower']
                features['bb_upper'] = upper_band
                features['bb_middle'] = middle_band
                features['bb_lower'] = lower_band
                # width derived if needed
                try:
                    features['bb_width'] = (upper_band - lower_band) / (middle_band.replace(0, np.nan))
                except Exception:
                    pass
            
            # Calculate Stochastic Oscillator
            if 'calculate_stochastic' in tech_config and tech_config['calculate_stochastic']:
                stoch_config = tech_config.get('stochastic_config', {})
                stoch_res = calculate_stochastic_oscillator(
                    data['high'], data['low'], data['close'],
                    k_period=stoch_config.get('k_period', 14),
                    d_period=stoch_config.get('d_period', 3)
                )
                try:
                    k_ser, d_ser = stoch_res
                except Exception:
                    k_ser = stoch_res['k']
                    d_ser = stoch_res['d']
                features['stoch_k'] = k_ser
                features['stoch_d'] = d_ser

            # Calculate Williams %R
            if 'calculate_williams_r' in tech_config and tech_config['calculate_williams_r']:
                williams_window = tech_config.get('williams_window', 14)
                features['williams_r'] = calculate_williams_r(
                    data['high'], data['low'], data['close'], williams_window
                )

            # Calculate ADX for trend strength
            if 'calculate_adx' in tech_config and tech_config['calculate_adx']:
                adx_window = tech_config.get('adx_window', 14)
                adx_ind = ADXIndicator(data['high'], data['low'], data['close'], window=adx_window)
                features['adx'] = adx_ind.adx()

        # Extract microstructure features
        if 'microstructure' in self.config:
            micro_config = self.config['microstructure']
            
            # Calculate spread
            if 'calculate_spread' in micro_config and micro_config['calculate_spread']:
                # Check if we have bid/ask columns or use high/low as fallback
                if 'bid_price' in data.columns and 'ask_price' in data.columns:
                    features['spread'] = calculate_spread(data['bid_price'], data['ask_price'])
                else:
                    features['spread'] = calculate_spread(data['high'], data['low'])
            
            # Calculate microprice
            if 'calculate_microprice' in micro_config and micro_config['calculate_microprice']:
                if 'bid_price' in data.columns and 'bid_size' in data.columns and \
                   'ask_price' in data.columns and 'ask_size' in data.columns:
                    features['microprice'] = calculate_microprice(
                        data['bid_price'], data['bid_size'], 
                        data['ask_price'], data['ask_size']
                    )
            
            # Calculate queue imbalance
            if 'calculate_queue_imbalance' in micro_config and micro_config['calculate_queue_imbalance']:
                if 'bid_size' in data.columns and 'ask_size' in data.columns:
                    features['queue_imbalance'] = calculate_queue_imbalance(
                        data['bid_size'], data['ask_size']
                    )
            elif 'calculate_imbalance' in micro_config and micro_config['calculate_imbalance']:
                # Handle test case where calculate_imbalance is used instead
                if 'bid_size' in data.columns and 'ask_size' in data.columns:
                    features['queue_imbalance'] = calculate_queue_imbalance(
                        data['bid_size'], data['ask_size']
                    )
            
            # Calculate order flow imbalance (z-scored) and ofi_best (raw best-level proxy)
            if 'calculate_order_flow_imbalance' in micro_config and micro_config['calculate_order_flow_imbalance']:
                need = {'bid_price','bid_size','ask_price','ask_size'}
                if need.issubset(set(data.columns)):
                    features['order_flow_imbalance'] = calculate_order_flow_imbalance(
                        data['bid_price'], data['bid_size'],
                        data['ask_price'], data['ask_size']
                    )
                    try:
                        features['ofi_best'] = calculate_ofi_best(
                            data['bid_price'], data['bid_size'], data['ask_price'], data['ask_size']
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to compute ofi_best: {e}")
                elif 'resolved' in locals() and resolved is not None:
                    # If L1 data is not available but ofi_proxy is requested, compute it
                    if "ofi_proxy" in resolved and "ofi_proxy" not in features.columns:
                        try:
                            features['ofi_proxy'] = OHLCV_MICRO['ofi_proxy'](data)
                        except Exception as e:
                            self.logger.warning(f"Failed to compute ofi_proxy: {e}")
                else:
                    missing = sorted(list(need.difference(set(data.columns))))
                    # Only log warning if we're actually trying to compute L1 features
                    if 'calculate_order_flow_imbalance' in micro_config and micro_config['calculate_order_flow_imbalance']:
                        self.logger.warning(f"Cannot compute ofi_best/OFI: missing raw fields {missing}")
            
            # Calculate VWAP
            if 'calculate_vwap' in micro_config and micro_config['calculate_vwap']:
                # Use Polygon VWAP if available and configured to do so
                polygon_vwap = None
                if self.use_polygon_vwap and 'vwap' in data.columns:
                    polygon_vwap = data['vwap']
                features['vwap'] = calculate_vwap(data['close'], data['volume'], polygon_vwap)

            # Additional microstructure/features by name when possible
            try:
                # bar_imbalance, signed_vol_delta
                if 'open' in data.columns and 'close' in data.columns and 'volume' in data.columns:
                    features['bar_imbalance'] = compute_bar_imbalance(data['open'], data['close'], data['volume'])
                    features['signed_vol_delta'] = compute_signed_vol_delta(data['close'], data['volume'])
                # spread_bps from quotes or OHLC
                if {'bid_price','ask_price'}.issubset(data.columns):
                    features['spread_bps'] = compute_spread_bps(bid=data['bid_price'], ask=data['ask_price'])
                elif {'high','low','close'}.issubset(data.columns):
                    try:
                        features['spread_bps'] = OHLCV_MICRO['spread_bps_hl'](data)
                        self.logger.warning("Using HL proxy for spread_bps; level-1 quotes missing")
                    except Exception as e:
                        self.logger.warning(f"Failed to compute spread_bps via HL proxy: {e}")
                else:
                    # Only log warning if we're actually trying to compute spread_bps
                    if 'calculate_spread' in micro_config and micro_config['calculate_spread']:
                        self.logger.warning("Cannot compute spread_bps: missing both bid/ask and high/low/close for fallback")
                # quote_intensity
                if 'transactions' in data.columns or 'volume' in data.columns:
                    # Only compute quote_intensity if it's in resolved features or explicitly requested
                    if 'resolved' in locals() and resolved is not None and "quote_intensity" in resolved:
                        features['quote_intensity'] = compute_quote_intensity(data.get('transactions'), data.get('volume'))
                    elif resolved and 'quote_intensity' in resolved:
                        # Fallback if resolved is not available in locals
                        features['quote_intensity'] = compute_quote_intensity(data.get('transactions'), data.get('volume'))
                # queue_imbalance already computed above when bid/ask sizes present
            except Exception:
                pass
            
            # Calculate TWAP
            if 'calculate_twap' in micro_config and micro_config['calculate_twap']:
                twap_window = micro_config.get('twap_window', 5)
                features['twap'] = calculate_twap(data['close'], window=twap_window)
            
            # Calculate price impact
            if 'calculate_price_impact' in micro_config and micro_config['calculate_price_impact']:
                if 'volume' in data.columns:
                    # Check if we have bid/ask data for proper price impact calculation
                    if 'bid_price' in data.columns and 'ask_price' in data.columns:
                        features['price_impact'] = calculate_price_impact(
                            data['close'], data['volume'], data['bid_price'], data['ask_price']
                        )
                    else:
                        # Fallback: simple price impact approximation using close vs open
                        features['price_impact'] = (data['close'] - data['open']) / data['open']

            # Resolve and compute explicit OHLCV microstructure names if requested
            try:
                requested_names = micro_config.get('names', None)
                # Also allow computing OHLCV proxies if present in resolved list
                if 'resolved' in locals() and resolved is not None:
                    extra = [n for n in resolved if isinstance(n, str) and n in OHLCV_MICRO]
                    if extra:
                        if requested_names is None:
                            requested_names = list(extra)
                        else:
                            requested_names = list(dict.fromkeys(list(requested_names) + extra))
                if requested_names is not None:
                    if not isinstance(requested_names, (list, tuple)):
                        raise ValueError("microstructure.names must be a list of feature names")
                    known = set(OHLCV_MICRO.keys())
                    unknown = [n for n in requested_names if n not in known]
                    if unknown:
                        raise ValueError(
                            f"Unknown microstructure feature name(s): {unknown}. "
                            f"Known OHLCV-only names: {sorted(known)}"
                        )
                    # Compute each requested OHLCV-only micro feature without relying on bid/ask
                    # Use only standard OHLCV columns; functions will validate required fields.
                    df_ohlcv = data.copy()
                    for name in requested_names:
                        func = OHLCV_MICRO[name]
                        try:
                            ser = func(df_ohlcv)
                            # Attach with the exact requested name (override if exists)
                            features[name] = ser
                        except Exception as e:
                            raise ValueError(f"Failed to compute microstructure feature '{name}': {e}")
            except Exception as e:
                # Surface configuration/unknown-name errors clearly
                if isinstance(e, ValueError):
                    raise
                # Non-fatal runtime errors should not kill the whole pipeline
                self.logger.warning(f"microstructure.names computation skipped: {e}")

        # If we have a resolved list and proxies are requested (and L1 may be absent), compute them
        try:
            if 'resolved' in locals() and resolved is not None:
                for name in resolved:
                    if isinstance(name, str) and name in OHLCV_MICRO and name not in features.columns:
                        try:
                            features[name] = OHLCV_MICRO[name](data)
                        except Exception:
                            pass
                # Assert required proxies if they were requested
                required = [f for f in ("ofi_proxy", "signed_vol_delta") if f in resolved]
                missing = [f for f in required if f not in features.columns]
                if missing:
                    raise AssertionError(f"Requested micro proxies missing from features: {missing}")
        except Exception as e:
            try:
                self.logger.warning(str(e))
            except Exception:
                pass
        
        # Extract time features
        if 'time' in self.config:
            time_config = self.config['time']
            
            # Extract time of day features
            if 'extract_time_of_day' in time_config and time_config['extract_time_of_day']:
                time_features = extract_time_of_day_features(data.index)
                features = pd.concat([features, time_features], axis=1)
            elif 'time_of_day' in time_config and time_config['time_of_day']:
                # Handle test case where time_of_day is used instead
                time_features = extract_time_of_day_features(data.index)
                features = pd.concat([features, time_features], axis=1)
            
            # Extract day of week features
            if 'extract_day_of_week' in time_config and time_config['extract_day_of_week']:
                dow_features = extract_day_of_week_features(data.index)
                features = pd.concat([features, dow_features], axis=1)
            
            # Extract session features
            if 'extract_session_features' in time_config and time_config['extract_session_features']:
                session_features = extract_session_features(data.index)
                features = pd.concat([features, session_features], axis=1)
            elif 'session_features' in time_config and time_config['session_features']:
                # Handle test case where session_features is used instead
                session_features = extract_session_features(data.index)
                features = pd.concat([features, session_features], axis=1)

        # One-hot encode session_phase
        if 'session_phase' in features.columns:
            features = pd.concat([
                features.drop(columns=['session_phase']),
                pd.get_dummies(features['session_phase'], prefix='session')
            ], axis=1)
        
        # Add VIX data and derived features
        if 'vix' in self.config and self.config['vix'].get('enabled', False):
            vix_path = self.config['vix'].get('path', 'data/external/vix.parquet')
            try:
                if os.path.exists(vix_path):
                    vix_df = pd.read_parquet(vix_path)
                    # Normalize index to NY-tz DatetimeIndex
                    if not isinstance(vix_df.index, pd.DatetimeIndex):
                        if 'timestamp' in vix_df.columns:
                            vix_df.index = pd.to_datetime(vix_df['timestamp'], utc=True, errors='coerce')
                        else:
                            vix_df.index = pd.to_datetime(vix_df.index, utc=True, errors='coerce')
                    if vix_df.index.tz is None:
                        vix_df.index = vix_df.index.tz_localize('UTC')
                    vix_df.index = vix_df.index.tz_convert('America/New_York')
                    vix_df = vix_df.sort_index()
                    base = vix_df.copy()
                    # Resolve a close column under various common names
                    close_col = None
                    for cand in ['vix_close', 'close', 'Close', 'VIXCLS', 'value']:
                        if cand in base.columns:
                            close_col = cand
                            break
                    if close_col is not None:
                        base['vix_close'] = pd.to_numeric(base[close_col], errors='coerce')
                        base['vix_ret'] = base['vix_close'].pct_change()
                        base['vix_ma20'] = base['vix_close'].rolling(20, min_periods=5).mean()
                        base['vix_z20'] = (base['vix_close'] - base['vix_ma20']) / (base['vix_close'].rolling(20, min_periods=5).std() + 1e-9)
                        keep = [c for c in ['vix_close','vix_ret','vix_ma20','vix_z20'] if c in base.columns]
                        if keep:
                            base = base[keep]
                            base = base.reindex(features.index, method='ffill')
                            features = features.join(base)
                    else:
                        self.logger.info("VIX file found but no recognizable close column; skipping VIX features.")
                else:
                    self.logger.info(f"VIX file not found at {vix_path}; skipping VIX features.")
            except Exception as e:
                self.logger.warning(f"VIX features join failed: {e}")

        # Volume–Price Analysis (VPA) and ICT-inspired features
        try:
            close = data['close']
            open_ = data['open'] if 'open' in data.columns else data['close'].shift(1)
            high = data['high'] if 'high' in data.columns else data['close']
            low = data['low'] if 'low' in data.columns else data['close']
            vol = data['volume'] if 'volume' in data.columns else pd.Series(0.0, index=data.index)

            tr = (high - low).abs()
            bar_range = (high - low).replace(0, np.nan)
            body = (close - open_)
            # Bar imbalance and effort/result
            features['bar_imbalance'] = (body / bar_range).clip(-1, 1).fillna(0.0)
            features['effort_result'] = (tr * vol).fillna(0.0)
            # Relative volume (simple rolling baseline)
            rv_window = self.technical_config.get('rvol_window', 390)
            features['rvol'] = (vol / (vol.rolling(rv_window, min_periods=30).mean() + 1e-9)).fillna(1.0)
            # Delta volume (up vs down bars)
            direction = np.sign(close.diff().fillna(0.0))
            features['delta_vol'] = (direction * vol).fillna(0.0)
            # Previous day reference levels and distances
            by_day = data.groupby(data.index.normalize())
            pdh = by_day['high'].transform('max').shift(1)
            pdl = by_day['low'].transform('min').shift(1)
            pdc = by_day['close'].transform('last').shift(1)
            features['dist_pdh_bp'] = (close - pdh)
            features['dist_pdl_bp'] = (close - pdl)
            features['dist_pdc_bp'] = (close - pdc)
            # Opening range (first 15 mins)
            minute = data.index.minute + data.index.hour * 60
            session_start_min = data.index.normalize().map(lambda d: 9*60+30)
            mins_from_open = minute - session_start_min
            is_or = (mins_from_open >= 0) & (mins_from_open < 15)
            or_high = by_day['high'].transform(lambda s: s[is_or.reindex(s.index, fill_value=False)].max())
            or_low = by_day['low'].transform(lambda s: s[is_or.reindex(s.index, fill_value=False)].min())
            features['dist_orh_bp'] = (close - or_high)
            features['dist_orl_bp'] = (close - or_low)
        except Exception:
            pass

        # Add Volatility of Volatility
        if 'vol_of_vol' in self.config.get('technical', {}) and self.config['technical']['vol_of_vol'].get('enabled', False):
            window = self.config['technical']['vol_of_vol'].get('window', 14)
            vol_window = self.config['technical']['vol_of_vol'].get('vol_window', 14)
            features['vol_of_vol'] = calculate_vol_of_vol(data['high'], data['low'], data['close'], window, vol_window)

        # Add SMA Slope
        if 'sma_slope' in self.config.get('technical', {}) and self.config['technical']['sma_slope'].get('enabled', False):
            window = self.config['technical']['sma_slope'].get('window', 20)
            slope_window = self.config['technical']['sma_slope'].get('slope_window', 5)
            features['sma_slope'] = calculate_sma_slope(data['close'], window, slope_window)

        # Add On-Balance Volume (OBV)
        if 'obv' in self.config.get('technical', {}) and self.config['technical']['obv'].get('enabled', False):
            # OBV requires 'close' and 'volume' columns
            if 'close' in data.columns and 'volume' in data.columns:
                features['obv'] = calculate_obv(data)
            else:
                self.logger.warning("OBV calculation skipped: 'close' or 'volume' column missing.")

        # Add Fair Value Gaps (FVG)
        if 'fvg' in self.config.get('microstructure', {}) and self.config['microstructure']['fvg'].get('enabled', False):
            features['fvg'] = calculate_fvg(data['high'], data['low'])
            # Rolling density (shifted to avoid lookahead)
            try:
                win = int(self.config.get('microstructure', {}).get('fvg', {}).get('density_window', 50))
                features['fvg_density'] = features['fvg'].rolling(win, min_periods=1).sum().shift(1)
            except Exception:
                pass

        # ICT features
        try:
            if self.ict_config.get('enabled', True):
                # Previous day mid (equilibrium) distance
                by_day = data.groupby(data.index.normalize())
                pdh = by_day['high'].transform('max').shift(1)
                pdl = by_day['low'].transform('min').shift(1)
                pdm = (pdh + pdl) / 2.0
                features['dist_pdm_mid'] = (data['close'] - pdm)

                # Opening range distances (first N minutes)
                or_minutes = int(self.ict_config.get('opening_range_minutes', 15))
                idx = data.index
                mins = idx.hour * 60 + idx.minute
                session_open_min = 9 * 60 + 30
                in_or = (mins >= session_open_min) & (mins < session_open_min + or_minutes)
                # Compute per-day ORH/ORL and map back
                orh = by_day.apply(lambda df: df.loc[in_or[df.index], 'high'].max() if not df.loc[in_or[df.index]].empty else pd.NA)
                orl = by_day.apply(lambda df: df.loc[in_or[df.index], 'low'].min() if not df.loc[in_or[df.index]].empty else pd.NA)
                features['dist_orh_bp'] = (data['close'] - idx.normalize().map(orh))
                features['dist_orl_bp'] = (data['close'] - idx.normalize().map(orl))

                # Displacement bar and density
                k_atr = float(self.ict_config.get('displacement_k_atr', 1.5))
                atr_series = features.get('atr', calculate_atr(data['high'], data['low'], data['close']))
                body = (data['close'] - data['open']).abs()
                wick = (data['high'] - data['low']) - body
                disp = (body > k_atr * (atr_series + 1e-9)) & ((wick / (body + 1e-9)) < 1.0)
                features['disp_bar'] = disp.astype(int)
                features['disp_density'] = features['disp_bar'].rolling(50, min_periods=1).sum().shift(1)
        except Exception:
            pass

        # VPA enhancements
        try:
            if self.vpa_config.get('enabled', False):
                # Climax volume flag via RVOL threshold
                if 'rvol' in features.columns:
                    thr = float(self.vpa_config.get('climax_rvol_threshold', 3.0))
                    features['climax_vol'] = (features['rvol'] >= thr).astype(int)

                # Churn index and z-score
                tr = (data['high'] - data['low']).abs()
                churn = (data['volume'] / (tr.replace(0, pd.NA)))
                churn = churn.ffill().fillna(0.0)
                # Opt in to future behavior explicitly to avoid downcasting warning
                churn = churn.infer_objects(copy=False)
                features['churn'] = churn.astype(float)
                zwin = int(self.vpa_config.get('zscore_window', 100))
                mu = features['churn'].rolling(zwin, min_periods=10).mean()
                sd = features['churn'].rolling(zwin, min_periods=10).std()
                features['churn_z'] = ((features['churn'] - mu) / (sd + 1e-9)).fillna(0.0)

                # Imbalance persistence and direction EMA (shifted for causality)
                ret = data['close'].pct_change().fillna(0.0)
                sgn = np.sign(ret)
                persist_win = int(self.vpa_config.get('persistence_window', 10))
                features['imbalance_persist'] = pd.Series(sgn, index=data.index).rolling(persist_win, min_periods=1).mean().shift(1)
                span = int(self.vpa_config.get('direction_ema_span', 10))
                features['direction_ema'] = ret.ewm(span=span, adjust=False).mean().shift(1)

                # Intrabar volatility proxy
                features['intrabar_vol'] = ((data['high'] - data['low']) / data['close']).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                # VWAP derivatives (if vwap exists)
                if 'vwap' in data.columns:
                    features['dist_vwap'] = (data['close'] - data['vwap']).astype(float)
                    if 'atr' in features.columns:
                        features['dist_vwap_atr'] = (features['dist_vwap'] / (features['atr'] + 1e-6)).astype(float)
        except Exception:
            pass

        # ICT: Liquidity pools (equal highs/lows) proximity
        try:
            if self.ict_config.get('enabled', True):
                tol = float(self.ict_config.get('eq_tolerance_bp', 5.0))  # basis points tolerance
                win = int(self.ict_config.get('eq_window', 50))
                price = data['close']
                bps = lambda x: x * 1e-4
                eq_high = pd.Series(0, index=data.index, dtype=int)
                eq_low = pd.Series(0, index=data.index, dtype=int)
                # rolling detection of approximate equal highs/lows
                roll_high = data['high'].rolling(win, min_periods=5)
                roll_low = data['low'].rolling(win, min_periods=5)
                # mark if last two local highs within tolerance
                last_high = roll_high.max()
                last_low = roll_low.min()
                eq_high = ((last_high - data['high']).abs() <= bps(price)).astype(int)
                eq_low = ((data['low'] - last_low).abs() <= bps(price)).astype(int)
                features['eq_high_flag'] = eq_high
                features['eq_low_flag'] = eq_low
                # distance to last eq high/low levels in price terms
                last_eq_high = last_high.ffill()
                last_eq_low = last_low.ffill()
                features['dist_eq_high'] = (price - last_eq_high).fillna(0.0)
                features['dist_eq_low'] = (price - last_eq_low).fillna(0.0)
        except Exception:
            pass

        # VIX term-structure ratios if provided in VIX parquet
        try:
            if 'vix' in self.config and self.config['vix'].get('enabled', False):
                vix_path = self.config['vix'].get('path', 'data/external/vix.parquet')
                if os.path.exists(vix_path):
                    vix_df = pd.read_parquet(vix_path)
                    if not isinstance(vix_df.index, pd.DatetimeIndex):
                        if 'timestamp' in vix_df.columns:
                            vix_df.index = pd.to_datetime(vix_df['timestamp'], utc=True, errors='coerce')
                        else:
                            vix_df.index = pd.to_datetime(vix_df.index, utc=True, errors='coerce')
                    if vix_df.index.tz is None:
                        vix_df.index = vix_df.index.tz_localize('UTC')
                    vix_df.index = vix_df.index.tz_convert('America/New_York')
                    vix_df = vix_df.sort_index()
                    cols = {c.lower(): c for c in vix_df.columns}
                    have_9d = cols.get('vix9d') or cols.get('vix_9d')
                    have_1m = cols.get('vix') or cols.get('vix_1m') or cols.get('close')
                    have_3m = cols.get('vix3m') or cols.get('vix_3m')
                    base = pd.DataFrame(index=vix_df.index)
                    if have_1m:
                        base['vix_1m'] = pd.to_numeric(vix_df[have_1m], errors='coerce')
                    if have_9d:
                        base['vix_9d'] = pd.to_numeric(vix_df[have_9d], errors='coerce')
                    if have_3m:
                        base['vix_3m'] = pd.to_numeric(vix_df[have_3m], errors='coerce')
                    if not base.empty:
                        if 'vix_9d' in base and 'vix_1m' in base:
                            base['vix_9d_ratio'] = base['vix_9d'] / (base['vix_1m'] + 1e-9)
                        if 'vix_1m' in base and 'vix_3m' in base:
                            base['vix_1m_3m_ratio'] = base['vix_1m'] / (base['vix_3m'] + 1e-9)
                        base = base.reindex(features.index, method='ffill')
                        features = features.join(base.drop(columns=[c for c in ['vix_9d','vix_1m','vix_3m'] if c in base.columns]))
        except Exception:
            pass

        # SMT divergence features (intermarket divergence). Two modes supported:
        # 1) vs SPY: compare instrument momentum to SPY momentum
        # 2) SPY vs QQQ: overall market divergence context
        try:
            if self.smt_config.get('enabled', False):
                mom_span = int(self.smt_config.get('momentum_span', 5))
                # Helper to load benchmark close series from parquet
                def _load_close_series(p: str) -> pd.Series:
                    dfb = pd.read_parquet(p)
                    idx = dfb.index
                    if not isinstance(idx, pd.DatetimeIndex):
                        return pd.Series(dtype=float)
                    if idx.tz is None:
                        idx = idx.tz_localize('UTC').tz_convert('America/New_York')
                    close = dfb['close'] if 'close' in dfb.columns else dfb.iloc[:, 0]
                    s = pd.Series(close.values, index=idx).sort_index()
                    return s

                # 1) vs SPY if path provided
                spy_path = self.smt_config.get('paths', {}).get('SPY')
                if spy_path and os.path.exists(spy_path):
                    spy_close = _load_close_series(spy_path)
                    # align and compute short-horizon momentum (EMA of returns)
                    spy_ret = spy_close.pct_change().fillna(0.0)
                    spy_mom = spy_ret.ewm(span=mom_span, adjust=False).mean()
                    # primary instrument momentum
                    prim_ret = data['close'].pct_change().fillna(0.0)
                    prim_mom = prim_ret.ewm(span=mom_span, adjust=False).mean()
                    spy_mom = spy_mom.reindex(features.index).ffill()
                    prim_mom = prim_mom.reindex(features.index).ffill()
                    features['smt_vs_spy'] = (prim_mom - spy_mom).shift(1).fillna(0.0)

                # 2) SPY vs QQQ global divergence if both provided
                spy_path = self.smt_config.get('paths', {}).get('SPY')
                qqq_path = self.smt_config.get('paths', {}).get('QQQ')
                if spy_path and qqq_path and os.path.exists(spy_path) and os.path.exists(qqq_path):
                    s_spy = _load_close_series(spy_path)
                    s_qqq = _load_close_series(qqq_path)
                    r_spy = s_spy.pct_change().fillna(0.0)
                    r_qqq = s_qqq.pct_change().fillna(0.0)
                    m_spy = r_spy.ewm(span=mom_span, adjust=False).mean()
                    m_qqq = r_qqq.ewm(span=mom_span, adjust=False).mean()
                    m_spy = m_spy.reindex(features.index).ffill()
                    m_qqq = m_qqq.reindex(features.index).ffill()
                    features['smt_spy_qqq'] = (m_spy - m_qqq).shift(1).fillna(0.0)
        except Exception:
            pass

        # Levels: pivots, rolling support/resistance, day open levels
        try:
            levels_cfg = self.config.get('levels', {})
            if levels_cfg.get('enabled', False):
                idx = data.index
                day_idx = idx.normalize()
                # Daily OHLC
                daily_o = data['open'].groupby(day_idx).first()
                daily_h = data['high'].groupby(day_idx).max()
                daily_l = data['low'].groupby(day_idx).min()
                daily_c = data['close'].groupby(day_idx).last()

                # Current day open and prior day open
                curr_open_map = daily_o
                prior_open_map = daily_o.shift(1)
                features['dist_cdo_bp'] = (data['close'] - day_idx.map(curr_open_map)).astype(float)
                features['dist_pdo_bp'] = (data['close'] - day_idx.map(prior_open_map)).astype(float)

                # Gap from prior close
                prior_close_map = daily_c.shift(1)
                day_open_const = day_idx.map(curr_open_map)
                features['gap_open_prev_close'] = (day_open_const - day_idx.map(prior_close_map)).astype(float)

                # Pivots (use prior day's HLC if configured)
                if levels_cfg.get('pivot', True):
                    use_prior = str(levels_cfg.get('pivots_from', 'prior_day')).lower() == 'prior_day'
                    H = daily_h.shift(1) if use_prior else daily_h
                    L = daily_l.shift(1) if use_prior else daily_l
                    C = daily_c.shift(1) if use_prior else daily_c
                    PP = (H + L + C) / 3.0
                    R1 = 2 * PP - L
                    S1 = 2 * PP - H
                    R2 = PP + (H - L)
                    S2 = PP - (H - L)
                    features['dist_pp_bp'] = (data['close'] - day_idx.map(PP)).astype(float)
                    features['dist_r1_bp'] = (data['close'] - day_idx.map(R1)).astype(float)
                    features['dist_s1_bp'] = (data['close'] - day_idx.map(S1)).astype(float)
                    features['dist_r2_bp'] = (data['close'] - day_idx.map(R2)).astype(float)
                    features['dist_s2_bp'] = (data['close'] - day_idx.map(S2)).astype(float)

                # Rolling support/resistance (past-only)
                rw = int(levels_cfg.get('roll_window', 20))
                roll_max = data['high'].rolling(rw, min_periods=5).max().shift(1)
                roll_min = data['low'].rolling(rw, min_periods=5).min().shift(1)
                features['dist_rollmax_bp'] = (data['close'] - roll_max).astype(float)
                features['dist_rollmin_bp'] = (data['close'] - roll_min).astype(float)

                # Session VWAP distance (intraday from session open)
                try:
                    vol = data['volume'].clip(lower=0)
                    typical = data['close']
                    # Cumulative from session open
                    v_cum = vol.groupby(day_idx).cumsum()
                    pv_cum = (typical * vol).groupby(day_idx).cumsum()
                    session_vwap = (pv_cum / v_cum).replace([np.inf, -np.inf], np.nan)
                    features['dist_session_vwap'] = (data['close'] - session_vwap).astype(float)
                except Exception:
                    pass

                # Normalize some distances by ATR if configured
                if levels_cfg.get('use_atr_norm', True):
                    atr_series = features.get('atr', calculate_atr(data['high'], data['low'], data['close']))
                    eps = 1e-6
                    for col in ['dist_cdo_bp','dist_pdo_bp','gap_open_prev_close','dist_pp_bp','dist_r1_bp','dist_s1_bp','dist_r2_bp','dist_s2_bp','dist_rollmax_bp','dist_rollmin_bp','dist_session_vwap']:
                        if col in features.columns:
                            features[col + '_atr'] = (features[col] / (atr_series + eps)).astype(float)

                # True swings (confirmed pivots) with causal confirmation (shift by right window)
                try:
                    swings = levels_cfg.get('swings', {'left': 3, 'right': 3})
                    left = int(swings.get('left', 3))
                    right = int(swings.get('right', 3))
                    win = left + right + 1
                    rh = data['high'].rolling(win, center=True).apply(lambda x: float(np.argmax(x) == left), raw=False)
                    rl = data['low'].rolling(win, center=True).apply(lambda x: float(np.argmin(x) == left), raw=False)
                    swing_high_flag = rh.shift(right).fillna(0).astype(int)
                    swing_low_flag = rl.shift(right).fillna(0).astype(int)
                    features['swing_high_flag'] = swing_high_flag
                    features['swing_low_flag'] = swing_low_flag
                    # Last swing levels and distances
                    last_sh = data['high'].where(swing_high_flag == 1).ffill()
                    last_sl = data['low'].where(swing_low_flag == 1).ffill()
                    features['dist_last_swing_high'] = (data['close'] - last_sh).astype(float)
                    features['dist_last_swing_low'] = (data['close'] - last_sl).astype(float)
                    # Break-of-structure flags (buffer via ATR)
                    atr_series = features.get('atr', calculate_atr(data['high'], data['low'], data['close']))
                    buf = 0.1 * atr_series
                    bos_up = (data['close'] > (last_sh + buf)).astype(int)
                    bos_dn = (data['close'] < (last_sl - buf)).astype(int)
                    features['bos_up'] = bos_up
                    features['bos_down'] = bos_dn
                except Exception:
                    pass
        except Exception:
            pass
        
        # Optional low-variance filter first
        min_var = float(self.feature_selection_config.get('min_variance', 0.0))
        if min_var and min_var > 0.0:
            try:
                features = self._drop_low_variance(features, min_var)
            except Exception as e:
                self.logger.warning(f"Variance filtering skipped: {e}")

        # Optional correlation filtering to reduce redundancy
        corr_thresh = float(self.feature_selection_config.get('correlation_threshold', 0.0))
        if corr_thresh and corr_thresh > 0.0:
            try:
                features = self._drop_highly_correlated(features, corr_thresh)
            except Exception as e:
                self.logger.warning(f"Correlation filtering skipped: {e}")

        # Log the final feature count after all processing
        self.logger.info("Extracted %d features", len(features.columns))

        # Optional regime features (low-cost tags; causal rolling, t-1)
        try:
            if isinstance(self.regime_config, dict) and self.regime_config.get('enabled', False):
                vol_win = int(self.regime_config.get('vol_window', 60))
                trend_win = int(self.regime_config.get('trend_window', 60))
                # Rolling volatility on close returns
                r = data['close'].pct_change().fillna(0.0)
                vol = r.rolling(vol_win, min_periods=max(5, vol_win//5)).std().shift(1)
                features['regime_vol'] = vol.astype(float)
                # Tercile buckets (0,1,2) using expanding quantiles (causal)
                q1 = vol.expanding(min_periods=10).quantile(1/3)
                q2 = vol.expanding(min_periods=10).quantile(2/3)
                bucket = (vol > q2).astype(int) * 2 + ((vol > q1) & (vol <= q2)).astype(int)
                features['regime_vol_bucket'] = bucket.fillna(0).astype(int)
                # Trend slope via rolling linear regression proxy: EMA of returns
                trend = r.ewm(span=max(3, trend_win//6), adjust=False).mean().shift(1)
                features['regime_trend'] = trend.astype(float)
                features['regime_trend_sign'] = np.sign(trend).fillna(0).astype(int)
        except Exception:
            # Never break the pipeline on optional tags
            pass

        # Note: Warmup bars are kept to maintain alignment with OHLCV data
        # The training process will handle any necessary warmup period
        # max_lb = 120
        # features = features.iloc[max_lb:].copy()

        return features

    def _drop_highly_correlated(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Drop one of each pair of features whose absolute correlation exceeds threshold.

        Keeps earlier columns; removes later ones among highly correlated pairs.
        """
        if features.empty:
            return features
        X = features.select_dtypes(include=[np.number]).copy()
        # Fill NaNs to compute corr
        X = X.ffill().bfill()
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        pruned = features.drop(columns=[c for c in to_drop if c in features.columns])
        self.logger.info("Correlation filter dropped %d features (threshold=%.2f)", len(to_drop), threshold)
        return pruned

    def _drop_low_variance(self, features: pd.DataFrame, min_var: float = 1e-8) -> pd.DataFrame:
        """Drop numeric features with variance below min_var."""
        X = features.select_dtypes(include=[np.number])
        variances = X.var()
        to_keep = variances[variances >= min_var].index.tolist()
        dropped = [c for c in X.columns if c not in to_keep]
        pruned = features.drop(columns=dropped)
        self.logger.info("Variance filter dropped %d features (min_var=%.2e)", len(dropped), min_var)
        return pruned

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using the specified method.
        
        Args:
            features: DataFrame of features to normalize
            
        Returns:
            Normalized features
        """
        if not self.normalization_config:
            return features
            
        method = self.normalization_config.get('method', 'standardize')
        fit_on_train = self.normalization_config.get('fit_on_train', True)
        
        if method == 'standardize':
            if self.scaler is None or not fit_on_train:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                
                # Fit on numeric features only
                numeric_features = features.select_dtypes(include=[np.number])
                if len(numeric_features.columns) > 0:
                    # Fill NaN values with mean before fitting
                    numeric_features_filled = numeric_features.fillna(numeric_features.mean())
                    self.scaler.fit(numeric_features_filled)
                    self.normalization_stats = {
                        'method': 'standardize',
                        'columns': list(numeric_features.columns),
                        'mean': {col: float(val) for col, val in zip(numeric_features.columns, self.scaler.mean_)},
                        'scale': {col: float(val) for col, val in zip(numeric_features.columns, getattr(self.scaler, 'scale_', np.ones_like(self.scaler.mean_)))},
                    }
            
            # Transform numeric features
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 0:
                # Fill NaN values with mean before transforming
                numeric_features_filled = numeric_features.fillna(numeric_features.mean())
                normalized_values = self.scaler.transform(numeric_features_filled)
                features[numeric_features.columns] = normalized_values
                
        elif method == 'minmax':
            if self.scaler is None or not fit_on_train:
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
                
                # Fit on numeric features only
                numeric_features = features.select_dtypes(include=[np.number])
                if len(numeric_features.columns) > 0:
                    # Fill NaN values with mean before fitting
                    numeric_features_filled = numeric_features.fillna(numeric_features.mean())
                    self.scaler.fit(numeric_features_filled)
                    self.normalization_stats = {
                        'method': 'minmax',
                        'columns': list(numeric_features.columns),
                        'min': {col: float(val) for col, val in zip(numeric_features.columns, getattr(self.scaler, 'data_min_', np.zeros(len(numeric_features.columns))))},
                        'max': {col: float(val) for col, val in zip(numeric_features.columns, getattr(self.scaler, 'data_max_', np.ones(len(numeric_features.columns))))},
                    }
            
            # Transform numeric features
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 0:
                # Fill NaN values with mean before transforming
                numeric_features_filled = numeric_features.fillna(numeric_features.mean())
                normalized_values = self.scaler.transform(numeric_features_filled)
                features[numeric_features.columns] = normalized_values

        return features

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Select features using the specified method.
        
        Args:
            features: DataFrame of features to select from
            
        Returns:
            Selected features
        """
        # Reset per-fit importance capture
        self.feature_scores_ = None
        # Backward-compat: map alternate keys
        if 'selection_method' in self.feature_selection_config and 'method' not in self.feature_selection_config:
            self.feature_selection_config['method'] = self.feature_selection_config['selection_method']
        if 'max_features' in self.feature_selection_config and 'k' not in self.feature_selection_config:
            # unify to k for univariate and variance-based selection
            self.feature_selection_config['k'] = int(self.feature_selection_config['max_features'])
        method = self.feature_selection_config.get('method', 'univariate')
        
        if method in ('univariate', 'k_best'):
            if self.feature_selector is None:
                k = int(self.feature_selection_config.get('k', 10))
                self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            
            # Calculate returns if not already present
            if 'returns' not in features.columns:
                # We need to calculate returns from the original data
                # This is a workaround for the test case
                returns = calculate_returns(features.iloc[:, 0])  # Use first column as proxy
                features['returns'] = returns
            
            # Remove rows with NaN values in both features and returns
            features_clean = features.dropna()
            
            if len(features_clean) > 0:
                # Prepare features and target for selection
                # Drop the synthetic target and restrict to numeric columns only for selector
                X = features_clean.drop(columns=['returns'])
                # Ensure non-numeric labels like 'ticker' are excluded from selector input
                X = X.select_dtypes(include=[np.number])
                y = features_clean['returns'].shift(-1).ffill()
                
                # Remove any remaining NaN values
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) > 0 and len(y_clean) > 0:
                    _ = self.feature_selector.fit(X_clean, y_clean)
                    self.selected_features = X_clean.columns[self.feature_selector.get_support()].tolist()
                    scores = getattr(self.feature_selector, 'scores_', None)
                    pvalues = getattr(self.feature_selector, 'pvalues_', None)
                    if scores is not None:
                        df_scores = pd.DataFrame({'feature': X_clean.columns, 'score': scores})
                        if pvalues is not None:
                            df_scores['pvalue'] = pvalues
                        df_scores = df_scores.sort_values('score', ascending=False)
                        self.feature_scores_ = df_scores.reset_index(drop=True)
                else:
                    # If no valid data after cleaning, select all features
                    self.selected_features = features.columns.tolist()
            else:
                # If no valid data, select all features
                self.selected_features = features.columns.tolist()
            
        elif method == 'variance':
            # Select top-k features by variance (drop non-numeric)
            k = int(self.feature_selection_config.get('k', 10))
            num = features.select_dtypes(include=[np.number])
            if num.empty:
                self.selected_features = features.columns.tolist()
            else:
                vars_ = num.var(axis=0).sort_values(ascending=False)
                top = vars_.index[:k].tolist()
                self.selected_features = top
        elif method == 'manual':
            if 'selected_features' in self.feature_selection_config:
                self.selected_features = self.feature_selection_config['selected_features']
                features = features[self.selected_features]
            else:
                raise ValueError("Manual selection requires 'selected_features' parameter")
        
        self.logger.info("Selected features: %s", self.selected_features)
        if self.selected_features is not None:
            keep_cols = [col for col in features.columns if col in self.selected_features]
            # Always preserve 'ticker' label if present for multi-ticker flows
            if 'ticker' in features.columns and 'ticker' not in keep_cols:
                keep_cols.append('ticker')
            return features[keep_cols]
        else:
            return features

    def audit_features(self, features: pd.DataFrame, *, stage: str = 'manual') -> pd.DataFrame:
        """Run feature audit and append to internal report list."""
        report = self._run_feature_audit(features, stage=stage)
        self._audit_reports.append(report)
        return report

    def _run_feature_audit(self, features: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Inspect features for NaNs, constants, and potential leakage."""
        if features.empty:
            return pd.DataFrame(columns=['feature', 'nan_ratio', 'inf_ratio', 'is_constant', 'head_nan_ratio', 'tail_nan_ratio', 'dtype', 'stage'])

        has_ticker_column = 'ticker' in features.columns
        has_ticker_index = isinstance(features.index, pd.MultiIndex) and 'ticker' in features.index.names

        if has_ticker_column or has_ticker_index:
            if has_ticker_column:
                grouped = features.groupby('ticker')
            else:
                tickers = features.index.get_level_values('ticker').unique()
                grouped = ((t, features.xs(t, level='ticker')) for t in tickers)

            for ticker, frame in grouped:
                if frame.empty:
                    continue
                idx = frame.index
                if isinstance(idx, pd.MultiIndex) and 'timestamp' in idx.names:
                    ts_idx = idx.get_level_values('timestamp')
                elif isinstance(idx, pd.DatetimeIndex):
                    ts_idx = idx
                else:
                    # Unable to validate index for this group; continue best-effort
                    continue
                if not ts_idx.is_monotonic_increasing:
                    raise ValueError(
                        f"Feature index is not monotonically increasing for ticker {ticker}; potential misalignment detected"
                    )
                if ts_idx.duplicated().any():
                    raise ValueError(
                        f"Feature index contains duplicates for ticker {ticker}; potential leakage or alignment issue"
                    )
        else:
            ts_index: Optional[pd.Index] = None
            if isinstance(features.index, pd.MultiIndex) and 'timestamp' in features.index.names:
                ts_index = features.index.get_level_values('timestamp')
            elif isinstance(features.index, pd.DatetimeIndex):
                ts_index = features.index

            if ts_index is not None:
                if not ts_index.is_monotonic_increasing:
                    raise ValueError("Feature index is not monotonically increasing; potential misalignment detected")
                if ts_index.duplicated().any():
                    raise ValueError("Feature index contains duplicates; potential leakage or alignment issue")

        rows: List[Dict[str, Any]] = []
        suspicious_names: List[str] = []
        suspicious_trailing: List[str] = []
        window = min(len(features), 10)

        for col in features.columns:
            if col == 'ticker':
                continue
            series = features[col]
            nan_ratio = float(series.isna().mean())
            inf_ratio = float(np.isinf(series).mean()) if series.dtype.kind in {'f', 'i'} else 0.0
            is_constant = bool(series.nunique(dropna=True) <= 1)
            head_slice = series.head(window)
            tail_slice = series.tail(window)
            head_nan_ratio = float(head_slice.isna().mean()) if len(head_slice) else 0.0
            tail_nan_ratio = float(tail_slice.isna().mean()) if len(tail_slice) else 0.0

            if self.LEAKAGE_PATTERN.search(str(col)):
                suspicious_names.append(col)
            if tail_nan_ratio >= 0.8 and head_nan_ratio <= 0.2:
                suspicious_trailing.append(col)

            rows.append({
                'feature': col,
                'nan_ratio': nan_ratio,
                'inf_ratio': inf_ratio,
                'is_constant': is_constant,
                'head_nan_ratio': head_nan_ratio,
                'tail_nan_ratio': tail_nan_ratio,
                'dtype': str(series.dtype),
                'stage': stage,
            })

        if suspicious_names:
            raise ValueError(f"Potential look-ahead feature names detected: {suspicious_names}")
        if suspicious_trailing:
            raise ValueError(
                "Features with trailing NaNs detected (possible forward shift): "
                f"{suspicious_trailing}"
            )

        report = pd.DataFrame(rows).sort_values('feature').reset_index(drop=True)
        return report

    def write_reports(
        self,
        output_dir: Union[str, Path],
        *,
        prefix: Optional[str] = None,
        features: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Persist selected features, audit reports, and normalization stats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        prefix_str = f"{prefix}_" if prefix else ""

        if features is not None:
            self.audit_features(features, stage='export')

        selected = list(self.selected_features or [])
        if not selected and features is not None:
            selected = [col for col in features.columns if col != 'ticker']
        selected_df = pd.DataFrame({'feature': selected})
        selected_df.to_csv(output_path / f"{prefix_str}features_selected.csv", index=False)

        if self.feature_scores_ is not None and not self.feature_scores_.empty:
            scores_df = self.feature_scores_.copy()
            scores_df['selected'] = scores_df['feature'].isin(selected)
            scores_df.to_csv(output_path / f"{prefix_str}feature_importances.csv", index=False)

        if self._audit_reports:
            audit_df = pd.concat(self._audit_reports, ignore_index=True)
            audit_df = audit_df.drop_duplicates(subset=['stage', 'feature']).sort_values(['stage', 'feature'])
            audit_df.to_csv(output_path / f"{prefix_str}feature_audit.csv", index=False)

        if self.normalization_stats:
            norm_path = output_path / f"{prefix_str}normalization_stats.json"
            with norm_path.open('w', encoding='utf-8') as fh:
                json.dump(self.normalization_stats, fh, indent=2)

        return output_path


def build_features_for_window(
    *,
    ohlcv: pd.DataFrame,
    base_features: pd.DataFrame,
    run_name: str,
    ticker: str,
    window_k: int,
    window_dir: Union[str, os.PathLike[str], Path],
    screen_dir: Optional[Union[str, os.PathLike[str], Path]],
    feature_pack: Optional[str],
    start: Union[pd.Timestamp, str],
    end: Union[pd.Timestamp, str],
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Resolve curated features for a single walk-forward window.

    This helper selects the curated subset for a window, persists bookkeeping, and
    serves feature caches. It assumes ``base_features`` already contains the
    computed indicators for the requested ticker/window span.
    """

    log = logger or logging.getLogger("features.window")

    window_path = Path(window_dir)
    window_path.mkdir(parents=True, exist_ok=True)

    screen_path = Path(screen_dir) if screen_dir is not None else window_path
    screen_path.mkdir(parents=True, exist_ok=True)

    has_l1 = all(col in ohlcv.columns for col in ("bid_price", "ask_price", "bid_size", "ask_size"))

    curated_from_screen = resolve_curated_pack(
        screen_path,
        has_l1=has_l1,
        logger=log,
        window_k=window_k,
        ticker=ticker,
    )

    available = [col for col in base_features.columns if col != "ticker"]
    pack_keys: List[str] = []
    if feature_pack:
        pack_keys = [p.strip() for p in str(feature_pack).split(",") if p.strip()]

    resolved = resolve_feature_pack(available, pack_keys, curated=curated_from_screen) if pack_keys else curated_from_screen
    if not resolved:
        resolved = available

    resolved_order = dict.fromkeys(resolved)
    final_cols = [col for col in base_features.columns if col != "ticker" and col in resolved_order]
    if not final_cols:
        final_cols = available

    base_hash = (
        f"{run_name}|{ticker}|{pd.Timestamp(start):%Y%m%d}-{pd.Timestamp(end):%Y%m%d}|"
        f"{','.join(pack_keys) or 'manual'}|{screen_path}"
    )
    cfg_hash = augment_cfg_hash(
        base_hash,
        run_name=run_name,
        features=final_cols,
        micro_module_path=Path('src/features/micro_ohlcv.py'),
        curated_token=LAST_CURATED_CACHE_TOKEN,
    )
    cache_path = feature_cache_path(
        run_name=run_name,
        ticker=ticker,
        split=f"window{window_k:02d}",
        start=start,
        end=end,
        cfg_hash=cfg_hash,
    )

    feats: Optional[pd.DataFrame] = None
    if use_cache and cache_path.exists():
        try:
            log.info("Feature cache HIT -> %s", cache_path)
            cached = load_features(cache_path)
            missing_cols = [c for c in final_cols if c not in cached.columns]
            if missing_cols:
                log.warning(
                    "Feature cache %s missing columns %s; recomputing",
                    cache_path,
                    missing_cols,
                )
            else:
                keep = ['ticker'] + final_cols if 'ticker' in cached.columns else final_cols
                feats = cached.loc[:, keep].copy()
        except Exception as exc:
            log.warning("Failed to load feature cache %s (%s); recomputing", cache_path, exc)
            feats = None

    if feats is None:
        status = "BYPASS" if not use_cache else "MISS"
        log.info(
            "Feature cache %s -> computing %d features for window %02d ticker %s",
            status,
            len(final_cols),
            window_k,
            ticker,
        )
        cols_to_keep = ['ticker'] + final_cols if 'ticker' in base_features.columns else final_cols
        feats = base_features.loc[:, [c for c in cols_to_keep if c in base_features.columns]].copy()
        if use_cache:
            save_features(feats, cache_path)

    curated = [c for c in final_cols if c != 'ticker']

    try:
        (window_path / 'features_used.txt').write_text("\n".join(curated) + "\n")
        # Also write to screen_path for consistency
        if screen_path != window_path:
            (screen_path / 'features_used.txt').write_text("\n".join(curated) + "\n")
    except Exception:
        pass

    missing_micro = [f for f in MICROSTRUCTURE_OHLCV if f in curated and f not in feats.columns]
    if missing_micro:
        raise AssertionError(f"Requested microstructure proxies missing from features: {missing_micro}")

    return feats, curated
