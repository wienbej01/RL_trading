"""
Multi-ticker observation normalization for RL trading.

This module implements observation normalization techniques for multi-ticker
RL environments, including ticker-specific normalization, cross-ticker
normalization, and dynamic normalization adaptation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseNormalizer:
    """
    Base class for observation normalizers.
    
    Provides a framework for implementing different normalization
    strategies for multi-ticker RL observations.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize base normalizer.
        
        Args:
            **kwargs: Additional normalizer-specific parameters
        """
        self.kwargs = kwargs
        self.is_fitted = False
        
    def fit(self, data: Union[np.ndarray, th.Tensor]) -> 'BaseNormalizer':
        """
        Fit normalizer to data.
        
        Args:
            data: Data to fit on
            
        Returns:
            Self
        """
        raise NotImplementedError("Subclasses must implement fit")
        
    def transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Transform data using fitted normalizer.
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        raise NotImplementedError("Subclasses must implement transform")
        
    def fit_transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Fit normalizer and transform data.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Normalized data
        """
        return self.fit(data).transform(data)
        
    def inverse_transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        raise NotImplementedError("Subclasses must implement inverse_transform")
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get information about the normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        return {
            'type': self.__class__.__name__,
            'is_fitted': self.is_fitted
        }


class StandardNormalizer(BaseNormalizer):
    """
    Standard normalizer using z-score normalization.
    
    Normalizes data to have zero mean and unit variance.
    """
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        """
        Initialize standard normalizer.
        
        Args:
            epsilon: Small value to avoid division by zero
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        
    def fit(self, data: Union[np.ndarray, th.Tensor]) -> 'StandardNormalizer':
        """
        Fit standard normalizer to data.
        
        Args:
            data: Data to fit on
            
        Returns:
            Self
        """
        if isinstance(data, th.Tensor):
            data = data.detach().cpu().numpy()
            
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + self.epsilon
        self.is_fitted = True
        
        return self
        
    def transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Transform data using standard normalizer.
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        normalized = (data - self.mean) / self.std
        
        if is_tensor:
            normalized = th.from_numpy(normalized).to(data.device)
            
        return normalized
        
    def inverse_transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        denormalized = data * self.std + self.mean
        
        if is_tensor:
            denormalized = th.from_numpy(denormalized).to(data.device)
            
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get information about the normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        info = super().get_normalizer_info()
        info.update({
            'epsilon': self.epsilon,
            'mean_shape': self.mean.shape if self.mean is not None else None,
            'std_shape': self.std.shape if self.std is not None else None
        })
        return info


class MinMaxNormalizer(BaseNormalizer):
    """
    Min-max normalizer.
    
    Normalizes data to a specified range, typically [0, 1] or [-1, 1].
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), epsilon: float = 1e-8, **kwargs):
        """
        Initialize min-max normalizer.
        
        Args:
            feature_range: Target range for normalized data
            epsilon: Small value to avoid division by zero
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min = None
        self.max = None
        
    def fit(self, data: Union[np.ndarray, th.Tensor]) -> 'MinMaxNormalizer':
        """
        Fit min-max normalizer to data.
        
        Args:
            data: Data to fit on
            
        Returns:
            Self
        """
        if isinstance(data, th.Tensor):
            data = data.detach().cpu().numpy()
            
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.is_fitted = True
        
        return self
        
    def transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Transform data using min-max normalizer.
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        min_val, max_val = self.feature_range
        scale = (max_val - min_val) / (self.max - self.min + self.epsilon)
        normalized = min_val + scale * (data - self.min)
        
        if is_tensor:
            normalized = th.from_numpy(normalized).to(data.device)
            
        return normalized
        
    def inverse_transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        min_val, max_val = self.feature_range
        scale = (max_val - min_val) / (self.max - self.min + self.epsilon)
        denormalized = self.min + (data - min_val) / scale
        
        if is_tensor:
            denormalized = th.from_numpy(denormalized).to(data.device)
            
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get information about the normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        info = super().get_normalizer_info()
        info.update({
            'feature_range': self.feature_range,
            'epsilon': self.epsilon,
            'min_shape': self.min.shape if self.min is not None else None,
            'max_shape': self.max.shape if self.max is not None else None
        })
        return info


class RobustNormalizer(BaseNormalizer):
    """
    Robust normalizer using median and quantiles.
    
    Normalizes data using median and interquartile range, which is
    more robust to outliers than standard normalization.
    """
    
    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0), epsilon: float = 1e-8, **kwargs):
        """
        Initialize robust normalizer.
        
        Args:
            quantile_range: Quantile range for robust scaling
            epsilon: Small value to avoid division by zero
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.quantile_range = quantile_range
        self.epsilon = epsilon
        self.median = None
        self.iqr = None
        
    def fit(self, data: Union[np.ndarray, th.Tensor]) -> 'RobustNormalizer':
        """
        Fit robust normalizer to data.
        
        Args:
            data: Data to fit on
            
        Returns:
            Self
        """
        if isinstance(data, th.Tensor):
            data = data.detach().cpu().numpy()
            
        self.median = np.median(data, axis=0)
        q1, q3 = np.percentile(data, self.quantile_range, axis=0)
        self.iqr = q3 - q1 + self.epsilon
        self.is_fitted = True
        
        return self
        
    def transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Transform data using robust normalizer.
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        normalized = (data - self.median) / self.iqr
        
        if is_tensor:
            normalized = th.from_numpy(normalized).to(data.device)
            
        return normalized
        
    def inverse_transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        denormalized = data * self.iqr + self.median
        
        if is_tensor:
            denormalized = th.from_numpy(denormalized).to(data.device)
            
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get information about the normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        info = super().get_normalizer_info()
        info.update({
            'quantile_range': self.quantile_range,
            'epsilon': self.epsilon,
            'median_shape': self.median.shape if self.median is not None else None,
            'iqr_shape': self.iqr.shape if self.iqr is not None else None
        })
        return info


class TickerNormalizer:
    """
    Ticker-specific normalizer for multi-ticker observations.
    
    Maintains separate normalizers for each ticker to account for
    ticker-specific characteristics and scales.
    """
    
    def __init__(
        self,
        normalizer_type: str = 'standard',
        tickers: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize ticker normalizer.
        
        Args:
            normalizer_type: Type of normalizer to use for each ticker
            tickers: List of ticker symbols
            **kwargs: Additional parameters for normalizers
        """
        self.normalizer_type = normalizer_type
        self.tickers = tickers or []
        self.kwargs = kwargs
        
        # Create normalizers for each ticker
        self.normalizers = {}
        for ticker in self.tickers:
            self.normalizers[ticker] = self._create_normalizer()
            
    def _create_normalizer(self) -> BaseNormalizer:
        """
        Create a normalizer instance based on the specified type.
        
        Returns:
            Normalizer instance
        """
        if self.normalizer_type == 'standard':
            return StandardNormalizer(**self.kwargs)
        elif self.normalizer_type == 'minmax':
            return MinMaxNormalizer(**self.kwargs)
        elif self.normalizer_type == 'robust':
            return RobustNormalizer(**self.kwargs)
        else:
            logger.warning(f"Unknown normalizer type: {self.normalizer_type}, using standard")
            return StandardNormalizer(**self.kwargs)
            
    def add_ticker(self, ticker: str) -> None:
        """
        Add a new ticker to the normalizer.
        
        Args:
            ticker: Ticker symbol to add
        """
        if ticker not in self.normalizers:
            self.normalizers[ticker] = self._create_normalizer()
            self.tickers.append(ticker)
            
    def fit(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> 'TickerNormalizer':
        """
        Fit ticker normalizers to data.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Self
        """
        for ticker, ticker_data in data.items():
            if ticker not in self.normalizers:
                self.add_ticker(ticker)
            self.normalizers[ticker].fit(ticker_data)
            
        return self
        
    def transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Transform data using ticker normalizers.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary mapping tickers to normalized data
        """
        normalized = {}
        for ticker, ticker_data in data.items():
            if ticker in self.normalizers:
                normalized[ticker] = self.normalizers[ticker].transform(ticker_data)
            else:
                logger.warning(f"No normalizer found for ticker {ticker}, using raw data")
                normalized[ticker] = ticker_data
                
        return normalized
        
    def fit_transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Fit ticker normalizers and transform data.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary mapping tickers to normalized data
        """
        return self.fit(data).transform(data)
        
    def inverse_transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Dictionary mapping tickers to normalized data
            
        Returns:
            Dictionary mapping tickers to original scale data
        """
        denormalized = {}
        for ticker, ticker_data in data.items():
            if ticker in self.normalizers:
                denormalized[ticker] = self.normalizers[ticker].inverse_transform(ticker_data)
            else:
                logger.warning(f"No normalizer found for ticker {ticker}, using raw data")
                denormalized[ticker] = ticker_data
                
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """
        Get information about the ticker normalizers.
        
        Returns:
            Dictionary mapping tickers to their normalizer information
        """
        return {
            ticker: normalizer.get_normalizer_info()
            for ticker, normalizer in self.normalizers.items()
        }


class CrossTickerNormalizer(BaseNormalizer):
    """
    Cross-ticker normalizer for multi-ticker observations.
    
    Normalizes features across all tickers to ensure consistent
    scaling and enable cross-ticker comparisons.
    """
    
    def __init__(
        self,
        normalizer_type: str = 'standard',
        aggregation_method: str = 'concat',
        **kwargs
    ):
        """
        Initialize cross-ticker normalizer.
        
        Args:
            normalizer_type: Type of normalizer to use
            aggregation_method: Method for aggregating data across tickers ('concat', 'mean', 'max')
            **kwargs: Additional parameters for normalizer
        """
        super().__init__(**kwargs)
        self.normalizer_type = normalizer_type
        self.aggregation_method = aggregation_method
        self.base_normalizer = self._create_normalizer()
        
    def _create_normalizer(self) -> BaseNormalizer:
        """
        Create a base normalizer instance.
        
        Returns:
            Normalizer instance
        """
        if self.normalizer_type == 'standard':
            return StandardNormalizer(**self.kwargs)
        elif self.normalizer_type == 'minmax':
            return MinMaxNormalizer(**self.kwargs)
        elif self.normalizer_type == 'robust':
            return RobustNormalizer(**self.kwargs)
        else:
            logger.warning(f"Unknown normalizer type: {self.normalizer_type}, using standard")
            return StandardNormalizer(**self.kwargs)
            
    def _aggregate_data(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Union[np.ndarray, th.Tensor]:
        """
        Aggregate data across tickers.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Aggregated data
        """
        if self.aggregation_method == 'concat':
            # Concatenate all ticker data
            data_list = list(data.values())
            if isinstance(data_list[0], th.Tensor):
                return th.cat(data_list, dim=0)
            else:
                return np.concatenate(data_list, axis=0)
        elif self.aggregation_method == 'mean':
            # Compute mean across tickers
            data_list = list(data.values())
            if isinstance(data_list[0], th.Tensor):
                return th.stack(data_list, dim=0).mean(dim=0)
            else:
                return np.stack(data_list, axis=0).mean(axis=0)
        elif self.aggregation_method == 'max':
            # Compute max across tickers
            data_list = list(data.values())
            if isinstance(data_list[0], th.Tensor):
                return th.stack(data_list, dim=0).max(dim=0)[0]
            else:
                return np.stack(data_list, axis=0).max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
    def fit(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> 'CrossTickerNormalizer':
        """
        Fit cross-ticker normalizer to data.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Self
        """
        aggregated = self._aggregate_data(data)
        self.base_normalizer.fit(aggregated)
        self.is_fitted = True
        
        return self
        
    def transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Transform data using cross-ticker normalizer.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary mapping tickers to normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
            
        normalized = {}
        for ticker, ticker_data in data.items():
            normalized[ticker] = self.base_normalizer.transform(ticker_data)
            
        return normalized
        
    def fit_transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Fit cross-ticker normalizer and transform data.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary mapping tickers to normalized data
        """
        return self.fit(data).transform(data)
        
    def inverse_transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Dictionary mapping tickers to normalized data
            
        Returns:
            Dictionary mapping tickers to original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        denormalized = {}
        for ticker, ticker_data in data.items():
            denormalized[ticker] = self.base_normalizer.inverse_transform(ticker_data)
            
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get information about the cross-ticker normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        info = super().get_normalizer_info()
        info.update({
            'normalizer_type': self.normalizer_type,
            'aggregation_method': self.aggregation_method,
            'base_normalizer': self.base_normalizer.get_normalizer_info()
        })
        return info


class MultiTickerNormalizer:
    """
    Multi-ticker observation normalizer combining ticker-specific and cross-ticker normalization.
    
    This normalizer applies both ticker-specific normalization to account for
    ticker characteristics and cross-ticker normalization to ensure consistent scaling.
    """
    
    def __init__(
        self,
        ticker_normalizer_type: str = 'standard',
        cross_ticker_normalizer_type: str = 'standard',
        cross_ticker_aggregation: str = 'concat',
        normalization_order: str = 'ticker_first',
        **kwargs
    ):
        """
        Initialize multi-ticker normalizer.
        
        Args:
            ticker_normalizer_type: Type of normalizer for ticker-specific normalization
            cross_ticker_normalizer_type: Type of normalizer for cross-ticker normalization
            cross_ticker_aggregation: Method for aggregating data across tickers
            normalization_order: Order of normalization ('ticker_first' or 'cross_first')
            **kwargs: Additional parameters for normalizers
        """
        self.ticker_normalizer_type = ticker_normalizer_type
        self.cross_ticker_normalizer_type = cross_ticker_normalizer_type
        self.cross_ticker_aggregation = cross_ticker_aggregation
        self.normalization_order = normalization_order
        self.kwargs = kwargs
        
        # Initialize normalizers
        self.ticker_normalizer = None
        self.cross_ticker_normalizer = None
        
    def _create_normalizers(self, tickers: List[str]) -> None:
        """
        Create normalizer instances.
        
        Args:
            tickers: List of ticker symbols
        """
        self.ticker_normalizer = TickerNormalizer(
            normalizer_type=self.ticker_normalizer_type,
            tickers=tickers,
            **self.kwargs
        )
        
        self.cross_ticker_normalizer = CrossTickerNormalizer(
            normalizer_type=self.cross_ticker_normalizer_type,
            aggregation_method=self.cross_ticker_aggregation,
            **self.kwargs
        )
        
    def fit(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> 'MultiTickerNormalizer':
        """
        Fit multi-ticker normalizer to data.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Self
        """
        tickers = list(data.keys())
        self._create_normalizers(tickers)
        
        if self.normalization_order == 'ticker_first':
            # Apply ticker-specific normalization first
            ticker_normalized = self.ticker_normalizer.fit_transform(data)
            # Then fit cross-ticker normalizer
            self.cross_ticker_normalizer.fit(ticker_normalized)
        else:  # cross_first
            # Apply cross-ticker normalization first
            self.cross_ticker_normalizer.fit(data)
            # Then fit ticker-specific normalizer
            cross_normalized = self.cross_ticker_normalizer.transform(data)
            self.ticker_normalizer.fit(cross_normalized)
            
        return self
        
    def transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Transform data using multi-ticker normalizer.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary mapping tickers to normalized data
        """
        if self.ticker_normalizer is None or self.cross_ticker_normalizer is None:
            raise ValueError("Normalizer must be fitted before transforming data")
            
        if self.normalization_order == 'ticker_first':
            # Apply ticker-specific normalization first
            ticker_normalized = self.ticker_normalizer.transform(data)
            # Then apply cross-ticker normalization
            normalized = self.cross_ticker_normalizer.transform(ticker_normalized)
        else:  # cross_first
            # Apply cross-ticker normalization first
            cross_normalized = self.cross_ticker_normalizer.transform(data)
            # Then apply ticker-specific normalization
            normalized = self.ticker_normalizer.transform(cross_normalized)
            
        return normalized
        
    def fit_transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Fit multi-ticker normalizer and transform data.
        
        Args:
            data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary mapping tickers to normalized data
        """
        return self.fit(data).transform(data)
        
    def inverse_transform(self, data: Dict[str, Union[np.ndarray, th.Tensor]]) -> Dict[str, Union[np.ndarray, th.Tensor]]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Dictionary mapping tickers to normalized data
            
        Returns:
            Dictionary mapping tickers to original scale data
        """
        if self.ticker_normalizer is None or self.cross_ticker_normalizer is None:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        if self.normalization_order == 'ticker_first':
            # Inverse cross-ticker normalization first
            cross_denormalized = self.cross_ticker_normalizer.inverse_transform(data)
            # Then inverse ticker-specific normalization
            denormalized = self.ticker_normalizer.inverse_transform(cross_denormalized)
        else:  # cross_first
            # Inverse ticker-specific normalization first
            ticker_denormalized = self.ticker_normalizer.inverse_transform(data)
            # Then inverse cross-ticker normalization
            denormalized = self.cross_ticker_normalizer.inverse_transform(ticker_denormalized)
            
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int, Dict]]:
        """
        Get information about the multi-ticker normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        return {
            'ticker_normalizer_type': self.ticker_normalizer_type,
            'cross_ticker_normalizer_type': self.cross_ticker_normalizer_type,
            'cross_ticker_aggregation': self.cross_ticker_aggregation,
            'normalization_order': self.normalization_order,
            'ticker_normalizer': self.ticker_normalizer.get_normalizer_info() if self.ticker_normalizer else None,
            'cross_ticker_normalizer': self.cross_ticker_normalizer.get_normalizer_info() if self.cross_ticker_normalizer else None
        }


class RunningNormalizer(BaseNormalizer):
    """
    Running normalizer for online normalization.
    
    Updates normalization statistics incrementally as new data arrives,
    suitable for online learning scenarios.
    """
    
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-8, **kwargs):
        """
        Initialize running normalizer.
        
        Args:
            momentum: Momentum for updating running statistics
            epsilon: Small value to avoid division by zero
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.count = 0
        
    def fit(self, data: Union[np.ndarray, th.Tensor]) -> 'RunningNormalizer':
        """
        Fit running normalizer to data.
        
        Args:
            data: Data to fit on
            
        Returns:
            Self
        """
        if isinstance(data, th.Tensor):
            data = data.detach().cpu().numpy()
            
        batch_mean = np.mean(data, axis=0)
        batch_var = np.var(data, axis=0)
        batch_count = data.shape[0]
        
        if self.running_mean is None:
            # First batch
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.count = batch_count
        else:
            # Update running statistics
            self.count += batch_count
            delta = batch_mean - self.running_mean
            m_a = self.count - batch_count
            m_b = batch_count
            
            # Update mean
            self.running_mean = (m_a * self.running_mean + m_b * batch_mean) / self.count
            
            # Update variance using Welford's algorithm
            self.running_var = (m_a * self.running_var + m_b * batch_var + 
                              m_a * m_b * delta**2 / self.count) / self.count
            
        self.is_fitted = True
        
        return self
        
    def transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Transform data using running normalizer.
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        normalized = (data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        if is_tensor:
            normalized = th.from_numpy(normalized).to(data.device)
            
        return normalized
        
    def update(self, data: Union[np.ndarray, th.Tensor]) -> 'RunningNormalizer':
        """
        Update running normalizer with new data.
        
        Args:
            data: New data to update with
            
        Returns:
            Self
        """
        return self.fit(data)
        
    def inverse_transform(self, data: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, th.Tensor]:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        is_tensor = isinstance(data, th.Tensor)
        
        if is_tensor:
            data = data.detach().cpu().numpy()
            
        denormalized = data * np.sqrt(self.running_var + self.epsilon) + self.running_mean
        
        if is_tensor:
            denormalized = th.from_numpy(denormalized).to(data.device)
            
        return denormalized
        
    def get_normalizer_info(self) -> Dict[str, Union[str, float, int]]:
        """
        Get information about the running normalizer.
        
        Returns:
            Dictionary with normalizer information
        """
        info = super().get_normalizer_info()
        info.update({
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'count': self.count,
            'running_mean_shape': self.running_mean.shape if self.running_mean is not None else None,
            'running_var_shape': self.running_var.shape if self.running_var is not None else None
        })
        return info


def create_normalizer(
    normalizer_type: str,
    **kwargs
) -> BaseNormalizer:
    """
    Create a normalizer instance of the specified type.
    
    Args:
        normalizer_type: Type of normalizer to create
        **kwargs: Additional parameters for the normalizer
        
    Returns:
        Normalizer instance
    """
    if normalizer_type == 'standard':
        return StandardNormalizer(**kwargs)
    elif normalizer_type == 'minmax':
        return MinMaxNormalizer(**kwargs)
    elif normalizer_type == 'robust':
        return RobustNormalizer(**kwargs)
    elif normalizer_type == 'running':
        return RunningNormalizer(**kwargs)
    else:
        logger.warning(f"Unknown normalizer type: {normalizer_type}, using standard")
        return StandardNormalizer(**kwargs)