"""
Feature engineering pipeline for the RL trading system.

This module provides a comprehensive feature engineering pipeline
that combines technical indicators, microstructure features, and time-based features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from .technical_indicators import TechnicalIndicators
from .microstructure_features import MicrostructureFeatures
from .time_features import TimeFeatures

logger = get_logger(__name__)


@dataclass
class FeaturePipelineConfig:
    """Configuration for the feature pipeline."""
    technical_indicators: bool = True
    microstructure_features: bool = True
    time_features: bool = True
    feature_selection: bool = True
    normalization: bool = True
    feature_importance: bool = False


class FeaturePipeline:
    """
    Comprehensive feature engineering pipeline.
    
    This class orchestrates the calculation of all features including
    technical indicators, microstructure features, and time-based features.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize feature pipeline.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.config = settings.get('feature_pipeline', {})
        
        # Initialize feature calculators
        self.technical_indicators = TechnicalIndicators(settings)
        self.microstructure_features = MicrostructureFeatures(settings)
        self.time_features = TimeFeatures(settings)
        
        # Feature storage
        self.feature_importance: Dict[str, float] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
    def build_feature_matrix(self, 
                           data: pd.DataFrame,
                           vix_data: Optional[pd.DataFrame] = None,
                           macro_flags: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build comprehensive feature matrix.
        
        Args:
            data: DataFrame with OHLCV data
            vix_data: Optional VIX data
            macro_flags: Optional macroeconomic flags
            
        Returns:
            DataFrame with all features
        """
        if data.empty:
            logger.error("Input data is empty")
            return pd.DataFrame()
        
        logger.info(f"Building feature matrix for {len(data)} rows")
        
        # Initialize feature matrix
        feature_matrix = pd.DataFrame(index=data.index)
        
        # Add technical indicators
        if self.config.get('technical_indicators', True):
            logger.info("Calculating technical indicators...")
            tech_features = self.technical_indicators.calculate_all_indicators(data)
            feature_matrix = pd.concat([feature_matrix, tech_features], axis=1)
        
        # Add microstructure features
        if self.config.get('microstructure_features', True):
            logger.info("Calculating microstructure features...")
            micro_features = self.microstructure_features.calculate_all_microstructure_features(data)
            feature_matrix = pd.concat([feature_matrix, micro_features], axis=1)
        
        # Add time-based features
        if self.config.get('time_features', True):
            logger.info("Calculating time-based features...")
            time_features = self.time_features.calculate_all_time_features(data)
            feature_matrix = pd.concat([feature_matrix, time_features], axis=1)
        
        # Add external features (VIX, macro)
        if vix_data is not None:
            logger.info("Adding VIX features...")
            vix_features = self._add_vix_features(feature_matrix, vix_data)
            feature_matrix = pd.concat([feature_matrix, vix_features], axis=1)
        
        if macro_flags is not None:
            logger.info("Adding macro features...")
            macro_features = self._add_macro_features(feature_matrix, macro_flags)
            feature_matrix = pd.concat([feature_matrix, macro_features], axis=1)
        
        # Feature selection
        if self.config.get('feature_selection', True):
            logger.info("Performing feature selection...")
            feature_matrix = self._select_features(feature_matrix)
        
        # Normalization
        if self.config.get('normalization', True):
            logger.info("Normalizing features...")
            feature_matrix = self._normalize_features(feature_matrix)
        
        # Calculate feature importance
        if self.config.get('feature_importance', False):
            logger.info("Calculating feature importance...")
            self._calculate_feature_importance(feature_matrix)
        
        # Store feature statistics
        self._calculate_feature_statistics(feature_matrix)
        
        logger.info(f"Feature matrix built with {feature_matrix.shape[1]} features")
        
        return feature_matrix
    
    def _add_vix_features(self, feature_matrix: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
        """Add VIX-based features."""
        vix_features = pd.DataFrame(index=feature_matrix.index)
        
        # VIX level features
        vix_features['vix_level'] = vix_data['close'].reindex(feature_matrix.index, method='ffill')
        vix_features['vix_return'] = vix_features['vix_level'].pct_change()
        vix_features['vix_volatility'] = vix_features['vix_level'].rolling(window=20).std()
        
        # VIX regime
        vix_features['vix_regime'] = pd.cut(
            vix_features['vix_level'],
            bins=[0, 13, 20, 30, 100],
            labels=['low', 'normal', 'high', 'extreme']
        )
        
        # VIX momentum
        vix_features['vix_momentum'] = (
            vix_features['vix_level'] - vix_features['vix_level'].rolling(window=20).mean()
        ) / vix_features['vix_level'].rolling(window=20).mean()
        
        # VIX correlation with price
        vix_features['vix_price_correlation'] = (
            feature_matrix['close'].pct_change().rolling(window=20).corr(
                vix_features['vix_return']
            )
        )
        
        return vix_features
    
    def _add_macro_features(self, feature_matrix: pd.DataFrame, macro_flags: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic features."""
        macro_features = pd.DataFrame(index=feature_matrix.index)
        
        # Event flags
        for col in macro_flags.columns:
            if macro_flags[col].dtype == bool:
                macro_features[f'macro_{col}'] = macro_flags[col].reindex(feature_matrix.index, method='ffill')
        
        # Event intensity
        macro_features['macro_event_intensity'] = macro_flags.sum(axis=1).reindex(feature_matrix.index, method='ffill')
        
        # Event timing
        macro_features['macro_event_timing'] = (
            macro_flags.any(axis=1).reindex(feature_matrix.index, method='ffill').astype(int)
        )
        
        return macro_features
    
    def _select_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Perform feature selection."""
        # Remove highly correlated features
        correlation_threshold = 0.95
        correlation_matrix = feature_matrix.corr().abs()
        
        # Find highly correlated features
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Select features to remove
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > correlation_threshold)
        ]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        
        # Remove features with low variance
        variance_threshold = 0.001
        low_variance = feature_matrix.var() < variance_threshold
        to_drop.extend(low_variance[low_variance].index.tolist())
        
        # Remove features with too many NaN values
        nan_threshold = 0.1
        nan_ratio = feature_matrix.isnull().mean()
        to_drop.extend(nan_ratio[nan_ratio > nan_threshold].index.tolist())
        
        # Remove duplicate features
        to_drop = list(set(to_drop))
        
        # Drop selected features
        selected_features = feature_matrix.drop(columns=to_drop)
        
        logger.info(f"Selected {selected_features.shape[1]} features from {feature_matrix.shape[1]}")
        
        return selected_features
    
    def _normalize_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Normalize features."""
        normalized_matrix = feature_matrix.copy()
        
        # Calculate statistics for each feature
        for column in normalized_matrix.columns:
            if normalized_matrix[column].dtype in ['float64', 'float32']:
                # Remove NaN values for calculation
                clean_values = normalized_matrix[column].dropna()
                
                if len(clean_values) > 0:
                    mean_val = clean_values.mean()
                    std_val = clean_values.std()
                    
                    # Z-score normalization
                    if std_val > 0:
                        normalized_matrix[column] = (
                            (normalized_matrix[column] - mean_val) / std_val
                        )
                    
                    # Store statistics
                    self.feature_stats[column] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': clean_values.min(),
                        'max': clean_values.max(),
                        'skew': clean_values.skew(),
                        'kurtosis': clean_values.kurtosis()
                    }
        
        return normalized_matrix
    
    def _calculate_feature_importance(self, feature_matrix: pd.DataFrame) -> None:
        """Calculate feature importance using random forest."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.impute import SimpleImputer
            
            # Prepare data
            X = feature_matrix.dropna()
            if len(X) == 0:
                logger.warning("No clean data for feature importance calculation")
                return
            
            # Use price returns as target
            y = X['close'].pct_change().dropna()
            X = X.loc[y.index]
            
            if len(X) < 100:
                logger.warning("Insufficient data for feature importance calculation")
                return
            
            # Impute missing values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            
            # Train random forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_imputed, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            feature_names = X.columns
            
            # Store importance
            self.feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 most important features:")
            for feature, importance in sorted_importance[:10]:
                logger.info(f"  {feature}: {importance:.4f}")
                
        except ImportError:
            logger.warning("scikit-learn not available for feature importance calculation")
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
    
    def _calculate_feature_statistics(self, feature_matrix: pd.DataFrame) -> None:
        """Calculate feature statistics."""
        self.feature_stats = {}
        
        for column in feature_matrix.columns:
            if feature_matrix[column].dtype in ['float64', 'float32']:
                clean_values = feature_matrix[column].dropna()
                
                if len(clean_values) > 0:
                    self.feature_stats[column] = {
                        'mean': clean_values.mean(),
                        'std': clean_values.std(),
                        'min': clean_values.min(),
                        'max': clean_values.max(),
                        'skew': clean_values.skew(),
                        'kurtosis': clean_values.kurtosis(),
                        'missing_ratio': feature_matrix[column].isnull().mean(),
                        'zero_ratio': (feature_matrix[column] == 0).mean()
                    }
    
    def get_feature_summary(self) -> Dict[str, any]:
        """Get feature pipeline summary."""
        summary = {
            'total_features': len(self.feature_stats),
            'feature_importance': self.feature_importance,
            'feature_stats': self.feature_stats,
            'config': self.config
        }
        
        return summary
    
    def save_feature_importance(self, filepath: str) -> None:
        """Save feature importance to file."""
        if not self.feature_importance:
            logger.warning("No feature importance data to save")
            return
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} for k, v in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        importance_df.to_csv(filepath, index=False)
        logger.info(f"Feature importance saved to {filepath}")
    
    def save_feature_statistics(self, filepath: str) -> None:
        """Save feature statistics to file."""
        if not self.feature_stats:
            logger.warning("No feature statistics data to save")
            return
        
        stats_df = pd.DataFrame(self.feature_stats).T
        stats_df.to_csv(filepath)
        logger.info(f"Feature statistics saved to {filepath}")
    
    def load_feature_importance(self, filepath: str) -> None:
        """Load feature importance from file."""
        try:
            importance_df = pd.read_csv(filepath)
            self.feature_importance = dict(
                zip(importance_df['feature'], importance_df['importance'])
            )
            logger.info(f"Feature importance loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
    
    def load_feature_statistics(self, filepath: str) -> None:
        """Load feature statistics from file."""
        try:
            stats_df = pd.read_csv(filepath, index_col=0)
            self.feature_stats = stats_df.to_dict('index')
            logger.info(f"Feature statistics loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading feature statistics: {e}")
    
    def validate_features(self, feature_matrix: pd.DataFrame) -> Dict[str, any]:
        """Validate feature matrix."""
        validation_results = {
            'total_features': feature_matrix.shape[1],
            'total_samples': feature_matrix.shape[0],
            'missing_values': feature_matrix.isnull().sum().to_dict(),
            'infinite_values': np.isinf(feature_matrix).sum().to_dict(),
            'constant_features': [],
            'highly_correlated': []
        }
        
        # Check for constant features
        for column in feature_matrix.columns:
            if feature_matrix[column].nunique() == 1:
                validation_results['constant_features'].append(column)
        
        # Check for highly correlated features
        correlation_matrix = feature_matrix.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        highly_correlated = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
            if correlated_features:
                highly_correlated.append((column, correlated_features))
        
        validation_results['highly_correlated'] = highly_correlated
        
        return validation_results
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.feature_stats.keys())
    
    def get_feature_importance_ranking(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n features by importance."""
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:n]