"""
Memory optimization utilities for the Multi-Ticker RL Trading System.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MemoryOptimizer:
    """Memory optimization utilities for the Multi-Ticker RL Trading System."""
    
    @staticmethod
    def optimize_data_loading(config: Dict[str, Any]) -> pd.DataFrame:
        """
        Optimize data loading for memory efficiency.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            pd.DataFrame: Optimized data
        """
        logger.info("Optimizing data loading for memory efficiency")
        
        # Use chunked loading for large datasets
        if config['data'].get('use_chunked_loading', False):
            chunk_size = config['data'].get('chunk_size', '1M')
            return MemoryOptimizer._load_data_in_chunks(
                tickers=config['data']['tickers'],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date'],
                chunk_size=chunk_size,
                config=config
            )
        else:
            # Standard loading with optimization
            from ..data.multiticker_data_loader import MultiTickerDataLoader
            data_loader = MultiTickerDataLoader(config['data'])
            data = data_loader.load_data()
            
            # Apply memory optimizations
            data = MemoryOptimizer._optimize_dataframe_memory(data)
            
            return data
    
    @staticmethod
    def _load_data_in_chunks(tickers: list, start_date: str, end_date: str, 
                           chunk_size: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Load data in chunks to optimize memory usage.
        
        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            chunk_size: Chunk size (e.g., '1M' for 1 month)
            config: Configuration dictionary
            
        Returns:
            pd.DataFrame: Combined data from all chunks
        """
        logger.info(f"Loading data in chunks of size {chunk_size}")
        
        from ..data.multiticker_data_loader import MultiTickerDataLoader
        
        # Generate date chunks
        dates = pd.date_range(start_date, end_date, freq=chunk_size)
        
        all_data = []
        
        for i in range(len(dates) - 1):
            chunk_start = dates[i]
            chunk_end = dates[i + 1] - pd.Timedelta(days=1)
            
            logger.info(f"Loading chunk {i+1}/{len(dates)-1}: {chunk_start} to {chunk_end}")
            
            # Update config for this chunk
            chunk_config = config.copy()
            chunk_config['data']['start_date'] = chunk_start.strftime('%Y-%m-%d')
            chunk_config['data']['end_date'] = chunk_end.strftime('%Y-%m-%d')
            
            # Load chunk
            data_loader = MultiTickerDataLoader(chunk_config['data'])
            chunk_data = data_loader.load_data()
            
            # Optimize chunk memory
            chunk_data = MemoryOptimizer._optimize_dataframe_memory(chunk_data)
            
            all_data.append(chunk_data)
        
        # Combine chunks
        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            logger.info(f"Combined data shape: {combined_data.shape}")
            return combined_data
        else:
            logger.warning("No data loaded from chunks")
            return pd.DataFrame()
    
    @staticmethod
    def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Memory-optimized DataFrame
        """
        logger.info(f"Optimizing DataFrame memory. Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Make a copy to avoid modifying original
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = optimized_df[col].astype(np.float32)
        
        # Optimize object columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique_values = len(optimized_df[col].unique())
            num_total_values = len(optimized_df[col])
            
            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        logger.info(f"Optimized memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return optimized_df
    
    @staticmethod
    def optimize_feature_extraction(features: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize feature extraction for memory efficiency.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            pd.DataFrame: Memory-optimized features
        """
        logger.info("Optimizing feature extraction for memory efficiency")
        
        # Apply DataFrame memory optimization
        features = MemoryOptimizer._optimize_dataframe_memory(features)
        
        # Additional feature-specific optimizations
        # Remove highly correlated features
        if features.shape[1] > 100:  # Only for large feature sets
            correlation_matrix = features.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            high_corr_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)
            ]
            
            if high_corr_features:
                logger.info(f"Removing {len(high_corr_features)} highly correlated features")
                features = features.drop(columns=high_corr_features)
        
        return features
    
    @staticmethod
    def optimize_model_training(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model training for memory efficiency.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict[str, Any]: Optimized configuration
        """
        logger.info("Optimizing model training for memory efficiency")
        
        # Reduce batch size if memory is limited
        if config['training'].get('batch_size', 64) > 32:
            config['training']['batch_size'] = 32
            logger.info("Reduced batch size to 32 for memory efficiency")
        
        # Enable gradient checkpointing
        config['training']['gradient_checkpointing'] = True
        logger.info("Enabled gradient checkpointing")
        
        # Use mixed precision training
        config['training']['mixed_precision'] = True
        logger.info("Enabled mixed precision training")
        
        # Optimize memory usage
        config['training']['max_grad_norm'] = 0.5
        logger.info("Set max gradient norm to 0.5")
        
        # Enable early stopping
        config['training']['early_stopping'] = True
        logger.info("Enabled early stopping")
        
        return config
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                'percent': process.memory_percent(),    # Memory usage percentage
            }
        except ImportError:
            # Fallback to basic memory info if psutil is not available
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                return {
                    'rss': usage.ru_maxrss / 1024,  # MB (Linux returns KB)
                    'vms': 0,  # Not available without psutil
                    'percent': 0  # Not available without psutil
                }
            except Exception:
                # Final fallback if resource module is not available
                return {
                    'rss': 0,
                    'vms': 0,
                    'percent': 0
                }
    
    @staticmethod
    def log_memory_usage(context: str = ""):
        """
        Log current memory usage.
        
        Args:
            context: Context string for logging
        """
        memory_stats = MemoryOptimizer.get_memory_usage()
        
        logger.info(f"Memory usage {context}: "
                   f"RSS: {memory_stats['rss']:.2f} MB, "
                   f"VMS: {memory_stats['vms']:.2f} MB, "
                   f"Percent: {memory_stats['percent']:.2f}%")
    
    @staticmethod
    def clear_memory():
        """Clear unused memory."""
        import gc
        
        logger.info("Clearing unused memory")
        gc.collect()
        
        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA cache")
        except ImportError:
            pass