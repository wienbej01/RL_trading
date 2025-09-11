"""
Computation optimization utilities for the Multi-Ticker RL Trading System.
"""

import os
import logging
from typing import Dict, Any, Union, Optional
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ComputationOptimizer:
    """Computation optimization utilities for the Multi-Ticker RL Trading System."""
    
    @staticmethod
    def optimize_for_gpu() -> bool:
        """
        Optimize system for GPU computation.
        
        Returns:
            bool: True if GPU optimization was applied, False otherwise
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                logger.info("Optimizing for GPU computation")
                
                # Set default device to GPU
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                logger.info("Set default tensor type to CUDA float")
                
                # Enable cuDNN benchmark mode
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
                
                # Disable cuDNN deterministic mode for better performance
                torch.backends.cudnn.deterministic = False
                logger.info("Disabled cuDNN deterministic mode")
                
                # Set memory allocation strategy
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                logger.info("Set CUDA memory allocation strategy")
                
                return True
            else:
                logger.warning("CUDA not available, skipping GPU optimization")
                return False
                
        except ImportError:
            logger.warning("PyTorch not available, skipping GPU optimization")
            return False
    
    @staticmethod
    def optimize_for_cpu() -> bool:
        """
        Optimize system for CPU computation.
        
        Returns:
            bool: True if CPU optimization was applied, False otherwise
        """
        try:
            import torch
            
            logger.info("Optimizing for CPU computation")
            
            # Set number of threads for CPU operations
            torch.set_num_threads(os.cpu_count())
            logger.info(f"Set PyTorch threads to {os.cpu_count()}")
            
            # Enable MKL if available
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                logger.info("Enabled MKL for CPU operations")
            
            # Set inter-op parallelism
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(1)
                logger.info("Set inter-op threads to 1")
            
            return True
            
        except ImportError:
            logger.warning("PyTorch not available, skipping CPU optimization")
            return False
    
    @staticmethod
    def optimize_data_parallelism(model, config: Dict[str, Any]) -> tuple:
        """
        Optimize model for data parallelism.
        
        Args:
            model: PyTorch model to optimize
            config: Configuration dictionary
            
        Returns:
            tuple: (optimized_model, optimized_config)
        """
        try:
            import torch
            
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                logger.info(f"Optimizing for {torch.cuda.device_count()} GPUs")
                
                # Use DataParallel for multi-GPU training
                model = torch.nn.DataParallel(model)
                logger.info("Enabled DataParallel for multi-GPU training")
                
                # Adjust batch size for multi-GPU
                if 'training' in config:
                    original_batch_size = config['training'].get('batch_size', 32)
                    config['training']['batch_size'] = original_batch_size * torch.cuda.device_count()
                    logger.info(f"Adjusted batch size from {original_batch_size} to {config['training']['batch_size']}")
                
                # Enable gradient accumulation for large effective batch sizes
                if 'training' in config:
                    config['training']['gradient_accumulation_steps'] = torch.cuda.device_count()
                    logger.info(f"Set gradient accumulation steps to {torch.cuda.device_count()}")
            
            return model, config
            
        except ImportError:
            logger.warning("PyTorch not available, skipping data parallelism optimization")
            return model, config
    
    @staticmethod
    def optimize_tensor_operations() -> bool:
        """
        Optimize tensor operations for better performance.
        
        Returns:
            bool: True if optimization was applied, False otherwise
        """
        try:
            import torch
            
            logger.info("Optimizing tensor operations")
            
            # Enable automatic mixed precision
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logger.info("Automatic Mixed Precision (AMP) is available")
            
            # Set memory-efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("Memory-efficient attention is available")
            
            # Enable flash attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                try:
                    # Test if flash attention is available
                    query = torch.randn(1, 8, 64, 64, device='cuda')
                    key = torch.randn(1, 8, 64, 64, device='cuda')
                    value = torch.randn(1, 8, 64, 64, device='cuda')
                    
                    with torch.backends.cuda.sdp_kernel(enable_flash=True):
                        torch.nn.functional.scaled_dot_product_attention(query, key, value)
                    
                    logger.info("Flash attention is available and enabled")
                except Exception as e:
                    logger.warning(f"Flash attention not available: {e}")
            
            return True
            
        except ImportError:
            logger.warning("PyTorch not available, skipping tensor operation optimization")
            return False
    
    @staticmethod
    def optimize_numpy_operations() -> bool:
        """
        Optimize NumPy operations for better performance.
        
        Returns:
            bool: True if optimization was applied, False otherwise
        """
        logger.info("Optimizing NumPy operations")
        
        # Set number of threads for NumPy operations
        n_threads = os.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
        
        logger.info(f"Set NumPy threads to {n_threads}")
        
        return True
    
    @staticmethod
    def optimize_pandas_operations() -> bool:
        """
        Optimize Pandas operations for better performance.
        
        Returns:
            bool: True if optimization was applied, False otherwise
        """
        logger.info("Optimizing Pandas operations")
        
        try:
            import pandas as pd
            
            # Set Pandas options for better performance
            pd.set_option('mode.chained_assignment', None)  # Disable SettingWithCopyWarning
            pd.set_option('compute.use_bottleneck', True)   # Use bottleneck for faster computations
            pd.set_option('compute.use_numexpr', True)      # Use numexpr for faster computations
            
            logger.info("Set Pandas performance options")
            
            return True
            
        except ImportError:
            logger.warning("Pandas not available, skipping Pandas optimization")
            return False
    
    @staticmethod
    def optimize_system_resources() -> bool:
        """
        Optimize system resource usage.
        
        Returns:
            bool: True if optimization was applied, False otherwise
        """
        logger.info("Optimizing system resources")
        
        # Set process priority to high if possible
        try:
            import psutil
            p = psutil.Process(os.getpid())
            
            # Windows
            if os.name == 'nt':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("Set process priority to HIGH on Windows")
            # Unix-like
            else:
                p.nice(-5)  # Increase priority (lower is better)
                logger.info("Increased process priority on Unix-like system")
                
        except ImportError as e:
            logger.warning(f"psutil not available, skipping process priority setting: {e}")
        except (PermissionError, Exception) as e:
            logger.warning(f"Could not set process priority: {e}")
        
        # Set memory limits if configured
        memory_limit_gb = os.environ.get('RL_MEMORY_LIMIT_GB')
        if memory_limit_gb:
            try:
                import resource
                
                # Convert GB to bytes
                memory_limit_bytes = int(float(memory_limit_gb) * 1024 * 1024 * 1024)
                
                # Set memory limit (soft and hard)
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                logger.info(f"Set memory limit to {memory_limit_gb} GB")
                
            except (ImportError, ValueError, resource.error) as e:
                logger.warning(f"Could not set memory limit: {e}")
        
        return True
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information for optimization decisions.
        
        Returns:
            Dict[str, Any]: System information
        """
        info = {
            'cpu_count': os.cpu_count(),
            'platform': os.name,
        }
        
        # Add GPU information if available
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        except ImportError:
            info['cuda_available'] = False
        
        # Add memory information if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['memory_total'] = memory.total / 1024**3  # GB
            info['memory_available'] = memory.available / 1024**3  # GB
            info['memory_percent'] = memory.percent
        except ImportError:
            pass
        
        return info
    
    @staticmethod
    def apply_all_optimizations(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all available optimizations.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict[str, Any]: Optimized configuration
        """
        logger.info("Applying all computation optimizations")
        
        # Get system information
        system_info = ComputationOptimizer.get_system_info()
        logger.info(f"System info: {system_info}")
        
        # Apply optimizations based on system capabilities
        if system_info.get('cuda_available', False):
            ComputationOptimizer.optimize_for_gpu()
        else:
            ComputationOptimizer.optimize_for_cpu()
        
        # Apply general optimizations
        ComputationOptimizer.optimize_tensor_operations()
        ComputationOptimizer.optimize_numpy_operations()
        ComputationOptimizer.optimize_pandas_operations()
        ComputationOptimizer.optimize_system_resources()
        
        # Add optimization settings to config
        if 'optimization' not in config:
            config['optimization'] = {}
        
        config['optimization']['gpu_enabled'] = system_info.get('cuda_available', False)
        config['optimization']['cpu_count'] = system_info.get('cpu_count', 1)
        config['optimization']['memory_total_gb'] = system_info.get('memory_total', 0)
        
        logger.info("Applied all computation optimizations")
        
        return config