"""
Compatibility shim: export OHLCV_MICRO from microstructure.

Some configs or callers may import from `src.features.microstructure_ohlcv`.
This module re-exports the mapping defined in `microstructure.py`.
"""
from .microstructure import OHLCV_MICRO  # re-export

__all__ = ["OHLCV_MICRO"]

