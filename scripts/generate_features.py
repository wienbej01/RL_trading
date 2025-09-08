#!/usr/bin/env python3
"""
Generate features for BBVA 2020 validation.

Usage: python scripts/generate_features.py

Requires: data/raw/BBVA_1min.parquet exists (from polygon_ingestor).
Outputs: data/features/BBVA_features.parquet with 8 spec features.
Deterministic, explicit errors.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Absolute imports from src
sys.path.insert(0, str(Path(__file__).parent.parent))  # add root
from src.features.pipeline import FeaturePipeline
from src.data.data_loader import UnifiedDataLoader

def main():
    """Generate features."""
    try:
        # Load consolidated BBVA data (existing)
        df = pd.read_parquet('data/raw/BBVA_1min.parquet')
        if df.empty:
            raise ValueError("No BBVA data loaded; check data/raw/BBVA_1min.parquet")

        # Load config for pipeline
        from src.utils.config_loader import Settings
        settings = Settings.from_paths('configs/settings.yaml')
        config = settings.to_dict().get('features', {})

        # Generate features
        pipeline = FeaturePipeline(config)
        features = pipeline.transform(df)

        # Save
        out_path = Path('data/features/BBVA_features.parquet')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out_path)
        print(f"Generated features saved to {out_path}; shape: {features.shape}")
        print(features.columns.tolist())  # verify 8 features

    except ImportError as e:
        print(f"Import error: {e} - ensure src/ structure with __init__.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating features: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()