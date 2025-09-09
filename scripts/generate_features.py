#!/usr/bin/env python3
"""
Generate features using FeaturePipeline and save to parquet.
Usage: python scripts/generate_features.py --config configs/settings.yaml --data data/raw/BBVA_1min.parquet --output data/features/BBVA_features.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any

# Add repo root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.pipeline import FeaturePipeline
from src.utils.config_loader import Settings


def main():
    parser = argparse.ArgumentParser(description="Generate features for RL training")
    parser.add_argument("--config", default="configs/settings.yaml", help="Config file path")
    parser.add_argument("--data", required=True, help="Path to raw OHLCV parquet")
    parser.add_argument("--output", required=True, help="Output path for features parquet")
    args = parser.parse_args()

    # Load settings
    settings = Settings.from_paths(args.config)

    # Load raw data
    print(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # Instantiate pipeline
    pipeline = FeaturePipeline(settings.get("features", {}))

    # Generate features
    print("Generating features...")
    features = pipeline.fit_transform(df)

    # Save features
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output)
    print(f"Features saved to {args.output} with {len(features.columns)} columns")


if __name__ == "__main__":
    main()