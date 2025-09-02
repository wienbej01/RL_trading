#!/usr/bin/env python3
"""
Aggregate SPY data from partitioned files into a single parquet file for training.
"""

import pandas as pd
import glob
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_spy_data(start_date: str = "2024-01-01", end_date: str = "2025-06-30"):
    """
    Aggregate SPY data from partitioned parquet files.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    base_path = Path("rl-intraday/data/polygon/historical/symbol=SPY")
    output_file = Path("rl-intraday/data/raw/spy_1min.parquet")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Aggregating SPY data from {start_date} to {end_date}")

    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    end_ts = pd.Timestamp(end_date, tz="America/New_York")

    # Find all data files
    data_files = sorted(base_path.glob("year=*/month=*/day=*/data.parquet"))

    logger.info(f"Found {len(data_files)} data files")

    if not data_files:
        raise ValueError("No data files found")

    # Load and concatenate data
    dfs = []
    for file_path in data_files:
        try:
            df = pd.read_parquet(file_path)
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York", ambiguous='infer')
            else:
                df.index = df.index.tz_convert("America/New_York")

            # Filter by date range
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            if mask.any():
                dfs.append(df[mask])
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    if not dfs:
        raise ValueError("No data found in the specified date range")

    # Concatenate all data
    combined_df = pd.concat(dfs, ignore_index=False)
    combined_df = combined_df.sort_index()

    # Remove duplicates
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in combined_df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {combined_df.columns.tolist()}")

    # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
    combined_df = combined_df.between_time('09:30', '16:00')

    logger.info(f"Final dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    # Save to parquet
    combined_df.to_parquet(output_file)
    logger.info(f"Saved aggregated data to {output_file}")

    return combined_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SPY data for training")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-06-30", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    aggregate_spy_data(args.start_date, args.end_date)