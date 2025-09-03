#!/usr/bin/env python3
"""
Download VIX data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_vix_data(start_date: str = "2023-12-01", end_date: str = "2025-07-01"):
    """
    Download VIX data and save it to a parquet file.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    output_file = Path("rl-intraday/data/external/vix.parquet")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading VIX data from {start_date} to {end_date}")

    # Download VIX data
    vix_data = yf.download("^VIX", start=start_date, end=end_date)

    if vix_data.empty:
        raise ValueError("No VIX data downloaded. Check the date range and ticker.")

    # Select relevant columns and rename
    cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
    vix_data = vix_data[[col for col in cols_to_keep if col in vix_data.columns]]
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = ['_'.join(col).strip() for col in vix_data.columns.values]
    vix_data.columns = [f"vix_{col.lower()}" for col in vix_data.columns]

    logger.info(f"Downloaded {len(vix_data)} rows of VIX data.")

    # Save to parquet
    vix_data.to_parquet(output_file)
    logger.info(f"Saved VIX data to {output_file}")

    return vix_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VIX data")
    parser.add_argument("--start-date", default="2023-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-07-01", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    download_vix_data(args.start_date, args.end_date)
