#!/usr/bin/env python3
"""
Download VIX data from Yahoo Finance (yfinance).

Fetches ^VIX (1M), optionally ^VIX9D and ^VIX3M if available, and writes a
unified parquet at data/external/vix.parquet with columns:
  - vix    (Close)
  - vix9d  (Close, if downloaded)
  - vix3m  (Close, if downloaded)

Timestamps normalized to America/New_York.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    return df


def _extract_close(df: pd.DataFrame, ticker_hint: str | None = None) -> pd.Series | None:
    """Extract close series from a yfinance DataFrame, handling MultiIndex columns.
    Returns a Series or None if not found.
    """
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # level 0 likely OHLCV, level 1 ticker
            try:
                s = df.xs('Close', level=0, axis=1)
                if isinstance(s, pd.DataFrame):
                    if ticker_hint and ticker_hint in s.columns:
                        s = s[ticker_hint]
                    else:
                        s = s.iloc[:, 0]
            except Exception:
                # fallback: select first column that contains 'Close'
                close_cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == 'Close']
                if close_cols:
                    s = df[close_cols[0]]
                else:
                    return None
        else:
            if 'Close' in df.columns:
                s = df['Close']
            else:
                return None
        s = pd.to_numeric(pd.Series(s), errors='coerce')
        return s
    except Exception:
        return None


def download_vix_data(start_date: str = "2020-01-01", end_date: str = "2025-07-01",
                      include_9d: bool = True, include_3m: bool = True):
    """
    Download VIX data and save it to a parquet file.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    output_file = Path("data/external/vix.parquet")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading VIX data from {start_date} to {end_date}")

    # Download main VIX (^VIX)
    vix_df = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=False, progress=False)
    if vix_df.empty:
        raise ValueError("No VIX (^VIX) data downloaded. Check the date range and ticker.")
    vix_df = _to_ny_index(vix_df)
    vix = _extract_close(vix_df, ticker_hint='^VIX')
    if vix is not None:
        vix.name = 'vix'

    # Optional 9D and 3M
    vix9d = None
    vix3m = None
    if include_9d:
        try:
            d9 = yf.download("^VIX9D", start=start_date, end=end_date, auto_adjust=False, progress=False)
            if not d9.empty:
                d9 = _to_ny_index(d9)
                vix9d = _extract_close(d9, ticker_hint='^VIX9D')
                if vix9d is not None:
                    vix9d.name = 'vix9d'
        except Exception:
            pass
    if include_3m:
        try:
            d3 = yf.download("^VIX3M", start=start_date, end=end_date, auto_adjust=False, progress=False)
            if not d3.empty:
                d3 = _to_ny_index(d3)
                vix3m = _extract_close(d3, ticker_hint='^VIX3M')
                if vix3m is not None:
                    vix3m.name = 'vix3m'
        except Exception:
            pass

    # Build unified frame
    out_idx = vix.index if vix is not None else None
    if vix9d is not None:
        out_idx = vix9d.index if out_idx is None else out_idx.union(vix9d.index)
    if vix3m is not None:
        out_idx = vix3m.index if out_idx is None else out_idx.union(vix3m.index)
    out = pd.DataFrame(index=out_idx)
    if vix is not None:
        out['vix'] = vix.reindex(out.index)
    if vix9d is not None:
        out['vix9d'] = vix9d.reindex(out.index)
    if vix3m is not None:
        out['vix3m'] = vix3m.reindex(out.index)

    logger.info(f"Downloaded VIX rows: {len(out)} with columns: {list(out.columns)}")

    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_file)
    logger.info(f"Saved VIX data to {output_file}")

    return out

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VIX data (Yahoo Finance)")
    parser.add_argument("--start-date", default="2023-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-07-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-9d", action='store_true', help="Do not fetch ^VIX9D")
    parser.add_argument("--no-3m", action='store_true', help="Do not fetch ^VIX3M")

    args = parser.parse_args()

    download_vix_data(args.start_date, args.end_date, include_9d=not args.no_9d, include_3m=not args.no_3m)
