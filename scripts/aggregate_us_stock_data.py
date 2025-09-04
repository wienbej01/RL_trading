#!/usr/bin/env python3
"""
Aggregate a US stock's minute aggregates (downloaded via collect_polygon_us_stocks.py)
into a single RTH-filtered Parquet for training/backtesting.

Input layout expected:
  data/polygon_plus/us_stocks/aggregates/symbol=<TICKER>/year=YYYY/month=MM/day=DD/data.parquet

Output:
  data/raw/<TICKER>_1min.parquet

Usage:
  python scripts/aggregate_us_stock_data.py --ticker BBVA --start-date 2020-01-01 --end-date 2025-09-04
"""
import argparse
from pathlib import Path
import pandas as pd


def aggregate_ticker(ticker: str, start_date: str, end_date: str) -> Path:
    ticker = ticker.upper()
    base_path = Path("data/polygon_plus/us_stocks/aggregates") / f"symbol={ticker}"
    if not base_path.exists():
        raise FileNotFoundError(f"Aggregates not found at {base_path}. Run collect_polygon_us_stocks.py first.")

    out_path = Path("data/raw") / f"{ticker}_1min.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    end_ts = pd.Timestamp(end_date, tz="America/New_York")

    files = sorted(base_path.glob("year=*/month=*/day=*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"No day files under {base_path}")

    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York", ambiguous='infer')
            else:
                df.index = df.index.tz_convert("America/New_York")
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            if mask.any():
                dfs.append(df.loc[mask])
        except Exception:
            continue

    if not dfs:
        raise ValueError("No data in selected range")

    all_df = pd.concat(dfs, axis=0).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep='first')]
    # RTH filter and final tidy
    all_df = all_df.between_time('09:30', '16:00')

    all_df.to_parquet(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Aggregate Polygon+ US stock aggregates into single Parquet")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    args = ap.parse_args()

    out = aggregate_ticker(args.ticker, args.start_date, args.end_date)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

