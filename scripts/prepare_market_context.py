#!/usr/bin/env python3
"""
Prepare market context datasets (SPY, QQQ, VIX term structure) for SMT & volatility features.

Steps:
  1) Download SPY, QQQ minute aggregates via upgraded downloader (parallel)
  2) Aggregate to single Parquet per ticker under data/raw/
  3) Download VIX term structure (VIX, VIX9D, VIX3M) with minute or daily fallback

Usage:
  export POLYGON_API_KEY=YOUR_KEY
  python scripts/prepare_market_context.py --start-date 2020-01-01 --end-date 2025-09-04 --concurrency 20
"""
import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print("→", " ".join(cmd))
    r = subprocess.run(cmd, check=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="Prepare SPY/QQQ/VIX context data for features")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    # 1) Download SPY and QQQ via parallel downloader
    run([sys.executable, "scripts/collect_polygon_us_stocks.py",
         "--tickers", "SPY,QQQ",
         "--start-date", args.start_date,
         "--end-date", args.end_date,
         "--types", "aggregates",
         "--concurrency", str(args.concurrency)])

    # 2) Aggregate to single parquet files
    for tk in ("SPY", "QQQ"):
        run([sys.executable, "scripts/aggregate_us_stock_data.py",
             "--ticker", tk,
             "--start-date", args.start_date,
             "--end-date", args.end_date])

    # 3) Download VIX term structure
    run([sys.executable, "scripts/download_vix_term_structure.py",
         "--start-date", args.start_date,
         "--end-date", args.end_date,
         "--output", "data/external/vix.parquet"])

    print("Market context prepared: data/raw/SPY_1min.parquet, data/raw/QQQ_1min.parquet, data/external/vix.parquet")


if __name__ == "__main__":
    main()

