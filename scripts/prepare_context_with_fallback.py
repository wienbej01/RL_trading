#!/usr/bin/env python3
"""
Prepare SPY/QQQ/VIX context with fallback strategies:

- Try upgraded US stocks downloader for SPY/QQQ minute aggregates (5y from today)
- Fallback to original rate-limited downloader if needed
- For VIX term structure, try Polygon indices; fallback to CBOE/FRED daily via vix_loader

Usage examples:
  export POLYGON_API_KEY=YOUR_KEY
  # 5 years back from today (auto clamp by upgraded API availability)
  python scripts/prepare_context_with_fallback.py --years 5 --concurrency 20
  # Or explicit date range (falls back to 2y if original is limited)
  python scripts/prepare_context_with_fallback.py --start-date 2023-01-01 --end-date 2025-09-04 --concurrency 20
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("→", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def compute_range(years: int | None, start: str | None, end: str | None) -> tuple[str, str]:
    if years:
        today = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=-4)))
        end_date = today.date().isoformat()
        start_date = (today - dt.timedelta(days=365 * years)).date().isoformat()
        return start_date, end_date
    assert start and end, "Either --years or both --start-date and --end-date are required"
    return start, end


def main():
    ap = argparse.ArgumentParser(description="Prepare SPY/QQQ/VIX with fallback downloaders")
    ap.add_argument("--years", type=int, default=None, help="Years back from today (e.g., 5)")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    start_date, end_date = compute_range(args.years, args.start_date, args.end_date)

    # 1) SPY/QQQ via upgraded downloader first
    rc = run([sys.executable, "scripts/collect_polygon_us_stocks.py",
              "--tickers", "SPY,QQQ",
              "--start-date", start_date,
              "--end-date", end_date,
              "--types", "aggregates",
              "--concurrency", str(args.concurrency)])
    if rc != 0:
        # Fallback to original rate-limited script (day-wise); supports "--symbols"
        print("Upgraded downloader failed; falling back to original rate-limited collector for SPY, QQQ")
        rc1 = run([sys.executable, "scripts/collect_polygon_data.py",
                   "--symbols", "SPY",
                   "--start-date", start_date, "--end-date", end_date])
        rc2 = run([sys.executable, "scripts/collect_polygon_data.py",
                   "--symbols", "QQQ",
                   "--start-date", start_date, "--end-date", end_date])
        if rc1 != 0 or rc2 != 0:
            print("Warning: original collector also failed for SPY/QQQ")

    # Aggregate SPY/QQQ
    for tk in ("SPY", "QQQ"):
        run([sys.executable, "scripts/aggregate_us_stock_data.py",
             "--ticker", tk, "--start-date", start_date, "--end-date", end_date])

    # 2) VIX term structure: try Polygon indices first (minute -> daily fallback inside)
    rc = run([sys.executable, "scripts/download_vix_term_structure.py",
              "--start-date", start_date, "--end-date", end_date, "--output", "data/external/vix.parquet"])
    if rc != 0:
        # Fallback to vix_loader (CBOE/FRED daily) from our data module
        print("Polygon VIX download failed; falling back to CBOE/FRED daily via vix_loader")
        code = """
import pandas as pd
from src.data.vix_loader import VIXDataLoader
from src.utils.config_loader import Settings
v = VIXDataLoader(Settings())
try:
    df = v.get_vix_data_for_date_range('%s','%s')
    out='data/external/vix.parquet'
    from pathlib import Path
    Path('data/external').mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print('Wrote', out, 'with columns:', list(df.columns))
except Exception as e:
    print('VIX fallback failed:', e)
    import pandas as pd
    from pathlib import Path
    Path('data/external').mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_parquet('data/external/vix.parquet')
    print('Wrote empty data/external/vix.parquet')
""" % (start_date, end_date)
        run([sys.executable, "-c", code])

    print("Prepared context files: data/raw/SPY_1min.parquet, data/raw/QQQ_1min.parquet, data/external/vix.parquet (may be empty)")


if __name__ == "__main__":
    main()
