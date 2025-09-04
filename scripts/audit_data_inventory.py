#!/usr/bin/env python3
"""
Audit data inventory in the repo to understand what is available for training/backtesting.

Scans typical locations and reports:
- Polygon+ US stocks aggregates (per-day partitions) and coverage per symbol
- Original Polygon historical (per-day partitions)
- Raw single-file minute datasets under data/raw/<TICKER>_1min.parquet
- Feature stores under data/features/<TICKER>_features.parquet
- External context under data/external (e.g., vix.parquet with available columns)
- Models (models/<TICKER>_trained_model(.zip)) and vecnormalize.pkl presence

Usage:
  python scripts/audit_data_inventory.py [--ticker BBVA] [--detailed]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import sys


def fmt_range(idx: pd.DatetimeIndex | pd.Index) -> str:
    if len(idx) == 0:
        return "<empty>"
    try:
        lo = idx.min()
        hi = idx.max()
        return f"{lo} .. {hi}"
    except Exception:
        return "<n/a>"


def list_symbols(root: Path, pattern: str = "symbol=*") -> list[str]:
    syms = []
    for p in root.glob(pattern):
        if p.is_dir():
            try:
                syms.append(p.name.split("=", 1)[1])
            except Exception:
                pass
    return sorted(set(syms))


def audit_polygon_plus_us_stocks(ticker: Optional[str], detailed: bool):
    base = Path("data/polygon_plus/us_stocks/aggregates")
    if not base.exists():
        return
    symbols = [ticker] if ticker else list_symbols(base)
    if not symbols:
        return
    print("\n== Polygon+ US Stocks Aggregates ==")
    for sym in symbols:
        root = base / f"symbol={sym}"
        days = list(root.glob("year=*/month=*/day=*/data.parquet"))
        print(f"  {sym}: {len(days)} day files")
        if detailed:
            # Show a small sample with dates
            for p in sorted(days)[:3]:
                parts = p.parts
                y = [x for x in parts if x.startswith('year=')]
                m = [x for x in parts if x.startswith('month=')]
                d = [x for x in parts if x.startswith('day=')]
                yd = f"{y[0] if y else ''}/{m[0] if m else ''}/{d[0] if d else ''}"
                print(f"    - {yd}")


def audit_polygon_original(ticker: Optional[str], detailed: bool):
    base = Path("data/polygon/historical")
    if not base.exists():
        return
    symbols = [ticker] if ticker else list_symbols(base)
    if not symbols:
        return
    print("\n== Original Polygon Historical ==")
    for sym in symbols:
        root = base / f"symbol={sym}"
        days = list(root.glob("year=*/month=*/day=*/data.parquet"))
        print(f"  {sym}: {len(days)} day files")
        if detailed:
            for p in sorted(days)[:3]:
                parts = p.parts
                y = [x for x in parts if x.startswith('year=')]
                m = [x for x in parts if x.startswith('month=')]
                d = [x for x in parts if x.startswith('day=')]
                yd = f"{y[0] if y else ''}/{m[0] if m else ''}/{d[0] if d else ''}"
                print(f"    - {yd}")


def brief_parquet_info(path: Path) -> tuple[int, str, str]:
    try:
        df = pd.read_parquet(path)
        n = len(df)
        rng = fmt_range(df.index)
        tz = getattr(df.index, 'tz', None)
        tzs = str(tz) if tz is not None else "naive"
        return n, rng, tzs
    except Exception as e:
        return -1, f"<error: {e}>", "?"


def audit_raw_single_files(ticker: Optional[str]):
    base = Path("data/raw")
    if not base.exists():
        return
    files = sorted(base.glob("*_1min.parquet"))
    if ticker:
        files = [p for p in files if p.name.startswith(ticker.upper() + "_")]
    if not files:
        return
    print("\n== Raw Single-File Minute Datasets ==")
    for p in files:
        n, rng, tzs = brief_parquet_info(p)
        print(f"  {p.name}: rows={n} range=[{rng}] tz={tzs}")


def audit_feature_stores(ticker: Optional[str], detailed: bool):
    base = Path("data/features")
    if not base.exists():
        return
    files = sorted(base.glob("*.parquet"))
    if ticker:
        files = [p for p in files if p.name.startswith(ticker.upper() + "_") or ticker.upper() in p.name]
    if not files:
        return
    print("\n== Feature Stores ==")
    for p in files:
        try:
            df = pd.read_parquet(p)
            cols = list(df.columns)
            print(f"  {p.name}: rows={len(df)} cols={len(cols)}")
            if detailed:
                sample = ", ".join(cols[:15]) + (" ..." if len(cols) > 15 else "")
                print(f"    - columns: {sample}")
        except Exception as e:
            print(f"  {p.name}: <error: {e}>")


def audit_external():
    base = Path("data/external")
    if not base.exists():
        return
    print("\n== External Context ==")
    vix = base / "vix.parquet"
    if vix.exists():
        try:
            df = pd.read_parquet(vix)
            cols = list(df.columns)
            print(f"  vix.parquet: rows={len(df)} cols={len(cols)} range=[{fmt_range(df.index)}]")
            print(f"    - columns: {cols}")
        except Exception as e:
            print(f"  vix.parquet: <error: {e}>")


def audit_models(ticker: Optional[str]):
    base = Path("models")
    if not base.exists():
        return
    files = sorted(base.glob("*_trained_model*"))
    if ticker:
        files = [p for p in files if p.name.startswith(ticker.upper() + "_")]
    if not files:
        return
    print("\n== Models ==")
    for p in files:
        vec = p.parent / "vecnormalize.pkl"
        print(f"  {p.name}: vecnormalize={'yes' if vec.exists() else 'no'}")


def main():
    ap = argparse.ArgumentParser(description="Audit repository data inventory")
    ap.add_argument("--ticker", default=None, help="Optional ticker filter (e.g., BBVA)")
    ap.add_argument("--detailed", action="store_true", help="Show more details (first few day partitions, columns sample)")
    args = ap.parse_args()

    print("Data Inventory Report")
    print("=====================")
    audit_polygon_plus_us_stocks(args.ticker, args.detailed)
    audit_polygon_original(args.ticker, args.detailed)
    audit_raw_single_files(args.ticker)
    audit_feature_stores(args.ticker, args.detailed)
    audit_external()
    audit_models(args.ticker)


if __name__ == "__main__":
    sys.exit(main())

