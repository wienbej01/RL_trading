#!/usr/bin/env python3
"""
Ingest external VIX data (e.g., manually downloaded from Databento/FRED/CBOE)
and convert to a unified parquet at data/external/vix.parquet, with best-effort
column detection for:
  - vix (1M) close
  - vix9d close (optional)
  - vix3m close (optional)

Supports CSV or Parquet inputs. Forward-fills to create a continuous series.

Usage examples:
  python scripts/ingest_external_vix.py --inputs data/external/my_vix.csv
  python scripts/ingest_external_vix.py --inputs data/external/vix.parquet data/external/vix3m.parquet

You can also pass ticker hints for mapping:
  --map vix=close --map vix9d=last --map vix3m=Close
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    # default to CSV
    return pd.read_csv(path)


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    # try known timestamp columns
    for c in ('timestamp','date','datetime','time','dt','ts'):
        if c in df.columns:
            idx = pd.to_datetime(df[c], utc=True, errors='coerce')
            break
    else:
        # try index
        idx = pd.to_datetime(df.index, utc=True, errors='coerce')
    df = df.loc[idx.notna()].copy()
    df.index = idx[idx.notna()]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('America/New_York')
    df = df.sort_index()
    return df


def pick_close(df: pd.DataFrame, hint: str | None) -> pd.Series | None:
    if hint and hint in df.columns:
        return pd.to_numeric(df[hint], errors='coerce')
    # try common names
    for cand in ('close','Close','vix','VIX','last','PX_LAST','PRICE','settle','SETTLE'):
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors='coerce')
    # if single numeric column, take it
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return pd.to_numeric(df[num_cols[0]], errors='coerce')
    return None


def main():
    ap = argparse.ArgumentParser(description="Ingest external VIX data into unified parquet")
    ap.add_argument("--inputs", nargs='+', required=True, help="Input CSV/Parquet files (one or more)")
    ap.add_argument("--map", action='append', default=[], help="Column map hint like vix=close or vix3m=Close")
    ap.add_argument("--output", default="data/external/vix.parquet")
    args = ap.parse_args()

    # parse hints
    hints: Dict[str,str] = {}
    for m in args.map:
        if '=' in m:
            k,v = m.split('=',1)
            hints[k.strip().lower()] = v.strip()

    series: Dict[str,pd.Series] = {}
    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"Skipping missing: {p}")
            continue
        try:
            df = load_any(p)
            df = normalize_index(df)
            for tag in ('vix','vix9d','vix3m'):
                if tag in series:
                    continue
                s = pick_close(df, hints.get(tag))
                if s is not None:
                    series[tag] = pd.Series(s.values, index=df.index)
        except Exception as e:
            print(f"Failed to ingest {p}: {e}")

    if not series:
        raise SystemExit("No usable VIX series found in inputs")

    # build dataframe and save
    idx = sorted(set().union(*[s.index for s in series.values()]))
    out = pd.DataFrame(index=idx)
    for k,s in series.items():
        out[k] = s.reindex(out.index)
    out = out.sort_index()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    print(f"Wrote {args.output} with columns: {list(out.columns)} range: {out.index.min()} .. {out.index.max()}")


if __name__ == "__main__":
    main()

