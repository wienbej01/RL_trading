#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from src.utils.config_loader import Settings

def normalize_one(fp: Path) -> bool:
    try:
        df = pd.read_parquet(fp)
        # Accept common time keys
        time_candidates = [c for c in ["timestamp","datetime","time","dt","ts","t"] if c in df.columns]
        if not time_candidates:
            # allow DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"[skip] {fp} (no timestamp col/index)")
                return False
            df = df.reset_index().rename(columns={"index":"timestamp"})
        else:
            ts_col = time_candidates[0]
            if ts_col != "timestamp":
                df = df.rename(columns={ts_col: "timestamp"})

        # Coerce to UTC (ms assumed) then set index tz and sort
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", unit="ms")
        df = df.loc[ts.notna()].copy()
        df["timestamp"] = ts
        df = df.sort_values("timestamp").set_index("timestamp")
        # idempotent write
        df.to_parquet(fp)
        print(f"[ok] {fp}")
        return True
    except Exception as e:
        print(f"[err] {fp}: {e}")
        return False

def main():
    settings = Settings(config_path="configs/settings.yaml")
    raw_dir = Path(settings.get("paths","polygon_raw_dir", default="data/polygon/historical"))
    sym = "SPY"
    files = sorted((raw_dir / f"symbol={sym}").rglob("data.parquet"))
    if not files:
        print(f"No files under {raw_dir}/symbol={sym}")
        return
    ok = 0
    for fp in files:
        ok += normalize_one(fp)
    print(f"Fixed {ok}/{len(files)} files")

if __name__ == "__main__":
    main()
