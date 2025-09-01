#!/usr/bin/env python3
import os, sys, glob, datetime as dt
from pathlib import Path
import pandas as pd

ROOT = Path("/home/jacobw/RL_trading/rl-intraday")
DATA_ROOT = ROOT / "data/polygon/historical/symbol=SPY"
START = pd.Timestamp("2024-01-01", tz="America/New_York")
END   = pd.Timestamp("2025-06-30", tz="America/New_York")

def ensure_deps():
    try:
        import pandas_market_calendars as mcal  # noqa
        return
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_market_calendars"])
ensure_deps()
import pandas_market_calendars as mcal  # noqa

def nyse_days(start, end):
    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=start.date(), end_date=end.date())
    days = pd.to_datetime(sched.index).tz_localize("America/New_York")
    return [d.normalize() for d in days]

def rows_in_parquet(pq_path: Path) -> int:
    try:
        df = pd.read_parquet(pq_path)
        return len(df)
    except Exception:
        return 0

def main():
    days = nyse_days(START, END)
    missing = []
    partial = []
    for d in days:
        y = d.year; m = f"{d.month:02d}"; dd = f"{d.day:02d}"
        pq = DATA_ROOT / f"year={y}/month={m}/day={dd}/data.parquet"
        if not pq.exists():
            missing.append(d.date())
            continue
        n = rows_in_parquet(pq)
        if n < 375:  # treat as partial (RTH baseline)
            partial.append((d.date(), n))

    print("=== SPY 1m Coverage Check (NYSE RTH baseline ≥375 rows) ===")
    print(f"Range: {START.date()} → {END.date()}   Trading days: {len(days)}")
    print(f"Missing days: {len(missing)}")
    print(f"Partial days (<375 rows): {len(partial)}")
    if missing:
        print("\n-- Missing days --")
        for d in missing: print(d)
    if partial:
        print("\n-- Partial days --")
        for d, n in partial: print(f"{d}  rows={n}")

    # Month-bucketed plan
    from collections import defaultdict
    plan = defaultdict(list)
    for d in missing: plan[(d.year, d.month)].append((d, "missing"))
    for d, n in partial: plan[(d.year, d.month)].append((d, f"partial({n})"))

    print("\n=== TODO (by year-month) ===")
    for (y, m), items in sorted(plan.items()):
        tag = f"{y}-{m:02d}"
        kinds = ", ".join([i[1] for i in items])
        print(f"{tag}: {len(items)} days → {kinds.count('missing')} missing, {sum(1 for _,k in items if k.startswith('partial'))} partial")

if __name__ == "__main__":
    sys.exit(main())
