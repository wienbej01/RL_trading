#!/usr/bin/env python3
import os, sys, time, json
from pathlib import Path
import datetime as dt
import pandas as pd
import requests

ROOT = Path("/home/jacobw/RL_trading/rl-intraday")
DATA_ROOT = ROOT / "data/polygon/historical/symbol=SPY"
START = pd.Timestamp("2024-01-01", tz="America/New_York")
END   = pd.Timestamp("2025-06-30", tz="America/New_York")
RTH_MIN_ROWS = 375
RATE_LIMIT_CALLS = 5
RATE_LIMIT_WINDOW = 60.0

def require_api_key():
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        print("[ERROR] POLYGON_API_KEY not set.")
        sys.exit(2)
    return key

def ensure_deps():
    try:
        import pandas_market_calendars as mcal  # noqa
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_market_calendars"])
ensure_deps()
import pandas_market_calendars as mcal  # noqa

def nyse_days(start, end):
    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=start.date(), end_date=end.date())
    days = pd.to_datetime(sched.index).tz_localize("America/New_York")
    return [d.normalize() for d in days]

def day_parquet_path(day: pd.Timestamp) -> Path:
    y, m, d = day.year, f"{day.month:02d}", f"{day.day:02d}"
    return DATA_ROOT / f"year={y}/month={m}/day={d}/data.parquet"

def rows_if_exists(p: Path) -> int:
    if not p.exists(): return 0
    try:
        return len(pd.read_parquet(p))
    except Exception:
        return 0

def fetch_day(symbol: str, day: pd.Timestamp, api_key: str) -> pd.DataFrame:
    start = (day.tz_convert("America/New_York") + pd.Timedelta("09:30")).tz_convert("UTC")
    end   = (day.tz_convert("America/New_York") + pd.Timedelta("16:00")).tz_convert("UTC")
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    results = data.get("results", [])
    if not results:
        return pd.DataFrame(columns=["t","o","h","l","c","v","n","vw"])
    df = pd.DataFrame(results)
    # Convert ‘t’ (ms since epoch) to tz-aware NY time
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("ts").sort_index()
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap","n":"transactions"})
    df = df[["open","high","low","close","volume","vwap","transactions"]]
    # keep only RTH 09:30-16:00
    df = df.between_time("09:30","16:00")
    return df

class RateLimiter:
    def __init__(self, calls, window):
        self.calls = calls
        self.window = window
        self.times = []
    def wait(self):
        now = time.time()
        self.times = [t for t in self.times if now - t < self.window]
        if len(self.times) >= self.calls:
            sleep_for = self.window - (now - self.times[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.times.append(time.time())

def main():
    api_key = require_api_key()
    rl = RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_WINDOW)
    days = nyse_days(START, END)

    to_fetch = []
    for d in days:
        pq = day_parquet_path(d)
        n = rows_if_exists(pq)
        if n >= RTH_MIN_ROWS:
            continue
        to_fetch.append((d, n))

    print(f"[PLAN] {len(to_fetch)} trading days need fetch/refetch.")

    for idx, (d, existing) in enumerate(to_fetch, 1):
        y, m, dd = d.year, f"{d.month:02d}", f"{d.day:02d}"
        out = day_parquet_path(d)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Respect rate limit (5/min)
        rl.wait()
        for attempt in range(1, 5):
            try:
                df = fetch_day("SPY", d, api_key)
                if len(df) < RTH_MIN_ROWS:
                    print(f"[WARN] {d.date()} fetched {len(df)} rows (<{RTH_MIN_ROWS}). Keeping but will remain 'partial'.")
                df.to_parquet(out)
                print(f"[OK] {idx}/{len(to_fetch)} {d.date()} → rows={len(df)}  path={out}")
                break
            except Exception as e:
                print(f"[ERR] {d.date()} attempt {attempt}: {e}")
                time.sleep(2 * attempt)
        else:
            print(f"[FAIL] {d.date()} after retries; continuing")

    # Summary rerun
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/find_missing_intraday_spy.py"], check=False)
    except Exception as e:
        print(f"[WARN] Could not re-run gap detector: {e}")

if __name__ == "__main__":
    sys.exit(main())