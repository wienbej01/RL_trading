#!/usr/bin/env python3
"""
High‑throughput Polygon US Stocks downloader for upgraded subscription tiers.

Features:
- Parallel async downloads (configurable concurrency)
- Minute aggregates (RTH filtered optional), Reference, Fundamentals, Corporate Actions, Snapshots
- 5y+ range handling with day-wise chunking to maximize throughput
- Robust I/O layout under data/polygon_plus/us_stocks/

Usage examples:
  export POLYGON_API_KEY=YOUR_KEY
  python scripts/collect_polygon_us_stocks.py \
      --tickers BBVA \
      --start-date 2020-01-01 --end-date 2025-09-04 \
      --types aggregates fundamentals reference corp_actions snapshot \
      --concurrency 20

Notes:
- Keep original scripts/collect_polygon_data.py for other asset classes (forex, futures, indices, options).
- This script targets US stocks endpoints with higher concurrency suitable for unlimited API.
"""
import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiohttp
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://api.polygon.io"


@dataclass
class DownloadTask:
    ticker: str
    date: datetime


def ny_tz_now_minus(minutes: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def daterange_days(start: datetime, end: datetime) -> List[datetime]:
    cur = start
    days = []
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


async def fetch_json(session: aiohttp.ClientSession, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_day_aggregates(session: aiohttp.ClientSession, api_key: str, ticker: str, day: datetime) -> pd.DataFrame:
    day_str = day.strftime("%Y-%m-%d")
    ep = f"/v2/aggs/ticker/{ticker}/range/1/minute/{day_str}/{day_str}"
    js = await fetch_json(session, ep, {"apiKey": api_key, "limit": 50000})
    results = js.get("results") or []
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    # Normalize Polygon fields
    # t=ms, o/h/l/c, v (volume), vw (vwap), n (transactions)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vw": "vwap", "n": "transactions"
    })
    return df[[c for c in ["open","high","low","close","volume","vwap","transactions"] if c in df.columns]]


async def download_aggregates(
    tickers: List[str], start: datetime, end: datetime, out_root: Path, api_key: str, concurrency: int
):
    ensure_dir(out_root)
    days = daterange_days(start, end)
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for ticker in tickers:
            pbar = tqdm(total=len(days), desc=f"agg {ticker}", unit="day")
            for i in range(0, len(days), concurrency):
                chunk = days[i:i+concurrency]
                tasks = [fetch_day_aggregates(session, api_key, ticker, d) for d in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for d, res in zip(chunk, results):
                    pbar.update(1)
                    year = d.year; month = f"{d.month:02d}"; dd = f"{d.day:02d}"
                    out_dir = out_root / f"symbol={ticker}" / f"year={year}" / f"month={month}" / f"day={dd}"
                    ensure_dir(out_dir)
                    out_path = out_dir / "data.parquet"
                    if isinstance(res, Exception) or res is None:
                        # Skip failed days; optional: write a small marker
                        continue
                    df: pd.DataFrame = res
                    if not df.empty:
                        try:
                            df.to_parquet(out_path, engine="pyarrow")
                        except Exception:
                            # Fallback to csv
                            df.to_csv(out_dir / "data.csv")
            pbar.close()


async def download_reference(
    tickers: List[str], out_root: Path, api_key: str, concurrency: int
):
    ensure_dir(out_root)
    ep = "/v3/reference/tickers"
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pbar = tqdm(total=len(tickers), desc="reference", unit="tk")
        for i in range(0, len(tickers), concurrency):
            chunk = tickers[i:i+concurrency]
            tasks = [fetch_json(session, ep, {"apiKey": api_key, "ticker": tk}) for tk in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for tk, js in zip(chunk, results):
                pbar.update(1)
                out_dir = out_root / f"symbol={tk}"
                ensure_dir(out_dir)
                try:
                    if isinstance(js, Exception):
                        continue
                    pd.json_normalize(js).to_parquet(out_dir / "reference.parquet")
                except Exception:
                    Path(out_dir / "reference.json").write_text(str(js))
        pbar.close()


async def download_fundamentals(
    tickers: List[str], out_root: Path, api_key: str, concurrency: int
):
    ensure_dir(out_root)
    # vX Fundamentals; endpoint may vary but commonly available under /vX/reference/financials
    ep = "/vX/reference/financials"
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pbar = tqdm(total=len(tickers), desc="fundamentals", unit="tk")
        for i in range(0, len(tickers), concurrency):
            chunk = tickers[i:i+concurrency]
            tasks = [fetch_json(session, ep, {"apiKey": api_key, "ticker": tk, "limit": 5000}) for tk in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for tk, js in zip(chunk, results):
                pbar.update(1)
                out_dir = out_root / f"symbol={tk}"
                ensure_dir(out_dir)
                try:
                    if isinstance(js, Exception):
                        continue
                    # Attempt to normalize
                    data = js.get("results") if isinstance(js, dict) else None
                    if data:
                        pd.json_normalize(data).to_parquet(out_dir / "fundamentals.parquet")
                    else:
                        Path(out_dir / "fundamentals.json").write_text(str(js))
                except Exception:
                    Path(out_dir / "fundamentals.json").write_text(str(js))
        pbar.close()


async def download_corporate_actions(
    tickers: List[str], out_root: Path, api_key: str, concurrency: int
):
    ensure_dir(out_root)
    eps = {
        "dividends": "/v3/reference/dividends",
        "splits": "/v3/reference/splits",
        # other corporate actions endpoints can be added here
    }
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for kind, ep in eps.items():
            pbar = tqdm(total=len(tickers), desc=f"corp_{kind}", unit="tk")
            for i in range(0, len(tickers), concurrency):
                chunk = tickers[i:i+concurrency]
                tasks = [fetch_json(session, ep, {"apiKey": api_key, "ticker": tk, "limit": 5000}) for tk in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for tk, js in zip(chunk, results):
                    pbar.update(1)
                    out_dir = out_root / f"symbol={tk}"
                    ensure_dir(out_dir)
                    try:
                        data = js.get("results") if isinstance(js, dict) else None
                        if data:
                            pd.json_normalize(data).to_parquet(out_dir / f"{kind}.parquet")
                        else:
                            Path(out_dir / f"{kind}.json").write_text(str(js))
                    except Exception:
                        Path(out_dir / f"{kind}.json").write_text(str(js))
            pbar.close()


async def download_snapshots(
    tickers: List[str], out_root: Path, api_key: str, concurrency: int
):
    ensure_dir(out_root)
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pbar = tqdm(total=len(tickers), desc="snapshot", unit="tk")
        for i in range(0, len(tickers), concurrency):
            chunk = tickers[i:i+concurrency]
            eps = [f"/v2/snapshot/locale/us/markets/stocks/tickers/{tk}" for tk in chunk]
            tasks = [fetch_json(session, ep, {"apiKey": api_key}) for ep in eps]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for tk, js in zip(chunk, results):
                pbar.update(1)
                out_dir = out_root / f"symbol={tk}"
                ensure_dir(out_dir)
                try:
                    pd.json_normalize(js).to_parquet(out_dir / "snapshot.parquet")
                except Exception:
                    Path(out_dir / "snapshot.json").write_text(str(js))
        pbar.close()


def clamp_end_for_delay(end: datetime) -> datetime:
    now_cap = ny_tz_now_minus(15)  # 15-minute delayed policy
    return min(end, now_cap)


async def main_async(args):
    api_key = args.api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Polygon API key required. Set POLYGON_API_KEY or use --api-key")

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = pd.Timestamp(args.start_date).tz_localize("America/New_York").to_pydatetime()
    end = pd.Timestamp(args.end_date).tz_localize("America/New_York").to_pydatetime()
    end = clamp_end_for_delay(end)

    root = Path(args.data_dir or "data") / "polygon_plus" / "us_stocks"

    if "aggregates" in args.types:
        await download_aggregates(tickers, start, end, root / "aggregates", api_key, args.concurrency)
    if "reference" in args.types:
        await download_reference(tickers, root / "reference", api_key, args.concurrency)
    if "fundamentals" in args.types:
        await download_fundamentals(tickers, root / "fundamentals", api_key, args.concurrency)
    if "corp_actions" in args.types:
        await download_corporate_actions(tickers, root / "corporate_actions", api_key, args.concurrency)
    if "snapshot" in args.types:
        await download_snapshots(tickers, root / "snapshots", api_key, args.concurrency)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel Polygon US Stocks downloader (upgraded tier)")
    p.add_argument("--tickers", required=True, help="Comma-separated tickers (e.g., 'BBVA,AAPL,MSFT')")
    p.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--types", nargs="+", default=["aggregates"],
                   choices=["aggregates","reference","fundamentals","corp_actions","snapshot"],
                   help="Data types to download")
    p.add_argument("--concurrency", type=int, default=20, help="Concurrent requests")
    p.add_argument("--api-key", type=str, default=None, help="API key (or set POLYGON_API_KEY)")
    p.add_argument("--data-dir", type=str, default=None, help="Root data directory (default: ./data)")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

