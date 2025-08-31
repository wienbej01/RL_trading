# rl-intraday/src/data/data_loader.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from ..utils.config_loader import Settings

logger = logging.getLogger(__name__)


def _to_datetime(x) -> datetime:
    if isinstance(x, datetime):
        return x
    return pd.to_datetime(x).to_pydatetime()


def _normalize_range(start, end) -> Tuple[datetime, datetime]:
    """Make end exclusive; if end has no time (00:00), bump to next day."""
    s = _to_datetime(start)
    e = _to_datetime(end)
    if e.time() == datetime.min.time():
        e = e + timedelta(days=1)
    return s, e

def _detect_ts_col(df: pd.DataFrame) -> str | None:
    for c in ("timestamp", "datetime", "time", "dt"):
        if c in df.columns:
            return c
    return None

class UnifiedDataLoader:
    """
    Unified loader for different data sources with simple caching.

    Directory expectations for Polygon RAW (already in your repo):
      <polygon_raw_dir>/symbol=SPY/year=2024/month=01/day=09/data.parquet

    Cache file naming:
      <cache_dir>/SPY_20240101_20240201_ohlcv_1min.parquet
    """

    def __init__(
        self,
        data_source: str = "polygon",
        config_path: str | None = None,
        cache_enabled: bool | None = None,
        default_timeframe: str = "1min",
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings(config_path=config_path)
        self.paths = self.settings.paths

        self.data_source = (data_source or "polygon").lower()
        self.data_root = Path(self.paths.get("data_root"))
        self.cache_dir = Path(self.paths.get("cache_dir"))
        if self.data_source == "polygon":
            self.raw_dir = Path(self.paths.get("polygon_raw_dir"))
        else:
            self.raw_dir = (self.data_root / self.data_source / "historical").resolve()

        # >>> IMPORTANT: this was missing and caused your crash
        self.cache_enabled = True if cache_enabled is None else bool(cache_enabled)
        self.default_timeframe = default_timeframe

        # Ensure cache folder exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Data root: %s | Cache: %s | Raw (%s): %s",
            self.data_root, self.cache_dir, self.data_source, self.raw_dir
        )
        self.data_directories = [self.raw_dir]  # for legacy logs
        self.logger.info("Data directories: %s", self.data_directories)

    # -------------------- Public API --------------------

    def load(
        self,
        symbol: str,
        start,
        end,
        timeframe: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load OHLCV for the given range, using cache if available."""
        tf = timeframe or self.default_timeframe
        s, e = _normalize_range(start, end)

        cache_path = self._cache_path(symbol, s, e, tf)
        if self.cache_enabled and cache_path.exists():
            self.logger.info("Loading cached data for %s: %s", symbol, cache_path)
            return self._read_parquet_safe(cache_path)

        # Build from RAW
        if self.data_source == "polygon":
            df = self._load_polygon_ohlcv(symbol, s, e)
        else:
            raise NotImplementedError(f"Data source not supported: {self.data_source}")

        if df.empty:
            self.logger.warning("No data found for %s between %s and %s", symbol, s, e)
            return df

        # (Optional) normalize/clean minimal columns here if needed

        # Save cache
        if self.cache_enabled:
            try:
                df.to_parquet(cache_path, index=False)
                self.logger.info("Wrote cache: %s", cache_path)
            except Exception as ex:
                self.logger.warning("Could not write cache %s: %s", cache_path, ex)

        return df

    # -------------------- Polygon helpers --------------------

    def _iter_polygon_files(self, symbol: str, start: datetime, end: datetime) -> Iterable[Path]:
        """
        Yield daily parquet paths inside RAW dir constrained by date range.
        Expected partitioning: symbol=SPY/year=YYYY/month=MM/day=DD/data.parquet
        """
        sym_dir = self.raw_dir / f"symbol={symbol.upper()}"
        if not sym_dir.exists():
            self.logger.debug("Symbol dir does not exist: %s", sym_dir)
            return []

        # We parse the year/month/day from the partition names
        candidates: List[Path] = []
        for p in sym_dir.rglob("data.parquet"):
            parts = {seg.split("=", 1)[0]: seg.split("=", 1)[1]
                     for seg in p.parts if "=" in seg}
            try:
                y = int(parts["year"])
                m = int(parts["month"])
                d = int(parts["day"])
                dt = datetime(y, m, d)
            except Exception:
                continue
            if start.date() <= dt.date() < end.date():
                candidates.append(p)

        candidates.sort()
        return candidates

    def _load_polygon_ohlcv(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        files = list(self._iter_polygon_files(symbol, start, end))
        if not files:
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as ex:
                self.logger.warning("Failed to read %s: %s", f, ex)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Normalize common columns if present
        ts_col = None
        for c in ("timestamp", "datetime", "time", "dt"):
            if c in df.columns:
                ts_col = c
                break
        if ts_col:
            # keep UTC tz-naive or convert; the sanitizer later can localize to US/Eastern
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        # Coerce basic numerics if present
        for c in ("open", "high", "low", "close", "volume", "vw", "n", "trades", "ticks"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # -------------------- Cache helpers --------------------

    def _cache_path(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Path:
        fname = f"{symbol.upper()}_{start:%Y%m%d}_{end:%Y%m%d}_ohlcv_{timeframe}.parquet"
        return (self.cache_dir / fname).resolve()

    # -------------------- IO --------------------

    def _read_parquet_safe(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception as ex:
            self.logger.warning("Failed to read cached file %s: %s", path, ex)
            return pd.DataFrame()
