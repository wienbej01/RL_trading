# src/data/data_loader.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    # Project-local Settings loader
    from src.utils.config_loader import Settings
except Exception:  # pragma: no cover
    # Allow this module to be imported outside the repo for tooling/fixes
    Settings = None  # type: ignore


logger = logging.getLogger(__name__)


# ----------------------------
# Robust timestamp handling
# ----------------------------

_TS_CANDIDATES: Iterable[str] = (
    "timestamp",
    "datetime",
    "time",
    "dt",
    "ts",
    "t",
    "sip_timestamp",
    "participant_timestamp",
    "unix",
)


def _detect_ts_col(df: pd.DataFrame) -> Optional[str]:
    for c in _TS_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _canonicalize_timestamp(
    df: pd.DataFrame,
    *,
    market_tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Ensure a canonical time index:

    - Find/derive a 'timestamp' column (or index) and parse it.
    - Make UTC tz-aware, then convert to market_tz (default: America/New_York).
    - Sort ascending, drop duplicates, set index name to 'timestamp'.
    - Return an empty, typed frame if the input is empty.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    ts_col = _detect_ts_col(df)
    if ts_col is None:
        # Try datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            tmp = df.reset_index().rename(columns={"index": "timestamp"})
            ts_col = "timestamp"
        else:
            # Try index as epoch ms → ns
            idx = pd.to_datetime(df.index, utc=True, errors="coerce", unit="ms")
            if idx.isna().all():
                idx = pd.to_datetime(df.index, utc=True, errors="coerce", unit="ns")
            if idx.notna().any():
                tmp = df.loc[idx.notna()].copy().reset_index(drop=True)
                tmp["timestamp"] = idx[idx.notna()].to_series(index=tmp.index)
                df = tmp
                ts_col = "timestamp"
            else:
                raise ValueError("No recognizable timestamp column or index in RAW parquet")
    else:
        df = df.rename(columns={ts_col: "timestamp"})
        ts_col = "timestamp"

    # First try general parser
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    # If entirely NaT, try epoch ms then ns
    if ts.isna().all():
        try:
            ts = pd.to_datetime(df[ts_col].astype("int64"), utc=True, errors="coerce", unit="ms")
        except Exception:
            pass
        if ts.isna().all():
            try:
                ts = pd.to_datetime(df[ts_col].astype("int64"), utc=True, errors="coerce", unit="ns")
            except Exception:
                pass

    out = df.loc[ts.notna()].copy()
    if out.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    out["timestamp"] = ts[ts.notna()]
    out = out.sort_values("timestamp").set_index("timestamp")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    # Convert to market tz (NY) for downstream consistency
    out.index = out.index.tz_convert(market_tz)

    # Drop duplicates just in case
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "timestamp"
    return out


# ----------------------------
# Optional RTH + resample + day-prune
# ----------------------------

def _enforce_rth_resample(
    df: pd.DataFrame,
    *,
    market_tz: str = "America/New_York",
    rth_start: str = "09:30",
    rth_end: str = "16:00",
    strict_resample: bool = True,
) -> pd.DataFrame:
    """
    Enforce Regular Trading Hours and exact 1-minute bars.
    - Convert index to market tz if needed.
    - Keep [09:30, 16:00) local time.
    - Resample to 1 minute with OHLCV aggregations.
    """
    if df.empty:
        return df

    local = df.copy()
    if local.index.tz is None:
        # Shouldn't happen after _canonicalize_timestamp, but keep safe
        local.index = local.index.tz_localize("UTC").tz_convert(market_tz)
    else:
        local.index = local.index.tz_convert(market_tz)

    # RTH slice
    local = local.between_time(rth_start, rth_end, inclusive="left")

    # Build aggregation by available columns
    agg = {}
    if "open" in local.columns:
        agg["open"] = "first"
    if "high" in local.columns:
        agg["high"] = "max"
    if "low" in local.columns:
        agg["low"] = "min"
    if "close" in local.columns:
        agg["close"] = "last"
    if "volume" in local.columns:
        agg["volume"] = "sum"
    if "vwap" in local.columns:
        # VWAP last is common for 1min bar
        agg["vwap"] = "last"
    if "transactions" in local.columns:
        agg["transactions"] = "sum"

    if not agg:
        # If no standard columns present, just count rows (edge case)
        local["rows"] = 1
        agg = {"rows": "sum"}

    out = local
    if strict_resample:
        out = local.resample("1min").agg(agg)

    return out


def _prune_sparse_days(
    df: pd.DataFrame,
    *,
    min_minutes_per_day: int = 370  # ~95% of 390
) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df.groupby(df.index.normalize()).size()
    bad_days = counts[counts < min_minutes_per_day].index
    if len(bad_days):
        return df[~df.index.normalize().isin(bad_days)]
    return df


# ----------------------------
# Loader
# ----------------------------

@dataclass
class LoaderPaths:
    data_root: Path
    cache_dir: Path
    polygon_raw_dir: Path


class UnifiedDataLoader:
    """
    Unified loader for historical OHLCV.

    - Reads RAW polygon partitions under: paths.polygon_raw_dir/symbol=SYMBOL/year=YYYY/month=MM/day=DD/data.parquet
    - Canonicalizes timestamps on read.
    - Optionally applies RTH + strict resample + day-prune (via config flags).
    - Writes/reads cache files to speed future loads.
    """

    def __init__(
        self,
        *,
        data_source: str = "polygon",
        config_path: Optional[str] = None,
        cache_enabled: bool = True,
        default_timeframe: str = "1min",
    ) -> None:
        self.data_source = data_source
        self.cache_enabled = cache_enabled
        self.default_timeframe = default_timeframe

        if Settings is not None and config_path:
            self.settings = Settings(config_path=config_path)
        else:
            self.settings = None  # type: ignore

        self.paths = self._resolve_paths()
        logger.info(
            "Data root: %s | Cache: %s | Raw (polygon): %s",
            self.paths.data_root,
            self.paths.cache_dir,
            self.paths.polygon_raw_dir,
        )
        logger.info("Data directories: [%s]", self.paths.polygon_raw_dir)

        # Config: RTH/resample/prune behavior (off by default to preserve behavior)
        self.session_tz = self._get_cfg("data", "session", "tz", default="America/New_York")
        self.rth_start = self._get_cfg("data", "session", "rth_start", default="09:30")
        self.rth_end = self._get_cfg("data", "session", "rth_end", default="16:00")
        self.apply_rth_resample = bool(self._get_cfg("data", "resample", "apply", default=False))
        self.strict_resample = bool(self._get_cfg("data", "resample", "strict", default=True))
        self.min_minutes_per_day = int(self._get_cfg("data", "prune", "min_minutes_per_day", default=0))

        # Ensure cache dir exists
        try:
            self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # --------- public API ---------

    def load_ohlcv(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        timeframe: str | None = None,
        use_cache: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV for [start, end], inclusive on start, exclusive on end cache-key.

        If cache exists and cache_enabled, loads cache; otherwise, builds from RAW partitions.
        """
        tf = timeframe or self.default_timeframe
        if tf != "1min":
            raise ValueError(f"Only '1min' timeframe supported by this loader (got {tf})")

        use_cache = self.cache_enabled if use_cache is None else use_cache

        cache_path = self._cache_path(symbol, start, end, tf)
        if use_cache and cache_path.exists():
            logger.info("Loading cached data for %s: %s", symbol, cache_path)
            df = pd.read_parquet(cache_path)
            # Guarantee index is DatetimeIndex (backward compat)
            if not isinstance(df.index, pd.DatetimeIndex):
                df = _canonicalize_timestamp(df, market_tz=self.session_tz)
            return df

        # Build from RAW
        file_paths = self._enumerate_raw_paths(symbol, start, end)
        df = self._load_raw_partitions(file_paths, market_tz=self.session_tz)

        if self.apply_rth_resample:
            df = _enforce_rth_resample(
                df,
                market_tz=self.session_tz,
                rth_start=self.rth_start,
                rth_end=self.rth_end,
                strict_resample=self.strict_resample,
            )
            if self.min_minutes_per_day and self.min_minutes_per_day > 0:
                df = _prune_sparse_days(df, min_minutes_per_day=self.min_minutes_per_day)

        if use_cache:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                logger.warning("Failed to write cache %s: %s", cache_path, e)

        return df

    # --------- internals ---------

    def _resolve_paths(self) -> LoaderPaths:
        """
        Resolve data_root, cache_dir, and polygon_raw_dir from YAML (with sane defaults).
        """
        if self.settings is not None:
            data_root = Path(self._get_cfg("paths", "data_root", default="data")).resolve()
            cache_dir = Path(self._get_cfg("paths", "cache_dir", default=str(data_root / "cache"))).resolve()
            # Project’s RAW layout default
            polygon_raw_dir = Path(
                self._get_cfg("paths", "polygon_raw_dir", default=str(data_root / "polygon" / "historical"))
            ).resolve()
        else:
            # Fallback defaults (repo-relative)
            cwd = Path.cwd()
            data_root = (cwd / "data").resolve()
            cache_dir = (data_root / "cache").resolve()
            polygon_raw_dir = (data_root / "polygon" / "historical").resolve()

        return LoaderPaths(data_root=data_root, cache_dir=cache_dir, polygon_raw_dir=polygon_raw_dir)

    def _get_cfg(self, *keys: str, default=None):
        """
        Convenience getter for nested keys via Settings.
        """
        if self.settings is None:
            return default
        try:
            return self.settings.get(*keys, default=default)
        except Exception:
            return default

    def _cache_path(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> Path:
        """
        Build a cache filename like: SPY_YYYYMMDD_YYYYMMDD_ohlcv_1min.parquet
        End part uses exclusive (end + 1 day) to mirror your existing convention.
        """
        start_key = pd.Timestamp(start).strftime("%Y%m%d")
        # end exclusive → add 1 day for the key
        end_key = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        name = f"{symbol}_{start_key}_{end_key}_ohlcv_{timeframe}.parquet"
        return self.paths.cache_dir / name

    def _enumerate_raw_paths(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
        """
        Enumerate daily parquet files under:
        polygon_raw_dir/symbol=SYMBOL/year=YYYY/month=MM/day=DD/data.parquet
        for each day in [start, end] (inclusive).
        """
        start_d = pd.Timestamp(start).normalize()
        end_d = pd.Timestamp(end).normalize()
        days = pd.date_range(start_d, end_d, freq="D")

        base = self.paths.polygon_raw_dir
        out: List[Path] = []
        for d in days:
            y = d.year
            m = f"{d.month:02d}"
            dd = f"{d.day:02d}"
            path = base / f"symbol={symbol}" / f"year={y}" / f"month={m}" / f"day={dd}" / "data.parquet"
            if path.exists():
                out.append(path)
        return out

    def _load_raw_partitions(
        self,
        file_paths: Sequence[Path],
        *,
        market_tz: str | None = None,
        log_prefix: str = "RAW",
    ) -> pd.DataFrame:
        """
        Load a sequence of RAW parquet partitions and return one canonical DataFrame.
        Calls _canonicalize_timestamp immediately after pd.read_parquet.
        """
        tz = market_tz or "America/New_York"

        parts: List[pd.DataFrame] = []
        total = 0
        bad = 0

        for path in file_paths:
            total += 1
            try:
                part = pd.read_parquet(path)
                part = _canonicalize_timestamp(part, market_tz=tz)  # << critical
                if not part.empty:
                    parts.append(part)
                else:
                    bad += 1
                    logger.warning("%s: empty frame after canonicalization: %s", log_prefix, path)
            except Exception as e:
                bad += 1
                logger.warning("%s: failed to load %s: %s", log_prefix, path, e)

        if not parts:
            empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
            logger.info("%s: loaded 0/%d partitions (bad=%d) -> empty frame", log_prefix, total, bad)
            return empty

        df = pd.concat(parts, axis=0, copy=False)
        # Safety: ensure sorted, deduped, proper name
        if not isinstance(df.index, pd.DatetimeIndex):
            df = _canonicalize_timestamp(df, market_tz=tz)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "timestamp"

        first, last = (df.index.min(), df.index.max()) if not df.empty else ("NA", "NA")
        logger.info(
            "%s: loaded %d/%d partitions (bad=%d); rows=%d range=[%s .. %s] tz=%s",
            log_prefix,
            len(parts),
            total,
            bad,
            len(df),
            first,
            last,
            df.index.tz,
        )

        return df
