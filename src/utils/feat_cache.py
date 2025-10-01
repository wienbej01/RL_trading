from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


def feature_cache_path(
    run_name: str,
    ticker: str,
    split: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    cfg_hash: str,
) -> Path:
    """Return the canonical feature cache path for a run/ticker/split window.

    Layout: results/cache/features/<run-name>/<ticker>/<split>_<start>_<end>_<cfg_hash>.parquet
    Dates are formatted as YYYYMMDD (local to their tz); end is inclusive key.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    start_key = s.strftime("%Y%m%d")
    end_key = e.strftime("%Y%m%d")
    rel = Path("results/cache/features") / run_name / ticker
    rel.mkdir(parents=True, exist_ok=True)
    fname = f"{split}_{start_key}_{end_key}_{cfg_hash}.parquet"
    return rel / fname


def augment_cfg_hash(
    base_hash: str,
    *,
    run_name: str = "",
    features: list[str] | None = None,
    micro_module_path: Path | None = None,
    curated_token: str | None = None,
) -> str:
    """Augment a base cfg_hash with run name, resolved feature list, and micro module version.

    This helps invalidate caches when curated resolvers or proxy implementations change.
    """
    try:
        import hashlib
        h = hashlib.sha1()
        h.update(str(base_hash).encode("utf-8"))
        if run_name:
            h.update(str(run_name).encode("utf-8"))
        if features:
            # Stable order
            for f in features:
                h.update(str(f).encode("utf-8"))
        if curated_token:
            h.update(str(curated_token).encode("utf-8"))
        if micro_module_path and micro_module_path.exists():
            try:
                src = micro_module_path.read_bytes()
                h.update(src)
            except Exception:
                pass
        return h.hexdigest()[:16]
    except Exception:
        return str(base_hash)


def save_features(df: pd.DataFrame, path: Path) -> None:
    """Save features DataFrame to parquet with pyarrow, preserving index and tz.

    - Writes with index to round-trip DatetimeIndex (tz-aware) precisely.
    - Logs INFO on save.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=True)
        logger.info("Saved feature cache to %s (rows=%d, cols=%d)", path, len(df), df.shape[1])
    except Exception as e:
        logger.warning("Failed to write feature cache %s: %s", path, e)


def load_features(path: Path) -> pd.DataFrame:
    """Load features DataFrame from parquet, ensuring index dtype/tz preserved.

    Returns an empty DataFrame if file is missing.
    """
    if not path.exists():
        logger.info("Feature cache miss (not found): %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    # Ensure DatetimeIndex with tz awareness remains intact
    if not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to promote a 'timestamp' column if present
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.loc[ts.notna()].set_index(ts[ts.notna()])
            df.index.name = "timestamp"
    logger.info("Loaded feature cache from %s (rows=%d, cols=%d)", path, len(df), df.shape[1])
    return df
