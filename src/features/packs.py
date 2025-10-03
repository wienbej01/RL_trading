from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Optional
import hashlib
import logging

import numpy as np


# Named feature packs. These are intent-based and resolved against actual
# available feature columns at selection time (prefix/suffix matches).
PACKS = {
    "price_vol": [
        "sma_", "ema_", "atr", "rsi_", "macd", "macd_signal", "macd_histogram", "macd_line",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "returns", "log_returns",
        "sma_slope", "vol_of_vol",
    ],
    "microstructure": [
        "spread", "spread_pct", "microprice", "queue_imbalance", "order_flow_imbalance",
        "bar_imbalance", "effort_result", "price_impact",
    ],
    "context": [
        "vwap", "twap", "dist_", "rvol", "delta_vol", "minute", "day_of_week",
        "session_", "regime_", "vix", "vix_z",
    ],
    "crosssec": [
        "momentum_rank", "rvol_rank", "liquidity_rank", "value_area_", "ticker_onehot_", "id_",
    ],
}

# Minimum microstructure features we strive to always include when available
MICROSTRUCTURE_MIN = [
    "order_flow_imbalance",
    "queue_imbalance",
    "microprice",
    "spread",
    "spread_pct",
    "price_impact",
    "bar_imbalance",
    "effort_result",
]

# OHLCV-only microstructure proxies to include by default in curated packs
MICROSTRUCTURE_OHLCV = [
    "ofi_proxy",
    "bar_imbalance",
    "signed_vol_delta",
    "spread_bps_hl",
    "quote_intensity_proxy",
    "queue_imbalance_proxy",
]

def curated_cache_token(resolved: Sequence[str]) -> str:
    """Stable cache token for a resolved curated list.

    Returns a short SHA1 prefix of the feature list joined by '\n'.
    Can be incorporated into feature cache keys to invalidate when the
    curated resolution changes.
    """
    try:
        import hashlib
        payload = "\n".join([str(x) for x in resolved])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return str(len(resolved))

logger = logging.getLogger(__name__)
LAST_CURATED_CACHE_TOKEN: Optional[str] = None
LAST_CURATED_RESOLVED: Optional[List[str]] = None

# --- Curated resolution (new helpers) ---
try:
    from .resolver import resolve_curated  # type: ignore
except Exception:
    resolve_curated = None  # type: ignore
try:
    from .micro_ohlcv import MICROSTRUCTURE_OHLCV as _MICRO_OHLCV  # type: ignore
except Exception:
    _MICRO_OHLCV = MICROSTRUCTURE_OHLCV  # fallback to local constant if available

def _load_curated_list(screen_dir: Path) -> List[str]:
    p = screen_dir / "curated_topN.txt"
    if not p.exists():
        return []
    lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    global LAST_CURATED_CACHE_TOKEN
    try:
        LAST_CURATED_CACHE_TOKEN = hashlib.md5(("|".join(lines)).encode()).hexdigest()
    except Exception:
        LAST_CURATED_CACHE_TOKEN = curated_cache_token(lines)
    return lines

def resolve_curated_pack(screen_dir: Path, has_l1: bool, logger: logging.Logger, window_k: int, ticker: str) -> List[str]:
    curated_topN = _load_curated_list(screen_dir)
    final: List[str]
    includes_micro: bool
    missing: List[str]
    if callable(resolve_curated):  # type: ignore
        rr = resolve_curated(curated_topN, has_l1=has_l1)  # type: ignore
        final = list(rr.final_features)
        includes_micro = bool(rr.includes_micro)
        missing = list(rr.micro_missing)
    else:
        # Fallback: union curated with OHLCV proxies
        final = list(dict.fromkeys(list(curated_topN) + list(_MICRO_OHLCV)))
        present = set(final)
        missing = [f for f in _MICRO_OHLCV if f not in present]
        includes_micro = (len(missing) < len(_MICRO_OHLCV))
    try:
        logger.info(
            f"[features] window={window_k:02d} ticker={ticker} curated count={len(final)} "
            f"includes_micro={includes_micro} missing={missing} first10={final[:10]}"
        )
    except Exception:
        pass
    # Expose resolved list globally for pipeline
    try:
        global LAST_CURATED_RESOLVED
        LAST_CURATED_RESOLVED = list(final)
    except Exception:
        pass
    return final


def _match(available: Sequence[str], patterns: Iterable[str]) -> List[str]:
    sel: List[str] = []
    for p in patterns:
        for c in available:
            if c == p or c.startswith(p) or c.endswith(p):
                sel.append(c)
    # unique order-preserving
    seen = set()
    out = []
    for c in sel:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_features_for_pack(available: Sequence[str], selection: Sequence[str], curated: Sequence[str] | None = None) -> List[str]:
    """Resolve a final feature list from packs and optional curated list.

    - `available`: actual feature columns present
    - `selection`: pack names (keys in PACKS) or raw feature names
    - `curated`: optional extra list to union (e.g., from screening)
    """
    chosen: List[str] = []
    for key in selection:
        if key in PACKS:
            chosen.extend(_match(available, PACKS[key]))
        else:
            chosen.extend(_match(available, [key]))
    if curated:
        chosen.extend(_match(available, curated))
        # Fallback: ensure basic microstructure signals are present if available
        if not any((m in chosen) for m in MICROSTRUCTURE_MIN):
            for m in MICROSTRUCTURE_MIN:
                if any((c == m or c.startswith(m) or c.endswith(m)) for c in available):
                    chosen.extend(_match(available, [m]))
    # unique
    seen = set()
    final: List[str] = []
    for c in chosen:
        if c not in seen:
            seen.add(c)
            final.append(c)
    return final


def resolve_feature_pack(available: Sequence[str], selection: Sequence[str], curated: Sequence[str] | None = None) -> List[str]:
    """Compatibility alias that delegates to get_features_for_pack."""
    return get_features_for_pack(available, selection, curated)


def propose_curated_from_consensus(consensus_path: Path, top_n: int = 40) -> List[str]:
    """Read consensus importance parquet and return top N feature names.
    Returns empty list if file is missing or unreadable.
    """
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(consensus_path)
        col = "consensus" if "consensus" in df.columns else (df.columns[-1] if df.columns.size > 1 else None)
        if col is None:
            return []
        df = df.sort_values(col, ascending=False)
        feats = df["feature"].astype(str).tolist()[: int(top_n)]
        # Update cache token from proposed curated list
        try:
            global LAST_CURATED_CACHE_TOKEN
            LAST_CURATED_CACHE_TOKEN = curated_cache_token(feats)
        except Exception:
            pass
        return feats
    except Exception:
        return []


def load_curated_topN(path: Path, top_n: int | None = None) -> List[str]:
    """Load curated_topN.txt and update LAST_CURATED_CACHE_TOKEN.

    - Reads one feature name per line, ignoring blanks and comments (#).
    - If file missing or unreadable, returns [] and sets token to '0'.
    - If top_n provided, truncates to that length.
    """
    global LAST_CURATED_CACHE_TOKEN
    curated: List[str] = []
    try:
        if path.exists():
            text = path.read_text().splitlines()
            for ln in text:
                s = str(ln).strip()
                if not s or s.startswith('#'):
                    continue
                curated.append(s)
            if isinstance(top_n, int) and top_n > 0:
                curated = curated[: int(top_n)]
            # Update token based on just the curated list
            LAST_CURATED_CACHE_TOKEN = curated_cache_token(curated)
        else:
            # Missing file: deterministic token
            LAST_CURATED_CACHE_TOKEN = "0"
    except Exception:
        try:
            LAST_CURATED_CACHE_TOKEN = "0"
        except Exception:
            pass
        curated = []
    return curated


# --- append near existing helpers ---
def select_features_strict(
    avail_cols: Sequence[str],
    pack_name: str,
    curated_list: Sequence[str] | None = None,
    packs: dict | None = None,
    *,
    win: Optional[str] = None,
    tic: Optional[str] = None,
) -> List[str]:
    """
    Enforce exact semantics:
      - pack_name == 'curated'                 -> curated_list (required)
      - pack_name startswith 'curated_minus:'  -> curated_list minus specified pack
      - pack_name in packs (e.g., 'microstructure') -> exactly that pack
    Always intersect with avail_cols and preserve curated_list order.
    """
    packs = packs or PACKS
    A = [c for c in avail_cols if c != "ticker"]
    aset = set(A)

    if pack_name == "curated":
        assert curated_list, "curated_list required for 'curated'"
        # Do not intersect with available; this is the requested final list to compute
        base = list(curated_list)
        final = list(dict.fromkeys(base + MICROSTRUCTURE_OHLCV))
        # Add MICROSTRUCTURE_MIN features that are in avail_cols
        for m in MICROSTRUCTURE_MIN:
            if m in aset and m not in final:
                final.append(m)
        # Publish resolved list and token for cache keys
        try:
            global LAST_CURATED_RESOLVED
            LAST_CURATED_RESOLVED = list(final)
        except Exception:
            pass
        try:
            global LAST_CURATED_CACHE_TOKEN
            LAST_CURATED_CACHE_TOKEN = curated_cache_token(final)
        except Exception:
            pass
        # Log with required format
        try:
            w = str(win) if win is not None else "NA"
            t = str(tic) if tic is not None else "NA"
            present = set(final)
            micro_missing = [f for f in MICROSTRUCTURE_OHLCV if f not in present]
            includes_micro = len(micro_missing) < len(MICROSTRUCTURE_OHLCV)
            logger.info(
                f"[features] window={w} ticker={t} curated count={len(final)} "
                f"includes_micro={includes_micro} missing={micro_missing} first10={final[:10]}"
            )
        except Exception:
            pass
        return final

    if pack_name.startswith("curated_minus:"):
        p = pack_name.split(":", 1)[1]
        assert curated_list, "curated_list required for 'curated_minus:*'"
        drop = set(packs.get(p, []))
        return [f for f in curated_list if (f in aset and f not in drop)]

    # exact pack
    if pack_name in packs:
        return [f for f in packs[pack_name] if f in aset]

    raise ValueError(f"Unknown pack_name: {pack_name}")
