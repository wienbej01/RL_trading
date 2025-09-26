from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

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
    # unique
    seen = set()
    final: List[str] = []
    for c in chosen:
        if c not in seen:
            seen.add(c)
            final.append(c)
    return final


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
        return feats
    except Exception:
        return []

