from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from .micro_ohlcv import MICROSTRUCTURE_OHLCV
except Exception:
    # Fallback: local definition if import path differs
    MICROSTRUCTURE_OHLCV = [
        "ofi_proxy",
        "bar_imbalance",
        "signed_vol_delta",
        "spread_bps_hl",
        "quote_intensity_proxy",
        "queue_imbalance_proxy",
    ]


@dataclass
class CuratedResolution:
    final_features: List[str]
    includes_micro: bool
    micro_missing: List[str]


def resolve_curated(curated_topN: List[str], *, has_l1: bool = False) -> CuratedResolution:
    # Union curated with OHLCV proxies (curated order first)
    final = list(dict.fromkeys(list(curated_topN) + list(MICROSTRUCTURE_OHLCV)))
    present = set(final)
    micro_missing = [f for f in MICROSTRUCTURE_OHLCV if f not in present]
    includes_micro = (len(micro_missing) < len(MICROSTRUCTURE_OHLCV))
    return CuratedResolution(final_features=final, includes_micro=includes_micro, micro_missing=micro_missing)

