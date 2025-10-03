import numpy as np

from src.features.packs import select_features_strict, MICROSTRUCTURE_MIN


def test_curated_pack_includes_microstructure_min():
    # Mock available columns, including some microstructure keys
    avail = [
        'feat_a', 'feat_b', 'feat_c',
        'order_flow_imbalance', 'queue_imbalance', 'microprice',
    ]
    curated = ['feat_a', 'feat_c']  # pretend curated_topN contains only these
    out = select_features_strict(avail, 'curated', curated_list=curated)
    # curated should be preserved
    assert 'feat_a' in out and 'feat_c' in out
    # ensure microstructure minimum keys present when available in avail
    ms_present = [m for m in MICROSTRUCTURE_MIN if m in avail]
    for m in ms_present:
        assert m in out, f"expected microstructure feature '{m}' to be included"

