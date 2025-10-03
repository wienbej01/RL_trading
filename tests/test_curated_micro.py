from src.features.packs import select_features_strict, MICROSTRUCTURE_OHLCV


def test_curated_union_micro_ohlcv():
    avail = ['f1','f2','f3'] + MICROSTRUCTURE_OHLCV
    curated = ['f1','f2']
    out = select_features_strict(avail, 'curated', curated_list=curated)
    # Curated preserved in order
    assert out[:2] == curated
    # All OHLCV proxies present in the resolved list
    for m in MICROSTRUCTURE_OHLCV:
        assert m in out
