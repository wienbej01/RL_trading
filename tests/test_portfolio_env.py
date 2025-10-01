import numpy as np
import pandas as pd

from src.sim.portfolio_env import PortfolioRLEnv, PortfolioEnvConfig


def _make_minute_index(n=5):
    # Create 1-minute spaced index within a single day in market tz
    start = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    return pd.date_range(start=start, periods=n, freq="1min")


def _make_ohlcv(idx: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    # Close increases by 0.1 each bar, high/low create ~1.0 ATR band
    close = base + 0.1 * np.arange(len(idx), dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = close.copy()
    vol = np.full(len(idx), 1000.0, dtype=float)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=idx)


def _make_features(idx: pd.DatetimeIndex, cols=("f1", "f2")) -> pd.DataFrame:
    data = {c: np.linspace(0.0, 1.0, len(idx), dtype=float) for c in cols}
    return pd.DataFrame(data, index=idx)


def test_observation_shape_and_action_mapping():
    idx = _make_minute_index(6)
    ohlcv_map = {
        "SPY": _make_ohlcv(idx, base=100.0),
        "QQQ": _make_ohlcv(idx, base=350.0),
    }
    features_map = {
        "SPY": _make_features(idx, cols=("f1", "f2")),
        "QQQ": _make_features(idx, cols=("g1",)),
    }

    cfg = PortfolioEnvConfig(
        cash=100_000.0,
        reward_scaling=1.0,
        units_per_ticker=100,
        risk_budget_per_ticker=1_000.0,
        max_gross_exposure=10.0,  # large cap to avoid scaling in this test
        turnover_penalty=0.0,
        exposure_penalty=0.0,
        position_holding_penalty=0.0,
        fixed_tickers=["SPY", "QQQ"],
        enforce_intraday=True,
        min_hold_minutes=1,
        max_hold_minutes=240,
    )

    env = PortfolioRLEnv(ohlcv_map=ohlcv_map, features_map=features_map, env_cfg=cfg)

    obs, info = env.reset()
    # feature_dim = 2 (SPY) + 1 (QQQ) + N positions (2)
    assert obs.shape[0] == 2 + 1 + 2

    # Step with actions [2, 1] => [long, flat] because 0→-1, 1→0, 2→1
    obs, reward, done, truncated, info = env.step(np.array([2, 1]))
    assert np.all(env.pos == np.array([1, 0]))

    # ATR ~ 1.0 => units ≈ floor(1000 / (1 * 1)) = 1000 but capped to 100
    assert int(env.units[0]) == 100
    assert int(env.units[1]) == 0


def test_eod_flatten_behavior():
    # Three bars: after second step, we approach last bar where EOD flatten applies
    idx = _make_minute_index(3)
    ohlcv_map = {"SPY": _make_ohlcv(idx, base=100.0)}
    features_map = {"SPY": _make_features(idx, cols=("f1",))}
    cfg = PortfolioEnvConfig(
        units_per_ticker=50,
        risk_budget_per_ticker=500.0,
        fixed_tickers=["SPY"],
        enforce_intraday=True,
        min_hold_minutes=1,
        max_hold_minutes=240,
        max_gross_exposure=10.0,
    )
    env = PortfolioRLEnv(ohlcv_map=ohlcv_map, features_map=features_map, env_cfg=cfg)
    env.reset()

    # Enter long before the last bar
    env.step(np.array([2]))  # long
    assert int(env.pos[0]) == 1
    # Next step is last bar; EOD flatten forces desired[:] = 0
    env.step(np.array([2]))  # desire long, but EOD flatten should override
    assert int(env.pos[0]) == 0

