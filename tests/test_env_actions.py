import numpy as np
import pandas as pd

from src.sim.env_intraday_rl import IntradayRLEnv, EnvConfig


def _make_ohlcv(start="2024-05-01 09:30", days=3, tz="America/New_York") -> pd.DataFrame:
    # Build a few RTH days of 1-min bars with simple drifting prices
    minutes_per_day = 390
    total = minutes_per_day * days
    idx = pd.date_range(start=pd.Timestamp(start, tz=tz), periods=total, freq="1min")
    # Trim to RTH window per day (09:30–16:00), inclusive of open, exclusive of close
    idx = idx[idx.indexer_between_time("09:30", "16:00", include_end=False)]
    # Synthetic price path with small noise
    base = 100.0
    steps = np.cumsum(np.random.RandomState(0).normal(0, 0.02, size=len(idx)))
    close = base + steps
    open_ = close + np.random.RandomState(1).normal(0, 0.01, size=len(idx))
    high = np.maximum(open_, close) + 0.02
    low = np.minimum(open_, close) - 0.02
    vol = np.full(len(idx), 1000.0)
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "vwap": close,
    }, index=idx)
    df.index.name = "timestamp"
    return df


def _make_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    # Minimal numeric feature set aligned to OHLCV index
    return pd.DataFrame({
        "feat_close": np.linspace(0.0, 1.0, len(idx)),
        "feat_trend": np.sin(np.linspace(0, 10, len(idx))),
    }, index=idx)


def test_action_space_and_defaults():
    ohlcv = _make_ohlcv(days=1)
    feats = _make_features(ohlcv.index)
    env = IntradayRLEnv(ohlcv=ohlcv, features=feats, env_config=EnvConfig())
    # Action space
    assert hasattr(env, 'action_space') and getattr(env.action_space, 'n', None) == 3, "Action space must be Discrete(3)"
    # Defaults
    assert bool(getattr(env.env_config, 'allow_shorts', False)) is True, "allow_shorts must default to True"
    assert bool(getattr(env.env_config, 'allow_long', False)) is True, "allow_long must default to True"


def test_long_short_reachable_in_seeded_episode():
    # With a sufficiently long episode, both sides should be reachable when shorts are allowed.
    ohlcv = _make_ohlcv(days=3)
    feats = _make_features(ohlcv.index)
    cfg = EnvConfig(max_steps=1000)  # allow long episode
    env = IntradayRLEnv(ohlcv=ohlcv, features=feats, env_config=cfg)
    obs, info = env.reset(seed=123)
    long_seen = False
    short_seen = False
    steps = 0
    # Cycle actions: 2→long, 1→flat, 0→short
    while steps < 1200 and not (long_seen and short_seen):
        a = [2, 1, 0][steps % 3]
        obs, reward, done, truncated, info = env.step(a)
        # Inspect current position sign to detect opens
        if env.pos > 0:
            long_seen = True
        elif env.pos < 0:
            short_seen = True
        steps += 1
        if done:
            # Restart next day if episode truncated/terminated early
            obs, info = env.reset()
    assert long_seen, "Long side appears clamped or unreachable under default settings"
    assert short_seen, "Short side appears clamped or unreachable under default settings"

