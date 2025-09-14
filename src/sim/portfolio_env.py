"""
Multi-ticker portfolio environment for RL.

This env aggregates multiple single-ticker OHLCV/feature streams into a single
portfolio state with per-ticker positions. It is intentionally minimal to
enable end-to-end multi-ticker training/backtesting:

- Action space: MultiDiscrete([3] * N) per ticker {-1,0,1} via mapping 0→-1,1→0,2→1
- Position size: unit exposure per ticker (can be extended to sizing later)
- Reward: portfolio PnL change (sum of per-ticker bar PnL), optional scaling
- EOD: flatten positions at session end

Assumptions:
- Input `ohlcv_map` and `features_map` are dicts ticker→DataFrame, aligned on a
  common DatetimeIndex with market tz (America/New_York). If not, we compute the
  intersection of indices and align/ffill.
- Features per ticker are numeric; we concatenate features across tickers into a
  single observation vector, plus per-ticker current positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

from ..utils.config_loader import Settings
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PortfolioEnvConfig:
    cash: float = 100_000.0
    reward_scaling: float = 1.0
    units_per_ticker: int = 100  # shares/contracts per ticker when fully long/short
    atr_window: int = 14
    risk_budget_per_ticker: float = 1000.0  # $ at ATR risk per ticker
    max_gross_exposure: float = 1.0  # max sum(|units*price|)/equity
    turnover_penalty: float = 0.0  # reward penalty per unit turnover
    exposure_penalty: float = 0.0  # penalty per unit of exposure violation (in $)
    # Intraday and trade-discipline constraints
    enforce_intraday: bool = True  # flatten at end of session; block entries too close to EOD
    min_hold_minutes: int = 5      # minimum holding period in minutes
    max_hold_minutes: int = 240    # maximum holding period in minutes
    max_entries_per_day: int = 3   # limit number of entries (incl. flips) per day
    allowed_trade_tickers: Optional[List[str]] = None  # if set, only these tickers are tradable
    position_holding_penalty: float = 0.0  # penalty per open position per bar
    # Fixed ticker universe/order to enforce observation/action shape parity with training
    fixed_tickers: Optional[List[str]] = None


class PortfolioRLEnv(Env):
    """Simple portfolio environment with per-ticker unit positions."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *,
        ohlcv_map: Dict[str, pd.DataFrame],
        features_map: Dict[str, pd.DataFrame],
        config: Optional[Dict[str, Any]] = None,
        settings: Optional[Settings] = None,
        env_cfg: Optional[PortfolioEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = env_cfg or PortfolioEnvConfig()
        self.settings = settings or Settings.from_yaml('configs/settings.yaml')
        # Desired ticker universe and order
        provided_tickers = list(self.cfg.fixed_tickers) if (self.cfg and self.cfg.fixed_tickers) else sorted(list(ohlcv_map.keys()))
        assert provided_tickers, "No tickers provided to PortfolioRLEnv"

        # Identify tickers with usable OHLCV for index alignment
        idx_sources: List[pd.DatetimeIndex] = []
        data_tickers: List[str] = []
        for t in provided_tickers:
            df = ohlcv_map.get(t)
            if df is None or df.empty:
                continue
            if 'close' not in df.columns:
                continue
            idx_sources.append(df.index)
            data_tickers.append(t)
        if not idx_sources:
            raise ValueError("Empty OHLCV data for all tickers")
        # Compute intersection; if too small, fall back to the longest single index
        common_idx = idx_sources[0]
        for idx in idx_sources[1:]:
            common_idx = common_idx.intersection(idx)
        common_idx = common_idx.sort_values()
        if len(common_idx) < 2:
            # Fallback: choose the longest available index among data tickers
            lens = [(t, len(ohlcv_map[t].index)) for t in data_tickers]
            lens.sort(key=lambda x: x[1], reverse=True)
            common_idx = ohlcv_map[lens[0][0]].index

        # Reindex/align and keep needed columns (pad missing tickers with zeros)
        self.ohlcv: Dict[str, pd.DataFrame] = {}
        self.X: Dict[str, pd.DataFrame] = {}
        feat_sizes: List[int] = []
        valid_tickers: List[str] = []
        for t in provided_tickers:
            df_src = ohlcv_map.get(t)
            if df_src is None or df_src.empty or 'close' not in df_src.columns:
                # Create zero OHLCV placeholders
                zeros = pd.DataFrame(index=common_idx, data={
                    'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0, 'vwap': 0.0
                })
                self.ohlcv[t] = zeros
                # Create zero feature vector; infer columns from provided features if present
                Xsrc = features_map.get(t)
                if Xsrc is not None and not Xsrc.empty:
                    Xcols = list(Xsrc.columns)
                else:
                    # Default to a single close feature if unknown
                    Xcols = ['close']
                X = pd.DataFrame(0.0, index=common_idx, columns=Xcols)
                self.X[t] = X
                feat_sizes.append(len(Xcols))
                valid_tickers.append(t)
                continue
            # Real data path
            o = df_src.reindex(common_idx).ffill().bfill()
            cols = [c for c in ["open","high","low","close","volume","vwap"] if c in o.columns]
            if 'close' not in cols:
                cols = cols + ['close']
                o['close'] = 0.0
            o = o[cols]
            self.ohlcv[t] = o
            X = features_map.get(t)
            if X is None or X.empty:
                X = pd.DataFrame({"close": o["close"]}, index=o.index)
            else:
                X = X.reindex(common_idx).ffill().bfill()
            X = X.select_dtypes(include=[np.number]).copy()
            self.X[t] = X
            feat_sizes.append(X.shape[1])
            valid_tickers.append(t)

        # Finalize ticker list to exactly provided order
        self.tickers = provided_tickers

        self.index = common_idx
        self.n = len(self.index)
        self.N = len(self.tickers)
        self.feat_sizes = feat_sizes
        self.feature_dim = int(sum(feat_sizes))

        # Determine bar duration in minutes robustly (ignore large gaps and day changes)
        try:
            if len(self.index) >= 2:
                diffs = (self.index[1:] - self.index[:-1]).asi8 / 60_000_000_000
                diffs = np.asarray(diffs, dtype=float)
                # Keep small, positive gaps (<= 5 minutes) to estimate bar interval
                diffs = diffs[(diffs > 0) & (diffs <= 5 * 1.0)]
                if diffs.size == 0:
                    self.bar_minutes = 1.0
                else:
                    # Use a lower quantile to avoid skew by occasional 2-5 min gaps
                    self.bar_minutes = float(np.quantile(diffs, 0.5))
                # Clamp to reasonable range for intraday data
                self.bar_minutes = max(1.0, min(self.bar_minutes, 60.0))
            else:
                self.bar_minutes = 1.0
        except Exception:
            self.bar_minutes = 1.0

        # Spaces: MultiDiscrete(3 per ticker) → map to {-1,0,1}
        self.action_space = MultiDiscrete([3] * self.N)
        # Obs = concatenated features + current positions per ticker
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.feature_dim + self.N,), dtype=np.float32)

        # Exec params
        try:
            self.point_value = float(self.settings.get("execution", "point_value", default=1.0))
            self.tc_per_trade = float(self.settings.get("execution", "commission_per_contract", default=0.0))
        except Exception:
            self.point_value = 1.0
            self.tc_per_trade = 0.0

        # Precompute ATR per ticker
        self.atr_map: Dict[str, pd.Series] = {}
        w = max(2, int(self.cfg.atr_window))
        for t in self.tickers:
            o = self.ohlcv[t]
            try:
                h = o['high']
                l = o['low']
                c = o['close']
                prev_c = c.shift(1)
                tr = (h - l).abs()
                tr = pd.concat([tr, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
                atr = tr.rolling(w, min_periods=1).mean().bfill()
                self.atr_map[t] = atr
            except Exception:
                self.atr_map[t] = pd.Series(0.0, index=o.index)

        # State
        self.reset()

    def _obs(self, i: int) -> np.ndarray:
        # concatenate features in ticker order, then positions
        feats: List[np.ndarray] = []
        for t in self.tickers:
            x = self.X[t].iloc[i].to_numpy(dtype=np.float32, copy=False)
            # sanitize
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(x)
        pos = self.pos.astype(np.float32)
        out = np.concatenate([*feats, pos], axis=0).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.i = 1  # start at index 1 so prev bar exists
        self.cash = float(self.cfg.cash)
        self.pos = np.zeros((self.N,), dtype=np.int32)
        self.equity = float(self.cash)
        self.equity_curve: List[float] = [self.equity]
        # Preserve last snapshots across auto-reset; do not clear here
        if not hasattr(self, '_last_equity_curve'):
            self._last_equity_curve = None
        if not hasattr(self, '_last_history'):
            self._last_history: Optional[List[Dict[str, Any]]] = None
        self.history: List[Dict[str, Any]] = []
        # Trades tracking (per-ticker open trade, and list of closed trades)
        self._open_trades: Dict[str, Optional[Dict[str, Any]]] = {t: None for t in self.tickers}
        self.trades: List[Dict[str, Any]] = []
        # Signed units per ticker
        self.units = np.zeros((self.N,), dtype=np.int64)
        # Holding durations in bars per ticker
        self.hold_bars = np.zeros((self.N,), dtype=np.int32)
        # Per-day entry counter and current day tracker
        self.current_day = pd.Timestamp(self.index[self.i]).date() if len(self.index) else None
        self.daily_entries = 0
        # Allowed tickers mask
        allowed = set(self.cfg.allowed_trade_tickers or [])
        self.allowed_mask = np.array([(t in allowed) for t in self.tickers], dtype=bool) if allowed else None
        return self._obs(self.i), {}

    def step(self, action: np.ndarray):
        # map 0→-1, 1→0, 2→1 per ticker
        a = np.asarray(action).reshape(-1)
        desired = (a - 1).astype(np.int32)
        desired = np.clip(desired, -1, 1)

        # Time and session bookkeeping
        ts_cur = self.index[self.i]
        day_cur = ts_cur.date()
        # Day change: reset counters
        if self.current_day is None or day_cur != self.current_day:
            self.current_day = day_cur
            self.daily_entries = 0
            # Reset holding bars because we consider overnight not allowed
            self.hold_bars[:] = 0
        # End-of-day check: if next timestamp is a new day (or we're at the end), force flat
        is_last_bar = (self.i >= self.n - 1)
        if not is_last_bar:
            next_day = self.index[self.i + 1].date()
            eod_incoming = (next_day != day_cur)
        else:
            eod_incoming = True

        # Apply allowed tickers gating
        if self.allowed_mask is not None:
            desired = desired.copy()
            desired[~self.allowed_mask] = 0

        prev_pos = self.pos.copy()

        # Convert minute constraints to bars
        min_hold_bars = int(max(0, np.ceil(self.cfg.min_hold_minutes / max(1e-6, self.bar_minutes))))
        max_hold_bars = int(max(1, np.floor(self.cfg.max_hold_minutes / max(1e-6, self.bar_minutes))))
        # Remaining bars today (including current step outcome before advancing)
        if eod_incoming:
            remaining_bars_today = 1
        else:
            # Count how many consecutive indices remain with same day
            j = self.i
            remaining = 0
            while j < self.n and self.index[j].date() == day_cur:
                remaining += 1
                j += 1
            remaining_bars_today = remaining

        # Enforce intraday discipline
        if self.cfg.enforce_intraday:
            desired = desired.copy()
            # Do not allow new entries when too close to EOD to satisfy min hold
            if remaining_bars_today < max(1, min_hold_bars):
                new_entry_mask = (prev_pos == 0) & (desired != 0)
                desired[new_entry_mask] = 0
            # Force flat at end of day
            if eod_incoming:
                desired[:] = 0

        # Enforce minimum hold: keep existing position until min bars are met
        must_hold_mask = (prev_pos != 0) & (self.hold_bars < max(0, min_hold_bars))
        desired[must_hold_mask] = prev_pos[must_hold_mask]

        # Enforce maximum hold: close positions that reached max bars
        must_exit_mask = (prev_pos != 0) & (self.hold_bars >= max_hold_bars)
        desired[must_exit_mask] = 0

        # Enforce max entries per day (including flips as entries)
        # Entry candidates are changes from 0→±1 or flips between ±1
        cand_mask = ((prev_pos == 0) & (desired != 0)) | ((prev_pos != 0) & (desired != 0) & (desired != prev_pos))
        if np.any(cand_mask):
            slots = max(0, int(self.cfg.max_entries_per_day) - int(self.daily_entries))
            if slots <= 0:
                # No entries allowed: cancel flips and new entries
                # Keep existing positions; do not allow changing sign
                keep_mask = (prev_pos != 0)
                desired[~keep_mask] = 0
                desired[keep_mask] = prev_pos[keep_mask]
            else:
                idx = np.where(cand_mask)[0]
                if len(idx) > slots:
                    # Prioritize by ATR descending as a simple proxy for opportunity
                    atr_vals = np.array([float(self.atr_map[self.tickers[k]].iloc[self.i]) for k in idx])
                    order = np.argsort(-np.nan_to_num(atr_vals, nan=0.0))
                    allow_idx = set(idx[order[:slots]].tolist())
                    for k in idx:
                        if k not in allow_idx:
                            # If currently flat: cancel entry; if flipping: keep prior sign
                            if prev_pos[k] == 0:
                                desired[k] = 0
                            else:
                                desired[k] = prev_pos[k]

        # Compute transaction costs for position changes in units
        # ATR-based unit sizing magnitude per ticker (capped to avoid blow-ups)
        target_units = np.zeros((self.N,), dtype=np.int64)
        for k, t in enumerate(self.tickers):
            atr_val = self.atr_map.get(t, None)
            try:
                atr = float(atr_val.iloc[self.i]) if atr_val is not None and len(atr_val) > self.i else 0.0
            except Exception:
                atr = 0.0
            atr = max(atr, 1e-6)
            # units = risk_budget / (ATR * point_value)
            denom = atr * max(1e-6, float(self.point_value))
            safe_units = int(getattr(self.cfg, 'units_per_ticker', 100))
            try:
                mag = int(max(0.0, np.floor(float(self.cfg.risk_budget_per_ticker) / denom)))
            except Exception:
                mag = safe_units
            # Cap magnitude to safe limit
            if mag <= 0 or not np.isfinite(mag):
                mag = safe_units
            else:
                mag = int(min(mag, safe_units))
            target_units[k] = int(desired[k]) * mag

        # Apply gross exposure cap by scaling units if needed
        try:
            prices = np.array([
                float(np.nan_to_num(self.ohlcv[t]['close'].iloc[self.i], nan=0.0, posinf=0.0, neginf=0.0))
                for t in self.tickers
            ], dtype=float)
        except Exception:
            prices = np.zeros((self.N,), dtype=float)
        gross_val = float(np.sum(np.abs(target_units) * prices * float(self.point_value)))
        cap = float(self.cfg.max_gross_exposure) * max(1e-6, float(self.equity))
        if cap > 0 and gross_val > cap:
            scale = cap / gross_val
            target_units = (target_units.astype(float) * scale).astype(np.int64)
        change_units = target_units - self.units
        if np.any(change_units != 0):
            # charge per unit changed (commission per share/contract)
            tc = float(np.sum(np.abs(change_units))) * float(self.tc_per_trade)
            self.cash -= tc

        # Set new positions (sign) and units
        self.pos = desired
        self.units = target_units

        # Bar PnL from prev close to current close per ticker
        pnl = 0.0
        for k, t in enumerate(self.tickers):
            prev_p = float(np.nan_to_num(self.ohlcv[t]["close"].iloc[self.i - 1], nan=0.0, posinf=0.0, neginf=0.0))
            cur_p = float(np.nan_to_num(self.ohlcv[t]["close"].iloc[self.i], nan=0.0, posinf=0.0, neginf=0.0))
            pnl += (cur_p - prev_p) * float(self.units[k]) * float(self.point_value)
        # Reward shaping penalties
        turnover = float(np.sum(np.abs(change_units)))
        # Gross exposure value at current bar
        gross_exposure_val = float(np.sum(np.abs(self.units) * prices * float(self.point_value)))
        exposure_violation = max(0.0, gross_exposure_val - cap) if cap > 0 else 0.0

        # Holding penalty: per open position per bar
        open_count = int(np.sum(self.pos != 0))
        hold_pen = float(self.cfg.position_holding_penalty) * float(open_count)

        shaped = pnl \
            - float(self.cfg.turnover_penalty) * turnover \
            - float(self.cfg.exposure_penalty) * exposure_violation \
            - hold_pen
        # sanitize shaped
        if not np.isfinite(shaped):
            shaped = 0.0

        self.cash += float(shaped)
        self.equity = self.cash
        self.equity_curve.append(self.equity)
        # Record step history
        try:
            ts = self.index[self.i]
        except Exception:
            ts = None
        step_turnover = float(np.sum(np.abs(change_units)))
        # Update holding durations and daily entry count after applying new pos
        new_daily_entries = 0
        for k in range(self.N):
            if prev_pos[k] == 0 and self.pos[k] != 0:
                # new entry
                new_daily_entries += 1
                self.hold_bars[k] = 1
            elif prev_pos[k] != 0 and self.pos[k] == 0:
                # exit
                self.hold_bars[k] = 0
            elif prev_pos[k] != 0 and self.pos[k] != 0:
                if self.pos[k] == prev_pos[k]:
                    self.hold_bars[k] = int(self.hold_bars[k]) + 1
                else:
                    # flip counts as exit+entry; reset hold and count as entry
                    new_daily_entries += 1
                    self.hold_bars[k] = 1
            else:
                self.hold_bars[k] = 0
        self.daily_entries += new_daily_entries
        self.history.append({
            'timestamp': ts,
            'equity': float(self.equity),
            'turnover': step_turnover,
            'open_positions': open_count,
            'daily_entries': int(self.daily_entries),
            **{f'pos_{t}': int(self.pos[k]) for k, t in enumerate(self.tickers)},
            **{f'units_{t}': int(self.units[k]) for k, t in enumerate(self.tickers)}
        })

        # Trade bookkeeping: close/flip/open based on position transitions
        cur_close_prices: Dict[str, float] = {}
        prev_close_prices: Dict[str, float] = {}
        for k, t in enumerate(self.tickers):
            try:
                prev_close_prices[t] = float(np.nan_to_num(self.ohlcv[t]['close'].iloc[self.i - 1], nan=0.0))
                cur_close_prices[t] = float(np.nan_to_num(self.ohlcv[t]['close'].iloc[self.i], nan=0.0))
            except Exception:
                prev_close_prices[t] = 0.0
                cur_close_prices[t] = 0.0
        for k, t in enumerate(self.tickers):
            prev = int(prev_pos[k])
            cur = int(self.pos[k])
            units = int(self.units[k])
            # Flip or exit closes existing trade
            if prev != 0 and (cur == 0 or np.sign(cur) != np.sign(prev)):
                ot = self._open_trades.get(t)
                if ot is not None:
                    exit_time = ts
                    exit_price = cur_close_prices.get(t, 0.0)
                    duration_bars = max(1, int(self.i - int(ot.get('entry_i', self.i))))
                    duration_minutes = float(duration_bars) * float(getattr(self, 'bar_minutes', 1.0))
                    dir_sign = 1 if ot.get('direction') == 'long' else -1
                    pnl = dir_sign * (exit_price - float(ot.get('entry_price', 0.0))) * float(abs(int(ot.get('units', 0)))) * float(self.point_value)
                    rec = {
                        'ticker': t,
                        'direction': ot.get('direction'),
                        'entry_time': ot.get('entry_time'),
                        'exit_time': exit_time,
                        'entry_price': float(ot.get('entry_price', 0.0)),
                        'exit_price': float(exit_price),
                        'units': int(abs(int(ot.get('units', 0)))),
                        'duration_bars': int(duration_bars),
                        'duration_minutes': float(duration_minutes),
                        'pnl': float(pnl)
                    }
                    self.trades.append(rec)
                self._open_trades[t] = None
            # Entry or flip opens a new trade
            if cur != 0 and (prev == 0 or np.sign(cur) != np.sign(prev)) and units != 0:
                entry_time = ts
                entry_price = cur_close_prices.get(t, 0.0)
                self._open_trades[t] = {
                    'ticker': t,
                    'direction': 'long' if cur > 0 else 'short',
                    'entry_time': entry_time,
                    'entry_i': int(self.i),
                    'entry_price': float(entry_price),
                    'units': int(abs(units)),
                }

        # Advance time
        self.i += 1
        terminated = bool(self.i >= self.n)
        truncated = False
        reward = float(self.cfg.reward_scaling) * float(shaped)
        if not np.isfinite(reward):
            reward = 0.0
        obs = self._obs(min(self.i, self.n - 1))
        info = {"equity": float(self.equity)}
        if terminated:
            # Snapshot final equity curve for evaluation retrieval
            try:
                import pandas as _pd
                self._last_equity_curve = _pd.Series(self.equity_curve, index=self.index[: len(self.equity_curve)])
            except Exception:
                self._last_equity_curve = self.equity_curve[:]
            # Snapshot history
            try:
                self._last_history = list(self.history)
            except Exception:
                self._last_history = None
            # Snapshot trades before potential auto-reset by VecEnv
            try:
                self._last_trades = list(self.trades)
            except Exception:
                self._last_trades = []
        return obs, float(reward), terminated, truncated, info

    # Helper to export equity curve
    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self.equity_curve, index=self.index[: len(self.equity_curve)])

    def get_history_df(self) -> pd.DataFrame:
        import pandas as _pd
        data = self.history if self.history else (self._last_history or [])
        if not data:
            return _pd.DataFrame()
        df = _pd.DataFrame(data)
        if 'timestamp' in df.columns and df['timestamp'].notna().any():
            df = df.set_index('timestamp')
        return df

    def get_trades(self) -> List[Dict[str, Any]]:
        # Return closed trades; if just finished an episode, use the last snapshot
        if hasattr(self, '_last_trades') and self.trades == []:
            return list(self._last_trades)
        return list(self.trades)
