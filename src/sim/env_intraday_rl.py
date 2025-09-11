"""
Reinforcement Learning environment for intraday trading.

This module provides a comprehensive RL environment with triple-barrier exits,
risk management, and end-of-day flatten functionality. Supports both Polygon
and Databento data formats with automatic detection and handling.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging
import time
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Discrete

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import DifferentialSharpe
from .execution import ExecutionEngine, estimate_tc, ExecParams
from .risk import RiskManager, RiskConfig

logger = get_logger(__name__)


@dataclass
class EnvConfig:
    """Environment configuration."""
    cash: float = 100000.0
    max_steps: int = 390  # 1 minute bars in RTH
    render_mode: str = 'human'
    reward_type: str = 'dsr'  # 'dsr', 'pnl', 'sharpe'
    penalty_factor: float = 0.1
    reward_scaling: float = 0.1


class IntradayRLEnv(Env):
    """
    Reinforcement Learning environment for intraday trading.
    
    This environment provides realistic trading simulation with:
    - Triple-barrier exits (stop loss, take profit, timeout)
    - Risk management and position sizing
    - End-of-day flatten
    - Realistic transaction costs
    - Multiple asset support
    """
    
    def __init__(self,
                 ohlcv: pd.DataFrame,
                 features: pd.DataFrame,
                 cash: float = 100000.0,
                 exec_params: Optional[Dict] = None,
                 risk_cfg: Optional[RiskConfig] = None,
                 point_value: float = 5.0,
                 env_config: Optional[EnvConfig] = None,
                 config: Optional[Dict] = None):
        """
        Initialize RL environment.

        Args:
            ohlcv: DataFrame with OHLCV data
            features: DataFrame with features
            cash: Initial cash
            exec_params: Execution parameters
            risk_cfg: Risk configuration
            point_value: Point value (e.g., $5 for MES)
            env_config: Environment configuration
            config: Configuration dictionary (for compatibility)
        """
        super().__init__()

        # Detect data source and validate data format
        self._detect_data_source(ohlcv, features)
        self._validate_data_format(ohlcv, features)

        # Configuration
        self.cash = cash
        self.point_value = point_value
        self.env_config = env_config or EnvConfig(cash=cash)
        self.config = config  # Store config for compatibility

        # Data
        self.ohlcv = ohlcv[['open', 'high', 'low', 'close', 'volume']].copy()
        self.features = features.reindex(ohlcv.index, method='ffill').copy()
        self.df = ohlcv.copy()  # For compatibility
        self.X = features.copy()  # For compatibility

        # Ensure alignment
        self.ohlcv = self.ohlcv.loc[self.features.index]

        # Initialize components
        logger.debug("Initializing ExecutionEngine with settings from configs/settings.yaml")
        # Use ExecutionEngine (there is no separate ExecutionSimulator class)
        self.exec_sim = ExecutionEngine(settings=Settings.from_paths('configs/settings.yaml'))
        if exec_params:
            logger.debug(f"Processing exec_params: {exec_params}")
            # Handle both dictionary and dataclass ExecParams objects
            if isinstance(exec_params, ExecParams):
                # If it's already an ExecParams dataclass, use it directly
                logger.debug("Using ExecParams dataclass directly")
                self.exec_sim.exec_params = exec_params
            elif isinstance(exec_params, dict):
                # Map parameter names from test format to ExecParams format
                exec_params_mapped = {
                    'commission_per_contract': exec_params.get('transaction_cost', 2.5),
                    'slippage_bps': exec_params.get('slippage', 0.01) * 100,  # Convert to bps
                    'tick_value': exec_params.get('tick_value', 1.25),
                    'spread_ticks': exec_params.get('spread_ticks', 1),
                    'impact_bps': exec_params.get('impact_bps', 0.5),
                    'min_commission': exec_params.get('min_commission', 1.0),
                    'liquidity_threshold': exec_params.get('liquidity_threshold', 1000)
                }
                logger.debug(f"Mapped exec_params: {exec_params_mapped}")
                self.exec_sim.exec_params = ExecParams(**exec_params_mapped)
            else:
                logger.warning(f"Unsupported exec_params type: {type(exec_params)}, using defaults")
                self.exec_sim.exec_params = ExecParams()

        # Create a proper Settings object from the config
        logger.debug("Creating Settings object for configuration")
        settings = Settings.from_paths('configs/settings.yaml')
        logger.debug(f"Checking for config attribute: hasattr(self, 'config') = {hasattr(self, 'config')}")

        if hasattr(self, 'config') and self.config:
            logger.debug(f"Found config attribute with content: {self.config}")
            # Ensure config keys are strings and update proper Settings format
            config_dict = {
                str(key): value
                for key, value in self.config.items()
            }
            logger.debug(f"Converted config_dict: {config_dict}")

            # Set defaults for required keys
            settings._config.update({
                'risk': {
                    'max_position_size': config_dict.get('max_position_size', 5),
                    'max_daily_loss_r': config_dict.get('risk', {}).get('max_daily_loss_r', 0.05),
                    'stop_r_multiple': config_dict.get('risk', {}).get('stop_r_multiple', 2.0),
                    'VaR_level': config_dict.get('risk', {}).get('VaR_level', 0.95),
                    'cvar_level': config_dict.get('risk', {}).get('cvar_level', 0.99)
                },
                'execution': {
                    'transaction_cost': config_dict.get('transaction_cost', 2.5),
                    'slippage': config_dict.get('slippage', 0.01)
                }
            })
            logger.debug(f"Updated settings._config: {settings._config}")
        else:
            logger.warning("No config attribute found or config is empty - using default settings")

        logger.debug(f"Final settings object: {settings._config}")
        self.settings = settings
        logger.debug("Initializing RiskManager with settings")
        self.risk_manager = RiskManager(settings)
        if risk_cfg:
            logger.debug(f"Updating RiskManager with risk_cfg: {risk_cfg}")
            self.risk_manager.risk_config = risk_cfg

        # State tracking
        self.reset_state()

        # Precompute day start indices (first bar at/after 09:30)
        try:
            times = self.df.index.time
            day_mask = (self.df.index.hour == 9) & (self.df.index.minute >= 30)
            # First index per calendar day at or after 09:30
            day_keys = [d.date() for d in self.df.index]
            first_seen = {}
            self._day_starts = []
            for idx, (t, is_rth) in enumerate(zip(day_keys, day_mask)):
                if t not in first_seen and is_rth:
                    first_seen[t] = idx
                    self._day_starts.append(idx)
            # Fallback: if none found, treat the first index as start
            if not self._day_starts and len(self.df) > 0:
                self._day_starts = [0]
            self._day_ptr = -1  # will advance on first reset
        except Exception:
            self._day_starts = [0]
            self._day_ptr = -1

        # Action space: -1 (short), 0 (flat), 1 (long)
        self.action_space = Discrete(3)

        # Observation space: features + position + unrealized P&L
        feature_dim = self.features.shape[1]
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(feature_dim + 2,),
            dtype=np.float32
        )

        # Reward tracking
        self.dsr = DifferentialSharpe()

        logger.info(f"Environment initialized with {len(self.ohlcv)} bars, {feature_dim} features from {getattr(self, 'data_source', 'unknown')} data")
    
    def reset_state(self):
        """Reset environment state."""
        self.cash = self.env_config.cash
        self.pos = 0
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.equity = self.cash
        self.max_equity = self.cash
        self.min_equity = self.cash
        self.equity_curve = [self.cash]
        self.daily_pnl = 0.0
        self.day_start_equity = self.cash
        self.i = 0
        self.done = False
        self.trades = []
        # Daily trade tracking for reward shaping
        self._daily_date = None
        self._daily_trade_count = 0
        self._day_return_sum = 0.0
        self._day_return_count = 0
        # Lagrangian multiplier for activity soft-constraint
        try:
            self._lambda_activity = float(self.config.get('env', {}).get('reward', {}).get('activity', {}).get('lambda_init', 0.0)) if isinstance(self.config, dict) else 0.0
        except Exception:
            self._lambda_activity = 0.0
        # Drawdown acceleration tracking (EMA of dd slope)
        self._prev_dd = 0.0
        self._dd_slope_ema = 0.0
        # Last action direction for churn penalty
        self._last_action_dir = 0
        
        # Reset risk manager
        self.risk_manager.reset_daily_metrics()
    
    def reset(self, seed=None, options: Dict[str, Any] = None):
        """
        Reset environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        # Reset state
        self.reset_state()

        # Determine day start mode (sequential or random) for episode starts
        mode = 'sequential'
        try:
            mode = str(self.config.get('env', {}).get('trading', {}).get('episode_day_mode', 'sequential')) if isinstance(self.config, dict) else 'sequential'
        except Exception:
            mode = 'sequential'
        if mode not in ('sequential', 'random'):
            mode = 'sequential'

        # Choose start index
        if mode == 'random':
            try:
                self.i = int(np.random.choice(self._day_starts))
            except Exception:
                self.i = 0
        else:
            # sequential: advance pointer day by day
            self._day_ptr = (self._day_ptr + 1) % max(1, len(self._day_starts))
            self.i = int(self._day_starts[self._day_ptr])

        # Ensure we land within RTH
        while self.i < len(self.df) and not self._tod(self.df.index[self.i]):
            self.i += 1
        
        ts = self.df.index[self.i]
        obs = self._obs(ts, float(self.df["close"].iloc[self.i]))
        
        return obs, {}
    
    def step(self, action: int):
        step_start_time = time.time()
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (-1, 0, 1)
            
        Returns:
            Observation, reward, done, truncated, info
        """
        # Normalize action to a scalar int (handles numpy types/arrays)
        try:
            # Fast path for numpy/int-like
            if isinstance(action, (np.integer, int)):
                act = int(action)
            else:
                # Flatten arrays/lists to a single scalar
                act = int(np.asarray(action).astype(int).item())
        except Exception:
            act = 1  # default to hold

        # Current bar
        ts = self.df.index[self.i]
        # Reset daily counters on date change
        try:
            cur_date = ts.date()
            if self._daily_date is None or cur_date != self._daily_date:
                self._daily_date = cur_date
                self._daily_trade_count = 0
        except Exception:
            pass
        row = self.df.iloc[self.i]
        price = float(row["close"])
        # Handle NaN price values
        if np.isnan(price) or np.isinf(price):
            logger.warning(f"Invalid price detected: {price}, skipping step")
            # Increment index to move to next step
            self.i += 1
            # Check if we've reached the end
            if self.i >= len(self.df):
                # Return final observation and end episode
                next_ts = self.df.index[min(self.i - 1, len(self.df) - 1)]
                obs = self._obs(next_ts, 0.0)
                info = {"equity": float(self.equity)}
                return obs, float(0.0), True, False, info
            else:
                # Update equity curve for skipped step (maintain previous equity)
                # Ensure equity is not NaN before appending
                if np.isnan(self.equity) or np.isinf(self.equity):
                    self.equity = self.cash  # Reset to cash if equity became invalid
                self.equity_curve.append(float(self.equity))
                # Return safe observation with zero reward and continue
                next_ts = self.df.index[self.i]
                obs = self._obs(next_ts, 0.0)
                info = {"equity": float(self.equity)}
                return obs, float(0.0), False, False, info

        atr_val = float(self.X.loc[ts].get("atr", 0.5))  # fallback small ATR
        if not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = 0.5

        # Track equity prior to this step to compute realized pnl delta
        prev_equity = float(getattr(self, "equity", self.cash))

        # Execute action mapping used by this environment (internal actions 0,1,2 map to -1,0,1 dir)
        # Robust containment check now that act is an int
        desired_dir = {-1: -1, 0: 0, 1: 1}[act - 1] if act in (0, 1, 2) else 0
        reward = 0.0
        info: Dict[str, Any] = {}

        # Flatten at EOD regardless of action
        if self._eod(ts):
            if self.pos != 0:
                # charge closing cost
                tc = estimate_tc(self.pos, price, self.exec_sim)
                # Log close event at EOD
                try:
                    open_ts = getattr(self, '_open_ts', None)
                    entry_tc = float(getattr(self, '_entry_tc', 0.0))
                    qty = int(abs(self.pos))
                    direction = 'long' if self.pos > 0 else 'short'
                    exit_price = float(price)
                    gross_pnl = (exit_price - self.entry_price) * (1 if direction == 'long' else -1) * qty * self.point_value
                    net_pnl = gross_pnl - entry_tc - float(tc)
                    self.trades.append({
                        'ts': ts,
                        'pos': 0,
                        'price': float(exit_price),
                        'action': 'close',
                        'reason': 'eod',
                        'entry_time': open_ts if open_ts is not None else ts,
                        'exit_time': ts,
                        'entry_price': float(self.entry_price),
                        'exit_price': float(exit_price),
                        'quantity': qty,
                        'direction': direction,
                        'pnl': float(net_pnl),
                        'duration_min': float(((ts - open_ts).total_seconds() / 60.0) if open_ts is not None else 0.0),
                        'commission_entry': float(entry_tc),
                        'commission_exit': float(tc)
                    })
                except Exception:
                    pass
                self.cash -= tc
                self.pos = 0
                self.entry_price = None
                self.stop_price = None
                self.tp_price = None
            # EOD: finalize equity for the day
            self.equity = self.cash
            self.max_equity = max(self.max_equity, self.equity)
            self.min_equity = min(self.min_equity, self.equity)
            self.equity_curve.append(self.equity)
            obs = self._obs(ts, price)
            info["equity"] = float(self.equity)
            # Update Lagrange multipliers at EOD
            try:
                tgt = int(self.config.get('env', {}).get('reward', {}).get('activity', {}).get('target_per_day', 0)) if isinstance(self.config, dict) else 0
                eta = float(self.config.get('env', {}).get('reward', {}).get('activity', {}).get('lagrange_eta', 0.0)) if isinstance(self.config, dict) else 0.0
                if tgt > 0 and eta > 0.0:
                    gap = float(self._daily_trade_count - tgt)
                    self._lambda_activity = max(0.0, float(self._lambda_activity) + eta * gap)
            except Exception:
                pass
            try:
                side_eta = float(self.config.get('env', {}).get('reward', {}).get('side_balance', {}).get('lagrange_eta', 0.01)) if isinstance(self.config, dict) else 0.01
                if not hasattr(self, '_lambda_side'):
                    self._lambda_side = float(self.config.get('env', {}).get('reward', {}).get('side_balance', {}).get('lambda_init', 0.0)) if isinstance(self.config, dict) else 0.0
                gap_side = abs(int(getattr(self, '_daily_longs_open', 0)) - int(getattr(self, '_daily_shorts_open', 0)))
                self._lambda_side = max(0.0, float(self._lambda_side) + side_eta * float(gap_side))
            except Exception:
                pass
            return obs, float(0.0), True, False, info

        # Update realized PnL between bars
        if self.i > 0:
            prev_price = float(self.df["close"].iloc[self.i - 1])
            bar_pnl = self._bar_pnl(prev_price, price, self.pos)
            self.cash += bar_pnl

        # Check triple-barrier exits (if in a trade)
        hit_exit = False
        if self.pos != 0 and self.entry_price is not None and self.stop_price is not None and self.tp_price is not None:
            if (self.pos > 0 and (row["low"] <= self.stop_price or row["high"] >= self.tp_price)) or \
               (self.pos < 0 and (row["high"] >= self.stop_price or row["low"] <= self.tp_price)):
                # Exit at barrier (assume touch -> fill)
                exit_price = self.tp_price if (
                    (self.pos > 0 and row["high"] >= self.tp_price) or
                    (self.pos < 0 and row["low"] <= self.tp_price)
                ) else self.stop_price
                pnl_exit = (exit_price - self.entry_price) * self.pos * self.point_value
                tc_exit = estimate_tc(self.pos, float(exit_price), self.exec_sim)
                self.cash += pnl_exit - tc_exit
                # Log trade close with PnL/duration metadata
                try:
                    open_ts = getattr(self, '_open_ts', None)
                    entry_tc = float(getattr(self, '_entry_tc', 0.0))
                    qty = int(abs(self.pos))
                    direction = 'long' if self.pos > 0 else 'short'
                    gross_pnl = (exit_price - self.entry_price) * (1 if direction == 'long' else -1) * qty * self.point_value
                    net_pnl = gross_pnl - entry_tc - float(tc_exit)
                    self.trades.append({
                        'ts': ts,
                        'pos': 0,
                        'price': float(exit_price),
                        'action': 'close',
                        'entry_time': open_ts if open_ts is not None else ts,
                        'exit_time': ts,
                        'entry_price': float(self.entry_price),
                        'exit_price': float(exit_price),
                        'quantity': qty,
                        'direction': direction,
                        'pnl': float(net_pnl),
                        'duration_min': float(((ts - open_ts).total_seconds() / 60.0) if open_ts is not None else 0.0),
                        'commission_entry': float(entry_tc),
                        'commission_exit': float(tc_exit)
                    })
                except Exception:
                    pass
                self.pos = 0
                self.entry_price = None
                self.stop_price = None
                self.tp_price = None
                hit_exit = True

        # Time-stop (max holding minutes)
        try:
            max_hold = int(self.config.get('env', {}).get('trading', {}).get('max_holding_minutes', 0)) if isinstance(self.config, dict) else 0
        except Exception:
            max_hold = 0
        if max_hold and self.pos != 0:
            # approximate bar count equals minutes since entry
            # track entry step index lazily
            if not hasattr(self, 'entry_index') or self.entry_index is None:
                self.entry_index = self.i
            held = self.i - int(self.entry_index)
            if held >= max_hold:
                # flatten at market price with transaction cost
                tc_exit = estimate_tc(self.pos, price, self.exec_sim)
                try:
                    # Log trade close with PnL/duration metadata
                    open_ts = getattr(self, '_open_ts', None)
                    entry_tc = float(getattr(self, '_entry_tc', 0.0))
                    qty = int(abs(self.pos))
                    direction = 'long' if self.pos > 0 else 'short'
                    exit_price = float(price)
                    gross_pnl = (exit_price - self.entry_price) * (1 if direction == 'long' else -1) * qty * self.point_value
                    net_pnl = gross_pnl - entry_tc - float(tc_exit)
                    self.trades.append({
                        'ts': ts,
                        'pos': 0,
                        'price': float(exit_price),
                        'action': 'close',
                        'entry_time': open_ts if open_ts is not None else ts,
                        'exit_time': ts,
                        'entry_price': float(self.entry_price),
                        'exit_price': float(exit_price),
                        'quantity': qty,
                        'direction': direction,
                        'pnl': float(net_pnl),
                        'duration_min': float(((ts - open_ts).total_seconds() / 60.0) if open_ts is not None else 0.0),
                        'commission_entry': float(entry_tc),
                        'commission_exit': float(tc_exit)
                    })
                except Exception:
                    pass
                self.cash -= tc_exit
                self.pos = 0
                self.entry_price = None
                self.stop_price = None
                self.tp_price = None
                hit_exit = True

        # Trading windows control (no-trade in first/last N minutes)
        try:
            open_min = int(self.config.get('env', {}).get('trading', {}).get('no_trade_open_minutes', 0)) if isinstance(self.config, dict) else 0
            close_min = int(self.config.get('env', {}).get('trading', {}).get('no_trade_close_minutes', 0)) if isinstance(self.config, dict) else 0
        except Exception:
            open_min = close_min = 0
        minutes_since_midnight = ts.hour * 60 + ts.minute
        session_open = 9 * 60 + 30
        session_close = 16 * 60
        in_no_trade = (open_min and session_open <= minutes_since_midnight < session_open + open_min) or \
                      (close_min and session_close - close_min <= minutes_since_midnight < session_close)

        # Apply desired action if flat (or change direction after exit) and not in no-trade window
        # Optional training-time forcing: with small probability, flip a hold into a side to seed exploration
        try:
            force_eps = float(self.config.get('env', {}).get('trading', {}).get('force_open_epsilon', 0.0)) if isinstance(self.config, dict) else 0.0
            force_frac = float(self.config.get('env', {}).get('trading', {}).get('force_warmup_frac', 0.0)) if isinstance(self.config, dict) else 0.0
        except Exception:
            force_eps = 0.0
            force_frac = 0.0

        # Warm-up forcing only in first portion of the session
        in_warmup = False
        try:
            session_open = 9 * 60 + 30
            session_close = 16 * 60
            minutes_since_midnight = ts.hour * 60 + ts.minute
            if session_open <= minutes_since_midnight < session_close and (session_close - session_open) > 0:
                frac_elapsed = (minutes_since_midnight - session_open) / float(session_close - session_open)
                in_warmup = (frac_elapsed <= force_frac)
        except Exception:
            in_warmup = False

        if self.pos == 0 and desired_dir == 0 and not in_no_trade and force_eps > 0.0 and in_warmup:
            try:
                if np.random.rand() < force_eps:
                    desired_dir = 1 if np.random.rand() < 0.5 else -1
            except Exception:
                pass

        if desired_dir != 0 and self.pos == 0 and not hit_exit and not in_no_trade:
            # Cap trades per hour
            try:
                max_tph = int(self.config.get('env', {}).get('trading', {}).get('max_trades_per_hour', 0)) if isinstance(self.config, dict) else 0
            except Exception:
                max_tph = 0
            if max_tph:
                recent = [t for t in self.trades if t.get('action') == 'open' and (ts - t['ts']) <= pd.Timedelta(minutes=60)]
                if len(recent) >= max_tph:
                    desired_dir = 0
            contracts = self._risk_sized_contracts(price, atr_val)
            if contracts > 0:
                # open position
                self.pos = contracts * int(np.sign(desired_dir))
                self.entry_price = price
                self._set_barrier_prices(self.pos, price, atr_val)
                # pay entry cost
                entry_tc = estimate_tc(self.pos, price, self.exec_sim)
                self.cash -= entry_tc
                try:
                    self.trades.append({
                        'ts': ts,
                        'pos': int(self.pos),
                        'price': float(price),
                        'action': 'open'
                    })
                    # Track open trade context for close logging
                    self._open_ts = ts
                    self._entry_tc = float(entry_tc)
                    # Increment daily trade count
                    self._daily_trade_count = int(self._daily_trade_count) + 1
                    # Side-balance counters (opens)
                    if self.pos > 0:
                        self._daily_longs_open = int(getattr(self, '_daily_longs_open', 0)) + 1
                    elif self.pos < 0:
                        self._daily_shorts_open = int(getattr(self, '_daily_shorts_open', 0)) + 1
                except Exception:
                    pass
                self.entry_index = self.i
                self.scale_out_done = False
            else:
                # Helpful debug breadcrumbs when trades fail to open
                logger.debug(
                    "Skip open: contracts=0 (risk sizing) | price=%.4f atr=%.4f equity=%.2f stop_r=%.3f",
                    price, atr_val, self.equity,
                    float(getattr(self.risk_manager.risk_config, 'stop_r_multiple', 1.0))
                )
        elif desired_dir != 0 and self.pos == 0 and in_no_trade:
            logger.debug("Skip open: in no-trade window at %s", ts)

        # Partial scale-out when in profit (optional)
        try:
            scale_r = float(self.config.get('env', {}).get('trading', {}).get('scale_out_r_multiple', 0.0)) if isinstance(self.config, dict) else 0.0
        except Exception:
            scale_r = 0.0
        if scale_r and self.pos != 0 and not self.scale_out_done and self.entry_price is not None and atr_val > 0:
            r_unreal = ((price - self.entry_price) * np.sign(self.pos)) / (atr_val * max(1e-6, float(self.risk_manager.risk_config.stop_r_multiple)))
            if r_unreal >= scale_r and abs(self.pos) > 1:
                half = int(abs(self.pos) // 2) * int(np.sign(self.pos))
                self.cash += (price - self.entry_price) * half * self.point_value
                self.cash -= estimate_tc(half, price, self.exec_sim)
                self.pos -= half
                try:
                    self.trades.append({'ts': ts, 'pos': int(self.pos), 'price': float(price), 'action': 'scale_out'})
                except Exception:
                    pass
                self.scale_out_done = True

        # Update equity and equity curve
        self.equity = self.cash
        self.max_equity = max(self.max_equity, self.equity)
        self.min_equity = min(self.min_equity, self.equity)
        self.equity_curve.append(self.equity)

        # Compute pnl for reward as realized delta in equity this step
        pnl = float(self.equity - prev_equity)

        # Compute risk penalties per modified_step logic
        # Drawdown as fraction from peak equity
        if self.max_equity > 0:
            current_dd = (self.max_equity - self.equity) / max(self.max_equity, 1e-6)
        else:
            current_dd = 0.0

        # Track/update max_drawdown ratio attribute for scaling
        if not hasattr(self, "max_drawdown"):
            self.max_drawdown = 0.0
        self.max_drawdown = max(float(self.max_drawdown), float(current_dd))

        # Drawdown penalty scales with ratio of current drawdown to max drawdown
        drawdown_penalty = (current_dd / self.max_drawdown) if self.max_drawdown > 0 else 0.0

        # Realized daily loss relative to start-of-day equity
        self.realized_drawdown = (self.day_start_equity - self.equity) / max(self.day_start_equity, 1e-6)
        max_daily_pct = float(self.risk_manager.risk_config.max_daily_loss_r) * 0.01
        realised_fraction = (self.realized_drawdown / max_daily_pct) if max_daily_pct > 0 else 0.0
        risk_penalty = max(0.0, float(realised_fraction))  # penalize only when drawdown is present

        # --- Reward Calculation ---
        # Compute bar return for optional directional shaping (de-meaned within day)
        try:
            prev_price = float(self.df["close"].iloc[self.i - 1]) if self.i > 0 else price
            bar_return = (price - prev_price) / max(prev_price, 1e-8)
        except Exception:
            bar_return = 0.0

        # Update daily mean return trackers (use previous mean to de-mean current return)
        try:
            day_mean_ret = (self._day_return_sum / max(1, self._day_return_count))
        except Exception:
            day_mean_ret = 0.0
        # Remove de-meaning bias - use raw bar_return for directional rewards
        # The de-meaning was penalizing correct directional bets
        eff_bar_return = float(bar_return)  # Use raw return, not de-meaned
        # Update running stats after computing eff return
        try:
            self._day_return_sum += float(bar_return)
            self._day_return_count += 1
        except Exception:
            pass
        # Compute normalized PnL using ATR if requested
        try:
            pnl_cap = float(self.config.get('env', {}).get('reward', {}).get('pnl_cap', 1.0)) if isinstance(self.config, dict) else 1.0
        except Exception:
            pnl_cap = 1.0
        pnl_norm = 0.0
        try:
            denom = max(1e-6, atr_val * float(getattr(self, 'point_value', 1.0)))
            pnl_norm = float(np.clip(pnl / denom, -pnl_cap, pnl_cap))
        except Exception:
            pnl_norm = 0.0

        # Drawdown acceleration penalty (EMA of dd slope)
        try:
            dd = (self.equity / max(self.max_equity, 1e-6)) - 1.0
            dd_slope = float(dd - self._prev_dd)
            self._prev_dd = float(dd)
            # EMA update
            dd_alpha = 0.2
            self._dd_slope_ema = (1 - dd_alpha) * self._dd_slope_ema + dd_alpha * dd_slope
        except Exception:
            dd_slope = 0.0

        if self.env_config.reward_type == 'pnl':
            reward = pnl
        elif self.env_config.reward_type == 'dsr':
            step_return = pnl / prev_equity if prev_equity != 0 else 0.0
            reward = self.dsr.update(step_return)
        elif self.env_config.reward_type == 'sharpe':
            # Note: simplified Sharpe ratio proxy per step
            step_return = pnl / prev_equity if prev_equity != 0 else 0.0
            reward = step_return / (np.std(self.equity_curve) + 1e-8) if len(self.equity_curve) > 1 else 0.0
        elif self.env_config.reward_type == 'blend':
            # Blended reward: alpha * DSR + beta * raw_pnl - penalties
            alpha = float(self.config.get('env', {}).get('reward', {}).get('alpha', 0.5)) if isinstance(self.config, dict) else 0.5
            beta = float(self.config.get('env', {}).get('reward', {}).get('beta', 0.5)) if isinstance(self.config, dict) else 0.5
            step_return = pnl / prev_equity if prev_equity != 0 else 0.0
            dsr_val = self.dsr.update(step_return)
            reward = alpha * dsr_val + beta * pnl
            # Microstructure penalty (optional): discourage trading in poor liquidity
            try:
                rvol = float(self.X.loc[ts].get('rvol', 1.0))
                spread = float(self.X.loc[ts].get('spread', 0.0))
                mu_pen = float(self.config.get('env', {}).get('reward', {}).get('micro_penalty', 0.0)) if isinstance(self.config, dict) else 0.0
                # time-of-day widening: add extra penalty during first/last widen window
                widen_min = int(self.config.get('env', {}).get('trading', {}).get('widen_spread_minutes', 0)) if isinstance(self.config, dict) else 0
                minutes_since_midnight = ts.hour * 60 + ts.minute
                session_open = 9 * 60 + 30
                session_close = 16 * 60
                in_widen = (widen_min and session_open <= minutes_since_midnight < session_open + widen_min) or \
                           (widen_min and session_close - widen_min <= minutes_since_midnight < session_close)
                widen_factor = 1.0 + (0.25 if in_widen else 0.0)
                # apply micro penalty only under poor liquidity
                if rvol < 1.0:
                    reward -= mu_pen * widen_factor * (max(0.0, 1.5 - rvol) + spread)
            except Exception:
                pass
            # Optional activity shaping toward target trades/day
            try:
                if self.entry_index == self.i:
                    target = int(self.config.get('env', {}).get('reward', {}).get('trade_target_per_day', 2)) if isinstance(self.config, dict) else 2
                    bonus = float(self.config.get('env', {}).get('reward', {}).get('trade_activity_bonus', 0.0)) if isinstance(self.config, dict) else 0.0
                    penalty = float(self.config.get('env', {}).get('reward', {}).get('trade_activity_penalty', 0.0)) if isinstance(self.config, dict) else 0.0
                    if self._daily_trade_count <= target:
                        reward += bonus
                    else:
                        # penalize excess trades beyond target
                        reward -= penalty * max(0, int(self._daily_trade_count) - target)
                else:
                    # Backward-compatible open bonus
                    open_bonus = float(self.config.get('env', {}).get('reward', {}).get('open_bonus', 0.0)) if isinstance(self.config, dict) else 0.0
                    if open_bonus > 0 and self.entry_index == self.i:
                        reward += open_bonus
            except Exception:
                pass
            reward -= float(drawdown_penalty) + float(risk_penalty)
        elif self.env_config.reward_type == 'directional':
            # Pure directional shaping on chosen action (short=-1, hold=0, long=1)
            # Encourages taking a side even before a position is opened.
            try:
                dir_w = float(self.config.get('env', {}).get('reward', {}).get('dir_weight', 100.0)) if isinstance(self.config, dict) else 100.0
            except Exception:
                dir_w = 100.0
            reward = dir_w * float(desired_dir) * float(eff_bar_return)
            reward -= float(drawdown_penalty) + float(risk_penalty)
        elif self.env_config.reward_type == 'hybrid':
            # Hybrid: blend of DSR, PnL, and directional shaping
            try:
                alpha = float(self.config.get('env', {}).get('reward', {}).get('alpha', 0.2)) if isinstance(self.config, dict) else 0.2
                beta = float(self.config.get('env', {}).get('reward', {}).get('beta', 0.3)) if isinstance(self.config, dict) else 0.3
                dir_w = float(self.config.get('env', {}).get('reward', {}).get('dir_weight', 100.0)) if isinstance(self.config, dict) else 100.0
            except Exception:
                alpha, beta, dir_w = 0.2, 0.3, 100.0
            step_return = pnl / prev_equity if prev_equity != 0 else 0.0
            dsr_val = self.dsr.update(step_return)
            reward = alpha * dsr_val + beta * pnl + dir_w * float(desired_dir) * float(eff_bar_return)
            reward -= float(drawdown_penalty) + float(risk_penalty)
        elif self.env_config.reward_type == 'hybrid2':
            # Best-practice hybrid: normalized PnL + DSR + drift-neutral directional + activity soft-constraint + churn + dd accel
            try:
                alpha = float(self.config.get('env', {}).get('reward', {}).get('alpha', 0.15)) if isinstance(self.config, dict) else 0.15
                beta = float(self.config.get('env', {}).get('reward', {}).get('beta', 0.25)) if isinstance(self.config, dict) else 0.25
                dir_w = float(self.config.get('env', {}).get('reward', {}).get('dir_weight', 300.0)) if isinstance(self.config, dict) else 300.0
                churn_pen = float(self.config.get('env', {}).get('reward', {}).get('churn_penalty', 0.0)) if isinstance(self.config, dict) else 0.0
                dd_accel_pen = float(self.config.get('env', {}).get('reward', {}).get('dd_accel_penalty', 0.0)) if isinstance(self.config, dict) else 0.0
            except Exception:
                alpha, beta, dir_w, churn_pen, dd_accel_pen = 0.15, 0.25, 300.0, 0.0, 0.0
            # Regime weighting via VIX if available
            try:
                vix = float(self.X.loc[ts].get('vix', np.nan))
                if vix == vix:
                    vix_low = float(self.config.get('env', {}).get('reward', {}).get('regime_weights', {}).get('vix_low', 15.0))
                    vix_high = float(self.config.get('env', {}).get('reward', {}).get('regime_weights', {}).get('vix_high', 25.0))
                    w_low = float(self.config.get('env', {}).get('reward', {}).get('regime_weights', {}).get('dir_weight_low', 0.7))
                    w_high = float(self.config.get('env', {}).get('reward', {}).get('regime_weights', {}).get('dir_weight_high', 1.2))
                    regime_mult = w_low if vix < vix_low else (w_high if vix > vix_high else 1.0)
                else:
                    regime_mult = 1.0
            except Exception:
                regime_mult = 1.0
            # Activity term (apply on opens tracked separately below)
            activity_term = 0.0
            # Churn penalty: penalize side flips when flat
            if self.pos == 0 and desired_dir != 0 and self._last_action_dir != 0 and desired_dir != self._last_action_dir:
                activity_term -= churn_pen
            # Drawdown acceleration penalty (only penalize positive slope)
            if self._dd_slope_ema > 0:
                activity_term -= dd_accel_pen * float(self._dd_slope_ema)
            # Core blend
            step_return = pnl / prev_equity if prev_equity != 0 else 0.0
            dsr_val = self.dsr.update(step_return)
            reward = alpha * dsr_val + beta * pnl_norm + dir_w * regime_mult * float(desired_dir) * float(eff_bar_return) + activity_term
            reward -= float(drawdown_penalty) + float(risk_penalty)
        else:  # Default to pnl with penalties
            reward = pnl - float(drawdown_penalty) - float(risk_penalty)

        # Optional hold penalty to discourage persistent inactivity when flat
        try:
            hold_pen = float(self.config.get('env', {}).get('reward', {}).get('hold_penalty', 0.0)) if isinstance(self.config, dict) else 0.0
            if hold_pen > 0.0 and desired_dir == 0 and self.pos == 0:
                reward -= hold_pen
        except Exception:
            pass

        # Optional activity shaping toward target trades/day (apply for blend/hybrid/hybrid2)
        try:
            if self.entry_index == self.i:
                # Open just happened this bar
                target = int(self.config.get('env', {}).get('reward', {}).get('trade_target_per_day',
                                         self.config.get('env', {}).get('reward', {}).get('activity', {}).get('target_per_day', 2))) if isinstance(self.config, dict) else 2
                bonus = float(self.config.get('env', {}).get('reward', {}).get('trade_activity_bonus',
                                         self.config.get('env', {}).get('reward', {}).get('activity', {}).get('bonus', 0.0))) if isinstance(self.config, dict) else 0.0
                if self._daily_trade_count <= target:
                    reward += bonus
                else:
                    # Lagrangian penalty: penalize each open beyond target by current lambda
                    reward -= float(self._lambda_activity)
                # Side-balance penalty: penalize imbalance between long/short opens
                try:
                    side_eta = float(self.config.get('env', {}).get('reward', {}).get('side_balance', {}).get('lagrange_eta', 0.01)) if isinstance(self.config, dict) else 0.01
                    if not hasattr(self, '_lambda_side'):
                        self._lambda_side = float(self.config.get('env', {}).get('reward', {}).get('side_balance', {}).get('lambda_init', 0.0)) if isinstance(self.config, dict) else 0.0
                    gap = abs(int(getattr(self, '_daily_longs_open', 0)) - int(getattr(self, '_daily_shorts_open', 0)))
                    reward -= float(self._lambda_side) * float(gap)
                except Exception:
                    pass
            else:
                open_bonus = float(self.config.get('env', {}).get('reward', {}).get('open_bonus', 0.0)) if isinstance(self.config, dict) else 0.0
                if open_bonus > 0 and getattr(self, 'entry_index', None) == self.i:
                    reward += open_bonus
        except Exception:
            pass

        reward *= self.env_config.reward_scaling
        reward = np.clip(reward, -1, 1)
        # Track last action dir for churn evaluation next step
        try:
            self._last_action_dir = int(np.sign(desired_dir))
        except Exception:
            pass

        # Daily kill-switch: terminate when realized drawdown exceeds threshold
        done = False
        if self.realized_drawdown > max_daily_pct:
            # Force flat and charge closing costs
            if self.pos != 0:
                tc = estimate_tc(self.pos, price, self.exec_sim)
                try:
                    # Log close due to kill-switch
                    open_ts = getattr(self, '_open_ts', None)
                    entry_tc = float(getattr(self, '_entry_tc', 0.0))
                    qty = int(abs(self.pos))
                    direction = 'long' if self.pos > 0 else 'short'
                    exit_price = float(price)
                    gross_pnl = (exit_price - self.entry_price) * (1 if direction == 'long' else -1) * qty * self.point_value
                    net_pnl = gross_pnl - entry_tc - float(tc)
                    self.trades.append({
                        'ts': ts,
                        'pos': 0,
                        'price': float(exit_price),
                        'action': 'close',
                        'reason': 'kill_switch',
                        'entry_time': open_ts if open_ts is not None else ts,
                        'exit_time': ts,
                        'entry_price': float(self.entry_price),
                        'exit_price': float(exit_price),
                        'quantity': qty,
                        'direction': direction,
                        'pnl': float(net_pnl),
                        'duration_min': float(((ts - open_ts).total_seconds() / 60.0) if open_ts is not None else 0.0),
                        'commission_entry': float(entry_tc),
                        'commission_exit': float(tc)
                    })
                except Exception:
                    pass
                self.cash -= tc
                self.pos = 0
                self.entry_price = None
                self.stop_price = None
                self.tp_price = None
            self.equity = self.cash
            done = True

        # Advance index/end-of-data check
        self.i += 1
        if self.i >= len(self.df):
            done = True

        next_ts = self.df.index[min(self.i, len(self.df) - 1)]
        next_price = float(self.df["close"].iloc[min(self.i, len(self.df) - 1)])

        # Handle NaN price values for observation
        if np.isnan(next_price) or np.isinf(next_price):
            next_price = 0.0

        obs = self._obs(next_ts, next_price)

        logger.info("Before returning from step")
        # Diagnostic info
        info = {
            "pnl": float(pnl),
            "drawdown_penalty": float(drawdown_penalty),
            "risk_penalty": float(risk_penalty),
            "realized_drawdown": float(self.realized_drawdown),
        }
        info["equity"] = float(self.equity)
        logger.info(f"Total step time: {time.time() - step_start_time:.6f}s")
        return obs, float(reward), bool(done), False, info
    
    def _obs(self, ts, price):
        """Construct observation vector."""
        # Get features at timestamp
        if ts in self.X.index:
            feats = self.X.loc[ts].values.astype(np.float32)
        else:
            # Fallback to last available features
            feats = self.X.iloc[-1].values.astype(np.float32)

        # Handle NaN values in features
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Position and unrealized P&L
        unreal = 0.0
        if self.entry_price is not None:
            unreal = (price - self.entry_price) * (np.sign(self.pos)) * self.point_value
        obs = np.concatenate([feats, np.array([self.pos, unreal], dtype=np.float32)])
        return obs
    
    def _bar_pnl(self, prev_price, curr_price, pos):
        """Calculate P&L for the bar."""
        if pos == 0:
            return 0.0
        return (curr_price - prev_price) * pos * self.point_value
    
    def _risk_sized_contracts(self, price, atr):
        """Calculate position size based on risk rules using the tighter of ATR and swing stops."""
        try:
            stop_r = float(self.risk_manager.risk_config.stop_r_multiple)
        except Exception:
            stop_r = 1.0
        atr_stop = max(atr * stop_r, 1e-6)
        # Swing-based stop distance from features
        buffer_atr = 0.1 * max(atr, 1e-6)
        swing_stop = None
        try:
            if self.pos == 0:
                # if we are about to open long (desired_dir handled outside), estimate both directions by proximity
                last_sl = float(self.X.loc[self.df.index[self.i]].get('dist_last_swing_low', np.nan))
                last_sh = float(self.X.loc[self.df.index[self.i]].get('dist_last_swing_high', np.nan))
                if last_sl == last_sl:  # not NaN
                    swing_stop_long = max((price - (price - last_sl) + buffer_atr), 1e-6)
                else:
                    swing_stop_long = np.inf
                if last_sh == last_sh:
                    swing_stop_short = max(((price + last_sh) - price + buffer_atr), 1e-6)
                else:
                    swing_stop_short = np.inf
                # pick the smaller as an estimate of plausible stop distance
                swing_stop = min(swing_stop_long, swing_stop_short)
        except Exception:
            swing_stop = None
        stop_distance = atr_stop
        if swing_stop is not None and np.isfinite(swing_stop):
            stop_distance = max(1e-6, min(atr_stop, swing_stop))

        # Calculate contracts from 2% risk rule
        risk_per_trade = self.equity * float(self.risk_manager.risk_config.risk_per_trade_frac)
        point_value = float(getattr(self, 'point_value', 1.0))
        risk_per_contract = stop_distance * point_value
        raw_contracts = int(max(0, risk_per_trade / max(risk_per_contract, 1e-6)))

        # Apply leverage/notional cap
        max_notional = self.equity * float(getattr(self.risk_manager.risk_config, 'max_leverage', 1.0))
        by_notional = int(max_notional / max(price, 1e-6))
        max_pos = int(getattr(self.risk_manager.risk_config, 'max_position_size', 100000))
        contracts_int = int(max(0, min(raw_contracts, by_notional, max_pos)))
        return contracts_int
    
    def _set_barrier_prices(self, pos, price, atr):
        """Set stop loss and take profit prices."""
        # Use ATR-based as baseline
        stop, tp = self.risk_manager.calculate_stop_prices(price, pos, atr)
        # Use swing stop if tighter
        try:
            last_sl = float(self.X.loc[self.df.index[self.i]].get('dist_last_swing_low', np.nan))
            last_sh = float(self.X.loc[self.df.index[self.i]].get('dist_last_swing_high', np.nan))
            buffer_atr = 0.1 * atr
            if pos > 0 and last_sl == last_sl:
                swing_stop_price = price - max((price - (price - last_sl) + buffer_atr), 1e-6)
                stop = max(min(stop, swing_stop_price), 0.0)
            if pos < 0 and last_sh == last_sh:
                swing_stop_price = price + max(((price + last_sh) - price + buffer_atr), 1e-6)
                stop = min(max(stop, swing_stop_price), price + 10 * atr)
        except Exception:
            pass
        self.stop_price, self.tp_price = stop, tp
    
    def _tod(self, ts):
        """Check if timestamp is within trading day."""
        return ts.time() >= pd.Timestamp('09:30').time() and ts.time() <= pd.Timestamp('16:00').time()
    
    def _eod(self, ts):
        """Check if timestamp is end of day."""
        return ts.time() >= pd.Timestamp('16:00').time()

    def _detect_data_source(self, ohlcv: pd.DataFrame, features: pd.DataFrame):
        """
        Detect the data source (Polygon or Databento) based on data characteristics.

        Args:
            ohlcv: OHLCV DataFrame
            features: Features DataFrame
        """
        # Check for Polygon-specific indicators
        polygon_indicators = ['vwap', 'transactions']
        databento_indicators = ['bid', 'ask']  # Databento uses 'bid'/'ask' instead of 'bid_price'/'ask_price'

        polygon_score = sum(1 for col in polygon_indicators if col in ohlcv.columns)
        databento_score = sum(1 for col in databento_indicators if col in ohlcv.columns)

        if polygon_score > databento_score:
            self.data_source = 'polygon'
        elif databento_score > polygon_score:
            self.data_source = 'databento'
        else:
            self.data_source = 'unknown'

        logger.info(f"Detected data source: {self.data_source}")

    def _validate_data_format(self, ohlcv: pd.DataFrame, features: pd.DataFrame):
        """
        Validate that the data format is compatible with the environment.

        Args:
            ohlcv: OHLCV DataFrame
            features: Features DataFrame

        Raises:
            ValueError: If data format is invalid
        """
        # Check required OHLCV columns
        required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in required_ohlcv_cols if col not in ohlcv.columns]
        if missing_ohlcv:
            raise ValueError(f"Missing required OHLCV columns: {missing_ohlcv}")

        # Check for DatetimeIndex
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("OHLCV data must have DatetimeIndex")

        if not isinstance(features.index, pd.DatetimeIndex):
            raise ValueError("Features data must have DatetimeIndex")

        # Check data alignment
        if len(ohlcv) != len(features):
            logger.warning(f"OHLCV ({len(ohlcv)}) and features ({len(features)}) have different lengths")

        # Check for NaN values in critical columns
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in critical_cols:
            if col in ohlcv.columns:
                nan_count = ohlcv[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in {col} column")

        # Polygon-specific validations
        if getattr(self, 'data_source', None) == 'polygon':
            # Check timestamp precision (Polygon uses milliseconds)
            if len(ohlcv) > 0:
                first_ts = ohlcv.index[0]
                if hasattr(first_ts, 'nanosecond'):
                    # If nanoseconds are present, likely from Polygon millisecond conversion
                    if first_ts.nanosecond > 0:
                        logger.debug("Detected millisecond-precision timestamps (Polygon format)")

            # Log Polygon-specific fields if present
            polygon_fields = ['vwap', 'transactions']
            present_fields = [field for field in polygon_fields if field in ohlcv.columns]
            if present_fields:
                logger.info(f"Polygon-specific fields detected: {present_fields}")

        logger.info("Data format validation completed successfully")

    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.i}, Equity: ${self.equity:,.2f}, Position: {self.pos}")

    def close(self):
        """Close environment."""
        pass
    
    def get_equity_curve(self):
        """Get equity curve."""
        return pd.Series(self.equity_curve, index=self.df.index[:len(self.equity_curve)])
    
    def get_trades(self):
        """Get trade history."""
        return self.trades
    
    def get_performance_metrics(self):
        """Get performance metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change().dropna()
        
        metrics = {
            'total_return': (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0],
            'annual_return': (1 + returns.mean()) ** 252 - 1,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (equity_curve / equity_curve.cummax() - 1).min(),
            'calmar_ratio': returns.mean() / abs((equity_curve / equity_curve.cummax() - 1).min()) * 252,
            'win_rate': len(returns[returns > 0]) / len(returns),
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
        }
        
        return metrics


# For compatibility with the original specification
def create_env(settings: Settings) -> IntradayRLEnv:
    """
    Create environment from settings.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Initialized environment
    """
    # This would be implemented to load data and create environment
    # For now, return a placeholder
    return IntradayRLEnv(
        ohlcv=pd.DataFrame(),
        features=pd.DataFrame(),
        cash=100000.0,
        exec_params={},
        risk_cfg=RiskConfig(),
        point_value=5.0
    )


# Alias for compatibility
class IntradayRLEnvironment(IntradayRLEnv):
    """
    Compatibility wrapper for IntradayRLEnvironment to match test expectations.
    
    This class provides the interface expected by the tests while internally
    using the enhanced IntradayRLEnv implementation.
    """
    
    def __init__(self, market_data=None, config=None, **kwargs):
        """
        Initialize environment with test-compatible interface.
        
        Args:
            market_data: DataFrame with market data (OHLCV and microstructure)
            config: Configuration dictionary
            **kwargs: Additional arguments for compatibility
        """
        logger.debug(f"IntradayRLEnvironment.__init__ called with market_data={market_data is not None}, config={config is not None}")
        logger.debug(f"Config content: {config}")
        
        if market_data is None or config is None:
            logger.debug("Using default initialization for empty environment")
            # Default initialization for empty environment
            super().__init__(
                ohlcv=pd.DataFrame(),
                features=pd.DataFrame(),
                cash=100000.0,
                exec_params={},
                risk_cfg=RiskConfig(),
                point_value=5.0
            )
            return
        
        logger.debug("Processing market_data and config for full initialization")
        # Extract OHLCV data
        ohlcv = market_data[['open', 'high', 'low', 'close', 'volume']].copy()
        logger.debug(f"Extracted OHLCV data shape: {ohlcv.shape}")
        
        # Extract features from config
        features_list = []
        if 'features' in config:
            logger.debug(f"Processing features config: {config['features']}")
            # Technical features
            if 'technical' in config['features']:
                logger.debug(f"Processing technical features: {config['features']['technical']}")
                for feat in config['features']['technical']:
                    if feat.startswith('sma_'):
                        window = int(feat.split('_')[1])
                        ohlcv[feat] = ohlcv['close'].rolling(window=window).mean()
                    elif feat.startswith('rsi_'):
                        window = int(feat.split('_')[1])
                        delta = ohlcv['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                        rs = gain / loss
                        ohlcv[feat] = 100 - (100 / (1 + rs))
                    elif feat.startswith('atr_'):
                        window = int(feat.split('_')[1])
                        high_low = ohlcv['high'] - ohlcv['low']
                        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift())
                        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift())
                        ranges = pd.concat([high_low, high_close, low_close], axis=1)
                        true_range = ranges.max(axis=1)
                        ohlcv[feat] = true_range.rolling(window=window).mean()
            
            # Microstructure features
            if 'microstructure' in config['features']:
                logger.debug(f"Processing microstructure features: {config['features']['microstructure']}")
                for feat in config['features']['microstructure']:
                    if feat == 'spread':
                        ohlcv[feat] = market_data['ask_price'] - market_data['bid_price']
                    elif feat == 'imbalance':
                        ohlcv[feat] = (market_data['ask_size'] - market_data['bid_size']) / \
                                    (market_data['ask_size'] + market_data['bid_size'])
            
            # Time features
            if 'time' in config['features']:
                logger.debug(f"Processing time features: {config['features']['time']}")
                for feat in config['features']['time']:
                    if feat == 'hour':
                        ohlcv[feat] = ohlcv.index.hour
                    elif feat == 'minute':
                        ohlcv[feat] = ohlcv.index.minute
                    elif feat == 'time_to_close':
                        # Calculate minutes until 16:00 close
                        close_time = pd.Timestamp('16:00').time()
                        current_time = ohlcv.index.time
                        ohlcv[feat] = [(close_time.hour - t.hour) * 60 + (close_time.minute - t.minute) 
                                      for t in current_time]
        
        # Create features DataFrame
        feature_cols = []
        if 'features' in config:
            feature_cols = (config['features'].get('technical', []) + 
                          config['features'].get('microstructure', []) + 
                          config['features'].get('time', []))
        
        logger.debug(f"Feature columns: {feature_cols}")
        features = ohlcv[feature_cols].copy() if feature_cols else pd.DataFrame(index=ohlcv.index)
        logger.debug(f"Features DataFrame shape: {features.shape}")
        
        # Create execution parameters
        exec_params = {
            'transaction_cost': config.get('transaction_cost', 2.5),
            'slippage': config.get('slippage', 0.01)
        }
        logger.debug(f"Execution parameters: {exec_params}")
        
        # Create risk configuration
        risk_cfg = RiskConfig()
        risk_cfg.max_position_size = config.get('max_position_size', 5)
        logger.debug(f"Risk config: max_position_size={risk_cfg.max_position_size}")
        
        # Initialize parent class
        logger.debug("Calling parent IntradayRLEnv.__init__")
        super().__init__(
            ohlcv=ohlcv,
            features=features,
            cash=config.get('initial_cash', 100000.0),
            exec_params=exec_params,
            risk_cfg=risk_cfg,
            point_value=5.0,
            config=config  # Pass the config to parent class
        )
        
        # Store test-compatible attributes
        logger.debug("Storing test-compatible attributes")
        self.config = config
        self.market_data = market_data
        
        # Initialize test-compatible state variables
        self.current_step = 0
        self.position = 0
        self.trade_history = []
        self.current_price = 0.0
        self.entry_price = None
        self.entry_time = None
        
        logger.debug("IntradayRLEnvironment initialization completed")

    def _detect_data_source_from_market_data(self, market_data: pd.DataFrame) -> str:
        """Detect data source from market data characteristics."""
        # Check for Polygon-specific columns
        polygon_indicators = ['vwap', 'transactions', 'bid_exchange', 'ask_exchange']
        databento_indicators = ['bid', 'ask']

        polygon_score = sum(1 for col in polygon_indicators if col in market_data.columns)
        databento_score = sum(1 for col in databento_indicators if col in market_data.columns)

        if polygon_score > databento_score:
            return 'polygon'
        elif databento_score > polygon_score:
            return 'databento'
        else:
            return 'unknown'

    def reset(self):
        """Reset environment with test-compatible interface."""
        # Call parent reset
        obs, info = super().reset()
        
        # Update test-compatible state
        self.current_step = 0
        self.position = self.pos
        self.trade_history = []
        
        return obs
    
    def step(self, action):
        """Step environment with test-compatible interface."""
        # Map test actions (0=hold, 1=buy, 2=sell) to internal actions (-1, 0, 1)
        internal_action = {0: 0, 1: 1, 2: -1}[action]
        
        # Call parent step
        obs, reward, done, truncated, info = super().step(internal_action + 1)  # Convert to 0,1,2
        
        # Update test-compatible state
        self.current_step += 1
        self.position = self.pos
        
        # Update trade history if position changed
        if self.pos != 0 and len(self.trades) > len(self.trade_history):
            self.trade_history.append({
                'step': self.current_step,
                'position': self.pos,
                'price': self.entry_price,
                'timestamp': self.df.index[self.i-1] if self.i > 0 else self.df.index[0]
            })
        
        # Update current price
        if self.i < len(self.df):
            self.current_price = float(self.df["close"].iloc[self.i])
        
        # Create test-compatible info dictionary
        info = {
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.equity,
            'unrealized_pnl': self.equity - self.cash
        }
        
        return obs, reward, done, info
    
    def get_cash(self):
        """Get current cash for test compatibility."""
        return self.cash
    
    def get_portfolio_value(self):
        """Get current portfolio value for test compatibility."""
        return self.equity
    
    def check_triple_barrier(self):
        """Check triple barrier conditions for test compatibility."""
        # This would be implemented based on the current price and barrier prices
        return False  # Placeholder


# Keep the original alias for backward compatibility
# IntradayRLEnv = IntradayRLEnv  // Remove this circular reference
