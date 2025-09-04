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
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Discrete

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import DifferentialSharpe
from .execution import ExecutionEngine, ExecutionSimulator, estimate_tc, ExecParams
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
        logger.debug("Initializing ExecutionSimulator with empty Settings()")
        self.exec_sim = ExecutionSimulator(Settings.from_paths('configs/settings.yaml'))
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
        
        # Seek first bar within RTH
        while self.i < len(self.df) and not self._tod(self.df.index[self.i]):
            self.i += 1
        
        ts = self.df.index[self.i]
        obs = self._obs(ts, float(self.df["close"].iloc[self.i]))
        
        return obs, {}
    
    def step(self, action: int):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (-1, 0, 1)
            
        Returns:
            Observation, reward, done, truncated, info
        """
        # Current bar
        ts = self.df.index[self.i]
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
        desired_dir = {-1: -1, 0: 0, 1: 1}[action - 1] if action in (0, 1, 2) else 0
        reward = 0.0
        info: Dict[str, Any] = {}

        # Flatten at EOD regardless of action
        if self._eod(ts):
            if self.pos != 0:
                # charge closing cost
                tc = estimate_tc(self.pos, price, self.exec_sim)
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
        if desired_dir != 0 and self.pos == 0 and not hit_exit and not in_no_trade:
            contracts = self._risk_sized_contracts(price, atr_val)
            if contracts > 0:
                # open position
                self.pos = contracts * int(np.sign(desired_dir))
                self.entry_price = price
                self._set_barrier_prices(self.pos, price, atr_val)
                # pay entry cost
                self.cash -= estimate_tc(self.pos, price, self.exec_sim)
                try:
                    self.trades.append({
                        'ts': ts,
                        'pos': int(self.pos),
                        'price': float(price)
                    })
                except Exception:
                    pass

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
                widen_factor = 1.0 + (0.5 if in_widen else 0.0)
                reward -= mu_pen * widen_factor * (max(0.0, 1.5 - rvol) + spread)
            except Exception:
                pass
            reward -= float(drawdown_penalty) + float(risk_penalty)
        else:  # Default to pnl with penalties
            reward = pnl - float(drawdown_penalty) - float(risk_penalty)

        reward *= self.env_config.reward_scaling
        reward = np.clip(reward, -1, 1)

        # Daily kill-switch: terminate when realized drawdown exceeds threshold
        done = False
        if self.realized_drawdown > max_daily_pct:
            # Force flat and charge closing costs
            if self.pos != 0:
                self.cash -= estimate_tc(self.pos, price, self.exec_sim)
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

        # Diagnostic info
        info = {
            "pnl": float(pnl),
            "drawdown_penalty": float(drawdown_penalty),
            "risk_penalty": float(risk_penalty),
            "realized_drawdown": float(self.realized_drawdown),
        }
        info["equity"] = float(self.equity)
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
        """Calculate position size based on risk management."""
        contracts = self.risk_manager.calculate_position_size(
            self.equity, price,
            price - atr * self.risk_manager.risk_config.stop_r_multiple,
            atr
        )

        # Handle NaN values - comprehensive check
        if np.isnan(contracts) or np.isinf(contracts) or contracts is None:
            contracts = 0.0

        # Ensure contracts is a valid number before int conversion
        try:
            contracts_int = int(max(0, contracts))
        except (ValueError, OverflowError):
            logger.warning(f"Invalid contracts value: {contracts}, defaulting to 0")
            contracts_int = 0

        return contracts_int
    
    def _set_barrier_prices(self, pos, price, atr):
        """Set stop loss and take profit prices."""
        self.stop_price, self.tp_price = self.risk_manager.calculate_stop_prices(
            price, pos, atr
        )
    
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
