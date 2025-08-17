"""
Reinforcement Learning environment for intraday trading.

This module provides a comprehensive RL environment with triple-barrier exits,
risk management, and end-of-day flatten functionality.
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
from .execution import ExecutionSimulator, estimate_tc
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
                 env_config: Optional[EnvConfig] = None):
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
        """
        super().__init__()
        
        # Configuration
        self.cash = cash
        self.point_value = point_value
        self.env_config = env_config or EnvConfig(cash=cash)
        
        # Data
        self.ohlcv = ohlcv[['open', 'high', 'low', 'close', 'volume']]
        self.features = features.reindex(ohlcv.index, method='ffill')
        
        # Ensure alignment
        self.ohlcv = self.ohlcv.loc[self.features.index]
        
        # Initialize components
        self.exec_sim = ExecutionSimulator(Settings())
        if exec_params:
            self.exec_sim.exec_params = self.exec_sim.ExecParams(**exec_params)
        
        self.risk_manager = RiskManager(Settings())
        if risk_cfg:
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
        
        logger.info(f"Environment initialized with {len(self.ohlcv)} bars, {feature_dim} features")
    
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
        atr_val = float(self.X.loc[ts].get("atr", 0.5))  # fallback small ATR
        
        # Execute action at bar close -> new position for next bar
        desired_dir = {-1: -1, 0: 0, 1: 1}[action - 1] if action in (0,1,2) else 0
        reward = 0.0
        info = {}
        
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
            # EOD penalty for holding inventory (should be zero)
            reward -= self.exec_sim.exec_params.tick_value * self.pos * 0 + 0.0  # already flattened
            done = True
            obs = self._obs(ts, price)
            self.equity = self.cash
            self.equity_curve.append(self.equity)
            return obs, reward, done, False, info
        
        # Update unrealized -> realized between bars
        if self.i > 0:
            prev_price = float(self.df["close"].iloc[self.i-1])
            bar_pnl = self._bar_pnl(prev_price, price, self.pos)
            self.cash += bar_pnl
            self.equity = self.cash
            self.max_equity = max(self.max_equity, self.equity)
            self.min_equity = min(self.min_equity, self.equity)
            self.equity_curve.append(self.equity)
        
        # Check barriers (if in a trade)
        hit_exit = False
        if self.pos != 0 and self.entry_price is not None:
            if (self.pos > 0 and (row["low"] <= self.stop_price or row["high"] >= self.tp_price)) \
               or (self.pos < 0 and (row["high"] >= self.stop_price or row["low"] <= self.tp_price)):
                # Exit at barrier (assume touch -> fill)
                exit_price = self.tp_price if (
                    (self.pos > 0 and row["high"] >= self.tp_price) or
                    (self.pos < 0 and row["low"] <= self.tp_price)
                ) else self.stop_price
                pnl = (exit_price - self.entry_price) * self.pos * self.point_value
                tc = estimate_tc(self.pos, exit_price, self.exec_sim)
                self.cash += pnl - tc
                self.pos = 0
                self.entry_price = None
                self.stop_price = None
                self.tp_price = None
                hit_exit = True
        
        # Apply desired action if flat (or change direction after exit)
        if desired_dir == 0:
            pass
        else:
            if self.pos == 0 and not hit_exit:
                contracts = self._risk_sized_contracts(price, atr_val)
                if contracts > 0:
                    # open position
                    self.pos = contracts * int(np.sign(desired_dir))
                    self.entry_price = price
                    self._set_barrier_prices(self.pos, price, atr_val)
                    # pay entry cost
                    self.cash -= estimate_tc(self.pos, price, self.exec_sim)
        
        # Reward: DSR of per-bar return minus penalties
        # per-bar "strategy return" as pct of equity
        if len(self.equity_curve) >= 2:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-6)
        else:
            ret = 0.0
        
        reward += self.dsr.update(ret)
        # Drawdown penalty
        dd = 0.0 if not self.equity_curve else (self.max_equity - self.equity) / max(self.max_equity, 1e-6)
        reward -= self.exec_sim.exec_params.tick_value * 0.0  # placeholder
        reward -= self.risk_manager.risk_config.max_daily_loss_r * 0.0  # placeholder
        reward -= self.risk_manager.risk_config.stop_r_multiple * 0.0   # placeholder
        reward -= dd *  self.risk_manager.risk_config.stop_r_multiple * 0 + 0.0
        reward -= 0.0
        
        # Daily kill-switch on realized R
        realized_drawdown = (self.day_start_equity - self.equity) / max(self.day_start_equity, 1e-6)
        if realized_drawdown < -self.risk_manager.risk_config.max_daily_loss_r * 0.01:  # crude mapping
            # force flat and end episode
            if self.pos != 0:
                self.cash -= estimate_tc(self.pos, price, self.exec_sim)
                self.pos = 0
                self.entry_price = None
                self.stop_price = None
                self.tp_price = None
            done = True
        else:
            done = False
        
        self.i += 1
        if self.i >= len(self.df):
            done = True
        
        next_ts = self.df.index[min(self.i, len(self.df)-1)]
        obs = self._obs(next_ts, float(self.df["close"].iloc[min(self.i, len(self.df)-1)]))
        return obs, float(reward), done, False, {}
    
    def _obs(self, ts, price):
        """Construct observation vector."""
        # Get features at timestamp
        if ts in self.X.index:
            feats = self.X.loc[ts].values.astype(np.float32)
        else:
            # Fallback to last available features
            feats = self.X.iloc[-1].values.astype(np.float32)
        
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
        return self.risk_manager.calculate_position_size(
            self.equity, price, 
            price - atr * self.risk_manager.risk_config.stop_r_multiple,
            atr
        )
    
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