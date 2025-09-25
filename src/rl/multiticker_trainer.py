"""
Minimal multi-ticker RL trainer that reuses the single-asset IntradayRLEnv by
creating one environment per ticker and training a shared policy across them via
Stable-Baselines3 VecEnv.

Scope:
- Assumes input `data` and `features` are DataFrames containing multiple tickers,
  identified by a `ticker` column. Index must be a DatetimeIndex.
- Builds one IntradayRLEnv per ticker with aligned OHLCV and features.
- Trains a single RecurrentPPO policy across parallel envs.
- Provides a simple backtest runner that evaluates on the provided test split.

This is a pragmatic stepping stone toward a full portfolio-aware multi-ticker
environment while enabling multi-ticker training/backtesting end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..sim.env_intraday_rl import IntradayRLEnv, EnvConfig
from ..sim.portfolio_env import PortfolioRLEnv, PortfolioEnvConfig
from ..sim.execution import ExecParams
from ..sim.risk import RiskConfig
from .train import evaluate_model  # reuse existing evaluator


logger = get_logger(__name__)

def _set_global_seeds(seed: int) -> None:
    try:
        import numpy as _np
        import random as _rd
        import torch as _torch
        _rd.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def linear_schedule(start: float, end: float):
    """Return a callable for SB3 that maps progress_remaining (1->0) to value."""
    start = float(start)
    end = float(end)
    def fn(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)
    return fn


class EvalAndLrCallback(BaseCallback):
    """Periodic evaluation + LR/clip heartbeat + optional LR reduce on plateau.

    - Runs evaluation every eval_freq steps on provided eval_env
    - Saves best model by mean reward
    - Halves LR if no improvement for `patience` evals
    """
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, out_dir: Path,
                 min_lr: float = 1e-5, patience: int = 3, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.best_mean_reward = -float('inf')
        self.no_improve = 0
        self.min_lr = float(min_lr)
        self.patience = int(max(1, patience))
        self.out_dir = Path(out_dir)
        (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step % self.eval_freq != 0:
            return True
        try:
            mean_r, std_r = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                                            deterministic=True, return_episode_rewards=False)
        except Exception:
            return True
        # Heartbeat: lr and clip_range if callable
        try:
            lr = float(self.model.lr_schedule(1.0)) if callable(self.model.lr_schedule) else float(self.model.learning_rate)
        except Exception:
            lr = float(getattr(self.model, 'learning_rate', 0.0))
        try:
            cr = float(self.model.clip_range(1.0)) if callable(self.model.clip_range) else float(self.model.clip_range)
        except Exception:
            cr = float(getattr(self.model, 'clip_range', 0.0))
        self.logger.record("eval/mean_reward", mean_r)
        self.logger.record("train/lr", lr)
        self.logger.record("train/clip_range", cr)
        if self.verbose:
            print(f"[eval] step={step} meanR={mean_r:.3f} lr={lr:.2e} clip={cr:.3f}")

        # Save best
        if mean_r > self.best_mean_reward:
            self.best_mean_reward = mean_r
            try:
                self.model.save(str(self.out_dir / 'checkpoints' / 'best_model'))
                # persist vecnorm if present
                try:
                    env = self.model.get_env()
                    if isinstance(env, VecNormalize):
                        env.save(str(self.out_dir / 'checkpoints' / 'vecnorm.pkl'))
                except Exception:
                    pass
            except Exception:
                pass
            self.no_improve = 0
        else:
            self.no_improve += 1
            # Reduce LR on plateau
            if self.no_improve >= self.patience:
                try:
                    current_lr = float(self.model.lr_schedule(1.0)) if callable(self.model.lr_schedule) else float(self.model.learning_rate)
                except Exception:
                    current_lr = float(getattr(self.model, 'learning_rate', 0.0))
                new_lr = max(self.min_lr, current_lr * 0.5)
                try:
                    # Update optimizer and schedule base
                    for pg in self.model.policy.optimizer.param_groups:
                        pg['lr'] = new_lr
                    # Replace schedule to a flat at new_lr from now on
                    self.model.lr_schedule = linear_schedule(new_lr, new_lr)
                    if self.verbose:
                        print(f"[eval] plateau detected → lr halved to {new_lr:.2e}")
                except Exception:
                    pass
                self.no_improve = 0
        return True

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-aware DatetimeIndex in America/New_York order."""
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.loc[ts.notna()].copy()
        df['timestamp'] = ts
        df = df.sort_values('timestamp').set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=True, errors='coerce', unit='ms')
        if idx.isna().all():
            idx = pd.to_datetime(df.index, utc=True, errors='coerce')
        df = df.loc[idx.notna()].copy()
        df.index = idx[idx.notna()]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.sort_index().tz_convert('America/New_York')


def _build_env_from_frames(
    settings: Settings,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    *,
    point_value: float = 1.0,
) -> IntradayRLEnv:
    """Create an IntradayRLEnv from aligned OHLCV and features frames."""
    # Execution and risk parameters from settings with safe defaults
    exec_params = ExecParams(
        tick_value=float(settings.get("execution", "tick_value", default=0.01)),
        spread_ticks=int(settings.get("execution", "spread_ticks", default=1)),
        impact_bps=float(settings.get("execution", "impact_bps", default=0.5)),
        commission_per_contract=float(settings.get("execution", "commission_per_contract", default=0.0035)),
    )
    risk_cfg = RiskConfig(
        risk_per_trade_frac=float(settings.get("risk", "risk_per_trade_frac", default=0.02)),
        stop_r_multiple=float(settings.get("risk", "stop_r_multiple", default=1.0)),
        tp_r_multiple=float(settings.get("risk", "tp_r_multiple", default=1.5)),
        max_daily_loss_r=float(settings.get("risk", "max_daily_loss_r", default=3.0)),
    )
    reward_type = str(settings.get("env", "reward", "kind", default="dsr"))
    reward_scaling = float(settings.get("env", "reward_scaling", default=0.1))
    max_steps = int(settings.get("env", "max_steps", default=390))
    env_cfg = EnvConfig(
        cash=100_000.0,
        max_steps=max_steps,
        reward_type=reward_type,
        reward_scaling=reward_scaling,
    )
    # Align indices and columns
    o = _ensure_dt_index(ohlcv)
    X = _ensure_dt_index(features)
    X = X.reindex(o.index).ffill().bfill()
    o = o.dropna(subset=[c for c in ["open","high","low","close"] if c in o.columns])
    return IntradayRLEnv(
        ohlcv=o[[c for c in ["open","high","low","close","volume"] if c in o.columns]].copy(),
        features=X.copy(),
        cash=100_000.0,
        exec_params=exec_params,
        risk_cfg=risk_cfg,
        point_value=float(settings.get("execution", "point_value", default=point_value)),
        env_config=env_cfg,
        config=settings.to_dict() if hasattr(settings, 'to_dict') else settings._cfg,
    )


def _extract_tickers(df: pd.DataFrame) -> List[str]:
    if 'ticker' in df.columns:
        return sorted(list(pd.unique(df['ticker'])))
    # Also support MultiIndex with level named 'ticker'
    if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
        return sorted(list(df.index.get_level_values('ticker').unique()))
    raise ValueError("Multi-ticker data requires a 'ticker' column or MultiIndex level named 'ticker'.")


def _slice_by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if 'ticker' in df.columns:
        return df[df['ticker'] == ticker].drop(columns=['ticker'], errors='ignore')
    if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
        return df.xs(ticker, level='ticker')
    return df


@dataclass
class _HP:
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.7
    ent_coef: float = 0.015
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    target_kl: float = 0.075
    device: str = "auto"
    seed: int = 42
    total_steps: int = 100_000


def _read_hparams(cfg: Dict[str, Any]) -> _HP:
    # Map from either 'ppo', 'train' or 'training' blocks; prefer 'ppo'.
    def G(*keys, default=None):
        cur = cfg
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur
    blk = cfg.get('ppo', cfg.get('train', cfg.get('training', {}))) or {}
    # Prefer rl.ppo if present
    try:
        rl_blk = cfg.get('rl', {}).get('ppo', {}) if isinstance(cfg, dict) else {}
        if rl_blk:
            blk = rl_blk
    except Exception:
        pass
    def get_k(k, dv):
        return blk.get(k, dv)
    hp = _HP(
        learning_rate=float(get_k('learning_rate', 1e-4)),
        n_steps=int(get_k('n_steps', 2048)),
        batch_size=int(get_k('batch_size', 4096)),
        gamma=float(get_k('gamma', 0.99)),
        gae_lambda=float(get_k('gae_lambda', 0.95)),
        clip_range=float(get_k('clip_range', 0.2)),
        vf_coef=float(get_k('vf_coef', 0.7)),
        ent_coef=float(get_k('ent_coef', 0.015)),
        max_grad_norm=float(get_k('max_grad_norm', 0.5)),
        n_epochs=int(get_k('n_epochs', 10)),
        target_kl=float(get_k('target_kl', 0.075)),
        device=str(get_k('device', 'auto')),
        seed=int(get_k('seed', 42)),
        total_steps=int(get_k('total_timesteps', get_k('total_steps', 100_000))),
    )
    return hp


class MultiTickerRLTrainer:
    """
    Train and backtest a shared PPO-LSTM policy across multiple tickers by
    running one IntradayRLEnv instance per ticker in parallel.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        # Gracefully construct Settings; allow passing a config dict only
        try:
            # If a config file path is embedded in meta, use that
            cfg_file = (config.get('__meta__') or {}).get('config_file')
            self.settings = Settings.from_yaml(cfg_file) if cfg_file else Settings.from_yaml()
        except Exception:
            # Fallback: create with overrides from cfg['paths'] if present
            self.settings = Settings.from_paths(paths=config.get('paths', {}))
        self.hp = _read_hparams(config)
        self.model: Optional[RecurrentPPO] = None
        self._train_tickers: Optional[List[str]] = None

    def _make_envs(self, data: pd.DataFrame, features: pd.DataFrame, tickers: List[str]) -> DummyVecEnv:
        envs: List[Any] = []
        for t in tickers:
            df_t = _slice_by_ticker(data, t)
            X_t = _slice_by_ticker(features, t)
            envs.append(lambda df=df_t, X=X_t: _build_env_from_frames(self.settings, df, X))
        return DummyVecEnv(envs)

    def train(
        self,
        *,
        data: pd.DataFrame,
        features: pd.DataFrame,
        output_dir: Path,
    ) -> RecurrentPPO:
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tickers = _extract_tickers(data)
        # Use full ticker universe to keep observation shape stable across train/test

        logger.info("Training across tickers: %s", ", ".join(tickers))
        # Persist train tickers for backtest parity
        self._train_tickers = list(tickers)
        # Prefer portfolio env when multiple tickers
        force_port = False
        try:
            force_port = bool(self.cfg.get('env', {}).get('portfolio', {}).get('force', False))
        except Exception:
            force_port = False
        # Persist training feature columns per ticker for test-time alignment
        self._train_feat_cols: Dict[str, List[str]] = {}

        if len(tickers) > 1 or force_port:
            o_map = {t: _slice_by_ticker(data, t) for t in tickers}
            X_map = {t: _slice_by_ticker(features, t) for t in tickers}
            # Optional: add ticker identity one-hot columns for embedding-like signal
            try:
                id_cfg = (self.cfg.get('features', {}).get('ticker_identity', {}) if isinstance(self.cfg, dict) else {}) or {}
                if bool(id_cfg.get('enabled', False)):
                    id_cols = [f'id_{t}' for t in tickers]
                    for t in tickers:
                        X_t = X_map.get(t)
                        if X_t is None:
                            continue
                        for col in id_cols:
                            X_t[col] = 1.0 if col == f'id_{t}' else 0.0
                        X_map[t] = X_t
                        logger.debug("Added ticker identity columns to features for %s", t)
            except Exception:
                pass
            # record train feature columns per ticker
            for t in tickers:
                try:
                    self._train_feat_cols[t] = list(X_map[t].columns)
                except Exception:
                    self._train_feat_cols[t] = []
            def make_port_env():
                # Pull portfolio env settings from config so training honors activity/cadence shaping
                port_cfg = (self.cfg.get('env', {}).get('portfolio', {}) if isinstance(self.cfg, dict) else {}) or {}
                env_cfg = PortfolioEnvConfig(
                    cash=float(port_cfg.get('cash', 100_000.0)),
                    reward_scaling=float(port_cfg.get('reward_scaling', 1.0)),
                    enforce_intraday=bool(port_cfg.get('enforce_intraday', True)),
                    min_hold_minutes=int(port_cfg.get('min_hold_minutes', 5)),
                    max_hold_minutes=int(port_cfg.get('max_hold_minutes', 240)),
                    max_entries_per_day=int(port_cfg.get('max_entries_per_day', 3)),
                    position_holding_penalty=float(port_cfg.get('position_holding_penalty', 0.0)),
                    fixed_tickers=tickers,
                    allowed_trade_tickers=None,
                )
                return PortfolioRLEnv(
                    ohlcv_map=o_map,
                    features_map=X_map,
                    settings=self.settings,
                    env_cfg=env_cfg,
                )
            # Portfolio env is stateful across tickers; keep single process for correctness
            vec_env = DummyVecEnv([make_port_env])
        else:
            # Parallelize single-ticker via SubprocVecEnv when multiple envs requested
            n_envs = int(self.cfg.get('rl', {}).get('n_envs', 1) if isinstance(self.cfg, dict) else 1)
            n_envs = max(1, n_envs)
            fns = []
            def make_single():
                return _build_env_from_frames(self.settings, data, features)
            for _ in range(n_envs):
                fns.append(make_single)
            vec_env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv([make_single])

        policy_kwargs = dict(
            net_arch={"pi": [256, 256], "vf": [256, 256]},
            activation_fn=__import__("torch", fromlist=["nn"]).nn.ReLU,
            ortho_init=True,
            normalize_images=False,
        )
        # Optional normalization config
        # Normalization settings (backward compatible)
        norm_cfg = (self.cfg.get('normalize', {}) if isinstance(self.cfg, dict) else {}) or {}
        rl_vn = (self.cfg.get('rl', {}).get('vecnormalize', {}) if isinstance(self.cfg, dict) else {}) or {}
        norm_obs = bool(rl_vn.get('norm_obs', norm_cfg.get('obs', False)))
        norm_rew = bool(rl_vn.get('norm_reward', norm_cfg.get('reward', False)))
        clip_obs = float(rl_vn.get('clip_obs', 10.0))
        clip_reward = float(rl_vn.get('clip_reward', 10.0))

        # Seed
        seed = int(self.cfg.get('rl', {}).get('seed', getattr(self.hp, 'seed', 42))) if isinstance(self.cfg, dict) else getattr(self.hp, 'seed', 42)
        _set_global_seeds(seed)

        # Schedules
        lr_sched = linear_schedule(3e-4, 1e-5) if str(self.cfg.get('rl', {}).get('ppo', {}).get('lr_schedule', '')).startswith('linear') else self.hp.learning_rate
        clip_sched = linear_schedule(0.2, 0.1) if str(self.cfg.get('rl', {}).get('ppo', {}).get('clip_schedule', '')).startswith('linear') else self.hp.clip_range

        self.model = RecurrentPPO(
            'MlpLstmPolicy',
            vec_env,
            learning_rate=lr_sched,
            n_steps=max(1, int(self.hp.n_steps)),
            batch_size=max(64, int(self.hp.batch_size)),
            gamma=self.hp.gamma,
            gae_lambda=self.hp.gae_lambda,
            clip_range=clip_sched,
            vf_coef=self.hp.vf_coef,
            ent_coef=float(self.cfg.get('rl', {}).get('ppo', {}).get('ent_coef', self.hp.ent_coef)) if isinstance(self.cfg, dict) else self.hp.ent_coef,
            max_grad_norm=self.hp.max_grad_norm,
            n_epochs=self.hp.n_epochs,
            target_kl=float(self.cfg.get('rl', {}).get('ppo', {}).get('target_kl', self.hp.target_kl)) if isinstance(self.cfg, dict) else self.hp.target_kl,
            policy_kwargs=policy_kwargs,
            device=self.hp.device,
            verbose=1,
            seed=seed,
            tensorboard_log=str((output_dir / 'logs' / 'tensorboard').resolve()),
        )
        # Wrap with VecNormalize if requested
        if norm_obs or norm_rew:
            vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_rew, clip_obs=clip_obs, clip_reward=clip_reward)
            self.model.set_env(vec_env)
        # Build a small eval env on a held-out tail slice if possible (single-ticker path)
        eval_cb = None
        try:
            if len(tickers) == 1:
                idx = data.index
                if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1000:
                    cutoff = int(len(idx) * 0.9)
                    d_eval = data.iloc[cutoff:]
                    X_eval = features.iloc[cutoff:]
                else:
                    d_eval = data.tail(1000)
                    X_eval = features.tail(1000)
                def _make_eval():
                    return _build_env_from_frames(self.settings, d_eval, X_eval)
                eval_env = DummyVecEnv([_make_eval])
                eval_cb = EvalAndLrCallback(eval_env=eval_env,
                                            eval_freq=int(self.cfg.get('rl', {}).get('eval', {}).get('eval_freq', 100000)),
                                            n_eval_episodes=int(self.cfg.get('rl', {}).get('eval', {}).get('n_eval_episodes', 5)),
                                            out_dir=output_dir,
                                            patience=3,
                                            verbose=1)
        except Exception:
            eval_cb = None

        total_steps = int(self.hp.total_steps)
        self.model.learn(total_timesteps=total_steps, progress_bar=True, callback=eval_cb)

        # Save artifacts
        (output_dir).mkdir(parents=True, exist_ok=True)
        # Persist artifacts
        ckpt_dir = output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_path = ckpt_dir / 'model_last'
        # Robust save: handle environments where gym lacks __version__ attribute (SB3 system info)
        try:
            self.model.save(str(model_path))
        except AttributeError as e:
            # SB3 get_system_info may access gym.__version__; patch it if missing and retry once
            try:
                import gym as _gym  # type: ignore
                if not hasattr(_gym, "__version__"):
                    setattr(_gym, "__version__", "0.0.0")
                    logger.warning("Patched gym.__version__='0.0.0' for SB3 save compatibility")
                self.model.save(str(model_path))
            except Exception:
                raise e
        try:
            if isinstance(self.model.get_env(), VecNormalize):
                self.model.get_env().save(str(ckpt_dir / 'vecnorm.pkl'))
        except Exception:
            pass
        # Write minimal training summary
        try:
            import json as _json, time as _time
            summ = {
                'seed': seed,
                'total_timesteps': total_steps,
                'saved': str(model_path),
                'best_model': str((ckpt_dir / 'best_model').with_suffix('.zip')),
                'vecnorm': str(ckpt_dir / 'vecnorm.pkl'),
                'timestamp': _time.time(),
            }
            met_dir = output_dir / 'metrics'
            met_dir.mkdir(parents=True, exist_ok=True)
            (met_dir / 'training_summary.json').write_text(_json.dumps(summ, indent=2))
        except Exception:
            pass
        logger.info("Saved multi-ticker model to %s", model_path)
        return self.model

    def backtest(
        self,
        *,
        model: Optional[RecurrentPPO],
        data: pd.DataFrame,
        features: pd.DataFrame,
        output_dir: Path,
        eval_episodes: int = 1,
        allowed_tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if model is None:
            if self.model is None:
                raise ValueError("No model provided and trainer has no trained model")
            model = self.model
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use training tickers for backtest to ensure obs shape parity
        tickers = list(self._train_tickers) if getattr(self, '_train_tickers', None) else _extract_tickers(data)
        # Use the same portfolio env for evaluation when multiple tickers
        force_port = False
        try:
            force_port = bool(self.cfg.get('env', {}).get('portfolio', {}).get('force', False))
        except Exception:
            force_port = False
        if len(tickers) > 1 or force_port:
            # Use full training ticker universe and enforce fixed shape.
            train_tickers = list(self._train_tickers) if getattr(self, '_train_tickers', None) else list(_extract_tickers(data))
            # Slice frames per ticker
            raw_o_map = {t: _slice_by_ticker(data, t) for t in train_tickers}
            raw_X_map = {t: _slice_by_ticker(features, t) for t in train_tickers}
            # Determine common index from tickers that have data; prefer allowed tickers if provided
            prefer = set(allowed_tickers) if allowed_tickers is not None else set(train_tickers)
            idx_sources = [raw_o_map[t].index for t in train_tickers if (t in prefer and not raw_o_map[t].empty)]
            if not idx_sources:
                # fallback to any training ticker with data
                idx_sources = [df.index for df in raw_o_map.values() if not getattr(df, 'empty', True)]
            if not idx_sources:
                raise ValueError("No OHLCV data for any training ticker in test window")
            common_idx = idx_sources[0]
            for idx in idx_sources[1:]:
                common_idx = common_idx.intersection(idx)
            common_idx = common_idx.sort_values()
            if len(common_idx) < 2:
                # pick the longest single index
                lens = [(t, len(raw_o_map[t].index)) for t in train_tickers]
                lens.sort(key=lambda x: x[1], reverse=True)
                common_idx = raw_o_map[lens[0][0]].index

            # Build OHLCV map with zero-padded placeholders for missing tickers
            o_map: Dict[str, pd.DataFrame] = {}
            X_map: Dict[str, pd.DataFrame] = {}
            for t in train_tickers:
                df_t = raw_o_map.get(t)
                if df_t is None or df_t.empty:
                    # placeholder OHLCV
                    o_pad = pd.DataFrame(index=common_idx, data={
                        'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0, 'vwap': 0.0
                    })
                    o_map[t] = o_pad
                else:
                    o = df_t.reindex(common_idx).ffill().bfill()
                    keep = [c for c in ["open","high","low","close","volume","vwap"] if c in o.columns]
                    if 'close' not in keep:
                        keep = keep + ['close']
                        o['close'] = 0.0
                    o_map[t] = o[keep]
                # Features: strict to training columns if known, else use raw
                cols = list((getattr(self, '_train_feat_cols', {}) or {}).get(t, []))
                if not cols:
                    try:
                        cols = list(raw_X_map.get(t, pd.DataFrame()).columns)
                    except Exception:
                        cols = ['close']
                X_aligned = pd.DataFrame(0.0, index=common_idx, columns=cols, dtype=float)
                rawX = raw_X_map.get(t)
                if rawX is not None and not rawX.empty:
                    rawX = rawX.reindex(common_idx)
                    common_cols = [c for c in cols if c in rawX.columns]
                    if common_cols:
                        try:
                            X_aligned.loc[:, common_cols] = rawX[common_cols].astype(float).values
                        except Exception:
                            pass
                X_map[t] = X_aligned
            def make_env():
                # Configure portfolio env with intraday constraints and allowed tickers for backtest
                port_cfg = (self.cfg.get('env', {}).get('portfolio', {}) if isinstance(self.cfg, dict) else {}) or {}
                env_cfg = PortfolioEnvConfig(
                    cash=float(port_cfg.get('cash', 100_000.0)),
                    reward_scaling=float(port_cfg.get('reward_scaling', 1.0)),
                    enforce_intraday=bool(port_cfg.get('enforce_intraday', True)),
                    min_hold_minutes=int(port_cfg.get('min_hold_minutes', 5)),
                    max_hold_minutes=int(port_cfg.get('max_hold_minutes', 240)),
                    max_entries_per_day=int(port_cfg.get('max_entries_per_day', 3)),
                    position_holding_penalty=float(port_cfg.get('position_holding_penalty', 0.0)),
                    allowed_trade_tickers=[t for t in train_tickers if (allowed_tickers is None or t in allowed_tickers)],
                    fixed_tickers=train_tickers,
                )
                return PortfolioRLEnv(ohlcv_map=o_map, features_map=X_map, settings=self.settings, env_cfg=env_cfg)
            vec_env = DummyVecEnv([make_env])
            metrics = evaluate_model(model, vec_env, num_episodes=max(1, eval_episodes))
            # Persist equity/history and recompute metrics from equity curve
            try:
                env0 = vec_env.envs[0]
                # Save history CSV
                hist = env0.get_history_df()
                if not hist.empty:
                    out_csv = output_dir / 'portfolio_history.csv'
                    hist.to_csv(out_csv)
                # Equity curve
                import pandas as _pd
                if getattr(env0, '_last_equity_curve', None) is not None:
                    eq = env0._last_equity_curve if isinstance(env0._last_equity_curve, _pd.Series) else _pd.Series(env0._last_equity_curve)
                else:
                    eq = env0.get_equity_curve()
                # Compute returns and portfolio stats
                ret = eq.pct_change().dropna()
                stats = {}
                if len(ret) > 1 and float(eq.iloc[0]) != 0.0:
                    stats = {
                        'total_return': float((eq.iloc[-1] - eq.iloc[0]) / abs(eq.iloc[0])),
                        'annual_return': float((1 + ret.mean()) ** 252 - 1),
                        'annual_volatility': float(ret.std() * np.sqrt(252)),
                        'sharpe_ratio': float((ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252)),
                        'max_drawdown': float(((eq / eq.cummax()) - 1).min()),
                        'sortino_ratio': float((ret.mean() / (ret[ret < 0].std() + 1e-12)) * np.sqrt(252)) if len(ret[ret < 0]) > 0 else float('inf'),
                    }
                    if stats['max_drawdown'] != 0:
                        stats['calmar_ratio'] = float((ret.mean() * 252) / abs(stats['max_drawdown']))
                # Merge metrics with stats
                metrics.update(stats)
                # Trades and trade-level performance
                try:
                    trades = env0.get_trades()
                except Exception:
                    trades = []
                import pandas as _pd
                tdf = _pd.DataFrame(trades)
                # Always save trades log (may be empty)
                if tdf.empty:
                    # Create with expected columns for consistency
                    tdf = _pd.DataFrame(columns=[
                        'ticker','direction','entry_time','exit_time','entry_price','exit_price','units','duration_bars','duration_minutes','pnl'
                    ])
                # Enrich with computed fields and run metadata
                try:
                    tdf = tdf.copy()
                    tdf['trade_id'] = _pd.RangeIndex(start=1, stop=len(tdf)+1)
                    # PnL percent relative to notional at entry (per-trade)
                    eps = 1e-9
                    notional = (_pd.to_numeric(tdf.get('entry_price', _pd.Series(dtype=float))) * _pd.to_numeric(tdf.get('units', _pd.Series(dtype=float))).abs()).replace(0, _pd.NA)
                    tdf['return_pct'] = _pd.to_numeric(tdf.get('pnl', _pd.Series(dtype=float))) / (notional.replace(_pd.NA, eps) + eps)
                    # Metadata columns
                    tdf['run_seed'] = seed if 'seed' in locals() else getattr(self.hp, 'seed', 0)
                    try:
                        # Derive window from data index used for backtest
                        tdf['window_start'] = _pd.Timestamp(data.index.min()).isoformat()
                        tdf['window_end'] = _pd.Timestamp(data.index.max()).isoformat()
                    except Exception:
                        pass
                except Exception:
                    pass
                tdf.to_csv(output_dir / 'trades.csv', index=False)
                # Aggregate trade stats (zeros if none)
                total_trades = int(len(tdf))
                long_mask = (tdf['direction'] == 'long') if 'direction' in tdf else _pd.Series([], dtype=bool)
                short_mask = (tdf['direction'] == 'short') if 'direction' in tdf else _pd.Series([], dtype=bool)
                trade_stats = {
                    'total_trades': total_trades,
                    'long_trades': int(long_mask.sum()) if total_trades else 0,
                    'short_trades': int(short_mask.sum()) if total_trades else 0,
                    'avg_duration_minutes': float(_pd.to_numeric(tdf.get('duration_minutes', _pd.Series(dtype=float))).mean()) if total_trades else 0.0,
                    'avg_duration_minutes_long': float(_pd.to_numeric(tdf.loc[long_mask, 'duration_minutes']).mean()) if total_trades and long_mask.any() else 0.0,
                    'avg_duration_minutes_short': float(_pd.to_numeric(tdf.loc[short_mask, 'duration_minutes']).mean()) if total_trades and short_mask.any() else 0.0,
                    'avg_pnl': float(_pd.to_numeric(tdf.get('pnl', _pd.Series(dtype=float))).mean()) if total_trades else 0.0,
                    'avg_pnl_long': float(_pd.to_numeric(tdf.loc[long_mask, 'pnl']).mean()) if total_trades and long_mask.any() else 0.0,
                    'avg_pnl_short': float(_pd.to_numeric(tdf.loc[short_mask, 'pnl']).mean()) if total_trades and short_mask.any() else 0.0,
                    'win_rate': float((tdf.get('pnl', _pd.Series(dtype=float)) > 0).mean()) if total_trades else 0.0,
                    'profit_factor': float(tdf.loc[tdf.get('pnl', _pd.Series(dtype=float)) > 0, 'pnl'].sum() / (abs(tdf.loc[tdf.get('pnl', _pd.Series(dtype=float)) < 0, 'pnl'].sum()) + 1e-12)) if total_trades else 0.0,
                }
                # Expose under metrics and summary
                metrics.update(trade_stats)
            except Exception as e:
                logger.warning(f"Portfolio metrics export failed: {e}")
            # Build summary for portfolio evaluation
            summary = {
                'tickers': list(o_map.keys()),
                'portfolio_metrics': metrics,
            }

        else:
            # Single-ticker evaluation
            single_env = DummyVecEnv([lambda: _build_env_from_frames(self.settings, data, features)])
            metrics = evaluate_model(model, single_env, num_episodes=max(1, eval_episodes))
            summary = {
                'tickers': tickers,
                'per_ticker_metrics': {tickers[0]: metrics},
            }
        with (output_dir / 'summary.json').open('w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        return summary

    # Optional hooks used by the pipeline script
    def generate_backtest_plots(self, results_dir: Path) -> None:  # pragma: no cover
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception:
            logger.info("matplotlib not available; skipping plots.")
            return
        # Placeholder: real portfolio/equity plots can be added later
        pass

    def get_backtest_summary(self) -> Dict[str, Any]:
        # Expose a last-known summary if desired (not persisted across processes)
        return {}
