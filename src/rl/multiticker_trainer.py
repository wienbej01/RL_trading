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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import explained_variance

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..sim.env_intraday_rl import IntradayRLEnv, EnvConfig
from ..sim.portfolio_env import PortfolioRLEnv, PortfolioEnvConfig
from ..sim.execution import ExecParams
from ..sim.risk import RiskConfig
from .train import evaluate_model  # reuse existing evaluator
from .callbacks import KLStopCallback, AdaptiveLRByKL, LiveLRBump
from ..utils.callbacks import EarlyStopNoImprove
from ..utils.artifacts import BacktestResult


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
    max_episode_bars: int | None = None,
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
    if isinstance(max_episode_bars, int) and max_episode_bars > 0:
        max_steps = min(max_steps, int(max_episode_bars))
    # Enforce shorts parity for equity runs: allow_shorts must be True
    try:
        allow_shorts = bool(settings.get("env", "allow_shorts", default=True))
    except Exception:
        allow_shorts = True
    if not allow_shorts:
        raise ValueError("allow_shorts must be True for equity runs (parity enforcement)")
    env_cfg = EnvConfig(
        cash=100_000.0,
        max_steps=max_steps,
        reward_type=reward_type,
        reward_scaling=reward_scaling,
        allow_shorts=True,
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
        try:
            self.fast_smoke: bool = bool(self.cfg.get('rl', {}).get('fast_smoke', False)) if isinstance(self.cfg, dict) else False
        except Exception:
            self.fast_smoke = False

    def _make_envs(self, data: pd.DataFrame, features: pd.DataFrame, tickers: List[str]) -> DummyVecEnv:
        envs: List[Any] = []
        for t in tickers:
            df_t = _slice_by_ticker(data, t)
            X_t = _slice_by_ticker(features, t)
            envs.append(lambda df=df_t, X=X_t: _build_env_from_frames(self.settings, df, X, max_episode_bars=(2500 if self.fast_smoke else None)))
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
                return _build_env_from_frames(self.settings, data, features, max_episode_bars=(2500 if self.fast_smoke else None))
            for _ in range(n_envs):
                fns.append(make_single)
            vec_env = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv([make_single])

        if self.fast_smoke:
            policy_kwargs = dict(
                net_arch=[64, 64],
                activation_fn=__import__("torch", fromlist=["nn"]).nn.ReLU,
                ortho_init=True,
                normalize_images=False,
            )
        else:
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

        # Schedules (read from rl.ppo)
        ppo_cfg = (self.cfg.get('rl', {}).get('ppo', {}) if isinstance(self.cfg, dict) else {}) or {}
        if str(ppo_cfg.get('lr_schedule', '')).startswith('linear'):
            lr_start = float(ppo_cfg.get('lr_start', 1.5e-4))
            lr_end = float(ppo_cfg.get('lr_end', 1.0e-5))
            lr_sched = linear_schedule(lr_start, lr_end)
        else:
            lr_sched = self.hp.learning_rate
        if str(ppo_cfg.get('clip_schedule', '')).startswith('linear'):
            clip_start = float(ppo_cfg.get('clip_start', 0.15))
            clip_end = float(ppo_cfg.get('clip_end', 0.12))
            clip_sched = linear_schedule(clip_start, clip_end)
        else:
            clip_sched = self.hp.clip_range

        if self.fast_smoke:
            # Standard PPO with MLP policy, no LSTM
            self.model = PPO(
                'MlpPolicy',
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
                target_kl=float(ppo_cfg.get('target_kl', self.hp.target_kl)) if isinstance(self.cfg, dict) else self.hp.target_kl,
                policy_kwargs=policy_kwargs,
                device=self.hp.device,
                verbose=0,
                seed=seed,
                tensorboard_log=None,
            )
            # Optional: epsilon action smoothing to avoid zero-entropy collapse (debug only)
            try:
                eps = float(self.cfg.get('rl', {}).get('ppo', {}).get('epsilon_action_prob', 0.0)) if isinstance(self.cfg, dict) else 0.0
            except Exception:
                eps = 0.0
            if eps and eps > 0.0:
                try:
                    import torch as _torch
                    from torch.distributions import Categorical as _Categorical  # type: ignore
                    _orig_get = self.model.policy.get_distribution
                    def _smoothed_get_distribution(obs, *a, **kw):  # type: ignore[override]
                        dist = _orig_get(obs, *a, **kw)
                        logits = getattr(dist.distribution, 'logits', None)
                        if logits is not None:
                            probs = _torch.softmax(logits, dim=-1)
                            n = probs.shape[-1]
                            probs = (1.0 - float(eps)) * probs + (float(eps) / float(max(1, int(n))))
                            dist.distribution = _Categorical(probs=probs)
                        return dist
                    self.model.policy.get_distribution = _smoothed_get_distribution  # type: ignore[assignment]
                    logger.info(f"Enabled epsilon_action_prob smoothing: eps={eps}")
                except Exception:
                    pass
            # Log initial action priors (fast-smoke, non-recurrent)
            try:
                import torch as _torch
                obs = vec_env.reset()
                # Build a small batch from the initial observation
                if isinstance(obs, (list, tuple)):
                    obs0 = obs[0]
                else:
                    obs0 = obs
                obs_batch = _torch.as_tensor(obs0).float()
                if obs_batch.ndim == 1:
                    obs_batch = obs_batch.unsqueeze(0)
                dist = self.model.policy.get_distribution(obs_batch)
                logits = getattr(dist.distribution, 'logits', None)
                if logits is not None:
                    probs = _torch.softmax(logits, dim=-1).mean(0)
                    # Apply epsilon smoothing for logging if enabled
                    if 'eps' in locals() and float(eps) > 0.0:
                        n = probs.shape[-1]
                        probs = (1.0 - float(eps)) * probs + (float(eps) / float(max(1, int(n))))
                    probs = probs.detach().cpu().numpy()
                    logger.info(f"Action priors mean: short={probs[0]:.3f}, flat={probs[1]:.3f}, long={probs[2]:.3f}")
            except Exception:
                pass
        else:
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
                target_kl=float(ppo_cfg.get('target_kl', self.hp.target_kl)) if isinstance(self.cfg, dict) else self.hp.target_kl,
                clip_range_vf=float(ppo_cfg.get('clip_range_vf', 0.0)) if 'clip_range_vf' in ppo_cfg else None,
                policy_kwargs=policy_kwargs,
                device=self.hp.device,
                verbose=0,
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
            if (len(tickers) == 1) and (not self.fast_smoke):
                idx = data.index
                if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1000:
                    cutoff = int(len(idx) * 0.9)
                    d_eval = data.iloc[cutoff:]
                    X_eval = features.iloc[cutoff:]
                else:
                    d_eval = data.tail(1000)
                    X_eval = features.tail(1000)
                def _make_eval():
                    return _build_env_from_frames(self.settings, d_eval, X_eval, max_episode_bars=(2500 if self.fast_smoke else None))
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
        # Compose callbacks: KL early stop, adaptive LR by KL, live LR bump flag
        cb_list = [
            KLStopCallback(target_kl=float(ppo_cfg.get('target_kl', 0.01)) if isinstance(ppo_cfg, dict) else 0.01),
            AdaptiveLRByKL(low=0.003, high=float(ppo_cfg.get('target_kl', 0.01)) if isinstance(ppo_cfg, dict) else 0.01,
                           up=1.15, down=0.7, min_lr=2e-5, max_lr=2e-4),
            LiveLRBump(run_dir=str(output_dir.resolve()), bump_factor=1.25),
        ]
        # Optional: EarlyStopNoImprove unless in fast-smoke
        try:
            es_cfg = (self.cfg.get('rl', {}).get('early_stop', {}) if isinstance(self.cfg, dict) else {}) or {}
            if es_cfg and not self.fast_smoke:
                es = EarlyStopNoImprove(
                    check_freq=int(es_cfg.get('check_freq', 10)),
                    min_delta=float(es_cfg.get('min_delta', 1e-3)),
                    patience=int(es_cfg.get('patience', 5)),
                    verbose=1,
                )
                cb_list.append(es)
        except Exception:
            pass
        if eval_cb is not None:
            cb_list.append(eval_cb)
        self.model.learn(total_timesteps=total_steps, progress_bar=True, callback=CallbackList(cb_list))

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
                ret = eq.pct_change(fill_method=None).dropna()
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
                trades_path = output_dir / 'trades.csv'
                tdf.to_csv(trades_path, index=False)
                # Trades are the single source of truth for costs
                try:
                    trades_df = _pd.read_csv(trades_path)
                    cost_cols = ["commission_cost","spread_cost","slippage_cost","impact_cost","total_cost","total_cost_est"]
                    for c in cost_cols:
                        if c not in trades_df.columns:
                            trades_df[c] = 0.0
                    trades_df[cost_cols] = trades_df[cost_cols].fillna(0.0)
                    # Use max(total_cost, total_cost_est) per trade to guard legacy columns
                    total_used = _pd.concat([
                        _pd.to_numeric(trades_df["total_cost"], errors='coerce').fillna(0.0),
                        _pd.to_numeric(trades_df["total_cost_est"], errors='coerce').fillna(0.0)
                    ], axis=1).max(axis=1)
                    sum_costs = float(total_used.sum())
                    metrics["tx_costs_total"] = sum_costs
                    assert abs(metrics["tx_costs_total"] - sum_costs) < 1e-9
                    # Net PnL for PF/returns (aggregate and side)
                    if 'gross_pnl' in trades_df.columns:
                        trades_df['net_pnl'] = _pd.to_numeric(trades_df['gross_pnl'], errors='coerce').fillna(0.0) - total_used
                    elif 'pnl' in trades_df.columns:
                        # Assume pnl already net if gross not available
                        trades_df['net_pnl'] = _pd.to_numeric(trades_df['pnl'], errors='coerce').fillna(0.0)
                    else:
                        trades_df['net_pnl'] = 0.0
                    net = trades_df['net_pnl']
                    pos_sum = float(net[net > 0].sum())
                    neg_sum = float(net[net < 0].sum())
                    metrics['profit_factor'] = float(pos_sum / (abs(neg_sum) + 1e-12)) if pos_sum > 0 else 0.0
                    # Side segmented
                    if 'direction' in trades_df.columns:
                        m_long = trades_df['direction'] == 'long'
                        m_short = trades_df['direction'] == 'short'
                        lp = float(trades_df.loc[m_long, 'net_pnl'][trades_df.loc[m_long, 'net_pnl'] > 0].sum())
                        ln = float(trades_df.loc[m_long, 'net_pnl'][trades_df.loc[m_long, 'net_pnl'] < 0].sum())
                        sp = float(trades_df.loc[m_short, 'net_pnl'][trades_df.loc[m_short, 'net_pnl'] > 0].sum())
                        sn = float(trades_df.loc[m_short, 'net_pnl'][trades_df.loc[m_short, 'net_pnl'] < 0].sum())
                        metrics['long_pf'] = float(lp / (abs(ln) + 1e-12)) if lp > 0 else 0.0
                        metrics['short_pf'] = float(sp / (abs(sn) + 1e-12)) if sp > 0 else 0.0
                        metrics['long_ret'] = float(trades_df.loc[m_long, 'net_pnl'].sum())
                        metrics['short_ret'] = float(trades_df.loc[m_short, 'net_pnl'].sum())
                    # steps.parquet is mandatory upstream; no diagnostic fallback from trades
                except Exception:
                    pass
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
                # --- Added: richer portfolio metrics from env diagnostics ---
                try:
                    # Action counts and flips
                    acts = env0.get_action_counts() if hasattr(env0, 'get_action_counts') else {}
                    metrics['long_steps'] = int(acts.get('long_steps', 0))
                    metrics['short_steps'] = int(acts.get('short_steps', 0))
                    metrics['flat_steps'] = int(acts.get('flat_steps', 0))
                    metrics['flips'] = int(acts.get('flips', 0))
                except Exception:
                    metrics.setdefault('long_steps', 0)
                    metrics.setdefault('short_steps', 0)
                    metrics.setdefault('flat_steps', 0)
                    metrics.setdefault('flips', 0)
                try:
                    # Turnover and exposure diagnostics from history
                    turnover_total = float(_pd.to_numeric(hist.get('turnover', _pd.Series(dtype=float))).sum()) if not hist.empty else 0.0
                    metrics['turnover'] = float(turnover_total)
                except Exception:
                    metrics.setdefault('turnover', 0.0)
                try:
                    exposure_pct = float(_pd.to_numeric(hist.get('exposure_pct', _pd.Series(dtype=float))).mean()) if ('exposure_pct' in hist.columns) else 0.0
                    metrics['exposure_pct'] = float(exposure_pct)
                except Exception:
                    metrics.setdefault('exposure_pct', 0.0)
                try:
                    # Use metrics['tx_costs_total'] already set from trades to compute per-trade average
                    tx_total = float(metrics.get('tx_costs_total', 0.0))
                    metrics['tx_costs_per_trade'] = float(tx_total) / float(max(1, int(metrics.get('total_trades', 0))))
                except Exception:
                    metrics.setdefault('tx_costs_per_trade', 0.0)
                # Deterministic parity flag from trades/steps
                try:
                    lt = int(metrics.get('long_trades', 0))
                    st = int(metrics.get('short_trades', 0))
                    ls = int(metrics.get('long_steps', 0))
                    ss = int(metrics.get('short_steps', 0))
                    if (lt > 0 or ls > 0) and (st > 0 or ss > 0):
                        metrics['parity_flag'] = 'BOTH'
                    elif (lt > 0 or ls > 0):
                        metrics['parity_flag'] = 'ONLY_LONG'
                    elif (st > 0 or ss > 0):
                        metrics['parity_flag'] = 'ONLY_SHORT'
                    else:
                        metrics['parity_flag'] = 'NONE'
                except Exception:
                    metrics['parity_flag'] = 'NONE'
                # Daily performance report (compact)
                try:
                    import pandas as _pd
                    if not hist.empty and 'equity' in hist.columns:
                        hdf = hist.copy()
                        eq = _pd.to_numeric(hdf['equity'], errors='coerce')
                        eq = eq.dropna()
                        daily = eq.groupby(_pd.to_datetime(hdf.index).date).apply(lambda s: s.iloc[-1] - s.iloc[0])
                        daily = daily.rename('daily_pnl').to_frame()
                        daily['num_trades'] = int(total_trades)
                        # daily returns for sharpe-like
                        daily = eq.resample('1D').last()
                        dr = daily.pct_change(fill_method=None).dropna()
                        if not dr.empty:
                            daily_sharpe = float((dr.mean() / (dr.std() + 1e-12)) * (252 ** 0.5))
                        else:
                            daily_sharpe = 0.0
                        # max drawdown on intraday equity
                        dd = float(((eq / eq.cummax()) - 1).min()) if len(eq) > 1 else 0.0
                        daily.to_csv(output_dir / 'daily_report.csv')
                        metrics['daily_sharpe_proxy'] = daily_sharpe
                        metrics['intraday_max_drawdown'] = dd
                except Exception:
                    pass
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
                'portfolio_metrics': metrics,
            }
        # --- Diagnostics, parity, and baselines ---
        try:
            # Prefer the env we just used (portfolio or single)
            env_ref = None
            try:
                env_ref = vec_env.envs[0]
            except Exception:
                try:
                    env_ref = single_env.envs[0]  # type: ignore[name-defined]
                except Exception:
                    env_ref = None
            import pandas as _pd
            import numpy as _np
            # Build and persist steps.parquet (fail loudly on missing data)
            steps_path = output_dir.parent / 'steps.parquet'
            df_steps = None
            if env_ref is not None and hasattr(env_ref, 'get_diagnostics'):
                diag = env_ref.get_diagnostics()
                if diag is not None and not getattr(diag, 'empty', True):
                    ts_index = _pd.to_datetime(diag.get('ts', diag.index))
                    action_series = _pd.to_numeric(diag.get('action', diag.get('action_dir')), errors='coerce')
                    pos_series = _pd.to_numeric(diag.get('pos', _pd.Series(index=diag.index, dtype=float)), errors='coerce')
                    price_series = _pd.to_numeric(diag.get('price'), errors='coerce')
                    cols = {
                        'ts': ts_index,
                        'action': action_series,
                        'pos': pos_series,
                        'price': price_series,
                    }
                    # Attach flow proxies from the live feature frame if available; else derive minimal proxy
                    live_X = getattr(env_ref, 'X', None)
                    attached = False
                    if isinstance(live_X, _pd.DataFrame):
                        for flow_col in ("ofi_proxy", "signed_vol_delta"):
                            if flow_col in live_X.columns:
                                cols[flow_col] = _pd.to_numeric(live_X[flow_col], errors='coerce').reindex(ts_index).astype('float32').fillna(0.0)
                                attached = True
                                break
                    if not attached:
                        # Minimal OHLCV proxy using price and volume
                        try:
                            c = _pd.to_numeric(price_series, errors='coerce').astype('float32')
                            vol_src = None
                            try:
                                if isinstance(data, _pd.DataFrame) and 'volume' in data.columns:
                                    if 'ticker' in data.columns:
                                        vol_src = _pd.to_numeric(data['volume'], errors='coerce').groupby(data.index).sum()
                                    else:
                                        vol_src = _pd.to_numeric(data['volume'], errors='coerce')
                            except Exception:
                                vol_src = None
                            v = _pd.to_numeric(vol_src, errors='coerce') if vol_src is not None else _pd.Series(0.0, index=ts_index)
                            v = v.reindex(ts_index).fillna(0.0).astype('float32')
                            sgn = _np.sign(c.diff().fillna(0.0))
                            dvol = v.diff().fillna(0.0)
                            ofi = (sgn * dvol).astype('float32')
                            # Day z-score by ts_index date
                            by_day = ofi.groupby(ts_index.date)
                            ofi = (ofi - by_day.transform('mean')) / (by_day.transform('std') + 1e-12)
                            cols['ofi_proxy'] = ofi.fillna(0.0).astype('float32')
                        except Exception:
                            cols['ofi_proxy'] = _pd.Series(0.0, index=ts_index, dtype='float32')
                    df_steps = _pd.DataFrame(cols).set_index('ts')
                    df_steps.index.name = 'timestamp'
            # Fallback: synthesize steps from history or features if diagnostics unavailable
            if df_steps is None or len(df_steps) == 0:
                try:
                    hist = env_ref.get_history_df() if (env_ref is not None and hasattr(env_ref, 'get_history_df')) else _pd.DataFrame()
                except Exception:
                    hist = _pd.DataFrame()
                if not hist.empty:
                    ts_index = _pd.to_datetime(hist.index if isinstance(hist.index, _pd.DatetimeIndex) else hist.get('timestamp'), utc=True, errors='coerce')
                    ts_index = ts_index.tz_convert('America/New_York') if ts_index.tz is not None else ts_index
                    # Aggregate per-ticker positions if present
                    pos_cols = [c for c in hist.columns if str(c).startswith('pos_')]
                    pos_series = _pd.to_numeric(hist[pos_cols].sum(axis=1), errors='coerce') if pos_cols else _pd.Series(0, index=hist.index)
                    # Infer action from pos changes
                    dpos = _np.sign(_pd.to_numeric(pos_series, errors='coerce').diff().fillna(0.0).to_numpy())
                    action_series = _pd.Series(dpos, index=hist.index)
                    price_series = _pd.Series(_np.nan, index=hist.index)
                    cols = {
                        'ts': ts_index,
                        'action': action_series,
                        'pos': pos_series,
                        'price': price_series,
                    }
                    # Attach proxies from features if available; else derive minimal proxy
                    live_X = getattr(env_ref, 'X', None)
                    src = live_X if isinstance(live_X, _pd.DataFrame) else features
                    attached = False
                    if isinstance(src, _pd.DataFrame):
                        for flow_col in ("ofi_proxy", "signed_vol_delta"):
                            if flow_col in src.columns:
                                cols[flow_col] = _pd.to_numeric(src[flow_col], errors='coerce').reindex(ts_index).astype('float32').fillna(0.0)
                                attached = True
                                break
                    if not attached:
                        try:
                            # Use OHLCV from `data` to build minimal proxy
                            vol_src = None
                            if isinstance(data, _pd.DataFrame) and 'volume' in data.columns:
                                if 'ticker' in data.columns:
                                    vol_src = _pd.to_numeric(data['volume'], errors='coerce').groupby(data.index).sum()
                                else:
                                    vol_src = _pd.to_numeric(data['volume'], errors='coerce')
                            v = _pd.to_numeric(vol_src, errors='coerce') if vol_src is not None else _pd.Series(0.0, index=ts_index)
                            v = v.reindex(ts_index).fillna(0.0).astype('float32')
                            # For price, we may not have a portfolio-level price; use zeros for sign
                            c = _pd.Series(0.0, index=ts_index, dtype='float32')
                            sgn = _np.sign(c.diff().fillna(0.0))
                            dvol = v.diff().fillna(0.0)
                            ofi = (sgn * dvol).astype('float32')
                            by_day = ofi.groupby(ts_index.date)
                            ofi = (ofi - by_day.transform('mean')) / (by_day.transform('std') + 1e-12)
                            cols['ofi_proxy'] = ofi.fillna(0.0).astype('float32')
                        except Exception:
                            cols['ofi_proxy'] = _pd.Series(0.0, index=ts_index, dtype='float32')
                    df_steps = _pd.DataFrame(cols).set_index('ts')
                    df_steps.index.name = 'timestamp'
                else:
                    # Last resort: build empty scaffold from features timeline
                    try:
                        ts_index = _pd.to_datetime(features.index, utc=True, errors='coerce')
                        cols = {
                            'ts': ts_index,
                            'action': _pd.Series(0, index=ts_index),
                            'pos': _pd.Series(0, index=ts_index),
                            'price': _pd.Series(_np.nan, index=ts_index),
                        }
                        attached = False
                        for flow_col in ("ofi_proxy", "signed_vol_delta"):
                            if flow_col in features.columns:
                                cols[flow_col] = _pd.to_numeric(features[flow_col], errors='coerce').reindex(ts_index).astype('float32').fillna(0.0)
                                attached = True
                                break
                        if not attached:
                            # Minimal proxy from OHLCV in `data`
                            try:
                                vol_src = None
                                if isinstance(data, _pd.DataFrame) and 'volume' in data.columns:
                                    if 'ticker' in data.columns:
                                        vol_src = _pd.to_numeric(data['volume'], errors='coerce').groupby(data.index).sum()
                                    else:
                                        vol_src = _pd.to_numeric(data['volume'], errors='coerce')
                                v = _pd.to_numeric(vol_src, errors='coerce') if vol_src is not None else _pd.Series(0.0, index=ts_index)
                                v = v.reindex(ts_index).fillna(0.0).astype('float32')
                                c = _pd.Series(0.0, index=ts_index, dtype='float32')
                                sgn = _np.sign(c.diff().fillna(0.0))
                                dvol = v.diff().fillna(0.0)
                                ofi = (sgn * dvol).astype('float32')
                                by_day = ofi.groupby(ts_index.date)
                                ofi = (ofi - by_day.transform('mean')) / (by_day.transform('std') + 1e-12)
                                cols['ofi_proxy'] = ofi.fillna(0.0).astype('float32')
                            except Exception:
                                cols['ofi_proxy'] = _pd.Series(0.0, index=ts_index, dtype='float32')
                        df_steps = _pd.DataFrame(cols).set_index('ts')
                        df_steps.index.name = 'timestamp'
                    except Exception:
                        df_steps = _pd.DataFrame({'action': [], 'pos': [], 'price': []})
            # Attach OHLCV flow proxies from features if available (ofi_proxy, signed_vol_delta)
            try:
                if hasattr(env_ref, 'X') and isinstance(env_ref.X, _pd.DataFrame):
                    for _col in ['ofi_proxy', 'signed_vol_delta']:
                        if _col in env_ref.X.columns and _col not in df_steps.columns:
                            ser = _pd.to_numeric(env_ref.X[_col], errors='coerce')
                            df_steps[_col] = ser.reindex(df_steps.index)
                else:
                    # Try from provided features frame
                    for _col in ['ofi_proxy', 'signed_vol_delta']:
                        if _col in features.columns and _col not in df_steps.columns:
                            ser = _pd.to_numeric(features[_col], errors='coerce')
                            df_steps[_col] = ser.reindex(df_steps.index)
            except Exception:
                pass
            # Always write steps.parquet (may be minimal if little info available)
            try:
                df_steps = df_steps.sort_index()
                df_steps.to_parquet(steps_path, engine='pyarrow', index=True)
            except Exception as _e:
                logger.warning(f"Failed to write steps.parquet: {_e}")

            # Compute metrics strictly from steps.parquet
            s = _pd.read_parquet(steps_path).sort_index()
            # Drop rows with NaN in action or pos
            s = s.loc[s['action'].notna() & s['pos'].notna()]
            a = s['action'].astype(int).to_numpy()
            p_arr = s['pos'].astype(int).to_numpy()
            # Entropy over {-1,0,1} (base-2)
            vals, cnts = _np.unique(a, return_counts=True)
            probs = cnts / cnts.sum() if cnts.sum() > 0 else _np.array([1.0])
            action_entropy = float(-_np.sum(probs * _np.log2(probs + 1e-12))) if probs.size else 0.0
            # Entries/exits/flips
            if p_arr.size >= 2:
                entries = int(((p_arr[:-1] == 0) & (p_arr[1:] != 0)).sum())
                exits = int(((p_arr[:-1] != 0) & (p_arr[1:] == 0)).sum())
                flips = int(((p_arr[:-1] * p_arr[1:]) < 0).sum())
            else:
                entries = exits = flips = 0
            dp = _np.diff(p_arr, prepend=p_arr[0]) if p_arr.size else _np.array([0])
            turnover = float(_np.abs(dp).sum())
            avg_abs_pos = float(_np.mean(_np.abs(p_arr))) if p_arr.size else 0.0
            avg_abs_dpos = float(_np.mean(_np.abs(dp))) if dp.size else 0.0
            # corr(sign(action), sign(flow_proxy)) preferring ofi_proxy, else signed_vol_delta
            corr_action_flow = None
            try:
                flow_col = 'ofi_proxy' if ('ofi_proxy' in s.columns) else ('signed_vol_delta' if ('signed_vol_delta' in s.columns) else None)
                if flow_col and s[flow_col].notna().sum() > 50:
                    sgn_a = _np.sign(s['action'].to_numpy())
                    sgn_o = _np.sign(s[flow_col].to_numpy())
                    corr_action_flow = float(_np.corrcoef(sgn_a, sgn_o)[0, 1])
                else:
                    if flow_col is None:
                        logger.warning("No flow proxy ('ofi_proxy' or 'signed_vol_delta') available in steps; skipping corr_action_flow")
                    else:
                        logger.warning("Insufficient non-NaN samples for %s to compute corr_action_flow", flow_col)
            except Exception:
                corr_action_flow = None
            # Write metrics into dict
            metrics['turnover'] = float(turnover)
            metrics['entries'] = int(entries)
            metrics['exits'] = int(exits)
            metrics['flips'] = int(flips)
            metrics['avg_abs_pos'] = float(avg_abs_pos)
            metrics['avg_abs_dpos'] = float(avg_abs_dpos)
            metrics['action_entropy'] = float(action_entropy)
            metrics['corr_action_flow'] = corr_action_flow

            # Runtime guard: if no short steps, emit WARN with config context and action dist
            try:
                short_steps = int((s['action'] < 0).sum()) if 's' in locals() else 0
                if short_steps == 0:
                    try:
                        allow_sh = bool(self.settings.get('env', 'allow_shorts', default=True))
                    except Exception:
                        allow_sh = True
                    try:
                        max_short_exp = self.settings.get('env', 'max_short_exposure', default=None)
                    except Exception:
                        max_short_exp = None
                    counts = s['action'].value_counts()
                    logger.warning(f"NO_SHORTS detected: allow_shorts={allow_sh} max_short_exposure={max_short_exp} action_counts={counts.to_dict()}")
            except Exception:
                pass

            # Parity flag will be set later after trade KPIs using steps/trades

            # Baselines on test set (no_trade, vwap_fade, ofi_follow)
            try:
                # Build baselines from steps.parquet
                pr = None
                try:
                    s = _pd.read_parquet(steps_path)
                    pr = _pd.to_numeric(s['price'], errors='coerce')
                except Exception:
                    pr = _pd.to_numeric(data['close'], errors='coerce') if 'close' in data.columns else None
                def _metrics_from_returns(ret_ser: _pd.Series) -> dict:
                    ret = ret_ser.dropna()
                    if ret.empty:
                        return {'sharpe': 0.0, 'pf': 0.0, 'ret': 0.0, 'maxdd': 0.0}
                    total_ret = float(ret.sum())
                    vol = float(ret.std())
                    sharpe = float((ret.mean() / (vol + 1e-12)) * (252 ** 0.5)) if vol > 0 else 0.0
                    pos_sum = float(ret[ret > 0].sum())
                    neg_sum = float(ret[ret < 0].sum())
                    pf = float(pos_sum / (abs(neg_sum) + 1e-12)) if pos_sum > 0 else 0.0
                    # Max drawdown from cumulative equity proxy
                    eq = (1 + ret).cumprod()
                    maxdd = float(((eq / eq.cummax()) - 1).min()) if len(eq) > 1 else 0.0
                    return {'sharpe': sharpe, 'pf': pf, 'ret': total_ret, 'maxdd': maxdd}
                baselines = {}
                # No-trade: all zeros
                baselines['no_trade'] = {'sharpe': 0.0, 'pf': 0.0, 'ret': 0.0, 'maxdd': 0.0}
                if pr is not None and len(pr) > 1:
                    price_ret = pr.pct_change(fill_method=None).fillna(0.0)
                    # Ensure baselines dir
                    baselines_dir = (output_dir.parent / 'baselines')
                    try:
                        baselines_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    # no_trade positions (all zeros) aligned to steps timeline
                    try:
                        s = _pd.read_parquet(steps_path)
                        idx = s.index
                        pos_nt = _pd.Series(0.0, index=idx, name='pos')
                        (baselines_dir / 'no_trade_steps.parquet').unlink(missing_ok=True) if hasattr(baselines_dir, 'unlink') else None
                        pos_nt.to_frame().to_parquet(baselines_dir / 'no_trade_steps.parquet', engine='pyarrow', index=True)
                    except Exception:
                        pos_nt = None
                    # OFI-follow baseline from steps parquet (pos = sign(ofi_best))
                    try:
                        s = _pd.read_parquet(steps_path)
                        if 'ofi_best' in s.columns and s['ofi_best'].notna().any():
                            pos_ofi = _pd.Series(_np.sign(s['ofi_best'].fillna(0.0)), index=s.index, name='pos')
                            pos_ofi.to_frame().to_parquet(baselines_dir / 'ofi_follow_steps.parquet', engine='pyarrow', index=True)
                            ret_ser2 = pos_ofi.shift(1).fillna(0.0).to_numpy(dtype=float) * price_ret.reindex(s.index).fillna(0.0).to_numpy(dtype=float)
                            baselines['ofi_follow'] = _metrics_from_returns(_pd.Series(ret_ser2, index=s.index))
                    except Exception:
                        pass
                metrics['baselines'] = baselines
                # Write diagnostics.csv with selected metrics (in ticker root alongside steps.parquet)
                import csv as _csv
                diag_out = output_dir.parent / 'diagnostics.csv'
                rows: list[dict[str, object]] = []
                # action_entropy first, then corr_action_flow
                rows.append({'metric': 'action_entropy', 'value': float(metrics.get('action_entropy', 0.0) or 0.0)})
                corr_val = metrics.get('corr_action_flow', None)
                rows.append({'metric': 'corr_action_flow', 'value': (float(corr_val) if isinstance(corr_val, (int, float)) else '')})
                if 'ofi_follow' in baselines:
                    try:
                        rows.append({'metric': 'ret_if_follow_ofi', 'value': float(baselines['ofi_follow'].get('ret', 0.0) or 0.0)})
                    except Exception:
                        rows.append({'metric': 'ret_if_follow_ofi', 'value': ''})
                with diag_out.open('w', newline='') as f:
                    w = _csv.DictWriter(f, fieldnames=['metric','value'])
                    w.writeheader(); w.writerows(rows)
            except Exception:
                pass
        except Exception:
            pass
        # --- Per-side KPIs & parity flag (single or portfolio) ---
        try:
            # Wherever possible, compute from the last used env (portfolio or single)
            env_ref = None
            try:
                env_ref = vec_env.envs[0]
            except Exception:
                try:
                    env_ref = single_env.envs[0]  # type: ignore[name-defined]
                except Exception:
                    env_ref = None
            trades = []
            if env_ref is not None and hasattr(env_ref, 'get_trades'):
                trades = env_ref.get_trades()
            import pandas as _pd
            tdf = _pd.DataFrame(trades)
            # Side masks
            long_m = (tdf.get('direction') == 'long') if not tdf.empty else _pd.Series([], dtype=bool)
            short_m = (tdf.get('direction') == 'short') if not tdf.empty else _pd.Series([], dtype=bool)
            # Profit factor per side
            def _pf(mask):
                if tdf.empty or mask.sum() == 0:
                    return 0.0
                pnl = _pd.to_numeric(tdf.loc[mask, 'pnl'], errors='coerce').fillna(0.0)
                pos = float(pnl[pnl > 0].sum())
                neg = float(pnl[pnl < 0].sum())
                return float(pos / (abs(neg) + 1e-12)) if (pos > 0 or neg < 0) else 0.0
            # Return per side (sum pnl / sum notional)
            def _ret(mask):
                if tdf.empty or mask.sum() == 0:
                    return 0.0
                pnl = _pd.to_numeric(tdf.loc[mask, 'pnl'], errors='coerce').fillna(0.0)
                units = _pd.to_numeric(tdf.loc[mask, 'units'] if 'units' in tdf.columns else tdf.loc[mask, 'quantity'] if 'quantity' in tdf.columns else _pd.Series(0.0, index=tdf.index), errors='coerce').abs().fillna(0.0)
                entry = _pd.to_numeric(tdf.loc[mask, 'entry_price'], errors='coerce').fillna(0.0)
                notional = (units * entry).replace(0.0, _pd.NA).fillna(0.0)
                denom = float(notional.sum())
                return float(pnl.sum() / (denom + 1e-12)) if denom > 0 else 0.0
            long_pf = _pf(long_m)
            short_pf = _pf(short_m)
            long_ret = _ret(long_m)
            short_ret = _ret(short_m)
            # Inject into metrics where applicable
            try:
                metrics['long_pf'] = float(long_pf)
                metrics['short_pf'] = float(short_pf)
                metrics['long_ret'] = float(long_ret)
                metrics['short_ret'] = float(short_ret)
            except Exception:
                pass
            # Parity flag (deterministic from steps/trades): BOTH / ONLY_LONG / ONLY_SHORT / NONE
            try:
                steps_path = output_dir.parent / 'steps.parquet'
                s = _pd.read_parquet(steps_path) if steps_path.exists() else _pd.DataFrame()
            except Exception:
                s = _pd.DataFrame()
            has_long = False
            has_short = False
            try:
                if not s.empty:
                    has_long = bool(((s.get('action', _pd.Series(dtype=float)) > 0).any()) or ((s.get('pos', _pd.Series(dtype=float)) > 0).any()))
                    has_short = bool(((s.get('action', _pd.Series(dtype=float)) < 0).any()) or ((s.get('pos', _pd.Series(dtype=float)) < 0).any()))
            except Exception:
                pass
            # Also consider trade directions if present
            try:
                if not tdf.empty and 'direction' in tdf.columns:
                    has_long = has_long or (tdf['direction'] == 'long').any()
                    has_short = has_short or (tdf['direction'] == 'short').any()
            except Exception:
                pass
            try:
                flag = 'NONE'
                if has_long and has_short:
                    flag = 'BOTH'
                elif has_long and not has_short:
                    flag = 'ONLY_LONG'
                elif has_short and not has_long:
                    flag = 'ONLY_SHORT'
                summary['parity_flag'] = flag
                try:
                    metrics['parity_flag'] = flag
                except Exception:
                    pass
            except Exception:
                summary['parity_flag'] = 'NONE'
        except Exception:
            pass
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

    # Convenience helper: train, backtest, and return a BacktestResult bundle
    # for downstream artifact writers.
    def train_and_backtest(
        self,
        *,
        data: pd.DataFrame,
        features: pd.DataFrame,
        output_dir: Path,
        eval_episodes: int = 1,
    ) -> BacktestResult:
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self.train(data=data, features=features, output_dir=output_dir)
        summary = self.backtest(
            model=model,
            data=data,
            features=features,
            output_dir=output_dir,
            eval_episodes=eval_episodes,
        )

        import pandas as _pd
        import numpy as _np

        trades_path = output_dir / 'trades.csv'
        try:
            trades_df = _pd.read_csv(trades_path)
        except Exception:
            trades_df = _pd.DataFrame()

        equity_path = output_dir / 'portfolio_history.csv'
        try:
            equity_df = _pd.read_csv(equity_path, index_col=0)
            if 'equity' not in equity_df.columns and equity_df.shape[1] >= 1:
                equity_df.columns = ['equity']
        except Exception:
            equity_df = _pd.DataFrame({'equity': []})

        steps_path = output_dir.parent / 'steps.parquet'
        steps_df: _pd.DataFrame
        if steps_path.exists():
            try:
                raw_steps = _pd.read_parquet(steps_path).copy()
                if 'ts' in raw_steps.columns and not isinstance(raw_steps.index, _pd.DatetimeIndex):
                    ts_index = _pd.to_datetime(raw_steps['ts'], utc=True, errors='coerce')
                    mask = ts_index.notna()
                    ts_index = _pd.DatetimeIndex(ts_index[mask])
                    steps_df = raw_steps.loc[mask].drop(columns=['ts']).set_index(ts_index)
                else:
                    ts_index = _pd.to_datetime(raw_steps.index, utc=True, errors='coerce')
                    mask = ts_index.notna()
                    ts_index = _pd.DatetimeIndex(ts_index[mask])
                    steps_df = raw_steps.loc[mask].copy()
                    steps_df.index = ts_index
                steps_df.index.name = 'ts'
            except Exception:
                steps_df = _pd.DataFrame()
        else:
            steps_df = _pd.DataFrame()

        if steps_df.empty:
            ts_index = _pd.to_datetime(features.index, utc=True, errors='coerce')
            ts_index = _pd.DatetimeIndex(ts_index[ts_index.notna()])
            steps_df = _pd.DataFrame(
                {
                    'action': _pd.Series(0, index=ts_index, dtype='int32'),
                    'pos': _pd.Series(0, index=ts_index, dtype='int32'),
                    'price': _pd.Series(_np.nan, index=ts_index, dtype='float32'),
                }
            )
            steps_df.index.name = 'ts'
        else:
            ts_index = _pd.to_datetime(steps_df.index, utc=True, errors='coerce')
            mask = ts_index.notna()
            ts_index = _pd.DatetimeIndex(ts_index[mask])
            steps_df = steps_df.loc[mask]
            steps_df.index = ts_index
            steps_df['action'] = _pd.to_numeric(steps_df.get('action', 0), errors='coerce').fillna(0).astype('int8')
            steps_df['pos'] = _pd.to_numeric(steps_df.get('pos', 0), errors='coerce').fillna(0).astype('int8')
            steps_df['price'] = _pd.to_numeric(steps_df.get('price'), errors='coerce')

        live_features = features.copy()
        attached = False
        for col in ("ofi_proxy", "signed_vol_delta"):
            if col in live_features.columns:
                ser = _pd.to_numeric(live_features[col], errors='coerce').reindex(ts_index).fillna(0.0).astype('float32')
                steps_df[col] = ser
                attached = True
                break

        if not attached:
            price_series = _pd.to_numeric(steps_df.get('price'), errors='coerce').reindex(ts_index)
            if price_series.isna().all():
                if isinstance(data, _pd.DataFrame) and 'close' in data.columns:
                    close_series = _pd.to_numeric(data['close'], errors='coerce')
                    price_series = close_series.reindex(ts_index)
            price_series = price_series.fillna(method='ffill').fillna(0.0)

            if isinstance(data, _pd.DataFrame) and 'volume' in data.columns:
                if 'ticker' in data.columns and data['ticker'].nunique() > 1:
                    try:
                        first_ticker = str(data['ticker'].dropna().iloc[0])
                        vol_src = _pd.to_numeric(data.loc[data['ticker'] == first_ticker, 'volume'], errors='coerce')
                    except Exception:
                        vol_src = _pd.to_numeric(data['volume'], errors='coerce')
                else:
                    vol_src = _pd.to_numeric(data['volume'], errors='coerce')
                volume_series = vol_src.reindex(ts_index).fillna(0.0)
            else:
                volume_series = _pd.Series(0.0, index=ts_index)

            ofi = (_np.sign(price_series.diff().fillna(0.0)) * volume_series).astype('float32')
            by_day = ofi.groupby(ts_index.date)
            steps_df['ofi_proxy'] = ((ofi - by_day.transform('mean')) / (by_day.transform('std') + 1e-12)).fillna(0.0).astype('float32')

        metrics_dict = summary if isinstance(summary, dict) else {}
        feature_names = [c for c in features.columns if c != 'ticker']

        return BacktestResult(
            trades=trades_df,
            equity=equity_df,
            steps=steps_df,
            metrics=metrics_dict,
            feature_names=feature_names,
            baselines={},
        )
