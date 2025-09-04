"""
Training infrastructure for PPO-LSTM reinforcement learning.

This module provides the complete training pipeline including
walk-forward validation, model management, and training utilities.
"""
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import re
import json

from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Alias for compatibility
# MlpLstmPolicy = ActorCriticPolicy

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..features.pipeline import FeaturePipeline
from ..sim.env_intraday_rl import IntradayRLEnv, EnvConfig, RiskConfig
from ..sim.execution import ExecParams


logger = get_logger(__name__)


def _make_env_worker(config_path: str, data_path: str, features_path: str):
    """Top-level factory for SubprocVecEnv picklability."""
    settings = Settings.from_yaml(config_path)
    return build_env(settings, data_path, features_path)


class RLTrainer:
    """
    RL trainer wrapper class for easy model training and management.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize RL trainer.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.training_config = TrainingConfig()
        self.model = None
        
    def train(self, data_path: str, features_path: str, model_path: str) -> RecurrentPPO:
        """
        Train RL model.
        
        Args:
            data_path: Path to training data
            features_path: Path to features
            model_path: Path to save model
            
        Returns:
            Trained PPO model
        """
        self.model = train_ppo_lstm(
            settings=self.settings,
            data_path=data_path,
            features_path=features_path,
            model_path=model_path,
            training_config=self.training_config
        )
        return self.model
    
    def evaluate(self, data_path: str, features_path: str, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            data_path: Path to evaluation data
            features_path: Path to features
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        env = build_env(self.settings, data_path, features_path)
        return evaluate_model(self.model, env, num_episodes)
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load trained model."""
        self.model = RecurrentPPO.load(path)


class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-5
    n_steps: int = 2048
    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    n_minibatches: int = 32
    target_kl: float = 0.016
    tensorboard_log: str = "runs/tensorboard"
    save_freq: int = 10000
    device: str = "auto"
    seed: int = 42
    eval_freq: int = 10000
    eval_episodes: int = 5
    save_replay_buffer: bool = False
    verbose: int = 1


class TrainingCallback(BaseCallback):
    """
    Custom callback for training monitoring and logging.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called after each step."""
        # Log training progress
        if self.locals.get('dones', [False])[0]:
            episode_reward = self.locals.get('episode_rewards', [0])[0]
            episode_length = self.locals.get('episode_lengths', [0])[0]
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Calculate mean reward over last 100 episodes
            if len(self.episode_rewards) >= 100:
                mean_reward = np.mean(self.episode_rewards[-100:])
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("models/best_model")
                    logger.info(f"New best model saved with mean reward: {mean_reward:.2f}")
                
                if self.verbose > 0:
                    logger.info(f"Episode {len(self.episode_rewards)}: "
                              f"Reward: {episode_reward:.2f}, "
                              f"Length: {episode_length}, "
                              f"Mean (100): {mean_reward:.2f}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        logger.info("Training completed")
        logger.info(f"Best mean reward: {self.best_mean_reward:.2f}")


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal entropy coefficient from initial to final over total timesteps."""
    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int):
        super().__init__(verbose=0)
        self.initial = float(initial_ent)
        self.final = float(final_ent)
        self.total = int(max(1, total_timesteps))

    def _on_step(self) -> bool:
        # progress in [0,1]
        t = min(self.model.num_timesteps, self.total)
        frac = t / self.total
        new_ent = self.initial + (self.final - self.initial) * frac
        try:
            # SB3 stores ent_coef on the model
            self.model.ent_coef = float(new_ent)
        except Exception:
            pass
        return True

def build_env(settings: Settings, data_path: str, features_path: str) -> IntradayRLEnv:
    """
    Build the training environment from OHLCV and feature store, using YAML-driven
    execution, risk, and environment parameters so that training and backtest are aligned.

    Args:
        settings: Settings object loaded from configs/settings.yaml
        data_path: Path to OHLCV parquet (must contain columns like open/high/low/close/volume)
        features_path: Path to features parquet (index or column with timestamps)

    Returns:
        An initialized IntradayRLEnv ready for training.
    """
    import pandas as pd
    import numpy as np
    from ..sim.env_intraday_rl import IntradayRLEnv, EnvConfig
    from ..sim.risk import RiskConfig
    from ..sim.execution import ExecParams

    # ---------------------------
    # 1) Load OHLCV and normalize timestamp/index
    # ---------------------------
    df = pd.read_parquet(data_path)

    # Accept either a 'timestamp' column or a DatetimeIndex (optionally epoch ms)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.loc[ts.notna()].copy()
        df['timestamp'] = ts
        df = df.sort_values('timestamp').set_index('timestamp')
    else:
        # If index is not datetime, try to interpret as epoch ms
        if not isinstance(df.index, pd.DatetimeIndex):
            idx = pd.to_datetime(df.index, utc=True, errors='coerce', unit='ms')
            df = df.loc[idx.notna()].copy()
            df.index = idx[idx.notna()]
        # Ensure monotonic and tz-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()

    # Convert to market timezone used by the rest of the stack
    df.index = df.index.tz_convert("America/New_York")

    # Keep standard OHLCV columns that exist
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not keep:
        raise ValueError("OHLCV parquet must include at least one of: open/high/low/close/volume")
    df = df[keep]

    # ---------------------------
    # 2) Load features and align to OHLCV index
    # ---------------------------
    X = pd.read_parquet(features_path)

    # Features may carry their own timestamp; normalize similarly
    if 'timestamp' in X.columns:
        tx = pd.to_datetime(X['timestamp'], utc=True, errors='coerce')
        X = X.loc[tx.notna()].copy()
        X['timestamp'] = tx
        X = X.sort_values('timestamp').set_index('timestamp')
    else:
        if not isinstance(X.index, pd.DatetimeIndex):
            # try epoch ms on index
            idx = pd.to_datetime(X.index, utc=True, errors='coerce', unit='ms')
            X = X.loc[idx.notna()].copy()
            X.index = idx[idx.notna()]
        if X.index.tz is None:
            X.index = X.index.tz_localize("UTC")
        X = X.sort_index()

    X.index = X.index.tz_convert("America/New_York")

    # Align features to price index, forward/back-fill for small gaps,
    # then drop remaining NaNs in critical OHLCV fields.
    X = X.reindex(df.index).ffill().bfill()
    df = df.dropna(subset=["open", "high", "low", "close"])

    # ---------------------------
    # 3) Pull execution/risk/env params from YAML
    # ---------------------------
    # Execution parameters (costs & microstructure)
    exec_params = ExecParams(
        tick_value=float(settings.get("execution", "tick_value", default=0.01)),
        spread_ticks=int(settings.get("execution", "spread_ticks", default=1)),
        impact_bps=float(settings.get("execution", "impact_bps", default=0.5)),
        commission_per_contract=float(settings.get("execution", "commission_per_contract", default=0.0035)),
    )

    # Risk configuration (position sizing & kill-switch thresholds)
    risk_cfg = RiskConfig(
        risk_per_trade_frac=float(settings.get("risk", "risk_per_trade_frac", default=0.02)),
        stop_r_multiple=float(settings.get("risk", "stop_r_multiple", default=1.0)),
        tp_r_multiple=float(settings.get("risk", "tp_r_multiple", default=1.5)),
        max_daily_loss_r=float(settings.get("risk", "max_daily_loss_r", default=3.0)),
    )

    # Instrument point value (SPY ETF should be 1.0; futures differ)
    point_value = float(settings.get("execution", "point_value", default=1.0))

    # Environment config — single, consolidated instance
    reward_type = str(settings.get("env", "reward", "kind", default="dsr"))
    reward_scaling = float(settings.get("env", "reward_scaling", default=0.1))
    max_steps = int(settings.get("env", "max_steps", default=390))  # episodic training horizon

    env_cfg = EnvConfig(
        cash=100_000.0,
        max_steps=max_steps,
        reward_type=reward_type,
        reward_scaling=reward_scaling
    )

    # ---------------------------
    # 4) Build environment (single EnvConfig here)
    # ---------------------------
    env = IntradayRLEnv(
        ohlcv=df,
        features=X,
        cash=100_000.0,
        exec_params=exec_params,
        risk_cfg=risk_cfg,
        point_value=point_value,
        env_config=env_cfg,
        config=settings.to_dict()  # pass full YAML dict for any downstream needs
    )

    return env


def train_ppo_lstm(settings: Settings, 
                  data_path: str, 
                  features_path: str,
                  model_path: str,
                  training_config: TrainingConfig) -> RecurrentPPO:
    """
    Train PPO-LSTM model.
    
    Args:
        settings: Configuration settings
        data_path: Path to training data
        features_path: Path to features
        model_path: Path to save model
        training_config: Training configuration
        
    Returns:
        Trained PPO model
    """
    logger.info("Starting PPO-LSTM training...")
    
    # Set random seeds (avoid CUDA calls if CPU-only)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    
    # Build one environment (for single-env fallback)
    env = build_env(settings, data_path, features_path)
    # Save feature metadata next to model later
    feature_cols = []
    try:
        feature_cols = list(getattr(env, 'features', getattr(env, 'X', pd.DataFrame())).columns)
    except Exception:
        pass
    
    # Create vectorized environment (support multi-env with subprocess workers)
    from stable_baselines3.common.vec_env import VecNormalize
    n_envs = int(settings.get("train", "n_envs", default=1))
    if n_envs > 1:
        # Use SubprocVecEnv for parallel env stepping; pass config path for picklable factory
        try:
            from functools import partial
            config_path = settings.meta.get("config_file") if hasattr(settings, "meta") else "configs/settings.yaml"
            env_fn = partial(_make_env_worker, config_path, data_path, features_path)
            vec_env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            logger.info(f"Using SubprocVecEnv with n_envs={n_envs}")
        except Exception as e:
            # Fallback to DummyVecEnv replication
            logger.warning(f"SubprocVecEnv init failed ({e}); falling back to DummyVecEnv x{n_envs}")
            vec_env = DummyVecEnv([lambda: build_env(settings, data_path, features_path) for _ in range(n_envs)])
    else:
        vec_env = DummyVecEnv([lambda: env])
        logger.info("Using DummyVecEnv with n_envs=1")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Policy configuration
    policy_kwargs = dict(
        net_arch=dict(
            pi=settings.get("train", "policy_kwargs", "net_arch_pi", default=[128, 64]),
            vf=settings.get("train", "policy_kwargs", "net_arch_vf", default=[128, 64])
        ),
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            eps=1e-5
        ),
        # LSTM specific arguments
        lstm_hidden_size=settings.get("train", "policy_kwargs", "lstm_hidden_size", default=128),
        n_lstm_layers=1, # A reasonable default
        enable_critic_lstm=True,
    )
    
    # Resolve device (avoid CUDA/NVML queries when user forces CPU)
    desired_device = str(getattr(training_config, 'device', 'auto'))
    if desired_device == 'cpu':
        effective_device = 'cpu'
        logger.info("Using device=cpu (CUDA checks skipped)")
    else:
        auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        effective_device = auto_device if desired_device == 'auto' else desired_device
        logger.info(
            f"Torch CUDA available={torch.cuda.is_available()} | torch.cuda.device_count={torch.cuda.device_count()} | device={effective_device}"
        )
        # Only seed CUDA if we are actually using it
        if effective_device == 'cuda':
            try:
                torch.cuda.manual_seed_all(training_config.seed)
            except Exception:
                pass

    # Create RecurrentPPO model with LSTM policy
    # Learning rate schedule (optional): settings.train.lr_schedule: 'linear'|'cosine'|'none'
    lr_value = training_config.learning_rate
    try:
        lr_sched = str(settings.get('train', 'lr_schedule', default='none')).lower()
    except Exception:
        lr_sched = 'none'

    def make_lr_schedule(init_lr: float):
        if lr_sched == 'linear':
            # SB3 passes remaining_progress from 1 -> 0
            return lambda progress: init_lr * progress
        if lr_sched == 'cosine':
            import math
            return lambda progress: init_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return init_lr

    lr_param = make_lr_schedule(training_config.learning_rate)

    model = RecurrentPPO(
        'MlpLstmPolicy',
        vec_env,
        learning_rate=lr_param,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        gamma=training_config.gamma,
        gae_lambda=training_config.gae_lambda,
        clip_range=training_config.clip_range,
        vf_coef=training_config.vf_coef,
        ent_coef=training_config.ent_coef,
        max_grad_norm=training_config.max_grad_norm,
        n_epochs=training_config.n_epochs,
        target_kl=training_config.target_kl,
        tensorboard_log=training_config.tensorboard_log,
        policy_kwargs=policy_kwargs,
        device=effective_device,
        verbose=training_config.verbose,
        seed=training_config.seed
    )
    
    # Hint to PyTorch for small-CPU + GPU setups
    try:
        torch.set_num_threads(int(settings.get("train", "torch_threads", default=1)))
    except Exception:
        pass
    
    # Create callbacks
    callback = TrainingCallback(verbose=training_config.verbose)
    
    # Train model
    # Ensure integer type even if YAML has comments/strings like "10000#100000"
    _raw_steps = settings.get("train", "total_steps", default=1_200_000)
    if isinstance(_raw_steps, (int, np.integer)):
        total_steps = int(_raw_steps)
    else:
        m = re.search(r"[-+]?\d+", str(_raw_steps))
        total_steps = int(m.group()) if m else 1_200_000
    # Optional entropy anneal: train.ent_anneal_final can override final coefficient
    try:
        ent_final = float(settings.get('train', 'ent_anneal_final', default=None))
    except Exception:
        ent_final = None

    callbacks = [callback]
    # Ensure integer type even if YAML has comments/strings like "10000#100000"
    _raw_steps = settings.get("train", "total_steps", default=1_200_000)
    if isinstance(_raw_steps, (int, np.integer)):
        total_steps = int(_raw_steps)
    else:
        m = re.search(r"[-+]?\d+", str(_raw_steps))
        total_steps = int(m.group()) if m else 1_200_000
    if ent_final is not None:
        callbacks.append(EntropyAnnealCallback(initial_ent=model.ent_coef, final_ent=ent_final, total_timesteps=total_steps))

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks if len(callbacks) > 1 else callback,
        log_interval=1000,
        progress_bar=True
    )
    
    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save the VecNormalize statistics
    vec_env.save(str(Path(model_path).parent / "vecnormalize.pkl"))
    logger.info(f"VecNormalize statistics saved to {Path(model_path).parent / 'vecnormalize.pkl'}")
    # Save feature column metadata for parity checks
    try:
        meta = {
            'features': feature_cols,
            'saved_at': datetime.now().isoformat(),
            'data_path': data_path,
            'features_path': features_path
        }
        with open(str(Path(model_path) ) + "_features.json", 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved feature metadata to {str(Path(model_path))+'_features.json'}")
    except Exception as e:
        logger.warning(f"Failed to save feature metadata: {e}")
    
    return model


def evaluate_model(model: RecurrentPPO, env: DummyVecEnv, num_episodes: int = 5) -> Dict[str, float]:
    """
    Evaluate trained model.
    
    Args:
        model: Trained PPO model
        env: Evaluation environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating model for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    equity_curves = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = None
        episode_start = np.ones((env.num_envs,), dtype=bool)
        done = np.array([False])
        episode_reward = 0.0
        episode_length = 0

        while not done.all():
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=True
            )
            obs, reward, done, info = env.step(action)
            episode_start = done
            episode_reward += float(np.array(reward).sum())
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        equity_curves.append(env.envs[0].get_equity_curve())
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'total_return': 0.0,
        'annual_return': 0.0,
        'annual_volatility': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'calmar_ratio': 0.0,
        'win_rate': 0.0,
        'profit_factor': 0.0
    }
    
    # Calculate performance metrics
    if equity_curves:
        combined_equity = pd.concat(equity_curves, axis=1).mean(axis=1)
        returns = combined_equity.pct_change().dropna()
        
        if len(returns) > 0:
            metrics.update({
                'total_return': (combined_equity.iloc[-1] - combined_equity.iloc[0]) / combined_equity.iloc[0],
                'annual_return': (1 + returns.mean()) ** 252 - 1,
                'annual_volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': (combined_equity / combined_equity.cummax() - 1).min(),
                'calmar_ratio': returns.mean() / abs((combined_equity / combined_equity.cummax() - 1).min()) * 252,
                'win_rate': len(returns[returns > 0]) / len(returns),
                'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
            })
    
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return metrics


def walk_forward_training(settings: Settings,
                         data_path: str,
                         features_path: str,
                         output_dir: str,
                         training_config: TrainingConfig) -> Dict[str, Any]:
    """
    Perform walk-forward training with validation.
    
    Args:
        settings: Configuration settings
        data_path: Path to training data
        features_path: Path to features
        output_dir: Output directory for results
        training_config: Training configuration
        
    Returns:
        Walk-forward results
    """
    logger.info("Starting walk-forward training...")
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Get date index
    date_index = pd.to_datetime(df.index.date).unique()
    
    # Walk-forward parameters
    train_days = settings.get("walkforward", "train_days", default=30)
    test_days = settings.get("walkforward", "test_days", default=10)
    embargo_minutes = settings.get("walkforward", "embargo_minutes", default=60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Walk-forward results
    wf_results = []
    
    # Generate walk-forward windows
    for i in range(0, len(date_index) - train_days - test_days + 1, test_days):
        train_start = pd.Timestamp(date_index[i], tz='America/New_York')
        train_end = pd.Timestamp(date_index[i + train_days - 1], tz='America/New_York') + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        # Apply embargo
        test_start_day = date_index[i + train_days]
        test_start = pd.Timestamp(test_start_day, tz='America/New_York') + pd.Timedelta(minutes=embargo_minutes)
        test_end = pd.Timestamp(date_index[i + train_days + test_days - 1], tz='America/New_York') + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        logger.info(f"Walk-forward fold {i//test_days + 1}:")
        logger.info(f"  Train: {train_start.date()} to {train_end.date()}")
        logger.info(f"  Test: {test_start.date()} to {test_end.date()}")
        
        # Split data
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        # Split features
        print("Training data index:", train_df.index)
        features_df = pd.read_parquet(features_path)
        print("Features data index:", features_df.index)
        train_features = features_df.loc[train_df.index]
        test_features = pd.read_parquet(features_path).loc[test_df.index]
        
        # Save fold data
        fold_dir = output_path / f"fold_{i//test_days + 1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_parquet(fold_dir / "train_data.parquet")
        test_df.to_parquet(fold_dir / "test_data.parquet")
        train_features.to_parquet(fold_dir / "train_features.parquet")
        test_features.to_parquet(fold_dir / "test_features.parquet")
        
        # Train model
        model_path = fold_dir / "model"
        train_ppo_lstm(
            settings=settings,
            data_path=str(fold_dir / "train_data.parquet"),
            features_path=str(fold_dir / "train_features.parquet"),
            model_path=str(model_path),
            training_config=training_config
        )
        
        # Load trained model
        model = RecurrentPPO.load(str(model_path))
        
        # Evaluate on test set with VecNormalize stats from the fold
        test_env = build_env(settings, str(fold_dir / "test_data.parquet"), str(fold_dir / "test_features.parquet"))
        vec_test_env = DummyVecEnv([lambda: test_env])
        # Load vecnormalize if exists next to the fold model
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            vn_path = fold_dir / "vecnormalize.pkl"
            if vn_path.exists():
                vec_test_env = VecNormalize.load(str(vn_path), vec_test_env)
                vec_test_env.training = False
        except Exception as e:
            logger.warning(f"VecNormalize load skipped in WF eval: {e}")
        test_metrics = evaluate_model(model, vec_test_env, num_episodes=3)
        
        # Save results
        fold_results = {
            'fold': i//test_days + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'test_metrics': test_metrics,
            'model_path': str(model_path)
        }
        
        wf_results.append(fold_results)
        
        # Save equity curve
        equity_curve = test_env.envs[0].get_equity_curve()
        equity_curve.to_csv(fold_dir / "equity_curve.csv")
        
        # Save trades
        trades = test_env.envs[0].get_trades()





def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train PPO-LSTM trading model")
    parser.add_argument("--config", default="configs/settings.yaml", help="Configuration file")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol for naming outputs (e.g., SPY, BBVA)")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--features", required=True, help="Path to features")
    parser.add_argument("--output", default=None, help="Output model path (defaults to models/<TICKER>_trained_model if --ticker set)")
    parser.add_argument("--walkforward", action="store_true", help="Perform walk-forward training")
    parser.add_argument("--wf-output", default="runs/walkforward", help="Walk-forward output directory")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained model")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Load settings
    settings = Settings.from_paths(args.config)
    
    # Training configuration
    training_config = TrainingConfig()
    training_config.learning_rate = float(settings.get("train", "learning_rate", default=1e-5))
    training_config.n_steps = int(settings.get("train", "n_steps", default=2048))
    training_config.batch_size = int(settings.get("train", "batch_size", default=2048))
    training_config.gamma = float(settings.get("train", "gamma", default=0.99))
    training_config.gae_lambda = float(settings.get("train", "gae_lambda", default=0.95))
    training_config.clip_range = float(settings.get("train", "clip_range", default=0.2))
    training_config.vf_coef = float(settings.get("train", "vf_coef", default=0.5))
    training_config.ent_coef = float(settings.get("train", "ent_coef", default=0.0))
    training_config.verbose = int(settings.get("train", "verbose", default=1))
    # Device selection (cpu/auto)
    try:
        training_config.device = str(settings.get("train", "device", default="auto"))
    except Exception:
        training_config.device = "auto"
    
    # Resolve default output path using ticker if not provided
    if args.output is None:
        default_out = f"models/{args.ticker}_trained_model" if args.ticker else "models/trained_model"
        args.output = default_out

    # Perform training
    if args.walkforward:
        # Walk-forward training
        results = walk_forward_training(
            settings=settings,
            data_path=args.data,
            features_path=args.features,
            output_dir=args.wf_output,
            training_config=training_config
        )
        
        logger.info("Walk-forward training completed successfully!")
        
    else:
        # Standard training
        model = train_ppo_lstm(
            settings=settings,
            data_path=args.data,
            features_path=args.features,
            model_path=args.output,
            training_config=training_config
        )
        
        logger.info("Training completed successfully!")
        
        # Evaluate model if requested
        if args.eval:
            env = build_env(settings, args.data, args.features)
            vec_env = DummyVecEnv([lambda: env])
            metrics = evaluate_model(model, vec_env, num_episodes=args.eval_episodes)
            
            logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
