"""
Backtest evaluator for the RL trading system.

This module provides comprehensive backtesting evaluation capabilities
including performance analysis, risk metrics, and strategy comparison.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import calculate_performance_metrics, calculate_risk_metrics
from ..sim.env_intraday_rl import IntradayRLEnv
from ..sim.execution import ExecParams
from ..features.pipeline import FeaturePipeline

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100000.0
    risk_per_trade_frac: float = 0.02
    stop_loss_r_multiple: float = 1.0
    take_profit_r_multiple: float = 1.5
    max_daily_loss_r: float = 3.0
    commission_per_contract: float = 0.6
    spread_ticks: int = 1
    impact_bps: float = 0.5
    point_value: float = 5.0
    tick_value: float = 1.25
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    symbol: str = "MES"
    exchange: str = "CME"
    currency: str = "USD"
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    transaction_costs: bool = True
    slippage_model: str = "linear"  # linear, constant, percentage
    slippage_bps: float = 0.5
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    position_sizing: str = "fixed"  # fixed, percentage, kelly
    max_position_size: int = 10
    max_leverage: float = 2.0
    enable_short_selling: bool = True
    enable_margin: bool = True
    margin_requirement: float = 0.5


@dataclass
class BacktestResult:
    """Backtest result data structure."""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    upside_capture: float
    downside_capture: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float
    payoff_ratio: float
    # Directional trade stats
    num_long_trades: int
    num_short_trades: int
    avg_pnl_long: float
    avg_pnl_short: float
    avg_duration_min: float
    avg_duration_long_min: float
    avg_duration_short_min: float
    long_win_rate: float
    short_win_rate: float
    trades_per_day: float
    total_commission: float
    total_slippage: float
    total_transaction_costs: float
    final_equity: float
    peak_equity: float
    equity_curve: List[Dict[str, Any]]
    trade_history: List[Dict[str, Any]]
    position_history: List[Dict[str, Any]]
    risk_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    config: BacktestConfig


class BacktestEvaluator:
    """
    Comprehensive backtest evaluator.
    
    This class provides detailed backtesting evaluation with
    performance analysis, risk metrics, and strategy comparison.
    """
    
    def __init__(self, settings: Settings, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest evaluator.
        
        Args:
            settings: Configuration settings
            config: Backtest configuration
        """
        self.settings = settings
        self.config = config or self._load_config()
        
        # Backtest state
        self.equity_curve = []
        self.trade_history = []
        self.position_history = []
        self.risk_metrics = {}
        self.performance_metrics = {}
        
        # Results
        self.backtest_result = None
        
        # Configuration
        self.backtest_enabled = settings.get("evaluation", "backtest_enabled", default=True)
        
        logger.info("Backtest evaluator initialized")
    
    def _load_config(self) -> BacktestConfig:
        """
        Load backtest configuration from settings.
        
        Returns:
            Backtest configuration
        """
        config = BacktestConfig()
        
        config.initial_capital = self.settings.get("evaluation", "initial_capital", default=100000.0)
        config.risk_per_trade_frac = self.settings.get("evaluation", "risk_per_trade_frac", default=0.02)
        config.stop_loss_r_multiple = self.settings.get("evaluation", "stop_loss_r_multiple", default=1.0)
        config.take_profit_r_multiple = self.settings.get("evaluation", "take_profit_r_multiple", default=1.5)
        config.max_daily_loss_r = self.settings.get("evaluation", "max_daily_loss_r", default=3.0)
        config.commission_per_contract = self.settings.get("evaluation", "commission_per_contract", default=0.6)
        config.spread_ticks = self.settings.get("evaluation", "spread_ticks", default=1)
        config.impact_bps = self.settings.get("evaluation", "impact_bps", default=0.5)
        config.point_value = self.settings.get("evaluation", "point_value", default=5.0)
        config.tick_value = self.settings.get("evaluation", "tick_value", default=1.25)
        config.start_date = self.settings.get("evaluation", "start_date", default="2023-01-01")
        config.end_date = self.settings.get("evaluation", "end_date", default="2024-12-31")
        config.symbol = self.settings.get("evaluation", "symbol", default="MES")
        config.exchange = self.settings.get("evaluation", "exchange", default="CME")
        config.currency = self.settings.get("evaluation", "currency", default="USD")
        config.benchmark_symbol = self.settings.get("evaluation", "benchmark_symbol", default="SPY")
        config.risk_free_rate = self.settings.get("evaluation", "risk_free_rate", default=0.02)
        config.transaction_costs = self.settings.get("evaluation", "transaction_costs", default=True)
        config.slippage_model = self.settings.get("evaluation", "slippage_model", default="linear")
        config.slippage_bps = self.settings.get("evaluation", "slippage_bps", default=0.5)
        config.rebalance_frequency = self.settings.get("evaluation", "rebalance_frequency", default="daily")
        config.position_sizing = self.settings.get("evaluation", "position_sizing", default="fixed")
        config.max_position_size = self.settings.get("evaluation", "max_position_size", default=10)
        config.max_leverage = self.settings.get("evaluation", "max_leverage", default=2.0)
        config.enable_short_selling = self.settings.get("evaluation", "enable_short_selling", default=True)
        config.enable_margin = self.settings.get("evaluation", "enable_margin", default=True)
        config.margin_requirement = self.settings.get("evaluation", "margin_requirement", default=0.5)
        
        return config
    
    def run_backtest(self, model_path: str, data_path: str) -> BacktestResult:
        """
        Run backtest with trained model.
        
        Args:
            model_path: Path to trained model
            data_path: Path to backtest data
            
        Returns:
            Backtest result
        """
        if not self.backtest_enabled:
            logger.warning("Backtesting is disabled")
            return None
        
        try:
            logger.info("Starting backtest...")
            
            # Load model (support RecurrentPPO and PPO)
            model = None
            try:
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(model_path)
                logger.info("Loaded RecurrentPPO model")
                use_recurrent = True
            except Exception:
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
                logger.info("Loaded PPO model")
                use_recurrent = False

            # Try to load feature column metadata saved at training
            self._feature_list = None
            try:
                meta_path = Path(str(model_path) + "_features.json")
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    feats = meta.get('features')
                    if isinstance(feats, list) and feats:
                        self._feature_list = feats
                        logger.info(f"Loaded {len(self._feature_list)} training feature columns from metadata")
            except Exception as e:
                logger.warning(f"Feature metadata load skipped: {e}")
            
            # Load data
            df_all = pd.read_parquet(data_path)
            # Normalize to naive timestamps for env compatibility
            try:
                df_all = df_all.tz_localize(None)
            except Exception:
                # If already naive
                pass

            # Filter by date range if it intersects data; otherwise use full range
            try:
                cfg_start = pd.to_datetime(self.config.start_date)
                cfg_end = pd.to_datetime(self.config.end_date)
                data_min = df_all.index.min()
                data_max = df_all.index.max()
                # If configured window intersects data
                if (cfg_end >= data_min) and (cfg_start <= data_max):
                    start_date = max(cfg_start, data_min)
                    end_date = min(cfg_end, data_max)
                    df = df_all.loc[(df_all.index >= start_date) & (df_all.index <= end_date)]
                else:
                    logger.info("Config evaluation window outside data range; using full data range")
                    df = df_all
            except Exception:
                df = df_all

            if len(df) == 0:
                logger.warning("No rows after date filtering; using full dataset for backtest")
                df = df_all
            
            # Initialize environment
            env = self._create_environment(df)
            
            # Run backtest with VecNormalize if available (normalized obs to match training)
            logger.info("Running backtest...")
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                base_env = DummyVecEnv([lambda: env])
                vn_path = Path(model_path).with_name('vecnormalize.pkl')
                if vn_path.exists():
                    vec_env = VecNormalize.load(str(vn_path), base_env)
                    vec_env.training = False
                    logger.info(f"Loaded VecNormalize stats from {vn_path}")
                else:
                    vec_env = base_env
                    logger.info("VecNormalize stats not found; proceeding without normalization")
            except Exception as e:
                logger.warning(f"VecNormalize setup failed: {e}. Proceeding without it.")
                from stable_baselines3.common.vec_env import DummyVecEnv
                vec_env = DummyVecEnv([lambda: env])

            import numpy as _np
            obs = vec_env.reset()
            done = _np.zeros((vec_env.num_envs,), dtype=bool)
            step = 0
            state = None
            episode_start = _np.ones((vec_env.num_envs,), dtype=bool)
            action_counts = {"short": 0, "hold": 0, "long": 0}
            first_actions = []
            while not done.all():
                if use_recurrent:
                    action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                # SB3 returns shape (n_env,); tally mapping as per raw IntradayRLEnv: 0=short,1=hold,2=long
                try:
                    acts = [int(a) for a in action]
                except Exception:
                    acts = [int(action)]
                if step < 10:
                    first_actions.extend(acts)
                for a in acts:
                    if a == 0:
                        action_counts["short"] += 1
                    elif a == 1:
                        action_counts["hold"] += 1
                    elif a == 2:
                        action_counts["long"] += 1
                obs, reward, done, info = vec_env.step(_np.array(acts))
                episode_start = done
                step += 1
            logger.info(f"Backtest steps={step}, first_actions={first_actions}, action_counts={action_counts}")
            
            # Extract results
            self._extract_results(env)
            
            # Calculate metrics
            self._calculate_metrics()
            
            # Create backtest result
            self.backtest_result = self._create_backtest_result()
            
            logger.info("Backtest completed successfully")
            
            return self.backtest_result
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None
    
    def _create_environment(self, df: pd.DataFrame) -> IntradayRLEnv:
        """
        Create RL environment for backtest.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            RL environment
        """
        # Create feature pipeline configuration and transform data
        features_cfg = self.settings.get('features', default={})
        feature_pipeline = FeaturePipeline(features_cfg if isinstance(features_cfg, dict) else {})
        X = feature_pipeline.transform(df)

        # Align features to training feature list if available
        try:
            feat_list = getattr(self, '_feature_list', None)
            if isinstance(feat_list, list) and feat_list:
                # Add any missing columns as zeros, drop extras
                for col in feat_list:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[feat_list]
                logger.info(f"Aligned features to training list: {len(feat_list)} columns")
        except Exception as e:
            logger.warning(f"Feature alignment skipped: {e}")
        
        # Create execution parameters
        exec_params = ExecParams(
            tick_value=self.config.tick_value,
            spread_ticks=self.config.spread_ticks,
            impact_bps=self.config.impact_bps,
            commission_per_contract=self.config.commission_per_contract,
        )
        
        # Create risk configuration
        from ..sim.env_intraday_rl import RiskConfig
        risk_cfg = RiskConfig(
            risk_per_trade_frac=self.config.risk_per_trade_frac,
            stop_r_multiple=self.config.stop_loss_r_multiple,
            tp_r_multiple=self.config.take_profit_r_multiple,
            max_daily_loss_r=self.config.max_daily_loss_r,
        )
        
        # Create environment
        env = IntradayRLEnv(
            ohlcv=df[["open", "high", "low", "close", "volume"]],
            features=X,
            cash=self.config.initial_capital,
            exec_params=exec_params,
            risk_cfg=risk_cfg,
            point_value=self.config.point_value,
            config=self.settings.to_dict() if hasattr(self.settings, 'to_dict') else None,
        )
        
        return env
    
    def _extract_results(self, env: IntradayRLEnv):
        """
        Extract results from environment.
        
        Args:
            env: RL environment
        """
        # Extract equity curve
        if hasattr(env, 'equity_curve'):
            self.equity_curve = env.equity_curve
            # align timeline for plotting/report
            try:
                if hasattr(env, 'df') and hasattr(env.df, 'index'):
                    self.timeline = list(env.df.index[:len(self.equity_curve)])
            except Exception:
                self.timeline = []
        
        # Extract trade history (prefer rich 'trades' if available)
        if hasattr(env, 'trade_history') and env.trade_history:
            self.trade_history = env.trade_history
        elif hasattr(env, 'trades'):
            self.trade_history = env.trades
        
        # Extract position history
        if hasattr(env, 'position_history'):
            self.position_history = env.position_history
    
    def _calculate_metrics(self):
        """Calculate performance and risk metrics."""
        if not self.equity_curve:
            return
        
        # Handle equity as list of floats
        if isinstance(self.equity_curve, list) and self.equity_curve and not isinstance(self.equity_curve[0], dict):
            equity_series = pd.Series(self.equity_curve)
            returns = equity_series.pct_change().dropna()
        else:
            equity_df = pd.DataFrame(self.equity_curve)
            if 'timestamp' in equity_df.columns:
                equity_df.set_index('timestamp', inplace=True)
            returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate performance metrics
        self.performance_metrics = calculate_performance_metrics(returns)
        
        # Calculate risk metrics
        self.risk_metrics = calculate_risk_metrics(returns)
    
    def _create_backtest_result(self) -> BacktestResult:
        """
        Create backtest result.
        
        Returns:
            Backtest result
        """
        if not self.equity_curve:
            return None
        
        # Convert equity to series
        if isinstance(self.equity_curve, list) and self.equity_curve and not isinstance(self.equity_curve[0], dict):
            equity_series = pd.Series(self.equity_curve)
            returns = equity_series.pct_change().dropna()
            equity_df = equity_series.to_frame('equity')
        else:
            equity_df = pd.DataFrame(self.equity_curve)
            if 'timestamp' in equity_df.columns:
                equity_df.set_index('timestamp', inplace=True)
            returns = equity_df['equity'].pct_change().dropna()
        
        # Basic performance metrics
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annual_return - self.config.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        sortino_ratio = (annual_return - self.config.risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0.0
        
        # Drawdown analysis
        cumulative_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk
        returns_sorted = np.sort(returns)
        cvar_95 = returns_sorted[returns_sorted <= var_95].mean() if len(returns_sorted[returns_sorted <= var_95]) > 0 else var_95
        cvar_99 = returns_sorted[returns_sorted <= var_99].mean() if len(returns_sorted[returns_sorted <= var_99]) > 0 else var_99
        
        # Trade analysis
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # P&L statistics
        if self.trade_history:
            pnl_values = [t.get('pnl', 0) for t in self.trade_history]
            avg_win = np.mean([p for p in pnl_values if p > 0]) if winning_trades > 0 else 0.0
            avg_loss = np.mean([p for p in pnl_values if p < 0]) if losing_trades > 0 else 0.0
            largest_win = max(pnl_values) if pnl_values else 0.0
            largest_loss = min(pnl_values) if pnl_values else 0.0
            
            # Profit factor
            gross_profit = sum(p for p in pnl_values if p > 0)
            gross_loss = abs(sum(p for p in pnl_values if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            avg_win = 0.0
            avg_loss = 0.0
            largest_win = 0.0
            largest_loss = 0.0
            profit_factor = 0.0
        
        # Directional and consecutive analysis
        # Use only closed trades when available
        closed = [t for t in self.trade_history if t.get('action') == 'close'] if self.trade_history else []
        long_trades = [t for t in closed if t.get('direction') == 'long']
        short_trades = [t for t in closed if t.get('direction') == 'short']
        num_long_trades = len(long_trades)
        num_short_trades = len(short_trades)
        # Durations (minutes)
        def _avg(lst, key):
            vals = [float(t.get(key, 0.0)) for t in lst if key in t]
            return float(np.mean(vals)) if vals else 0.0
        avg_duration_min = _avg(closed, 'duration_min')
        avg_duration_long_min = _avg(long_trades, 'duration_min')
        avg_duration_short_min = _avg(short_trades, 'duration_min')
        # Avg PnL by side
        def _avg_pnl(lst):
            pnls = [float(t.get('pnl', 0.0)) for t in lst if 'pnl' in t]
            return float(np.mean(pnls)) if pnls else 0.0
        avg_pnl_long = _avg_pnl(long_trades)
        avg_pnl_short = _avg_pnl(short_trades)
        # Win rates by side
        def _winrate(lst):
            if not lst:
                return 0.0
            wins = sum(1 for t in lst if float(t.get('pnl', 0.0)) > 0.0)
            return wins / len(lst)
        long_win_rate = _winrate(long_trades)
        short_win_rate = _winrate(short_trades)
        # Trades per day
        trades_per_day = 0.0
        if closed:
            dates = pd.to_datetime([t.get('exit_time') or t.get('ts') for t in closed]).date
            if len(dates) > 0:
                per_day = pd.Series(1, index=pd.Index(dates)).groupby(level=0).sum()
                trades_per_day = float(per_day.mean()) if len(per_day) > 0 else 0.0

        # Consecutive analysis
        max_consecutive_wins = self._calculate_consecutive([t.get('pnl', 0) > 0 for t in self.trade_history])
        max_consecutive_losses = self._calculate_consecutive([t.get('pnl', 0) < 0 for t in self.trade_history])
        
        # Recovery factor
        total_net_profit = sum(t.get('pnl', 0) for t in self.trade_history)
        recovery_factor = total_net_profit / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Payoff ratio
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        # Transaction costs
        total_commission = sum(t.get('commission', 0) for t in self.trade_history)
        total_slippage = sum(t.get('slippage', 0) for t in self.trade_history)
        total_transaction_costs = total_commission + total_slippage
        
        # Final metrics
        final_equity = equity_df['equity'].iloc[-1]
        peak_equity = cumulative_max.iloc[-1]
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=1.0,  # Placeholder
            alpha=0.0,  # Placeholder
            information_ratio=0.0,  # Placeholder
            tracking_error=0.0,  # Placeholder
            upside_capture=0.0,  # Placeholder
            downside_capture=0.0,  # Placeholder
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=0.0,  # Placeholder
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            recovery_factor=recovery_factor,
            payoff_ratio=payoff_ratio,
            num_long_trades=num_long_trades,
            num_short_trades=num_short_trades,
            avg_pnl_long=avg_pnl_long,
            avg_pnl_short=avg_pnl_short,
            avg_duration_min=avg_duration_min,
            avg_duration_long_min=avg_duration_long_min,
            avg_duration_short_min=avg_duration_short_min,
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            trades_per_day=trades_per_day,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_transaction_costs=total_transaction_costs,
            final_equity=final_equity,
            peak_equity=peak_equity,
            equity_curve=self.equity_curve,
            trade_history=self.trade_history,
            position_history=self.position_history,
            risk_metrics=self.risk_metrics,
            performance_metrics=self.performance_metrics,
            config=self.config
        )
    
    def _calculate_consecutive(self, series: List[bool]) -> int:
        """
        Calculate maximum consecutive values.
        
        Args:
            series: Boolean series
            
        Returns:
            Maximum consecutive count
        """
        if not series:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def save_backtest_report(self, output_path: str):
        """
        Save backtest report.
        
        Args:
            output_path: Output path for report
        """
        if not self.backtest_result:
            logger.warning("No backtest result to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        if isinstance(self.equity_curve, list) and self.equity_curve and not isinstance(self.equity_curve[0], dict):
            eq_list = [
                {
                    'timestamp': (self.timeline[i].isoformat() if (self.timeline and i < len(self.timeline) and hasattr(self.timeline[i], 'isoformat')) else i),
                    'equity': float(v),
                    'pnl': 0.0
                }
                for i, v in enumerate(self.equity_curve)
            ]
        else:
            eq_list = [
                {
                    'timestamp': entry['timestamp'].isoformat() if isinstance(entry.get('timestamp', None), (datetime,)) else entry.get('timestamp'),
                    'equity': entry.get('equity'),
                    'pnl': entry.get('pnl', 0)
                }
                for entry in self.equity_curve
            ]

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'backtest_config': self.config.__dict__,
            'backtest_result': {
                'total_return': self.backtest_result.total_return,
                'annual_return': self.backtest_result.annual_return,
                'annual_volatility': self.backtest_result.annual_volatility,
                'sharpe_ratio': self.backtest_result.sharpe_ratio,
                'sortino_ratio': self.backtest_result.sortino_ratio,
                'calmar_ratio': self.backtest_result.calmar_ratio,
                'max_drawdown': self.backtest_result.max_drawdown,
                'current_drawdown': self.backtest_result.current_drawdown,
                'var_95': self.backtest_result.var_95,
                'var_99': self.backtest_result.var_99,
                'cvar_95': self.backtest_result.cvar_95,
                'cvar_99': self.backtest_result.cvar_99,
                'beta': self.backtest_result.beta,
                'alpha': self.backtest_result.alpha,
                'information_ratio': self.backtest_result.information_ratio,
                'tracking_error': self.backtest_result.tracking_error,
                'upside_capture': self.backtest_result.upside_capture,
                'downside_capture': self.backtest_result.downside_capture,
                'total_trades': self.backtest_result.total_trades,
                'winning_trades': self.backtest_result.winning_trades,
                'losing_trades': self.backtest_result.losing_trades,
                'win_rate': self.backtest_result.win_rate,
                'avg_win': self.backtest_result.avg_win,
                'avg_loss': self.backtest_result.avg_loss,
                'profit_factor': self.backtest_result.profit_factor,
                'largest_win': self.backtest_result.largest_win,
                'largest_loss': self.backtest_result.largest_loss,
                'avg_trade_duration': self.backtest_result.avg_trade_duration,
                'max_consecutive_wins': self.backtest_result.max_consecutive_wins,
                'max_consecutive_losses': self.backtest_result.max_consecutive_losses,
                'recovery_factor': self.backtest_result.recovery_factor,
                'payoff_ratio': self.backtest_result.payoff_ratio,
                'num_long_trades': self.backtest_result.num_long_trades,
                'num_short_trades': self.backtest_result.num_short_trades,
                'avg_pnl_long': self.backtest_result.avg_pnl_long,
                'avg_pnl_short': self.backtest_result.avg_pnl_short,
                'avg_duration_min': self.backtest_result.avg_duration_min,
                'avg_duration_long_min': self.backtest_result.avg_duration_long_min,
                'avg_duration_short_min': self.backtest_result.avg_duration_short_min,
                'long_win_rate': self.backtest_result.long_win_rate,
                'short_win_rate': self.backtest_result.short_win_rate,
                'trades_per_day': self.backtest_result.trades_per_day,
                'total_commission': self.backtest_result.total_commission,
                'total_slippage': self.backtest_result.total_slippage,
                'total_transaction_costs': self.backtest_result.total_transaction_costs,
                'final_equity': self.backtest_result.final_equity,
                'peak_equity': self.backtest_result.peak_equity
            },
            'equity_curve': eq_list,
            'trade_history': self.trade_history,
            'position_history': self.position_history,
            'risk_metrics': self.risk_metrics,
            'performance_metrics': self.performance_metrics
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Backtest report saved to {output_path}")
    
    def generate_backtest_plots(self, output_dir: str):
        """
        Generate backtest plots.
        
        Args:
            output_dir: Output directory for plots
        """
        if not self.backtest_result:
            logger.warning("No backtest result to plot")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        if isinstance(self.equity_curve, list) and self.equity_curve and not isinstance(self.equity_curve[0], dict):
            if self.timeline and len(self.timeline) == len(self.equity_curve):
                equity_df = pd.DataFrame({'equity': self.equity_curve}, index=pd.DatetimeIndex(self.timeline))
            else:
                equity_df = pd.DataFrame({'equity': self.equity_curve})
        else:
            equity_df = pd.DataFrame(self.equity_curve)
            if 'timestamp' in equity_df.columns:
                equity_df.set_index('timestamp', inplace=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Equity curve
        axes[0, 0].plot(equity_df.index, equity_df['equity'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True)
        
        # Drawdown curve
        cumulative_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cumulative_max) / cumulative_max
        axes[0, 1].fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(equity_df.index, drawdown)
        axes[0, 1].set_title('Drawdown Curve')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Returns distribution
        returns = equity_df['equity'].pct_change().dropna()
        axes[0, 2].hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[0, 2].axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
        axes[0, 2].set_title('Returns Distribution')
        axes[0, 2].set_xlabel('Returns')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].grid(True)
        
        # Trade P&L
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            axes[1, 0].hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Monthly returns heatmap
        monthly_returns = equity_df['equity'].resample('M').last().pct_change()
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) > 0:
            # Create year-month matrix
            monthly_returns.index = monthly_returns.index.to_period('M')
            monthly_returns_matrix = monthly_returns.unstack()
            
            sns.heatmap(monthly_returns_matrix, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Monthly Returns Heatmap')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Year')
        
        # Rolling metrics
        window = 252  # 1 year
        if len(returns) > window:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            axes[1, 2].plot(rolling_sharpe.index, rolling_sharpe)
            axes[1, 2].set_title(f'Rolling Sharpe Ratio ({window} days)')
            axes[1, 2].set_xlabel('Date')
            axes[1, 2].set_ylabel('Sharpe Ratio')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "backtest_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Backtest plots saved to {output_dir}")
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """
        Get backtest summary.
        
        Returns:
            Backtest summary dictionary
        """
        if not self.backtest_result:
            return {}
        
        return {
            'total_return': self.backtest_result.total_return,
            'annual_return': self.backtest_result.annual_return,
            'sharpe_ratio': self.backtest_result.sharpe_ratio,
            'max_drawdown': self.backtest_result.max_drawdown,
            'win_rate': self.backtest_result.win_rate,
            'total_trades': self.backtest_result.total_trades,
            'profit_factor': self.backtest_result.profit_factor,
            'final_equity': self.backtest_result.final_equity,
            'peak_equity': self.backtest_result.peak_equity,
            'total_commission': self.backtest_result.total_commission,
            'total_slippage': self.backtest_result.total_slippage,
            'total_transaction_costs': self.backtest_result.total_transaction_costs
        }
    
    def compare_strategies(self, other_results: List[BacktestResult]) -> Dict[str, Any]:
        """
        Compare this backtest result with other strategies.
        
        Args:
            other_results: List of other backtest results
            
        Returns:
            Comparison results
        """
        if not self.backtest_result:
            return {}
        
        comparison = {
            'primary_strategy': {
                'name': 'RL Trading Strategy',
                'total_return': self.backtest_result.total_return,
                'annual_return': self.backtest_result.annual_return,
                'sharpe_ratio': self.backtest_result.sharpe_ratio,
                'max_drawdown': self.backtest_result.max_drawdown,
                'win_rate': self.backtest_result.win_rate,
                'profit_factor': self.backtest_result.profit_factor
            },
            'comparison_strategies': [],
            'best_strategy': None,
            'worst_strategy': None,
            'rankings': {}
        }
        
        # Add comparison strategies
        for i, result in enumerate(other_results):
            strategy_name = f'Strategy {i+1}'
            strategy_data = {
                'name': strategy_name,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor
            }
            comparison['comparison_strategies'].append(strategy_data)
        
        # Determine best and worst strategies
        all_strategies = [comparison['primary_strategy']] + comparison['comparison_strategies']
        
        # Best by total return
        best_by_return = max(all_strategies, key=lambda x: x['total_return'])
        comparison['best_strategy'] = best_by_return['name']
        
        # Worst by max drawdown
        worst_by_drawdown = min(all_strategies, key=lambda x: x['max_drawdown'])
        comparison['worst_strategy'] = worst_by_drawdown['name']
        
        # Rankings
        rankings = {
            'total_return': sorted(all_strategies, key=lambda x: x['total_return'], reverse=True),
            'sharpe_ratio': sorted(all_strategies, key=lambda x: x['sharpe_ratio'], reverse=True),
            'max_drawdown': sorted(all_strategies, key=lambda x: x['max_drawdown']),
            'win_rate': sorted(all_strategies, key=lambda x: x['win_rate'], reverse=True),
            'profit_factor': sorted(all_strategies, key=lambda x: x['profit_factor'], reverse=True)
        }
        
        comparison['rankings'] = rankings
        
        return comparison
    
    def reset_backtest(self):
        """Reset backtest data."""
        self.equity_curve = []
        self.trade_history = []
        self.position_history = []
        self.risk_metrics = {}
        self.performance_metrics = {}
        self.backtest_result = None
        
        logger.info("Backtest data reset")
