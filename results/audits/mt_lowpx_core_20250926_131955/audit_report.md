# RL Run Audit — mt_lowpx_core

## Overview
- Commit: 57b1ba42f03b0647fad3e161207d746e68a4b491
- Libs: {'python': '3.13.3', 'stable_baselines3': '2.7.0', 'torch': '2.8.0+cu128', 'numpy': '2.3.2'}
- Seed: 123

## Data & Splits
- Train tickers (10): ['F', 'CCL', 'AAL', 'NCLH', 'T', 'PBR', 'RIVN', 'LCID', 'SNAP', 'KDP']
- Backtest tickers (1): ['SNAP']
- Train: {'start': '2024-04-01', 'end': '2024-08-15', 'wall_days': 137}
- Test: {'start': '2024-08-16', 'end': '2024-09-30', 'wall_days': 46}
- Portfolio env: True
- Bar freq: None

## Features & Normalization
- Features used (52): ['rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'macd_line', 'adx', 'spread', 'price_impact', 'bar_imbalance', 'effort_result', 'rvol', 'delta_vol', 'dist_pdh_bp', 'dist_pdc_bp', 'imbalance_persist', 'direction_ema', 'intrabar_vol', 'eq_high_flag', 'eq_low_flag', 'dist_eq_high', 'dist_eq_low', 'dist_cdo_bp', 'dist_pdo_bp', 'gap_open_prev_close', 'dist_pp_bp', 'dist_r1_bp', 'dist_s1_bp', 'dist_r2_bp', 'dist_s2_bp', 'dist_rollmax_bp', 'dist_rollmin_bp', 'dist_session_vwap', 'dist_cdo_bp_atr', 'dist_pdo_bp_atr', 'dist_pp_bp_atr', 'dist_r1_bp_atr', 'dist_s1_bp_atr', 'dist_r2_bp_atr', 'dist_s2_bp_atr', 'dist_rollmax_bp_atr', 'dist_rollmin_bp_atr', 'dist_session_vwap_atr', 'swing_high_flag', 'swing_low_flag', 'dist_last_swing_high', 'dist_last_swing_low', 'bos_up', 'bos_down', 'regime_trend', 'regime_trend_sign']...
- VecNormalize: obs=True, reward=True, path=results/mt_lowpx_core/models/checkpoints/vecnorm.pkl

## Strategy & Execution
- Execution: {}
- Entry/exit: {'force_open_epsilon': 0.05, 'force_warmup_frac': 0.25, 'max_entries_per_day': 2, 'max_trades_per_hour': 3}

## PPO Hyperparameters
- Params: {'n_steps': 2048, 'batch_size': 2048, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.015, 'vf_coef': 0.7, 'max_grad_norm': 0.5, 'target_kl': 0.01, 'lr_schedule': 'linear', 'lr_start': 0.00015, 'lr_end': 1e-05, 'clip_schedule': 'linear', 'clip_start': 0.15, 'clip_end': 0.12, 'clip_range_vf': 0.5, 'policy_kwargs': {'net_arch': {'pi': [256, 256], 'vf': [256, 256]}, 'activation_fn': 'ReLU', 'ortho_init': True}}
- Callbacks: {'kl_stop': {'target_kl': 0.01}, 'adaptive_lr_by_kl': {'low': 0.003, 'high': 0.01}, 'live_lr_bump': {'flag': '.lr_bump'}}

## Backtest Results
- Metrics: {'mean_reward': -42.57099743152503, 'std_reward': 0.0, 'mean_length': 9452.0, 'std_length': 0.0, 'max_reward': -42.57099743152503, 'min_reward': -42.57099743152503, 'total_return': -0.00042571000001349603, 'annual_return': -1.135037514410353e-05, 'annual_volatility': 6.0692008626494864e-05, 'sharpe_ratio': -0.18701698276399784, 'max_drawdown': -0.0006350394953201777, 'calmar_ratio': -0.01787359586870016, 'win_rate': 0.4473684210526316, 'profit_factor': 1.377183695076738, 'sortino_ratio': -0.04828839965011895, 'total_trades': 38, 'long_trades': 2, 'short_trades': 36, 'avg_duration_minutes': 10.552631578947368, 'avg_duration_minutes_long': 10.0, 'avg_duration_minutes_short': 10.583333333333334, 'avg_pnl': 0.37500000000000955, 'avg_pnl_long': -1.4649999999999608, 'avg_pnl_short': 0.4772222222222301, 'daily_sharpe_proxy': -4.640386161487043, 'intraday_max_drawdown': -0.0006350394953201777}

## Derived Ratios
- {'train_test_day_ratio': 2.9782608695652173, 'n_train_tickers': 10, 'n_backtest_tickers': 1}

## Likely Failure Modes
- PORTFOLIO_IN_NAME_ONLY: Multiple train tickers requested but backtest produced a single-ticker output.