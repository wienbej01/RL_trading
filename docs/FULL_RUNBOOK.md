# Full Runbook

1) Generate features (adjust paths as needed)
```
make gen_features DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  FEATURES=results/oos_eval_demo/spy_features.parquet CONFIG=configs/settings.yaml
```

2) Train RL model (single-asset example)
```
make rl_train DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  FEATURES=results/oos_eval_demo/spy_features.parquet \
  MODEL=results/oos_eval_demo/model CONFIG=configs/settings.yaml
```

3) OOS backtest (held-out tickers/windows)
```
make backtest_oos DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  FEATURES=results/oos_eval_demo/spy_features.parquet \
  MODEL=results/oos_eval_demo/model \
  OOS_OUT=results/oos_eval_demo \
  CONFIG=configs/settings.yaml TICKERS="SPY"
```

4) Tuning sweep (small number of trials)
```
make tune DATA=data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  FEATURES=results/oos_eval_demo/spy_features.parquet \
  CONFIG=configs/settings.yaml
```

Notes
- Multi-ticker end-to-end pipeline is available via `scripts/run_multiticker_pipeline.py`.
- For IBKR paper trading, see `scripts/run_paper_trading.py` and `src/trading/` modules.
