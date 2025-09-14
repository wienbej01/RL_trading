# Dry Runs & Sanity Checks

Commands executed to validate wiring (token-safe):

1) Generate features (SPY sample)
```
PYTHONPATH=. venv/bin/python scripts/generate_spy_features.py \
  --data data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  --output results/oos_eval_dryrun/spy_features.parquet \
  --config configs/settings.yaml
```

2) Tiny train (sanity)
```
PYTHONPATH=. venv/bin/python -m src.rl.train \
  --config configs/settings.yaml \
  --data data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  --features results/oos_eval_dryrun/spy_features.parquet \
  --ticker SPY --output results/oos_eval_dryrun/model_tiny \
  --total-steps 500
```

3) OOS evaluator (1 window)
```
PYTHONPATH=. venv/bin/python scripts/oos_eval.py \
  --config configs/settings.yaml \
  --data data/cache/SPY_20240101_20240701_ohlcv_1min.parquet \
  --features results/oos_eval_dryrun/spy_features.parquet \
  --model results/oos_eval_dryrun/model_tiny \
  --test-tickers SPY \
  --windows 1 \
  --output results/oos_eval_dryrun
```

Artifacts
- Features parquet, tiny model, and `oos_results.json` under `results/oos_eval_dryrun/`.
