# Minimal wiring for existing entrypoints

PY := venv/bin/python
PP := PYTHONPATH=.

# Defaults (override on CLI: make rl_train DATA=... FEATURES=...)
CONFIG ?= configs/settings.yaml
DATA ?= data/cache/SPY_20240101_20240701_ohlcv_1min.parquet
FEATURES ?= results/oos_eval_dryrun/spy_features.parquet
MODEL ?= results/model_demo
OOS_OUT ?= results/oos_eval_demo
TICKERS ?= SPY AAPL MSFT

.PHONY: tickers_train tickers_test gen_features rl_train backtest_oos tune

tickers_train:
	@echo "Training tickers come from the CLI or config depending on the path."
	@echo "Single-asset: use --data/--features (ticker implied)."
	@echo "Multi-ticker: scripts/run_multiticker_pipeline.py --tickers $(TICKERS) ..."

tickers_test:
	@echo "Backtest/OOS test tickers: pass --test-tickers (pipeline/oos_eval)."
	@echo "Example: scripts/oos_eval.py --test-tickers $(TICKERS) ..."

gen_features:
	$(PP) $(PY) scripts/generate_spy_features.py --data $(DATA) --output $(FEATURES) --config $(CONFIG)

rl_train:
	$(PP) $(PY) -m src.rl.train --config $(CONFIG) --data $(DATA) --features $(FEATURES) --ticker DEMO --output $(MODEL) --total-steps 5000

backtest_oos:
	$(PP) $(PY) scripts/oos_eval.py --config $(CONFIG) --data $(DATA) --features $(FEATURES) --model $(MODEL) --test-tickers $(TICKERS) --output $(OOS_OUT)

tune:
	$(PP) $(PY) scripts/tune_orchestrator.py --config $(CONFIG) --data $(DATA) --features $(FEATURES) --output results/tuning_run --test-tickers $(TICKERS) --trials 4 --total-steps 5000

