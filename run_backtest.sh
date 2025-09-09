#!/bin/bash

# Set PYTHONPATH to the project root
export PYTHONPATH=/home/jacobw/RL_trading/rl-intraday

# Activate the virtual environment
source /home/jacobw/RL_trading/rl-intraday/venv/bin/activate

# Run the backtest
python /home/jacobw/RL_trading/rl-intraday/scripts/walkforward_train_eval.py \
    --config /home/jacobw/RL_trading/rl-intraday/configs/settings.yaml \
    --data /home/jacobw/RL_trading/rl-intraday/data/raw/BBVA_1min.parquet \
    --features /home/jacobw/RL_trading/rl-intraday/data/features/BBVA_features.parquet \
    --output /home/jacobw/RL_trading/rl-intraday/runs/wf_bbva_debug \
    --train-days 5 --test-days 2
