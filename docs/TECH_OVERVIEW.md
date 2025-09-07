# Technical Overview

This document provides a technical overview of the RL Intraday Trading System.

## 1. System Intent & Boundaries

The system is a reinforcement learning-based trading platform for intraday futures trading, specifically targeting E-mini S&P 500 (MES) contracts. It is designed to be a complete end-to-end solution, from data collection and feature engineering to model training, backtesting, and paper trading.

**In Scope:**

*   Data collection from Polygon.io.
*   Feature engineering pipeline.
*   RL model training (PPO with LSTM).
*   Walk-forward validation.
*   Backtesting and evaluation.
*   Paper trading with Interactive Brokers.

**Out of Scope:**

*   Live trading with real money.
*   Support for other data providers out-of-the-box (though the system is extensible).
*   Advanced order execution logic (e.g., TWAP/VWAP execution).

## 2. Architecture Overview

The system is composed of several interconnected modules that handle different aspects of the trading pipeline.

```
[ Polygon.io ] -> [ Data Loader ] -> [ Feature Pipeline ] -> [ RL Environment ] -> [ PPO Model ]
      ^                   |
      |                   v
      +-----------[ IBKR Gateway ] <- [ Execution Engine ] <- [ Backtest/Paper Trading ] <-+
```

*   **Data Loader:** Responsible for fetching historical and real-time data from Polygon.io. See `src/data/data_loader.py`.
*   **Feature Pipeline:** Generates a rich set of features from the raw data, including technical indicators and microstructure features. See `src/features/pipeline.py`.
*   **RL Environment:** A custom OpenAI Gym-compatible environment for training the RL agent. See `src/sim/env_intraday_rl.py`.
*   **PPO Model:** The reinforcement learning agent, based on Proximal Policy Optimization (PPO) with an LSTM policy. See `src/rl/ppo_lstm_policy.py`.
*   **Execution Engine:** Simulates order execution for backtesting and paper trading. See `src/sim/execution.py`.
*   **Backtest/Paper Trading:** The main entry points for running simulations and paper trading sessions. See `examples/polygon_rl_backtest_example.py` and `scripts/run_paper_trading.py`.

## 3. Data Pipeline

*   **Sources:** The primary data source is Polygon.io, providing historical and real-time market data.
*   **Schema:** The data is stored in Parquet files, partitioned by symbol, year, month, and day. The schema includes OHLCV data, as well as VWAP and transaction counts from Polygon.
*   **Caches:** The system uses a caching mechanism to store pre-processed data and features, reducing the need for repeated computations. The cache is located at `data/cache/`.
*   **Artifacts:**
    *   Raw data is stored in `data/polygon/historical/`.
    *   Aggregated data is stored in `data/raw/`.
    *   Generated features are stored in `data/features/`.
    *   Trained models are saved in `models/`.

## 4. Training / Backtest Flow

### Training

The training process is initiated via the `src/rl/train.py` script.

**CLI Example:**

```bash
PYTHONPATH=/home/jacobw/RL_trading/rl-intraday python -m src.rl.train \
  --config /home/jacobw/RL_trading/rl-intraday/configs/settings.yaml \
  --data data/raw/spy_1min.parquet \
  --features data/features/SPY_features.parquet \
  --output /home/jacobw/RL_trading/rl-intraday/models/trained_model
```

**Key Configuration (`configs/settings.yaml`):**

*   `train`: Hyperparameters for the PPO agent.
*   `features`: Configuration for the feature engineering pipeline.
*   `walkforward`: Parameters for walk-forward validation.

### Backtesting & Evaluation

Use the consolidated report script, which builds features, aligns them to the
training feature list, and applies observation normalization using the saved
VecNormalize statistics (Gymnasium‑compatible):

```bash
PYTHONPATH=$(pwd) venv/bin/python scripts/generate_performance_report.py \
  --model runs/wf_bbva_debug/fold_179/model.zip \
  --data data/raw/BBVA_20250101_20250630_1min.parquet \
  --start 2025-01-01 --end 2025-06-30 \
  --out reports/BBVA_20250101_20250630_norm.json
```

Notes:
- The evaluator aligns features to the training list found in
  `model_features.json` or `model.zip_features.json` and fills NaNs using
  `reindex → ffill/bfill → fillna(0)` to avoid feature collapse.
- For normalization parity with training, it loads `vecnormalize.pkl` next to
  the model and normalizes observations on reset/step.

## 5. Environments

*   **Virtual Environment:** A local Python virtual environment is located at `/home/jacobw/RL_trading/rl-intraday/venv`.
*   **PYTHONPATH:** The `rl-intraday` directory must be added to the `PYTHONPATH` to ensure that the modules can be found.
*   **Dependencies:** Key dependencies include `sb3-contrib`, `stable-baselines3`, `torch`, `pandas`, `numpy`, and `polygon-api-client`. A full list can be found in `requirements.txt`.
*   **Gymnasium Compatibility:** The project uses a Gymnasium‑safe `DummyVecEnv`
    wrapper that returns 5‑tuple steps. The evaluator applies manual
    VecNormalize to remain compatible with Gymnasium envs.

## 6. Observability

*   **Logs:** Logs are generated by the various scripts and are printed to the console. The `end_to_end_test.log` file in the root directory contains detailed logs of the test runs.
*   **Metrics:** The backtesting script outputs a summary of performance metrics, including total return, win rate, and Sharpe ratio.
*   **Reports:** The backtest report saves JSON summaries and a `backtest_results.png` plot in the `reports/` directory.
*   **Tensorboard:** The training script logs data to Tensorboard, which can be found in the `runs/tensorboard` directory.

## 7. Data Completeness & Risks

1.  **PYTHONPATH:** Ensure `rl-intraday` is in `PYTHONPATH` when invoking modules.
2.  **Data Aggregation:** Use `scripts/fetch_polygon_range.py` for quick symbol ranges, or
    `scripts/collect_polygon_us_stocks.py` + `scripts/aggregate_us_stock_data.py` for
    high‑throughput, partitioned downloads.
3.  **SMT/VIX Requirements:** When `features.smt.enabled: true`, ensure
    `data/raw/SPY_1min.parquet` and `data/raw/QQQ_1min.parquet` cover the evaluation range.
    When `features.vix.enabled: true`, ensure `data/external/vix.parquet` covers the range.
4.  **Normalization Parity:** Evaluation loads `vecnormalize.pkl` and normalizes obs manually.
    Missing or mismatched stats can degrade policy behavior.

## 8. Glossary

*   **PPO:** Proximal Policy Optimization, the reinforcement learning algorithm used in this project.
*   **LSTM:** Long Short-Term Memory, a type of recurrent neural network used in the policy.
*   **Walk-Forward Validation:** A method for testing the performance of a trading strategy on historical data that avoids lookahead bias.
*   **Sharpe Ratio:** A measure of risk-adjusted return.
*   **Max Drawdown:** The maximum observed loss from a peak to a trough of a portfolio.
*   **OHLCV:** Open, High, Low, Close, Volume - the standard data points for a financial instrument.
*   **VWAP:** Volume-Weighted Average Price.
*   **RTH:** Regular Trading Hours.
*   **IBKR:** Interactive Brokers, the supported broker for paper and live trading.
*   **TWS:** Trader Workstation, the trading platform from Interactive Brokers.
