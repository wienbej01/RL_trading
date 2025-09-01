Intraday Trading System User Guide

## Overview

The RL Intraday Trading System is a comprehensive reinforcement learning-based trading platform designed for intraday futures trading, specifically optimized for E-mini S&P 500 (MES) contracts. The system integrates advanced RL algorithms, professional risk management, and real-time monitoring capabilities.

## 1. Quick Start

### System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 50GB free space (100GB recommended for data)
- **Network**: Stable internet connection for market data

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/rl-intraday.git
    cd rl-intraday
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set PYTHONPATH:**
    ```bash
    export PYTHONPATH=$(pwd)
    ```
    (Note: `$(pwd)` assumes you are in the `rl-intraday` directory)
5.  **Sanity Checks:**
    ```bash
    python -V
    pip -V
    ```

## 2. Data Setup

### Data Sources

The system primarily uses Polygon.io for historical and real-time market data.

### Data Location

Raw data from Polygon.io is stored in `data/polygon/historical/` and is partitioned by symbol, year, month, and day.

### Required Environment Variables

*   **Polygon.io API Key:**
    ```bash
    export POLYGON_API_KEY="your_polygon_api_key_here"
    ```
    Get your API key from [Polygon.io](https://polygon.io/)

*   **Databento API (Alternative data source):**
    ```bash
    export DATABENTO_API_KEY="your_databento_api_key_here"
    ```

### Data Aggregation

The training and backtesting scripts expect a single aggregated data file. You need to aggregate the partitioned data using the `aggregate_spy_data.py` script.

**Command:**
```bash
python scripts/aggregate_spy_data.py --start-date 2024-01-01 --end-date 2025-06-30
```
This will create `data/raw/spy_1min.parquet`.

### Feature Generation

After aggregating the data, generate features using the `generate_spy_features.py` script.

**Command:**
```bash
python scripts/generate_spy_features.py --data data/raw/spy_1min.parquet --output data/features/SPY_features.parquet
```
This will create `data/features/SPY_features.parquet`.

## 3. Training

The training process uses a PPO agent with an LSTM policy.

### Training Command Example (SPY 2024-01-01 → 2025-06-30)

```bash
PYTHONPATH=$(pwd) python -m src.rl.train \
  --config configs/settings.yaml \
  --data data/raw/spy_1min.parquet \
  --features data/features/SPY_features.parquet \
  --output models/trained_model
```

### Output Artifacts

*   **Trained Model:** The trained model is saved as a `.zip` file (e.g., `trained_model.zip`) in the `models/` directory.
*   **Tensorboard Logs:** Training progress and metrics are logged to Tensorboard, located in `runs/tensorboard/`.

## 4. Backtest

Backtesting allows you to evaluate the performance of your trained model on historical data.

### Backtest Command Example

```bash
PYTHONPATH=$(pwd) python examples/polygon_rl_backtest_example.py \
  --symbol SPY \
  --start-date 2024-01-01 \
  --end-date 2025-06-30 \
  --episodes 10 \
  --model-path models/trained_model.zip \
  --plot
```

### Output Reports

*   **Console Output:** Detailed episode-by-episode results and overall performance metrics are printed to the console.
*   **Plot:** If the `--plot` flag is used, a plot of the backtest results is saved as `backtest_results_SPY.png` in the root directory of the project.

## 5. Troubleshooting

Here are some common issues and their solutions:

1.  **`ModuleNotFoundError: No module named 'src'`**
    *   **Cause:** The `rl-intraday` directory is not in your `PYTHONPATH`.
    *   **Fix:** Run `export PYTHONPATH=$(pwd)` from the `rl-intraday` directory.

2.  **`ValueError: No data files found` (during aggregation)**
    *   **Cause:** The `aggregate_spy_data.py` script cannot find the partitioned data files. This might be due to incorrect relative paths or missing data.
    *   **Fix:** Ensure you are running the script from the root of the `rl-intraday` directory and that data has been collected using `scripts/collect_polygon_data.py`. Verify the `base_path` in `aggregate_spy_data.py` is correct.

3.  **`TypeError: PPOLSTMPolicy.__init__() got an unexpected keyword argument 'use_sde'`**
    *   **Cause:** The `PPOLSTMPolicy` class was not fully compatible with `stable-baselines3`'s policy interface.
    *   **Fix:** This has been addressed in the latest version of the codebase. Ensure your repository is up-to-date.

4.  **`KeyError: 'episode_rewards'` (during training callback)**
    *   **Cause:** A training callback was trying to access `episode_rewards` from `self.locals` before it was available.
    *   **Fix:** This has been addressed in the latest version of the codebase. Ensure your repository is up-to-date.

5.  **`ImportError: Trying to log data to tensorboard but tensorboard is not installed.`**
    *   **Cause:** The `tensorboard` package is missing.
    *   **Fix:** Run `pip install tensorboard`.

6.  **`WARNING - OHLCV (...) and features (...) have different lengths`**
    *   **Cause:** Mismatch between raw data and generated features, potentially due to data gaps or processing issues.
    *   **Fix:** Investigate data quality and feature generation logic. Ensure `generate_spy_features.py` is run on the correct aggregated data.

7.  **`WARNING - No config attribute found or config is empty - using default settings`**
    *   **Cause:** The `settings.yaml` file is not being loaded correctly or is empty.
    *   **Fix:** Verify the path to `configs/settings.yaml` in your commands and ensure the file contains valid YAML.

## 6. Repro Tips

*   **Deterministic Seeds:** For reproducible training runs, ensure `seed` is set in `configs/settings.yaml` and passed to the training script.
*   **Config Pinning:** Always specify the `--config` file explicitly for training and backtesting to ensure consistent settings.
*   **Recommended Log Levels:** Set `logging.level: INFO` in `configs/settings.yaml` for general operation. Use `DEBUG` for detailed troubleshooting.

## 7. FAQ

*   **Q: How do I collect more historical data?**
    *   A: Use `python scripts/collect_polygon_data.py --symbols SYMBOL --start-date YYYY-MM-DD --end-date YYYY-MM-DD`.
*   **Q: Can I use a different data source?**
    *   A: The system is designed to be extensible. You would need to implement a new data loader in `src/data/` and potentially adjust the feature pipeline.
*   **Q: Why are my backtest returns zero/negative?**
    *   A: This indicates the model is not learning an effective trading strategy. This is a common challenge in RL for trading. Focus on hyperparameter tuning, feature engineering, and potentially exploring different RL algorithms or environment reward structures.
*   **Q: How do I view Tensorboard logs?**
    *   A: Navigate to the `rl-intraday` directory and run `tensorboard --logdir runs/tensorboard`. Then open your browser to the address provided (usually `http://localhost:6006`).