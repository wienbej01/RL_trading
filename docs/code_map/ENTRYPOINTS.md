# Entry Points

## Overview
This document catalogs all the entry points into the RL trading system, including CLI scripts, Jupyter notebooks, and other ways users can interact with the codebase. Understanding these entry points is crucial for implementing the multi-ticker and reward overhaul.

## Entry Point Classification

### Primary Entry Points
Main scripts and interfaces that users will commonly interact with.

### Secondary Entry Points
Supporting scripts and utilities that provide additional functionality.

### Development Entry Points
Scripts and tools used during development and testing.

## Primary Entry Points

### Training Scripts

#### `scripts/train.py`
**Purpose**: Main training script for single-ticker RL models

**Usage**:
```bash
python scripts/train.py --config configs/settings.yaml --model-dir models/
```

**Key Features**:
- Loads configuration from YAML
- Creates data loader and feature pipeline
- Builds RL environment
- Trains PPO-LSTM model
- Saves model checkpoints
- Logs training metrics

**Configuration**:
- `--config`: Path to configuration file
- `--model-dir`: Directory to save models
- `--log-dir`: Directory for logs
- `--resume`: Resume training from checkpoint

**Dependencies**:
- `src/data/data_loader.py`
- `src/features/pipeline.py`
- `src/sim/env_intraday_rl.py`
- `src/rl/train.py`
- `src/rl/ppo_lstm_policy.py`

#### `scripts/train_multiticker.py` (Planned)
**Purpose**: Multi-ticker RL model training

**Usage**:
```bash
python scripts/train_multiticker.py --config configs/multiticker_settings.yaml --model-dir models/
```

**Key Features**:
- Multi-ticker data loading
- Cross-ticker feature engineering
- Multi-ticker RL environment
- Portfolio-level training
- Leave-One-Ticker-Out cross-validation

### Evaluation Scripts

#### `scripts/evaluate.py`
**Purpose**: Model evaluation and backtesting

**Usage**:
```bash
python scripts/evaluate.py --model models/ppo_lstm_model.zip --config configs/settings.yaml --output results/
```

**Key Features**:
- Loads trained model
- Runs backtest evaluation
- Calculates performance metrics
- Generates reports and visualizations
- Saves results to output directory

**Configuration**:
- `--model`: Path to trained model
- `--config`: Path to configuration file
- `--output`: Output directory for results
- `--start-date`: Evaluation start date
- `--end-date`: Evaluation end date

**Dependencies**:
- `src/evaluation/backtest_evaluator.py`
- `src/evaluation/metrics.py`
- `src/evaluation/visualization.py`

#### `scripts/evaluate_multiticker.py` (Planned)
**Purpose**: Multi-ticker model evaluation

**Usage**:
```bash
python scripts/evaluate_multiticker.py --model models/multiticker_model.zip --config configs/multiticker_settings.yaml --output results/
```

**Key Features**:
- Multi-ticker backtesting
- Portfolio performance analysis
- Ticker-specific attribution
- Cross-ticker correlation analysis

### Hyperparameter Optimization Scripts

#### `scripts/optimize.py`
**Purpose**: Hyperparameter optimization using Optuna

**Usage**:
```bash
python scripts/optimize.py --config configs/settings.yaml --study-name ppo_lstm_optimization --n-trials 100
```

**Key Features**:
- Optuna study management
- Hyperparameter search space definition
- Multi-objective optimization
- Parallel execution support
- Study visualization and analysis

**Configuration**:
- `--config`: Path to configuration file
- `--study-name`: Name of Optuna study
- `--n-trials`: Number of optimization trials
- `--storage`: Optuna storage URL
- `--n-jobs`: Number of parallel jobs

**Dependencies**:
- `src/rl/optuna_optimizer.py` (to be created)
- `src/rl/objective_functions.py` (to be created)

#### `scripts/optimize_multiticker.py` (Planned)
**Purpose**: Multi-ticker hyperparameter optimization

**Usage**:
```bash
python scripts/optimize_multiticker.py --config configs/multiticker_settings.yaml --study-name multiticker_optimization --n-trials 100
```

**Key Features**:
- Multi-ticker HPO search space
- Portfolio-level objectives
- Leave-One-Ticker-Out validation
- Multi-objective optimization

### Walk-Forward Optimization Scripts

#### `scripts/walkforward.py`
**Purpose**: Walk-Forward Optimization for time series validation

**Usage**:
```bash
python scripts/walkforward.py --config configs/settings.yaml --n-folds 5 --output results/
```

**Key Features**:
- Time series cross-validation
- Rolling window training
- Out-of-sample testing
- Fold aggregation and analysis
- WFO visualization

**Configuration**:
- `--config`: Path to configuration file
- `--n-folds`: Number of WFO folds
- `--train-size`: Training window size
- `--test-size`: Test window size
- `--output`: Output directory for results

**Dependencies**:
- `src/rl/walkforward.py`
- `src/evaluation/wfo_metrics.py` (to be created)

#### `scripts/walkforward_multiticker.py` (Planned)
**Purpose**: Multi-ticker Walk-Forward Optimization

**Usage**:
```bash
python scripts/walkforward_multiticker.py --config configs/multiticker_settings.yaml --n-folds 5 --output results/
```

**Key Features**:
- Leave-One-Ticker-Out cross-validation
- Multi-ticker WFO
- Embargo periods between folds
- Regime-aware fold splitting

## Secondary Entry Points

### Data Management Scripts

#### `scripts/download_data.py`
**Purpose**: Download market data from external sources

**Usage**:
```bash
python scripts/download_data.py --tickers AAPL MSFT GOOGL --start-date 2023-01-01 --end-date 2023-12-31 --source polygon
```

**Key Features**:
- Data download from Polygon/Databento
- Data validation and cleaning
- Partitioning by symbol and date
- Caching for performance

**Configuration**:
- `--tickers`: List of ticker symbols
- `--start-date`: Start date for data download
- `--end-date`: End date for data download
- `--source`: Data source (polygon/databento)
- `--output-dir`: Output directory for data

**Dependencies**:
- `src/data/downloaders/polygon_downloader.py` (to be created)
- `src/data/downloaders/databento_downloader.py` (to be created)

#### `scripts/validate_data.py`
**Purpose**: Validate and clean market data

**Usage**:
```bash
python scripts/validate_data.py --data-dir data/ --tickers AAPL MSFT GOOGL --output validation_report.json
```

**Key Features**:
- Data quality checks
- Missing data detection
- Outlier identification
- Validation report generation

### Feature Engineering Scripts

#### `scripts/extract_features.py`
**Purpose**: Extract and save features for training

**Usage**:
```bash
python scripts/extract_features.py --config configs/settings.yaml --tickers AAPL MSFT GOOGL --output features/
```

**Key Features**:
- Feature extraction from raw data
- Feature normalization and selection
- Feature caching
- Feature statistics generation

**Dependencies**:
- `src/features/pipeline.py`
- `src/features/feature_cache.py` (to be created)

#### `scripts/analyze_features.py`
**Purpose**: Analyze feature importance and correlations

**Usage**:
```bash
python scripts/analyze_features.py --features-dir features/ --output feature_analysis.html
```

**Key Features**:
- Feature correlation analysis
- Feature importance ranking
- Feature distribution analysis
- Interactive visualization

### Monitoring Scripts

#### `scripts/monitor_training.py`
**Purpose**: Monitor training progress in real-time

**Usage**:
```bash
python scripts/monitor_training.py --log-dir logs/ --port 8080
```

**Key Features**:
- Real-time training metrics dashboard
- TensorBoard integration
- Alert system for training issues
- Performance visualization

**Dependencies**:
- `src/monitoring/dashboard.py`
- `src/monitoring/alerts.py`

#### `scripts/monitor_performance.py`
**Purpose**: Monitor model performance in production

**Usage**:
```bash
python scripts/monitor_performance.py --model models/ppo_lstm_model.zip --config configs/settings.yaml
```

**Key Features**:
- Live performance tracking
- Performance degradation detection
- Risk limit monitoring
- Automated alerting

## Development Entry Points

### Testing Scripts

#### `scripts/run_tests.py`
**Purpose**: Run the test suite

**Usage**:
```bash
python scripts/run_tests.py --module src.rl --verbose
```

**Key Features**:
- Unit test execution
- Test coverage reporting
- Performance benchmarking
- Test result visualization

#### `scripts/test_multiticker.py` (Planned)
**Purpose**: Run multi-ticker specific tests

**Usage**:
```bash
python scripts/test_multiticker.py --module src.rl.multiticker --verbose
```

**Key Features**:
- Multi-ticker unit tests
- Integration tests
- Performance benchmarks
- Stress testing

### Benchmarking Scripts

#### `scripts/benchmark.py`
**Purpose**: Benchmark system performance

**Usage**:
```bash
python scripts/benchmark.py --config configs/settings.yaml --n-runs 10 --output benchmark_results.json
```

**Key Features**:
- Training speed benchmarking
- Inference latency measurement
- Memory usage profiling
- Scalability analysis

#### `scripts/benchmark_multiticker.py` (Planned)
**Purpose**: Multi-ticker performance benchmarking

**Usage**:
```bash
python scripts/benchmark_multiticker.py --config configs/multiticker_settings.yaml --n-runs 10 --output benchmark_results.json
```

**Key Features**:
- Multi-ticker scaling analysis
- Cross-ticker performance comparison
- Portfolio optimization benchmarks

### Documentation Scripts

#### `scripts/generate_docs.py`
**Purpose**: Generate API documentation

**Usage**:
```bash
python scripts/generate_docs.py --output docs/api/ --format html
```

**Key Features**:
- API documentation generation
- Code examples extraction
- Documentation validation
- Multi-format output support

## Jupyter Notebooks

### Analysis Notebooks

#### `notebooks/analysis/exploratory_analysis.ipynb`
**Purpose**: Exploratory data analysis and visualization

**Key Features**:
- Data loading and inspection
- Statistical analysis
- Interactive visualizations
- Feature correlation analysis

#### `notebooks/analysis/feature_analysis.ipynb`
**Purpose**: Feature engineering analysis

**Key Features**:
- Feature extraction examples
- Feature importance analysis
- Feature selection techniques
- Feature visualization

#### `notebooks/analysis/multiticker_analysis.ipynb` (Planned)
**Purpose**: Multi-ticker data analysis

**Key Features**:
- Multi-ticker data loading
- Cross-ticker correlation analysis
- Sector-based analysis
- Portfolio construction examples

### Training Notebooks

#### `notebooks/training/single_ticker_training.ipynb`
**Purpose**: Step-by-step single-ticker model training

**Key Features**:
- Environment setup
- Data loading and preprocessing
- Feature engineering
- Model training
- Results analysis

#### `notebooks/training/multiticker_training.ipynb` (Planned)
**Purpose**: Multi-ticker model training tutorial

**Key Features**:
- Multi-ticker data preparation
- Cross-ticker feature engineering
- Multi-ticker environment setup
- Portfolio-level training
- Performance analysis

### Evaluation Notebooks

#### `notebooks/evaluation/backtest_analysis.ipynb`
**Purpose**: Backtest result analysis

**Key Features**:
- Backtest result loading
- Performance metrics calculation
- Risk analysis
- Interactive visualizations

#### `notebooks/evaluation/multiticker_evaluation.ipynb` (Planned)
**Purpose**: Multi-ticker evaluation analysis

**Key Features**:
- Multi-ticker backtest results
- Portfolio performance analysis
- Ticker attribution analysis
- Risk decomposition

### Research Notebooks

#### `notebooks/research/reward_function_experiments.ipynb`
**Purpose**: Reward function experimentation

**Key Features**:
- Different reward function implementations
- Reward shaping experiments
- Performance comparison
- Sensitivity analysis

#### `notebooks/research/hyperparameter_experiments.ipynb`
**Purpose**: Hyperparameter sensitivity analysis

**Key Features**:
- Hyperparameter grid search
- Sensitivity visualization
- Optimal parameter identification
- Robustness analysis

## Configuration Files

### Main Configuration

#### `configs/settings.yaml`
**Purpose**: Main system configuration

**Key Sections**:
- Data source settings
- Feature engineering parameters
- Environment configuration
- Training hyperparameters
- Risk management rules
- Logging and monitoring settings

#### `configs/multiticker_settings.yaml` (Planned)
**Purpose**: Multi-ticker system configuration

**Key Sections**:
- Multi-ticker data settings
- Cross-ticker feature parameters
- Portfolio environment configuration
- Multi-ticker training settings
- Portfolio risk management

### Environment Configuration

#### `configs/environments/single_ticker_env.yaml`
**Purpose**: Single-ticker environment configuration

**Key Settings**:
- Reward function parameters
- Risk limits
- Position sizing rules
- Transaction costs

#### `configs/environments/multiticker_env.yaml` (Planned)
**Purpose**: Multi-ticker environment configuration

**Key Settings**:
- Portfolio reward parameters
- Cross-ticker risk limits
- Portfolio position sizing
- Multi-ticker transaction costs

### Training Configuration

#### `configs/training/ppo_lstm_config.yaml`
**Purpose**: PPO-LSTM training configuration

**Key Settings**:
- Learning parameters
- Network architecture
- Regularization settings
- Training schedule

#### `configs/training/multiticker_training_config.yaml` (Planned)
**Purpose**: Multi-ticker training configuration

**Key Settings**:
- Multi-ticker learning parameters
- Portfolio network architecture
- Cross-ticker regularization
- Multi-ticker training schedule

## Entry Point Dependencies

### Core Dependencies
- `src/utils/config_loader.py`: Configuration management
- `src/utils/logging.py`: Logging utilities
- `src/data/data_loader.py`: Data loading
- `src/features/pipeline.py`: Feature engineering
- `src/sim/env_intraday_rl.py`: RL environment
- `src/rl/train.py`: Training logic
- `src/rl/ppo_lstm_policy.py`: PPO-LSTM policy
- `src/evaluation/backtest_evaluator.py`: Evaluation logic

### Multi-Ticker Dependencies (To Be Implemented)
- `src/data/multiticker_data_loader.py`: Multi-ticker data loading
- `src/features/multiticker_pipeline.py`: Multi-ticker feature engineering
- `src/sim/multiticker_env.py`: Multi-ticker RL environment
- `src/rl/multiticker_trainer.py`: Multi-ticker training
- `src/evaluation/multiticker_evaluator.py`: Multi-ticker evaluation

### Monitoring Dependencies
- `src/monitoring/dashboard.py`: Monitoring dashboard
- `src/monitoring/alerts.py`: Alert system
- `src/monitoring/metrics_logger.py`: Metrics logging

### Optimization Dependencies
- `src/rl/optuna_optimizer.py`: Hyperparameter optimization
- `src/rl/walkforward.py`: Walk-Forward Optimization
- `src/rl/objective_functions.py`: Objective functions

## Entry Point Extensions for Multi-Ticker Support

### New Entry Points to Implement
1. `scripts/train_multiticker.py`: Multi-ticker training script
2. `scripts/evaluate_multiticker.py`: Multi-ticker evaluation script
3. `scripts/optimize_multiticker.py`: Multi-ticker hyperparameter optimization
4. `scripts/walkforward_multiticker.py`: Multi-ticker WFO
5. `scripts/benchmark_multiticker.py`: Multi-ticker benchmarking
6. `scripts/test_multiticker.py`: Multi-ticker testing

### New Notebooks to Create
1. `notebooks/analysis/multiticker_analysis.ipynb`: Multi-ticker analysis
2. `notebooks/training/multiticker_training.ipynb`: Multi-ticker training
3. `notebooks/evaluation/multiticker_evaluation.ipynb`: Multi-ticker evaluation
4. `notebooks/research/multiticker_research.ipynb`: Multi-ticker research

### New Configuration Files
1. `configs/multiticker_settings.yaml`: Multi-ticker main configuration
2. `configs/environments/multiticker_env.yaml`: Multi-ticker environment
3. `configs/training/multiticker_training_config.yaml`: Multi-ticker training

## Entry Point Usage Patterns

### Single Ticker Workflow
```bash
# 1. Download data
python scripts/download_data.py --tickers AAPL --start-date 2023-01-01 --end-date 2023-12-31

# 2. Extract features
python scripts/extract_features.py --config configs/settings.yaml --tickers AAPL

# 3. Train model
python scripts/train.py --config configs/settings.yaml --model-dir models/

# 4. Evaluate model
python scripts/evaluate.py --model models/ppo_lstm_model.zip --config configs/settings.yaml --output results/

# 5. Optimize hyperparameters
python scripts/optimize.py --config configs/settings.yaml --study-name ppo_optimization --n-trials 50

# 6. Walk-Forward Optimization
python scripts/walkforward.py --config configs/settings.yaml --n-folds 5 --output results/
```

### Multi-Ticker Workflow (Planned)
```bash
# 1. Download multi-ticker data
python scripts/download_data.py --tickers AAPL MSFT GOOGL --start-date 2023-01-01 --end-date 2023-12-31

# 2. Extract multi-ticker features
python scripts/extract_features.py --config configs/multiticker_settings.yaml --tickers AAPL MSFT GOOGL

# 3. Train multi-ticker model
python scripts/train_multiticker.py --config configs/multiticker_settings.yaml --model-dir models/

# 4. Evaluate multi-ticker model
python scripts/evaluate_multiticker.py --model models/multiticker_model.zip --config configs/multiticker_settings.yaml --output results/

# 5. Optimize multi-ticker hyperparameters
python scripts/optimize_multiticker.py --config configs/multiticker_settings.yaml --study-name multiticker_optimization --n-trials 50

# 6. Multi-ticker Walk-Forward Optimization
python scripts/walkforward_multiticker.py --config configs/multiticker_settings.yaml --n-folds 5 --output results/