# Public and Semi-Public APIs

## Overview
This document catalogs the public and semi-public APIs, classes, and functions in the RL trading system. Understanding these interfaces is essential for implementing the multi-ticker and reward overhaul.

## API Classification

### Public APIs
Intended for external users and documented for general use. These APIs have stable interfaces and backward compatibility guarantees.

### Semi-Public APIs
Internal APIs that may be used by advanced users or for extending the system. These interfaces may change between versions but are generally stable within a major version.

### Internal APIs
Private implementation details that should not be relied upon by external code. These interfaces can change at any time.

## Data Module APIs

### UnifiedDataLoader (Public)

```python
class UnifiedDataLoader:
    """
    Unified data loader for Polygon and Databento market data.
    
    This class provides a consistent interface for loading market data
    from different sources, with support for caching, validation, and
    preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary containing data source settings
        """
        pass
    
    def load_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        data_type: str = "ohlcv"
    ) -> pd.DataFrame:
        """
        Load market data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: Type of data to load ("ohlcv", "quotes", "trades")
            
        Returns:
            DataFrame with market data
        """
        pass
    
    def load_multiple_tickers(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load market data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        pass
    
    def get_ticker_metadata(self, ticker: str) -> Dict[str, Any]:
        """
        Get metadata for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker metadata
        """
        pass
```

## Feature Module APIs

### FeaturePipeline (Public)

```python
class FeaturePipeline:
    """
    Feature engineering pipeline for the RL trading system.
    
    This class provides a unified interface for extracting and transforming
    features from market data, including technical indicators, microstructure
    features, and time-based features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature pipeline.
        
        Args:
            config: Configuration dictionary specifying which features to extract
        """
        pass
    
    def fit(self, data: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit the feature pipeline on data.
        
        Args:
            data: Data to fit on
            
        Returns:
            Self
        """
        pass
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed features
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit pipeline and transform data.
        
        Args:
            data: Training data
            
        Returns:
            Transformed features
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names
        """
        pass
```

## RL Module APIs

### RLTrainer (Public)

```python
class RLTrainer:
    """
    RL trainer for model training and management.
    
    This class provides a high-level interface for training RL models,
    including PPO-LSTM training, walk-forward optimization, and model evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL trainer.
        
        Args:
            config: Training configuration
        """
        pass
    
    def train(
        self, 
        env: gym.Env, 
        model_config: Dict[str, Any] = None
    ) -> 'RLTrainer':
        """
        Train an RL model.
        
        Args:
            env: RL environment
            model_config: Model configuration overrides
            
        Returns:
            Self
        """
        pass
    
    def walk_forward_training(
        self, 
        env_builder: Callable,
        train_periods: List[Tuple[str, str]],
        test_periods: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Perform walk-forward training.
        
        Args:
            env_builder: Function to create environments
            train_periods: List of (start_date, end_date) for training
            test_periods: List of (start_date, end_date) for testing
            
        Returns:
            Dictionary with training results
        """
        pass
    
    def evaluate_model(
        self, 
        env: gym.Env, 
        n_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            env: RL environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    def save_model(self, path: str):
        """
        Save trained model.
        
        Args:
            path: Path to save model
        """
        pass
    
    def load_model(self, path: str):
        """
        Load trained model.
        
        Args:
            path: Path to load model from
        """
        pass
```

### PPOLSTMPolicy (Semi-Public)

```python
class PPOLSTMPolicy(ActorCriticPolicy):
    """
    PPO-LSTM policy with custom feature extraction.
    
    This policy extends the standard ActorCriticPolicy with LSTM-based
    feature extraction for sequential data.
    """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs
    ):
        """
        Initialize the PPO-LSTM policy.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            lr_schedule: Learning rate schedule
            **kwargs: Additional arguments
        """
        pass
    
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass of the policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_prob)
        """
        pass
```

## Simulation Module APIs

### IntradayRLEnv (Public)

```python
class IntradayRLEnv(gym.Env):
    """
    Intraday RL trading environment.
    
    This environment simulates intraday trading with various reward types,
    risk management, and position sizing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration
        """
        pass
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        
        Returns:
            Initial observation
        """
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        pass
    
    def close(self):
        """Close the environment."""
        pass
```

## Evaluation Module APIs

### BacktestEvaluator (Public)

```python
class BacktestEvaluator:
    """
    Backtest evaluation framework.
    
    This class provides comprehensive backtesting capabilities including
    performance metrics calculation, risk analysis, and visualization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtest evaluator.
        
        Args:
            config: Evaluation configuration
        """
        pass
    
    def run_backtest(
        self, 
        model: 'PPOLSTMPolicy', 
        env: gym.Env,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            model: Trained model
            env: RL environment
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with backtest results
        """
        pass
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            results: Backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    def generate_report(self, results: Dict[str, Any], output_path: str):
        """
        Generate a backtest report.
        
        Args:
            results: Backtest results
            output_path: Path to save report
        """
        pass
    
    def plot_results(self, results: Dict[str, Any], output_dir: str):
        """
        Plot backtest results.
        
        Args:
            results: Backtest results
            output_dir: Directory to save plots
        """
        pass
```

## Configuration Module APIs

### Settings (Public)

```python
class Settings:
    """
    Configuration settings for the RL trading system.
    
    This class provides a unified interface for accessing configuration
    settings from YAML files and environment variables.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        paths_override: Optional[Dict[str, Any]] = None,
        secrets_override: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize settings.
        
        Args:
            config_path: Path to configuration file
            paths_override: Override for path settings
            secrets_override: Override for secret settings
        """
        pass
    
    def get(self, *keys, default=None):
        """
        Get nested configuration value.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Settings dictionary
        """
        pass
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Settings':
        """
        Create settings from YAML file.
        
        Args:
            config_path: Path to YAML file
            
        Returns:
            Settings instance
        """
        pass
```

## Utility Functions

### Environment Building (Public)

```python
def build_env(config: Dict[str, Any]) -> gym.Env:
    """
    Build an RL environment from configuration.
    
    Args:
        config: Environment configuration
        
    Returns:
        Configured RL environment
    """
    pass

def build_multi_ticker_env(config: Dict[str, Any]) -> gym.Env:
    """
    Build a multi-ticker RL environment from configuration.
    
    Args:
        config: Multi-ticker environment configuration
        
    Returns:
        Configured multi-ticker RL environment
    """
    pass
```

### Training Functions (Public)

```python
def train_ppo_lstm(
    env: gym.Env,
    config: Dict[str, Any],
    model_path: Optional[str] = None
) -> 'RLTrainer':
    """
    Train a PPO-LSTM model.
    
    Args:
        env: RL environment
        config: Training configuration
        model_path: Path to save model
        
    Returns:
        Trained RL trainer
    """
    pass

def walk_forward_training(
    env_builder: Callable,
    config: Dict[str, Any],
    train_periods: List[Tuple[str, str]],
    test_periods: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Perform walk-forward training.
    
    Args:
        env_builder: Function to create environments
        config: Training configuration
        train_periods: List of training periods
        test_periods: List of testing periods
        
    Returns:
        Dictionary with training results
    """
    pass
```

## Multi-Ticker Extension APIs (To Be Implemented)

### MultiTickerDataLoader (Planned)

```python
class MultiTickerDataLoader:
    """
    Multi-ticker data loader with cross-ticker alignment.
    
    This class extends the UnifiedDataLoader to handle multiple tickers
    with proper alignment, normalization, and correlation analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-ticker data loader."""
        pass
    
    def load_aligned_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Load and align data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Aligned multi-ticker data
        """
        pass
    
    def get_correlation_matrix(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get correlation matrix for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Correlation matrix
        """
        pass
```

### MultiTickerFeaturePipeline (Planned)

```python
class MultiTickerFeaturePipeline:
    """
    Multi-ticker feature pipeline with cross-ticker features.
    
    This class extends the FeaturePipeline to handle multiple tickers
    with ticker-specific normalization and cross-ticker features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-ticker feature pipeline."""
        pass
    
    def extract_cross_ticker_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract cross-ticker features.
        
        Args:
            data: Dictionary of ticker data
            
        Returns:
            Cross-ticker features
        """
        pass
    
    def normalize_ticker_features(self, ticker: str, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            features: Feature matrix
            
        Returns:
            Normalized features
        """
        pass
```

### MultiTickerRLEnv (Planned)

```python
class MultiTickerRLEnv(gym.Env):
    """
    Multi-ticker RL trading environment.
    
    This environment extends the IntradayRLEnv to handle multiple tickers
    with portfolio management and cross-ticker risk controls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-ticker environment."""
        pass
    
    def step(self, actions: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take actions for multiple tickers.
        
        Args:
            actions: Dictionary of ticker-action pairs
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Portfolio state dictionary
        """
        pass
```

## API Versioning and Compatibility

### Versioning Scheme
- Major version (X.0.0): Breaking changes
- Minor version (0.X.0): New features, backward compatible
- Patch version (0.0.X): Bug fixes, backward compatible

### Deprecation Policy
- Deprecated APIs will be marked for at least one minor version
- Removal will occur in the next major version
- Migration guides will be provided for breaking changes

### Backward Compatibility
- Public APIs maintain backward compatibility within major versions
- Semi-public APIs may change with minor version notice
- Internal APIs can change at any time

## API Usage Examples

### Basic Usage

```python
# Load configuration
settings = Settings.from_yaml("configs/settings.yaml")

# Load data
data_loader = UnifiedDataLoader(settings.to_dict())
data = data_loader.load_data("AAPL", "2023-01-01", "2023-12-31")

# Extract features
feature_pipeline = FeaturePipeline(settings.get("features"))
features = feature_pipeline.fit_transform(data)

# Create environment
env = build_env(settings.to_dict())

# Train model
trainer = RLTrainer(settings.get("training"))
trainer.train(env)
trainer.save_model("models/ppo_lstm_model")

# Evaluate model
evaluator = BacktestEvaluator(settings.get("evaluation"))
results = evaluator.run_backtest(trainer.model, env, "2023-01-01", "2023-12-31")
```

### Multi-Ticker Usage (Planned)

```python
# Multi-ticker configuration
multi_ticker_config = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "data_source": "polygon",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
}

# Load multi-ticker data
data_loader = MultiTickerDataLoader(multi_ticker_config)
aligned_data = data_loader.load_aligned_data(
    multi_ticker_config["tickers"],
    multi_ticker_config["start_date"],
    multi_ticker_config["end_date"]
)

# Extract multi-ticker features
feature_pipeline = MultiTickerFeaturePipeline(multi_ticker_config)
features = feature_pipeline.fit_transform(aligned_data)

# Create multi-ticker environment
env = build_multi_ticker_env(multi_ticker_config)

# Train multi-ticker model
trainer = RLTrainer(multi_ticker_config)
trainer.train(env)
```

## API Extensions for Multi-Ticker Support

### Data Loading Extensions
- Support for multiple tickers in UnifiedDataLoader
- Cross-ticker data alignment methods
- Ticker metadata management

### Feature Pipeline Extensions
- Multi-ticker feature extraction
- Cross-ticker correlation features
- Ticker-specific normalization

### Environment Extensions
- Multi-ticker action spaces
- Portfolio-level observations
- Cross-ticker reward calculations

### Training Extensions
- Multi-ticker training loops
- Leave-One-Ticker-Out cross-validation
- Multi-objective optimization