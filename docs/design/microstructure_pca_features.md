
# Microstructure PCA Features

## Overview

This document outlines the design for microstructure PCA (Principal Component Analysis) features in the multi-ticker RL trading system. Microstructure PCA features are derived from market microstructure data using dimensionality reduction techniques to capture the most important patterns and relationships in order flow, liquidity, and price dynamics.

## Requirements

### Functional Requirements
1. **Microstructure Data Processing**: Process raw microstructure data (order book, trades, quotes)
2. **PCA Feature Extraction**: Apply PCA to extract principal components from microstructure features
3. **Multi-Ticker Support**: Generate PCA features for multiple tickers with cross-ticker analysis
4. **Dynamic Adaptation**: Adapt PCA models to changing market conditions
5. **Feature Selection**: Select the most informative principal components
6. **Regime Awareness**: Adjust PCA features based on market regimes
7. **Configurable Parameters**: Allow configuration of PCA parameters and feature selection

### Non-Functional Requirements
1. **Performance**: Efficient PCA computation with minimal computational overhead
2. **Stability**: Stable feature extraction that doesn't introduce excessive noise
3. **Interpretability**: Provide interpretable descriptions of principal components
4. **Configurability**: Flexible configuration of PCA parameters
5. **Adaptability**: Adaptive to changing market conditions

## Architecture Overview

### Core Components

#### 1. MicrostructurePCAFeatureExtractor
The main class responsible for extracting PCA features from microstructure data.

```python
class MicrostructurePCAFeatureExtractor(RewardComponent):
    """
    Microstructure PCA feature extractor.
    
    Extracts features from market microstructure data using
    Principal Component Analysis (PCA) for dimensionality reduction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the microstructure PCA feature extractor.
        
        Args:
            config: Configuration dictionary with PCA settings
        """
        super().__init__(config)
        self.pca_models = {}  # PCA models for each ticker
        self.feature_scalers = {}  # Feature scalers for each ticker
        self.cross_ticker_pca = None  # Cross-ticker PCA model
        self.selected_components = config.get('selected_components', 5)
        self.update_frequency = config.get('update_frequency', 100)  # Update every N steps
        self.min_samples = config.get('min_samples', 50)  # Minimum samples for PCA
        self.variance_threshold = config.get('variance_threshold', 0.95)  # Explained variance threshold
        self.use_cross_ticker_pca = config.get('use_cross_ticker_pca', True)
        self.regime_aware = config.get('regime_aware', True)
        self.feature_history = {}  # Feature history for each ticker
        self.pca_history = {}  # PCA history for each ticker
        self.step_count = 0
        self.regime_detector = RegimeDetector(config.get('regime_detection', {}))
        
    def calculate(self, portfolio_state: PortfolioState,
                 market_data: Dict[str, MarketData],
                 actions: Dict[str, float],
                 previous_portfolio_state: PortfolioState) -> float:
        """
        Calculate the microstructure PCA feature reward.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data for all tickers
            actions: Actions taken for each ticker
            previous_portfolio_state: Previous portfolio state
            
        Returns:
            Microstructure PCA feature reward
        """
        # Extract microstructure features
        microstructure_features = self._extract_microstructure_features(market_data)
        
        # Update feature history
        self._update_feature_history(microstructure_features)
        
        # Update PCA models if needed
        self._update_pca_models()
        
        # Extract PCA features
        pca_features = self._extract_pca_features(microstructure_features)
        
        # Calculate reward based on PCA features
        reward = self._calculate_pca_reward(pca_features, actions)
        
        # Apply regime adjustment if enabled
        if self.regime_aware:
            reward = self._apply_regime_adjustment(reward, market_data)
            
        return self.normalize(reward)
        
    def get_name(self) -> str:
        return "microstructure_pca_features"
        
    def _extract_microstructure_features(self, market_data: Dict[str, MarketData]) -> Dict[str, pd.DataFrame]:
        """
        Extract raw microstructure features from market data.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Dictionary of ticker -> microstructure features DataFrame
        """
        microstructure_features = {}
        
        for ticker, data in market_data.items():
            # Create feature DataFrame
            features = pd.DataFrame()
            
            # Order book features
            if hasattr(data, 'order_book') and data.order_book is not None:
                order_book_features = self._extract_order_book_features(data.order_book)
                features = pd.concat([features, order_book_features], axis=1)
                
            # Trade features
            if hasattr(data, 'trades') and data.trades is not None:
                trade_features = self._extract_trade_features(data.trades)
                features = pd.concat([features, trade_features], axis=1)
                
            # Quote features
            if hasattr(data, 'quotes') and data.quotes is not None:
                quote_features = self._extract_quote_features(data.quotes)
                features = pd.concat([features, quote_features], axis=1)
                
            # Volume features
            if hasattr(data, 'volume') and data.volume is not None:
                volume_features = self._extract_volume_features(data)
                features = pd.concat([features, volume_features], axis=1)
                
            # Price features
            if hasattr(data, 'price') and data.price is not None:
                price_features = self._extract_price_features(data)
                features = pd.concat([features, price_features], axis=1)
                
            microstructure_features[ticker] = features
            
        return microstructure_features
        
    def _extract_order_book_features(self, order_book: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features from order book data.
        
        Args:
            order_book: Order book data
            
        Returns:
            Order book features DataFrame
        """
        features = pd.DataFrame()
        
        # Bid-ask spread
        if 'bid_price' in order_book and 'ask_price' in order_book:
            features['spread'] = order_book['ask_price'] - order_book['bid_price']
            features['relative_spread'] = features['spread'] / ((order_book['bid_price'] + order_book['ask_price']) / 2)
            
        # Order book imbalance
        if 'bid_size' in order_book and 'ask_size' in order_book:
            features['order_imbalance'] = (order_book['bid_size'] - order_book['ask_size']) / (order_book['bid_size'] + order_book['ask_size'])
            
        # Book depth
        if 'bid_levels' in order_book and 'ask_levels' in order_book:
            features['bid_depth'] = sum(order_book['bid_levels'])
            features['ask_depth'] = sum(order_book['ask_levels'])
            features['depth_imbalance'] = (features['bid_depth'] - features['ask_depth']) / (features['bid_depth'] + features['ask_depth'])
            
        # Price impact
        if 'bid_price' in order_book and 'ask_price' in order_book and 'bid_size' in order_book and 'ask_size' in order_book:
            features['price_impact'] = (order_book['ask_price'] - order_book['bid_price']) / (order_book['bid_size'] + order_book['ask_size'])
            
        return features
        
    def _extract_trade_features(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from trade data.
        
        Args:
            trades: Trade data DataFrame
            
        Returns:
            Trade features DataFrame
        """
        features = pd.DataFrame()
        
        if len(trades) > 0:
            # Trade frequency
            features['trade_frequency'] = len(trades)
            
            # Trade size statistics
            features['avg_trade_size'] = trades['size'].mean()
            features['std_trade_size'] = trades['size'].std()
            features['max_trade_size'] = trades['size'].max()
            features['min_trade_size'] = trades['size'].min()
            
            # Trade direction imbalance
            if 'direction' in trades.columns:
                buy_trades = trades[trades['direction'] == 'buy']
                sell_trades = trades[trades['direction'] == 'sell']
                
                buy_volume = buy_trades['size'].sum() if len(buy_trades) > 0 else 0
                sell_volume = sell_trades['size'].sum() if len(sell_trades) > 0 else 0
                
                features['trade_imbalance'] = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
                
            # Trade price impact
            if 'price' in trades.columns:
                features['price_volatility'] = trades['price'].std()
                features['price_range'] = trades['price'].max() - trades['price'].min()
                
        return features
        
    def _extract_quote_features(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from quote data.
        
        Args:
            quotes: Quote data DataFrame
            
        Returns:
            Quote features DataFrame
        """
        features = pd.DataFrame()
        
        if len(quotes) > 0:
            # Quote frequency
            features['quote_frequency'] = len(quotes)
            
            # Quote spread statistics
            if 'bid_price' in quotes.columns and 'ask_price' in quotes.columns:
                quotes['spread'] = quotes['ask_price'] - quotes['bid_price']
                features['avg_spread'] = quotes['spread'].mean()
                features['std_spread'] = quotes['spread'].std()
                features['min_spread'] = quotes['spread'].min()
                features['max_spread'] = quotes['spread'].max()
                
            # Quote size statistics
            if 'bid_size' in quotes.columns and 'ask_size' in quotes.columns:
                features['avg_bid_size'] = quotes['bid_size'].mean()
                features['avg_ask_size'] = quotes['ask_size'].mean()
                features['bid_size_volatility'] = quotes['bid_size'].std()
                features['ask_size_volatility'] = quotes['ask_size'].std()
                
        return features
        
    def _extract_volume_features(self, data: MarketData) -> pd.DataFrame:
        """
        Extract volume-based features.
        
        Args:
            data: Market data
            
        Returns:
            Volume features DataFrame
        """
        features = pd.DataFrame()
        
        if hasattr(data, 'volume') and data.volume is not None:
            features['volume'] = [data.volume]
            
            # Volume relative to average
            if hasattr(data, 'avg_volume') and data.avg_volume is not None:
                features['relative_volume'] = data.volume / data.avg_volume
                
            # Volume volatility
            if hasattr(data, 'volume_history') and data.volume_history is not None:
                features['volume_volatility'] = np.std(data.volume_history)
                
        return features
        
    def _extract_price_features(self, data: MarketData) -> pd.DataFrame:
        """
        Extract price-based features.
        
        Args:
            data: Market data
            
        Returns:
            Price features DataFrame
        """
        features = pd.DataFrame()
        
        if hasattr(data, 'price') and data.price is not None:
            features['price'] = [data.price]
            
            # Price change
            if hasattr(data, 'previous_price') and data.previous_price is not None:
                features['price_change'] = data.price - data.previous_price
                features['price_change_pct'] = features['price_change'] / data.previous_price
                
            # Price volatility
            if hasattr(data, 'price_history') and data.price_history is not None:
                features['price_volatility'] = np.std(data.price_history)
                
            # Price relative to moving averages
            if hasattr(data, 'sma') and data.sma is not None:
                features['price_sma_ratio'] = data.price / data.sma
                
            if hasattr(data, 'ema') and data.ema is not None:
                features['price_ema_ratio'] = data.price / data.ema
                
        return features
        
    def _update_feature_history(self, microstructure_features: Dict[str, pd.DataFrame]):
        """
        Update feature history for each ticker.
        
        Args:
            microstructure_features: Current microstructure features
        """
        for ticker, features in microstructure_features.items():
            if ticker not in self.feature_history:
                self.feature_history[ticker] = []
                
            self.feature_history[ticker].append(features)
            
            # Keep history limited
            max_history = self.config.get('max_history', 1000)
            if len(self.feature_history[ticker]) > max_history:
                self.feature_history[ticker].pop(0)
                
    def _update_pca_models(self):
        """Update PCA models if needed."""
        self.step_count += 1
        
        # Update models at specified frequency
        if self.step_count % self.update_frequency == 0:
            # Update individual ticker PCA models
            for ticker, history in self.feature_history.items():
                if len(history) >= self.min_samples:
                    # Combine historical features
                    combined_features = pd.concat(history, ignore_index=True)
                    
                    # Remove NaN values
                    combined_features = combined_features.dropna()
                    
                    if len(combined_features) >= self.min_samples:
                        # Update PCA model
                        self._update_ticker_pca(ticker, combined_features)
                        
            # Update cross-ticker PCA model if enabled
            if self.use_cross_ticker_pca:
                self._update_cross_ticker_pca()
                
    def _update_ticker_pca(self, ticker: str, features: pd.DataFrame):
        """
        Update PCA model for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            features: Feature DataFrame
        """
        # Scale features
        if ticker not in self.feature_scalers:
            self.feature_scalers[ticker] = StandardScaler()
            
        scaler = self.feature_scalers[ticker]
        scaled_features = scaler.fit_transform(features)
        
        # Create or update PCA model
        if ticker not in self.pca_models:
            self.pca_models[ticker] = PCA(n_components=self.variance_threshold)
        else:
            # Reuse existing model but refit
            self.pca_models[ticker] = PCA(n_components=self.variance_threshold)
            
        # Fit PCA model
        pca = self.pca_models[ticker]
        pca.fit(scaled_features)
        
        # Store PCA history
        if ticker not in self.pca_history:
            self.pca_history[ticker] = []
            
        self.pca_history[ticker].append({
            'timestamp': pd.Timestamp.now(),
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        })
        
        # Keep history limited
        max_history = self.config.get('max_pca_history', 100)
        if len(self.pca_history[ticker]) > max_history:
            self.pca_history[ticker].pop(0)
            
    def _update_cross_ticker_pca(self):
        """Update cross-ticker PCA model."""
        # Combine features from all tickers
        all_features = []
        
        for ticker, history in self.feature_history.items():
            if len(history) >= self.min_samples:
                # Combine historical features for this ticker
                ticker_features = pd.concat(history, ignore_index=True)
                
                # Remove NaN values
                ticker_features = ticker_features.dropna()
                
                if len(ticker_features) >= self.min_samples:
                    # Add ticker prefix to columns
                    ticker_features = ticker_features.add_prefix(f"{ticker}_")
                    all_features.append(ticker_features)
                    
        if all_features:
            # Combine all ticker features
            combined_features = pd.concat(all_features, axis=1)
            
            # Remove NaN values
            combined_features = combined_features.dropna()
            
            if len(combined_features) >= self.min_samples:
                # Scale features
                if not hasattr(self, 'cross_ticker_scaler'):
                    self.cross_ticker_scaler = StandardScaler()
                    
                scaled_features = self.cross_ticker_scaler.fit_transform(combined_features)
                
                # Create or update PCA model
                self.cross_ticker_pca = PCA(n_components=self.variance_threshold)
                
                # Fit PCA model
                self.cross_ticker_pca.fit(scaled_features)
                
    def _extract_pca_features(self, microstructure_features: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Extract PCA features from microstructure features.
        
        Args:
            microstructure_features: Current microstructure features
            
        Returns:
            Dictionary of ticker -> PCA features
        """
        pca_features = {}
        
        for ticker, features in microstructure_features.items():
            if ticker in self.pca_models and len(features) > 0:
                # Remove NaN values
                features = features.dropna()
                
                if len(features) > 0:
                    # Scale features
                    scaler = self.feature_scalers[ticker]
                    scaled_features = scaler.transform(features)
                    
                    # Transform using PCA
                    pca = self.pca_models[ticker]
                    transformed_features = pca.transform(scaled_features)
                    
                    # Select top components
                    n_components = min(self.selected_components, transformed_features.shape[1])
                    pca_features[ticker] = transformed_features[:, :n_components]
                    
        return pca_features
        
    def _calculate_pca_reward(self, pca_features: Dict[str, np.ndarray], actions: Dict[str, float]) -> float:
        """
        Calculate reward based on PCA features.
        
        Args:
            pca_features: PCA features for each ticker
            actions: Actions taken for each ticker
            
        Returns:
            PCA feature reward
        """
        if not pca_features:
            return 0.0
            
        reward = 0.0
        
        for ticker, features in pca_features.items():
            if ticker in actions:
                action = actions[ticker]
                
                # Simple reward based on first principal component and action alignment
                if len(features) > 0:
                    # Use first principal component as signal
                    signal = features[0, 0]
                    
                    # Reward for aligning action with signal
                    alignment_reward = signal * action
                    
                    # Scale by feature importance (explained variance)
                    if ticker in self.pca_models:
                        explained_variance = self.pca_models[ticker].explained_variance_ratio_[0]
                        alignment_reward *= explained_variance
                        
                    reward += alignment_reward
                    
        # Normalize by number of tickers
        reward /= len(pca_features)
        
        return reward
        
    def _apply_regime_adjustment(self, reward: float, market_data: Dict[str, MarketData]) -> float:
        """
        Apply regime-based adjustment to the reward.
        
        Args:
            reward: Original reward value
            market_data: Current market data for all tickers
            
        Returns:
            Adjusted reward value
        """
        if not self.regime_aware or not market_data:
            return reward
            
        # Detect current regime
        regime = self.regime_detector.detect_regime(market_data)
        
        # Adjust reward based on regime
        if regime == 'high_volatility':
            adjustment_factor = 0.8  # Reduce reward in high volatility
        elif regime == 'low_volatility':
            adjustment_factor = 1.2  # Increase reward in low volatility
        elif regime == 'trending':
            adjustment_factor = 1.1  # Slightly increase reward in trending markets
        elif regime == 'range_bound':
            adjustment_factor = 0.9  # Slightly reduce reward in range-bound markets
        else:  # normal regime
            adjustment_factor = 1.0
            
        return reward * adjustment_factor
```

#### 2. RegimeDetector
Component for detecting market regimes based on microstructure features.

```python
class RegimeDetector:
    """
    Market regime detector based on microstructure features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window = config.get('window', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.2)
        self.trend_threshold = config.get('trend_threshold', 0.1)
        self.feature_history = []
        
    def detect_regime(self, market_data: Dict[str, MarketData]) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Regime name
        """
        # Calculate regime features
        regime_features = self._calculate_regime_features(market_data)
        
        # Update feature history
        self.feature_history.append(regime_features)
        
        # Keep history limited
        if len(self.feature_history) > self.window:
            self.feature_history.pop(0)
            
        # Detect regime if we have enough history
        if len(self.feature_history) >= self.window:
            return self._classify_regime()
        else:
            return 'normal'
            
    def _calculate_regime_features(self, market_data: Dict[str, MarketData]) -> Dict[str, float]:
        """
        Calculate features for regime detection.
        
        Args:
            market_data: Current market data for all tickers
            
        Returns:
            Dictionary of regime features
        """
        features = {}
        
        # Calculate average volatility
        volatilities = []
        for ticker, data in market_data.items():
            if hasattr(data, 'volatility') and data.volatility is not None:
                volatilities.append(data.volatility)
                
        if volatilities:
            features['avg_volatility'] = np.mean(volatilities)
            features['volatility_dispersion'] = np.std(volatilities)
        else:
            features['avg_volatility'] = 0.0
            features['volatility_dispersion'] = 0.0
            
        # Calculate trend strength
        trends = []
        for ticker, data in market_data.items():
            if hasattr(data, 'trend') and data.trend is not None:
                trends.append(data.trend)
                
        if trends:
            features['avg_trend'] = np.mean(trends)
            features['trend_dispersion'] = np.std(trends)
        else:
            features['avg_trend'] = 0.0
            features['trend_dispersion'] = 0.0
            
        # Calculate correlation features
        if len(market_data) > 1:
            prices = []
            for ticker, data in market_data.items():
                if hasattr(data, 'price') and data.price is not None:
                    prices.append(data.price)
                    
            if len(prices) > 1:
                features['price_correlation'] = np.corrcoef(prices)[0, 1] if len(prices) == 2 else 0.0
            else:
                features['price_correlation'] = 0.0
        else:
            features['price_correlation'] = 0.0
            
        return features
        
    def _classify_regime(self) -> str:
        """
        Classify current market regime based on feature history.
        
        Returns:
            Regime name
        """
        # Calculate feature averages over the window
        avg_volatility = np.mean([f['avg_volatility'] for f in self.feature_history])
        avg_trend = np.mean([f['avg_trend'] for f in self.feature_history])
        avg_correlation = np.mean([f['price_correlation'] for f in self.feature_history])
        
        # Classify regime
        if avg_volatility > self.volatility_threshold:
            return 'high_volatility'
        elif avg_volatility < self.volatility_threshold * 0.5:
            return 'low_volatility'
        elif abs(avg_trend) > self.trend_threshold:
            return 'trending'
        elif avg_correlation > 0.7:
            return 'high_correlation'
        elif avg_correlation < -0.7:
            return 'low_correlation'
        else:
            return 'normal'
```

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Market Data     │    │ Portfolio       │    │ Actions         │
│ (Input)         │    │ State           │    │ (Input)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Microstructure  │    │ Feature         │    │ PCA             │
│ Feature         │    │ History         │    │ Models          │
│ Extraction      │◀───│ (Update)        │◀───│ (Update)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ PCA Feature     │
│ Extraction     │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ PCA Reward      │
│ Calculation     │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Regime          │
│ Adjustment      │
└─────────────────┘
         │
         │
         ▼
┌─────────────────┐
│ Normalized      │
│ Reward          │
│ (Output)        │
└─────────────────┘
```

## Configuration Structure

### Microstructure PCA Features Configuration

```yaml
# Microstructure PCA features configuration
microstructure_pca_features:
  enabled: true
  weight: 0.2
  
  # PCA parameters
  selected_components: 5  # Number of principal components to use
  update_frequency: 100  # Update PCA models every N steps
  min_samples: 50  # Minimum samples for PCA
  variance_threshold: 0.95  # Explained variance threshold
  use_cross_ticker_pca: true  # Enable cross-ticker PCA
  regime_aware: true  # Enable regime-aware adjustments
  
  # Feature extraction parameters
  max_history: 1000  # Maximum feature history to keep
  max_pca_history: 100  # Maximum PCA history to keep
  
  # Regime detection parameters
  regime_detection:
    window: 20  # Window for regime detection
    volatility_threshold: 0.2  # Threshold for high volatility regime
    trend_threshold: 0.1  # Threshold for trending regime
    
  # Order book features
  order_book_features:
    enabled: true
    include_spread: true
    include_order_imbalance: true
    include_depth: true
    include_price_impact: true
    
  # Trade features
  trade_features:
    enabled: true
    include_frequency: true
    include_size_stats: true
    include_direction_imbalance: true
    include_price_impact: true
    
  # Quote features
  quote_features:
    enabled: true
    include_frequency: true
    include_spread_stats: true
    include_size_stats: true
    
  # Volume features
  volume_features:
    enabled: true
    include_relative_volume: true
    include_volatility: true
    
  # Price features
  price_features:
    enabled: true
    include_change: true
    include_volatility: true
    include_ma_ratios: true
    
  # Normalization
  normalization:
    enabled: true
    method: standard  # standard, min_max, clip
    clip_threshold: 3.0
```

## Implementation Details

### Microstructure Feature Extraction Algorithm

```python
def _extract_microstructure_features(self, market_data: Dict[str, MarketData]) -> Dict[str, pd.DataFrame]:
    """
    Extract raw microstructure features from market data.
    
    Args:
        market_data: Current market data for all tickers
        
    Returns:
        Dictionary of ticker -> microstructure features DataFrame
    """
    microstructure_features = {}
    
    for ticker, data in market_data.items():
        # Create feature DataFrame
        features = pd.DataFrame()
        
        # Order book features
        if hasattr(data, 'order_book') and data.order_book is not None:
            order_book_features = self._extract_order_book_features(data.order_book)
            features = pd.concat([features, order_book_features], axis=1)
            
        # Trade features
        if hasattr(data, 'trades') and data.trades is not None:
            trade_features = self._extract_trade_features(data.trades)
            features = pd.concat([features, trade_features], axis=1)
            
        # Quote features
        if hasattr(data, 'quotes') and data.quotes is not None:
            quote_features = self._extract_quote_features(data.quotes)
            features = pd.concat([features, quote_features], axis=1)
            
        # Volume features
        if hasattr(data, 'volume') and data.volume is not None:
            volume_features = self._extract_volume_features(data)
            features = pd.concat([features, volume_features], axis=1)
            
        # Price features
        if hasattr(data, 'price') and data.price is not None:
            price_features = self._extract_price_features(data)
            features = pd.concat([features, price_features], axis=1)
            
        microstructure_features[ticker] = features
            
    return microstructure_features
```

### PCA Model Update Algorithm

```python
def _update_pca_models(self):
    """Update PCA models if needed."""
    self.step_count += 1
    
    # Update models at specified frequency
    if self.step_count % self.update_frequency == 0:
        # Update individual ticker PCA models
        for ticker, history in self.feature_history.items():
            if len(history) >= self.min_samples:
                # Combine historical features
                combined_features = pd.concat(history, ignore_index=True)
                
                # Remove NaN values
                combined_features = combined_features.dropna()
                
                if len(combined_features) >= self.min_samples:
                    # Update PCA model
                    self._update_ticker_pca(ticker, combined_features)
                    
        # Update cross-ticker PCA model if enabled
        if self.use_cross_ticker_pca:
            self._update_cross_ticker_pca()
```

### PCA Feature Extraction Algorithm

```python
def _extract_pca_features(self, microstructure_features: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    """
    Extract PCA features from microstructure features.
    
    Args:
        microstructure_features: Current microstructure features
        
    Returns:
        Dictionary of ticker -> PCA features
    """
    pca_features = {}
    
    for ticker, features in microstructure_features.items():
        if ticker in self.pca_models and len(features) > 0:
            # Remove NaN values
            features = features.dropna()
            
            if len(features) > 0:
                # Scale features
                scaler = self.feature_scalers[ticker]
                scaled_features = scaler.transform(features)
                
                # Transform using PCA
                pca = self.pca_models[ticker]
                transformed_features = pca.transform(scaled_features)
                
                # Select top components
                n_components = min(self.selected_components, transformed_features.shape[1])
                pca_features[ticker] = transformed_features[:, :n_components]
                
    return pca_features
```

### PCA Reward Calculation Algorithm

```python
def _calculate_pca_reward(self, pca_features: Dict[str, np.ndarray], actions: Dict[str, float]) -> float:
    """
    Calculate reward based on PCA features.
    
    Args:
        pca_features: PCA features for each ticker
        actions: Actions taken for each ticker
        
    Returns:
        PCA feature reward
    """
    if not pca_features:
        return 0.0
        
    reward = 0.0
    
    for ticker, features in pca_features.items():
        if ticker in actions:
            action = actions[ticker]
            
            # Simple reward based on first principal component and action alignment
            if len(features) > 0:
                # Use first principal component as signal
                signal = features[0, 0]
                
                # Reward for aligning action with signal
                alignment_reward = signal * action
                
                # Scale by feature importance (explained variance)
                if ticker in self.pca_models:
                    explained_variance = self.pca_models[ticker].explained_variance_ratio_[0]
                    alignment_reward *= explained_variance
                    
                reward += alignment_reward
                
    # Normalize by number of tickers
    reward /= len(pca_features)
    
    return reward
```

## Best Practices

### Feature Engineering
1. **Feature Diversity**: Include a diverse set of microstructure features to capture different market aspects
2. **Feature Quality**: Ensure features are properly cleaned and normalized before PCA
3. **Feature Relevance**: Focus on features that are relevant