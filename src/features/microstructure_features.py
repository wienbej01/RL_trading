"""
Microstructure features for the RL trading system.

This module implements market microstructure features including
order book analysis, liquidity metrics, and trade flow analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from ..utils.config_loader import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure features."""
    window: int
    levels: int
    liquidity_threshold: float
    price_impact_threshold: float


class MicrostructureFeatures:
    """
    Market microstructure features implementation.
    
    This class provides advanced microstructure features including
    order book analysis, liquidity metrics, and trade flow analysis.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize microstructure features.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.config = settings.get('microstructure', {})
        self.default_window = self.config.get('window', 20)
        self.default_levels = self.config.get('levels', 5)
        
    def calculate_all_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        
        Args:
            data: DataFrame with market data (OHLCV + order book)
            
        Returns:
            DataFrame with all microstructure features
        """
        if data.empty:
            return pd.DataFrame()
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        
        # Calculate order book features
        result = self._add_order_book_features(result, data)
        
        # Calculate liquidity features
        result = self._add_liquidity_features(result, data)
        
        # Calculate trade flow features
        result = self._add_trade_flow_features(result, data)
        
        # Calculate price impact features
        result = self._add_price_impact_features(result, data)
        
        # Calculate market depth features
        result = self._add_market_depth_features(result, data)
        
        # Calculate volatility clustering features
        result = self._add_volatility_clustering_features(result, data)
        
        return result
    
    def _add_order_book_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add order book-based features."""
        # Bid-Ask Spread
        result['bid_ask_spread'] = data['ask'] - data['bid']
        result['bid_ask_spread_pct'] = result['bid_ask_spread'] / ((data['ask'] + data['bid']) / 2)
        
        # Mid Price
        result['mid_price'] = (data['ask'] + data['bid']) / 2
        
        # Order Imbalance
        result['order_imbalance'] = (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size'])
        
        # Weighted Mid Price
        result['weighted_mid_price'] = (
            (data['bid'] * data['bid_size'] + data['ask'] * data['ask_size']) /
            (data['bid_size'] + data['ask_size'])
        )
        
        # Effective Spread
        result['effective_spread'] = 2 * np.abs(data['close'] - result['mid_price'])
        
        # Realized Spread
        result['realized_spread'] = 2 * (data['close'] - result['mid_price']).shift(1)
        
        # Price Impact
        result['price_impact'] = data['close'] - result['mid_price']
        
        # Order Book Slope
        if 'bid_size_1' in data.columns and 'ask_size_1' in data.columns:
            result['bid_slope'] = (data['bid_size_1'] - data['bid_size']) / (data['bid_1'] - data['bid'])
            result['ask_slope'] = (data['ask_size_1'] - data['ask_size']) / (data['ask'] - data['ask_1'])
        
        # Order Book Curvature
        if 'bid_size_2' in data.columns and 'ask_size_2' in data.columns:
            result['bid_curvature'] = (
                (data['bid_size_2'] - 2 * data['bid_size_1'] + data['bid_size']) /
                ((data['bid_2'] - data['bid']) ** 2)
            )
            result['ask_curvature'] = (
                (data['ask_size_2'] - 2 * data['ask_size_1'] + data['ask_size']) /
                ((data['ask'] - data['ask_2']) ** 2)
            )
        
        return result
    
    def _add_liquidity_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-based features."""
        # Total Order Book Depth
        result['total_depth'] = data['bid_size'] + data['ask_size']
        
        # Bid-Ask Depth Ratio
        result['depth_ratio'] = data['bid_size'] / (data['ask_size'] + 1e-6)
        
        # Effective Liquidity
        result['effective_liquidity'] = (
            (data['bid_size'] * data['ask_size']) /
            (data['bid_size'] + data['ask_size'] + 1e-6)
        )
        
        # Liquidity Score
        result['liquidity_score'] = (
            (1 / (result['bid_ask_spread_pct'] + 1e-6)) *
            (result['total_depth'] / result['total_depth'].rolling(window=20).mean())
        )
        
        # Liquidity Regime
        result['liquidity_regime'] = pd.cut(
            result['liquidity_score'],
            bins=[-np.inf, 0.5, 1.5, np.inf],
            labels=['low', 'normal', 'high']
        )
        
        # Liquidity Shock
        result['liquidity_shock'] = (
            result['total_depth'] - result['total_depth'].rolling(window=20).mean()
        ) / result['total_depth'].rolling(window=20).std()
        
        # Liquidity Divergence
        result['liquidity_divergence'] = (
            result['bid_size'] / result['bid_size'].rolling(window=20).mean() -
            result['ask_size'] / result['ask_size'].rolling(window=20).mean()
        )
        
        # Liquidity Pressure
        result['liquidity_pressure'] = (
            (data['close'] - result['mid_price']) / result['bid_ask_spread_pct']
        )
        
        return result
    
    def _add_trade_flow_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add trade flow-based features."""
        # Trade Flow Imbalance
        result['trade_flow_imbalance'] = (
            data['volume'] * np.sign(data['close'] - data['open'])
        )
        
        # Buy-Sell Ratio
        result['buy_sell_ratio'] = (
            np.where(data['close'] > data['open'], data['volume'], 0) /
            (np.where(data['close'] <= data['open'], data['volume'], 0) + 1e-6)
        )
        
        # Accumulation/Distribution
        result['accumulation_distribution'] = (
            (data['close'] - data['low']) - (data['high'] - data['close'])
        ) / (data['high'] - data['low'] + 1e-6) * data['volume']
        
        # Money Flow
        result['money_flow'] = (
            (data['close'] + data['high'] + data['low']) / 3 * data['volume']
        )
        
        # Volume-Weighted Price
        result['volume_weighted_price'] = (
            data['close'] * data['volume']
        ) / (data['volume'] + 1e-6)
        
        # Trade Flow Momentum
        result['trade_flow_momentum'] = result['trade_flow_imbalance'].rolling(window=5).sum()
        
        # Trade Flow Acceleration
        result['trade_flow_acceleration'] = (
            result['trade_flow_momentum'] - result['trade_flow_momentum'].shift(1)
        )
        
        # Trade Flow Divergence
        result['trade_flow_divergence'] = (
            result['trade_flow_imbalance'] - result['trade_flow_imbalance'].rolling(window=20).mean()
        )
        
        # Trade Flow Volatility
        result['trade_flow_volatility'] = (
            result['trade_flow_imbalance'].rolling(window=20).std()
        )
        
        return result
    
    def _add_price_impact_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price impact-based features."""
        # Price Impact per Unit Volume
        result['price_impact_per_volume'] = (
            result['price_impact'] / (data['volume'] + 1e-6)
        )
        
        # Price Impact per Dollar Volume
        result['price_impact_per_dollar_volume'] = (
            result['price_impact'] / (data['close'] * data['volume'] + 1e-6)
        )
        
        # Price Impact Persistence
        result['price_impact_persistence'] = (
            result['price_impact'].rolling(window=5).mean() / 
            (result['price_impact'].rolling(window=20).mean() + 1e-6)
        )
        
        # Price Impact Asymmetry
        result['price_impact_asymmetry'] = (
            np.where(result['price_impact'] > 0, result['price_impact'], 0).rolling(window=20).mean() /
            (np.where(result['price_impact'] < 0, -result['price_impact'], 0).rolling(window=20).mean() + 1e-6)
        )
        
        # Price Impact Regime
        result['price_impact_regime'] = pd.cut(
            result['price_impact_per_volume'],
            bins=[-np.inf, -0.1, 0.1, np.inf],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Price Impact Shock
        result['price_impact_shock'] = (
            result['price_impact_per_volume'] - 
            result['price_impact_per_volume'].rolling(window=20).mean()
        ) / result['price_impact_per_volume'].rolling(window=20).std()
        
        return result
    
    def _add_market_depth_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market depth-based features."""
        # Depth at Best Price
        result['depth_at_best'] = data['bid_size'] + data['ask_size']
        
        # Depth at Multiple Levels
        if 'bid_size_2' in data.columns and 'ask_size_2' in data.columns:
            result['depth_at_2'] = data['bid_size_1'] + data['bid_size_2'] + data['ask_size_1'] + data['ask_size_2']
        
        # Depth Concentration
        result['depth_concentration'] = (
            (data['bid_size'] + data['ask_size']) /
            (result['total_depth'] + 1e-6)
        )
        
        # Depth Slope
        if 'bid_size_1' in data.columns and 'ask_size_1' in data.columns:
            result['depth_slope'] = (
                (data['bid_size_1'] - data['bid_size']) / (data['bid_1'] - data['bid']) +
                (data['ask_size'] - data['ask_size_1']) / (data['ask_1'] - data['ask'])
            )
        
        # Depth Curvature
        if 'bid_size_2' in data.columns and 'ask_size_2' in data.columns:
            result['depth_curvature'] = (
                (data['bid_size_2'] - 2 * data['bid_size_1'] + data['bid_size']) / ((data['bid_2'] - data['bid']) ** 2) +
                (data['ask_size_2'] - 2 * data['ask_size_1'] + data['ask_size']) / ((data['ask'] - data['ask_2']) ** 2)
            )
        
        # Depth Imbalance
        result['depth_imbalance'] = (
            (data['bid_size'] - data['ask_size']) /
            (data['bid_size'] + data['ask_size'] + 1e-6)
        )
        
        # Depth Volatility
        result['depth_volatility'] = (
            result['total_depth'].rolling(window=20).std() /
            (result['total_depth'].rolling(window=20).mean() + 1e-6)
        )
        
        return result
    
    def _add_volatility_clustering_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering features."""
        # Realized Volatility
        result['realized_volatility'] = (
            np.log(data['close'] / data['close'].shift())
        ).rolling(window=20).std() * np.sqrt(252) * 100
        
        # Volatility Clustering
        result['volatility_clustering'] = (
            result['realized_volatility'] / 
            result['realized_volatility'].rolling(window=20).mean()
        )
        
        # Volatility Persistence
        result['volatility_persistence'] = (
            result['realized_volatility'].rolling(window=5).mean() /
            (result['realized_volatility'].rolling(window=20).mean() + 1e-6)
        )
        
        # Volatility Jump
        result['volatility_jump'] = (
            result['realized_volatility'] - 
            result['realized_volatility'].shift(1)
        )
        
        # Volatility Regime
        result['volatility_regime'] = pd.cut(
            result['realized_volatility'],
            bins=[-np.inf, 10, 20, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Volatility Shock
        result['volatility_shock'] = (
            result['realized_volatility'] - 
            result['realized_volatility'].rolling(window=20).mean()
        ) / result['realized_volatility'].rolling(window=20).std()
        
        # Volatility Skew
        result['volatility_skew'] = (
            result['realized_volatility'].rolling(window=20).skew()
        )
        
        # Volatility Kurtosis
        result['volatility_kurtosis'] = (
            result['realized_volatility'].rolling(window=20).kurtosis()
        )
        
        return result
    
    def calculate_order_flow_pressure(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate order flow pressure.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with order flow pressure
        """
        # Order flow pressure combines multiple microstructure signals
        ofp = (
            self._calculate_buy_pressure(data) -
            self._calculate_sell_pressure(data)
        )
        
        return ofp
    
    def _calculate_buy_pressure(self, data: pd.DataFrame) -> pd.Series:
        """Calculate buy pressure."""
        buy_pressure = (
            np.where(data['close'] > data['open'], data['volume'], 0) +
            np.where(data['close'] > data['bid'], data['bid_size'], 0) +
            np.where(data['close'] < data['ask'], data['ask_size'], 0)
        )
        return pd.Series(buy_pressure, index=data.index)
    
    def _calculate_sell_pressure(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sell pressure."""
        sell_pressure = (
            np.where(data['close'] < data['open'], data['volume'], 0) +
            np.where(data['close'] < data['bid'], data['bid_size'], 0) +
            np.where(data['close'] > data['ask'], data['ask_size'], 0)
        )
        return pd.Series(sell_pressure, index=data.index)
    
    def calculate_market_liquidity_index(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate market liquidity index.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with market liquidity index
        """
        # Combine multiple liquidity metrics
        spread_component = 1 / (data['ask'] - data['bid'] + 1e-6)
        depth_component = (data['bid_size'] + data['ask_size']) / 1000
        volume_component = data['volume'] / data['volume'].rolling(window=20).mean()
        
        liquidity_index = (
            spread_component * 0.4 +
            depth_component * 0.3 +
            volume_component * 0.3
        )
        
        return liquidity_index
    
    def calculate_price_discovery_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price discovery efficiency.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with price discovery efficiency
        """
        # Price discovery efficiency measures how quickly prices reflect new information
        mid_price_changes = data['mid_price'].pct_change()
        trade_volume = data['volume']
        
        # Efficiency ratio
        efficiency = (
            mid_price_changes.abs() / 
            (trade_volume / trade_volume.rolling(window=20).mean() + 1e-6)
        )
        
        return efficiency.rolling(window=20).mean()
    
    def calculate_market_impact_coefficient(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate market impact coefficient.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with market impact coefficient
        """
        # Market impact coefficient measures the price impact per unit volume
        price_changes = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()
        
        # Impact coefficient
        impact_coefficient = (
            price_changes.abs() / 
            (volume_changes.abs() + 1e-6)
        )
        
        return impact_coefficient.rolling(window=20).mean()
    
    def calculate_information_asymmetry(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate information asymmetry.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with information asymmetry
        """
        # Information asymmetry measures the difference between informed and uninformed traders
        bid_ask_spread = data['ask'] - data['bid']
        order_imbalance = (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size'])
        
        # Information asymmetry index
        asymmetry = (
            bid_ask_spread * order_imbalance
        )
        
        return asymmetry.rolling(window=20).mean()
    
    def calculate_market_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate market quality score.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with market quality score
        """
        # Market quality score combines multiple microstructure metrics
        liquidity_score = self.calculate_market_liquidity_index(data)
        price_efficiency = self.calculate_price_discovery_efficiency(data)
        impact_coefficient = self.calculate_market_impact_coefficient(data)
        information_asymmetry = self.calculate_information_asymmetry(data)
        
        # Normalize and combine
        quality_score = (
            liquidity_score / liquidity_score.rolling(window=20).mean() * 0.3 +
            price_efficiency / price_efficiency.rolling(window=20).mean() * 0.3 +
            1 / (impact_coefficient + 1e-6) * 0.2 +
            1 / (information_asymmetry.abs() + 1e-6) * 0.2
        )
        
        return quality_score


def calculate_order_flow_imbalance(bid_price: pd.Series, bid_size: pd.Series, 
                                 ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Calculate order flow imbalance.
    
    Order flow imbalance measures the imbalance between buy and sell pressure
    in the order book, which can be predictive of short-term price movements.
    
    Args:
        bid_price: Series of bid prices
        bid_size: Series of bid sizes
        ask_price: Series of ask prices
        ask_size: Series of ask sizes
        
    Returns:
        Series containing order flow imbalance values
    """
    # Calculate mid price
    mid_price = (bid_price + ask_price) / 2
    
    # Calculate weighted order flow imbalance
    # OFI = (ask_size * ask_price - bid_size * bid_price) / mid_price
    ofi = (ask_size * ask_price - bid_size * bid_price) / mid_price
    
    # First value should be NaN (requires previous values for calculation)
    ofi.iloc[0] = np.nan
    
    return ofi


def calculate_microprice(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Calculate microprice based on order book imbalance.
    
    The microprice is a weighted average of bid and ask prices
    based on order book depth and imbalance.
    
    Args:
        bid_price: Series with bid prices
        bid_size: Series with bid sizes
        ask_price: Series with ask prices
        ask_size: Series with ask sizes
        
    Returns:
        Series with microprice values
    """
    if bid_price.empty or bid_size.empty or ask_price.empty or ask_size.empty:
        return pd.Series(dtype=float)
    
    # Calculate order book imbalance
    imbalance = bid_size / (bid_size + ask_size + 1e-6)
    microprice = bid_price * (1 - imbalance) + ask_price * imbalance
    
    return pd.Series(microprice, index=bid_price.index)


def calculate_queue_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """
    Calculate queue imbalance based on bid and ask sizes.
    
    Args:
        bid_size: Series with bid sizes
        ask_size: Series with ask sizes
        
    Returns:
        Series with queue imbalance values between -1 and 1
    """
    if bid_size.empty or ask_size.empty:
        return pd.Series(dtype=float)
    
    # Queue imbalance formula
    imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-6)
    
    return pd.Series(imbalance, index=bid_size.index)


def calculate_spread(bid_price: pd.Series, ask_price: pd.Series) -> pd.Series:
    """
    Calculate bid-ask spread.
    
    Args:
        bid_price: Series with bid prices
        ask_price: Series with ask prices
        
    Returns:
        Series with spread values
    """
    if bid_price.empty or ask_price.empty:
        return pd.Series(dtype=float)
    
    # Spread calculation
    spread = ask_price - bid_price
    
    return pd.Series(spread, index=bid_price.index)


def calculate_price_impact(trade_price: pd.Series, trade_size: pd.Series, 
                          bid_price: pd.Series, ask_price: pd.Series) -> pd.Series:
    """
    Calculate price impact based on trades and order book.
    
    Args:
        trade_price: Series with trade prices
        trade_size: Series with trade sizes
        bid_price: Series with bid prices
        ask_price: Series with ask prices
        
    Returns:
        Series with price impact values
    """
    if trade_price.empty or trade_size.empty or bid_price.empty or ask_price.empty:
        return pd.Series(dtype=float)
    
    # Calculate mid price
    mid_price = (bid_price + ask_price) / 2
    
    # Price impact calculation
    price_impact = (trade_price - mid_price) / mid_price
    
    return pd.Series(price_impact, index=trade_price.index)


def calculate_vwap(trade_price: pd.Series, trade_size: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        trade_price: Series with trade prices
        trade_size: Series with trade sizes
        
    Returns:
        Series with VWAP values
    """
    if trade_price.empty or trade_size.empty:
        return pd.Series(dtype=float)
    
    # VWAP calculation using cumulative sum
    cumulative_price_volume = (trade_price * trade_size).cumsum()
    cumulative_volume = trade_size.cumsum()
    
    vwap = cumulative_price_volume / (cumulative_volume + 1e-6)
    
    return pd.Series(vwap, index=trade_price.index)


def calculate_twap(trade_price: pd.Series, window: int = 10) -> pd.Series:
    """
    Calculate Time Weighted Average Price (TWAP).
    
    Args:
        trade_price: Series with trade prices
        window: Rolling window size
        
    Returns:
        Series with TWAP values
    """
    if trade_price.empty:
        return pd.Series(dtype=float)
    
    # TWAP is simply a rolling mean
    twap = trade_price.rolling(window=window).mean()
    
    return pd.Series(twap, index=trade_price.index)
