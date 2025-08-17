"""
Paper trading interface for RL trading system.

This module provides a comprehensive paper trading interface that
integrates trained RL models with IBKR for realistic trading simulation.
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path
import json
from queue import Queue

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..features.pipeline import FeaturePipeline
from ..sim.env_intraday_rl import IntradayRLEnv
from .ibkr_client import IBKRClient, PaperTradingInterface, IBKRConfig

logger = get_logger(__name__)


@dataclass
class PaperTradingConfig:
    """Paper trading configuration."""
    model_path: str = ""
    trading_symbol: str = "MES"
    trading_exchange: str = "CME"
    trading_currency: str = "USD"
    initial_capital: float = 100000.0
    max_position_size: int = 10
    risk_per_trade_frac: float = 0.02
    stop_loss_r_multiple: float = 1.0
    take_profit_r_multiple: float = 1.5
    max_daily_loss_r: float = 3.0
    update_frequency: int = 60  # seconds
    data_lookback: int = 120  # minutes
    enable_risk_management: bool = True
    log_trades: bool = True
    save_equity_curve: bool = True
    output_dir: str = "paper_trading_results"


class PaperTradingEngine:
    """
    Paper trading engine for RL trading system.
    
    This class integrates trained RL models with IBKR for realistic
    paper trading with risk management and performance monitoring.
    """
    
    def __init__(self, settings: Settings, config: PaperTradingConfig):
        """
        Initialize paper trading engine.
        
        Args:
            settings: Configuration settings
            config: Paper trading configuration
        """
        self.settings = settings
        self.config = config
        
        # Initialize components
        self.ibkr_client = IBKRClient(settings)
        self.paper_trading = PaperTradingInterface(settings)
        
        # Feature pipeline
        self.feature_pipeline = FeaturePipeline(settings)
        
        # RL environment (for simulation)
        self.rl_env = None
        
        # Trading state
        self.current_position = 0
        self.pending_orders = {}
        self.trade_history = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        # Model
        self.model = None
        self.model_loaded = False
        
        # Data management
        self.market_data_buffer = []
        self.feature_buffer = []
        self.last_update_time = None
        
        # Risk management
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = self.config.initial_capital * self.config.max_daily_loss_r
        
        # Logging
        self.trade_log = []
        self.equity_log = []
        
        logger.info("Paper trading engine initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize paper trading engine.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing paper trading engine...")
            
            # Connect to IBKR
            if not await self.ibkr_client.connect():
                logger.error("Failed to connect to IBKR")
                return False
            
            # Connect paper trading interface
            if not await self.paper_trading.connect():
                logger.error("Failed to connect paper trading interface")
                return False
            
            # Load RL model
            if not await self._load_model():
                logger.error("Failed to load RL model")
                return False
            
            # Initialize RL environment
            await self._initialize_rl_env()
            
            # Create output directory
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info("Paper trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing paper trading engine: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """
        Load trained RL model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            from stable_baselines3 import PPO
            
            if not Path(self.config.model_path).exists():
                logger.error(f"Model file not found: {self.config.model_path}")
                return False
            
            self.model = PPO.load(self.config.model_path)
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully from {self.config.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    async def _initialize_rl_env(self):
        """Initialize RL environment for simulation."""
        try:
            # Get current market data
            market_data = await self._get_historical_data()
            
            if market_data is None or len(market_data) == 0:
                logger.error("Failed to get historical data for RL environment")
                return
            
            # Calculate features
            features = self.feature_pipeline.calculate_features(market_data)
            
            # Initialize RL environment
            self.rl_env = IntradayRLEnv(
                ohlcv=market_data[["open", "high", "low", "close", "volume"]],
                features=features,
                cash=self.config.initial_capital,
                exec_params=self._get_execution_params(),
                risk_cfg=self._get_risk_config(),
                point_value=self._get_point_value()
            )
            
            logger.info("RL environment initialized")
            
        except Exception as e:
            logger.error(f"Error initializing RL environment: {e}")
    
    def _get_execution_params(self):
        """Get execution parameters."""
        from ..sim.execution import ExecParams
        
        return ExecParams(
            tick_value=self.settings.get("execution", "tick_value", default=1.25),
            spread_ticks=self.settings.get("execution", "spread_ticks", default=1),
            impact_bps=self.settings.get("execution", "impact_bps", default=0.5),
            commission_per_contract=self.settings.get("execution", "commission_per_contract", default=0.6)
        )
    
    def _get_risk_config(self):
        """Get risk configuration."""
        from ..sim.risk import RiskConfig
        
        return RiskConfig(
            risk_per_trade_frac=self.config.risk_per_trade_frac,
            stop_r_multiple=self.config.stop_loss_r_multiple,
            tp_r_multiple=self.config.take_profit_r_multiple,
            max_daily_loss_r=self.config.max_daily_loss_r
        )
    
    def _get_point_value(self) -> float:
        """Get point value for the trading symbol."""
        point_values = {
            "MES": 5.0,  # Micro E-mini S&P 500
            "ES": 50.0,  # E-mini S&P 500
            "NQ": 20.0,  # E-mini Nasdaq 100
            "YM": 5.0,   # E-mini Dow Jones
            "CL": 1000.0, # Crude Oil
            "ZN": 1000.0, # 10-Year Treasury Note
        }
        
        return point_values.get(self.config.trading_symbol, 5.0)
    
    async def _get_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Get historical data for feature calculation.
        
        Returns:
            Historical data DataFrame or None
        """
        try:
            # Get contract
            contract = await self.ibkr_client.get_contract(
                self.config.trading_symbol,
                self.config.trading_exchange,
                self.config.trading_currency
            )
            
            if not contract:
                logger.error(f"Failed to get contract for {self.config.trading_symbol}")
                return None
            
            # Request historical data
            bars = await self.ibkr_client.request_historical_data(
                contract,
                duration=f"{self.config.data_lookback} mins",
                bar_size="1 min"
            )
            
            if bars:
                df = self.ibkr_client.get_historical_dataframe(contract)
                return df
            else:
                logger.error("No historical data received")
                return None
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    async def _update_market_data(self):
        """Update market data and features."""
        try:
            # Get current market data
            market_data = await self._get_historical_data()
            
            if market_data is None or len(market_data) == 0:
                logger.warning("No market data available")
                return
            
            # Calculate features
            features = self.feature_pipeline.calculate_features(market_data)
            
            # Update buffers
            self.market_data_buffer = market_data
            self.feature_buffer = features
            
            # Update RL environment
            if self.rl_env:
                self.rl_env.update_data(market_data, features)
            
            self.last_update_time = datetime.now()
            
            logger.debug("Market data updated")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _get_trading_signal(self) -> Optional[int]:
        """
        Get trading signal from RL model.
        
        Returns:
            Trading signal (-1: short, 0: flat, 1: long) or None
        """
        try:
            if not self.model_loaded or not self.feature_buffer:
                return None
            
            # Get latest features
            latest_features = self.feature_buffer.iloc[-1]
            
            # Prepare observation
            obs = np.concatenate([
                latest_features.values.astype(np.float32),
                np.array([self.current_position, 0.0], dtype=np.float32)  # position, unrealized_pnl
            ])
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            return int(action)
            
        except Exception as e:
            logger.error(f"Error getting trading signal: {e}")
            return None
    
    async def _execute_trade(self, signal: int):
        """
        Execute trade based on signal.
        
        Args:
            signal: Trading signal (-1: short, 0: flat, 1: long)
        """
        try:
            if signal == 0:  # Flat
                if self.current_position != 0:
                    # Close position
                    await self._close_position()
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size == 0:
                logger.info("Position size is zero, skipping trade")
                return
            
            # Determine action
            if signal == 1 and self.current_position <= 0:  # Go long
                action = "BUY"
                quantity = position_size - self.current_position
            elif signal == -1 and self.current_position >= 0:  # Go short
                action = "SELL"
                quantity = abs(position_size) + self.current_position
            else:
                logger.info("No action needed (already in position)")
                return
            
            # Execute trade
            if quantity != 0:
                order_id = await self.paper_trading.execute_trade(
                    self.config.trading_symbol,
                    action,
                    abs(quantity),
                    order_type="MKT"
                )
                
                if order_id:
                    logger.info(f"Trade executed: {action} {abs(quantity)} {self.config.trading_symbol}")
                else:
                    logger.error("Failed to execute trade")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _calculate_position_size(self, signal: int) -> int:
        """
        Calculate position size based on risk management.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size
        """
        if not self.config.enable_risk_management:
            return self.config.max_position_size
        
        # Calculate risk per trade
        risk_per_trade = self.config.initial_capital * self.config.risk_per_trade_frac
        
        # Get current market price
        if not self.market_data_buffer:
            return 0
        
        current_price = self.market_data_buffer["close"].iloc[-1]
        
        # Calculate ATR for stop loss
        atr = self.feature_buffer["atr"].iloc[-1] if "atr" in self.feature_buffer.columns else 1.0
        
        # Calculate position size
        if atr > 0:
            risk_per_contract = atr * self.config.stop_loss_r_multiple * self._get_point_value()
            max_contracts = int(risk_per_trade / risk_per_contract)
        else:
            max_contracts = self.config.max_position_size
        
        # Apply position limit
        max_contracts = min(max_contracts, self.config.max_position_size)
        
        # Return position size based on signal
        return max_contracts if signal == 1 else -max_contracts
    
    async def _close_position(self):
        """Close current position."""
        try:
            if self.current_position > 0:
                # Close long position
                order_id = await self.paper_trading.execute_trade(
                    self.config.trading_symbol,
                    "SELL",
                    self.current_position,
                    order_type="MKT"
                )
            elif self.current_position < 0:
                # Close short position
                order_id = await self.paper_trading.execute_trade(
                    self.config.trading_symbol,
                    "BUY",
                    abs(self.current_position),
                    order_type="MKT"
                )
            
            if order_id:
                logger.info(f"Position closed: {self.current_position} {self.config.trading_symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Get account info
            account_info = await self.paper_trading.get_account_info()
            
            # Calculate equity
            equity = account_info.get('account_summary', {}).get('TotalCash', 0)
            if 'portfolio' in account_info:
                for position in account_info['portfolio']:
                    equity += position.get('market_value', 0)
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'position': self.current_position
            })
            
            # Calculate daily P&L
            if len(self.equity_curve) > 1:
                daily_pnl = equity - self.equity_curve[0]['equity']
                self.daily_pnl = daily_pnl
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
                await self._close_position()
            
            # Update performance metrics
            if len(self.equity_curve) > 1:
                equity_series = pd.Series([e['equity'] for e in self.equity_curve])
                returns = equity_series.pct_change().dropna()
                
                self.performance_metrics = {
                    'total_return': (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0],
                    'annual_return': (1 + returns.mean()) ** 252 - 1,
                    'annual_volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': (equity_series / equity_series.cummax() - 1).min(),
                    'current_drawdown': (equity_series.iloc[-1] / equity_series.cummax().iloc[-1] - 1),
                    'win_rate': len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0,
                    'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
                }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _log_trade(self, action: str, quantity: int, price: float):
        """
        Log trade information.
        
        Args:
            action: Action ("BUY" or "SELL")
            quantity: Quantity
            price: Price
        """
        if not self.config.log_trades:
            return
        
        trade_info = {
            'timestamp': datetime.now(),
            'symbol': self.config.trading_symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'position': self.current_position,
            'equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.config.initial_capital
        }
        
        self.trade_log.append(trade_info)
        
        # Log to file
        log_file = Path(self.config.output_dir) / "trades.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(trade_info) + '\n')
    
    async def _save_results(self):
        """Save trading results."""
        try:
            # Save equity curve
            if self.config.save_equity_curve and self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df.to_csv(Path(self.config.output_dir) / "equity_curve.csv", index=False)
            
            # Save performance metrics
            if self.performance_metrics:
                metrics_df = pd.DataFrame([self.performance_metrics])
                metrics_df.to_csv(Path(self.config.output_dir) / "performance_metrics.csv", index=False)
            
            # Save trade log
            if self.trade_log:
                trades_df = pd.DataFrame(self.trade_log)
                trades_df.to_csv(Path(self.config.output_dir) / "trades.csv", index=False)
            
            # Save configuration
            config_dict = {
                'config': self.config.__dict__,
                'performance_metrics': self.performance_metrics,
                'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.config.initial_capital,
                'total_trades': len(self.trade_log),
                'start_time': self.equity_curve[0]['timestamp'] if self.equity_curve else None,
                'end_time': self.equity_curve[-1]['timestamp'] if self.equity_curve else None
            }
            
            with open(Path(self.config.output_dir) / "results.json", 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Results saved to {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    async def run_trading_session(self, duration_minutes: Optional[int] = None):
        """
        Run trading session.
        
        Args:
            duration_minutes: Duration of trading session in minutes (None for indefinite)
        """
        try:
            logger.info("Starting trading session...")
            
            # Initialize start time
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes) if duration_minutes else None
            
            # Main trading loop
            while True:
                # Check if session should end
                if end_time and datetime.now() >= end_time:
                    logger.info("Trading session duration reached")
                    break
                
                # Check if market is open (simplified check)
                if not self._is_market_open():
                    await asyncio.sleep(60)  # Wait 1 minute
                    continue
                
                # Update market data
                await self._update_market_data()
                
                # Get trading signal
                signal = await self._get_trading_signal()
                
                # Execute trade
                if signal is not None:
                    await self._execute_trade(signal)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Log status
                if len(self.equity_curve) % 10 == 0:  # Log every 10 updates
                    logger.info(f"Trading update - Equity: {self.equity_curve[-1]['equity']:.2f}, "
                              f"Position: {self.current_position}, "
                              f"Daily P&L: {self.daily_pnl:.2f}")
                
                # Wait for next update
                await asyncio.sleep(self.config.update_frequency)
            
            # Close position at end of session
            await self._close_position()
            
            # Save results
            await self._save_results()
            
            logger.info("Trading session completed")
            
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
            await self._save_results()
    
    def _is_market_open(self) -> bool:
        """
        Check if market is open (simplified implementation).
        
        Returns:
            True if market is open
        """
        now = datetime.now()
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check time (simplified: 9:30 AM to 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    async def stop_trading_session(self):
        """Stop trading session."""
        logger.info("Stopping trading session...")
        
        # Close position
        await self._close_position()
        
        # Save results
        await self._save_results()
        
        # Disconnect
        await self.paper_trading.disconnect()
        await self.ibkr_client.disconnect()
        
        logger.info("Trading session stopped")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """
        Get trading summary.
        
        Returns:
            Trading summary dictionary
        """
        return {
            'config': self.config.__dict__,
            'performance_metrics': self.performance_metrics,
            'current_position': self.current_position,
            'total_trades': len(self.trade_log),
            'equity_curve_length': len(self.equity_curve),
            'start_time': self.equity_curve[0]['timestamp'] if self.equity_curve else None,
            'end_time': self.equity_curve[-1]['timestamp'] if self.equity_curve else None,
            'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.config.initial_capital
        }