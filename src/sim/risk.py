"""
Risk management module for the RL trading system.

This module provides comprehensive risk management including
position sizing, stop-loss, take-profit, and drawdown controls.
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
class RiskConfig:
    """Risk management configuration."""
    risk_per_trade_frac: float = 0.02  # 2% risk per trade
    stop_r_multiple: float = 1.0       # 1R stop loss
    tp_r_multiple: float = 1.5         # 1.5R take profit
    max_daily_loss_r: float = 3.0      # 3R maximum daily loss
    max_position_size: int = 100       # Maximum position size
    max_leverage: float = 1.0          # Maximum leverage
    drawdown_limit: float = 0.15       # 15% maximum drawdown
    var_confidence: float = 0.95       # 95% VaR confidence
    cvar_confidence: float = 0.95      # 95% CVaR confidence
    correlation_limit: float = 0.8     # Maximum correlation


class RiskManager:
    """
    Comprehensive risk management system.
    
    This class provides position sizing, risk limits, and drawdown controls
    for the RL trading system.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize risk manager.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.risk_config = RiskConfig(**settings.get('risk', {}))
        
        # Risk tracking
        self.daily_pnl: List[float] = []
        self.position_history: List[Dict] = []
        self.risk_metrics: Dict[str, float] = {}
        
        # Risk limits
        self.daily_loss_count = 0
        self.consecutive_losses = 0
        
    def calculate_position_size(self, 
                              account_equity: float, 
                              entry_price: float, 
                              stop_price: float,
                              atr: float = None) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            account_equity: Current account equity
            entry_price: Entry price
            stop_price: Stop loss price
            atr: Average True Range (optional)
            
        Returns:
            Position size in contracts
        """
        # Calculate risk per trade
        risk_per_trade = account_equity * self.risk_config.risk_per_trade_frac
        
        # Calculate stop distance
        if atr is not None:
            # Use ATR-based stop
            stop_distance = atr * self.risk_config.stop_r_multiple
        else:
            # Use price-based stop
            stop_distance = abs(entry_price - stop_price)
        
        # Calculate position size
        if stop_distance > 0:
            risk_per_contract = stop_distance * 5.0  # $5 per point for MES
            position_size = int(risk_per_trade / risk_per_contract)
        else:
            position_size = 0
        
        # Apply position limits
        position_size = min(position_size, self.risk_config.max_position_size)
        position_size = min(position_size, int(account_equity / entry_price * self.risk_config.max_leverage))
        
        return max(0, position_size)
    
    def calculate_stop_prices(self, 
                            entry_price: float, 
                            position_size: int,
                            atr: float = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit prices.
        
        Args:
            entry_price: Entry price
            position_size: Position size
            atr: Average True Range (optional)
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Calculate stop distance
        if atr is not None:
            stop_distance = atr * self.risk_config.stop_r_multiple
            tp_distance = atr * self.risk_config.tp_r_multiple
        else:
            # Default stop distance (1% of price)
            stop_distance = entry_price * 0.01
            tp_distance = entry_price * 0.015
        
        # Calculate prices
        if position_size > 0:  # Long position
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        elif position_size < 0:  # Short position
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        else:
            stop_loss = take_profit = entry_price
        
        return stop_loss, take_profit
    
    def check_risk_limits(self, 
                         position_size: int, 
                         account_equity: float,
                         current_price: float) -> Dict[str, bool]:
        """
        Check if position size exceeds risk limits.
        
        Args:
            position_size: Proposed position size
            account_equity: Current account equity
            current_price: Current market price
            
        Returns:
            Dictionary with risk limit checks
        """
        checks = {
            'position_size_ok': abs(position_size) <= self.risk_config.max_position_size,
            'leverage_ok': abs(position_size * current_price) <= account_equity * self.risk_config.max_leverage,
            'risk_per_trade_ok': True,  # Will be checked in calculate_position_size
            'daily_loss_ok': self.daily_loss_count < 5,  # Max 5 losing days
            'consecutive_losses_ok': self.consecutive_losses < 3,  # Max 3 consecutive losses
        }
        
        return checks
    
    def update_risk_metrics(self, 
                           account_equity: float,
                           current_position: int,
                           current_price: float) -> Dict[str, float]:
        """
        Update risk metrics.
        
        Args:
            account_equity: Current account equity
            current_position: Current position size
            current_price: Current market price
            
        Returns:
            Dictionary with updated risk metrics
        """
        # Calculate position value
        position_value = current_position * current_price
        
        # Calculate leverage
        leverage = abs(position_value) / account_equity if account_equity > 0 else 0
        
        # Calculate drawdown
        max_equity = max(self.daily_pnl + [account_equity]) if self.daily_pnl else account_equity
        drawdown = (max_equity - account_equity) / max_equity if max_equity > 0 else 0
        
        # Update risk metrics
        self.risk_metrics = {
            'account_equity': account_equity,
            'position_value': position_value,
            'leverage': leverage,
            'drawdown': drawdown,
            'position_size': current_position,
            'daily_pnl': self.daily_pnl[-1] if self.daily_pnl else 0,
            'consecutive_losses': self.consecutive_losses,
            'daily_loss_count': self.daily_loss_count
        }
        
        return self.risk_metrics
    
    def check_drawdown_limits(self, account_equity: float) -> bool:
        """
        Check if drawdown limits are exceeded.
        
        Args:
            account_equity: Current account equity
            
        Returns:
            True if within drawdown limits
        """
        if not self.daily_pnl:
            return True
        
        max_equity = max(self.daily_pnl + [account_equity])
        drawdown = (max_equity - account_equity) / max_equity if max_equity > 0 else 0
        
        return drawdown <= self.risk_config.drawdown_limit
    
    def check_daily_loss_limits(self, daily_pnl: float) -> bool:
        """
        Check if daily loss limits are exceeded.
        
        Args:
            daily_pnl: Daily P&L
            
        Returns:
            True if within daily loss limits
        """
        if daily_pnl < 0:
            self.daily_loss_count += 1
            self.consecutive_losses += 1
        else:
            self.daily_loss_count = 0
            self.consecutive_losses = 0
        
        # Check if daily loss exceeds limit
        max_daily_loss = self.risk_config.max_daily_loss_r * 1000  # Assuming $1000 account
        return abs(daily_pnl) <= max_daily_loss
    
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence: Confidence level (optional)
            
        Returns:
            VaR value
        """
        if confidence is None:
            confidence = self.risk_config.var_confidence
        
        if len(returns) == 0:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return var
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of returns
            confidence: Confidence level (optional)
            
        Returns:
            CVaR value
        """
        if confidence is None:
            confidence = self.risk_config.cvar_confidence
        
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    def calculate_portfolio_metrics(self, 
                                  positions: Dict[str, int],
                                  prices: Dict[str, float],
                                  returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            positions: Dictionary of positions by symbol
            prices: Dictionary of current prices by symbol
            returns: DataFrame of historical returns
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        
        for symbol, position in positions.items():
            if symbol in returns.columns and position != 0:
                portfolio_returns += returns[symbol] * position
        
        # Calculate risk metrics
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        portfolio_var = self.calculate_var(portfolio_returns)
        portfolio_cvar = self.calculate_cvar(portfolio_returns)
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Calculate maximum correlation
        max_correlation = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].max()
        
        metrics = {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var': portfolio_var,
            'portfolio_cvar': portfolio_cvar,
            'max_correlation': max_correlation,
            'diversification_ratio': 1.0 / max_correlation if max_correlation > 0 else 1.0
        }
        
        return metrics
    
    def should_reduce_position(self, 
                             current_position: int,
                             account_equity: float,
                             current_price: float) -> bool:
        """
        Determine if position should be reduced due to risk limits.
        
        Args:
            current_position: Current position size
            account_equity: Current account equity
            current_price: Current market price
            
        Returns:
            True if position should be reduced
        """
        # Check leverage
        position_value = abs(current_position * current_price)
        leverage = position_value / account_equity if account_equity > 0 else 0
        
        if leverage > self.risk_config.max_leverage:
            return True
        
        # Check drawdown
        if not self.check_drawdown_limits(account_equity):
            return True
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            return True
        
        return False
    
    def get_position_adjustment(self, 
                              current_position: int,
                              account_equity: float,
                              current_price: float) -> int:
        """
        Get position adjustment based on risk limits.
        
        Args:
            current_position: Current position size
            account_equity: Current account equity
            current_price: Current market price
            
        Returns:
            Position adjustment (positive to increase, negative to decrease)
        """
        if self.should_reduce_position(current_position, account_equity, current_price):
            # Reduce position to within limits
            max_position_value = account_equity * self.risk_config.max_leverage
            max_position = int(max_position_value / current_price)
            
            adjustment = max_position - current_position
        else:
            adjustment = 0
        
        return adjustment
    
    def record_position(self, 
                       symbol: str,
                       position: int,
                       entry_price: float,
                       timestamp: pd.Timestamp) -> None:
        """
        Record position change.
        
        Args:
            symbol: Trading symbol
            position: Position size
            entry_price: Entry price
            timestamp: Timestamp of position change
        """
        position_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'position': position,
            'entry_price': entry_price
        }
        
        self.position_history.append(position_record)
    
    def get_risk_report(self) -> Dict[str, any]:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dictionary with risk report
        """
        report = {
            'risk_config': self.risk_config.__dict__,
            'risk_metrics': self.risk_metrics,
            'daily_pnl': self.daily_pnl,
            'position_history': self.position_history,
            'risk_limits': {
                'drawdown_limit': self.risk_config.drawdown_limit,
                'max_leverage': self.risk_config.max_leverage,
                'max_position_size': self.risk_config.max_position_size,
                'risk_per_trade': self.risk_config.risk_per_trade_frac
            }
        }
        
        return report
    
    def reset_daily_metrics(self) -> None:
        """Reset daily risk metrics."""
        self.daily_pnl = []
        self.daily_loss_count = 0
        self.consecutive_losses = 0
        logger.info("Daily risk metrics reset")
    
    def save_risk_report(self, filepath: str) -> None:
        """Save risk report to file."""
        report = self.get_risk_report()
        
        # Convert to DataFrame for easier saving
        df = pd.DataFrame(report['position_history'])
        df.to_csv(filepath, index=False)
        
        logger.info(f"Risk report saved to {filepath}")
    
    def load_risk_report(self, filepath: str) -> None:
        """Load risk report from file."""
        try:
            df = pd.read_csv(filepath)
            self.position_history = df.to_dict('records')
            logger.info(f"Risk report loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading risk report: {e}")


class RiskMonitor:
    """
    Real-time risk monitoring system.
    
    This class provides real-time risk monitoring and alerts.
    """
    
    def __init__(self, risk_manager: RiskManager):
        """
        Initialize risk monitor.
        
        Args:
            risk_manager: Risk manager instance
        """
        self.risk_manager = risk_manager
        self.alerts: List[Dict] = []
        
    def monitor_position(self, 
                        symbol: str,
                        position: int,
                        account_equity: float,
                        current_price: float) -> List[Dict]:
        """
        Monitor position risk.
        
        Args:
            symbol: Trading symbol
            position: Position size
            account_equity: Account equity
            current_price: Current price
            
        Returns:
            List of risk alerts
        """
        alerts = []
        
        # Check risk limits
        risk_checks = self.risk_manager.check_risk_limits(position, account_equity, current_price)
        
        # Generate alerts
        for check_name, is_ok in risk_checks.items():
            if not is_ok:
                alert = {
                    'timestamp': pd.Timestamp.now(),
                    'type': 'risk_limit',
                    'symbol': symbol,
                    'message': f'{check_name} exceeded',
                    'severity': 'high'
                }
                alerts.append(alert)
                self.alerts.append(alert)
        
        # Check drawdown
        if not self.risk_manager.check_drawdown_limits(account_equity):
            alert = {
                'timestamp': pd.Timestamp.now(),
                'type': 'drawdown',
                'symbol': symbol,
                'message': 'Drawdown limit exceeded',
                'severity': 'critical'
            }
            alerts.append(alert)
            self.alerts.append(alert)
        
        # Check daily loss
        if self.risk_manager.daily_pnl:
            daily_pnl = self.risk_manager.daily_pnl[-1]
            if not self.risk_manager.check_daily_loss_limits(daily_pnl):
                alert = {
                    'timestamp': pd.Timestamp.now(),
                    'type': 'daily_loss',
                    'symbol': symbol,
                    'message': 'Daily loss limit exceeded',
                    'severity': 'high'
                }
                alerts.append(alert)
                self.alerts.append(alert)
        
        return alerts
    
    def get_alerts(self, severity: str = None) -> List[Dict]:
        """
        Get risk alerts.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of alerts
        """
        if severity is None:
            return self.alerts
        
        return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []
        logger.info("Risk alerts cleared")