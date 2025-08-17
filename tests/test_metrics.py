"""
Tests for metrics calculation module.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.utils.metrics import (
    calculate_performance_metrics,
    calculate_risk_metrics,
    calculate_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_beta,
    calculate_alpha,
    calculate_information_ratio,
    calculate_tracking_error,
    calculate_upside_capture,
    calculate_downside_capture
)


class TestDrawdownCalculations:
    """Test drawdown calculation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample equity curve
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.008, -0.03, 0.02, 0.01])
        self.equity = np.cumprod(1 + returns) * 100000
        self.returns = returns
    
    def test_calculate_drawdown_basic(self):
        """Test basic drawdown calculation."""
        drawdown = calculate_drawdown(self.equity)
        
        # Drawdown should be a numpy array
        assert isinstance(drawdown, np.ndarray)
        
        # Length should match equity curve
        assert len(drawdown) == len(self.equity)
        
        # Drawdown values should be <= 0
        assert np.all(drawdown <= 0)
        
        # First value should be 0 (no drawdown at start)
        assert drawdown[0] == 0
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        max_dd = calculate_max_drawdown(self.equity)
        
        # Max drawdown should be a negative float
        assert isinstance(max_dd, float)
        assert max_dd <= 0
        
        # Should be the minimum of all drawdowns
        drawdown = calculate_drawdown(self.equity)
        assert abs(max_dd - np.min(drawdown)) < 1e-10
    
    def test_drawdown_with_constant_equity(self):
        """Test drawdown calculation with constant equity."""
        constant_equity = np.array([100000, 100000, 100000, 100000])
        drawdown = calculate_drawdown(constant_equity)
        
        # All drawdowns should be zero
        assert np.all(drawdown == 0)
    
    def test_drawdown_with_monotonic_increase(self):
        """Test drawdown with monotonically increasing equity."""
        increasing_equity = np.array([100000, 110000, 120000, 130000])
        drawdown = calculate_drawdown(increasing_equity)
        
        # All drawdowns should be zero
        assert np.all(drawdown == 0)


class TestRatioCalculations:
    """Test ratio calculation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
        self.benchmark_returns = np.random.normal(0.0005, 0.015, 252)
        self.risk_free_rate = 0.02
        self.equity = np.cumprod(1 + self.returns) * 100000
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(
            self.returns, 
            risk_free_rate=self.risk_free_rate
        )
        
        # Sharpe ratio should be a float
        assert isinstance(sharpe, float)
        
        # Should be finite
        assert np.isfinite(sharpe)
        
        # Test with zero volatility (should return 0 or handle gracefully)
        zero_vol_returns = np.zeros(252)
        sharpe_zero = calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero == 0 or np.isnan(sharpe_zero)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(
            self.returns,
            risk_free_rate=self.risk_free_rate
        )
        
        # Sortino ratio should be a float
        assert isinstance(sortino, float)
        
        # Should be finite
        assert np.isfinite(sortino)
        
        # Sortino should typically be higher than Sharpe for same returns
        sharpe = calculate_sharpe_ratio(self.returns, self.risk_free_rate)
        # This is not always true, but often the case
        # assert sortino >= sharpe or abs(sortino - sharpe) < 0.1
    
    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(self.returns, self.equity)
        
        # Calmar ratio should be a float
        assert isinstance(calmar, float)
        
        # Should be finite (assuming non-zero max drawdown)
        if calculate_max_drawdown(self.equity) < 0:
            assert np.isfinite(calmar)
    
    def test_calculate_information_ratio(self):
        """Test Information ratio calculation."""
        info_ratio = calculate_information_ratio(
            self.returns,
            self.benchmark_returns
        )
        
        # Information ratio should be a float
        assert isinstance(info_ratio, float)
        
        # Should be finite if tracking error is non-zero
        tracking_err = calculate_tracking_error(self.returns, self.benchmark_returns)
        if tracking_err > 0:
            assert np.isfinite(info_ratio)


class TestRelativeMetrics:
    """Test relative performance metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
        self.benchmark_returns = np.random.normal(0.0005, 0.015, 252)
    
    def test_calculate_beta(self):
        """Test beta calculation."""
        beta = calculate_beta(self.returns, self.benchmark_returns)
        
        # Beta should be a float
        assert isinstance(beta, float)
        
        # Should be finite
        assert np.isfinite(beta)
        
        # Test with identical returns (beta should be 1)
        beta_identical = calculate_beta(self.returns, self.returns)
        assert abs(beta_identical - 1.0) < 1e-10
    
    def test_calculate_alpha(self):
        """Test alpha calculation."""
        alpha = calculate_alpha(
            self.returns,
            self.benchmark_returns,
            risk_free_rate=0.02
        )
        
        # Alpha should be a float
        assert isinstance(alpha, float)
        
        # Should be finite
        assert np.isfinite(alpha)
    
    def test_calculate_tracking_error(self):
        """Test tracking error calculation."""
        tracking_err = calculate_tracking_error(
            self.returns,
            self.benchmark_returns
        )
        
        # Tracking error should be a positive float
        assert isinstance(tracking_err, float)
        assert tracking_err >= 0
        
        # Should be finite
        assert np.isfinite(tracking_err)
        
        # Test with identical returns (tracking error should be 0)
        te_identical = calculate_tracking_error(self.returns, self.returns)
        assert abs(te_identical) < 1e-10
    
    def test_calculate_upside_capture(self):
        """Test upside capture ratio calculation."""
        upside = calculate_upside_capture(
            self.returns,
            self.benchmark_returns
        )
        
        # Upside capture should be a float
        assert isinstance(upside, float)
        
        # Should be finite if there are positive benchmark periods
        if np.any(self.benchmark_returns > 0):
            assert np.isfinite(upside)
        
        # Should be positive
        assert upside >= 0
    
    def test_calculate_downside_capture(self):
        """Test downside capture ratio calculation."""
        downside = calculate_downside_capture(
            self.returns,
            self.benchmark_returns
        )
        
        # Downside capture should be a float
        assert isinstance(downside, float)
        
        # Should be finite if there are negative benchmark periods
        if np.any(self.benchmark_returns < 0):
            assert np.isfinite(downside)
        
        # Should be positive
        assert downside >= 0


class TestRiskMetrics:
    """Test risk metric calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
    
    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        var_95 = calculate_var(self.returns, confidence=0.05)
        var_99 = calculate_var(self.returns, confidence=0.01)
        
        # VaR should be negative (represents loss)
        assert var_95 <= 0
        assert var_99 <= 0
        
        # 99% VaR should be more negative than 95% VaR
        assert var_99 <= var_95
        
        # VaR should be finite
        assert np.isfinite(var_95)
        assert np.isfinite(var_99)
    
    def test_calculate_cvar(self):
        """Test Conditional Value at Risk calculation."""
        cvar_95 = calculate_cvar(self.returns, confidence=0.05)
        cvar_99 = calculate_cvar(self.returns, confidence=0.01)
        
        # CVaR should be negative (represents loss)
        assert cvar_95 <= 0
        assert cvar_99 <= 0
        
        # 99% CVaR should be more negative than 95% CVaR
        assert cvar_99 <= cvar_95
        
        # CVaR should be more negative than corresponding VaR
        var_95 = calculate_var(self.returns, confidence=0.05)
        var_99 = calculate_var(self.returns, confidence=0.01)
        assert cvar_95 <= var_95
        assert cvar_99 <= var_99
    
    def test_var_cvar_relationship(self):
        """Test relationship between VaR and CVaR."""
        confidence_levels = [0.01, 0.05, 0.10]
        
        for confidence in confidence_levels:
            var = calculate_var(self.returns, confidence)
            cvar = calculate_cvar(self.returns, confidence)
            
            # CVaR should be more negative than VaR
            assert cvar <= var, f"CVaR should be <= VaR at {confidence} confidence"


class TestPerformanceMetrics:
    """Test comprehensive performance metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
        self.benchmark_returns = np.random.normal(0.0005, 0.015, 252)
    
    def test_calculate_performance_metrics(self):
        """Test comprehensive performance metrics calculation."""
        metrics = calculate_performance_metrics(
            self.returns,
            self.benchmark_returns,
            risk_free_rate=0.02
        )
        
        # Should return a dictionary
        assert isinstance(metrics, dict)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annual_return', 'annual_volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'beta', 'alpha', 'information_ratio',
            'tracking_error', 'upside_capture', 'downside_capture'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Invalid type for {metric}"
            assert np.isfinite(metrics[metric]) or np.isnan(metrics[metric]), f"Invalid value for {metric}"
    
    def test_calculate_risk_metrics(self):
        """Test comprehensive risk metrics calculation."""
        metrics = calculate_risk_metrics(self.returns)
        
        # Should return a dictionary
        assert isinstance(metrics, dict)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'var_95', 'var_99', 'cvar_95', 'cvar_99',
            'skewness', 'kurtosis', 'max_drawdown'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Invalid type for {metric}"
            assert np.isfinite(metrics[metric]) or np.isnan(metrics[metric]), f"Invalid value for {metric}"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_returns(self):
        """Test handling of empty returns array."""
        empty_returns = np.array([])
        
        with pytest.raises(ValueError):
            calculate_sharpe_ratio(empty_returns)
    
    def test_single_return(self):
        """Test handling of single return value."""
        single_return = np.array([0.01])
        
        # Should handle gracefully (might return 0 or NaN for ratios)
        sharpe = calculate_sharpe_ratio(single_return)
        assert np.isnan(sharpe) or sharpe == 0
    
    def test_all_zero_returns(self):
        """Test handling of all zero returns."""
        zero_returns = np.zeros(252)
        
        sharpe = calculate_sharpe_ratio(zero_returns, risk_free_rate=0.02)
        # With zero volatility, Sharpe ratio should be undefined (NaN) or 0
        assert np.isnan(sharpe) or sharpe == 0
    
    def test_constant_equity(self):
        """Test metrics with constant equity (no variation)."""
        constant_equity = np.full(252, 100000)
        
        max_dd = calculate_max_drawdown(constant_equity)
        assert max_dd == 0
    
    def test_infinite_values(self):
        """Test handling of infinite values in returns."""
        returns_with_inf = np.array([0.01, np.inf, -0.02, 0.015])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, FloatingPointError)) or np.isnan:
            calculate_sharpe_ratio(returns_with_inf)
    
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)  # Different length
        
        with pytest.raises(ValueError):
            calculate_beta(returns, benchmark_returns)