"""
Real-time monitoring dashboard for multi-ticker RL trading system.

This module provides a comprehensive real-time dashboard for monitoring
multi-ticker RL trading strategies, including performance metrics,
regime analysis, and alert management.
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask import Flask

from ..utils.logging import get_logger
from ..utils.config_loader import load_config
from .performance_tracker import PerformanceTracker
from .regime_analyzer import RegimeAnalyzer

logger = get_logger(__name__)


class AlertManager:
    """
    Alert manager for performance degradation and other events.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alerts = []
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.notification_callbacks = []
        
        # Default thresholds
        self.default_thresholds = {
            'drawdown': 0.1,  # 10% drawdown
            'sharpe_ratio': 0.5,  # Sharpe ratio below 0.5
            'win_rate': 0.4,  # Win rate below 40%
            'volatility': 0.3,  # Volatility above 30%
            'max_position_duration': 5,  # Max position duration in days
            'consecutive_losses': 5,  # Consecutive losses
            'regime_change': True,  # Alert on regime change
            'system_error': True  # Alert on system errors
        }
        
        # Merge with config thresholds
        self.thresholds = {**self.default_thresholds, **self.alert_thresholds}
        
    def add_notification_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add notification callback.
        
        Args:
            callback: Callback function to call when alert is triggered
        """
        self.notification_callbacks.append(callback)
        
    def check_performance_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for performance alerts.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Check drawdown
        if metrics.get('max_drawdown', 0) > self.thresholds['drawdown']:
            alert = {
                'type': 'drawdown',
                'severity': 'high' if metrics['max_drawdown'] > 0.15 else 'medium',
                'message': f"Drawdown exceeded threshold: {metrics['max_drawdown']:.2%} > {self.thresholds['drawdown']:.2%}",
                'timestamp': datetime.now(),
                'value': metrics['max_drawdown'],
                'threshold': self.thresholds['drawdown']
            }
            triggered_alerts.append(alert)
            
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < self.thresholds['sharpe_ratio']:
            alert = {
                'type': 'sharpe_ratio',
                'severity': 'medium',
                'message': f"Sharpe ratio below threshold: {metrics['sharpe_ratio']:.2f} < {self.thresholds['sharpe_ratio']:.2f}",
                'timestamp': datetime.now(),
                'value': metrics['sharpe_ratio'],
                'threshold': self.thresholds['sharpe_ratio']
            }
            triggered_alerts.append(alert)
            
        # Check win rate
        if metrics.get('win_rate', 0) < self.thresholds['win_rate']:
            alert = {
                'type': 'win_rate',
                'severity': 'medium',
                'message': f"Win rate below threshold: {metrics['win_rate']:.2%} < {self.thresholds['win_rate']:.2%}",
                'timestamp': datetime.now(),
                'value': metrics['win_rate'],
                'threshold': self.thresholds['win_rate']
            }
            triggered_alerts.append(alert)
            
        # Check volatility
        if metrics.get('annual_volatility', 0) > self.thresholds['volatility']:
            alert = {
                'type': 'volatility',
                'severity': 'medium',
                'message': f"Volatility exceeded threshold: {metrics['annual_volatility']:.2%} > {self.thresholds['volatility']:.2%}",
                'timestamp': datetime.now(),
                'value': metrics['annual_volatility'],
                'threshold': self.thresholds['volatility']
            }
            triggered_alerts.append(alert)
            
        # Add alerts to list
        for alert in triggered_alerts:
            self.add_alert(alert)
            
        return triggered_alerts
        
    def check_trade_alerts(self, trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for trade alerts.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Check position duration
        if 'entry_time' in trade and 'exit_time' in trade:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            duration = (exit_time - entry_time).total_seconds() / 86400  # days
            
            if duration > self.thresholds['max_position_duration']:
                alert = {
                    'type': 'position_duration',
                    'severity': 'low',
                    'message': f"Position duration exceeded threshold: {duration:.1f} days > {self.thresholds['max_position_duration']} days",
                    'timestamp': datetime.now(),
                    'value': duration,
                    'threshold': self.thresholds['max_position_duration'],
                    'ticker': trade.get('ticker', 'unknown')
                }
                triggered_alerts.append(alert)
                
        # Add alerts to list
        for alert in triggered_alerts:
            self.add_alert(alert)
            
        return triggered_alerts
        
    def check_regime_alert(self, from_regime: str, to_regime: str) -> Dict[str, Any]:
        """
        Check for regime change alert.
        
        Args:
            from_regime: Previous regime
            to_regime: New regime
            
        Returns:
            Triggered alert or None
        """
        if not self.thresholds['regime_change']:
            return None
            
        alert = {
            'type': 'regime_change',
            'severity': 'low',
            'message': f"Market regime changed from {from_regime} to {to_regime}",
            'timestamp': datetime.now(),
            'from_regime': from_regime,
            'to_regime': to_regime
        }
        
        self.add_alert(alert)
        return alert
        
    def add_system_error_alert(self, error_message: str, error_type: str = 'system_error') -> Dict[str, Any]:
        """
        Add system error alert.
        
        Args:
            error_message: Error message
            error_type: Type of error
            
        Returns:
            Created alert
        """
        if not self.thresholds['system_error']:
            return None
            
        alert = {
            'type': error_type,
            'severity': 'high',
            'message': f"System error: {error_message}",
            'timestamp': datetime.now()
        }
        
        self.add_alert(alert)
        return alert
        
    def add_alert(self, alert: Dict[str, Any]) -> None:
        """
        Add alert to list and trigger notifications.
        
        Args:
            alert: Alert dictionary
        """
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
        # Trigger notifications
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
                
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Args:
            severity: Filter by severity (optional)
            
        Returns:
            List of active alerts
        """
        # Get alerts from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        active_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        # Filter by severity
        if severity:
            active_alerts = [alert for alert in active_alerts if alert['severity'] == severity]
            
        # Sort by timestamp (newest first)
        active_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return active_alerts
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get alert summary.
        
        Returns:
            Alert summary dictionary
        """
        # Get alerts from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        # Count by severity
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for alert in recent_alerts:
            severity = alert['severity']
            if severity in severity_counts:
                severity_counts[severity] += 1
                
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            if alert_type not in type_counts:
                type_counts[alert_type] = 0
            type_counts[alert_type] += 1
            
        return {
            'total_alerts': len(recent_alerts),
            'severity_counts': severity_counts,
            'type_counts': type_counts,
            'most_recent_alert': recent_alerts[0] if recent_alerts else None
        }


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for multi-ticker RL trading system.
    """
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        regime_analyzer: Optional[RegimeAnalyzer] = None,
        config: Optional[Dict[str, Any]] = None,
        host: str = '127.0.0.1',
        port: int = 8050,
        debug: bool = False
    ):
        """
        Initialize monitoring dashboard.
        
        Args:
            performance_tracker: Performance tracker instance
            regime_analyzer: Regime analyzer instance (optional)
            config: Configuration dictionary
            host: Dashboard host
            port: Dashboard port
            debug: Debug mode
        """
        self.performance_tracker = performance_tracker
        self.regime_analyzer = regime_analyzer
        self.config = config or {}
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.config)
        
        # Setup dashboard
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.app.title = "Multi-Ticker RL Trading Dashboard"
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Background update thread
        self.update_thread = None
        self.stop_event = threading.Event()
        
    def _setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Multi-Ticker RL Trading Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Return"),
                        dbc.CardBody([
                            html.H4(id="total-return", children="0.00%")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sharpe Ratio"),
                        dbc.CardBody([
                            html.H4(id="sharpe-ratio", children="0.00")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Max Drawdown"),
                        dbc.CardBody([
                            html.H4(id="max-drawdown", children="0.00%")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Win Rate"),
                        dbc.CardBody([
                            html.H4(id="win-rate", children="0.00%")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Tabs
            dbc.Tabs([
                # Performance tab
                dbc.Tab(label="Performance", tab_id="performance-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="equity-curve")
                        ], width=8),
                        
                        dbc.Col([
                            dcc.Graph(id="drawdown-curve")
                        ], width=4)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="returns-distribution")
                        ], width=6),
                        
                        dbc.Col([
                            dcc.Graph(id="rolling-metrics")
                        ], width=6)
                    ])
                ]),
                
                # Tickers tab
                dbc.Tab(label="Tickers", tab_id="tickers-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="ticker-performance")
                        ], width=12)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="ticker-contribution")
                        ], width=12)
                    ])
                ]),
                
                # Regimes tab
                dbc.Tab(label="Regimes", tab_id="regimes-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="regime-equity")
                        ], width=8),
                        
                        dbc.Col([
                            dcc.Graph(id="regime-performance")
                        ], width=4)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="regime-transitions")
                        ], width=6),
                        
                        dbc.Col([
                            dcc.Graph(id="regime-durations")
                        ], width=6)
                    ])
                ]),
                
                # Alerts tab
                dbc.Tab(label="Alerts", tab_id="alerts-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Alert Summary"),
                                dbc.CardBody([
                                    html.Div(id="alert-summary")
                                ])
                            ])
                        ], width=12)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Recent Alerts"),
                                dbc.CardBody([
                                    html.Div(id="recent-alerts")
                                ])
                            ])
                        ], width=12)
                    ])
                ])
            ], id="tabs", active_tab="performance-tab"),
            
            # Auto-refresh interval
            dcc.Interval(
                id="interval-component",
                interval=30*1000,  # 30 seconds
                n_intervals=0
            ),
            
            # Store for data
            dcc.Store(id="performance-data"),
            dcc.Store(id="regime-data"),
            dcc.Store(id="alert-data")
        ], fluid=True)
        
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks."""
        
        # Update summary cards
        @self.app.callback(
            [
                Output("total-return", "children"),
                Output("sharpe-ratio", "children"),
                Output("max-drawdown", "children"),
                Output("win-rate", "children")
            ],
            [
                Input("interval-component", "n_intervals")
            ]
        )
        def update_summary_cards(n):
            # Calculate metrics
            metrics = self.performance_tracker.calculate_metrics()
            
            # Format values
            total_return = f"{metrics.total_return:.2%}"
            sharpe_ratio = f"{metrics.sharpe_ratio:.2f}"
            max_drawdown = f"{metrics.max_drawdown:.2%}"
            win_rate = f"{metrics.win_rate:.2%}"
            
            return total_return, sharpe_ratio, max_drawdown, win_rate
            
        # Update performance tab
        @self.app.callback(
            [
                Output("equity-curve", "figure"),
                Output("drawdown-curve", "figure"),
                Output("returns-distribution", "figure"),
                Output("rolling-metrics", "figure")
            ],
            [
                Input("interval-component", "n_intervals")
            ]
        )
        def update_performance_tab(n):
            # Calculate metrics
            metrics = self.performance_tracker.calculate_metrics()
            
            # Create equity curve figure
            equity_fig = self._create_equity_curve_figure()
            
            # Create drawdown curve figure
            drawdown_fig = self._create_drawdown_curve_figure()
            
            # Create returns distribution figure
            returns_fig = self._create_returns_distribution_figure()
            
            # Create rolling metrics figure
            rolling_fig = self._create_rolling_metrics_figure()
            
            return equity_fig, drawdown_fig, returns_fig, rolling_fig
            
        # Update tickers tab
        @self.app.callback(
            [
                Output("ticker-performance", "figure"),
                Output("ticker-contribution", "figure")
            ],
            [
                Input("interval-component", "n_intervals")
            ]
        )
        def update_tickers_tab(n):
            # Calculate metrics
            metrics = self.performance_tracker.calculate_metrics()
            
            # Create ticker performance figure
            ticker_perf_fig = self._create_ticker_performance_figure()
            
            # Create ticker contribution figure
            ticker_contrib_fig = self._create_ticker_contribution_figure()
            
            return ticker_perf_fig, ticker_contrib_fig
            
        # Update regimes tab
        @self.app.callback(
            [
                Output("regime-equity", "figure"),
                Output("regime-performance", "figure"),
                Output("regime-transitions", "figure"),
                Output("regime-durations", "figure")
            ],
            [
                Input("interval-component", "n_intervals")
            ]
        )
        def update_regimes_tab(n):
            # Default figures
            regime_equity_fig = go.Figure()
            regime_perf_fig = go.Figure()
            regime_trans_fig = go.Figure()
            regime_dur_fig = go.Figure()
            
            # Update if regime analyzer is available
            if self.regime_analyzer:
                # Create regime equity figure
                regime_equity_fig = self._create_regime_equity_figure()
                
                # Create regime performance figure
                regime_perf_fig = self._create_regime_performance_figure()
                
                # Create regime transitions figure
                regime_trans_fig = self._create_regime_transitions_figure()
                
                # Create regime durations figure
                regime_dur_fig = self._create_regime_durations_figure()
                
            return regime_equity_fig, regime_perf_fig, regime_trans_fig, regime_dur_fig
            
        # Update alerts tab
        @self.app.callback(
            [
                Output("alert-summary", "children"),
                Output("recent-alerts", "children")
            ],
            [
                Input("interval-component", "n_intervals")
            ]
        )
        def update_alerts_tab(n):
            # Get alert summary
            summary = self.alert_manager.get_alert_summary()
            
            # Create alert summary
            alert_summary = html.Div([
                html.P(f"Total Alerts: {summary['total_alerts']}"),
                html.P(f"High Severity: {summary['severity_counts']['high']}"),
                html.P(f"Medium Severity: {summary['severity_counts']['medium']}"),
                html.P(f"Low Severity: {summary['severity_counts']['low']}")
            ])
            
            # Get recent alerts
            alerts = self.alert_manager.get_active_alerts()
            
            # Create recent alerts
            recent_alerts = []
            for alert in alerts[:10]:  # Show top 10
                severity_color = {
                    'high': 'danger',
                    'medium': 'warning',
                    'low': 'info'
                }.get(alert['severity'], 'secondary')
                
                alert_item = dbc.Alert(
                    [
                        html.Strong(f"{alert['type'].replace('_', ' ').title()}"),
                        html.Br(),
                        alert['message'],
                        html.Br(),
                        html.Small(f"{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    ],
                    color=severity_color,
                    className="mb-2"
                )
                recent_alerts.append(alert_item)
                
            if not recent_alerts:
                recent_alerts = [html.P("No recent alerts")]
                
            return alert_summary, recent_alerts
            
    def _create_equity_curve_figure(self) -> go.Figure:
        """
        Create equity curve figure.
        
        Returns:
            Plotly figure
        """
        if not self.performance_tracker.timestamps:
            return go.Figure()
            
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': self.performance_tracker.timestamps,
            'equity': self.performance_tracker.equity_curve
        })
        equity_df.set_index('timestamp', inplace=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df['equity'],
            mode='lines',
            name='Portfolio Equity',
            line=dict(color='blue', width=2)
        ))
        
        # Add initial capital line
        fig.add_hline(
            y=self.performance_tracker.initial_capital,
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Capital"
        )
        
        # Add peak equity line
        metrics = self.performance_tracker.calculate_metrics()
        fig.add_hline(
            y=metrics.peak_equity,
            line_dash="dash",
            line_color="green",
            annotation_text="Peak Equity"
        )
        
        # Update layout
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig
        
    def _create_drawdown_curve_figure(self) -> go.Figure:
        """
        Create drawdown curve figure.
        
        Returns:
            Plotly figure
        """
        if not self.performance_tracker.drawdown_history:
            return go.Figure()
            
        # Create drawdown DataFrame
        drawdown_df = pd.DataFrame({
            'timestamp': self.performance_tracker.timestamps,
            'drawdown': self.performance_tracker.drawdown_history
        })
        drawdown_df.set_index('timestamp', inplace=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add drawdown curve
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        # Update layout
        fig.update_layout(
            title="Portfolio Drawdown Curve",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            template='plotly_dark'
        )
        
        # Format y-axis as percentage
        fig.update_yaxes(tickformat=".1%")
        
        return fig
        
    def _create_returns_distribution_figure(self) -> go.Figure:
        """
        Create returns distribution figure.
        
        Returns:
            Plotly figure
        """
        metrics = self.performance_tracker.calculate_metrics()
        
        if not metrics.daily_returns:
            return go.Figure()
            
        # Create figure
        fig = go.Figure()
        
        # Add returns histogram
        fig.add_trace(go.Histogram(
            x=metrics.daily_returns,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mu, sigma = stats.norm.fit(metrics.daily_returns)
        x = np.linspace(min(metrics.daily_returns), max(metrics.daily_returns), 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y * len(metrics.daily_returns) * (max(metrics.daily_returns) - min(metrics.daily_returns)) / 50,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        # Add VaR lines
        fig.add_vline(
            x=metrics.var_95,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"VaR 95%: {metrics.var_95:.2%}"
        )
        
        fig.add_vline(
            x=metrics.var_99,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR 99%: {metrics.var_99:.2%}"
        )
        
        # Update layout
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            template='plotly_dark'
        )
        
        # Format x-axis as percentage
        fig.update_xaxes(tickformat=".1%")
        
        return fig
        
    def _create_rolling_metrics_figure(self) -> go.Figure:
        """
        Create rolling metrics figure.
        
        Returns:
            Plotly figure
        """
        metrics = self.performance_tracker.calculate_metrics()
        
        if not metrics.daily_returns:
            return go.Figure()
            
        # Create returns DataFrame
        returns_df = pd.DataFrame({
            'timestamp': self.performance_tracker.timestamps[1:],  # Skip first as no return
            'returns': metrics.daily_returns
        })
        returns_df.set_index('timestamp', inplace=True)
        
        # Calculate rolling metrics
        window = 63  # ~3 months
        rolling_sharpe = returns_df['returns'].rolling(window).mean() / returns_df['returns'].rolling(window).std() * np.sqrt(252)
        rolling_vol = returns_df['returns'].rolling(window).std() * np.sqrt(252)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.1
        )
        
        # Add rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Rolling Metrics ({window} days)",
            template='plotly_dark',
            height=600
        )
        
        # Format y-axes
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=2, col=1)
        
        return fig
        
    def _create_ticker_performance_figure(self) -> go.Figure:
        """
        Create ticker performance figure.
        
        Returns:
            Plotly figure
        """
        metrics = self.performance_tracker.calculate_metrics()
        
        if not metrics.ticker_metrics:
            return go.Figure()
            
        # Create ticker metrics DataFrame
        ticker_data = []
        for ticker, ticker_metrics in metrics.ticker_metrics.items():
            ticker_data.append({
                'ticker': ticker,
                'total_return': ticker_metrics['total_return'],
                'sharpe_ratio': ticker_metrics['sharpe_ratio'],
                'max_drawdown': ticker_metrics['max_drawdown'],
                'win_rate': ticker_metrics['win_rate']
            })
            
        ticker_df = pd.DataFrame(ticker_data)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add total return
        fig.add_trace(
            go.Bar(
                x=ticker_df['ticker'],
                y=ticker_df['total_return'],
                name='Total Return',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Add Sharpe ratio
        fig.add_trace(
            go.Bar(
                x=ticker_df['ticker'],
                y=ticker_df['sharpe_ratio'],
                name='Sharpe Ratio',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Add max drawdown
        fig.add_trace(
            go.Bar(
                x=ticker_df['ticker'],
                y=ticker_df['max_drawdown'],
                name='Max Drawdown',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # Add win rate
        fig.add_trace(
            go.Bar(
                x=ticker_df['ticker'],
                y=ticker_df['win_rate'],
                name='Win Rate',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Ticker Performance Comparison",
            template='plotly_dark',
            height=600
        )
        
        # Format y-axes
        fig.update_yaxes(title_text="Return", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate", tickformat=".1%", row=2, col=2)
        
        return fig
        
    def _create_ticker_contribution_figure(self) -> go.Figure:
        """
        Create ticker contribution figure.
        
        Returns:
            Plotly figure
        """
        metrics = self.performance_tracker.calculate_metrics()
        
        if not metrics.ticker_metrics:
            return go.Figure()
            
        # Create ticker contribution data
        ticker_data = []
        for ticker, ticker_metrics in metrics.ticker_metrics.items():
            # Estimate contribution based on total return
            # This is a simplified approach - in practice, you'd use actual P&L
            contribution = ticker_metrics['total_return']
            ticker_data.append({
                'ticker': ticker,
                'contribution': contribution
            })
            
        ticker_df = pd.DataFrame(ticker_data)
        
        # Create figure
        fig = go.Figure()
        
        # Add contribution bars
        colors = ['green' if x >= 0 else 'red' for x in ticker_df['contribution']]
        
        fig.add_trace(go.Bar(
            x=ticker_df['ticker'],
            y=ticker_df['contribution'],
            name='Contribution',
            marker_color=colors
        ))
        
        # Update layout
        fig.update_layout(
            title="Ticker Contribution to Portfolio Return",
            xaxis_title="Ticker",
            yaxis_title="Contribution (%)",
            template='plotly_dark'
        )
        
        # Format y-axis as percentage
        fig.update_yaxes(tickformat=".1%")
        
        return fig
        
    def _create_regime_equity_figure(self) -> go.Figure:
        """
        Create regime equity figure.
        
        Returns:
            Plotly figure
        """
        if not self.regime_analyzer or not self.performance_tracker.timestamps:
            return go.Figure()
            
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': self.performance_tracker.timestamps,
            'equity': self.performance_tracker.equity_curve
        })
        equity_df.set_index('timestamp', inplace=True)
        
        # Align regimes with performance data
        aligned_regimes = self.regime_analyzer.regimes.reindex(equity_df.index, method='ffill')
        
        # Create figure
        fig = go.Figure()
        
        # Plot equity curve colored by regime
        for regime in aligned_regimes.unique():
            regime_mask = aligned_regimes == regime
            regime_equity = equity_df[regime_mask]['equity']
            
            if len(regime_equity) > 0:
                fig.add_trace(go.Scatter(
                    x=regime_equity.index,
                    y=regime_equity,
                    mode='lines',
                    name=f'Regime {regime}',
                    line=dict(width=2)
                ))
                
        # Update layout
        fig.update_layout(
            title="Portfolio Equity Curve by Regime",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template='plotly_dark'
        )
        
        return fig
        
    def _create_regime_performance_figure(self) -> go.Figure:
        """
        Create regime performance figure.
        
        Returns:
            Plotly figure
        """
        if not self.regime_analyzer or not self.regime_analyzer.regime_performance:
            return go.Figure()
            
        # Create regime performance DataFrame
        regime_data = []
        for regime, perf in self.regime_analyzer.regime_performance.items():
            regime_data.append({
                'regime': regime,
                'total_return': perf.total_return,
                'sharpe_ratio': perf.sharpe_ratio,
                'max_drawdown': perf.max_drawdown,
                'win_rate': perf.win_rate
            })
            
        regime_df = pd.DataFrame(regime_data)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add total return
        fig.add_trace(
            go.Bar(
                x=regime_df['regime'],
                y=regime_df['total_return'],
                name='Total Return',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # Add Sharpe ratio
        fig.add_trace(
            go.Bar(
                x=regime_df['regime'],
                y=regime_df['sharpe_ratio'],
                name='Sharpe Ratio',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Add max drawdown
        fig.add_trace(
            go.Bar(
                x=regime_df['regime'],
                y=regime_df['max_drawdown'],
                name='Max Drawdown',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # Add win rate
        fig.add_trace(
            go.Bar(
                x=regime_df['regime'],
                y=regime_df['win_rate'],
                name='Win Rate',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Regime Performance Comparison",
            template='plotly_dark',
            height=600
        )
        
        # Format y-axes
        fig.update_yaxes(title_text="Return", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate", tickformat=".1%", row=2, col=2)
        
        return fig
        
    def _create_regime_transitions_figure(self) -> go.Figure:
        """
        Create regime transitions figure.
        
        Returns:
            Plotly figure
        """
        if not self.regime_analyzer or not self.regime_analyzer.regime_transitions:
            return go.Figure()
            
        # Create transition matrix
        regimes = sorted(list(set(t['from_regime'] for t in self.regime_analyzer.regime_transitions) | 
                            set(t['to_regime'] for t in self.regime_analyzer.regime_transitions)))
        
        transition_matrix = np.zeros((len(regimes), len(regimes)))
        
        for transition in self.regime_analyzer.regime_transitions:
            from_idx = regimes.index(transition['from_regime'])
            to_idx = regimes.index(transition['to_regime'])
            transition_matrix[from_idx, to_idx] += 1
            
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=regimes,
            y=regimes,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Regime Transition Matrix",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            template='plotly_dark'
        )
        
        return fig
        
    def _create_regime_durations_figure(self) -> go.Figure:
        """
        Create regime durations figure.
        
        Returns:
            Plotly figure
        """
        if not self.regime_analyzer or not self.regime_analyzer.regime_durations:
            return go.Figure()
            
        # Create regime duration data
        regimes = list(self.regime_analyzer.regime_durations.keys())
        mean_durations = [self.regime_analyzer.regime_durations[regime]['mean_duration'] for regime in regimes]
        std_durations = [self.regime_analyzer.regime_durations[regime]['std_duration'] for regime in regimes]
        
        # Create figure
        fig = go.Figure()
        
        # Add duration bars with error bars
        fig.add_trace(go.Bar(
            x=regimes,
            y=mean_durations,
            error_y=dict(type='data', array=std_durations, visible=True),
            name='Average Duration',
            marker_color='blue'
        ))
        
        # Update layout
        fig.update_layout(
            title="Average Regime Duration",
            xaxis_title="Regime",
            yaxis_title="Duration (days)",
            template='plotly_dark'
        )
        
        return fig
        
    def start(self) -> None:
        """Start the dashboard."""
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start dashboard
        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=self.debug)
        
    def stop(self) -> None:
        """Stop the dashboard."""
        if self.update_thread:
            self.stop_event.set()
            self.update_thread.join()
            
    def _update_loop(self) -> None:
        """Background update loop."""
        while not self.stop_event.is_set():
            try:
                # Check for performance alerts
                metrics = self.performance_tracker.calculate_metrics()
                self.alert_manager.check_performance_alerts(metrics.to_dict())
                
                # Check for trade alerts
                for trade in self.performance_tracker.trade_history[-10:]:  # Check last 10 trades
                    self.alert_manager.check_trade_alerts(trade)
                    
                # Check for regime alerts
                if self.regime_analyzer and len(self.regime_analyzer.regime_transitions) > 0:
                    latest_transition = self.regime_analyzer.regime_transitions[-1]
                    transition_time = latest_transition['timestamp']
                    
                    # Check if transition is recent (within last hour)
                    if (datetime.now() - transition_time).total_seconds() < 3600:
                        self.alert_manager.check_regime_alert(
                            latest_transition['from_regime'],
                            latest_transition['to_regime']
                        )
                        
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                self.alert_manager.add_system_error_alert(str(e))
                
            # Sleep for 5 minutes
            time.sleep(300)