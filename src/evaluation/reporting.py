"""
Reporting module for the RL trading system.

This module provides comprehensive reporting capabilities including
performance reports, risk reports, and strategy analysis reports.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import calculate_performance_metrics, calculate_risk_metrics
from .backtest_evaluator import BacktestResult

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Report configuration."""
    output_dir: str = "reports"
    include_plots: bool = True
    include_raw_data: bool = False
    include_benchmark_comparison: bool = True
    include_risk_analysis: bool = True
    include_trade_analysis: bool = True
    include_performance_analysis: bool = True
    include_recommendations: bool = True
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 300
    include_executive_summary: bool = True
    include_technical_details: bool = True
    include_appendix: bool = True


@dataclass
class ExecutiveSummary:
    """Executive summary data structure."""
    period: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    key_findings: List[str]
    recommendations: List[str]
    risk_assessment: str
    performance_rating: str


class TradingReportGenerator:
    """
    Comprehensive trading report generator.
    
    This class provides detailed reporting capabilities for
    backtest results, performance analysis, and strategy evaluation.
    """
    
    def __init__(self, settings: Settings, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            settings: Configuration settings
            config: Report configuration
        """
        self.settings = settings
        self.config = config or self._load_config()
        
        # Report data
        self.backtest_results = []
        self.benchmark_results = []
        self.executive_summary = None
        
        # Configuration
        self.reporting_enabled = settings.get("evaluation", "reporting_enabled", default=True)
        
        logger.info("Report generator initialized")
    
    def _load_config(self) -> ReportConfig:
        """
        Load report configuration from settings.
        
        Returns:
            Report configuration
        """
        config = ReportConfig()
        
        config.output_dir = self.settings.get("evaluation", "report_output_dir", default="reports")
        config.include_plots = self.settings.get("evaluation", "include_plots", default=True)
        config.include_raw_data = self.settings.get("evaluation", "include_raw_data", default=False)
        config.include_benchmark_comparison = self.settings.get("evaluation", "include_benchmark_comparison", default=True)
        config.include_risk_analysis = self.settings.get("evaluation", "include_risk_analysis", default=True)
        config.include_trade_analysis = self.settings.get("evaluation", "include_trade_analysis", default=True)
        config.include_performance_analysis = self.settings.get("evaluation", "include_performance_analysis", default=True)
        config.include_recommendations = self.settings.get("evaluation", "include_recommendations", default=True)
        config.plot_format = self.settings.get("evaluation", "plot_format", default="png")
        config.plot_dpi = self.settings.get("evaluation", "plot_dpi", default=300)
        config.include_executive_summary = self.settings.get("evaluation", "include_executive_summary", default=True)
        config.include_technical_details = self.settings.get("evaluation", "include_technical_details", default=True)
        config.include_appendix = self.settings.get("evaluation", "include_appendix", default=True)
        
        return config
    
    def add_backtest_result(self, result: BacktestResult):
        """
        Add backtest result to report.
        
        Args:
            result: Backtest result
        """
        self.backtest_results.append(result)
        logger.info(f"Added backtest result to report")
    
    def add_benchmark_result(self, result: BacktestResult):
        """
        Add benchmark result to report.
        
        Args:
            result: Benchmark result
        """
        self.benchmark_results.append(result)
        logger.info(f"Added benchmark result to report")
    
    def generate_comprehensive_report(self, report_name: str) -> Path:
        """
        Generate comprehensive trading report.
        
        Args:
            report_name: Name of the report
            
        Returns:
            Path to the generated report
        """
        if not self.reporting_enabled:
            logger.warning("Reporting is disabled")
            return None
        
        if not self.backtest_results:
            logger.warning("No backtest results to report")
            return None
        
        try:
            # Create output directory
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate executive summary
            if self.config.include_executive_summary:
                self.executive_summary = self._generate_executive_summary()
            
            # Generate report sections
            report_sections = {}
            
            if self.config.include_performance_analysis:
                report_sections['performance_analysis'] = self._generate_performance_analysis()
            
            if self.config.include_risk_analysis:
                report_sections['risk_analysis'] = self._generate_risk_analysis()
            
            if self.config.include_trade_analysis:
                report_sections['trade_analysis'] = self._generate_trade_analysis()
            
            if self.config.include_benchmark_comparison and self.benchmark_results:
                report_sections['benchmark_comparison'] = self._generate_benchmark_comparison()
            
            if self.config.include_recommendations:
                report_sections['recommendations'] = self._generate_recommendations()
            
            if self.config.include_technical_details:
                report_sections['technical_details'] = self._generate_technical_details()
            
            if self.config.include_appendix:
                report_sections['appendix'] = self._generate_appendix()
            
            # Generate plots
            if self.config.include_plots:
                self._generate_plots(output_dir)
            
            # Save report
            report_path = self._save_report(report_name, output_dir, report_sections)
            
            logger.info(f"Comprehensive report generated: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def _generate_executive_summary(self) -> ExecutiveSummary:
        """
        Generate executive summary.
        
        Returns:
            Executive summary
        """
        if not self.backtest_results:
            return None
        
        # Aggregate results
        total_return = np.mean([r.total_return for r in self.backtest_results])
        annual_return = np.mean([r.annual_return for r in self.backtest_results])
        sharpe_ratio = np.mean([r.sharpe_ratio for r in self.backtest_results])
        max_drawdown = np.mean([r.max_drawdown for r in self.backtest_results])
        win_rate = np.mean([r.win_rate for r in self.backtest_results])
        total_trades = np.mean([r.total_trades for r in self.backtest_results])
        profit_factor = np.mean([r.profit_factor for r in self.backtest_results])
        
        # Generate key findings
        key_findings = []
        if sharpe_ratio > 1.0:
            key_findings.append("Strategy demonstrates strong risk-adjusted performance")
        if max_drawdown > -0.1:
            key_findings.append("Strategy shows acceptable drawdown levels")
        if win_rate > 0.5:
            key_findings.append("Strategy has high win rate")
        if profit_factor > 1.5:
            key_findings.append("Strategy shows good profit factor")
        
        # Generate recommendations
        recommendations = []
        if sharpe_ratio < 1.0:
            recommendations.append("Consider improving risk-adjusted performance")
        if max_drawdown < -0.2:
            recommendations.append("Implement stricter risk management")
        if win_rate < 0.4:
            recommendations.append("Improve trade selection criteria")
        if profit_factor < 1.2:
            recommendations.append("Review entry and exit strategies")
        
        # Risk assessment
        if sharpe_ratio > 1.5 and max_drawdown > -0.15:
            risk_assessment = "Low risk with good returns"
        elif sharpe_ratio > 1.0 and max_drawdown > -0.2:
            risk_assessment = "Moderate risk with acceptable returns"
        else:
            risk_assessment = "High risk - requires careful monitoring"
        
        # Performance rating
        if sharpe_ratio > 2.0 and max_drawdown > -0.1:
            performance_rating = "Excellent"
        elif sharpe_ratio > 1.5 and max_drawdown > -0.15:
            performance_rating = "Good"
        elif sharpe_ratio > 1.0 and max_drawdown > -0.2:
            performance_rating = "Average"
        else:
            performance_rating = "Poor"
        
        period = f"{self.backtest_results[0].config.start_date} to {self.backtest_results[0].config.end_date}"
        
        return ExecutiveSummary(
            period=period,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=profit_factor,
            key_findings=key_findings,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            performance_rating=performance_rating
        )
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """
        Generate performance analysis section.
        
        Returns:
            Performance analysis data
        """
        if not self.backtest_results:
            return {}
        
        # Aggregate performance metrics
        performance_metrics = {
            'returns': {
                'total_return': np.mean([r.total_return for r in self.backtest_results]),
                'annual_return': np.mean([r.annual_return for r in self.backtest_results]),
                'annual_volatility': np.mean([r.annual_volatility for r in self.backtest_results])
            },
            'risk_adjusted': {
                'sharpe_ratio': np.mean([r.sharpe_ratio for r in self.backtest_results]),
                'sortino_ratio': np.mean([r.sortino_ratio for r in self.backtest_results]),
                'calmar_ratio': np.mean([r.calmar_ratio for r in self.backtest_results])
            },
            'drawdown': {
                'max_drawdown': np.mean([r.max_drawdown for r in self.backtest_results]),
                'current_drawdown': np.mean([r.current_drawdown for r in self.backtest_results])
            },
            'var': {
                'var_95': np.mean([r.var_95 for r in self.backtest_results]),
                'var_99': np.mean([r.var_99 for r in self.backtest_results]),
                'cvar_95': np.mean([r.cvar_95 for r in self.backtest_results]),
                'cvar_99': np.mean([r.cvar_99 for r in self.backtest_results])
            }
        }
        
        return {
            'performance_metrics': performance_metrics,
            'performance_comparison': self._compare_performance_metrics()
        }
    
    def _generate_risk_analysis(self) -> Dict[str, Any]:
        """
        Generate risk analysis section.
        
        Returns:
            Risk analysis data
        """
        if not self.backtest_results:
            return {}
        
        # Aggregate risk metrics
        risk_metrics = {
            'volatility': {
                'annual_volatility': np.mean([r.annual_volatility for r in self.backtest_results]),
                'monthly_volatility': np.mean([r.annual_volatility for r in self.backtest_results]) / np.sqrt(12),
                'weekly_volatility': np.mean([r.annual_volatility for r in self.backtest_results]) / np.sqrt(52),
                'daily_volatility': np.mean([r.annual_volatility for r in self.backtest_results]) / np.sqrt(252)
            },
            'drawdown': {
                'max_drawdown': np.mean([r.max_drawdown for r in self.backtest_results]),
                'avg_drawdown': np.mean([abs(r.max_drawdown) for r in self.backtest_results])
            },
            'risk_adjusted': {
                'sharpe_ratio': np.mean([r.sharpe_ratio for r in self.backtest_results]),
                'sortino_ratio': np.mean([r.sortino_ratio for r in self.backtest_results]),
                'calmar_ratio': np.mean([r.calmar_ratio for r in self.backtest_results])
            }
        }
        
        # Risk assessment
        risk_assessment = self._assess_risk_level()
        
        return {
            'risk_metrics': risk_metrics,
            'risk_assessment': risk_assessment
        }
    
    def _generate_trade_analysis(self) -> Dict[str, Any]:
        """
        Generate trade analysis section.
        
        Returns:
            Trade analysis data
        """
        if not self.backtest_results:
            return {}
        
        # Aggregate trade metrics
        trade_metrics = {
            'trade_statistics': {
                'total_trades': np.mean([r.total_trades for r in self.backtest_results]),
                'winning_trades': np.mean([r.winning_trades for r in self.backtest_results]),
                'losing_trades': np.mean([r.losing_trades for r in self.backtest_results]),
                'win_rate': np.mean([r.win_rate for r in self.backtest_results])
            },
            'pnl_statistics': {
                'avg_win': np.mean([r.avg_win for r in self.backtest_results]),
                'avg_loss': np.mean([r.avg_loss for r in self.backtest_results]),
                'largest_win': np.mean([r.largest_win for r in self.backtest_results]),
                'largest_loss': np.mean([r.largest_loss for r in self.backtest_results]),
                'profit_factor': np.mean([r.profit_factor for r in self.backtest_results])
            },
            'trade_patterns': {
                'max_consecutive_wins': np.mean([r.max_consecutive_wins for r in self.backtest_results]),
                'max_consecutive_losses': np.mean([r.max_consecutive_losses for r in self.backtest_results]),
                'recovery_factor': np.mean([r.recovery_factor for r in self.backtest_results]),
                'payoff_ratio': np.mean([r.payoff_ratio for r in self.backtest_results])
            }
        }
        
        return {
            'trade_metrics': trade_metrics,
            'trade_patterns': self._analyze_trade_patterns()
        }
    
    def _generate_benchmark_comparison(self) -> Dict[str, Any]:
        """
        Generate benchmark comparison section.
        
        Returns:
            Benchmark comparison data
        """
        if not self.backtest_results or not self.benchmark_results:
            return {}
        
        # Compare performance metrics
        comparison = {
            'performance_comparison': {
                'total_return': {
                    'strategy': np.mean([r.total_return for r in self.backtest_results]),
                    'benchmark': np.mean([r.total_return for r in self.benchmark_results]),
                    'outperformance': np.mean([r.total_return for r in self.backtest_results]) - np.mean([r.total_return for r in self.benchmark_results])
                },
                'annual_return': {
                    'strategy': np.mean([r.annual_return for r in self.backtest_results]),
                    'benchmark': np.mean([r.annual_return for r in self.benchmark_results]),
                    'outperformance': np.mean([r.annual_return for r in self.backtest_results]) - np.mean([r.annual_return for r in self.benchmark_results])
                },
                'sharpe_ratio': {
                    'strategy': np.mean([r.sharpe_ratio for r in self.backtest_results]),
                    'benchmark': np.mean([r.sharpe_ratio for r in self.benchmark_results]),
                    'outperformance': np.mean([r.sharpe_ratio for r in self.backtest_results]) - np.mean([r.sharpe_ratio for r in self.benchmark_results])
                },
                'max_drawdown': {
                    'strategy': np.mean([r.max_drawdown for r in self.backtest_results]),
                    'benchmark': np.mean([r.max_drawdown for r in self.benchmark_results]),
                    'outperformance': np.mean([r.max_drawdown for r in self.backtest_results]) - np.mean([r.max_drawdown for r in self.benchmark_results])
                }
            }
        }
        
        # Generate comparison insights
        insights = self._generate_comparison_insights(comparison)
        
        return {
            'comparison': comparison,
            'insights': insights,
            'recommendations': self._generate_benchmark_recommendations(comparison)
        }
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations section.
        
        Returns:
            Recommendations data
        """
        if not self.backtest_results:
            return {}
        
        # Analyze performance and generate recommendations
        recommendations = {
            'performance_recommendations': self._generate_performance_recommendations(),
            'risk_recommendations': self._generate_risk_recommendations(),
            'strategy_recommendations': self._generate_strategy_recommendations(),
            'implementation_recommendations': self._generate_implementation_recommendations()
        }
        
        return recommendations
    
    def _generate_technical_details(self) -> Dict[str, Any]:
        """
        Generate technical details section.
        
        Returns:
            Technical details data
        """
        if not self.backtest_results:
            return {}
        
        # Technical details
        technical_details = {
            'backtest_parameters': self._get_backtest_parameters(),
            'model_details': self._get_model_details(),
            'data_details': self._get_data_details(),
            'execution_details': self._get_execution_details(),
            'risk_management_details': self._get_risk_management_details()
        }
        
        return technical_details
    
    def _generate_appendix(self) -> Dict[str, Any]:
        """
        Generate appendix section.
        
        Returns:
            Appendix data
        """
        if not self.backtest_results:
            return {}
        
        # Appendix
        appendix = {
            'glossary': self._generate_glossary(),
            'methodology': self._generate_methodology(),
            'data_sources': self._generate_data_sources(),
            'references': self._generate_references(),
            'raw_data': self._get_raw_data() if self.config.include_raw_data else None
        }
        
        return appendix
    
    def _generate_plots(self, output_dir: Path):
        """
        Generate plots for the report.
        
        Args:
            output_dir: Output directory for plots
        """
        if not self.backtest_results:
            return
        
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate equity curves
        self._generate_equity_curves(plots_dir)
        
        # Generate performance metrics
        self._generate_performance_metrics(plots_dir)
        
        # Generate risk metrics
        self._generate_risk_metrics(plots_dir)
        
        # Generate trade analysis
        self._generate_trade_analysis_plots(plots_dir)
        
        # Generate benchmark comparison
        if self.benchmark_results:
            self._generate_benchmark_comparison_plots(plots_dir)
        
        logger.info(f"Plots generated in {plots_dir}")
    
    def _generate_equity_curves(self, output_dir: Path):
        """Generate equity curves plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, result in enumerate(self.backtest_results):
            equity_df = pd.DataFrame(result.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            ax.plot(equity_df.index, equity_df['equity'], label=f'Strategy {i+1}')
        
        if self.benchmark_results:
            for i, result in enumerate(self.benchmark_results):
                equity_df = pd.DataFrame(result.equity_curve)
                equity_df.set_index('timestamp', inplace=True)
                ax.plot(equity_df.index, equity_df['equity'], '--', label=f'Benchmark {i+1}')
        
        ax.set_title('Equity Curves')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(output_dir / "equity_curves.png", dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_metrics(self, output_dir: Path):
        """Generate performance metrics plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sharpe ratio
        sharpe_ratios = [r.sharpe_ratio for r in self.backtest_results]
        axes[0, 0].bar(range(len(sharpe_ratios)), sharpe_ratios)
        axes[0, 0].set_title('Sharpe Ratio')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        
        # Max drawdown
        max_drawdowns = [r.max_drawdown for r in self.backtest_results]
        axes[0, 1].bar(range(len(max_drawdowns)), max_drawdowns)
        axes[0, 1].set_title('Maximum Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        
        # Win rate
        win_rates = [r.win_rate for r in self.backtest_results]
        axes[1, 0].bar(range(len(win_rates)), win_rates)
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].set_ylabel('Win Rate')
        
        # Profit factor
        profit_factors = [r.profit_factor for r in self.backtest_results]
        axes[1, 1].bar(range(len(profit_factors)), profit_factors)
        axes[1, 1].set_title('Profit Factor')
        axes[1, 1].set_ylabel('Profit Factor')
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_metrics.png", dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _generate_risk_metrics(self, output_dir: Path):
        """Generate risk metrics plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Annual volatility
        volatilities = [r.annual_volatility for r in self.backtest_results]
        axes[0, 0].bar(range(len(volatilities)), volatilities)
        axes[0, 0].set_title('Annual Volatility')
        axes[0, 0].set_ylabel('Volatility')
        
        # VaR
        var_95_values = [r.var_95 for r in self.backtest_results]
        var_99_values = [r.var_99 for r in self.backtest_results]
        x = range(len(var_95_values))
        axes[0, 1].bar(x, var_95_values, alpha=0.7, label='VaR 95%')
        axes[0, 1].bar(x, var_99_values, alpha=0.7, label='VaR 99%')
        axes[0, 1].set_title('Value at Risk')
        axes[0, 1].set_ylabel('VaR')
        axes[0, 1].legend()
        
        # CVaR
        cvar_95_values = [r.cvar_95 for r in self.backtest_results]
        cvar_99_values = [r.cvar_99 for r in self.backtest_results]
        axes[1, 0].bar(x, cvar_95_values, alpha=0.7, label='CVaR 95%')
        axes[1, 0].bar(x, cvar_99_values, alpha=0.7, label='CVaR 99%')
        axes[1, 0].set_title('Conditional Value at Risk')
        axes[1, 0].set_ylabel('CVaR')
        axes[1, 0].legend()
        
        # Beta
        betas = [r.beta for r in self.backtest_results]
        axes[1, 1].bar(range(len(betas)), betas)
        axes[1, 1].set_title('Beta')
        axes[1, 1].set_ylabel('Beta')
        
        plt.tight_layout()
        plt.savefig(output_dir / "risk_metrics.png", dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _generate_trade_analysis_plots(self, output_dir: Path):
        """Generate trade analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade statistics
        total_trades = [r.total_trades for r in self.backtest_results]
        winning_trades = [r.winning_trades for r in self.backtest_results]
        losing_trades = [r.losing_trades for r in self.backtest_results]
        
        x = range(len(total_trades))
        axes[0, 0].bar(x, total_trades, alpha=0.7, label='Total Trades')
        axes[0, 0].bar(x, winning_trades, alpha=0.7, label='Winning Trades')
        axes[0, 0].bar(x, losing_trades, alpha=0.7, label='Losing Trades')
        axes[0, 0].set_title('Trade Statistics')
        axes[0, 0].set_ylabel('Number of Trades')
        axes[0, 0].legend()
        
        # P&L distribution
        all_pnls = []
        for result in self.backtest_results:
            all_pnls.extend([t.get('pnl', 0) for t in result.trade_history])
        
        if all_pnls:
            axes[0, 1].hist(all_pnls, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_title('Trade P&L Distribution')
            axes[0, 1].set_xlabel('P&L ($)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Win rate
        win_rates = [r.win_rate for r in self.backtest_results]
        axes[1, 0].bar(range(len(win_rates)), win_rates)
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].set_ylabel('Win Rate')
        
        # Profit factor
        profit_factors = [r.profit_factor for r in self.backtest_results]
        axes[1, 1].bar(range(len(profit_factors)), profit_factors)
        axes[1, 1].set_title('Profit Factor')
        axes[1, 1].set_ylabel('Profit Factor')
        
        plt.tight_layout()
        plt.savefig(output_dir / "trade_analysis.png", dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _generate_benchmark_comparison_plots(self, output_dir: Path):
        """Generate benchmark comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance comparison
        strategy_returns = [r.total_return for r in self.backtest_results]
        benchmark_returns = [r.total_return for r in self.benchmark_results]
        
        x = range(len(strategy_returns))
        axes[0, 0].bar(x, strategy_returns, alpha=0.7, label='Strategy')
        axes[0, 0].bar(x, benchmark_returns, alpha=0.7, label='Benchmark')
        axes[0, 0].set_title('Total Return Comparison')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].legend()
        
        # Risk comparison
        strategy_vol = [r.annual_volatility for r in self.backtest_results]
        benchmark_vol = [r.annual_volatility for r in self.benchmark_results]
        
        axes[0, 1].bar(x, strategy_vol, alpha=0.7, label='Strategy')
        axes[0, 1].bar(x, benchmark_vol, alpha=0.7, label='Benchmark')
        axes[0, 1].set_title('Volatility Comparison')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].legend()
        
        # Sharpe ratio comparison
        strategy_sharpe = [r.sharpe_ratio for r in self.backtest_results]
        benchmark_sharpe = [r.sharpe_ratio for r in self.benchmark_results]
        
        axes[1, 0].bar(x, strategy_sharpe, alpha=0.7, label='Strategy')
        axes[1, 0].bar(x, benchmark_sharpe, alpha=0.7, label='Benchmark')
        axes[1, 0].set_title('Sharpe Ratio Comparison')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].legend()
        
        # Drawdown comparison
        strategy_dd = [r.max_drawdown for r in self.backtest_results]
        benchmark_dd = [r.max_drawdown for r in self.benchmark_results]
        
        axes[1, 1].bar(x, strategy_dd, alpha=0.7, label='Strategy')
        axes[1, 1].bar(x, benchmark_dd, alpha=0.7, label='Benchmark')
        axes[1, 1].set_title('Maximum Drawdown Comparison')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_comparison.png", dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _save_report(self, report_name: str, output_dir: Path, report_sections: Dict[str, Any]) -> Path:
        """
        Save the report to file.
        
        Args:
            report_name: Name of the report
            output_dir: Output directory
            report_sections: Report sections
            
        Returns:
            Path to the saved report
        """
        # Create report data
        report_data = {
            'report_name': report_name,
            'generated_at': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'executive_summary': self.executive_summary.__dict__ if self.executive_summary else None,
            'sections': report_sections,
            'backtest_results': [
                {
                    'symbol': r.config.symbol,
                    'total_return': r.total_return,
                    'annual_return': r.annual_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'win_rate': r.win_rate,
                    'total_trades': r.total_trades,
                    'profit_factor': r.profit_factor
                }
                for r in self.backtest_results
            ]
        }
        
        # Save as JSON
        report_path = output_dir / f"{report_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save as HTML
        html_path = output_dir / f"{report_name}.html"
        self._generate_html_report(report_data, html_path)
        
        return report_path
    
    def _generate_html_report(self, report_data: Dict[str, Any], output_path: Path):
        """
        Generate HTML report.
        
        Args:
            report_data: Report data
            output_path: Output path
        """
        # Create HTML content
        html_content = self._create_html_template(report_data)
        
        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _create_html_template(self, report_data: Dict[str, Any]) -> str:
        """
        Create HTML template for the report.
        
        Args:
            report_data: Report data
            
        Returns:
            HTML content
        """
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_data['report_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_data['report_name']}</h1>
                <p>Generated on: {report_data['generated_at']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {self._format_executive_summary(report_data.get('executive_summary'))}
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                {self._format_performance_analysis(report_data.get('sections', {}).get('performance_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Risk Analysis</h2>
                {self._format_risk_analysis(report_data.get('sections', {}).get('risk_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Trade Analysis</h2>
                {self._format_trade_analysis(report_data.get('sections', {}).get('trade_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._format_recommendations(report_data.get('sections', {}).get('recommendations', {}))}
            </div>
            
            <div class="section">
                <h2>Technical Details</h2>
                {self._format_technical_details(report_data.get('sections', {}).get('technical_details', {}))}
            </div>
            
            <div class="section">
                <h2>Appendix</h2>
                {self._format_appendix(report_data.get('sections', {}).get('appendix', {}))}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_executive_summary(self, summary: Optional[Dict[str, Any]]) -> str:
        """Format executive summary for HTML."""
        if not summary:
            return "<p>No executive summary available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Period: {summary.get('period', 'N/A')}</h3>
            <p>Total Return: {summary.get('total_return', 0):.2%}</p>
            <p>Annual Return: {summary.get('annual_return', 0):.2%}</p>
            <p>Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}</p>
            <p>Max Drawdown: {summary.get('max_drawdown', 0):.2%}</p>
            <p>Win Rate: {summary.get('win_rate', 0):.2%}</p>
            <p>Total Trades: {summary.get('total_trades', 0)}</p>
            <p>Profit Factor: {summary.get('profit_factor', 0):.2f}</p>
        </div>
        
        <div class="metric">
            <h3>Performance Rating: {summary.get('performance_rating', 'N/A')}</h3>
            <p>Risk Assessment: {summary.get('risk_assessment', 'N/A')}</p>
        </div>
        
        <div class="metric">
            <h3>Key Findings</h3>
            <ul>
                {''.join([f'<li>{finding}</li>' for finding in summary.get('key_findings', [])])}
            </ul>
        </div>
        
        <div class="metric">
            <h3>Recommendations</h3>
            <ul>
                {''.join([f'<li>{recommendation}</li>' for recommendation in summary.get('recommendations', [])])}
            </ul>
        </div>
        """
        
        return html
    
    def _format_performance_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis for HTML."""
        if not analysis:
            return "<p>No performance analysis available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Performance Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td>{analysis.get('performance_metrics', {}).get('returns', {}).get('total_return', 0):.2%}</td></tr>
                <tr><td>Annual Return</td><td>{analysis.get('performance_metrics', {}).get('returns', {}).get('annual_return', 0):.2%}</td></tr>
                <tr><td>Annual Volatility</td><td>{analysis.get('performance_metrics', {}).get('returns', {}).get('annual_volatility', 0):.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{analysis.get('performance_metrics', {}).get('risk_adjusted', {}).get('sharpe_ratio', 0):.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{analysis.get('performance_metrics', {}).get('risk_adjusted', {}).get('sortino_ratio', 0):.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{analysis.get('performance_metrics', {}).get('risk_adjusted', {}).get('calmar_ratio', 0):.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{analysis.get('performance_metrics', {}).get('drawdown', {}).get('max_drawdown', 0):.2%}</td></tr>
            </table>
        </div>
        """
        
        return html
    
    def _format_risk_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format risk analysis for HTML."""
        if not analysis:
            return "<p>No risk analysis available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Risk Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Annual Volatility</td><td>{analysis.get('risk_metrics', {}).get('volatility', {}).get('annual_volatility', 0):.2%}</td></tr>
                <tr><td>Max Drawdown</td><td>{analysis.get('risk_metrics', {}).get('drawdown', {}).get('max_drawdown', 0):.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{analysis.get('risk_metrics', {}).get('risk_adjusted', {}).get('sharpe_ratio', 0):.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{analysis.get('risk_metrics', {}).get('risk_adjusted', {}).get('sortino_ratio', 0):.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{analysis.get('risk_metrics', {}).get('risk_adjusted', {}).get('calmar_ratio', 0):.2f}</td></tr>
            </table>
        </div>
        
        <div class="metric">
            <h3>Risk Assessment</h3>
            <p>{analysis.get('risk_assessment', 'N/A')}</p>
        </div>
        """
        
        return html
    
    def _format_trade_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format trade analysis for HTML."""
        if not analysis:
            return "<p>No trade analysis available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Trade Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{analysis.get('trade_metrics', {}).get('trade_statistics', {}).get('total_trades', 0)}</td></tr>
                <tr><td>Winning Trades</td><td>{analysis.get('trade_metrics', {}).get('trade_statistics', {}).get('winning_trades', 0)}</td></tr>
                <tr><td>Losing Trades</td><td>{analysis.get('trade_metrics', {}).get('trade_statistics', {}).get('losing_trades', 0)}</td></tr>
                <tr><td>Win Rate</td><td>{analysis.get('trade_metrics', {}).get('trade_statistics', {}).get('win_rate', 0):.2%}</td></tr>
            </table>
        </div>
        
        <div class="metric">
            <h3>P&L Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Win</td><td>{analysis.get('trade_metrics', {}).get('pnl_statistics', {}).get('avg_win', 0):.2f}</td></tr>
                <tr><td>Average Loss</td><td>{analysis.get('trade_metrics', {}).get('pnl_statistics', {}).get('avg_loss', 0):.2f}</td></tr>
                <tr><td>Largest Win</td><td>{analysis.get('trade_metrics', {}).get('pnl_statistics', {}).get('largest_win', 0):.2f}</td></tr>
                <tr><td>Largest Loss</td><td>{analysis.get('trade_metrics', {}).get('pnl_statistics', {}).get('largest_loss', 0):.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{analysis.get('trade_metrics', {}).get('pnl_statistics', {}).get('profit_factor', 0):.2f}</td></tr>
            </table>
        </div>
        """
        
        return html
    
    def _format_recommendations(self, recommendations: Dict[str, Any]) -> str:
        """Format recommendations for HTML."""
        if not recommendations:
            return "<p>No recommendations available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Performance Recommendations</h3>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in recommendations.get('performance_recommendations', [])])}
            </ul>
        </div>
        
        <div class="metric">
            <h3>Risk Recommendations</h3>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in recommendations.get('risk_recommendations', [])])}
            </ul>
        </div>
        
        <div class="metric">
            <h3>Strategy Recommendations</h3>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in recommendations.get('strategy_recommendations', [])])}
            </ul>
        </div>
        
        <div class="metric">
            <h3>Implementation Recommendations</h3>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in recommendations.get('implementation_recommendations', [])])}
            </ul>
        </div>
        """
        
        return html
    
    def _format_technical_details(self, details: Dict[str, Any]) -> str:
        """Format technical details for HTML."""
        if not details:
            return "<p>No technical details available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Backtest Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Initial Capital</td><td>{details.get('backtest_parameters', {}).get('initial_capital', 0)}</td></tr>
                <tr><td>Risk per Trade</td><td>{details.get('backtest_parameters', {}).get('risk_per_trade_frac', 0):.2%}</td></tr>
                <tr><td>Stop Loss</td><td>{details.get('backtest_parameters', {}).get('stop_loss_r_multiple', 0)}R</td></tr>
                <tr><td>Take Profit</td><td>{details.get('backtest_parameters', {}).get('take_profit_r_multiple', 0)}R</td></tr>
                <tr><td>Max Daily Loss</td><td>{details.get('backtest_parameters', {}).get('max_daily_loss_r', 0)}R</td></tr>
            </table>
        </div>
        
        <div class="metric">
            <h3>Model Details</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>{details.get('model_details', {}).get('model_type', 'N/A')}</td></tr>
                <tr><td>Algorithm</td><td>{details.get('model_details', {}).get('algorithm', 'N/A')}</td></tr>
                <tr><td>Training Data</td><td>{details.get('model_details', {}).get('training_data', 'N/A')}</td></tr>
            </table>
        </div>
        """
        
        return html
    
    def _format_appendix(self, appendix: Dict[str, Any]) -> str:
        """Format appendix for HTML."""
        if not appendix:
            return "<p>No appendix available.</p>"
        
        html = f"""
        <div class="metric">
            <h3>Glossary</h3>
            <ul>
                {''.join([f'<li>{term}: {definition}</li>' for term, definition in appendix.get('glossary', {}).items()])}
            </ul>
        </div>
        
        <div class="metric">
            <h3>Methodology</h3>
            <p>{appendix.get('methodology', 'N/A')}</p>
        </div>
        
        <div class="metric">
            <h3>Data Sources</h3>
            <ul>
                {''.join([f'<li>{source}</li>' for source in appendix.get('data_sources', [])])}
            </ul>
        </div>
        
        <div class="metric">
            <h3>References</h3>
            <ul>
                {''.join([f'<li>{ref}</li>' for ref in appendix.get('references', [])])}
            </ul>
        </div>
        """
        
        return html
    
    # Helper methods (placeholder implementations)
    def _compare_performance_metrics(self) -> Dict[str, Any]:
        """Compare performance metrics."""
        return {}
    
    def _assess_risk_level(self) -> str:
        """Assess risk level."""
        return "Moderate risk"
    
    def _analyze_trade_patterns(self) -> Dict[str, Any]:
        """Analyze trade patterns."""
        return {}
    
    def _generate_comparison_insights(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate comparison insights."""
        return []
    
    def _generate_benchmark_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate benchmark recommendations."""
        return []
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        return []
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk recommendations."""
        return []
    
    def _generate_strategy_recommendations(self) -> List[str]:
        """Generate strategy recommendations."""
        return []
    
    def _generate_implementation_recommendations(self) -> List[str]:
        """Generate implementation recommendations."""
        return []
    
    def _get_backtest_parameters(self) -> Dict[str, Any]:
        """Get backtest parameters."""
        return {}
    
    def _get_model_details(self) -> Dict[str, Any]:
        """Get model details."""
        return {}
    
    def _get_data_details(self) -> Dict[str, Any]:
        """Get data details."""
        return {}
    
    def _get_execution_details(self) -> Dict[str, Any]:
        """Get execution details."""
        return {}
    
    def _get_risk_management_details(self) -> Dict[str, Any]:
        """Get risk management details."""
        return {}
    
    def _generate_glossary(self) -> Dict[str, str]:
        """Generate glossary."""
        return {}
    
    def _generate_methodology(self) -> str:
        """Generate methodology."""
        return "Methodology not implemented"
    
    def _generate_data_sources(self) -> List[str]:
        """Generate data sources."""
        return []
    
    def _generate_references(self) -> List[str]:
        """Generate references."""
        return []
    
    def _get_raw_data(self) -> Dict[str, Any]:
        """Get raw data."""
        return {}