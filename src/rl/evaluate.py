"""
Evaluation module for the RL trading system.

This module provides comprehensive evaluation tools for trained models,
including performance metrics, risk analysis, and reporting capabilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

from ..utils.config_loader import Settings
from ..utils.logging import get_logger
from ..utils.metrics import DifferentialSharpe, calculate_performance_metrics

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    This class provides tools for evaluating trained RL models
    with various performance metrics and risk analysis.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize model evaluator.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.results: Dict[str, Any] = {}
        
        # Evaluation parameters
        self.risk_free_rate = settings.get("evaluation", "risk_free_rate", default=0.02)
        self.timeframe = settings.get("evaluation", "timeframe", default="daily")
        self.benchmark_symbol = settings.get("evaluation", "benchmark_symbol", default="SPY")
        
    def evaluate_model(self, 
                      model_path: str,
                      env,
                      num_episodes: int = 10,
                      render: bool = False) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to trained model
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            render: Whether to render evaluation
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model from {model_path}...")
        
        # Load model
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        
        # Evaluation results
        episode_rewards = []
        episode_lengths = []
        equity_curves = []
        trade_lists = []
        
        for episode in range(num_episodes):
            logger.info(f"Evaluating episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            episode_equity = []
            
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track equity
                equity = env.equity
                episode_equity.append(equity)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            equity_curves.append(pd.Series(episode_equity, index=env.df.index[:len(episode_equity)]))
            trade_lists.append(env.get_trades())
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(equity_curves)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(trade_lists)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curves)
        
        # Combine results
        results = {
            'model_path': model_path,
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'performance_metrics': performance_metrics,
            'trade_statistics': trade_stats,
            'risk_metrics': risk_metrics,
            'equity_curves': equity_curves,
            'trade_lists': trade_lists
        }
        
        self.results = results
        logger.info("Model evaluation completed")
        
        return results
    
    def _calculate_performance_metrics(self, equity_curves: List[pd.Series]) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curves: List of equity curves
            
        Returns:
            Performance metrics
        """
        if not equity_curves:
            return {}
        
        # Combine equity curves
        combined_equity = pd.concat(equity_curves, axis=1).mean(axis=1)
        
        # Calculate returns
        returns = combined_equity.pct_change().dropna()
        
        # Calculate metrics
        metrics = calculate_performance_metrics(
            returns=returns,
            risk_free_rate=self.risk_free_rate,
            timeframe=self.timeframe
        )
        
        return metrics
    
    def _calculate_trade_statistics(self, trade_lists: List[List[Dict]]) -> Dict[str, float]:
        """
        Calculate trade statistics.
        
        Args:
            trade_lists: List of trade lists
            
        Returns:
            Trade statistics
        """
        if not trade_lists:
            return {}
        
        # Flatten trade lists
        all_trades = []
        for trades in trade_lists:
            all_trades.extend(trades)
        
        if not all_trades:
            return {}
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)
        
        # Calculate statistics
        stats = {
            'total_trades': len(all_trades),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean(),
            'profit_factor': trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()),
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_trade_duration': trades_df['duration'].mean() if 'duration' in trades_df.columns else 0,
            'max_consecutive_wins': self._calculate_consecutive(trades_df['pnl'] > 0),
            'max_consecutive_losses': self._calculate_consecutive(trades_df['pnl'] < 0)
        }
        
        return stats
    
    def _calculate_consecutive(self, series: pd.Series) -> int:
        """
        Calculate maximum consecutive values.
        
        Args:
            series: Boolean series
            
        Returns:
            Maximum consecutive count
        """
        if len(series) == 0:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _calculate_risk_metrics(self, equity_curves: List[pd.Series]) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            equity_curves: List of equity curves
            
        Returns:
            Risk metrics
        """
        if not equity_curves:
            return {}
        
        # Combine equity curves
        combined_equity = pd.concat(equity_curves, axis=1).mean(axis=1)
        
        # Calculate returns
        returns = combined_equity.pct_change().dropna()
        
        # Calculate risk metrics
        risk_metrics = {
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),
            'volatility': returns.std(),
            'downside_deviation': returns[returns < 0].std(),
            'sortino_ratio': (returns.mean() - self.risk_free_rate/252) / returns[returns < 0].std() * np.sqrt(252),
            'calmar_ratio': returns.mean() / abs((combined_equity / combined_equity.cummax() - 1).min()) * 252,
            'omega_ratio': self._calculate_omega_ratio(returns),
            'tail_ratio': abs(returns.quantile(0.05) / returns.quantile(0.95))
        }
        
        return risk_metrics
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Returns series
            threshold: Minimum acceptable return
            
        Returns:
            Omega ratio
        """
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        
        if len(losses) == 0:
            return np.inf
        
        return gains.sum() / abs(losses.sum())
    
    def generate_report(self, output_dir: str) -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Output directory for report
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        self._generate_html_report(output_path)
        
        # Generate plots
        self._generate_plots(output_path)
        
        # Save results as JSON
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _generate_html_report(self, output_path: Path) -> None:
        """
        Generate HTML report.
        
        Args:
            output_path: Output path
        """
        html_content = []
        
        # Header
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { margin: 10px 0; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
        """)
        
        # Model info
        html_content.append(f"""
        <h2>Model Information</h2>
        <p><strong>Model Path:</strong> {self.results['model_path']}</p>
        <p><strong>Number of Episodes:</strong> {self.results['num_episodes']}</p>
        """)
        
        # Performance metrics
        html_content.append("<h2>Performance Metrics</h2>")
        html_content.append("<table>")
        html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        for metric, value in self.results['performance_metrics'].items():
            if isinstance(value, (int, float)):
                value_str = f"{value:.4f}"
                if 'return' in metric.lower() and value > 0:
                    value_str = f'<span class="positive">{value_str}</span>'
                elif 'return' in metric.lower() and value < 0:
                    value_str = f'<span class="negative">{value_str}</span>'
            else:
                value_str = str(value)
            
            html_content.append(f"<tr><td>{metric}</td><td>{value_str}</td></tr>")
        
        html_content.append("</table>")
        
        # Trade statistics
        html_content.append("<h2>Trade Statistics</h2>")
        html_content.append("<table>")
        html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        for metric, value in self.results['trade_statistics'].items():
            if isinstance(value, (int, float)):
                value_str = f"{value:.4f}"
                if metric == 'win_rate' and value > 0.5:
                    value_str = f'<span class="positive">{value_str}</span>'
                elif metric == 'win_rate' and value < 0.5:
                    value_str = f'<span class="negative">{value_str}</span>'
            else:
                value_str = str(value)
            
            html_content.append(f"<tr><td>{metric}</td><td>{value_str}</td></tr>")
        
        html_content.append("</table>")
        
        # Risk metrics
        html_content.append("<h2>Risk Metrics</h2>")
        html_content.append("<table>")
        html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        for metric, value in self.results['risk_metrics'].items():
            if isinstance(value, (int, float)):
                value_str = f"{value:.4f}"
                if 'ratio' in metric.lower() and value > 1:
                    value_str = f'<span class="positive">{value_str}</span>'
                elif 'ratio' in metric.lower() and value < 1:
                    value_str = f'<span class="negative">{value_str}</span>'
            else:
                value_str = str(value)
            
            html_content.append(f"<tr><td>{metric}</td><td>{value_str}</td></tr>")
        
        html_content.append("</table>")
        
        # Episode statistics
        html_content.append("<h2>Episode Statistics</h2>")
        html_content.append("<table>")
        html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        episode_rewards = self.results['episode_rewards']
        episode_lengths = self.results['episode_lengths']
        
        html_content.append(f"<tr><td>Average Reward</td><td>{np.mean(episode_rewards):.2f}</td></tr>")
        html_content.append(f"<tr><td>Std Reward</td><td>{np.std(episode_rewards):.2f}</td></tr>")
        html_content.append(f"<tr><td>Average Length</td><td>{np.mean(episode_lengths):.2f}</td></tr>")
        html_content.append(f"<tr><td>Max Reward</td><td>{np.max(episode_rewards):.2f}</td></tr>")
        html_content.append(f"<tr><td>Min Reward</td><td>{np.min(episode_rewards):.2f}</td></tr>")
        
        html_content.append("</table>")
        
        # Footer
        html_content.append("</body></html>")
        
        # Save HTML
        with open(output_path / "evaluation_report.html", 'w') as f:
            f.write('\n'.join(html_content))
    
    def _generate_plots(self, output_path: Path) -> None:
        """
        Generate evaluation plots.
        
        Args:
            output_path: Output path
        """
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Equity curve plot
        plt.figure(figsize=(12, 8))
        
        for i, equity_curve in enumerate(self.results['equity_curves']):
            plt.plot(equity_curve.index, equity_curve.values, alpha=0.3, label=f'Episode {i+1}')
        
        # Plot average
        combined_equity = pd.concat(self.results['equity_curves'], axis=1).mean(axis=1)
        plt.plot(combined_equity.index, combined_equity.values, 'k-', linewidth=2, label='Average')
        
        plt.title('Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "equity_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Drawdown plot
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdowns
        drawdowns = []
        for equity_curve in self.results['equity_curves']:
            drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
            drawdowns.append(drawdown)
        
        # Plot average drawdown
        avg_drawdown = pd.concat(drawdowns, axis=1).mean(axis=1)
        plt.fill_between(avg_drawdown.index, avg_drawdown.values, 0, alpha=0.3, color='red')
        plt.plot(avg_drawdown.index, avg_drawdown.values, 'r-', linewidth=1)
        
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "drawdown_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Returns distribution
        plt.figure(figsize=(12, 6))
        
        # Calculate returns
        all_returns = []
        for equity_curve in self.results['equity_curves']:
            returns = equity_curve.pct_change().dropna()
            all_returns.extend(returns.values)
        
        plt.hist(all_returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        
        # Add normal distribution overlay
        mu, sigma = np.mean(all_returns), np.std(all_returns)
        x = np.linspace(min(all_returns), max(all_returns), 100)
        plt.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2), 'r-', linewidth=2)
        
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / "returns_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Trade P&L distribution
        if self.results['trade_lists']:
            plt.figure(figsize=(12, 6))
            
            all_trades = []
            for trades in self.results['trade_lists']:
                all_trades.extend(trades)
            
            if all_trades:
                pnl_values = [trade['pnl'] for trade in all_trades]
                
                plt.hist(pnl_values, bins=30, alpha=0.7, edgecolor='black')
                plt.axvline(0, color='red', linestyle='--', linewidth=2)
                
                plt.title('Trade P&L Distribution')
                plt.xlabel('P&L ($)')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_path / "trade_pnl_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info("Evaluation plots generated")
    
    def compare_models(self, 
                      model_results: List[Dict[str, Any]], 
                      output_dir: str) -> None:
        """
        Compare multiple models.
        
        Args:
            model_results: List of model evaluation results
            output_dir: Output directory for comparison
        """
        if not model_results:
            logger.warning("No model results to compare")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for result in model_results:
            model_name = Path(result['model_path']).stem
            metrics = result['performance_metrics']
            risk_metrics = result['risk_metrics']
            trade_stats = result['trade_statistics']
            
            row = {
                'model': model_name,
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'profit_factor': trade_stats.get('profit_factor', 0),
                'sortino_ratio': risk_metrics.get('sortino_ratio', 0),
                'calmar_ratio': risk_metrics.get('calmar_ratio', 0)
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Generate comparison plot
        plt.figure(figsize=(14, 8))
        
        # Select key metrics for comparison
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 3, i+1)
            
            # Normalize metrics for better visualization
            normalized_values = comparison_df[metric] / comparison_df[metric].abs().max()
            
            bars = plt.bar(comparison_df['model'], normalized_values)
            
            # Color bars based on performance
            for j, bar in enumerate(bars):
                if comparison_df.iloc[j][metric] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate comparison report
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        
        report.append("\nMODEL PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        report.append(comparison_df.to_string(index=False))
        
        # Find best model for each metric
        report.append("\nBEST MODEL BY METRIC:")
        report.append("-" * 40)
        
        for metric in metrics_to_plot:
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'model']
            best_value = comparison_df[metric].max()
            report.append(f"{metric.replace('_', ' ').title()}: {best_model} ({best_value:.4f})")
        
        # Save report
        with open(output_path / "model_comparison_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Model comparison saved to {output_path}")


def run_policy_on_env(model_path: str, env, save_dir: str) -> Dict[str, Any]:
    """
    Run a trained policy on an environment.
    
    Args:
        model_path: Path to trained model
        env: Environment to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    from stable_baselines3 import PPO
    
    # Load model
    model = PPO.load(model_path)
    
    # Run policy
    obs = env.reset()
    done = False
    equity_curve = []
    timestamps = []
    
    # Try to read timestamps from the env's data index
    if hasattr(env, "df"):
        ts_index = env.df.index
    else:
        ts_index = None
    
    step_i = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        
        # Pull equity from env if available
        eq = getattr(env, "equity", None)
        if eq is not None:
            equity_curve.append(float(eq))
            if ts_index is not None and env.i < len(ts_index):
                timestamps.append(ts_index[env.i])
            else:
                timestamps.append(pd.NaT)
        
        step_i += 1
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if equity_curve:
        out = pd.DataFrame({"timestamp": timestamps, "equity": equity_curve})
        out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
        out.to_csv(save_dir / "equity_curve.csv", index=False)
    
    return {"steps": step_i, "logged_points": len(equity_curve)}