"""
Visualization and analysis for Population-Based Training results.

This module provides comprehensive visualization capabilities for PBT results,
including population evolution, diversity metrics, and performance tracking.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from .pbt import PBTResults, PBTConfig, PBTMember
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PBTVisualizer:
    """
    Visualizer for Population-Based Training results.
    """
    
    def __init__(self, results: PBTResults, output_dir: str = "pbt_plots"):
        """
        Initialize PBT visualizer.
        
        Args:
            results: PBT results to visualize
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up figure parameters
        self.fig_size = (15, 10)
        self.dpi = 300
        self.font_size = 12
        
    def create_summary_dashboard(self) -> str:
        """
        Create a comprehensive summary dashboard.
        
        Returns:
            Path to saved dashboard
        """
        if not self.results.population_history:
            logger.error("No population history to visualize")
            return ""
            
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Fitness evolution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_fitness_evolution(ax1)
        
        # 2. Population diversity
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_diversity_evolution(ax2)
        
        # 3. Parameter evolution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_parameter_evolution(ax3)
        
        # 4. Best member lineage
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_best_member_lineage(ax4)
        
        # 5. Age distribution
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_age_distribution(ax5)
        
        # 6. Performance distribution
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_performance_distribution(ax6)
        
        # 7. Hyperparameter correlation
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_hyperparam_correlation(ax7)
        
        # 8. Reward parameter correlation
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_reward_param_correlation(ax8)
        
        # 9. Generation statistics
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_generation_stats(ax9)
        
        # 10. Population heatmap
        ax10 = fig.add_subplot(gs[3, :])
        self._plot_population_heatmap(ax10)
        
        # Add title
        fig.suptitle(
            f"PBT Summary Dashboard - {len(self.results.population_history)} Generations",
            fontsize=16,
            y=0.98
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / "pbt_dashboard.png"
        plt.savefig(dashboard_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved to {dashboard_path}")
        return str(dashboard_path)
        
    def _plot_fitness_evolution(self, ax: plt.Axes) -> None:
        """
        Plot fitness evolution over generations.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.generation_stats:
            ax.text(0.5, 0.5, "No generation statistics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Fitness Evolution")
            return
            
        # Extract fitness statistics
        generations = [stat['generation'] for stat in self.results.generation_stats]
        best_fitness = [stat['best_fitness'] for stat in self.results.generation_stats]
        mean_fitness = [stat['mean_fitness'] for stat in self.results.generation_stats]
        std_fitness = [stat['std_fitness'] for stat in self.results.generation_stats]
        
        # Plot fitness evolution
        ax.plot(generations, best_fitness, 'o-', label='Best Fitness', color='green', linewidth=2)
        ax.plot(generations, mean_fitness, 'o-', label='Mean Fitness', color='blue', linewidth=2)
        
        # Add standard deviation band
        ax.fill_between(
            generations, 
            np.array(mean_fitness) - np.array(std_fitness),
            np.array(mean_fitness) + np.array(std_fitness),
            alpha=0.2, color='blue', label='±1 Std Dev'
        )
        
        ax.set_title("Fitness Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_diversity_evolution(self, ax: plt.Axes) -> None:
        """
        Plot diversity evolution over generations.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.diversity_metrics:
            ax.text(0.5, 0.5, "No diversity metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Diversity Evolution")
            return
            
        # Extract diversity metrics
        generations = list(range(len(self.results.diversity_metrics)))
        hyperparam_diversity = [m['hyperparam_diversity'] for m in self.results.diversity_metrics]
        reward_diversity = [m['reward_diversity'] for m in self.results.diversity_metrics]
        architecture_diversity = [m['architecture_diversity'] for m in self.results.diversity_metrics]
        fitness_diversity = [m['fitness_diversity'] for m in self.results.diversity_metrics]
        
        # Plot diversity evolution
        ax.plot(generations, hyperparam_diversity, 'o-', label='Hyperparameter', color='blue')
        ax.plot(generations, reward_diversity, 'o-', label='Reward', color='green')
        ax.plot(generations, architecture_diversity, 'o-', label='Architecture', color='red')
        ax.plot(generations, fitness_diversity, 'o-', label='Fitness', color='purple')
        
        ax.set_title("Diversity Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Diversity Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_parameter_evolution(self, ax: plt.Axes) -> None:
        """
        Plot parameter evolution over generations.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Evolution")
            return
            
        # Extract parameter evolution for a few key parameters
        generations = [h['generation'] for h in self.results.population_history]
        
        # Get parameter names from first generation
        if not self.results.population_history[0]['members']:
            ax.text(0.5, 0.5, "No members in first generation", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Evolution")
            return
            
        first_member = self.results.population_history[0]['members'][0]
        hyperparams = first_member.get('hyperparams', {})
        reward_params = first_member.get('reward_params', {})
        
        # Select a few key parameters to plot
        key_params = []
        if 'learning_rate' in hyperparams:
            key_params.append(('learning_rate', 'hyperparams', 'Learning Rate'))
        if 'ent_coef' in hyperparams:
            key_params.append(('ent_coef', 'hyperparams', 'Entropy Coef'))
        if 'pnl_weight' in reward_params:
            key_params.append(('pnl_weight', 'reward_params', 'PNL Weight'))
        if 'drawdown_penalty' in reward_params:
            key_params.append(('drawdown_penalty', 'reward_params', 'Drawdown Penalty'))
            
        # Limit to 4 parameters
        key_params = key_params[:4]
        
        if not key_params:
            ax.text(0.5, 0.5, "No key parameters found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Evolution")
            return
            
        # Plot parameter evolution
        colors = plt.cm.tab10(np.linspace(0, 1, len(key_params)))
        
        for i, (param_name, param_type, label) in enumerate(key_params):
            param_values = []
            
            for gen_data in self.results.population_history:
                # Get mean parameter value for this generation
                values = []
                for member in gen_data['members']:
                    if param_type in member and param_name in member[param_type]:
                        values.append(member[param_type][param_name])
                        
                if values:
                    param_values.append(np.mean(values))
                else:
                    param_values.append(np.nan)
                    
            ax.plot(generations, param_values, 'o-', label=label, color=colors[i], alpha=0.7)
            
        ax.set_title("Parameter Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Parameter Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_best_member_lineage(self, ax: plt.Axes) -> None:
        """
        Plot best member lineage over generations.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Best Member Lineage")
            return
            
        # Extract best member from each generation
        generations = []
        best_fitness = []
        member_ids = []
        
        for gen_data in self.results.population_history:
            generations.append(gen_data['generation'])
            
            # Find best member in this generation
            best_member = None
            best_fitness_val = -np.inf
            
            for member in gen_data['members']:
                fitness = member.get('performance', {}).get('sharpe_ratio', 0.0)
                if fitness > best_fitness_val:
                    best_fitness_val = fitness
                    best_member = member
                    
            if best_member:
                best_fitness.append(best_fitness_val)
                member_ids.append(best_member['member_id'])
            else:
                best_fitness.append(np.nan)
                member_ids.append(None)
                
        # Plot best member lineage
        ax.plot(generations, best_fitness, 'o-', color='green', linewidth=2)
        
        # Add member ID labels
        for i, (gen, member_id) in enumerate(zip(generations, member_ids)):
            if member_id is not None and i % 2 == 0:  # Label every other point to avoid crowding
                ax.annotate(f"ID: {member_id}", (gen, best_fitness[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                
        ax.set_title("Best Member Lineage")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.grid(True, alpha=0.3)
        
    def _plot_age_distribution(self, ax: plt.Axes) -> None:
        """
        Plot age distribution of the final population.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Age Distribution")
            return
            
        # Get final population
        final_population = self.results.population_history[-1]['members']
        
        # Extract ages
        ages = [member.get('age', 0) for member in final_population]
        
        # Create histogram
        ax.hist(ages, bins=20, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_age = np.mean(ages)
        median_age = np.median(ages)
        
        ax.axvline(mean_age, color='red', linestyle='--', label=f'Mean: {mean_age:.1f}')
        ax.axvline(median_age, color='green', linestyle='--', label=f'Median: {median_age:.1f}')
        
        ax.set_title("Age Distribution (Final Population)")
        ax.set_xlabel("Age (Generations)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_performance_distribution(self, ax: plt.Axes) -> None:
        """
        Plot performance distribution of the final population.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Performance Distribution")
            return
            
        # Get final population
        final_population = self.results.population_history[-1]['members']
        
        # Extract performance metrics
        performance_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        
        for i, metric in enumerate(performance_metrics):
            values = []
            for member in final_population:
                if metric in member.get('performance', {}):
                    values.append(member['performance'][metric])
                    
            if values:
                # Create histogram for each metric
                ax.hist(values, bins=15, alpha=0.5, label=metric, edgecolor='black')
                
        ax.set_title("Performance Distribution (Final Population)")
        ax.set_xlabel("Metric Value")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_hyperparam_correlation(self, ax: plt.Axes) -> None:
        """
        Plot hyperparameter correlation heatmap.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Hyperparameter Correlation")
            return
            
        # Get final population
        final_population = self.results.population_history[-1]['members']
        
        # Extract hyperparameters
        hyperparam_data = {}
        for member in final_population:
            for param_name, param_value in member.get('hyperparams', {}).items():
                if param_name not in hyperparam_data:
                    hyperparam_data[param_name] = []
                hyperparam_data[param_name].append(param_value)
                
        # Create DataFrame
        df = pd.DataFrame(hyperparam_data)
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, "Insufficient numeric hyperparameters", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Hyperparameter Correlation")
            return
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Hyperparameter Correlation")
        
    def _plot_reward_param_correlation(self, ax: plt.Axes) -> None:
        """
        Plot reward parameter correlation heatmap.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Reward Parameter Correlation")
            return
            
        # Get final population
        final_population = self.results.population_history[-1]['members']
        
        # Extract reward parameters
        reward_param_data = {}
        for member in final_population:
            for param_name, param_value in member.get('reward_params', {}).items():
                if param_name not in reward_param_data:
                    reward_param_data[param_name] = []
                reward_param_data[param_name].append(param_value)
                
        # Create DataFrame
        df = pd.DataFrame(reward_param_data)
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, "Insufficient numeric reward parameters", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Reward Parameter Correlation")
            return
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Reward Parameter Correlation")
        
    def _plot_generation_stats(self, ax: plt.Axes) -> None:
        """
        Plot generation statistics.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.generation_stats:
            ax.text(0.5, 0.5, "No generation statistics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Generation Statistics")
            return
            
        # Extract statistics
        generations = [stat['generation'] for stat in self.results.generation_stats]
        best_fitness = [stat['best_fitness'] for stat in self.results.generation_stats]
        worst_fitness = [stat['worst_fitness'] for stat in self.results.generation_stats]
        mean_fitness = [stat['mean_fitness'] for stat in self.results.generation_stats]
        median_fitness = [stat['median_fitness'] for stat in self.results.generation_stats]
        
        # Plot statistics
        ax.plot(generations, best_fitness, 'o-', label='Best', color='green')
        ax.plot(generations, worst_fitness, 'o-', label='Worst', color='red')
        ax.plot(generations, mean_fitness, 'o-', label='Mean', color='blue')
        ax.plot(generations, median_fitness, 'o-', label='Median', color='purple')
        
        ax.set_title("Generation Statistics")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_population_heatmap(self, ax: plt.Axes) -> None:
        """
        Plot population heatmap showing fitness over generations.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.population_history:
            ax.text(0.5, 0.5, "No population history available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Population Heatmap")
            return
            
        # Create fitness matrix
        max_members = max(len(gen['members']) for gen in self.results.population_history)
        n_generations = len(self.results.population_history)
        
        fitness_matrix = np.full((n_generations, max_members), np.nan)
        
        for i, gen_data in enumerate(self.results.population_history):
            for j, member in enumerate(gen_data['members']):
                fitness = member.get('performance', {}).get('sharpe_ratio', np.nan)
                fitness_matrix[i, j] = fitness
                
        # Create heatmap
        im = ax.imshow(fitness_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fitness Score')
        
        ax.set_title("Population Fitness Heatmap")
        ax.set_xlabel("Member Index")
        ax.set_ylabel("Generation")
        
    def create_member_evolution_plot(self, member_id: int) -> str:
        """
        Create detailed plot for a specific member's evolution.
        
        Args:
            member_id: Member ID to plot
            
        Returns:
            Path to saved plot
        """
        if not self.results.population_history:
            logger.error("No population history available")
            return ""
            
        # Find member in population history
        member_history = []
        
        for gen_data in self.results.population_history:
            for member in gen_data['members']:
                if member['member_id'] == member_id:
                    member_history.append({
                        'generation': gen_data['generation'],
                        'member': member
                    })
                    break
                    
        if not member_history:
            logger.error(f"Member {member_id} not found in population history")
            return ""
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Member {member_id} Evolution", fontsize=16)
        
        # Plot 1: Fitness evolution
        ax1 = axes[0, 0]
        self._plot_member_fitness_evolution(member_history, ax1)
        
        # Plot 2: Hyperparameter evolution
        ax2 = axes[0, 1]
        self._plot_member_hyperparam_evolution(member_history, ax2)
        
        # Plot 3: Reward parameter evolution
        ax3 = axes[1, 0]
        self._plot_member_reward_param_evolution(member_history, ax3)
        
        # Plot 4: Age and generation
        ax4 = axes[1, 1]
        self._plot_member_age_generation(member_history, ax4)
        
        # Save plot
        plot_path = self.output_dir / f"member_{member_id}_evolution.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Member {member_id} evolution saved to {plot_path}")
        return str(plot_path)
        
    def _plot_member_fitness_evolution(self, member_history: List[Dict], ax: plt.Axes) -> None:
        """
        Plot member fitness evolution.
        
        Args:
            member_history: Member history data
            ax: Matplotlib axes
        """
        generations = [h['generation'] for h in member_history]
        fitness = [h['member'].get('performance', {}).get('sharpe_ratio', np.nan) for h in member_history]
        
        ax.plot(generations, fitness, 'o-', color='blue', linewidth=2)
        ax.set_title("Fitness Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(True, alpha=0.3)
        
    def _plot_member_hyperparam_evolution(self, member_history: List[Dict], ax: plt.Axes) -> None:
        """
        Plot member hyperparameter evolution.
        
        Args:
            member_history: Member history data
            ax: Matplotlib axes
        """
        generations = [h['generation'] for h in member_history]
        
        # Get hyperparameter names
        if not member_history:
            return
            
        hyperparams = member_history[0]['member'].get('hyperparams', {})
        param_names = list(hyperparams.keys())
        
        # Limit to 4 parameters
        param_names = param_names[:4]
        
        if not param_names:
            ax.text(0.5, 0.5, "No hyperparameters found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Hyperparameter Evolution")
            return
            
        # Plot parameter evolution
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
        
        for i, param_name in enumerate(param_names):
            values = []
            for h in member_history:
                value = h['member'].get('hyperparams', {}).get(param_name, np.nan)
                values.append(value)
                
            ax.plot(generations, values, 'o-', label=param_name, color=colors[i], alpha=0.7)
            
        ax.set_title("Hyperparameter Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Parameter Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_member_reward_param_evolution(self, member_history: List[Dict], ax: plt.Axes) -> None:
        """
        Plot member reward parameter evolution.
        
        Args:
            member_history: Member history data
            ax: Matplotlib axes
        """
        generations = [h['generation'] for h in member_history]
        
        # Get reward parameter names
        if not member_history:
            return
            
        reward_params = member_history[0]['member'].get('reward_params', {})
        param_names = list(reward_params.keys())
        
        # Limit to 4 parameters
        param_names = param_names[:4]
        
        if not param_names:
            ax.text(0.5, 0.5, "No reward parameters found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Reward Parameter Evolution")
            return
            
        # Plot parameter evolution
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
        
        for i, param_name in enumerate(param_names):
            values = []
            for h in member_history:
                value = h['member'].get('reward_params', {}).get(param_name, np.nan)
                values.append(value)
                
            ax.plot(generations, values, 'o-', label=param_name, color=colors[i], alpha=0.7)
            
        ax.set_title("Reward Parameter Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Parameter Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_member_age_generation(self, member_history: List[Dict], ax: plt.Axes) -> None:
        """
        Plot member age and generation.
        
        Args:
            member_history: Member history data
            ax: Matplotlib axes
        """
        generations = [h['generation'] for h in member_history]
        ages = [h['member'].get('age', 0) for h in member_history]
        
        ax.plot(generations, ages, 'o-', color='green', linewidth=2)
        ax.set_title("Age Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Age")
        ax.grid(True, alpha=0.3)
        
    def create_diversity_analysis_plot(self) -> str:
        """
        Create diversity analysis plot.
        
        Returns:
            Path to saved plot
        """
        if not self.results.diversity_metrics:
            logger.error("No diversity metrics available")
            return ""
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Diversity Analysis", fontsize=16)
        
        # Plot 1: Diversity evolution
        ax1 = axes[0, 0]
        self._plot_diversity_evolution(ax1)
        
        # Plot 2: Diversity correlation
        ax2 = axes[0, 1]
        self._plot_diversity_correlation(ax2)
        
        # Plot 3: Diversity vs fitness
        ax3 = axes[1, 0]
        self._plot_diversity_vs_fitness(ax3)
        
        # Plot 4: Diversity distribution
        ax4 = axes[1, 1]
        self._plot_diversity_distribution(ax4)
        
        # Save plot
        plot_path = self.output_dir / "diversity_analysis.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diversity analysis saved to {plot_path}")
        return str(plot_path)
        
    def _plot_diversity_correlation(self, ax: plt.Axes) -> None:
        """
        Plot diversity correlation heatmap.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.diversity_metrics:
            ax.text(0.5, 0.5, "No diversity metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Diversity Correlation")
            return
            
        # Create DataFrame from diversity metrics
        df = pd.DataFrame(self.results.diversity_metrics)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, "Insufficient diversity metrics", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Diversity Correlation")
            return
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Diversity Correlation")
        
    def _plot_diversity_vs_fitness(self, ax: plt.Axes) -> None:
        """
        Plot diversity vs fitness relationship.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.diversity_metrics or not self.results.generation_stats:
            ax.text(0.5, 0.5, "Insufficient data for diversity vs fitness plot", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Diversity vs Fitness")
            return
            
        # Extract data
        diversity_data = self.results.diversity_metrics
        fitness_data = self.results.generation_stats
        
        # Ensure same length
        min_len = min(len(diversity_data), len(fitness_data))
        diversity_data = diversity_data[:min_len]
        fitness_data = fitness_data[:min_len]
        
        # Extract metrics
        hyperparam_diversity = [m['hyperparam_diversity'] for m in diversity_data]
        reward_diversity = [m['reward_diversity'] for m in diversity_data]
        mean_fitness = [m['mean_fitness'] for m in fitness_data]
        
        # Create scatter plot
        ax.scatter(hyperparam_diversity, mean_fitness, label='Hyperparameter Diversity', alpha=0.7)
        ax.scatter(reward_diversity, mean_fitness, label='Reward Diversity', alpha=0.7)
        
        # Add trend lines
        if len(hyperparam_diversity) > 1:
            z = np.polyfit(hyperparam_diversity, mean_fitness, 1)
            p = np.poly1d(z)
            ax.plot(hyperparam_diversity, p(hyperparam_diversity), "r--", alpha=0.5)
            
        if len(reward_diversity) > 1:
            z = np.polyfit(reward_diversity, mean_fitness, 1)
            p = np.poly1d(z)
            ax.plot(reward_diversity, p(reward_diversity), "g--", alpha=0.5)
            
        ax.set_title("Diversity vs Fitness")
        ax.set_xlabel("Diversity Score")
        ax.set_ylabel("Mean Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_diversity_distribution(self, ax: plt.Axes) -> None:
        """
        Plot diversity distribution.
        
        Args:
            ax: Matplotlib axes
        """
        if not self.results.diversity_metrics:
            ax.text(0.5, 0.5, "No diversity metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Diversity Distribution")
            return
            
        # Extract diversity metrics
        hyperparam_diversity = [m['hyperparam_diversity'] for m in self.results.diversity_metrics]
        reward_diversity = [m['reward_diversity'] for m in self.results.diversity_metrics]
        architecture_diversity = [m['architecture_diversity'] for m in self.results.diversity_metrics]
        fitness_diversity = [m['fitness_diversity'] for m in self.results.diversity_metrics]
        
        # Create histograms
        ax.hist(hyperparam_diversity, bins=15, alpha=0.5, label='Hyperparameter', edgecolor='black')
        ax.hist(reward_diversity, bins=15, alpha=0.5, label='Reward', edgecolor='black')
        ax.hist(architecture_diversity, bins=15, alpha=0.5, label='Architecture', edgecolor='black')
        ax.hist(fitness_diversity, bins=15, alpha=0.5, label='Fitness', edgecolor='black')
        
        ax.set_title("Diversity Distribution")
        ax.set_xlabel("Diversity Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)