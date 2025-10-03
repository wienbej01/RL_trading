"""
Visualization and analysis for Optuna Hyperparameter Optimization results.

This module provides comprehensive visualization capabilities for HPO results,
including optimization history, parameter importance, and multi-objective analysis.
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
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_pareto_front, plot_contour, plot_parallel_coordinate
from optuna.visualization import plot_slice, plot_edf

from .optuna_hpo import HPOResults, HPOConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class HPOVisualizer:
    """
    Visualizer for Optuna Hyperparameter Optimization results.
    """
    
    def __init__(self, results: HPOResults, output_dir: str = "hpo_plots"):
        """
        Initialize HPO visualizer.
        
        Args:
            results: HPO results to visualize
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
        if self.results.study is None:
            logger.error("No study to visualize")
            return ""
            
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Optimization history
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_optimization_history(ax1)
        
        # 2. Parameter importance
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_param_importance(ax2)
        
        # 3. Trials distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_trials_distribution(ax3)
        
        # 4. Parameter relationships
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_parameter_relationships(ax4)
        
        # 5. Best parameters evolution
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_best_params_evolution(ax5)
        
        # 6. Objective value distribution
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_objective_distribution(ax6)
        
        # 7. Parallel coordinates (if multi-objective)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_parallel_coordinates(ax7)
        
        # 8. Slice plot
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_slice(ax8)
        
        # 9. EDF plot
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_edf(ax9)
        
        # 10. Contour plot
        ax10 = fig.add_subplot(gs[3, :])
        self._plot_contour(ax10)
        
        # Add title
        fig.suptitle(
            f"HPO Summary Dashboard - {len(self.results.study.trials)} Trials",
            fontsize=16,
            y=0.98
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / "hpo_dashboard.png"
        plt.savefig(dashboard_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved to {dashboard_path}")
        return str(dashboard_path)
        
    def _plot_optimization_history(self, ax: plt.Axes) -> None:
        """
        Plot optimization history.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None:
            ax.text(0.5, 0.5, "No study available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Optimization History")
            return
            
        try:
            # Get optimization history data
            trials = self.results.study.trials
            trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if self.results.config.multi_objective:
                # For multi-objective, plot each objective separately
                objectives = self.results.config.objectives
                colors = plt.cm.tab10(np.linspace(0, 1, len(objectives)))
                
                for i, obj_name in enumerate(objectives):
                    values = [t.values[i] for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
                    ax.plot(trial_numbers, values, 'o-', label=obj_name, color=colors[i], alpha=0.7)
                    
                ax.legend()
            else:
                # For single objective, plot the values
                values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
                ax.plot(trial_numbers, values, 'o-', color='blue', alpha=0.7)
                
                # Add best value line
                if values:
                    best_value = max(values) if self.results.config.direction == 'maximize' else min(values)
                    ax.axhline(y=best_value, color='red', linestyle='--', alpha=0.5, label='Best Value')
                    ax.legend()
                    
            ax.set_title("Optimization History")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Objective Value")
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"Could not plot optimization history: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Optimization History")
            
    def _plot_param_importance(self, ax: plt.Axes) -> None:
        """
        Plot parameter importance.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None or self.results.config.multi_objective:
            ax.text(0.5, 0.5, "Parameter importance not available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Importance")
            return
            
        try:
            # Get parameter importance
            importance = optuna.importance.get_param_importances(self.results.study)
            
            if not importance:
                ax.text(0.5, 0.5, "No parameter importance data", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Parameter Importance")
                return
                
            # Create horizontal bar chart
            params = list(importance.keys())
            values = list(importance.values())
            
            y_pos = np.arange(len(params))
            ax.barh(y_pos, values, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.invert_yaxis()  # To display from top to bottom
            ax.set_xlabel("Importance")
            ax.set_title("Parameter Importance")
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                
        except Exception as e:
            logger.warning(f"Could not plot parameter importance: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Importance")
            
    def _plot_trials_distribution(self, ax: plt.Axes) -> None:
        """
        Plot trials state distribution.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None:
            ax.text(0.5, 0.5, "No study available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Trials Distribution")
            return
            
        # Count trials by state
        trials = self.results.study.trials
        states = [t.state.name for t in trials]
        state_counts = pd.Series(states).value_counts()
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(state_counts)))
        wedges, texts, autotexts = ax.pie(
            state_counts.values, 
            labels=state_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        ax.set_title("Trials Distribution")
        
    def _plot_parameter_relationships(self, ax: plt.Axes) -> None:
        """
        Plot parameter relationships.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None or self.results.trials_df is None:
            ax.text(0.5, 0.5, "No trial data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Relationships")
            return
            
        try:
            # Get completed trials
            completed_trials = self.results.trials_df[
                self.results.trials_df['state'] == 'COMPLETE'
            ]
            
            if completed_trials.empty:
                ax.text(0.5, 0.5, "No completed trials", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Parameter Relationships")
                return
                
            # Get parameter columns
            param_cols = [col for col in completed_trials.columns 
                         if col.startswith('params_')]
            
            if len(param_cols) < 2:
                ax.text(0.5, 0.5, "Insufficient parameters for relationship plot", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Parameter Relationships")
                return
                
            # Select first two parameters
            param1 = param_cols[0]
            param2 = param_cols[1]
            
            # Create scatter plot
            x = completed_trials[param1]
            y = completed_trials[param2]
            c = completed_trials['value']
            
            scatter = ax.scatter(x, y, c=c, cmap='viridis', alpha=0.7)
            ax.set_xlabel(param1.replace('params_', ''))
            ax.set_ylabel(param2.replace('params_', ''))
            ax.set_title("Parameter Relationships")
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Objective Value')
            
        except Exception as e:
            logger.warning(f"Could not plot parameter relationships: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parameter Relationships")
            
    def _plot_best_params_evolution(self, ax: plt.Axes) -> None:
        """
        Plot best parameters evolution.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None or self.results.trials_df is None:
            ax.text(0.5, 0.5, "No trial data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Best Parameters Evolution")
            return
            
        try:
            # Get completed trials sorted by value
            completed_trials = self.results.trials_df[
                self.results.trials_df['state'] == 'COMPLETE'
            ].copy()
            
            if completed_trials.empty:
                ax.text(0.5, 0.5, "No completed trials", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Best Parameters Evolution")
                return
                
            # Sort by objective value
            if self.results.config.direction == 'maximize':
                completed_trials = completed_trials.sort_values('value', ascending=False)
            else:
                completed_trials = completed_trials.sort_values('value', ascending=True)
                
            # Get parameter columns
            param_cols = [col for col in completed_trials.columns 
                         if col.startswith('params_')]
            
            if not param_cols:
                ax.text(0.5, 0.5, "No parameter columns found", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Best Parameters Evolution")
                return
                
            # Select top 20 trials
            top_trials = completed_trials.head(20)
            
            # Normalize parameter values for plotting
            for col in param_cols:
                if top_trials[col].dtype in ['float64', 'int64']:
                    min_val = top_trials[col].min()
                    max_val = top_trials[col].max()
                    if max_val > min_val:
                        top_trials[f'{col}_norm'] = (top_trials[col] - min_val) / (max_val - min_val)
                    else:
                        top_trials[f'{col}_norm'] = 0.5
                        
            # Plot normalized parameter evolution
            for col in param_cols[:5]:  # Limit to first 5 parameters
                norm_col = f'{col}_norm'
                if norm_col in top_trials.columns:
                    ax.plot(
                        range(len(top_trials)), 
                        top_trials[norm_col], 
                        'o-', 
                        label=col.replace('params_', ''),
                        alpha=0.7
                    )
                    
            ax.set_title("Best Parameters Evolution (Normalized)")
            ax.set_xlabel("Top Trial Rank")
            ax.set_ylabel("Normalized Parameter Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"Could not plot best parameters evolution: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Best Parameters Evolution")
            
    def _plot_objective_distribution(self, ax: plt.Axes) -> None:
        """
        Plot objective value distribution.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None or self.results.trials_df is None:
            ax.text(0.5, 0.5, "No trial data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Objective Distribution")
            return
            
        try:
            # Get completed trials
            completed_trials = self.results.trials_df[
                self.results.trials_df['state'] == 'COMPLETE'
            ]
            
            if completed_trials.empty:
                ax.text(0.5, 0.5, "No completed trials", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Objective Distribution")
                return
                
            # Create histogram
            values = completed_trials['value']
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()
            
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', 
                      label=f'Median: {median_val:.3f}')
            
            ax.set_title("Objective Value Distribution")
            ax.set_xlabel("Objective Value")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"Could not plot objective distribution: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Objective Distribution")
            
    def _plot_parallel_coordinates(self, ax: plt.Axes) -> None:
        """
        Plot parallel coordinates.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None:
            ax.text(0.5, 0.5, "No study available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parallel Coordinates")
            return
            
        try:
            # Use Optuna's parallel coordinate plot
            fig = plot_parallel_coordinate(self.results.study)
            
            # Save to temporary file and load back
            temp_path = self.output_dir / "temp_parallel_coordinates.png"
            fig.write_image(temp_path)
            
            # Load and display on the given axes
            img = plt.imread(temp_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Parallel Coordinates")
            
            # Clean up temporary file
            temp_path.unlink()
            
        except Exception as e:
            logger.warning(f"Could not plot parallel coordinates: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Parallel Coordinates")
            
    def _plot_slice(self, ax: plt.Axes) -> None:
        """
        Plot slice plot.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None:
            ax.text(0.5, 0.5, "No study available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Slice Plot")
            return
            
        try:
            # Use Optuna's slice plot
            fig = plot_slice(self.results.study)
            
            # Save to temporary file and load back
            temp_path = self.output_dir / "temp_slice.png"
            fig.write_image(temp_path)
            
            # Load and display on the given axes
            img = plt.imread(temp_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Slice Plot")
            
            # Clean up temporary file
            temp_path.unlink()
            
        except Exception as e:
            logger.warning(f"Could not plot slice: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Slice Plot")
            
    def _plot_edf(self, ax: plt.Axes) -> None:
        """
        Plot EDF (Empirical Distribution Function).
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None:
            ax.text(0.5, 0.5, "No study available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("EDF Plot")
            return
            
        try:
            # Use Optuna's EDF plot
            fig = plot_edf(self.results.study)
            
            # Save to temporary file and load back
            temp_path = self.output_dir / "temp_edf.png"
            fig.write_image(temp_path)
            
            # Load and display on the given axes
            img = plt.imread(temp_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("EDF Plot")
            
            # Clean up temporary file
            temp_path.unlink()
            
        except Exception as e:
            logger.warning(f"Could not plot EDF: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("EDF Plot")
            
    def _plot_contour(self, ax: plt.Axes) -> None:
        """
        Plot contour plot.
        
        Args:
            ax: Matplotlib axes
        """
        if self.results.study is None:
            ax.text(0.5, 0.5, "No study available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Contour Plot")
            return
            
        try:
            # Use Optuna's contour plot
            fig = plot_contour(self.results.study)
            
            # Save to temporary file and load back
            temp_path = self.output_dir / "temp_contour.png"
            fig.write_image(temp_path)
            
            # Load and display on the given axes
            img = plt.imread(temp_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Contour Plot")
            
            # Clean up temporary file
            temp_path.unlink()
            
        except Exception as e:
            logger.warning(f"Could not plot contour: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Contour Plot")
            
    def create_pareto_front_plot(self) -> str:
        """
        Create Pareto front plot for multi-objective optimization.
        
        Returns:
            Path to saved plot
        """
        if self.results.study is None or not self.results.config.multi_objective:
            logger.warning("Pareto front plot only available for multi-objective studies")
            return ""
            
        try:
            # Use Optuna's Pareto front plot
            fig = plot_pareto_front(self.results.study)
            
            # Save plot
            pareto_path = self.output_dir / "pareto_front.png"
            fig.write_image(pareto_path)
            
            logger.info(f"Pareto front plot saved to {pareto_path}")
            return str(pareto_path)
            
        except Exception as e:
            logger.warning(f"Could not create Pareto front plot: {e}")
            return ""
            
    def create_trial_details_plot(self, trial_number: int) -> str:
        """
        Create detailed plot for a specific trial.
        
        Args:
            trial_number: Trial number to plot
            
        Returns:
            Path to saved plot
        """
        if self.results.study is None:
            logger.error("No study available")
            return ""
            
        # Find the trial
        trial = None
        for t in self.results.study.trials:
            if t.number == trial_number:
                trial = t
                break
                
        if trial is None:
            logger.error(f"Trial {trial_number} not found")
            return ""
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Trial {trial_number} Detailed Report", fontsize=16)
        
        # Plot 1: Trial summary
        ax1 = axes[0, 0]
        self._plot_trial_summary(trial, ax1)
        
        # Plot 2: Parameter values
        ax2 = axes[0, 1]
        self._plot_trial_params(trial, ax2)
        
        # Plot 3: Trial timeline
        ax3 = axes[1, 0]
        self._plot_trial_timeline(trial, ax3)
        
        # Plot 4: Trial comparison
        ax4 = axes[1, 1]
        self._plot_trial_comparison(trial, ax4)
        
        # Save plot
        plot_path = self.output_dir / f"trial_{trial_number}_details.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trial {trial_number} details saved to {plot_path}")
        return str(plot_path)
        
    def _plot_trial_summary(self, trial: optuna.Trial, ax: plt.Axes) -> None:
        """
        Plot trial summary.
        
        Args:
            trial: Optuna trial
            ax: Matplotlib axes
        """
        # Create summary text
        summary_text = f"Trial Number: {trial.number}\n"
        summary_text += f"State: {trial.state.name}\n"
        
        if trial.value is not None:
            summary_text += f"Value: {trial.value:.4f}\n"
            
        if trial.values is not None:
            summary_text += "Values:\n"
            for i, val in enumerate(trial.values):
                obj_name = self.results.config.objectives[i] if i < len(self.results.config.objectives) else f"Objective {i}"
                summary_text += f"  {obj_name}: {val:.4f}\n"
                
        if trial.datetime_start:
            summary_text += f"Start: {trial.datetime_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
        if trial.datetime_complete:
            summary_text += f"Complete: {trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            duration = trial.datetime_complete - trial.datetime_start
            summary_text += f"Duration: {duration.total_seconds():.2f} seconds\n"
            
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace')
        ax.set_title("Trial Summary")
        ax.axis('off')
        
    def _plot_trial_params(self, trial: optuna.Trial, ax: plt.Axes) -> None:
        """
        Plot trial parameters.
        
        Args:
            trial: Optuna trial
            ax: Matplotlib axes
        """
        if not trial.params:
            ax.text(0.5, 0.5, "No parameters", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Trial Parameters")
            return
            
        # Create parameter table
        param_names = list(trial.params.keys())
        param_values = list(trial.params.values())
        
        # Create bar chart
        y_pos = np.arange(len(param_names))
        bars = ax.barh(y_pos, param_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.invert_yaxis()  # To display from top to bottom
        ax.set_title("Trial Parameters")
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, param_values)):
            if isinstance(value, float):
                ax.text(value + 0.01, i, f'{value:.3f}', va='center')
            else:
                ax.text(value + 0.01, i, str(value), va='center')
                
    def _plot_trial_timeline(self, trial: optuna.Trial, ax: plt.Axes) -> None:
        """
        Plot trial timeline.
        
        Args:
            trial: Optuna trial
            ax: Matplotlib axes
        """
        if not trial.datetime_start or not trial.datetime_complete:
            ax.text(0.5, 0.5, "No timeline data", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Trial Timeline")
            return
            
        # Create timeline
        events = [
            (trial.datetime_start, "Start", "green"),
            (trial.datetime_complete, "Complete", "blue")
        ]
        
        # Plot timeline
        for i, (date, label, color) in enumerate(events):
            ax.axvline(x=date, color=color, linestyle='--', alpha=0.7)
            ax.text(date, i + 0.1, label, rotation=45, ha='left', va='bottom')
            
        ax.set_title("Trial Timeline")
        ax.set_ylabel("Events")
        ax.set_yticks([])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    def _plot_trial_comparison(self, trial: optuna.Trial, ax: plt.Axes) -> None:
        """
        Plot trial comparison with other trials.
        
        Args:
            trial: Optuna trial
            ax: Matplotlib axes
        """
        if self.results.trials_df is None:
            ax.text(0.5, 0.5, "No trial data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Trial Comparison")
            return
            
        try:
            # Get all completed trials
            completed_trials = self.results.trials_df[
                self.results.trials_df['state'] == 'COMPLETE'
            ]
            
            if completed_trials.empty:
                ax.text(0.5, 0.5, "No completed trials", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Trial Comparison")
                return
                
            # Create histogram of all values
            values = completed_trials['value']
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black', label='All Trials')
            
            # Highlight current trial
            if trial.value is not None:
                ax.axvline(trial.value, color='red', linestyle='--', 
                          linewidth=2, label=f'Trial {trial.number}')
                
            ax.set_title("Trial Comparison")
            ax.set_xlabel("Objective Value")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"Could not plot trial comparison: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Trial Comparison")