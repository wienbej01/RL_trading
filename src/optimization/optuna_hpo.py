"""
Optuna Hyperparameter Optimization for multi-ticker RL trading.

This module implements hyperparameter optimization using Optuna for multi-ticker
RL trading systems, including multi-objective optimization, reward function
parameter tuning, and parallel execution.
"""

import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_pareto_front, plot_contour

from ..utils.logging import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


class HPOConfig:
    """
    Configuration for Hyperparameter Optimization.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = 1,
        sampler: str = 'tpe',
        pruner: str = 'median',
        direction: str = 'maximize',
        multi_objective: bool = False,
        objectives: List[str] = None,
        study_name: str = None,
        storage: str = None,
        output_dir: str = 'hpo_results',
        timeout: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize HPO configuration.
        
        Args:
            n_trials: Number of trials
            n_jobs: Number of parallel jobs
            sampler: Sampler type ('tpe', 'nsgaii')
            pruner: Pruner type ('median', 'hyperband')
            direction: Optimization direction ('maximize', 'minimize')
            multi_objective: Whether to use multi-objective optimization
            objectives: List of objective names for multi-objective optimization
            study_name: Name of the study
            storage: Database URL for study storage
            output_dir: Directory to save HPO results
            timeout: Timeout in seconds
            random_state: Random state for reproducibility
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction
        self.multi_objective = multi_objective
        self.objectives = objectives or ['sharpe_ratio']
        self.study_name = study_name or f"hpo_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.output_dir = output_dir
        self.timeout = timeout
        self.random_state = random_state
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config_dict = {
            'n_trials': self.n_trials,
            'n_jobs': self.n_jobs,
            'sampler': self.sampler,
            'pruner': self.pruner,
            'direction': self.direction,
            'multi_objective': self.multi_objective,
            'objectives': self.objectives,
            'study_name': self.study_name,
            'storage': self.storage,
            'output_dir': self.output_dir,
            'timeout': self.timeout,
            'random_state': self.random_state
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    @classmethod
    def load(cls, filepath: str) -> 'HPOConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            HPOConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)


class HPOResults:
    """
    Container for HPO results and analysis.
    """
    
    def __init__(self, config: HPOConfig):
        """
        Initialize HPO results.
        
        Args:
            config: HPO configuration
        """
        self.config = config
        self.study = None
        self.best_trial = None
        self.best_params = None
        self.trials_df = None
        self.param_importance = None
        self.optimization_history = None
        
    def set_study(self, study: optuna.Study) -> None:
        """
        Set the Optuna study.
        
        Args:
            study: Optuna study
        """
        self.study = study
        self.best_trial = study.best_trial
        self.best_params = study.best_params
        self.trials_df = study.trials_dataframe()
        
    def save(self, filepath: str) -> None:
        """
        Save HPO results to file.
        
        Args:
            filepath: Path to save results
        """
        if self.study is None:
            logger.warning("No study to save")
            return
            
        results = {
            'config': {
                'n_trials': self.config.n_trials,
                'n_jobs': self.config.n_jobs,
                'sampler': self.config.sampler,
                'pruner': self.config.pruner,
                'direction': self.config.direction,
                'multi_objective': self.config.multi_objective,
                'objectives': self.config.objectives,
                'study_name': self.config.study_name,
                'storage': self.config.storage,
                'output_dir': self.config.output_dir,
                'timeout': self.config.timeout,
                'random_state': self.config.random_state
            },
            'best_trial': {
                'number': self.best_trial.number,
                'value': self.best_trial.value if not self.config.multi_objective else None,
                'values': self.best_trial.values if self.config.multi_objective else None,
                'params': self.best_trial.params,
                'datetime_start': self.best_trial.datetime_start.isoformat() if self.best_trial.datetime_start else None,
                'datetime_complete': self.best_trial.datetime_complete.isoformat() if self.best_trial.datetime_complete else None
            },
            'trials_df': self.trials_df.to_dict() if self.trials_df is not None else None,
            'param_importance': self.param_importance,
            'optimization_history': self.optimization_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'HPOResults':
        """
        Load HPO results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            HPOResults instance
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
        # Reconstruct config
        config = HPOConfig(
            n_trials=results['config']['n_trials'],
            n_jobs=results['config']['n_jobs'],
            sampler=results['config']['sampler'],
            pruner=results['config']['pruner'],
            direction=results['config']['direction'],
            multi_objective=results['config']['multi_objective'],
            objectives=results['config']['objectives'],
            study_name=results['config']['study_name'],
            storage=results['config']['storage'],
            output_dir=results['config']['output_dir'],
            timeout=results['config']['timeout'],
            random_state=results['config']['random_state']
        )
        
        # Create HPOResults instance
        hpo_results = cls(config)
        hpo_results.trials_df = pd.DataFrame(results['trials_df']) if results['trials_df'] else None
        hpo_results.param_importance = results['param_importance']
        hpo_results.optimization_history = results['optimization_history']
        
        return hpo_results


class MultiTickerHPO:
    """
    Hyperparameter Optimization for multi-ticker RL trading systems.
    """
    
    def __init__(
        self,
        config: HPOConfig,
        objective_func: Callable,
        search_space: Dict[str, Any],
        reward_params_space: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-ticker HPO.
        
        Args:
            config: HPO configuration
            objective_func: Objective function to optimize
            search_space: Hyperparameter search space
            reward_params_space: Reward parameter search space (optional)
        """
        self.config = config
        self.objective_func = objective_func
        self.search_space = search_space
        self.reward_params_space = reward_params_space or {}
        self.results = HPOResults(config)
        
        # Create sampler
        if config.sampler == 'tpe':
            self.sampler = TPESampler(seed=config.random_state)
        elif config.sampler == 'nsgaii':
            self.sampler = NSGAIISampler(seed=config.random_state)
        else:
            logger.warning(f"Unknown sampler: {config.sampler}, using TPE")
            self.sampler = TPESampler(seed=config.random_state)
            
        # Create pruner
        if config.pruner == 'median':
            self.pruner = MedianPruner()
        elif config.pruner == 'hyperband':
            self.pruner = HyperbandPruner()
        else:
            logger.warning(f"Unknown pruner: {config.pruner}, using Median")
            self.pruner = MedianPruner()
            
    def optimize(self) -> HPOResults:
        """
        Run hyperparameter optimization.
        
        Returns:
            HPOResults instance
        """
        logger.info(f"Starting HPO with {self.config.n_trials} trials")
        
        # Create study
        if self.config.multi_objective:
            directions = ['maximize'] * len(self.config.objectives)
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=self.sampler,
                pruner=self.pruner,
                directions=directions,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=self.sampler,
                pruner=self.pruner,
                direction=self.config.direction,
                load_if_exists=True
            )
            
        # Define objective wrapper
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in self.search_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                    
            # Sample reward parameters if provided
            reward_params = {}
            for param_name, param_config in self.reward_params_space.items():
                if param_config['type'] == 'float':
                    reward_params[param_name] = trial.suggest_float(
                        f"reward_{param_name}",
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    reward_params[param_name] = trial.suggest_int(
                        f"reward_{param_name}",
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    reward_params[param_name] = trial.suggest_categorical(
                        f"reward_{param_name}",
                        param_config['choices']
                    )
                    
            # Call objective function
            return self.objective_func(trial, params, reward_params)
            
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=True
        )
        
        # Store results
        self.results.set_study(study)
        
        # Calculate parameter importance
        try:
            if not self.config.multi_objective:
                self.results.param_importance = optuna.importance.get_param_importances(study)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            
        # Save results
        results_path = os.path.join(self.config.output_dir, "hpo_results.pkl")
        self.results.save(results_path)
        
        logger.info("HPO completed successfully")
        return self.results
        
    def create_visualizations(self) -> Dict[str, str]:
        """
        Create HPO visualization plots.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        if self.results.study is None:
            logger.warning("No study to visualize")
            return {}
            
        plots = {}
        
        # Optimization history
        try:
            fig = plot_optimization_history(self.results.study)
            path = os.path.join(self.config.output_dir, "optimization_history.png")
            fig.write_image(path)
            plots['optimization_history'] = path
        except Exception as e:
            logger.warning(f"Could not create optimization history plot: {e}")
            
        # Parameter importance
        try:
            if not self.config.multi_objective and self.results.param_importance is not None:
                fig = plot_param_importances(self.results.study)
                path = os.path.join(self.config.output_dir, "param_importance.png")
                fig.write_image(path)
                plots['param_importance'] = path
        except Exception as e:
            logger.warning(f"Could not create parameter importance plot: {e}")
            
        # Pareto front (for multi-objective)
        try:
            if self.config.multi_objective:
                fig = plot_pareto_front(self.results.study)
                path = os.path.join(self.config.output_dir, "pareto_front.png")
                fig.write_image(path)
                plots['pareto_front'] = path
        except Exception as e:
            logger.warning(f"Could not create Pareto front plot: {e}")
            
        # Contour plot
        try:
            if not self.config.multi_objective and len(self.search_space) >= 2:
                fig = plot_contour(self.results.study)
                path = os.path.join(self.config.output_dir, "contour_plot.png")
                fig.write_image(path)
                plots['contour_plot'] = path
        except Exception as e:
            logger.warning(f"Could not create contour plot: {e}")
            
        return plots
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive HPO report.
        
        Returns:
            Dictionary with report data
        """
        if self.results.study is None:
            return {}
            
        report = {
            'config': {
                'n_trials': self.config.n_trials,
                'n_jobs': self.config.n_jobs,
                'sampler': self.config.sampler,
                'pruner': self.config.pruner,
                'direction': self.config.direction,
                'multi_objective': self.config.multi_objective,
                'objectives': self.config.objectives,
                'study_name': self.config.study_name
            },
            'summary': {
                'n_trials_completed': len(self.results.study.trials),
                'best_trial_number': self.results.best_trial.number,
                'best_value': self.results.best_trial.value if not self.config.multi_objective else None,
                'best_values': self.results.best_trial.values if self.config.multi_objective else None,
                'best_params': self.results.best_trial.params
            },
            'trials_summary': {
                'completed_trials': len([t for t in self.results.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'pruned_trials': len([t for t in self.results.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len([t for t in self.results.study.trials if t.state == optuna.trial.TrialState.FAIL])
            }
        }
        
        # Add parameter importance if available
        if self.results.param_importance is not None:
            report['param_importance'] = self.results.param_importance.to_dict()
            
        # Add trials statistics
        if self.results.trials_df is not None:
            report['trials_stats'] = {
                'value_mean': self.results.trials_df['value'].mean(),
                'value_std': self.results.trials_df['value'].std(),
                'value_min': self.results.trials_df['value'].min(),
                'value_max': self.results.trials_df['value'].max()
            }
            
        return report


def create_default_search_space() -> Dict[str, Any]:
    """
    Create default hyperparameter search space for PPO-LSTM.
    
    Returns:
        Dictionary with search space configuration
    """
    return {
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-3,
            'log': True
        },
        'n_steps': {
            'type': 'int',
            'low': 128,
            'high': 2048,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 16,
            'high': 256,
            'log': True
        },
        'n_epochs': {
            'type': 'int',
            'low': 3,
            'high': 10
        },
        'gamma': {
            'type': 'float',
            'low': 0.9,
            'high': 0.999
        },
        'gae_lambda': {
            'type': 'float',
            'low': 0.8,
            'high': 0.95
        },
        'clip_range': {
            'type': 'float',
            'low': 0.1,
            'high': 0.3
        },
        'ent_coef': {
            'type': 'float',
            'low': 1e-4,
            'high': 0.1,
            'log': True
        },
        'vf_coef': {
            'type': 'float',
            'low': 0.1,
            'high': 1.0
        },
        'max_grad_norm': {
            'type': 'float',
            'low': 0.1,
            'high': 1.0
        },
        'lstm_hidden_size': {
            'type': 'int',
            'low': 32,
            'high': 256,
            'log': True
        },
        'lstm_num_layers': {
            'type': 'int',
            'low': 1,
            'high': 3
        },
        'lstm_dropout': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5
        }
    }


def create_default_reward_params_space() -> Dict[str, Any]:
    """
    Create default reward parameter search space.
    
    Returns:
        Dictionary with reward parameter search space configuration
    """
    return {
        'pnl_weight': {
            'type': 'float',
            'low': 0.1,
            'high': 2.0
        },
        'dsr_weight': {
            'type': 'float',
            'low': 0.1,
            'high': 2.0
        },
        'sharpe_weight': {
            'type': 'float',
            'low': 0.1,
            'high': 2.0
        },
        'drawdown_penalty': {
            'type': 'float',
            'low': 0.1,
            'high': 5.0
        },
        'activity_penalty': {
            'type': 'float',
            'low': 0.001,
            'high': 0.1,
            'log': True
        },
        'regime_weight': {
            'type': 'float',
            'low': 0.5,
            'high': 2.0
        }
    }


def create_hpo_config_from_yaml(config_path: str) -> HPOConfig:
    """
    Create HPOConfig from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        HPOConfig instance
    """
    config = load_config(config_path)
    hpo_config_dict = config.get('hpo', {})
    
    return HPOConfig(
        n_trials=hpo_config_dict.get('n_trials', 100),
        n_jobs=hpo_config_dict.get('n_jobs', 1),
        sampler=hpo_config_dict.get('sampler', 'tpe'),
        pruner=hpo_config_dict.get('pruner', 'median'),
        direction=hpo_config_dict.get('direction', 'maximize'),
        multi_objective=hpo_config_dict.get('multi_objective', False),
        objectives=hpo_config_dict.get('objectives', ['sharpe_ratio']),
        study_name=hpo_config_dict.get('study_name'),
        storage=hpo_config_dict.get('storage'),
        output_dir=hpo_config_dict.get('output_dir', 'hpo_results'),
        timeout=hpo_config_dict.get('timeout'),
        random_state=hpo_config_dict.get('random_state')
    )