"""
Population-Based Training (PBT) for multi-ticker RL trading.

This module implements Population-Based Training for evolving multi-ticker
RL trading strategies, including population mutation, reward function evolution,
and performance monitoring.
"""

import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import yaml
from copy import deepcopy

from ..utils.logging import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


class PBTConfig:
    """
    Configuration for Population-Based Training.
    """
    
    def __init__(
        self,
        population_size: int = 10,
        generations: int = 20,
        eval_steps: int = 10000,
        truncation_fraction: float = 0.2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.2,
        reward_mutation_rate: float = 0.15,
        hyperparam_mutation_rate: float = 0.1,
        architecture_mutation_rate: float = 0.05,
        output_dir: str = 'pbt_results',
        checkpoint_interval: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize PBT configuration.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            eval_steps: Number of steps to evaluate each member
            truncation_fraction: Fraction of population to truncate each generation
            mutation_rate: Base mutation rate
            crossover_rate: Crossover rate for breeding
            elite_fraction: Fraction of elite members to preserve
            reward_mutation_rate: Mutation rate for reward parameters
            hyperparam_mutation_rate: Mutation rate for hyperparameters
            architecture_mutation_rate: Mutation rate for architecture changes
            output_dir: Directory to save PBT results
            checkpoint_interval: Interval for saving checkpoints
            random_state: Random state for reproducibility
        """
        self.population_size = population_size
        self.generations = generations
        self.eval_steps = eval_steps
        self.truncation_fraction = truncation_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.reward_mutation_rate = reward_mutation_rate
        self.hyperparam_mutation_rate = hyperparam_mutation_rate
        self.architecture_mutation_rate = architecture_mutation_rate
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.random_state = random_state
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config_dict = {
            'population_size': self.population_size,
            'generations': self.generations,
            'eval_steps': self.eval_steps,
            'truncation_fraction': self.truncation_fraction,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_fraction': self.elite_fraction,
            'reward_mutation_rate': self.reward_mutation_rate,
            'hyperparam_mutation_rate': self.hyperparam_mutation_rate,
            'architecture_mutation_rate': self.architecture_mutation_rate,
            'output_dir': self.output_dir,
            'checkpoint_interval': self.checkpoint_interval,
            'random_state': self.random_state
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    @classmethod
    def load(cls, filepath: str) -> 'PBTConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            PBTConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)


class PBTMember:
    """
    Represents a member of the PBT population.
    """
    
    def __init__(
        self,
        member_id: int,
        hyperparams: Dict[str, Any],
        reward_params: Dict[str, Any],
        architecture_params: Optional[Dict[str, Any]] = None,
        model_state: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, float]] = None
    ):
        """
        Initialize PBT member.
        
        Args:
            member_id: Unique identifier for the member
            hyperparams: Hyperparameter configuration
            reward_params: Reward parameter configuration
            architecture_params: Architecture parameter configuration
            model_state: Model state dictionary
            performance: Performance metrics
        """
        self.member_id = member_id
        self.hyperparams = hyperparams
        self.reward_params = reward_params
        self.architecture_params = architecture_params or {}
        self.model_state = model_state
        self.performance = performance or {}
        self.age = 0
        self.generation = 0
        self.parent_id = None
        
    def update_performance(self, performance: Dict[str, float]) -> None:
        """
        Update performance metrics.
        
        Args:
            performance: Dictionary of performance metrics
        """
        self.performance.update(performance)
        
    def get_fitness(self, primary_metric: str = 'sharpe_ratio') -> float:
        """
        Get fitness score based on primary metric.
        
        Args:
            primary_metric: Primary metric for fitness evaluation
            
        Returns:
            Fitness score
        """
        return self.performance.get(primary_metric, 0.0)
        
    def copy(self) -> 'PBTMember':
        """
        Create a deep copy of the member.
        
        Returns:
            Copy of the member
        """
        return PBTMember(
            member_id=self.member_id,
            hyperparams=deepcopy(self.hyperparams),
            reward_params=deepcopy(self.reward_params),
            architecture_params=deepcopy(self.architecture_params),
            model_state=deepcopy(self.model_state),
            performance=deepcopy(self.performance)
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert member to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'member_id': self.member_id,
            'hyperparams': self.hyperparams,
            'reward_params': self.reward_params,
            'architecture_params': self.architecture_params,
            'performance': self.performance,
            'age': self.age,
            'generation': self.generation,
            'parent_id': self.parent_id
        }


class PBTResults:
    """
    Container for PBT results and analysis.
    """
    
    def __init__(self, config: PBTConfig):
        """
        Initialize PBT results.
        
        Args:
            config: PBT configuration
        """
        self.config = config
        self.population_history = []
        self.best_member = None
        self.generation_stats = []
        self.diversity_metrics = []
        
    def add_generation(self, population: List[PBTMember], generation: int) -> None:
        """
        Add generation data to results.
        
        Args:
            population: Current population
            generation: Generation number
        """
        # Store population snapshot
        generation_data = {
            'generation': generation,
            'members': [member.to_dict() for member in population],
            'timestamp': datetime.now().isoformat()
        }
        self.population_history.append(generation_data)
        
        # Calculate generation statistics
        fitness_scores = [member.get_fitness() for member in population]
        stats = {
            'generation': generation,
            'best_fitness': max(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'mean_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'median_fitness': np.median(fitness_scores)
        }
        self.generation_stats.append(stats)
        
        # Update best member
        best_idx = np.argmax(fitness_scores)
        if self.best_member is None or fitness_scores[best_idx] > self.best_member.get_fitness():
            self.best_member = population[best_idx].copy()
            
    def save(self, filepath: str) -> None:
        """
        Save PBT results to file.
        
        Args:
            filepath: Path to save results
        """
        results = {
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'eval_steps': self.config.eval_steps,
                'truncation_fraction': self.config.truncation_fraction,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_fraction': self.config.elite_fraction,
                'reward_mutation_rate': self.config.reward_mutation_rate,
                'hyperparam_mutation_rate': self.config.hyperparam_mutation_rate,
                'architecture_mutation_rate': self.config.architecture_mutation_rate,
                'output_dir': self.config.output_dir,
                'checkpoint_interval': self.config.checkpoint_interval,
                'random_state': self.config.random_state
            },
            'population_history': self.population_history,
            'best_member': self.best_member.to_dict() if self.best_member else None,
            'generation_stats': self.generation_stats,
            'diversity_metrics': self.diversity_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'PBTResults':
        """
        Load PBT results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            PBTResults instance
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
        # Reconstruct config
        config = PBTConfig(
            population_size=results['config']['population_size'],
            generations=results['config']['generations'],
            eval_steps=results['config']['eval_steps'],
            truncation_fraction=results['config']['truncation_fraction'],
            mutation_rate=results['config']['mutation_rate'],
            crossover_rate=results['config']['crossover_rate'],
            elite_fraction=results['config']['elite_fraction'],
            reward_mutation_rate=results['config']['reward_mutation_rate'],
            hyperparam_mutation_rate=results['config']['hyperparam_mutation_rate'],
            architecture_mutation_rate=results['config']['architecture_mutation_rate'],
            output_dir=results['config']['output_dir'],
            checkpoint_interval=results['config']['checkpoint_interval'],
            random_state=results['config']['random_state']
        )
        
        # Create PBTResults instance
        pbt_results = cls(config)
        pbt_results.population_history = results['population_history']
        pbt_results.generation_stats = results['generation_stats']
        pbt_results.diversity_metrics = results['diversity_metrics']
        
        # Reconstruct best member
        if results['best_member']:
            best_dict = results['best_member']
            pbt_results.best_member = PBTMember(
                member_id=best_dict['member_id'],
                hyperparams=best_dict['hyperparams'],
                reward_params=best_dict['reward_params'],
                architecture_params=best_dict['architecture_params'],
                performance=best_dict['performance']
            )
            pbt_results.best_member.age = best_dict['age']
            pbt_results.best_member.generation = best_dict['generation']
            pbt_results.best_member.parent_id = best_dict['parent_id']
            
        return pbt_results


class PopulationBasedTraining:
    """
    Population-Based Training for multi-ticker RL trading.
    """
    
    def __init__(
        self,
        config: PBTConfig,
        eval_func: Callable,
        hyperparam_space: Dict[str, Any],
        reward_param_space: Dict[str, Any],
        architecture_space: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PBT.
        
        Args:
            config: PBT configuration
            eval_func: Evaluation function for members
            hyperparam_space: Hyperparameter search space
            reward_param_space: Reward parameter search space
            architecture_space: Architecture parameter search space
        """
        self.config = config
        self.eval_func = eval_func
        self.hyperparam_space = hyperparam_space
        self.reward_param_space = reward_param_space
        self.architecture_space = architecture_space or {}
        self.results = PBTResults(config)
        self.current_generation = 0
        self.population = []
        
    def initialize_population(self) -> None:
        """
        Initialize the population with random members.
        """
        logger.info(f"Initializing population with {self.config.population_size} members")
        
        for i in range(self.config.population_size):
            # Sample hyperparameters
            hyperparams = self._sample_params(self.hyperparam_space)
            
            # Sample reward parameters
            reward_params = self._sample_params(self.reward_param_space)
            
            # Sample architecture parameters
            architecture_params = self._sample_params(self.architecture_space)
            
            # Create member
            member = PBTMember(
                member_id=i,
                hyperparams=hyperparams,
                reward_params=reward_params,
                architecture_params=architecture_params
            )
            
            self.population.append(member)
            
    def _sample_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample parameters from parameter space.
        
        Args:
            param_space: Parameter space configuration
            
        Returns:
            Sampled parameters
        """
        params = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'float':
                if param_config.get('log', False):
                    # Log-uniform sampling
                    log_low = np.log(param_config['low'])
                    log_high = np.log(param_config['high'])
                    value = np.exp(np.random.uniform(log_low, log_high))
                else:
                    # Uniform sampling
                    value = np.random.uniform(param_config['low'], param_config['high'])
                params[param_name] = value
                
            elif param_config['type'] == 'int':
                if param_config.get('log', False):
                    # Log-uniform sampling for integers
                    log_low = np.log(param_config['low'])
                    log_high = np.log(param_config['high'])
                    value = int(np.exp(np.random.uniform(log_low, log_high)))
                else:
                    # Uniform sampling for integers
                    value = np.random.randint(param_config['low'], param_config['high'] + 1)
                params[param_name] = value
                
            elif param_config['type'] == 'categorical':
                # Categorical sampling
                value = np.random.choice(param_config['choices'])
                params[param_name] = value
                
        return params
        
    def evaluate_population(self) -> None:
        """
        Evaluate all members in the population.
        """
        logger.info(f"Evaluating generation {self.current_generation}")
        
        for member in self.population:
            # Evaluate member
            performance = self.eval_func(
                member.hyperparams,
                member.reward_params,
                member.architecture_params,
                self.config.eval_steps
            )
            
            # Update member performance
            member.update_performance(performance)
            member.age += 1
            
    def evolve_population(self) -> None:
        """
        Evolve the population using selection, crossover, and mutation.
        """
        logger.info(f"Evolving generation {self.current_generation}")
        
        # Sort population by fitness
        self.population.sort(key=lambda m: m.get_fitness(), reverse=True)
        
        # Calculate number of members to keep
        n_elite = int(self.config.population_size * self.config.elite_fraction)
        n_truncate = int(self.config.population_size * self.config.truncation_fraction)
        
        # Keep elite members
        new_population = self.population[:n_elite]
        
        # Generate new members through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Select parents from top performers
            parent1, parent2 = self._select_parents(self.population[:n_truncate])
            
            # Create offspring through crossover
            offspring = self._crossover(parent1, parent2)
            
            # Mutate offspring
            offspring = self._mutate(offspring)
            
            # Add to new population
            new_population.append(offspring)
            
        # Update population
        self.population = new_population
        
        # Update generation counter
        self.current_generation += 1
        
        # Update member generation
        for member in self.population:
            member.generation = self.current_generation
            
    def _select_parents(self, candidates: List[PBTMember]) -> Tuple[PBTMember, PBTMember]:
        """
        Select two parents from candidates using tournament selection.
        
        Args:
            candidates: List of candidate members
            
        Returns:
            Selected parents
        """
        # Tournament selection
        tournament_size = min(3, len(candidates))
        
        # Select first parent
        tournament1 = np.random.choice(candidates, tournament_size, replace=False)
        parent1 = max(tournament1, key=lambda m: m.get_fitness())
        
        # Select second parent
        tournament2 = np.random.choice(candidates, tournament_size, replace=False)
        parent2 = max(tournament2, key=lambda m: m.get_fitness())
        
        return parent1, parent2
        
    def _crossover(self, parent1: PBTMember, parent2: PBTMember) -> PBTMember:
        """
        Create offspring through crossover of two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring member
        """
        if np.random.random() > self.config.crossover_rate:
            # No crossover, return copy of random parent
            return np.random.choice([parent1, parent2]).copy()
            
        # Create new member ID
        new_id = max([m.member_id for m in self.population]) + 1
        
        # Hyperparameter crossover
        offspring_hyperparams = self._crossover_params(
            parent1.hyperparams, parent2.hyperparams, self.hyperparam_space
        )
        
        # Reward parameter crossover
        offspring_reward_params = self._crossover_params(
            parent1.reward_params, parent2.reward_params, self.reward_param_space
        )
        
        # Architecture parameter crossover
        offspring_architecture_params = self._crossover_params(
            parent1.architecture_params, parent2.architecture_params, self.architecture_space
        )
        
        # Create offspring
        offspring = PBTMember(
            member_id=new_id,
            hyperparams=offspring_hyperparams,
            reward_params=offspring_reward_params,
            architecture_params=offspring_architecture_params
        )
        
        # Set parent ID
        offspring.parent_id = np.random.choice([parent1.member_id, parent2.member_id])
        
        return offspring
        
    def _crossover_params(
        self, 
        params1: Dict[str, Any], 
        params2: Dict[str, Any], 
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform crossover on parameter dictionaries.
        
        Args:
            params1: First parameter dictionary
            params2: Second parameter dictionary
            param_space: Parameter space configuration
            
        Returns:
            Crossover result
        """
        offspring_params = {}
        
        for param_name in param_space.keys():
            if param_name in params1 and param_name in params2:
                param_config = param_space[param_name]
                
                if param_config['type'] in ['float', 'int']:
                    # Arithmetic crossover for numeric parameters
                    alpha = np.random.random()
                    if param_config['type'] == 'float':
                        value = alpha * params1[param_name] + (1 - alpha) * params2[param_name]
                    else:
                        value = int(alpha * params1[param_name] + (1 - alpha) * params2[param_name])
                    offspring_params[param_name] = value
                    
                elif param_config['type'] == 'categorical':
                    # Uniform crossover for categorical parameters
                    value = np.random.choice([params1[param_name], params2[param_name]])
                    offspring_params[param_name] = value
                    
            elif param_name in params1:
                offspring_params[param_name] = params1[param_name]
            elif param_name in params2:
                offspring_params[param_name] = params2[param_name]
                
        return offspring_params
        
    def _mutate(self, member: PBTMember) -> PBTMember:
        """
        Mutate a member.
        
        Args:
            member: Member to mutate
            
        Returns:
            Mutated member
        """
        # Create copy
        mutated = member.copy()
        
        # Mutate hyperparameters
        if np.random.random() < self.config.hyperparam_mutation_rate:
            mutated.hyperparams = self._mutate_params(
                mutated.hyperparams, self.hyperparam_space
            )
            
        # Mutate reward parameters
        if np.random.random() < self.config.reward_mutation_rate:
            mutated.reward_params = self._mutate_params(
                mutated.reward_params, self.reward_param_space
            )
            
        # Mutate architecture parameters
        if np.random.random() < self.config.architecture_mutation_rate:
            mutated.architecture_params = self._mutate_params(
                mutated.architecture_params, self.architecture_space
            )
            
        return mutated
        
    def _mutate_params(
        self, 
        params: Dict[str, Any], 
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Mutate parameter dictionary.
        
        Args:
            params: Parameter dictionary to mutate
            param_space: Parameter space configuration
            
        Returns:
            Mutated parameters
        """
        mutated_params = params.copy()
        
        for param_name, param_config in param_space.items():
            if np.random.random() < self.config.mutation_rate:
                if param_name in mutated_params:
                    if param_config['type'] == 'float':
                        # Gaussian mutation for float parameters
                        std = (param_config['high'] - param_config['low']) * 0.1
                        value = mutated_params[param_name] + np.random.normal(0, std)
                        # Clamp to bounds
                        value = np.clip(value, param_config['low'], param_config['high'])
                        mutated_params[param_name] = value
                        
                    elif param_config['type'] == 'int':
                        # Random reset mutation for integer parameters
                        if np.random.random() < 0.5:
                            # Small change
                            change = np.random.choice([-1, 1])
                            value = mutated_params[param_name] + change
                            # Clamp to bounds
                            value = np.clip(value, param_config['low'], param_config['high'])
                            mutated_params[param_name] = int(value)
                        else:
                            # Random resample
                            mutated_params[param_name] = np.random.randint(
                                param_config['low'], param_config['high'] + 1
                            )
                            
                    elif param_config['type'] == 'categorical':
                        # Random resample for categorical parameters
                        mutated_params[param_name] = np.random.choice(param_config['choices'])
                        
        return mutated_params
        
    def run(self) -> PBTResults:
        """
        Run the PBT algorithm.
        
        Returns:
            PBT results
        """
        logger.info("Starting Population-Based Training")
        
        # Initialize population
        self.initialize_population()
        
        # Run for specified generations
        for generation in range(self.config.generations):
            logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate population
            self.evaluate_population()
            
            # Store generation results
            self.results.add_generation(self.population, generation)
            
            # Save checkpoint if needed
            if (generation + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.output_dir, f"checkpoint_gen_{generation + 1}.pkl"
                )
                self.results.save(checkpoint_path)
                logger.info(f"Saved checkpoint at generation {generation + 1}")
                
            # Evolve population (except for last generation)
            if generation < self.config.generations - 1:
                self.evolve_population()
                
        # Final evaluation
        self.evaluate_population()
        self.results.add_generation(self.population, self.config.generations)
        
        # Save final results
        final_path = os.path.join(self.config.output_dir, "pbt_final_results.pkl")
        self.results.save(final_path)
        
        logger.info("Population-Based Training completed successfully")
        return self.results
        
    def calculate_diversity(self) -> Dict[str, float]:
        """
        Calculate population diversity metrics.
        
        Returns:
            Dictionary of diversity metrics
        """
        if not self.population:
            return {}
            
        # Calculate hyperparameter diversity
        hyperparam_diversity = self._calculate_param_diversity(
            [m.hyperparams for m in self.population], self.hyperparam_space
        )
        
        # Calculate reward parameter diversity
        reward_diversity = self._calculate_param_diversity(
            [m.reward_params for m in self.population], self.reward_param_space
        )
        
        # Calculate architecture diversity
        architecture_diversity = self._calculate_param_diversity(
            [m.architecture_params for m in self.population], self.architecture_space
        )
        
        # Calculate fitness diversity
        fitness_scores = [m.get_fitness() for m in self.population]
        fitness_diversity = np.std(fitness_scores)
        
        return {
            'hyperparam_diversity': hyperparam_diversity,
            'reward_diversity': reward_diversity,
            'architecture_diversity': architecture_diversity,
            'fitness_diversity': fitness_diversity
        }
        
    def _calculate_param_diversity(
        self, 
        param_list: List[Dict[str, Any]], 
        param_space: Dict[str, Any]
    ) -> float:
        """
        Calculate diversity for a list of parameter dictionaries.
        
        Args:
            param_list: List of parameter dictionaries
            param_space: Parameter space configuration
            
        Returns:
            Diversity score
        """
        if not param_list:
            return 0.0
            
        diversity_scores = []
        
        for param_name in param_space.keys():
            param_values = []
            for params in param_list:
                if param_name in params:
                    param_values.append(params[param_name])
                    
            if not param_values:
                continue
                
            param_config = param_space[param_name]
            
            if param_config['type'] in ['float', 'int']:
                # Normalized standard deviation for numeric parameters
                values = np.array(param_values)
                if values.std() > 0:
                    range_val = param_config['high'] - param_config['low']
                    if range_val > 0:
                        normalized_std = values.std() / range_val
                        diversity_scores.append(normalized_std)
                        
            elif param_config['type'] == 'categorical':
                # Entropy for categorical parameters
                unique_values, counts = np.unique(param_values, return_counts=True)
                probabilities = counts / len(param_values)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                max_entropy = np.log(len(param_config['choices']))
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                    diversity_scores.append(normalized_entropy)
                    
        return np.mean(diversity_scores) if diversity_scores else 0.0


def create_default_hyperparam_space() -> Dict[str, Any]:
    """
    Create default hyperparameter search space for PPO-LSTM.
    
    Returns:
        Dictionary with hyperparameter search space configuration
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


def create_default_reward_param_space() -> Dict[str, Any]:
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


def create_default_architecture_space() -> Dict[str, Any]:
    """
    Create default architecture parameter search space.
    
    Returns:
        Dictionary with architecture parameter search space configuration
    """
    return {
        'use_attention': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'attention_heads': {
            'type': 'int',
            'low': 1,
            'high': 8
        },
        'use_skip_connections': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'normalization_type': {
            'type': 'categorical',
            'choices': ['batch', 'layer', 'none']
        }
    }


def create_pbt_config_from_yaml(config_path: str) -> PBTConfig:
    """
    Create PBTConfig from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        PBTConfig instance
    """
    config = load_config(config_path)
    pbt_config_dict = config.get('pbt', {})
    
    return PBTConfig(
        population_size=pbt_config_dict.get('population_size', 10),
        generations=pbt_config_dict.get('generations', 20),
        eval_steps=pbt_config_dict.get('eval_steps', 10000),
        truncation_fraction=pbt_config_dict.get('truncation_fraction', 0.2),
        mutation_rate=pbt_config_dict.get('mutation_rate', 0.1),
        crossover_rate=pbt_config_dict.get('crossover_rate', 0.7),
        elite_fraction=pbt_config_dict.get('elite_fraction', 0.2),
        reward_mutation_rate=pbt_config_dict.get('reward_mutation_rate', 0.15),
        hyperparam_mutation_rate=pbt_config_dict.get('hyperparam_mutation_rate', 0.1),
        architecture_mutation_rate=pbt_config_dict.get('architecture_mutation_rate', 0.05),
        output_dir=pbt_config_dict.get('output_dir', 'pbt_results'),
        checkpoint_interval=pbt_config_dict.get('checkpoint_interval', 5),
        random_state=pbt_config_dict.get('random_state')
    )