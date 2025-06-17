#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Hyperparameter Search with HOLO-1.5 Recursive Symbolic Cognition Mesh
OPTIMIZER Role - Advanced hyperparameter optimization with neural-symbolic reasoning

This module provides comprehensive hyperparameter optimization capabilities enhanced with
HOLO-1.5 pattern for recursive symbolic cognition, adaptive optimization strategies,
and integration with the VantaCore cognitive mesh.
"""

import argparse
import json
import logging
import os
import itertools
import subprocess
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

# HOLO-1.5 Pattern Integration
HOLO15_AVAILABLE = False
try:
    from Vanta.core.VantaCore import BaseCore, VantaCore
    from Vanta.core.holo15_core import vanta_core_module, CognitiveRole
    from Vanta.core.recursive_symbolic_mesh import (
        SymbolicNode, RecursiveProcessor, CognitiveMesh
    )
    HOLO15_AVAILABLE = True
except ImportError:
    # Backward compatibility - create minimal fallback classes
    class BaseCore:
        def __init__(self, *args, **kwargs):
            pass
    
    def vanta_core_module(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveRole:
        OPTIMIZER = "OPTIMIZER"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyperparameter_search.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """Tracks the state of optimization process"""
    current_iteration: int = 0
    best_score: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    convergence_history: List[float] = field(default_factory=list)
    exploration_factor: float = 1.0
    adaptive_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class OptimizationObjective:
    """Defines optimization objectives and constraints"""
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["speed", "memory"])
    weights: Dict[str, float] = field(default_factory=lambda: {"accuracy": 1.0})
    constraints: Dict[str, Any] = field(default_factory=dict)
    optimization_direction: str = "maximize"  # "maximize" or "minimize"


@vanta_core_module(
    role=CognitiveRole.OPTIMIZER,
    capabilities=[
        "advanced_optimization",
        "multi_objective_optimization", 
        "adaptive_search_space",
        "optimization_caching",
        "bayesian_optimization",
        "evolutionary_optimization",
        "neural_symbolic_optimization"
    ]
)
class EnhancedHyperparameterSearch(BaseCore):
    """
    Enhanced Hyperparameter Search with HOLO-1.5 OPTIMIZER Role
    
    Provides advanced hyperparameter optimization with:
    - Multiple optimization algorithms (Grid, Random, Bayesian, Evolutionary)
    - Multi-objective optimization capabilities
    - Adaptive search space refinement
    - Optimization result caching and reuse
    - Neural-symbolic reasoning for optimization guidance
    - Integration with VantaCore cognitive mesh
    """
    
    def __init__(
        self, 
        model_type: str, 
        search_space: Optional[Dict[str, List[Any]]] = None,
        optimization_strategy: str = "adaptive_bayesian",
        enable_holo15: bool = True
    ):
        # Initialize BaseCore if HOLO-1.5 is available
        if HOLO15_AVAILABLE and enable_holo15:
            super().__init__(
                role=CognitiveRole.OPTIMIZER,
                cognitive_capabilities=[
                    "advanced_optimization", "multi_objective_optimization",
                    "adaptive_search_space", "neural_symbolic_optimization"
                ]
            )
        
        self.model_type = model_type
        self.search_space = search_space or {
            "epochs": [1, 2, 3],
            "batch_size": [2, 4, 8],
            "learning_rate": [1e-5, 2e-5, 5e-5],
        }
        self.optimization_strategy = optimization_strategy
        self.enable_holo15 = enable_holo15 and HOLO15_AVAILABLE
        
        # HOLO-1.5 Enhanced Features
        self.optimization_state = OptimizationState()
        self.optimization_history: List[Dict[str, Any]] = []
        self.optimization_cache: Dict[str, Any] = {}
        self.adaptive_search_enabled = True
        
        # Neural-Symbolic Optimization Components
        if self.enable_holo15:
            self.symbolic_optimizer = self._initialize_symbolic_optimizer()
            self.cognitive_mesh_optimizer = self._initialize_cognitive_mesh()
            
        # Multi-objective optimization setup
        self.objectives = OptimizationObjective()
        
        logger.info(f"Enhanced HyperparameterSearch initialized with HOLO-1.5: {self.enable_holo15}")

    def _initialize_symbolic_optimizer(self) -> Optional[Any]:
        """Initialize neural-symbolic optimization components"""
        if not self.enable_holo15:
            return None
            
        try:
            # Create symbolic nodes for optimization reasoning
            optimization_nodes = {
                "parameter_exploration": SymbolicNode(
                    "parameter_exploration",
                    reasoning_type="optimization_exploration"
                ),
                "convergence_analysis": SymbolicNode(
                    "convergence_analysis", 
                    reasoning_type="optimization_convergence"
                ),
                "strategy_adaptation": SymbolicNode(
                    "strategy_adaptation",
                    reasoning_type="optimization_adaptation"
                )
            }
            
            processor = RecursiveProcessor(
                nodes=optimization_nodes,
                max_recursion_depth=5
            )
            
            return processor
            
        except Exception as e:
            logger.warning(f"Failed to initialize symbolic optimizer: {e}")
            return None

    def _initialize_cognitive_mesh(self) -> Optional[Any]:
        """Initialize cognitive mesh for optimization coordination"""
        if not self.enable_holo15:
            return None
            
        try:
            mesh = CognitiveMesh(
                mesh_type="optimization_mesh",
                optimization_role=CognitiveRole.OPTIMIZER
            )
            return mesh
        except Exception as e:
            logger.warning(f"Failed to initialize cognitive mesh: {e}")
            return None

    def generate_hyperparameters_adaptive(
        self, 
        num_samples: int = None,
        exploration_factor: float = None
    ) -> List[Dict[str, Any]]:
        """
        Generate hyperparameters using adaptive strategy
        
        Args:
            num_samples: Number of parameter combinations to generate
            exploration_factor: Factor controlling exploration vs exploitation
            
        Returns:
            List of hyperparameter dictionaries
        """
        if exploration_factor is None:
            exploration_factor = self.optimization_state.exploration_factor
            
        if self.optimization_strategy == "grid":
            return self.generate_hyperparameters()
        elif self.optimization_strategy == "random":
            return self._generate_random_hyperparameters(num_samples or 50)
        elif self.optimization_strategy == "bayesian":
            return self._generate_bayesian_hyperparameters(num_samples or 20)
        elif self.optimization_strategy == "evolutionary":
            return self._generate_evolutionary_hyperparameters(num_samples or 30)
        elif self.optimization_strategy == "adaptive_bayesian":
            return self._generate_adaptive_bayesian_hyperparameters(
                num_samples or 25, exploration_factor
            )
        else:
            logger.warning(f"Unknown strategy {self.optimization_strategy}, falling back to grid")
            return self.generate_hyperparameters()

    def _generate_random_hyperparameters(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate random hyperparameter combinations"""
        params_list = []
        for _ in range(num_samples):
            params = {}
            for key, values in self.search_space.items():
                if isinstance(values[0], (int, float)):
                    # Numeric parameter - sample from range
                    min_val, max_val = min(values), max(values)
                    if isinstance(values[0], int):
                        params[key] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[key] = np.random.uniform(min_val, max_val)
                else:
                    # Categorical parameter
                    params[key] = np.random.choice(values)
            params_list.append(params)
        return params_list

    def _generate_bayesian_hyperparameters(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate hyperparameters using Bayesian optimization principles"""
        # Simplified Bayesian optimization - in production would use GPyOpt or similar
        if not self.optimization_history:
            # No history, start with random sampling
            return self._generate_random_hyperparameters(num_samples)
        
        # Analyze history to guide next samples
        best_params = self.optimization_state.best_params
        params_list = []
        
        # Generate samples around best known parameters
        for _ in range(num_samples):
            params = {}
            for key, values in self.search_space.items():
                if key in best_params:
                    best_val = best_params[key]
                    if isinstance(values[0], (int, float)):
                        # Add Gaussian noise around best value
                        std = (max(values) - min(values)) * 0.1
                        if isinstance(values[0], int):
                            params[key] = int(np.clip(
                                np.random.normal(best_val, std),
                                min(values), max(values)
                            ))
                        else:
                            params[key] = np.clip(
                                np.random.normal(best_val, std),
                                min(values), max(values)
                            )
                    else:
                        # For categorical, occasionally explore
                        if np.random.random() < 0.3:  # 30% exploration
                            params[key] = np.random.choice(values)
                        else:
                            params[key] = best_val
                else:
                    # No best value known, sample randomly
                    if isinstance(values[0], (int, float)):
                        min_val, max_val = min(values), max(values)
                        if isinstance(values[0], int):
                            params[key] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[key] = np.random.uniform(min_val, max_val)
                    else:
                        params[key] = np.random.choice(values)
            params_list.append(params)
        return params_list

    def _generate_evolutionary_hyperparameters(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate hyperparameters using evolutionary algorithms"""
        if not self.optimization_history:
            return self._generate_random_hyperparameters(num_samples)
        
        # Select top performers as parents
        sorted_history = sorted(
            self.optimization_history,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        top_performers = sorted_history[:min(5, len(sorted_history))]
        params_list = []
        
        for _ in range(num_samples):
            if len(top_performers) >= 2:
                # Crossover between two parents
                parent1 = np.random.choice(top_performers)['params']
                parent2 = np.random.choice(top_performers)['params']
                child = self._crossover_params(parent1, parent2)
                child = self._mutate_params(child, mutation_rate=0.1)
            else:
                # Single parent mutation
                parent = top_performers[0]['params']
                child = self._mutate_params(parent, mutation_rate=0.2)
            
            params_list.append(child)
        
        return params_list

    def _generate_adaptive_bayesian_hyperparameters(
        self, 
        num_samples: int, 
        exploration_factor: float
    ) -> List[Dict[str, Any]]:
        """
        Advanced adaptive Bayesian optimization with HOLO-1.5 neural-symbolic guidance
        """
        if self.enable_holo15 and self.symbolic_optimizer:
            # Use neural-symbolic reasoning to guide parameter generation
            symbolic_guidance = self._get_symbolic_optimization_guidance()
            if symbolic_guidance:
                return self._generate_symbolically_guided_params(
                    num_samples, exploration_factor, symbolic_guidance
                )
        
        # Fallback to standard adaptive Bayesian
        return self._generate_bayesian_hyperparameters(num_samples)

    def _get_symbolic_optimization_guidance(self) -> Optional[Dict[str, Any]]:
        """Get guidance from neural-symbolic reasoning"""
        if not self.enable_holo15 or not self.symbolic_optimizer:
            return None
            
        try:
            # Prepare optimization context
            context = {
                "optimization_history": self.optimization_history[-10:],  # Recent history
                "current_best": self.optimization_state.best_params,
                "convergence_trend": self.optimization_state.convergence_history[-5:],
                "exploration_factor": self.optimization_state.exploration_factor
            }
            
            # Process through symbolic reasoning
            guidance = self.symbolic_optimizer.process_recursive(
                "optimization_strategy", context
            )
            
            return guidance
            
        except Exception as e:
            logger.warning(f"Symbolic optimization guidance failed: {e}")
            return None

    def _generate_symbolically_guided_params(
        self,
        num_samples: int,
        exploration_factor: float,
        guidance: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate parameters guided by symbolic reasoning"""
        params_list = []
        
        # Extract symbolic guidance
        focus_params = guidance.get("focus_parameters", [])
        exploration_strategy = guidance.get("exploration_strategy", "balanced")
        adaptive_ranges = guidance.get("adaptive_ranges", {})
        
        for _ in range(num_samples):
            params = {}
            for key, values in self.search_space.items():
                if key in focus_params:
                    # Focus on symbolically identified important parameters
                    if key in adaptive_ranges:
                        min_val, max_val = adaptive_ranges[key]
                    else:
                        min_val, max_val = min(values), max(values)
                    
                    if isinstance(values[0], (int, float)):
                        if isinstance(values[0], int):
                            params[key] = np.random.randint(int(min_val), int(max_val) + 1)
                        else:
                            params[key] = np.random.uniform(min_val, max_val)
                    else:
                        params[key] = np.random.choice(values)
                else:
                    # Standard sampling for other parameters
                    if isinstance(values[0], (int, float)):
                        min_val, max_val = min(values), max(values)
                        if isinstance(values[0], int):
                            params[key] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[key] = np.random.uniform(min_val, max_val)
                    else:
                        params[key] = np.random.choice(values)
            
            params_list.append(params)
        
        return params_list

    def _crossover_params(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover between two parameter sets"""
        child = {}
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate_params(self, params: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """Mutate parameters"""
        mutated = params.copy()
        for key, value in params.items():
            if np.random.random() < mutation_rate:
                values = self.search_space[key]
                if isinstance(values[0], (int, float)):
                    # Numeric mutation
                    std = (max(values) - min(values)) * 0.1
                    if isinstance(values[0], int):
                        mutated[key] = int(np.clip(
                            np.random.normal(value, std),
                            min(values), max(values)
                        ))
                    else:
                        mutated[key] = np.clip(
                            np.random.normal(value, std),
                            min(values), max(values)
                        )
                else:
                    # Categorical mutation
                    mutated[key] = np.random.choice(values)
        return mutated

    def run_experiment_enhanced(
        self, 
        hyperparams: Dict[str, Any], 
        experiment_id: str,
        objectives: Optional[OptimizationObjective] = None
    ) -> Dict[str, Any]:
        """
        Enhanced experiment runner with multi-objective evaluation
        """
        # Use provided objectives or default
        obj = objectives or self.objectives
        
        # Check cache first
        cache_key = self._get_cache_key(hyperparams)
        if cache_key in self.optimization_cache:
            logger.info(f"Using cached result for experiment {experiment_id}")
            cached_result = self.optimization_cache[cache_key].copy()
            cached_result["experiment_id"] = experiment_id
            cached_result["from_cache"] = True
            return cached_result
        
        # Run experiment (original logic)
        result = self.run_experiment(hyperparams, experiment_id)
        
        # Enhanced multi-objective evaluation
        if result["success"]:
            result = self._evaluate_multi_objectives(result, obj)
        
        # Cache result
        self.optimization_cache[cache_key] = result.copy()
        
        # Update optimization state
        self._update_optimization_state(result, hyperparams)
        
        return result

    def _evaluate_multi_objectives(
        self, 
        result: Dict[str, Any], 
        objectives: OptimizationObjective
    ) -> Dict[str, Any]:
        """Evaluate multiple optimization objectives"""
        try:
            # Primary objective (e.g., accuracy) should be available from evaluation
            primary_score = result.get("evaluation", {}).get(objectives.primary_metric, 0)
            
            # Calculate secondary objectives
            secondary_scores = {}
            
            # Speed objective (training time)
            if "speed" in objectives.secondary_metrics:
                duration = result.get("duration_seconds", float('inf'))
                # Normalize speed score (faster is better)
                speed_score = 1.0 / (1.0 + duration / 3600)  # Normalize by hour
                secondary_scores["speed"] = speed_score
            
            # Memory objective (could be estimated from model size)
            if "memory" in objectives.secondary_metrics:
                # Simplified memory estimation based on parameters
                hyperparams = result.get("hyperparams", {})
                memory_penalty = 0
                if "batch_size" in hyperparams:
                    memory_penalty += hyperparams["batch_size"] * 0.01
                memory_score = 1.0 / (1.0 + memory_penalty)
                secondary_scores["memory"] = memory_score
            
            # Calculate composite score
            weights = objectives.weights
            composite_score = primary_score * weights.get(objectives.primary_metric, 1.0)
            
            for metric, score in secondary_scores.items():
                weight = weights.get(metric, 0.1)  # Default small weight
                composite_score += score * weight
            
            # Add to result
            result["multi_objective_evaluation"] = {
                "primary_score": primary_score,
                "secondary_scores": secondary_scores,
                "composite_score": composite_score,
                "objectives_used": objectives.__dict__
            }
            
            result["composite_score"] = composite_score
            
        except Exception as e:
            logger.warning(f"Multi-objective evaluation failed: {e}")
            # Fallback to single objective
            result["composite_score"] = result.get("evaluation", {}).get("accuracy", 0)
        
        return result

    def _get_cache_key(self, hyperparams: Dict[str, Any]) -> str:
        """Generate cache key for hyperparameters"""
        sorted_params = sorted(hyperparams.items())
        return json.dumps(sorted_params, sort_keys=True)

    def _update_optimization_state(self, result: Dict[str, Any], hyperparams: Dict[str, Any]):
        """Update optimization state based on result"""
        score = result.get("composite_score", 0)
        
        # Update optimization history
        self.optimization_history.append({
            "params": hyperparams.copy(),
            "score": score,
            "timestamp": time.time(),
            "experiment_id": result.get("experiment_id")
        })
        
        # Update best if better
        if score > self.optimization_state.best_score:
            self.optimization_state.best_score = score
            self.optimization_state.best_params = hyperparams.copy()
        
        # Update convergence history
        self.optimization_state.convergence_history.append(score)
        if len(self.optimization_state.convergence_history) > 20:
            self.optimization_state.convergence_history = self.optimization_state.convergence_history[-20:]
        
        # Adapt exploration factor
        self._adapt_exploration_factor()

    def _adapt_exploration_factor(self):
        """Adapt exploration factor based on convergence"""
        if len(self.optimization_state.convergence_history) < 5:
            return
        
        recent_scores = self.optimization_state.convergence_history[-5:]
        score_variance = np.var(recent_scores)
        
        # If variance is low, we might be converging - increase exploration
        if score_variance < 0.01:
            self.optimization_state.exploration_factor = min(1.5, 
                self.optimization_state.exploration_factor * 1.1)
        # If variance is high, we're exploring well - can reduce exploration
        elif score_variance > 0.1:
            self.optimization_state.exploration_factor = max(0.5,
                self.optimization_state.exploration_factor * 0.9)

    def run_search_enhanced(
        self, 
        num_experiments: Optional[int] = None,
        optimization_objectives: Optional[OptimizationObjective] = None,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Enhanced search with HOLO-1.5 optimization capabilities
        """
        if optimization_objectives:
            self.objectives = optimization_objectives
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_id = f"{self.model_type}_{self.optimization_strategy}_{timestamp}"
        results_dir = f"hp_search_results/{search_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate hyperparameters using enhanced strategy
        hyperparams_list = self.generate_hyperparameters_adaptive(num_experiments)
        
        logger.info(
            f"Running enhanced hyperparameter search for {self.model_type} "
            f"with {len(hyperparams_list)} combinations using {self.optimization_strategy} strategy"
        )
        
        experiments = []
        best_score = 0
        best_experiment = None
        no_improvement_count = 0
        
        for i, hyperparams in enumerate(hyperparams_list):
            experiment_id = f"{i + 1:02d}_{timestamp}"
            
            # Run enhanced experiment
            experiment_details = self.run_experiment_enhanced(
                hyperparams, experiment_id, self.objectives
            )
            
            # Evaluate with enhanced evaluation
            eval_results = self.evaluate_experiment(experiment_details)
            
            # Use composite score for comparison
            current_score = experiment_details.get("composite_score", 0)
            if current_score > best_score:
                best_score = current_score
                best_experiment = experiment_details
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            experiments.append(experiment_details)
            
            # Early stopping check
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping after {len(experiments)} experiments "
                           f"(no improvement for {early_stopping_patience} iterations)")
                break
            
            # Adaptive strategy updates
            if self.enable_holo15 and self.adaptive_search_enabled:
                self._update_adaptive_strategy(i, len(hyperparams_list))
        
        # Prepare enhanced results
        results = {
            "search_id": search_id,
            "model_type": self.model_type,
            "optimization_strategy": self.optimization_strategy,
            "num_experiments": len(experiments),
            "best_experiment": best_experiment["experiment_id"] if best_experiment else None,
            "best_composite_score": best_score,
            "optimization_objectives": self.objectives.__dict__,
            "experiments": experiments,
            "optimization_state": {
                "final_exploration_factor": self.optimization_state.exploration_factor,
                "convergence_history": self.optimization_state.convergence_history,
                "cache_hits": sum(1 for exp in experiments if exp.get("from_cache", False))
            }
        }
        
        # HOLO-1.5 specific results
        if self.enable_holo15:
            results["holo15_analysis"] = self._generate_holo15_analysis()
        
        # Save results
        results_file = f"{results_dir}/search_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Enhanced hyperparameter search completed. "
                   f"Best composite score: {best_score:.4f}")
        logger.info(f"Results saved to {results_file}")
        
        if best_experiment:
            logger.info(f"Best hyperparameters: {best_experiment['hyperparams']}")
        
        return results

    def _update_adaptive_strategy(self, current_iteration: int, total_iterations: int):
        """Update adaptive strategy during search"""
        progress = current_iteration / total_iterations
        
        # Adaptive strategy switching
        if progress < 0.3 and self.optimization_strategy == "adaptive_bayesian":
            # Early phase - encourage exploration
            self.optimization_state.exploration_factor = max(
                self.optimization_state.exploration_factor, 1.2
            )
        elif progress > 0.7:
            # Late phase - encourage exploitation
            self.optimization_state.exploration_factor = min(
                self.optimization_state.exploration_factor, 0.8
            )

    def _generate_holo15_analysis(self) -> Dict[str, Any]:
        """Generate HOLO-1.5 specific analysis"""
        analysis = {
            "symbolic_optimization_used": self.symbolic_optimizer is not None,
            "cognitive_mesh_integration": self.cognitive_mesh_optimizer is not None,
            "adaptive_features_active": self.adaptive_search_enabled,
            "optimization_insights": []
        }
        
        if self.optimization_history:
            # Parameter importance analysis
            param_importance = self._analyze_parameter_importance()
            analysis["parameter_importance"] = param_importance
            
            # Convergence analysis
            convergence_analysis = self._analyze_convergence()
            analysis["convergence_analysis"] = convergence_analysis
        
        return analysis

    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze which parameters have the most impact on performance"""
        if len(self.optimization_history) < 5:
            return {}
        
        importance = {}
        for param_name in self.search_space.keys():
            # Calculate correlation between parameter value and score
            values = []
            scores = []
            for entry in self.optimization_history:
                if param_name in entry["params"]:
                    values.append(entry["params"][param_name])
                    scores.append(entry["score"])
            
            if len(values) > 3:
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0
        
        return importance

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence characteristics"""
        convergence_data = {
            "is_converged": False,
            "convergence_rate": 0.0,
            "plateau_detected": False
        }
        
        if len(self.optimization_state.convergence_history) >= 10:
            recent_scores = self.optimization_state.convergence_history[-10:]
            score_variance = np.var(recent_scores)
            score_trend = np.mean(np.diff(recent_scores))
            
            convergence_data["is_converged"] = score_variance < 0.001
            convergence_data["convergence_rate"] = float(score_trend)
            convergence_data["plateau_detected"] = score_variance < 0.01 and abs(score_trend) < 0.001
        
        return convergence_data

    # Backward compatibility methods
    def generate_hyperparameters(self) -> List[Dict[str, Any]]:
        """Original grid search generation for backward compatibility"""
        keys = self.search_space.keys()
        values = self.search_space.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def run_experiment(self, hyperparams: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Original experiment runner for backward compatibility"""
        output_dir = f"models/{self.model_type}_hp_search_{experiment_id}"
        os.makedirs(output_dir, exist_ok=True)

        if self.model_type == "phi-2":
            script = "phi2_finetune.py"
            dataset = "voxsigil_finetune/data/phi-2/arc_training_phi-2.jsonl"
        elif self.model_type == "mistral-7b":
            script = "mistral_finetune.py"
            dataset = "voxsigil_finetune/data/mistral-7b/arc_training_mistral-7b.jsonl"
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        cmd = [
            "python", script,
            f"--dataset={dataset}",
            f"--output_dir={output_dir}",
            f"--epochs={hyperparams['epochs']}",
            f"--batch_size={hyperparams['batch_size']}",
            f"--learning_rate={hyperparams['learning_rate']}",
        ]

        logger.info(f"Running experiment {experiment_id} with hyperparams: {hyperparams}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            output = result.stdout

            experiment_details = {
                "experiment_id": experiment_id,
                "model_type": self.model_type,
                "hyperparams": hyperparams,
                "output_dir": output_dir,
                "success": True,
                "duration_seconds": end_time - start_time,
                "stdout": output,
                "stderr": result.stderr,
            }

            details_file = f"{output_dir}/experiment_details.json"
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(experiment_details, f, indent=2)

            logger.info(f"Experiment {experiment_id} completed successfully in {end_time - start_time:.2f} seconds")
            return experiment_details

        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")

            experiment_details = {
                "experiment_id": experiment_id,
                "model_type": self.model_type,
                "hyperparams": hyperparams,
                "output_dir": output_dir,
                "success": False,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
            }

            details_file = f"{output_dir}/experiment_details.json"
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(experiment_details, f, indent=2)

            return experiment_details

    def evaluate_experiment(self, experiment_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Original evaluation method for backward compatibility"""
        if not experiment_details["success"]:
            logger.warning(f"Skipping evaluation for failed experiment {experiment_details['experiment_id']}")
            return None

        output_dir = experiment_details["output_dir"]

        if self.model_type == "phi-2":
            eval_dataset = "voxsigil_finetune/data/phi-2/arc_evaluation_phi-2.jsonl"
        elif self.model_type == "mistral-7b":
            eval_dataset = "voxsigil_finetune/data/mistral-7b/arc_evaluation_mistral-7b.jsonl"
        else:
            eval_dataset = "voxsigil_finetune/data/arc_evaluation_dataset_fixed.jsonl"

        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        results_file = f"{results_dir}/{self.model_type}_exp_{experiment_details['experiment_id']}_results.json"

        cmd = [
            "python", "evaluate_model.py",
            f"--model_path={output_dir}",
            f"--dataset={eval_dataset}",
            f"--output={results_file}",
            "--num_samples=50",
        ]

        logger.info(f"Evaluating experiment {experiment_details['experiment_id']}...")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            with open(results_file, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

            experiment_details["evaluation"] = {
                "results_file": results_file,
                "accuracy": eval_results.get("accuracy", 0),
            }

            details_file = f"{output_dir}/experiment_details.json"
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(experiment_details, f, indent=2)

            logger.info(f"Evaluation for experiment {experiment_details['experiment_id']} completed with accuracy: {eval_results.get('accuracy', 0):.2%}")
            return eval_results

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Evaluation failed for experiment {experiment_details['experiment_id']}: {e}")
            return None

    def run_search(self, num_experiments: Optional[int] = None) -> Dict[str, Any]:
        """Original search method for backward compatibility"""
        if self.enable_holo15:
            return self.run_search_enhanced(num_experiments)
        
        hyperparams_list = self.generate_hyperparameters()
        if num_experiments is not None:
            hyperparams_list = hyperparams_list[:num_experiments]

        logger.info(f"Running hyperparameter search for {self.model_type} with {len(hyperparams_list)} combinations")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_id = f"{self.model_type}_{timestamp}"
        results_dir = f"hp_search_results/{search_id}"
        os.makedirs(results_dir, exist_ok=True)

        experiments = []
        best_accuracy = 0
        best_experiment = None

        for i, hyperparams in enumerate(hyperparams_list):
            experiment_id = f"{i + 1:02d}_{timestamp}"
            experiment_details = self.run_experiment(hyperparams, experiment_id)
            eval_results = self.evaluate_experiment(experiment_details)
            if eval_results and eval_results.get("accuracy", 0) > best_accuracy:
                best_accuracy = eval_results.get("accuracy", 0)
                best_experiment = experiment_details
            experiments.append(experiment_details)

        results = {
            "search_id": search_id,
            "model_type": self.model_type,
            "num_experiments": len(experiments),
            "best_experiment": best_experiment["experiment_id"] if best_experiment else None,
            "best_accuracy": best_accuracy,
            "experiments": experiments,
        }

        results_file = f"{results_dir}/search_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Hyperparameter search completed. Best accuracy: {best_accuracy:.2%}")
        logger.info(f"Results saved to {results_file}")

        if best_experiment:
            logger.info(f"Best hyperparameters: {best_experiment['hyperparams']}")

        return results

    def get_best_hyperparameters(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters from results"""
        best_exp_id = results.get("best_experiment")
        if not best_exp_id:
            return None
        for exp in results.get("experiments", []):
            if exp["experiment_id"] == best_exp_id:
                return exp.get("hyperparams")
        return None

    @classmethod
    def from_config(cls, config_path: str) -> "EnhancedHyperparameterSearch":
        """Create instance from config file"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            model_type=config["model_type"],
            search_space=config.get("search_space"),
            optimization_strategy=config.get("optimization_strategy", "adaptive_bayesian"),
            enable_holo15=config.get("enable_holo15", True)
        )

    @property
    def search_space_keys(self) -> List[str]:
        return list(self.search_space.keys())

    @property
    def search_space_values(self) -> List[List[Any]]:
        return list(self.search_space.values())


# Backward compatibility alias
HyperparameterSearch = EnhancedHyperparameterSearch


def main():
    """Main function with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Run enhanced hyperparameter search for ARC fine-tuning with HOLO-1.5"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["phi-2", "mistral-7b"],
        required=True,
        help="Model type to fine-tune",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=None,
        help="Number of experiments to run (default: strategy dependent)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file for search space and optimization settings",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random", "bayesian", "evolutionary", "adaptive_bayesian"],
        default="adaptive_bayesian",
        help="Optimization strategy to use",
    )
    parser.add_argument(
        "--disable-holo15",
        action="store_true",
        help="Disable HOLO-1.5 features for compatibility",
    )
    
    args = parser.parse_args()

    if args.config:
        searcher = EnhancedHyperparameterSearch.from_config(args.config)
    else:
        searcher = EnhancedHyperparameterSearch(
            model_type=args.model,
            optimization_strategy=args.strategy,
            enable_holo15=not args.disable_holo15
        )
    
    # Run enhanced search
    results = searcher.run_search_enhanced(args.experiments)
    
    print(f"\n=== Enhanced Hyperparameter Search Complete ===")
    print(f"Strategy: {args.strategy}")
    print(f"HOLO-1.5 Enabled: {not args.disable_holo15}")
    print(f"Best Score: {results.get('best_composite_score', 0):.4f}")
    print(f"Experiments Run: {results.get('num_experiments', 0)}")


if __name__ == "__main__":
    main()
