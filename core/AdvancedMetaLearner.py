"""
Advanced Meta-Learning Module

This module provides meta-learning capabilities for adapting to new tasks
and improving learning efficiency across domains.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Import Vanta Core for integration
from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

# Import MetaLearnerInterface from unified Vanta interfaces
from Vanta.interfaces.specialized_interfaces import MetaLearnerInterface

# HOLO-1.5 Recursive Symbolic Cognition Mesh imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole
from Vanta.core.UnifiedVantaCore import VantaCore

logger = logging.getLogger("metaconsciousness.meta_learner")


# Helper function to get UnifiedVantaCore instance
def get_unified_core():
    """Get the UnifiedVantaCore singleton instance."""
    try:
        return UnifiedVantaCore()
    except Exception as e:
        logger.error(f"Error getting UnifiedVantaCore: {e}")
        return None


class PerformanceTracker:
    """
    Tracks performance metrics across different domains.
    """
    
    def __init__(self, max_history: int = 100):
        self.domain_performances = {}
        self.overall_performances = []
        self.max_history = max_history
        self.last_update_time = time.time()
    
    def update_domain_performance(self, domain: str, performance: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Update performance metrics for a specific domain.
        
        Args:
            domain: The domain name
            performance: Performance value (0-1)
            metadata: Additional metadata about the performance
        """
        if domain not in self.domain_performances:
            self.domain_performances[domain] = []
            
        entry = {
            "value": performance,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.domain_performances[domain].append(entry)
        
        # Trim history if needed
        if len(self.domain_performances[domain]) > self.max_history:
            self.domain_performances[domain] = self.domain_performances[domain][-self.max_history:]
            
        # Update overall performance record
        self.overall_performances.append({
            "domain": domain,
            "value": performance,
            "timestamp": time.time()
        })
        
        # Trim overall history
        if len(self.overall_performances) > self.max_history * 3:  # Allow more entries in overall
            self.overall_performances = self.overall_performances[-self.max_history * 3:]
            
        self.last_update_time = time.time()
    
    def get_domain_performance(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific domain or all domains.
        
        Args:
            domain: Optional domain name to get metrics for
            
        Returns:
            Dictionary with performance metrics
        """
        if domain:
            if domain not in self.domain_performances:
                return {"domain": domain, "available": False}
                
            performances = self.domain_performances[domain]
            if not performances:
                return {"domain": domain, "available": False}
                
            values = [p["value"] for p in performances]
            return {
                "domain": domain,
                "available": True,
                "current": values[-1],
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "trend": self._calculate_trend(values),
                "last_update": self.domain_performances[domain][-1]["timestamp"]
            }
        else:
            # Get metrics across all domains
            domain_summaries = {}
            for d in self.domain_performances:
                domain_summaries[d] = self.get_domain_performance(d)
                
            # Calculate overall stats
            all_values = [p["value"] for p in self.overall_performances]
            if not all_values:
                return {"domains": len(self.domain_performances), "available": False}
                
            return {
                "domains": len(self.domain_performances),
                "available": True,
                "domain_summaries": domain_summaries,
                "overall_average": sum(all_values) / len(all_values),
                "overall_min": min(all_values),
                "overall_max": max(all_values),
                "overall_count": len(all_values),
                "overall_trend": self._calculate_trend(all_values),
                "last_update": self.last_update_time
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a series of values."""
        if len(values) < 3:
            return "insufficient_data"
            
        # Use last 5 values or all if less than 5
        recent = values[-min(5, len(values)):]
        
        # Simple linear regression slope
        x = list(range(len(recent)))
        mean_x = sum(x) / len(x)
        mean_y = sum(recent) / len(recent)
        
        numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, recent))
        denominator = sum((x_i - mean_x) ** 2 for x_i in x)
        
        if denominator == 0:
            return "stable"
            
        slope = numerator / denominator
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"


class CrossDomainKnowledgeTransfer:
    """
    Handles knowledge transfer between different domains.
    """
    
    def __init__(self):
        self.domain_knowledge = {}
        self.transfer_history = []
        self.knowledge_similarity_cache = {}
    
    def store_domain_knowledge(self, domain: str, knowledge: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Store knowledge from a specific domain.
        
        Args:
            domain: The domain name
            knowledge: The knowledge to store
            metadata: Additional metadata about the knowledge
        """
        self.domain_knowledge[domain] = {
            "knowledge": knowledge,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Clear similarity cache involving this domain
        keys_to_remove = []
        for key in self.knowledge_similarity_cache:
            if domain in key:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.knowledge_similarity_cache[key]
    
    def get_domain_knowledge(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge for a specific domain.
        
        Args:
            domain: The domain name
            
        Returns:
            Domain knowledge if available
        """
        return self.domain_knowledge.get(domain)
    
    def transfer_knowledge(self, source_domain: str, target_domain: str, transfer_strength: float = 0.5) -> Dict[str, Any]:
        """
        Transfer knowledge from source domain to target domain.
        
        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            transfer_strength: How strongly to apply the transfer (0-1)
            
        Returns:
            Dictionary with transfer results
        """
        if source_domain not in self.domain_knowledge:
            return {"success": False, "error": f"Source domain {source_domain} not found"}
            
        if target_domain not in self.domain_knowledge:
            return {"success": False, "error": f"Target domain {target_domain} not found"}
        
        # Calculate similarity
        similarity = self._calculate_knowledge_similarity(source_domain, target_domain)
        
        # Record transfer
        transfer_record = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "similarity": similarity,
            "transfer_strength": transfer_strength,
            "timestamp": time.time()
        }
        
        self.transfer_history.append(transfer_record)
        
        return {
            "success": True,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "similarity": similarity,
            "transfer_strength": transfer_strength
        }
    
    def _calculate_knowledge_similarity(self, domain1: str, domain2: str) -> float:
        """
        Calculate similarity between knowledge from two domains.
        
        Args:
            domain1: First domain name
            domain2: Second domain name
            
        Returns:
            Similarity score (0-1)
        """
        # Check cache first
        cache_key = f"{domain1}:{domain2}"
        if cache_key in self.knowledge_similarity_cache:
            return self.knowledge_similarity_cache[cache_key]
            
        # Simple similarity calculation
        # In a real implementation, this would be more sophisticated
        if domain1 not in self.domain_knowledge or domain2 not in self.domain_knowledge:
            return 0.0
            
        # Calculate similarity based on metadata overlap
        meta1 = self.domain_knowledge[domain1].get("metadata", {})
        meta2 = self.domain_knowledge[domain2].get("metadata", {})
        
        # Get common keys
        common_keys = set(meta1.keys()) & set(meta2.keys())
        if not common_keys:
            similarity = 0.2  # Base similarity
        else:
            similarities = []
            for key in common_keys:
                if meta1[key] == meta2[key]:
                    similarities.append(1.0)
                else:
                    similarities.append(0.3)
                    
            similarity = sum(similarities) / len(similarities)
            
        # Store in cache
        self.knowledge_similarity_cache[cache_key] = similarity
        self.knowledge_similarity_cache[f"{domain2}:{domain1}"] = similarity  # Symmetric
        
        return similarity
    
    def get_transfer_history(self) -> List[Dict[str, Any]]:
        """Get knowledge transfer history."""
        return self.transfer_history
    
    def get_all_domain_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored domain knowledge."""
        return self.domain_knowledge


@vanta_core_module(
    name="advanced_meta_learner",
    subsystem="learning_management",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="Advanced meta-learning system for cross-domain knowledge transfer with HOLO-1.5 integration and API-driven communication enhancement",
    capabilities=[
        "cross_domain_transfer", "parameter_optimization", "performance_tracking", 
        "meta_learning", "task_adaptation", "knowledge_integration",
        "communication_enhancement", "tts_learning", "stt_learning", 
        "api_driven_learning", "speech_optimization"
    ],
    cognitive_load=4.0,
    symbolic_depth=4,
    collaboration_patterns=[
        "meta_learning", "knowledge_transfer", "cross_domain_learning", 
        "performance_optimization", "communication_adaptation", "api_integration"
    ]
)
class AdvancedMetaLearner(BaseCore, MetaLearnerInterface):
    """
    Advanced meta-learning system for cross-domain knowledge transfer with API-driven communication enhancement.

    This system:
    - Identifies common patterns across learning tasks
    - Transfers knowledge between domains
    - Adapts learning strategies based on task characteristics
    - Optimizes learning parameters for efficiency
    - Tracks performance metrics across domains
    - Learns from TTS/STT API interactions to improve communication
    - Provides VantaCore with enhanced communication capabilities through submodel learning
    """

    def __init__(self, vanta_core: VantaCore, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the advanced meta-learner with communication enhancement capabilities.

        Args:
            vanta_core: VantaCore instance for HOLO-1.5 integration
            config: Configuration dictionary
        """
        # Initialize BaseCore with HOLO-1.5 mesh capabilities
        super().__init__(vanta_core, config or {})
        
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.learning_tasks = {}
        self.meta_parameters = {
            "learning_rate": self.config.get("learning_rate", 0.1),
            "exploration_rate": self.config.get("exploration_rate", 0.2),
            "transfer_strength": self.config.get("transfer_strength", 0.5),
        }
        # Performance tracking
        self.task_performances = {}
        self.transfer_history = []
        self.max_history = self.config.get("max_history", 100)
        self.connected_to = []
        
        # Initialize performance tracker and knowledge transfer modules
        self.performance_tracker = PerformanceTracker(max_history=self.max_history)
        self.knowledge_transfer = CrossDomainKnowledgeTransfer()
        
        # Initialize communication enhancement learning system
        self.communication_learner = CommunicationEnhancementLearner(
            max_history=self.config.get("communication_history", 1000)
        )
        
        # TTS/STT engine references for API-driven learning
        self.tts_engine = None
        self.stt_engine = None
        self.communication_enhancement_active = self.config.get("communication_enhancement", True)
        
        # Connect to other systems
        self._connect_components()
        
        # Log initialization
        logger.info(
            f"Advanced meta-learner initialized: enabled={self.enabled}, "
            f"communication_enhancement={self.communication_enhancement_active}, "
            f"params={self.meta_parameters}"
        )

    async def initialize(self) -> bool:
        """Initialize the AdvancedMetaLearner for BaseCore compliance."""
        try:
            if self.enabled:
                logger.info("AdvancedMetaLearner initialized successfully with HOLO-1.5 enhancement")
            else:
                logger.info("AdvancedMetaLearner disabled by configuration")
            return True
        except Exception as e:
            logger.error(f"Error initializing AdvancedMetaLearner: {e}")
            return False
    def _connect_components(self) -> None:
        """Connect to other Vanta components."""
        # Get Vanta Core instance
        self.vanta_core = get_unified_core()

        # Connect to memory systems
        self.memory = None
        self.pattern_memory = None
        self.art_controller = None
        self.meta_cognitive = None

        # Register with Vanta Core
        if self.vanta_core:
            try:
                # Register this component with Vanta Core
                self.vanta_core.register_component(
                    "advanced_meta_learner",
                    self,
                    {
                        "type": "meta_learning_system",
                        "capabilities": ["cross_domain_transfer", "parameter_optimization"],
                    },
                )
                logger.info("Advanced meta-learner registered with Vanta Core")                    # Try to get required components
                try:
                        components = self.vanta_core.list_components()

                        # Look for memory components
                        if "memory" in components:
                            self.memory = self.vanta_core.get_component("memory")
                            self.connected_to.append("memory")

                        if "patterns" in components:
                            self.pattern_memory = self.vanta_core.get_component("patterns")
                            self.connected_to.append("patterns")

                        # Look for ART controller
                        if "art_controller" in components:
                            self.art_controller = self.vanta_core.get_component(
                                "art_controller"
                            )
                            self.connected_to.append("art_controller")

                        # Look for meta-cognitive engine
                        if "meta_learning.engine" in components:
                            self.meta_cognitive = self.vanta_core.get_component(
                                "meta_learning.engine"
                            )
                            self.connected_to.append("meta_learning.engine")
                            
                        # Look for TTS/STT engines for communication enhancement
                        if "async_tts_engine" in components:
                            self.tts_engine = self.vanta_core.get_component("async_tts_engine")
                            self.connected_to.append("async_tts_engine")
                            logger.info("Connected to TTS engine for communication learning")
                            
                        if "async_stt_engine" in components:
                            self.stt_engine = self.vanta_core.get_component("async_stt_engine")
                            self.connected_to.append("async_stt_engine")
                            logger.info("Connected to STT engine for communication learning")

                        logger.info(f"Connected to components: {self.connected_to}")

                except Exception as e:
                        logger.warning(f"Error connecting to components: {e}")
            except Exception as e:
                logger.error(f"Failed to register with Vanta Core: {e}")
        else:
            logger.warning("Vanta Core not available, operating in standalone mode")

    def register_task(self, task_id: str, task_features: Dict[str, Any]) -> None:
        """
        Register a new learning task.

        Args:
            task_id: Task identifier
            task_features: Features describing the task
        """
        if not self.enabled:
            return

        # Initialize task record
        self.learning_tasks[task_id] = {
            "features": task_features,
            "performances": [],
            "parameters": {
                "learning_rate": self.meta_parameters["learning_rate"],
                "exploration_rate": self.meta_parameters["exploration_rate"],
            },
            "created": time.time(),
            "last_updated": time.time(),
        }

        # Find similar tasks for knowledge transfer
        similar_tasks = self._find_similar_tasks(task_id, task_features)

        # Initialize parameters from similar tasks if available
        if similar_tasks:
            self._transfer_parameters(task_id, similar_tasks)

        logger.info(
            f"Registered new task: {task_id} with {len(similar_tasks)} similar tasks"
        )

    def _find_similar_tasks(
        self, current_task_id: str, task_features: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Find tasks that are similar to the given task.

        Args:
            current_task_id: Task to compare against
            task_features: Features of the current task

        Returns:
            List of (task_id, similarity) tuples
        """
        similarities = []

        for task_id, task in self.learning_tasks.items():
            # Skip current task
            if task_id == current_task_id:
                continue

            # Calculate similarity based on feature overlap
            similarity = self._calculate_similarity(task_features, task["features"])

            if similarity > 0.3:  # Only include tasks with sufficient similarity
                similarities.append((task_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def _calculate_similarity(
        self, features1: Dict[str, Any], features2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two feature sets.

        Args:
            features1: First feature set
            features2: Second feature set

        Returns:
            Similarity score (0-1)
        """
        # Get common keys
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        # Calculate similarity for each feature
        similarities = []

        for key in common_keys:
            if key in ["domain", "type", "category"]:
                # Categorical features - exact match check
                if features1[key] == features2[key]:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
            elif isinstance(features1[key], (int, float)) and isinstance(
                features2[key], (int, float)
            ):
                # Numerical features - relative distance
                max_val = max(abs(features1[key]), abs(features2[key]))
                if max_val > 0:
                    diff = abs(features1[key] - features2[key]) / max_val
                    similarities.append(1.0 - min(1.0, diff))
                else:
                    similarities.append(1.0)  # Both values are 0
            else:
                # Other features - skip
                continue

        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0

    def _transfer_parameters(
        self, target_task_id: str, similar_tasks: List[Tuple[str, float]]
    ) -> None:
        """
        Transfer parameters from similar tasks to target task.

        Args:
            target_task_id: Task to transfer parameters to
            similar_tasks: List of similar tasks with similarities
        """
        # Only use top 3 most similar tasks
        top_tasks = similar_tasks[:3]

        # Skip if no similar tasks
        if not top_tasks:
            return

        # Get target task
        target_task = self.learning_tasks[target_task_id]

        # Calculate weighted parameters from similar tasks
        weighted_params = {}
        total_weight = sum(similarity for _, similarity in top_tasks)

        for task_id, similarity in top_tasks:
            source_task = self.learning_tasks[task_id]
            weight = similarity / total_weight

            for param_name, param_value in source_task["parameters"].items():
                if param_name not in weighted_params:
                    weighted_params[param_name] = 0.0
                weighted_params[param_name] += param_value * weight

        # Apply transfer strength
        transfer_strength = self.meta_parameters["transfer_strength"]

        for param_name, weighted_value in weighted_params.items():
            # Combine original value with transferred value
            original_value = target_task["parameters"].get(
                param_name, self.meta_parameters.get(param_name, 0.1)
            )
            new_value = (
                1.0 - transfer_strength
            ) * original_value + transfer_strength * weighted_value

            # Update parameter
            target_task["parameters"][param_name] = new_value

        # Record transfer event
        transfer_record = {
            "timestamp": time.time(),
            "target_task": target_task_id,
            "source_tasks": [task_id for task_id, _ in top_tasks],
            "similarities": [similarity for _, similarity in top_tasks],
            "transfer_strength": transfer_strength,
            "parameters": target_task["parameters"].copy(),
        }

        self.transfer_history.append(transfer_record)

        # Trim history if needed
        if len(self.transfer_history) > self.max_history:
            self.transfer_history = self.transfer_history[-self.max_history :]

    def update_performance(self, task_id: str, performance: float) -> None:
        """
        Update performance metrics for a task.

        Args:
            task_id: Task identifier
            performance: Performance metric (0-1)
        """
        if not self.enabled or task_id not in self.learning_tasks:
            return

        task = self.learning_tasks[task_id]

        # Update performance history
        task["performances"].append(
            {
                "value": performance,
                "timestamp": time.time(),
                "parameters": task["parameters"].copy(),
            }
        )

        # Limit performance history
        if len(task["performances"]) > self.max_history:
            task["performances"] = task["performances"][-self.max_history :]

        # Update task timestamp
        task["last_updated"] = time.time()

        # Update global performance tracking
        if task_id not in self.task_performances:
            self.task_performances[task_id] = []

        self.task_performances[task_id].append(performance)

        # Limit global performance history
        if len(self.task_performances[task_id]) > self.max_history:
            self.task_performances[task_id] = self.task_performances[task_id][
                -self.max_history :
            ]

        # Adapt parameters based on performance
        self._adapt_parameters(task_id)

    def _adapt_parameters(self, task_id: str) -> None:
        """
        Adapt task parameters based on performance history.

        Args:
            task_id: Task identifier
        """
        task = self.learning_tasks[task_id]
        performances = task["performances"]

        # Need at least 3 performance points to adapt
        if len(performances) < 3:
            return

        # Get recent performances
        recent = performances[-3:]

        # Calculate trend (positive or negative)
        values = [p["value"] for p in recent]
        is_improving = values[-1] > values[0]

        # Adapt learning rate based on trend
        current_lr = task["parameters"]["learning_rate"]

        if is_improving:
            # If improving, slightly increase learning rate
            new_lr = min(1.0, current_lr * 1.05)
        else:
            # If not improving, decrease learning rate
            new_lr = max(0.001, current_lr * 0.9)

        task["parameters"]["learning_rate"] = new_lr

        # Adapt exploration rate based on performance variance
        if len(values) >= 3:
            variance = np.var(values)
            current_exr = task["parameters"]["exploration_rate"]

            if variance < 0.01:
                # Low variance, increase exploration
                new_exr = min(0.5, current_exr * 1.1)
            elif variance > 0.1:
                # High variance, decrease exploration
                new_exr = max(0.05, current_exr * 0.9)
            else:
                # Moderate variance, maintain exploration
                new_exr = current_exr

            task["parameters"]["exploration_rate"] = new_exr

        logger.debug(
            f"Adapted parameters for task {task_id}: LR={new_lr:.4f}, EXR={task['parameters']['exploration_rate']:.4f}"
        )

    def get_task_parameters(self, task_id: str) -> Dict[str, Any]:
        """
        Get current parameters for a task.

        Args:
            task_id: Task identifier

        Returns:
            Dict with task parameters
        """
        if not self.enabled or task_id not in self.learning_tasks:
            # Return default parameters
            return {
                "learning_rate": self.meta_parameters["learning_rate"],
                "exploration_rate": self.meta_parameters["exploration_rate"],
            }

        return self.learning_tasks[task_id]["parameters"].copy()

    def analyze_task_performance(self, task_id: str) -> Dict[str, Any]:
        """
        Analyze task performance over time.

        Args:
            task_id: Task identifier

        Returns:
            Dict with performance analysis
        """
        if not self.enabled or task_id not in self.learning_tasks:
            return {"error": "Task not found"}

        task = self.learning_tasks[task_id]
        performances = task["performances"]

        if not performances:
            return {"error": "No performance data available"}

        # Extract performance values and timestamps
        values = [p["value"] for p in performances]
        timestamps = [p["timestamp"] for p in performances]

        # Calculate basic statistics
        avg_performance = sum(values) / len(values)
        max_performance = max(values)
        min_performance = min(values)

        # Calculate trend (simple linear regression)
        if len(values) >= 2:
            x = np.array(range(len(values)))
            y = np.array(values)

            # Calculate slope using least squares
            mean_x = np.mean(x)
            mean_y = np.mean(y)

            numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
            denominator = sum((x_i - mean_x) ** 2 for x_i in x)

            if denominator != 0:
                slope = numerator / denominator
            else:
                slope = 0.0

            trend = (
                "improving"
                if slope > 0.01
                else "declining"
                if slope < -0.01
                else "stable"
            )
        else:
            trend = "unknown"
            slope = 0.0

        return {
            "task_id": task_id,
            "avg_performance": avg_performance,
            "max_performance": max_performance,
            "min_performance": min_performance,
            "latest_performance": values[-1],
            "performance_count": len(values),
            "trend": trend,
            "trend_slope": slope,
            "first_timestamp": timestamps[0],
            "last_timestamp": timestamps[-1],
        }

    def get_global_performance(self) -> Dict[str, Any]:
        """
        Get global performance metrics across all tasks.

        Returns:
            Dict with global performance metrics
        """
        if not self.enabled or not self.task_performances:
            return {"tasks": 0, "performances": 0}

        # Calculate per-task metrics
        task_metrics = {}
        for task_id, performances in self.task_performances.items():
            if performances:
                task_metrics[task_id] = {
                    "avg_performance": sum(performances) / len(performances),
                    "max_performance": max(performances),
                    "count": len(performances),
                }

        # Calculate global metrics
        all_performances = [
            p for perfs in self.task_performances.values() for p in perfs
        ]

        if not all_performances:
            return {"tasks": len(self.task_performances), "performances": 0}

        return {
            "tasks": len(self.task_performances),
            "performances": len(all_performances),
            "global_avg": sum(all_performances) / len(all_performances),
            "task_metrics": task_metrics,
        }

    def optimize_meta_parameters(self) -> Dict[str, Any]:
        """
        Optimize global meta-parameters based on all task performances.

        Returns:
            Dict with optimization results
        """
        if not self.enabled or not self.task_performances:
            return {"optimized": False, "reason": "No performance data available"}

        # Get tasks with sufficient performance data
        tasks_with_data = [
            task_id
            for task_id, performances in self.task_performances.items()
            if len(performances) >= 5
        ]

        if not tasks_with_data:
            return {"optimized": False, "reason": "Insufficient performance data"}

        # Analyze each task's parameter effectiveness
        task_param_effectiveness = {}

        for task_id in tasks_with_data:
            task = self.learning_tasks[task_id]
            performances = task["performances"]

            # Group performances by parameter values
            param_groups = {}

            for perf in performances:
                # Create key from rounded parameter values
                lr_key = round(perf["parameters"]["learning_rate"], 3)
                exr_key = round(perf["parameters"]["exploration_rate"], 3)
                key = f"lr{lr_key}_exr{exr_key}"

                if key not in param_groups:
                    param_groups[key] = []

                param_groups[key].append(perf["value"])

            # Find best performing parameter set
            best_avg = 0.0
            best_params = {}

            for key, perfs in param_groups.items():
                if len(perfs) < 2:
                    continue

                avg = sum(perfs) / len(perfs)

                if avg > best_avg:
                    best_avg = avg

                    # Extract parameters from key
                    parts = key.split("_")
                    lr = float(parts[0][2:])
                    exr = float(parts[1][3:])

                    best_params = {"learning_rate": lr, "exploration_rate": exr}

            if best_params:
                task_param_effectiveness[task_id] = {
                    "best_performance": best_avg,
                    "best_params": best_params,
                }

        # Calculate weighted average of best parameters
        if not task_param_effectiveness:
            return {
                "optimized": False,
                "reason": "Could not identify effective parameters",
            }

        weighted_params = {}
        total_performance = sum(
            info["best_performance"] for info in task_param_effectiveness.values()
        )

        for task_id, info in task_param_effectiveness.items():
            weight = info["best_performance"] / total_performance

            for param_name, param_value in info["best_params"].items():
                if param_name not in weighted_params:
                    weighted_params[param_name] = 0.0

                weighted_params[param_name] += param_value * weight

        # Update meta-parameters with damping
        old_params = self.meta_parameters.copy()
        damping = 0.7  # Gradual change

        for param_name, weighted_value in weighted_params.items():
            old_value = self.meta_parameters[param_name]
            new_value = old_value * damping + weighted_value * (1.0 - damping)
            self.meta_parameters[param_name] = new_value

        return {
            "optimized": True,
            "old_params": old_params,
            "new_params": self.meta_parameters.copy(),
            "tasks_analyzed": len(task_param_effectiveness),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get meta-learner status.

        Returns:
            Dict with status information
        """
        return {
            "enabled": self.enabled,
            "task_count": len(self.learning_tasks),
            "transfer_count": len(self.transfer_history),
            "meta_parameters": self.meta_parameters,
            "memory_connected": self.memory is not None,
            "pattern_memory_connected": self.pattern_memory is not None,
        }
    
    # --- Implementation of MetaLearnerInterface methods ---
    
    def get_heuristics(self) -> List[Dict[str, Any]]:
        """
        Get learning heuristics derived from task experience.
        
        Returns:
            List of heuristic dictionaries
        """
        if not self.enabled:
            return []
            
        heuristics = []
        for task_id, task in self.learning_tasks.items():
            # Create a heuristic from each task's learning experience
            if not task["performances"]:
                continue
                
            # Extract performance trend
            values = [p["value"] for p in task["performances"]]
            performance_avg = sum(values) / len(values) if values else 0
            
            # Get task domain and type
            domain = task["features"].get("domain", "general")
            task_type = task["features"].get("type", "unspecified")
            
            heuristic = {
                "id": f"task_{task_id}",
                "domain": domain,
                "task_type": task_type,
                "confidence": min(len(values) / 10, 1.0),  # Confidence grows with experience
                "performance": performance_avg,
                "parameters": task["parameters"].copy(),
                "features": {k: v for k, v in task["features"].items() 
                             if k not in ["raw_data", "full_context"]},  # Filter large data
                "created": task["created"],
                "last_updated": task["last_updated"]
            }
            
            heuristics.append(heuristic)
            
        return heuristics
        
    def update_heuristic(self, heuristic_id: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing heuristic.
        
        Args:
            heuristic_id: Identifier for the heuristic
            updates: Dictionary with updates to apply
        """
        if not self.enabled:
            return
            
        # Parse task_id from heuristic_id
        if heuristic_id.startswith("task_"):
            task_id = heuristic_id[5:]  # Remove "task_" prefix
        else:
            task_id = heuristic_id  # Try directly
            
        if task_id not in self.learning_tasks:
            logger.warning(f"Heuristic {heuristic_id} not found")
            return
            
        task = self.learning_tasks[task_id]
        
        # Apply updates
        if "parameters" in updates:
            for param, value in updates["parameters"].items():
                if param in task["parameters"]:
                    task["parameters"][param] = value
                    
        if "features" in updates:
            for feature, value in updates["features"].items():
                task["features"][feature] = value
                
        # Update timestamp
        task["last_updated"] = time.time()
        logger.debug(f"Updated heuristic {heuristic_id}")
        
    def add_heuristic(self, heuristic_data: Dict[str, Any]) -> Any:
        """
        Add a new heuristic from external source.
        
        Args:
            heuristic_data: Heuristic data dictionary
            
        Returns:
            Heuristic ID or None if failed
        """
        if not self.enabled:
            return None
            
        # Generate task ID
        task_id = f"ext_{int(time.time())}_{hash(str(heuristic_data)) % 1000}"
        
        # Create task features from heuristic
        task_features = heuristic_data.get("features", {}).copy()
        
        # Add domain if provided
        if "domain" in heuristic_data:
            task_features["domain"] = heuristic_data["domain"]
            
        # Add task type if provided
        if "task_type" in heuristic_data:
            task_features["type"] = heuristic_data["task_type"]
            
        # Register task
        self.register_task(task_id, task_features)
        
        # Apply parameters if provided
        if "parameters" in heuristic_data and task_id in self.learning_tasks:
            for param, value in heuristic_data["parameters"].items():
                self.learning_tasks[task_id]["parameters"][param] = value
                
        # Apply performance if provided
        if "performance" in heuristic_data and task_id in self.learning_tasks:
            self.update_performance(task_id, heuristic_data["performance"])
            
        logger.info(f"Added new heuristic as task {task_id}")
        return f"task_{task_id}"
    
    def get_transferable_knowledge(self) -> Optional[Any]:
        """
        Get knowledge that can be transferred to other components.
        
        Returns:
            Transferable knowledge package or None
        """
        if not self.enabled or not self.learning_tasks:
            return None
            
        # Create a knowledge package with high-performing tasks
        transferable_knowledge = {
            "source": "advanced_meta_learner",
            "timestamp": time.time(),
            "meta_parameters": self.meta_parameters.copy(),
            "heuristics": self.get_heuristics(),
            "domain_performances": {},
            "task_count": len(self.learning_tasks),
        }
        
        # Extract domain performance data
        for task_id, task in self.learning_tasks.items():
            domain = task["features"].get("domain", "general")
            
            if domain not in transferable_knowledge["domain_performances"]:
                transferable_knowledge["domain_performances"][domain] = {
                    "tasks": 0,
                    "performances": [],
                }
                
            # Add performance data
            domain_data = transferable_knowledge["domain_performances"][domain]
            domain_data["tasks"] += 1
            
            if task["performances"]:
                # Add recent performances
                recent_perfs = [p["value"] for p in task["performances"][-5:]]
                domain_data["performances"].extend(recent_perfs)
                
                # Update domain performance in tracker
                avg_perf = sum(recent_perfs) / len(recent_perfs)
                self.performance_tracker.update_domain_performance(
                    domain, avg_perf, {"task_id": task_id}
                )
                
        # Add any additional knowledge from the knowledge transfer system
        transferable_knowledge["cross_domain_data"] = self.knowledge_transfer.get_all_domain_knowledge()
            
        return transferable_knowledge
    
    def integrate_knowledge(self, knowledge: Any, source: Any) -> None:
        """
        Integrate knowledge from other components.
        
        Args:
            knowledge: Knowledge package to integrate
            source: Source of the knowledge
        """
        if not self.enabled or not knowledge:
            return
            
        logger.info(f"Integrating knowledge from {source}")
        
        # Handle different knowledge types
        if isinstance(knowledge, dict):
            # Check for heuristics
            if "heuristics" in knowledge and isinstance(knowledge["heuristics"], list):
                for heuristic in knowledge["heuristics"]:
                    # Don't re-integrate our own heuristics
                    if "source" in heuristic and heuristic["source"] == "advanced_meta_learner":
                        continue
                        
                    # Add the heuristic with source attribution
                    heuristic_with_source = heuristic.copy()
                    heuristic_with_source["source"] = source
                    self.add_heuristic(heuristic_with_source)
            
            # Check for domain knowledge
            if "domain_knowledge" in knowledge and isinstance(knowledge["domain_knowledge"], dict):
                for domain, domain_data in knowledge["domain_knowledge"].items():
                    # Store in cross-domain knowledge transfer system
                    self.knowledge_transfer.store_domain_knowledge(
                        domain, domain_data, {"source": source}
                    )
        
        # Handle other knowledge formats - pattern data, performance data, etc.
        # (This would be expanded based on actual knowledge formats from other components)
    
    def get_performance_metrics(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics across domains.
        
        Args:
            domain: Optional domain name to get metrics for
            
        Returns:
            Dict with performance metrics
        """
        if not self.enabled:
            return {"enabled": False}
            
        # Use the performance tracker to get metrics
        metrics = self.performance_tracker.get_domain_performance(domain)
        
        # Add meta-learner specific information
        metrics["meta_learner_version"] = "1.0"
        metrics["tasks_tracked"] = len(self.learning_tasks)
        
        # Add task count per domain if no specific domain requested
        if domain is None:
            domain_task_counts = {}
            for task_id, task in self.learning_tasks.items():
                task_domain = task["features"].get("domain", "general")
                if task_domain not in domain_task_counts:
                    domain_task_counts[task_domain] = 0
                domain_task_counts[task_domain] += 1
                metrics["domain_task_counts"] = domain_task_counts
            
        return metrics
        
    # === Communication Enhancement Methods for API-Driven Learning ===
    
    async def enhance_tts_communication(self, text: str, synthesis_request: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance TTS communication by learning from synthesis interactions.
        VantaCore can use this to improve speech synthesis through submodel learning.
        
        Args:
            text: Text being synthesized
            synthesis_request: TTS synthesis request parameters
            context: Optional context information (user preferences, conversation state, etc.)
            
        Returns:
            Enhanced synthesis parameters and learning insights
        """
        if not self.communication_enhancement_active:
            return {"enhancement_active": False, "original_request": synthesis_request}
            
        # Apply learned enhancements to synthesis request
        enhanced_request = synthesis_request.copy()
        
        # Get text features for analysis
        text_features = self.communication_learner._extract_text_features(text)
        
        # Apply learned optimizations
        if self.communication_learner.tts_performance_metrics:
            # Find best performing engine for this text complexity
            complexity = text_features.get("complexity_score", 0)
            best_engine = self._get_optimal_tts_engine(complexity)
            
            if best_engine and best_engine != enhanced_request.get("engine"):
                enhanced_request["preferred_engine"] = best_engine
                enhanced_request["engine_reason"] = f"optimal_for_complexity_{complexity:.2f}"
                
        # Add communication-specific enhancements
        enhanced_request["learning_context"] = {
            "text_features": text_features,
            "context": context,
            "enhancement_version": "1.0"
        }
        
        return {
            "enhancement_active": True,
            "original_request": synthesis_request,
            "enhanced_request": enhanced_request,
            "text_features": text_features,
            "recommendations": self.communication_learner._generate_tts_recommendations(
                text_features, {"success": True}, enhanced_request.get("engine", "default")
            )
        }
        
    async def learn_from_tts_result(self, text: str, synthesis_result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn from TTS synthesis results to improve future communications.
        VantaCore calls this after TTS synthesis to enable continuous learning.
        
        Args:
            text: Original text that was synthesized
            synthesis_result: Result from TTS engine
            context: Optional context information
            
        Returns:
            Learning insights and performance metrics
        """
        if not self.communication_enhancement_active:
            return {"learning_active": False}
            
        # Use communication learner to process the interaction
        learning_result = await self.communication_learner.learn_from_tts_interaction(
            text, synthesis_result, context
        )
        
        # Update domain knowledge if context provides domain information
        if context and "domain" in context:
            domain = context["domain"]
            domain_knowledge = {
                "text_synthesis": {
                    "text_features": learning_result.get("text_features", {}),
                    "engine_performance": learning_result.get("engine_metrics", {}),
                    "success_rate": float(synthesis_result.get("success", False))
                }
            }
            
            # Store in cross-domain knowledge transfer system
            self.knowledge_transfer.store_domain_knowledge(
                f"tts_{domain}", domain_knowledge, {
                    "source": "tts_learning", 
                    "timestamp": time.time()
                }
            )
            
            # Update performance tracker
            self.performance_tracker.update_domain_performance(
                f"tts_{domain}", 
                domain_knowledge["text_synthesis"]["success_rate"],
                {"text_length": len(text), "engine": synthesis_result.get("engine_used")}
            )
            
        return learning_result
        
    async def enhance_stt_communication(self, audio_context: Dict[str, Any], transcription_request: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance STT communication by applying learned optimizations.
        VantaCore can use this to improve speech recognition through submodel learning.
        
        Args:
            audio_context: Audio metadata and context
            transcription_request: STT transcription request parameters
            context: Optional context information
            
        Returns:
            Enhanced transcription parameters and insights
        """
        if not self.communication_enhancement_active:
            return {"enhancement_active": False, "original_request": transcription_request}
            
        # Apply learned enhancements to transcription request
        enhanced_request = transcription_request.copy()
        
        # Analyze audio context for optimization opportunities
        if self.communication_learner.stt_performance_metrics:
            metrics = self.communication_learner.stt_performance_metrics.get("stt", {})
            
            # Optimize duration based on learned performance patterns
            if metrics.get("processing_times"):
                avg_processing = sum(metrics["processing_times"]) / len(metrics["processing_times"])
                requested_duration = enhanced_request.get("duration", 10)
                
                # Suggest optimal duration based on learned patterns
                if avg_processing > 3000 and requested_duration > 8:  # Long processing, suggest shorter
                    enhanced_request["suggested_duration"] = min(6, requested_duration)
                    enhanced_request["optimization_reason"] = "reduce_processing_time"
                    
        # Add learning context
        enhanced_request["learning_context"] = {
            "audio_context": audio_context,
            "context": context,
            "enhancement_version": "1.0"
        }
        
        return {
            "enhancement_active": True,
            "original_request": transcription_request,
            "enhanced_request": enhanced_request,
            "optimization_applied": "processing_time_optimization" if "suggested_duration" in enhanced_request else "none"
        }
        
    async def learn_from_stt_result(self, audio_context: Dict[str, Any], transcription_result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn from STT transcription results to improve future communications.
        VantaCore calls this after STT transcription to enable continuous learning.
        
        Args:
            audio_context: Audio metadata and context
            transcription_result: Result from STT engine
            context: Optional context information
            
        Returns:
            Learning insights and performance metrics
        """
        if not self.communication_enhancement_active:
            return {"learning_active": False}
            
        # Use communication learner to process the interaction
        learning_result = await self.communication_learner.learn_from_stt_interaction(
            audio_context, transcription_result, context
        )
        
        # Update domain knowledge if context provides domain information
        if context and "domain" in context:
            domain = context["domain"]
            domain_knowledge = {
                "speech_recognition": {
                    "audio_features": learning_result.get("audio_features", {}),
                    "processing_metrics": learning_result.get("engine_metrics", {}),
                    "success_rate": 1.0 if transcription_result.get("text") else 0.0
                }
            }
            
            # Store in cross-domain knowledge transfer system
            self.knowledge_transfer.store_domain_knowledge(
                f"stt_{domain}", domain_knowledge, {
                    "source": "stt_learning", 
                    "timestamp": time.time()
                }
            )
            
            # Update performance tracker
            self.performance_tracker.update_domain_performance(
                f"stt_{domain}",
                domain_knowledge["speech_recognition"]["success_rate"],
                {"text_length": len(transcription_result.get("text", "")), "processing_time": transcription_result.get("processing_time_ms")}
            )
            
        return learning_result
        
    def _get_optimal_tts_engine(self, complexity_score: float) -> Optional[str]:
        """
        Get the optimal TTS engine for a given text complexity.
        
        Args:
            complexity_score: Text complexity score (0-1)
            
        Returns:
            Optimal engine name or None
        """
        if not self.communication_learner.tts_performance_metrics:
            return None
            
        best_engine = None
        best_score = 0.0
        
        for engine, metrics in self.communication_learner.tts_performance_metrics.items():
            if not metrics.get("text_complexity_handling"):
                continue
                
            # Find performance for similar complexity levels
            complexity_key = round(complexity_score, 1)  # Round to nearest 0.1
            performance_data = metrics["text_complexity_handling"].get(complexity_key, [])
            
            if performance_data:
                success_rate = sum(performance_data) / len(performance_data)
                
                # Factor in overall engine performance
                overall_success = sum(metrics.get("success_rate", [])) / len(metrics.get("success_rate", [1])) if metrics.get("success_rate") else 0.5
                
                # Combined score
                combined_score = (success_rate * 0.7) + (overall_success * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_engine = engine
                    
        return best_engine
        
    def get_communication_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive insights about communication learning and performance.
        
        Returns:
            Dict with communication insights and metrics
        """
        if not self.communication_enhancement_active:
            return {"enhancement_active": False}
            
        insights = self.communication_learner.get_communication_insights()
        
        # Add domain-specific communication performance
        domain_communication_performance = {}
        
        # Get TTS domain performance
        for domain, knowledge in self.knowledge_transfer.get_all_domain_knowledge().items():
            if domain.startswith("tts_") or domain.startswith("stt_"):
                domain_type = domain.split("_")[0]
                domain_name = "_".join(domain.split("_")[1:])
                
                if domain_name not in domain_communication_performance:
                    domain_communication_performance[domain_name] = {}
                    
                domain_communication_performance[domain_name][domain_type] = {
                    "last_update": knowledge.get("timestamp", 0),
                    "knowledge_available": True
                }
                
        insights["domain_performance"] = domain_communication_performance
        insights["enhancement_active"] = True
        insights["engines_connected"] = {
            "tts": self.tts_engine is not None,
            "stt": self.stt_engine is not None
        }
        
        return insights
        
    def toggle_communication_enhancement(self, active: Optional[bool] = None) -> bool:
        """
        Toggle or set communication enhancement learning.
        
        Args:
            active: Optional boolean to set state, None to toggle
            
        Returns:
            New enhancement state
        """
        if active is not None:
            self.communication_enhancement_active = active
        else:
            self.communication_enhancement_active = not self.communication_enhancement_active
            
        # Also toggle the underlying communication learner
        self.communication_learner.toggle_learning(self.communication_enhancement_active)
        
        logger.info(f"Communication enhancement learning: {'enabled' if self.communication_enhancement_active else 'disabled'}")
        return self.communication_enhancement_active


class CommunicationEnhancementLearner:
    """
    Learning system for enhancing TTS/STT communication through API-driven feedback.
    """
    
    def __init__(self, max_history: int = 1000):
        self.interaction_history = []
        self.tts_performance_metrics = {}
        self.stt_performance_metrics = {}
        self.communication_patterns = {}
        self.max_history = max_history
        self.learning_active = True
        
        # Communication quality metrics
        self.quality_indicators = {
            "speech_clarity": [],
            "recognition_accuracy": [],
            "response_timing": [],
            "user_satisfaction": [],
            "context_awareness": []
        }
        
    async def learn_from_tts_interaction(self, text: str, synthesis_result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn from TTS synthesis interactions to improve future speech generation.
        
        Args:
            text: Original text that was synthesized
            synthesis_result: Result from TTS engine
            context: Optional context information
            
        Returns:
            Learning insights and recommendations
        """
        if not self.learning_active:
            return {"learning_active": False}
            
        interaction = {
            "type": "tts",
            "timestamp": time.time(),
            "text": text,
            "result": synthesis_result,
            "context": context or {},
            "text_length": len(text),
            "success": synthesis_result.get("success", False),
            "duration_ms": synthesis_result.get("duration_ms", 0),
            "engine_used": synthesis_result.get("engine_used", "unknown")
        }
        
        # Add to history
        self.interaction_history.append(interaction)
        self._trim_history()
        
        # Analyze text patterns
        text_features = self._extract_text_features(text)
        
        # Update TTS performance metrics
        engine = synthesis_result.get("engine_used", "unknown")
        if engine not in self.tts_performance_metrics:
            self.tts_performance_metrics[engine] = {
                "success_rate": [],
                "avg_duration": [],
                "text_complexity_handling": {},
                "error_patterns": []
            }
            
        metrics = self.tts_performance_metrics[engine]
        metrics["success_rate"].append(float(synthesis_result.get("success", False)))
        if synthesis_result.get("duration_ms"):
            metrics["avg_duration"].append(synthesis_result["duration_ms"])
            
        # Track text complexity handling
        complexity = text_features.get("complexity_score", 0)
        if complexity not in metrics["text_complexity_handling"]:
            metrics["text_complexity_handling"][complexity] = []
        metrics["text_complexity_handling"][complexity].append(synthesis_result.get("success", False))
        
        # Track errors
        if not synthesis_result.get("success", False) and synthesis_result.get("error"):
            metrics["error_patterns"].append({
                "error": synthesis_result["error"],
                "text_features": text_features,
                "timestamp": time.time()
            })
            
        # Generate recommendations
        recommendations = self._generate_tts_recommendations(text_features, synthesis_result, engine)
        
        return {
            "learning_active": True,
            "interaction_recorded": True,
            "text_features": text_features,
            "recommendations": recommendations,
            "engine_metrics": self._get_engine_performance_summary(engine, "tts")
        }
        
    async def learn_from_stt_interaction(self, audio_data: Any, transcription_result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn from STT transcription interactions to improve speech recognition.
        
        Args:
            audio_data: Audio data that was transcribed (metadata only for learning)
            transcription_result: Result from STT engine
            context: Optional context information
            
        Returns:
            Learning insights and recommendations
        """
        if not self.learning_active:
            return {"learning_active": False}
            
        # Extract audio metadata without storing raw audio
        audio_features = self._extract_audio_features(transcription_result)
        
        interaction = {
            "type": "stt",
            "timestamp": time.time(),
            "audio_features": audio_features,
            "result": transcription_result,
            "context": context or {},
            "text": transcription_result.get("text", ""),
            "is_final": transcription_result.get("is_final", False),
            "processing_time_ms": transcription_result.get("processing_time_ms", 0),
            "confidence": transcription_result.get("confidence", None)
        }
        
        # Add to history
        self.interaction_history.append(interaction)
        self._trim_history()
        
        # Update STT performance metrics
        if "stt" not in self.stt_performance_metrics:
            self.stt_performance_metrics["stt"] = {
                "transcription_accuracy": [],
                "processing_times": [],
                "audio_quality_handling": {},
                "error_patterns": []
            }
            
        metrics = self.stt_performance_metrics["stt"]
        
        # Track processing times
        if transcription_result.get("processing_time_ms"):
            metrics["processing_times"].append(transcription_result["processing_time_ms"])
            
        # Track audio quality handling
        audio_quality = audio_features.get("quality_score", 0)
        if audio_quality not in metrics["audio_quality_handling"]:
            metrics["audio_quality_handling"][audio_quality] = []
        metrics["audio_quality_handling"][audio_quality].append({
            "success": bool(transcription_result.get("text")),
            "text_length": len(transcription_result.get("text", ""))
        })
        
        # Track errors
        if transcription_result.get("error"):
            metrics["error_patterns"].append({
                "error": transcription_result["error"],
                "audio_features": audio_features,
                "timestamp": time.time()
            })
            
        # Generate recommendations
        recommendations = self._generate_stt_recommendations(audio_features, transcription_result)
        
        return {
            "learning_active": True,
            "interaction_recorded": True,
            "audio_features": audio_features,
            "recommendations": recommendations,
            "engine_metrics": self._get_engine_performance_summary("stt", "stt")
        }
        
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text for TTS learning."""
        words = text.split()
        
        # Basic text features
        features = {
            "length": len(text),
            "word_count": len(words),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "punctuation_density": sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text) if text else 0,
            "uppercase_ratio": sum(1 for char in text if char.isupper()) / len(text) if text else 0,
            "complexity_score": 0
        }
        
        # Calculate complexity score
        complexity = (
            min(features["word_count"] / 10, 1.0) * 0.3 +  # Word count factor
            min(features["avg_word_length"] / 8, 1.0) * 0.3 +  # Word length factor
            min(features["punctuation_density"] * 10, 1.0) * 0.2 +  # Punctuation factor
            min(features["sentence_count"] / 5, 1.0) * 0.2  # Sentence count factor
        )
        features["complexity_score"] = complexity
        
        return features
        
    def _extract_audio_features(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from audio/transcription for STT learning."""
        text = transcription_result.get("text", "")
        
        features = {
            "transcribed_length": len(text),
            "processing_time_ms": transcription_result.get("processing_time_ms", 0),
            "is_final": transcription_result.get("is_final", False),
            "has_error": bool(transcription_result.get("error")),
            "quality_score": 0,
            "confidence": transcription_result.get("confidence")
        }
        
        # Estimate quality score based on available data
        if text and not transcription_result.get("error"):
            quality = 1.0
            
            # Adjust based on processing time (faster might indicate clearer audio)
            proc_time = features["processing_time_ms"]
            if proc_time > 0:
                # Normalized processing time factor
                time_factor = max(0.5, min(1.0, 2000 / proc_time))
                quality *= time_factor
                
            features["quality_score"] = quality
        else:
            features["quality_score"] = 0.2 if transcription_result.get("error") else 0.5
            
        return features
        
    def _generate_tts_recommendations(self, text_features: Dict[str, Any], synthesis_result: Dict[str, Any], engine: str) -> Dict[str, Any]:
        """Generate recommendations for TTS improvement."""
        recommendations = {
            "engine_selection": {},
            "text_preprocessing": [],
            "synthesis_parameters": {},
            "quality_improvements": []
        }
        
        # Engine selection recommendations
        if engine in self.tts_performance_metrics:
            metrics = self.tts_performance_metrics[engine]
            
            # Success rate analysis
            if metrics["success_rate"]:
                success_rate = sum(metrics["success_rate"]) / len(metrics["success_rate"])
                if success_rate < 0.8:
                    recommendations["engine_selection"]["consider_alternative"] = True
                    recommendations["engine_selection"]["current_success_rate"] = success_rate
                    
            # Duration analysis
            if metrics["avg_duration"]:
                avg_duration = sum(metrics["avg_duration"]) / len(metrics["avg_duration"])
                text_length = text_features.get("length", 0)
                if text_length > 0:
                    ms_per_char = avg_duration / text_length
                    if ms_per_char > 100:  # Arbitrary threshold
                        recommendations["synthesis_parameters"]["consider_faster_voice"] = True
                        
        # Text preprocessing recommendations
        complexity = text_features.get("complexity_score", 0)
        if complexity > 0.7:
            recommendations["text_preprocessing"].append("consider_simplification")
            
        if text_features.get("punctuation_density", 0) > 0.3:
            recommendations["text_preprocessing"].append("normalize_punctuation")
            
        # Quality improvements based on errors
        if not synthesis_result.get("success", False):
            error = synthesis_result.get("error", "")
            if "timeout" in error.lower():
                recommendations["quality_improvements"].append("reduce_text_length")
            elif "engine" in error.lower():
                recommendations["quality_improvements"].append("try_alternative_engine")
                
        return recommendations
        
    def _generate_stt_recommendations(self, audio_features: Dict[str, Any], transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for STT improvement."""
        recommendations = {
            "audio_quality": [],
            "processing_optimization": [],
            "accuracy_improvements": [],
            "error_handling": []
        }
        
        # Processing time recommendations
        proc_time = audio_features.get("processing_time_ms", 0)
        if proc_time > 5000:  # More than 5 seconds
            recommendations["processing_optimization"].append("consider_shorter_audio_segments")
            
        # Quality recommendations
        quality = audio_features.get("quality_score", 0)
        if quality < 0.5:
            recommendations["audio_quality"].append("improve_audio_input_quality")
            
        # Error handling recommendations
        if transcription_result.get("error"):
            error = transcription_result["error"].lower()
            if "timeout" in error:
                recommendations["error_handling"].append("reduce_audio_duration")
            elif "model" in error:
                recommendations["error_handling"].append("check_model_availability")
                
        # Accuracy improvements
        text_length = len(transcription_result.get("text", ""))
        if text_length == 0 and not transcription_result.get("error"):
            recommendations["accuracy_improvements"].append("check_audio_volume_levels")
            
        return recommendations
        
    def _get_engine_performance_summary(self, engine: str, engine_type: str) -> Dict[str, Any]:
        """Get performance summary for an engine."""
        if engine_type == "tts" and engine in self.tts_performance_metrics:
            metrics = self.tts_performance_metrics[engine]
            return {
                "engine": engine,
                "type": engine_type,
                "success_rate": sum(metrics["success_rate"]) / len(metrics["success_rate"]) if metrics["success_rate"] else 0,
                "avg_duration_ms": sum(metrics["avg_duration"]) / len(metrics["avg_duration"]) if metrics["avg_duration"] else 0,
                "total_interactions": len(metrics["success_rate"]),
                "error_count": len(metrics["error_patterns"])
            }
        elif engine_type == "stt" and "stt" in self.stt_performance_metrics:
            metrics = self.stt_performance_metrics["stt"]
            return {
                "engine": "stt",
                "type": engine_type,
                "avg_processing_ms": sum(metrics["processing_times"]) / len(metrics["processing_times"]) if metrics["processing_times"] else 0,
                "total_interactions": len(metrics["processing_times"]),
                "error_count": len(metrics["error_patterns"])
            }
        else:
            return {"engine": engine, "type": engine_type, "data_available": False}
            
    def _trim_history(self):
        """Trim interaction history to max_history size."""
        if len(self.interaction_history) > self.max_history:
            self.interaction_history = self.interaction_history[-self.max_history:]
            
    def get_communication_insights(self) -> Dict[str, Any]:
        """Get insights about communication patterns and performance."""
        insights = {
            "total_interactions": len(self.interaction_history),
            "tts_interactions": sum(1 for i in self.interaction_history if i["type"] == "tts"),
            "stt_interactions": sum(1 for i in self.interaction_history if i["type"] == "stt"),
            "learning_active": self.learning_active,
            "engines_tracked": {
                "tts": list(self.tts_performance_metrics.keys()),
                "stt": list(self.stt_performance_metrics.keys())
            }
        }
        
        # Add recent performance trends
        recent_interactions = self.interaction_history[-50:]  # Last 50 interactions
        if recent_interactions:
            insights["recent_trends"] = {
                "avg_tts_duration": 0,
                "avg_stt_processing": 0,
                "success_rates": {}
            }
            
            tts_durations = [i["result"].get("duration_ms", 0) for i in recent_interactions if i["type"] == "tts"]
            stt_times = [i["result"].get("processing_time_ms", 0) for i in recent_interactions if i["type"] == "stt"]
            
            if tts_durations:
                insights["recent_trends"]["avg_tts_duration"] = sum(tts_durations) / len(tts_durations)
            if stt_times:
                insights["recent_trends"]["avg_stt_processing"] = sum(stt_times) / len(stt_times)
                
        return insights
        
    def toggle_learning(self, active: bool = None) -> bool:
        """Toggle or set learning state."""
        if active is not None:
            self.learning_active = active
        else:
            self.learning_active = not self.learning_active
        return self.learning_active
