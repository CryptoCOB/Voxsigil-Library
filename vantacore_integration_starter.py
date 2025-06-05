"""
VantaCore Integration for Iterative Reasoning GridFormer
----------------------------------------------------------
This module facilitates integration between the trained Iterative Reasoning GridFormer
and the VantaCore meta-learning system, compliant with VoxSigil Schema 1.5-holo-alpha.

The GridFormerTaskProfile class creates a bridge between the trained model and VantaCore's
adaptive meta-learning capabilities, enabling cross-task knowledge transfer and
parameter optimization.
"""

import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import time
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Robust model import setup ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

try:
    from Gridformer.inference.iterative_reasoning_gridformer import (
        ReasoningConfig,
        create_iterative_gridformer,
    )
except Exception as e:
    logger.error(
        f"ERROR importing iterative_reasoning_gridformer from {MODELS_DIR}: {str(e)}"
    )
    raise


@dataclass
class GridFormerTaskProfile:
    """
    Task profile for GridFormer model integration with VantaCore

    This class maintains the performance tracking and parameter adaptation
    for a specific ARC task, enabling VantaCore to apply meta-learning
    across similar tasks.
    """

    task_id: str
    description: Optional[str] = None
    embeddings: Optional[torch.Tensor] = None

    # Task-specific parameters that can be adapted by VantaCore
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "confidence_threshold": 0.85,
            "max_iterations": 5,
            "pattern_consistency_weight": 0.3,
            "diversity_penalty": 0.1,
            "refinement_strength": 0.1,
        }
    )

    # Performance metrics tracked over time
    performance_history: List[Dict[str, float]] = field(default_factory=list)

    # Cross-task similarity data
    similar_tasks: List[Tuple[str, float]] = field(default_factory=list)

    def update_performance(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics for this task"""
        self.performance_history.append({"timestamp": time.time(), **metrics})
        logger.info(f"Updated performance for task {self.task_id}: {metrics}")

    def adapt_parameters(self, adaptation_data: Dict[str, Any]) -> None:
        """Adapt task-specific parameters based on performance feedback"""
        for param_name, adjustment in adaptation_data.items():
            if param_name in self.params:
                old_value = self.params[param_name]
                self.params[param_name] += adjustment
                logger.info(
                    f"Adapted {param_name} for task {self.task_id}: {old_value} -> {self.params[param_name]}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "params": self.params,
            "performance_history": self.performance_history,
            "similar_tasks": self.similar_tasks,
            # Embeddings are handled separately
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridFormerTaskProfile":
        """Create profile from dictionary"""
        profile = cls(task_id=data["task_id"], description=data.get("description"))
        profile.params = data.get("params", profile.params)
        profile.performance_history = data.get("performance_history", [])
        profile.similar_tasks = data.get("similar_tasks", [])
        return profile


class GridFormerVantaCoreConnector:
    """
    Connector for integrating trained GridFormer models with VantaCore

    This class provides the interface between VantaCore's meta-learning system
    and the Iterative Reasoning GridFormer model, enabling:

    1. Task-specific parameter adaptation
    2. Performance tracking and feedback
    3. Cross-task knowledge transfer
    4. Meta-parameter optimization
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        self.task_profiles: Dict[str, GridFormerTaskProfile] = {}

        # Load model
        self._load_model()
        logger.info(
            f"GridFormerVantaCoreConnector initialized with model from {model_path}"
        )

    def _load_model(self) -> None:
        """Load the GridFormer model from checkpoint"""
        try:
            # Load config if provided
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)
                reasoning_config = ReasoningConfig(**config_data)
            else:
                # Use default config
                reasoning_config = ReasoningConfig()

            # Create model
            self.model = create_iterative_gridformer(config=reasoning_config)

            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def register_task(self, task_id: str, description: Optional[str] = None) -> None:
        """Register a new ARC task for meta-learning"""
        if task_id in self.task_profiles:
            logger.warning(f"Task {task_id} already registered. Updating description.")
            self.task_profiles[task_id].description = description
        else:
            self.task_profiles[task_id] = GridFormerTaskProfile(
                task_id=task_id, description=description
            )
            logger.info(f"Registered new task: {task_id}")

    def process_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an ARC task using the current adapted parameters"""
        # Ensure task is registered
        if task_id not in self.task_profiles:
            self.register_task(task_id)

        # Get task-specific parameters
        task_profile = self.task_profiles[task_id]

        # Apply task-specific parameters to model
        self._apply_task_parameters(task_profile.params)

        # Process task using the model
        # (Implementation would convert task_data to model input format)
        # result = self.model(...)

        # For now, return a placeholder
        return {"status": "success", "message": "Task processing not yet implemented"}

    def _apply_task_parameters(self, params: Dict[str, Any]) -> None:
        """Apply task-specific parameters to the model"""
        # Implementation would update model.reasoning_config with task-specific values
        pass

    def update_task_performance(self, task_id: str, metrics: Dict[str, float]) -> None:
        """Update performance metrics for a task"""
        if task_id not in self.task_profiles:
            logger.warning(f"Task {task_id} not registered. Creating profile.")
            self.register_task(task_id)

        self.task_profiles[task_id].update_performance(metrics)

    def adapt_task_parameters(
        self, task_id: str, adaptation_data: Dict[str, Any]
    ) -> None:
        """Adapt parameters for a specific task based on performance"""
        if task_id not in self.task_profiles:
            logger.warning(f"Task {task_id} not registered. Cannot adapt parameters.")
            return

        self.task_profiles[task_id].adapt_parameters(adaptation_data)

    def save_profiles(self, output_path: str) -> None:
        """Save all task profiles to disk"""
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        profiles_data = {
            task_id: profile.to_dict()
            for task_id, profile in self.task_profiles.items()
        }

        with open(output_path, "w") as f:
            json.dump(profiles_data, f, indent=2)

        logger.info(f"Saved {len(profiles_data)} task profiles to {output_path}")

    def load_profiles(self, input_path: str) -> None:
        """Load task profiles from disk"""
        if not os.path.exists(input_path):
            logger.warning(f"Profile file {input_path} not found.")
            return

        with open(input_path, "r") as f:
            profiles_data = json.load(f)

        for task_id, profile_data in profiles_data.items():
            self.task_profiles[task_id] = GridFormerTaskProfile.from_dict(profile_data)

        logger.info(f"Loaded {len(profiles_data)} task profiles from {input_path}")


if __name__ == "__main__":
    print("VantaCore Integration Module for Iterative Reasoning GridFormer")
    print("Ready for integration with VantaCore meta-learning system")
    print("See GridFormer_VantaCore_Integration_Plan.md for next steps")
