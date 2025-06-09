#!/usr/bin/env python
"""
vanta_core.py - Unified Cognitive Engine for VoxSigil

Integrates Meta-Learning, BLT Encoding, and Hybrid RAG Middleware
for adaptive, high-performance cognitive processing, especially for ARC tasks.
"""

import datetime  # Ensure datetime is imported
import hashlib  # Added import for hashlib
import json
import logging
import os
import random
import sys
import time
import traceback  # Added import for traceback
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable

import numpy as np

# --- BEGIN PYTHONPATH MODIFICATION ---
logger = logging.getLogger("VoxSigil.VantaCore")
# Corrected sys.path modification
_PROJECT_ROOT = r"C:\\Users\\16479\\Desktop\\Voxsigil"  # Define the absolute path to the project root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Clean up temporary variables from global namespace to avoid polluting it
if "_PROJECT_ROOT" in locals() or "_PROJECT_ROOT" in globals():
    del _PROJECT_ROOT
# --- END PYTHONPATH MODIFICATION ---

try:
    # Ensure calculate_file_hash is defined, even if it's a placeholder
    from tools.utilities.utils import (
        calculate_file_hash as imported_calculate_file_hash,  # Changed import source
    )

    # Create a wrapper to match the expected signature
    def calculate_file_hash(file_path: str) -> str:
        """Wrapper for imported calculate_file_hash to match expected signature."""
        try:
            from pathlib import Path

            # Convert string to Path if needed and call the imported function
            result = imported_calculate_file_hash(Path(file_path))
            return result if result is not None else "fallback_hash_error"
        except Exception as ex:
            logger.error(f"Error in calculate_file_hash wrapper for {file_path}: {ex}")
            return "fallback_hash_error"

    logger.info("Successfully imported calculate_file_hash from Voxsigil_Library.utils")
except ImportError as e:
    logger.warning(
        f"Could not import calculate_file_hash from Voxsigil_Library.utils. Error: {e}. Using local fallback."
    )

    def calculate_file_hash(file_path: str) -> str:
        """Fallback hash function if utils.calculate_file_hash is not available."""
        logger.debug(f"Using fallback calculate_file_hash for {file_path}")
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                hasher.update(f.read())  # Added this line
            return hasher.hexdigest()
        except Exception as ex:
            logger.error(f"Fallback calculate_file_hash failed for {file_path}: {ex}")
            return "fallback_hash_error"


# Import type hints for interfaces
from typing import Protocol, runtime_checkable

# Attempt to import supervisor and BLT components using absolute paths
# Assuming Voxsigil_Library is the top-level package in PYTHONPATH
try:
    # Import interfaces with aliases to avoid conflicts
    from Vanta.interfaces.supervisor_connector_interface import (
        BaseSupervisorConnector as _BaseSupervisorConnector,
    )
    from Vanta.interfaces.blt_encoder_interface import BaseBLTEncoder as _BaseBLTEncoder
    from Vanta.interfaces.hybrid_middleware_interface import (
        BaseHybridMiddleware as _BaseHybridMiddleware,
    )

    # Import implementations - corrected paths based on project structure
    from BLT.blt_encoder import BLTEncoder as _BLTEncoder
    from BLT.hybrid_middleware import HybridMiddleware as _HybridMiddleware

    # Import a suitable supervisor connector implementation
    # Using RealSupervisorConnector from real_supervisor_connector
    from Vanta.interfaces.real_supervisor_connector import (
        RealSupervisorConnector as _RealSupervisorConnector,
    )

    BLT_COMPONENTS_AVAILABLE = True
    logger.info(
        "Successfully imported Supervisor, BLT, and Middleware interfaces and implementations."
    )

    # Set the actual interface classes
    BaseSupervisorConnector = _BaseSupervisorConnector  # type: ignore[assignment]
    BaseBLTEncoder = _BaseBLTEncoder  # type: ignore[assignment]
    BaseHybridMiddleware = _BaseHybridMiddleware  # type: ignore[assignment]
    RealSupervisorConnector = _RealSupervisorConnector  # type: ignore[assignment]
    BLTEncoder = _BLTEncoder  # type: ignore[assignment]
    HybridMiddleware = _HybridMiddleware  # type: ignore[assignment]

except ImportError as e:
    BLT_COMPONENTS_AVAILABLE = False
    logger.error(
        f"Failed to import BLT components or their implementations: {e}. Using fallback classes."
    )
    # Log the current sys.path for debugging
    logger.debug(f"Current sys.path: {sys.path}")
    # Also log the directory of the current file
    logger.debug(f"Vanta_Core.py directory: {os.path.dirname(__file__)}")
    # Detailed traceback
    logger.debug(traceback.format_exc())

    # Define fallback interface classes with required methods as stubs
    @runtime_checkable
    class BaseSupervisorConnector(Protocol):
        def get_sigil_content_as_dict(
            self, sigil_ref: str
        ) -> Optional[Dict[str, Any]]: ...
        def get_sigil_content_as_text(self, sigil_ref: str) -> Optional[str]: ...
        def create_sigil(
            self,
            desired_sigil_ref: str,
            initial_content: Any,
            sigil_type: str,
            tags: Optional[List[str]] = None,
            related_sigils: Optional[List[str]] = None,
        ) -> Optional[str]: ...
        def store_sigil_content(
            self, sigil_ref: str, content: Any, content_type: str = "application/json"
        ) -> bool: ...
        def search_sigils(
            self, query_criteria: Dict[str, Any], max_results: Optional[int] = None
        ) -> List[Dict[str, Any]]: ...
        def get_module_health(self, registration_sigil_ref: str) -> Dict[str, Any]: ...
        def register_module_with_supervisor(
            self,
            module_name: str,
            module_capabilities: Dict[str, Any],
            requested_sigil_ref: Optional[str] = None,
        ) -> Optional[str]: ...
        def perform_health_check(self, module_registration_sigil_ref: str) -> bool: ...

    @runtime_checkable
    class BaseBLTEncoder(Protocol):
        def encode(
            self, text_content: str, task_type: str = "general"
        ) -> List[float]: ...
        def encode_batch(
            self, text_contents: List[str], task_type: str = "general"
        ) -> List[List[float]]: ...
        def get_encoder_details(self) -> Dict[str, Any]: ...

    @runtime_checkable
    class BaseHybridMiddleware(Protocol):
        def process_arc_task(
            self,
            input_data_sigil_ref: str,
            task_definition_sigil_ref: str,
            task_parameters: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Optional[str], Optional[str]]: ...
        def get_middleware_capabilities(self) -> Dict[str, Any]: ...

    # Fallback implementations that provide the required methods
    class BLTEncoder:
        def encode(self, text_content: str, task_type: str = "general") -> List[float]:
            logger.warning("Using fallback BLTEncoder.encode")
            return [0.0] * 128  # Return dummy embedding

        def encode_batch(
            self, text_contents: List[str], task_type: str = "general"
        ) -> List[List[float]]:
            logger.warning("Using fallback BLTEncoder.encode_batch")
            return [[0.0] * 128 for _ in text_contents]

        def get_encoder_details(self) -> Dict[str, Any]:
            return {"name": "FallbackBLTEncoder", "version": "0.0.1"}

    class HybridMiddleware:
        def process_arc_task(
            self,
            input_data_sigil_ref: str,
            task_definition_sigil_ref: str,
            task_parameters: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Optional[str], Optional[str]]:
            logger.warning("Using fallback HybridMiddleware.process_arc_task")
            return None, None

        def get_middleware_capabilities(self) -> Dict[str, Any]:
            return {"name": "FallbackHybridMiddleware", "version": "0.0.1"}

    class RealSupervisorConnector:
        def get_sigil_content_as_dict(self, sigil_ref: str) -> Optional[Dict[str, Any]]:
            logger.warning(
                "Using fallback RealSupervisorConnector.get_sigil_content_as_dict"
            )
            return {}

        def get_sigil_content_as_text(self, sigil_ref: str) -> Optional[str]:
            logger.warning(
                "Using fallback RealSupervisorConnector.get_sigil_content_as_text"
            )
            return None

        def create_sigil(
            self,
            desired_sigil_ref: str,
            initial_content: Any,
            sigil_type: str,
            tags: Optional[List[str]] = None,
            related_sigils: Optional[List[str]] = None,
        ) -> Optional[str]:
            logger.warning("Using fallback RealSupervisorConnector.create_sigil")
            return None

        def store_sigil_content(
            self, sigil_ref: str, content: Any, content_type: str = "application/json"
        ) -> bool:
            logger.warning("Using fallback RealSupervisorConnector.store_sigil_content")
            return False

        def search_sigils(
            self, query_criteria: Dict[str, Any], max_results: Optional[int] = None
        ) -> List[Dict[str, Any]]:
            logger.warning("Using fallback RealSupervisorConnector.search_sigils")
            return []

        def get_module_health(self, registration_sigil_ref: str) -> Dict[str, Any]:
            logger.warning("Using fallback RealSupervisorConnector.get_module_health")
            return {"status": "unknown"}

        def register_module_with_supervisor(
            self,
            module_name: str,
            module_capabilities: Dict[str, Any],
            requested_sigil_ref: Optional[str] = None,
        ) -> Optional[str]:
            logger.warning(
                "Using fallback RealSupervisorConnector.register_module_with_supervisor"
            )
            return None

        def perform_health_check(self, module_registration_sigil_ref: str) -> bool:
            logger.warning(
                "Using fallback RealSupervisorConnector.perform_health_check"
            )
            return False


# --- Pydantic Check ---
try:
    from pydantic import (  # Changed from validator to field_validator
        BaseModel,
        field_validator,
    )
except ImportError as e:
    logger.error(f"Pydantic is not installed or failed to import: {e}")
    raise


class ConfigModel(BaseModel):
    """
    Pydantic model for validating configuration structure.
    """

    default_learning_rate: float = 0.05
    default_exploration_rate: float = 0.1
    transfer_strength: float = 0.3
    parameter_damping_factor: float = 0.7
    similarity_threshold_for_transfer: float = 0.75
    max_performance_history_per_task: int = 50
    min_perf_points_for_adaptation: int = 5
    min_perf_points_for_global_opt: int = 10

    @field_validator("*", mode="before")  # Corrected syntax
    def check_positive(cls, v):
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError("Values must be positive")
        return v

    class Config:
        # schema_extra = { # Old V1 style
        json_schema_extra = {  # New V2 style
            "example": {
                "default_learning_rate": 0.01,
                "default_exploration_rate": 0.2,
                "transfer_strength": 0.5,
                "parameter_damping_factor": 0.8,
                "similarity_threshold_for_transfer": 0.7,
                "max_performance_history_per_task": 100,
                "min_perf_points_for_adaptation": 10,
                "min_perf_points_for_global_opt": 20,
            }
        }


# --- END Pydantic Check ---


class VantaCognitiveEngine:
    """
    VantaCognitiveEngine - Advanced Cognitive Processing Engine for VoxSigil.
    Integrates Meta-Learning, BLT Encoding, and Hybrid RAG Middleware.
    Specialized for AI/ML workloads and advanced cognitive tasks.
    """

    # Remove class-level type annotations for interfaces
    config_sigil_ref: str
    config: Dict[str, Any]
    meta_parameters: Dict[str, Any]
    task_adaptation_profiles: Dict[str, Dict[str, Any]]
    cross_task_knowledge_index: Dict[str, List[float]]
    supervisor_registration_sigil: Optional[str]  # Added type hint
    last_supervisor_health_check_status: Optional[str]  # Added type hint
    """"""

    def __init__(
        self,
        config_sigil_ref: str,
        supervisor_connector: BaseSupervisorConnector,
        blt_encoder: BaseBLTEncoder,
        hybrid_middleware: BaseHybridMiddleware,
    ):
        """
        Initialize VantaCore.

        Args:
            config_sigil_ref (str): Sigil reference to the configuration for VantaCore.
            supervisor_connector (BaseSupervisorConnector): An object providing an interface to VoxSigil Supervisor services.
            blt_encoder (BaseBLTEncoder): Pre-instantiated BLT encoder implementing BaseBLTEncoder interface.
            hybrid_middleware (BaseHybridMiddleware): Pre-instantiated middleware implementing BaseHybridMiddleware.
        """
        self.supervisor_connector = supervisor_connector
        self.config_sigil_ref = config_sigil_ref
        # Ensure config is loaded before it might be used by other initializations
        self.config = self._load_configuration(config_sigil_ref)

        logger.info(f"Initializing VantaCore with config from '{config_sigil_ref}'...")

        # --- Meta-Learning Kernel Attributes ---
        self.meta_parameters: Dict[str, Any] = {
            "default_learning_rate": self.config.get("default_learning_rate", 0.05),
            "default_exploration_rate": self.config.get(
                "default_exploration_rate", 0.1
            ),
            "transfer_strength": self.config.get("transfer_strength", 0.3),
            "parameter_damping_factor": self.config.get(
                "parameter_damping_factor", 0.7
            ),
            "similarity_threshold_for_transfer": self.config.get(
                "similarity_threshold_for_transfer", 0.75
            ),
            "max_performance_history_per_task": self.config.get(
                "max_performance_history_per_task", 50
            ),
            "min_perf_points_for_adaptation": self.config.get(
                "min_perf_points_for_adaptation", 5
            ),
            "min_perf_points_for_global_opt": self.config.get(
                "min_perf_points_for_global_opt", 10
            ),
        }
        self.task_adaptation_profiles: Dict[str, Dict[str, Any]] = {}
        self.cross_task_knowledge_index: Dict[str, List[float]] = {}

        # --- Integrated Component Instances ---
        # BLT encoder and Hybrid Middleware components are now mandatory and type-checked by constructor signature
        self.blt_encoder = blt_encoder
        logger.info("Using provided BLT Encoder")

        self.hybrid_middleware = hybrid_middleware
        logger.info("Using provided Hybrid Middleware")

        self.supervisor_registration_sigil = None  # Initialize attribute
        self.last_supervisor_health_check_status = "pending"  # Initialize attribute

        logger.info(f"VantaCore initialized. Meta-parameters: {self.meta_parameters}")
        logger.info("BLT Encoder and Hybrid Middleware interfaces are set up.")

        self.initialize_supervisor_integration()

    def _load_configuration(self, config_sigil_ref: str) -> Dict[str, Any]:
        """Loads configuration from a VoxSigil definition."""
        try:
            config_content = self.supervisor_connector.get_sigil_content_as_dict(
                config_sigil_ref
            )
            if not config_content:
                logger.error(
                    f"VantaCore config sigil '{config_sigil_ref}' is empty or not found. Using defaults."
                )
                return {}
            logger.info(
                f"VantaCore configuration loaded successfully from '{config_sigil_ref}'."
            )
            # Assuming config structure might have a specific key for VantaCore settings
            return config_content.get(
                "vanta_core_settings",
                config_content.get(
                    "custom_attributes_vanta_extensions", config_content
                ),
            )
        except Exception as e:
            logger.error(
                f"Failed to load VantaCore configuration from '{config_sigil_ref}': {e}. Using defaults.",
                exc_info=True,
            )
            return {}

    def perform_supervisor_health_check(self) -> None:
        """Performs a health check with the supervisor and updates status."""
        if not self.supervisor_connector or not self.supervisor_registration_sigil:
            logger.warning(
                "Cannot perform supervisor health check: Supervisor not connected or VantaCore not registered."
            )
            self.last_supervisor_health_check_status = "degraded"
            return

        try:
            # In a real scenario, this would involve a call to a supervisor endpoint
            # For mock purposes, we'll simulate a successful health check
            health_status = self.supervisor_connector.get_module_health(
                self.supervisor_registration_sigil
            )

            if health_status and health_status.get("status") == "healthy":
                self.last_supervisor_health_check_status = "healthy"
                logger.info(
                    f"Supervisor health check successful for {self.supervisor_registration_sigil}. Status: healthy"
                )
            else:
                self.last_supervisor_health_check_status = "degraded"
                logger.warning(
                    f"Supervisor health check indicated a problem for {self.supervisor_registration_sigil}. Status: {health_status.get('status', 'unknown') if health_status else 'unknown'}"
                )

        except Exception as e:
            logger.error(
                f"Error during supervisor health check: {e}", exc_info=True
            )  # Removed extra backslash
            self.last_supervisor_health_check_status = "error"

    def process_input(
        self,
        input_data_sigil_ref: str,
        task_sigil_ref: str,
        task_description_sigil_ref: Optional[str] = None,
    ) -> Optional[str]:
        """
        Primary entry point for VantaCore to process an ARC task or similar input.
        Handles task registration, parameter adaptation, task execution via middleware, and performance update.

        Args:
            input_data_sigil_ref (str): Sigil ref to the input data for the task.
            task_sigil_ref (str): Sigil ref representing the task itself (e.g., an ARC problem definition).
            task_description_sigil_ref (Optional[str]): Optional sigil ref for the task description.
                If not provided, VantaCore will attempt to retrieve it from task metadata.

        Returns:
            Optional[str]: Sigil ref to the output/solution, or None on failure.
        """
        start_time = time.monotonic()
        logger.info(
            f"VantaCore processing input '{input_data_sigil_ref}' for task '{task_sigil_ref}'..."
        )

        # Validate input parameters
        if not input_data_sigil_ref or not task_sigil_ref:
            logger.error(
                "Invalid input: input_data_sigil_ref and task_sigil_ref must be non-empty"
            )
            return None

        # Check if task is registered, if not, try to register it
        if task_sigil_ref not in self.task_adaptation_profiles:
            # Try to get the task description sigil from metadata if not provided
            if not task_description_sigil_ref:
                try:
                    # Try to get information about the task from the supervisor
                    task_metadata = (
                        self.supervisor_connector.get_sigil_content_as_dict(
                            task_sigil_ref
                        )
                        or {}
                    )  # Ensure a dict even if the supervisor returns None
                    task_description_sigil_ref = task_metadata.get(
                        "description_sigil_ref"
                    )

                    if not task_description_sigil_ref:
                        # If no description sigil is found in metadata, use a fallback
                        task_description_sigil_ref = (
                            f"SigilRef:Descriptor_For_{task_sigil_ref}"
                        )
                        logger.warning(
                            f"No description sigil found in task metadata. Using fallback: '{task_description_sigil_ref}'"
                        )
                except Exception as e:
                    task_description_sigil_ref = (
                        f"SigilRef:Descriptor_For_{task_sigil_ref}"
                    )
                    logger.warning(
                        f"Error retrieving task metadata: {e}. Using fallback descriptor: '{task_description_sigil_ref}'"
                    )

            logger.info(
                f"Task '{task_sigil_ref}' not yet registered. Registering with descriptor '{task_description_sigil_ref}'."
            )
            self.register_arc_task_profile(task_sigil_ref, task_description_sigil_ref)

        # Get adapted parameters for this task based on learning history
        adapted_parameters = self.get_adapted_parameters_for_task(task_sigil_ref)

        # Prepare execution parameters with adaptive exploration and learning settings
        task_execution_params = adapted_parameters.copy()

        # Scale exploration rate to effective parameters the middleware can use
        exploration_rate = adapted_parameters.get(
            "exploration_rate", self.meta_parameters["default_exploration_rate"]
        )
        learning_rate = adapted_parameters.get(
            "learning_rate", self.meta_parameters["default_learning_rate"]
        )

        # Map exploration_rate [0-1] to temperature [0.1-0.9] for LLM sampling
        task_execution_params["effective_temperature"] = 0.1 + (exploration_rate * 0.8)

        # More attempts for exploration, but limit to reasonable range
        task_execution_params["max_solution_attempts"] = max(
            1, min(5, 1 + int(exploration_rate * 4))
        )

        # Add learning context for middleware to understand adaptation history
        task_execution_params["learning_context"] = {
            "learning_rate": learning_rate,
            "exploration_rate": exploration_rate,
            "adaptation_count": len(
                self.task_adaptation_profiles.get(task_sigil_ref, {}).get(
                    "performance_history", []
                )
            ),
            "is_transfer_learning_applied": task_sigil_ref
            in self.cross_task_knowledge_index,
        }

        logger.debug(
            f"Executing task '{task_sigil_ref}' with adapted parameters: {task_execution_params}"
        )

        try:
            # Execute the task using the HybridMiddleware
            output_solution_sigil_ref, perf_metric_sigil_ref = (
                self.hybrid_middleware.process_arc_task(
                    input_data_sigil_ref,
                    task_sigil_ref,
                    task_parameters=task_execution_params,
                )
            )

            # Process performance metrics if available
            if perf_metric_sigil_ref:
                # Update task performance to drive future adaptation
                self.update_task_performance(task_sigil_ref, perf_metric_sigil_ref)

                # Log performance details
                try:
                    perf_data = (
                        self.supervisor_connector.get_sigil_content_as_dict(
                            perf_metric_sigil_ref
                        )
                        or {}
                    )
                    perf_value = perf_data.get("achieved_performance")
                    if perf_value is not None:
                        logger.info(
                            f"Task '{task_sigil_ref}' performance: {perf_value:.4f}"
                        )
                except Exception as e:
                    logger.warning(f"Error retrieving performance data: {e}")
            else:
                logger.warning(
                    f"No performance metric sigil returned from middleware for task '{task_sigil_ref}'. Cannot update performance."
                )

            # Return the solution sigil, or handle failure
            if output_solution_sigil_ref:
                duration = time.monotonic() - start_time
                logger.info(
                    f"Task '{task_sigil_ref}' completed in {duration:.2f}s. Solution sigil: '{output_solution_sigil_ref}'"
                )
                return output_solution_sigil_ref
            else:
                logger.error(
                    f"Task '{task_sigil_ref}' processing by middleware did not yield a solution sigil."
                )
                # Log a failure performance metric with specific error type
                failure_sigil = "SigilRef:PerfMetric_Failure_NoSolutionGenerated"
                self.update_task_performance(task_sigil_ref, failure_sigil)
                return None

        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error(
                f"Exception during VantaCore process_input for task '{task_sigil_ref}' after {duration:.2f}s: {e}",
                exc_info=True,
            )
            # Record the specific exception type to help with adaptation
            failure_sigil = "SigilRef:PerfMetric_Failure_VantaCoreException"
            self.update_task_performance(task_sigil_ref, failure_sigil)
            return None

    def register_arc_task_profile(
        self, task_sigil_ref: str, task_description_sigil_ref: str
    ) -> None:
        """
        Registers a new ARC task type or instance, initializes its adaptation profile,
        and stores its semantic embedding for similarity calculations.
        """
        if task_sigil_ref in self.task_adaptation_profiles:
            logger.debug(f"Task profile for '{task_sigil_ref}' already registered.")
            return

        # Initialize profile with default meta-parameters
        self.task_adaptation_profiles[task_sigil_ref] = {
            "description_sigil_ref": task_description_sigil_ref,
            "parameters": {
                "learning_rate": self.meta_parameters["default_learning_rate"],
                "exploration_rate": self.meta_parameters["default_exploration_rate"],
                # Other task-specific tunable parameters could be initialized here
            },
            "performance_history": [],
            "last_adapted_timestamp": time.monotonic(),
        }
        logger.info(
            f"Registered new task profile: '{task_sigil_ref}' linked to description '{task_description_sigil_ref}'."
        )

        # Store semantic embedding of the task description for similarity
        try:  # Use supervisor_connector to get the text content of the description sigil
            description_content = self.supervisor_connector.get_sigil_content_as_text(
                task_description_sigil_ref
            )

            if description_content:
                # Use blt_encoder to encode the description content
                embedding = self.blt_encoder.encode(
                    text_content=description_content, task_type="arc_task_description"
                )
                self.cross_task_knowledge_index[task_sigil_ref] = embedding
                logger.info(
                    f"Stored BLT embedding for task description '{task_description_sigil_ref}'. Embedding dim: {len(embedding)}"
                )
            else:
                logger.warning(
                    f"Could not retrieve content for task description sigil '{task_description_sigil_ref}'. Embedding not stored."
                )
        except Exception as e:
            logger.error(
                f"Failed to encode/store embedding for task description '{task_description_sigil_ref}': {e}",
                exc_info=True,
            )

        # Attempt initial knowledge transfer from similar tasks
        similar_tasks = self._find_similar_task_profiles(task_sigil_ref)
        if similar_tasks:
            self._transfer_knowledge(task_sigil_ref, similar_tasks)
        else:
            logger.info(
                f"No sufficiently similar tasks found for initial knowledge transfer to '{task_sigil_ref}'."
            )

    def update_task_performance(
        self, task_sigil_ref: str, performance_metric_sigil_ref: str
    ) -> None:
        """
        Updates the performance history for a given task and triggers parameter adaptation.
        """
        if task_sigil_ref not in self.task_adaptation_profiles:
            logger.warning(
                f"Attempted to update performance for unregistered task '{task_sigil_ref}'. Please register first."
            )
            return

        profile = self.task_adaptation_profiles[task_sigil_ref]

        try:
            # Assume performance_metric_sigil_ref contains a field like "achieved_performance" (0-1)
            sigil_entry = self.supervisor_connector.get_sigil_content_as_dict(
                performance_metric_sigil_ref
            )
            metric_content = (
                sigil_entry.get("content", {}) if sigil_entry else {}
            )  # Safely get content

            if not metric_content or "achieved_performance" not in metric_content:
                # This is a critical part: handling failure sigils or bad data
                logger.error(
                    f"Invalid or missing performance metric data in sigil '{performance_metric_sigil_ref}'. Full entry: {sigil_entry}"
                )
                # Check if the sigil_ref itself indicates a known failure type
                if performance_metric_sigil_ref.startswith(
                    "SigilRef:PerfMetric_Failure"
                ):
                    performance_value = (
                        0.0  # Assign low performance for known failure signals
                    )
                    logger.info(
                        f"Recognized failure sigil '{performance_metric_sigil_ref}', assigning performance 0.0."
                    )
                else:
                    # If not a known failure string and content is bad, skip update
                    logger.warning(
                        f"Skipping performance update for task '{task_sigil_ref}' due to bad metric sigil and unknown failure type."
                    )
                    return
            else:
                performance_value = float(metric_content["achieved_performance"])

        except Exception as e:
            logger.error(
                f"Failed to parse performance metric from sigil '{performance_metric_sigil_ref}': {e}"
            )
            # If parsing fails, also check if it's a known failure sigil string
            if performance_metric_sigil_ref.startswith("SigilRef:PerfMetric_Failure"):
                performance_value = 0.0  # Assign low performance for failure signals
                logger.info(
                    f"Recognized failure sigil '{performance_metric_sigil_ref}' after parsing error, assigning performance 0.0."
                )
            else:
                # If parsing fails and it's not a known failure string, skip update
                logger.warning(
                    f"Skipping performance update for task '{task_sigil_ref}' due to parsing error and unknown sigil type."
                )
                return

        profile["performance_history"].append(
            {
                "value": performance_value,
                "parameters_used": profile[
                    "parameters"
                ].copy(),  # Parameters active when this performance was achieved
                "metric_sigil_ref": performance_metric_sigil_ref,
                "timestamp": time.monotonic(),
            }
        )

        # Trim history
        if (
            len(profile["performance_history"])
            > self.meta_parameters["max_performance_history_per_task"]
        ):
            profile["performance_history"] = profile["performance_history"][
                -self.meta_parameters["max_performance_history_per_task"] :
            ]

        logger.info(
            f"Performance updated for task '{task_sigil_ref}': {performance_value:.3f}. History size: {len(profile['performance_history'])}"
        )
        self._adapt_task_parameters(task_sigil_ref)

    def _adapt_task_parameters(self, task_sigil_ref: str) -> None:
        """
        Adapts task-specific parameters (learning_rate, exploration_rate)
        based on its performance history.
        """
        profile = self.task_adaptation_profiles.get(task_sigil_ref)
        if not profile or not profile.get(
            "performance_history"
        ):  # Added check for performance_history existence
            logger.debug(
                f"Task profile for '{task_sigil_ref}' not found or has no performance history. Skipping adaptation."
            )
            return

        if (
            len(profile["performance_history"])
            < self.meta_parameters["min_perf_points_for_adaptation"]
        ):
            logger.debug(
                f"Not enough performance data for task '{task_sigil_ref}' to adapt parameters (need {self.meta_parameters['min_perf_points_for_adaptation']}, have {len(profile['performance_history'])})."
            )
            return

        performances = profile["performance_history"]
        current_params = profile["parameters"]

        # Use a consistent number of recent points for metrics.
        # Consider at least min_perf_points_for_adaptation or a small number (e.g., 5), up to available history.
        min_points_for_stats = self.meta_parameters.get(
            "min_perf_points_for_adaptation", 3
        )  # Default to 3 if not in meta_params
        num_points_to_consider = min(len(performances), max(min_points_for_stats, 5))
        recent_values = [p["value"] for p in performances[-num_points_to_consider:]]

        if not recent_values:  # Should not happen if len(profile["performance_history"]) check passed, but as a safeguard
            logger.warning(
                f"No recent values found for task '{task_sigil_ref}' despite having performance history. Skipping adaptation."
            )
            return

        avg_perf = np.mean(recent_values)

        trend_improving = False
        # Refined trend calculation:
        if len(recent_values) >= 4:
            mid_point = len(recent_values) // 2
            # Ensure there are enough elements in both halves for mean calculation
            # And that mid_point is not such that it creates empty slices for very small len(recent_values)
            if (
                mid_point > 0 and (len(recent_values) - mid_point) > 0
            ):  # Ensure both slices are non-empty
                first_half_mean = np.mean(recent_values[:mid_point])
                second_half_mean = np.mean(recent_values[mid_point:])
                if second_half_mean > first_half_mean:
                    trend_improving = True
        elif (
            len(recent_values) >= 2
        ):  # Simpler trend for 2 or 3 points (last vs second last)
            if (
                recent_values[-1] > recent_values[-2]
            ):  # Check if last is greater than second to last
                trend_improving = True

        variance = np.var(recent_values) if len(recent_values) > 1 else 0.0
        new_lr = current_params["learning_rate"]
        new_er = current_params["exploration_rate"]

        # Adaptation logic based on performance and trend
        if avg_perf > 0.8 and variance < 0.01:  # High and stable
            new_er = max(0.01, new_er * 0.9)
            new_lr = max(0.01, new_lr * 0.95)
        elif avg_perf < 0.4 and variance < 0.05:  # Low and stable (stuck)
            new_er = min(0.9, new_er * 1.2 + 0.05)
            new_lr = min(0.2, new_lr * 1.1 + 0.02)
        elif trend_improving:
            new_er = min(
                0.7, new_er * 1.05
            )  # Slightly more exploration to find peak or continue improvement
            if avg_perf > 0.6:  # If improving and already good
                new_lr = max(0.01, new_lr * 0.98)  # Slightly decrease LR to stabilize
            else:  # If improving but still low/mid
                new_lr = min(0.2, new_lr * 1.02)  # Maintain or slightly increase LR
        elif (
            len(recent_values) >= 2 and not trend_improving
        ):  # Declining or stuck fluctuating (and we have at least 2 points to make this call)
            new_er = min(0.9, new_er * 1.15 + 0.03)  # Try exploring more
            new_lr = min(
                0.2, new_lr * 1.05 + 0.01
            )  # Learn a bit faster from new exploration

        # Ensure parameters stay within reasonable bounds
        current_params["learning_rate"] = round(max(0.01, min(0.2, new_lr)), 4)
        current_params["exploration_rate"] = round(max(0.01, min(0.9, new_er)), 4)
        profile["last_adapted_timestamp"] = time.monotonic()

        logger.info(
            f"Adapted parameters for task '{task_sigil_ref}': LR={current_params['learning_rate']:.4f}, ER={current_params['exploration_rate']:.4f} (AvgPerf: {avg_perf:.3f}, Var: {variance:.4f}, TrendImproving: {trend_improving}, PointsConsidered: {len(recent_values)})"
        )

    def _calculate_profile_similarity(
        self, task_sigil_ref1: str, task_sigil_ref2: str
    ) -> float:
        """Calculates semantic similarity between two task profiles based on their description embeddings."""
        emb1 = self.cross_task_knowledge_index.get(task_sigil_ref1)
        emb2 = self.cross_task_knowledge_index.get(task_sigil_ref2)

        if emb1 is None or emb2 is None:
            # logger.debug(f"Missing embedding for similarity: {task_sigil_ref1 if emb1 is None else ''} {task_sigil_ref2 if emb2 is None else ''}")
            return 0.0

        # Cosine similarity
        emb1_arr = np.array(emb1)
        emb2_arr = np.array(emb2)
        dot_product = np.dot(emb1_arr, emb2_arr)
        norm1 = np.linalg.norm(emb1_arr)
        norm2 = np.linalg.norm(emb2_arr)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(max(0.0, min(1.0, similarity)))  # Ensure value is between 0 and 1

    def _find_similar_task_profiles(
        self, current_task_sigil_ref: str
    ) -> List[Tuple[str, float]]:
        """Finds tasks similar to the current one using BLT embeddings of their descriptions."""
        similarities = []
        if current_task_sigil_ref not in self.cross_task_knowledge_index:
            logger.warning(
                f"Cannot find similar tasks for '{current_task_sigil_ref}', its embedding is missing."
            )
            return []

        for task_sigil_ref_other, _ in self.cross_task_knowledge_index.items():
            if task_sigil_ref_other == current_task_sigil_ref:
                continue

            similarity = self._calculate_profile_similarity(
                current_task_sigil_ref, task_sigil_ref_other
            )
            if similarity >= self.meta_parameters["similarity_threshold_for_transfer"]:
                similarities.append((task_sigil_ref_other, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            f"Found {len(similarities)} similar tasks for '{current_task_sigil_ref}' above threshold {self.meta_parameters['similarity_threshold_for_transfer']}."
        )
        return similarities

    def _transfer_knowledge(
        self,
        target_task_sigil_ref: str,
        source_task_profiles_with_similarity: List[Tuple[str, float]],
    ) -> None:
        """
        Transfers (initializes/biases) parameters for the target task
        based on a weighted average of parameters from similar source tasks.
        This is typically called when a task is newly registered or performing poorly.
        """
        target_profile = self.task_adaptation_profiles.get(target_task_sigil_ref)
        if not target_profile:
            logger.error(
                f"Cannot transfer knowledge: Target task '{target_task_sigil_ref}' profile not found."
            )
            return
        if not source_task_profiles_with_similarity:
            logger.info(
                f"No source tasks provided for knowledge transfer to '{target_task_sigil_ref}'."
            )
            return

        weighted_sum_params: Dict[str, float] = {}
        total_similarity_weight: float = 0.0

        # Consider only top N (e.g., 3) most similar tasks for transfer
        for source_task_sigil_ref, similarity in source_task_profiles_with_similarity[
            :3
        ]:
            source_profile = self.task_adaptation_profiles.get(source_task_sigil_ref)
            if source_profile and source_profile["parameters"]:
                for param_name, param_value in source_profile["parameters"].items():
                    if isinstance(
                        param_value, (int, float)
                    ):  # Only transfer numerical params
                        weighted_sum_params[param_name] = weighted_sum_params.get(
                            param_name, 0.0
                        ) + (param_value * similarity)
                total_similarity_weight += similarity

        if total_similarity_weight == 0:
            logger.info(
                f"Total similarity weight is zero, cannot perform weighted transfer to '{target_task_sigil_ref}'."
            )
            return

        # Apply transfer with damping (transfer_strength from meta_parameters)
        transfer_strength = self.meta_parameters["transfer_strength"]
        updated_params_for_log = {}
        for param_name, summed_weighted_value in weighted_sum_params.items():
            avg_weighted_value = summed_weighted_value / total_similarity_weight
            original_value = target_profile["parameters"].get(
                param_name,
                self.meta_parameters.get(f"default_{param_name}", avg_weighted_value),
            )

            transferred_value = (
                1.0 - transfer_strength
            ) * original_value + transfer_strength * avg_weighted_value
            target_profile["parameters"][param_name] = round(transferred_value, 4)
            updated_params_for_log[param_name] = target_profile["parameters"][
                param_name
            ]

        if updated_params_for_log:
            logger.info(
                f"Knowledge transferred to task '{target_task_sigil_ref}'. New parameters: {updated_params_for_log} from sources: {[s[0][:20] for s in source_task_profiles_with_similarity[:3]]}"
            )
        else:
            logger.info(
                f"No parameters updated via knowledge transfer for task '{target_task_sigil_ref}'."
            )

    def _optimize_global_meta_parameters(self) -> None:
        """
        Periodically called to adjust global meta-parameters based on the
        effectiveness of different parameter settings across all tasks.
        """
        logger.info("Attempting to optimize global meta-parameters...")
        tasks_contributing = 0
        # Stores {param_name: [list_of_best_values_from_tasks]}
        best_param_values_across_tasks: Dict[str, List[float]] = {
            "learning_rate": [],
            "exploration_rate": [],  # Add other meta-tunable task params if any
        }

        for task_sigil_ref, profile in self.task_adaptation_profiles.items():
            if (
                len(profile["performance_history"])
                < self.meta_parameters["min_perf_points_for_global_opt"]
            ):
                continue

            tasks_contributing += 1
            # For each task, find parameter settings that correlated with its top N performances
            # This is a simplification; true attribution is complex.
            # Sort history by performance
            sorted_history = sorted(
                profile["performance_history"], key=lambda x: x["value"], reverse=True
            )

            # Consider top 20% of performances or top K (e.g. 5)
            top_performances = sorted_history[: max(1, int(len(sorted_history) * 0.2))]

            for perf_entry in top_performances:
                params_used = perf_entry["parameters_used"]
                if "learning_rate" in params_used:
                    best_param_values_across_tasks["learning_rate"].append(
                        params_used["learning_rate"]
                    )
                if "exploration_rate" in params_used:
                    best_param_values_across_tasks["exploration_rate"].append(
                        params_used["exploration_rate"]
                    )

        if tasks_contributing < 3:  # Need data from a few tasks at least
            logger.info(
                f"Not enough tasks ({tasks_contributing}) with sufficient data for global meta-parameter optimization."
            )
            return

        damping = self.meta_parameters["parameter_damping_factor"]
        updated_meta_params_log = {}

        for param_name, value_list in best_param_values_across_tasks.items():
            if value_list:
                # Use median or mean of these "good" values
                # Median is more robust to outliers
                globally_effective_value = float(np.median(value_list))

                old_meta_value = self.meta_parameters.get(
                    f"default_{param_name}", globally_effective_value
                )  # Use default_ if exists
                new_meta_value = (old_meta_value * damping) + (
                    globally_effective_value * (1.0 - damping)
                )

                # Update the *default* meta-parameters
                self.meta_parameters[f"default_{param_name}"] = round(new_meta_value, 4)
                updated_meta_params_log[f"default_{param_name}"] = self.meta_parameters[
                    f"default_{param_name}"
                ]

        if updated_meta_params_log:
            logger.info(
                f"Global meta-parameters optimized based on {tasks_contributing} tasks. New defaults: {updated_meta_params_log}"
            )
            # logger.debug(f"Collected best param values for optimization: {best_param_values_across_tasks}")
        else:
            logger.info(
                "No updates made to global meta-parameters during this optimization cycle."
            )

    def get_adapted_parameters_for_task(self, task_sigil_ref: str) -> Dict[str, Any]:
        """
        Returns the current adapted parameters for a specific task.
        If the task is not registered or has no specific adapted parameters,
        it returns the global default meta-parameters.
        """
        profile = self.task_adaptation_profiles.get(task_sigil_ref)
        if profile and profile["parameters"]:
            return profile["parameters"].copy()
        else:
            # Return defaults if task not found or no params
            return {
                "learning_rate": self.meta_parameters["default_learning_rate"],
                "exploration_rate": self.meta_parameters["default_exploration_rate"],
                # Other parameters that tasks might have, initialized to a sensible default
            }

    def get_status_report(self) -> Dict[str, Any]:
        """Returns a comprehensive status report of the VantaCore Meta-Learning subsystem."""
        report = {
            "vanta_core_config_ref": self.config_sigil_ref,
            "meta_parameters": self.meta_parameters.copy(),
            "num_task_profiles_registered": len(self.task_adaptation_profiles),
            "num_tasks_in_knowledge_index": len(self.cross_task_knowledge_index),
            "task_profile_summary": [],
        }
        for task_sigil, profile in self.task_adaptation_profiles.items():
            avg_perf = (
                np.mean([p["value"] for p in profile["performance_history"]])
                if profile["performance_history"]
                else None
            )
            report["task_profile_summary"].append(
                {
                    "task_sigil_ref": task_sigil,
                    "current_lr": profile["parameters"].get("learning_rate"),
                    "current_er": profile["parameters"].get("exploration_rate"),
                    "num_perf_points": len(profile["performance_history"]),
                    "avg_perf": round(avg_perf, 3) if avg_perf is not None else None,
                    "last_adapted_ago_sec": round(
                        time.monotonic() - profile["last_adapted_timestamp"], 1
                    )
                    if profile.get("last_adapted_timestamp")
                    else None,
                    "has_embedding": task_sigil in self.cross_task_knowledge_index,
                }
            )
        return report

    # --- Supervisor Integration Methods (Milestone 3.3) ---

    def register_with_supervisor(self) -> bool:  # Return boolean for success
        """Registers VantaCore with the Supervisor and performs an initial health check."""
        if not self.supervisor_connector:
            logger.warning(
                "Supervisor connector not available. Cannot register VantaCore."
            )
            self.supervisor_registration_sigil = None
            self.last_supervisor_health_check_status = "degraded"
            return False  # Indicate failure

        try:
            module_name = self.config.get(
                "module_name", "VantaCoreDemoInstance"
            )  # Changed default for clarity
            version = self.config.get("version", "1.0.0-demo")

            capabilities = {
                "name": module_name,
                "version": version,
                "description": "Adaptive Meta-Learning Engine for ARC Tasks (Demo Instance)",
                "managed_task_types": ["ARC"],
                "supports_dynamic_configuration": True,
                "supports_knowledge_transfer": True,
                "interfaces": {
                    "blt_encoder": self.blt_encoder.get_encoder_details()
                    if self.blt_encoder
                    else "N/A",
                    "hybrid_middleware": self.hybrid_middleware.get_middleware_capabilities()
                    if self.hybrid_middleware
                    else "N/A",
                },
            }

            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            desired_reg_sigil = f"SigilRef:{module_name}_Registration_{timestamp}"

            registration_data = {
                "module_name": module_name,
                "version": version,
                "capabilities": capabilities,
                "timestamp": time.time(),
                "status": "pending_registration",
            }

            created_sigil_ref = self.supervisor_connector.create_sigil(
                desired_sigil_ref=desired_reg_sigil,
                initial_content=registration_data,
                sigil_type="system_component_registration",
            )

            if created_sigil_ref:
                self.supervisor_registration_sigil = created_sigil_ref
                logger.info(
                    f"VantaCore successfully initiated registration with Supervisor. Registration Sigil: {self.supervisor_registration_sigil}"
                )

                registration_data["status"] = "registered"
                self.supervisor_connector.store_sigil_content(
                    sigil_ref=self.supervisor_registration_sigil,
                    content=registration_data,
                    content_type="application/json",
                )
                logger.info(
                    f"Updated registration sigil {self.supervisor_registration_sigil} content with status 'registered'."
                )
                # Removed direct call to perform_supervisor_health_check here
                return True  # Indicate success
            else:
                logger.error(
                    f"Failed to create registration sigil with Supervisor for {module_name}."
                )
                self.supervisor_registration_sigil = None
                self.last_supervisor_health_check_status = "degraded"
                return False  # Indicate failure

        except Exception as e:
            logger.error(f"Error during supervisor registration: {e}")
            logger.debug(traceback.format_exc())
            self.supervisor_registration_sigil = None
            self.last_supervisor_health_check_status = "degraded"
            if "unexpected keyword argument" in str(e):  # Simplified check
                logger.error(
                    "This might be due to an outdated call signature for supervisor_connector.create_sigil()."
                )
            return False  # Indicate failure

    def validate_supervisor_health(self) -> Dict[str, Any]:
        """
        Validates that the supervisor connector is functioning properly.
        Performs a series of health checks on the supervisor connection.

        Returns:
            Dict[str, Any]: Health check results with status information
        """
        health_report = {
            "status": "unknown",
            "checks_passed": 0,
            "checks_failed": 0,
            "timestamp": time.time(),  # Corrected this section
            "details": {},
        }

        try:
            # Check 1: Can we retrieve our config?
            try:
                config = self.supervisor_connector.get_sigil_content_as_dict(
                    self.config_sigil_ref
                )
                if config:
                    health_report["details"]["config_retrieval"] = "passed"
                    health_report["checks_passed"] += 1
                else:
                    health_report["details"]["config_retrieval"] = (
                        "failed - empty config"
                    )
                    health_report["checks_failed"] += 1
            except Exception as e:
                health_report["details"]["config_retrieval"] = f"failed - {str(e)}"
                health_report["checks_failed"] += 1

            # Check 2: Can we create a test sigil?
            test_sigil_ref = f"SigilRef:VantaCore_HealthCheck_{int(time.time())}"
            try:
                success = self.supervisor_connector.create_sigil(
                    desired_sigil_ref=test_sigil_ref,  # Corrected: sigil_ref to desired_sigil_ref
                    initial_content={
                        "health_check": "test",
                        "timestamp": time.time(),
                    },  # Corrected: content to initial_content
                    sigil_type="health_check",  # Corrected: type_tag to sigil_type
                )
                if success:
                    health_report["details"]["sigil_creation"] = "passed"
                    health_report["checks_passed"] += 1
                else:
                    health_report["details"]["sigil_creation"] = (
                        "failed - creation returned False"
                    )
                    health_report["checks_failed"] += 1
            except Exception as e:
                health_report["details"]["sigil_creation"] = f"failed - {str(e)}"
                health_report["checks_failed"] += 1

            # Check 3: Can we search for sigils?
            try:
                results = self.supervisor_connector.search_sigils(
                    {"prefix": "SigilRef:Vanta"}
                )
                if results is not None:
                    health_report["details"]["sigil_search"] = (
                        f"passed - found {len(results)} results"
                    )
                    health_report["checks_passed"] += 1
                else:
                    health_report["details"]["sigil_search"] = (
                        "failed - search returned None"
                    )
                    health_report["checks_failed"] += 1
            except Exception as e:
                health_report["details"]["sigil_search"] = f"failed - {str(e)}"
                health_report["checks_failed"] += 1

            # Overall status
            if health_report["checks_failed"] == 0:
                health_report["status"] = "healthy"
            elif health_report["checks_passed"] > 0:
                health_report["status"] = "degraded"
            else:
                health_report["status"] = "critical"

            return health_report

        except Exception as e:
            logger.error(f"Error during supervisor health check: {e}", exc_info=True)
            return {
                "status": "critical",
                "checks_passed": 0,
                "checks_failed": 1,
                "timestamp": time.time(),
                "details": {"error": str(e)},
            }

    def update_supervisor_status(self) -> bool:
        """
        Updates the VantaCore status information in the supervisor.
        This is typically called periodically to keep the supervisor informed
        of the current state and performance of VantaCore.

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            supervisor_integration = self.config.get("supervisor_integration", {})
            if not supervisor_integration.get("register_status_updates", True):
                return False

            # Get current status report
            status_report = self.get_status_report()

            # Add additional supervisor-specific information
            status_update = {
                "component_type": "vanta_core",
                "status": "active",
                "timestamp": time.time(),
                "task_count": len(self.task_adaptation_profiles),
                "performance_summary": {
                    "total_tasks_processed": sum(
                        len(profile["performance_history"])
                        for profile in self.task_adaptation_profiles.values()
                    ),
                    "average_performance": np.mean(
                        [
                            np.mean(
                                [p["value"] for p in profile["performance_history"]]
                            )
                            for profile in self.task_adaptation_profiles.values()
                            if profile["performance_history"]
                        ]
                    )
                    if self.task_adaptation_profiles
                    else 0.0,
                },
                "detailed_status": status_report,
            }  # Create or update status sigil
            status_sigil_ref = f"SigilRef:VantaCore_Status_{int(time.time())}"
            success = self.supervisor_connector.create_sigil(
                desired_sigil_ref=status_sigil_ref,
                initial_content=status_update,
                sigil_type="component_status",
            )

            if success:
                logger.debug(
                    f"VantaCore status updated successfully: {status_sigil_ref}"
                )
                return True
            else:
                logger.warning("Failed to update VantaCore status in supervisor")
                return False

        except Exception as e:
            logger.error(f"Error updating supervisor status: {e}", exc_info=True)
            return False

    def validate_sigil_reference(self, sigil_ref: str) -> bool:
        """
        Validates that a sigil reference exists and is accessible.

        Args:
            sigil_ref: The sigil reference to validate

        Returns:
            bool: True if the sigil exists and is accessible, False otherwise
        """
        if not sigil_ref or not isinstance(sigil_ref, str):
            return False

        supervisor_integration = self.config.get("supervisor_integration", {})
        if not supervisor_integration.get("validate_sigil_refs", True):
            # Skip validation if disabled
            return True

        try:
            # Try to get the sigil content
            content = self.supervisor_connector.get_sigil_content_as_dict(sigil_ref)
            # If we get a non-empty dict, validation passed
            return content is not None and content != {}
        except Exception as e:
            logger.warning(f"Sigil validation failed for '{sigil_ref}': {e}")
            return False

    def initialize_supervisor_integration(self) -> None:
        """
        Initializes the supervisor integration features.
        This method should be called during VantaCore startup
        to set up all supervisor-related features.
        """
        supervisor_integration = self.config.get("supervisor_integration", {})

        # Skip if supervisor integration is explicitly disabled
        if supervisor_integration.get("supervisor_integration_enabled", True) is False:
            logger.info("Supervisor integration disabled in config")
            return

        # Register with supervisor
        if supervisor_integration.get("register_status_updates", True):
            success = self.register_with_supervisor()
            if not success:
                logger.warning(
                    "Failed to register with supervisor during initialization"
                )  # Corrected this line
                # Even if registration fails, we might still want to check health

        # Validate supervisor health
        health_report = self.validate_supervisor_health()
        if health_report["status"] != "healthy":
            logger.warning(
                f"Supervisor health check during initialization: {health_report['status']}"
            )
            logger.debug(f"Health check details: {health_report['details']}")
        else:
            logger.info("Supervisor health check passed during initialization")

    # === Methods expected by UnifiedVantaCore ===

    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the cognitive engine.
        Maps to the existing process_input method.
        """
        try:
            # Extract task parameters from task_data
            input_data_sigil_ref = task_data.get(
                "input_data_sigil_ref", "unified_task_input"
            )
            task_sigil_ref = task_data.get("task_sigil_ref", "unified_task")

            # Use existing process_input method
            solution_ref = self.process_input(
                input_data_sigil_ref=input_data_sigil_ref, task_sigil_ref=task_sigil_ref
            )

            return {
                "success": True,
                "solution_ref": solution_ref,
                "task_data": task_data,
            }
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return {"success": False, "error": str(e), "task_data": task_data}

    def get_status(self) -> Dict[str, Any]:
        """
        Get status from the cognitive engine.
        Maps to the existing get_status_report method.
        """
        try:
            return self.get_status_report()
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"status": "error", "error": str(e), "available": False}

    def train(self, training_data: Any) -> Dict[str, Any]:
        """
        Train the cognitive engine with provided data.
        This is a placeholder implementation that can be expanded.
        """
        try:
            logger.info(
                f"Training cognitive engine with data type: {type(training_data)}"
            )

            # For now, just log the training attempt
            # Future implementation could update meta-parameters, task profiles, etc.
            training_result = {
                "success": True,
                "message": "Training completed",
                "data_processed": str(training_data)[:100] + "..."
                if len(str(training_data)) > 100
                else str(training_data),
            }

            logger.info(f"Training result: {training_result['message']}")
            return training_result

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {"success": False, "error": str(e), "message": "Training failed"}

    def shutdown(self) -> None:
        """
        Shutdown the cognitive engine and clean up resources.
        """
        try:
            logger.info(
                "Shutting down VantaCognitiveEngine..."
            )  # Disconnect from supervisor if connected
            if hasattr(self, "supervisor_connector") and self.supervisor_connector:
                try:
                    # Try to call disconnect if available
                    disconnect_method = getattr(
                        self.supervisor_connector, "disconnect", None
                    )
                    if disconnect_method and callable(disconnect_method):
                        disconnect_method()
                        logger.info("Disconnected from supervisor")
                    else:
                        logger.info(
                            "Supervisor connector doesn't support disconnect method"
                        )
                except Exception as e:
                    logger.warning(f"Error disconnecting from supervisor: {e}")

            # Clear task profiles and performance history
            if hasattr(self, "task_adaptation_profiles"):
                self.task_adaptation_profiles.clear()

            if hasattr(self, "cross_task_knowledge_index"):
                self.cross_task_knowledge_index.clear()

            logger.info("VantaCognitiveEngine shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# === End of missing methods ===

# --- Example Usage (Conceptual - Supervisor would orchestrate this) ---
if __name__ == "__main__":
    # Use the real implementations instead of mocks
    # You must ensure these are properly configured and available in your environment

    # Example: instantiate the real supervisor connector, BLT encoder, and hybrid middleware
    # You may need to adjust constructor arguments as required by your real classes

    # Import the real classes (already imported above if available)
    # from Vanta.interfaces.real_supervisor_connector import RealSupervisorConnector
    # from BLT.blt_encoder import BLTEncoder
    # from BLT.hybrid_middleware import HybridMiddleware

    # Instantiate real components
    supervisor_connector = RealSupervisorConnector()
    blt_encoder = BLTEncoder()
    hybrid_middleware = HybridMiddleware()

    # Initialize VantaCognitiveEngine with real components
    vanta_core = VantaCognitiveEngine(
        config_sigil_ref="SigilRef:VantaCore_TestConfig_V1",
        supervisor_connector=supervisor_connector,
        blt_encoder=blt_encoder,
        hybrid_middleware=hybrid_middleware,
    )

    # Example: register a task and process input (adjust sigil refs as needed)
    # task_sigil = "SigilRef:YourTask"
    # task_desc_sigil = "SigilRef:YourTaskDesc"
    # vanta_core.register_arc_task_profile(task_sigil, task_desc_sigil)
    # solution_ref = vanta_core.process_input(input_data_sigil_ref="SigilRef:YourInput", task_sigil_ref=task_sigil)
    # print("Solution sigil:", solution_ref)

    # ...add your real workflow here...
