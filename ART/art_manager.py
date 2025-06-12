"""
ART Manager Module

This module provides the ARTManager class which orchestrates all ART (Adaptive Resonance Theory)
components including the core controller, trainer, bridges, and utilities.

The ARTManager serves as the main entry point for all ART operations and ensures
proper coordination between components.

Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh pattern for advanced
orchestration capabilities and VantaCore integration.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

from .art_controller import ARTController

# Core ART imports
from .art_logger import get_art_logger

# HOLO-1.5 Cognitive Mesh Integration
try:
    from ..core.vanta_registration import vanta_agent, CognitiveMeshRole, BaseAgent
    from ..core.base_agent import VantaAgentCapability
    HOLO_AVAILABLE = True
except ImportError:
    HOLO_AVAILABLE = False
    # Fallback decorators and classes
    def vanta_agent(**kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveMeshRole:
        MANAGER = "manager"
    
    class BaseAgent:
        pass
    
    class VantaAgentCapability:
        ORCHESTRATION = "orchestration"
        COMPONENT_MANAGEMENT = "component_management"
        RESOURCE_COORDINATION = "resource_coordination"


@vanta_agent(
    name="ARTManager",
    subsystem="art_orchestration",
    mesh_role=CognitiveMeshRole.MANAGER,
    capabilities=[
        VantaAgentCapability.ORCHESTRATION,
        VantaAgentCapability.COMPONENT_MANAGEMENT,
        VantaAgentCapability.RESOURCE_COORDINATION,
        "art_coordination",
        "bridge_management",
        "pattern_orchestration"
    ],
    cognitive_load=3.2,
    symbolic_depth=4
)
class ARTManager(BaseAgent if HOLO_AVAILABLE else object):
    """
    Main entry point for the ART (Adaptive Resonance Theory) module.
    Coordinates various ART components for pattern analysis, learning, and generation.

    Manages:
    - ARTController (core neural network)
    - ArtTrainer (training coordination)
    - GenerativeArt (art generation)
    - ArtEntropyBridge (entropy integration)
    - ARTRAGBridge (RAG integration)
    - ARTBLTBridge (BLT integration)
    - PatternAnalysis (pattern processing utilities)
    - DuplicationChecker (duplicate detection)
    
    Enhanced with HOLO-1.5 Recursive Symbolic Cognition Mesh:
    - Orchestral coordination of ART subsystems
    - Cognitive load balancing across components
    - Symbolic depth management for complex operations
    - Async initialization with VantaCore integration
    """

    def __init__(self, logger_instance=None, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ARTManager and its sub-components.

        Args:
            logger_instance: Optional custom logger instance.
            config: Optional configuration dictionary for ARTManager and its components.
                   Keys can include:
                   - 'art_controller': Config for ARTController
                   - 'art_trainer': Config for ArtTrainer
                   - 'generative_art': Config for GenerativeArt
                   - 'art_entropy_bridge': Config for ArtEntropyBridge
                   - 'art_rag_bridge': Config for ARTRAGBridge
                   - 'art_blt_bridge': Config for ARTBLTBridge
                   - 'pattern_analysis': Config for PatternAnalysis
                   - 'duplication_checker': Config for DuplicationChecker
        """
        self.logger = (
            logger_instance if logger_instance else get_art_logger(name="ARTManager")
        )
        self.config = config if config is not None else {}
        self.lock = threading.RLock()  # For thread safety

        # Component instances
        self.art_controller: ARTController = ARTController(logger_instance=self.logger)
        self.art_trainer = None
        self.generative_art = None
        self.entropy_bridge = None
        self.rag_bridge = None
        self.blt_bridge = None
        self.pattern_analyzer = None
        self.duplication_checker = None

        # Define guardian as a placeholder
        self.guardian: Optional[Any] = None

        # Management state
        self.initialized = False
        self.component_status = {}

        self.logger.info("Initializing ARTManager...")
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all ART components with proper error handling."""
        with self.lock:
            try:
                # 1. Initialize core ARTController first (required by other components)
                controller_config = self.config.get("art_controller", {})
                self.art_controller = ARTController(
                    logger_instance=self.logger, **controller_config
                )
                self.component_status["art_controller"] = "initialized"
                self.logger.info("ARTController initialized successfully")

                # 2. Initialize pattern analysis utilities
                try:
                    from tools.utilities.pattern_analysis import PatternAnalysis

                    pattern_analysis_config = self.config.get("pattern_analysis", {})
                    self.pattern_analyzer = PatternAnalysis(
                        logger_instance=self.logger, **pattern_analysis_config
                    )
                    self.component_status["pattern_analyzer"] = "initialized"
                    self.logger.info("PatternAnalysis initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import PatternAnalysis: {e}")
                    self.component_status["pattern_analyzer"] = "unavailable"
                # Stub fallback for pattern analyzer
                if self.pattern_analyzer is None:

                    class DummyPatternAnalyzer:
                        def scan(self, input_data, analysis_type=None):
                            return [float(len(str(input_data)))]

                        def get_hashable_representation(self, input_data):
                            return str(input_data)

                    self.pattern_analyzer = DummyPatternAnalyzer()
                    self.logger.warning("Using DummyPatternAnalyzer as fallback")

                # 3. Initialize duplication checker
                try:
                    from tools.utilities.duplication_checker import DuplicationChecker

                    duplication_checker_config = self.config.get(
                        "duplication_checker", {}
                    )
                    self.duplication_checker = DuplicationChecker(
                        logger_instance=self.logger, **duplication_checker_config
                    )
                    self.component_status["duplication_checker"] = "initialized"
                    self.logger.info("DuplicationChecker initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import DuplicationChecker: {e}")
                    self.component_status["duplication_checker"] = "unavailable"
                # Stub fallback for duplication checker
                if self.duplication_checker is None:

                    class DummyDuplicationChecker:
                        def is_duplicate_pattern(self, hash_val):
                            return False

                    self.duplication_checker = DummyDuplicationChecker()
                    self.logger.warning("Using DummyDuplicationChecker as fallback")

                # 4. Initialize ArtTrainer (depends on ARTController)
                try:
                    from .art_trainer import ArtTrainer

                    trainer_config = self.config.get("art_trainer", {})
                    self.art_trainer = ArtTrainer(
                        art_controller=self.art_controller,
                        config=trainer_config,
                        logger_instance=self.logger,
                    )
                    self.component_status["art_trainer"] = "initialized"
                    self.logger.info("ArtTrainer initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import ArtTrainer: {e}")
                    self.component_status["art_trainer"] = "unavailable"

                # 5. Initialize GenerativeArt
                try:
                    from .generative_art import GenerativeArt

                    generative_art_config = self.config.get("generative_art", {})
                    self.generative_art = GenerativeArt(
                        config=generative_art_config, logger_instance=self.logger
                    )
                    self.component_status["generative_art"] = "initialized"
                    self.logger.info("GenerativeArt initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import GenerativeArt: {e}")
                    self.component_status["generative_art"] = "unavailable"

                # 6. Initialize ArtEntropyBridge (will be configured later when entropy guardian is available)
                try:
                    from .art_entropy_bridge import ArtEntropyBridge

                    entropy_bridge_config = self.config.get("art_entropy_bridge", {})
                    # Initialize with ARTController, entropy guardian can be set later
                    self.entropy_bridge = ArtEntropyBridge(
                        art_controller=self.art_controller,
                        entropy_guardian=None,  # Can be set later via set_entropy_guardian
                        config=entropy_bridge_config,
                        logger_instance=self.logger,
                    )
                    self.component_status["entropy_bridge"] = "initialized"
                    self.logger.info("ArtEntropyBridge initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import ArtEntropyBridge: {e}")
                    self.component_status["entropy_bridge"] = "unavailable"

                # 7. Initialize ARTRAGBridge
                try:
                    from .art_rag_bridge import ARTRAGBridge

                    rag_bridge_config = self.config.get("art_rag_bridge", {})
                    self.rag_bridge = ARTRAGBridge(
                        art_manager=self,  # Pass self as art_manager
                        **rag_bridge_config,
                    )
                    self.component_status["rag_bridge"] = "initialized"
                    self.logger.info("ARTRAGBridge initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import ARTRAGBridge: {e}")
                    self.component_status["rag_bridge"] = "unavailable"

                # 8. Initialize ARTBLTBridge
                try:
                    from .blt.art_blt_bridge import ARTBLTBridge

                    blt_bridge_config = self.config.get("art_blt_bridge", {})
                    self.blt_bridge = ARTBLTBridge(
                        art_manager=self,  # Pass self as art_manager
                        config=blt_bridge_config,
                    )
                    self.component_status["blt_bridge"] = "initialized"
                    self.logger.info("ARTBLTBridge initialized successfully")
                except ImportError as e:
                    self.logger.warning(f"Could not import ARTBLTBridge: {e}")
                    self.component_status["blt_bridge"] = "unavailable"

                self.initialized = True
                self.logger.info(
                    "ARTManager and its components initialized successfully."
                )

            except Exception as e:
                self.logger.error(
                    f"Error during ARTManager initialization: {e}", exc_info=True
                )
                self.initialized = False
                raise

    # === Component Access Methods ===

    def get_art_controller(self) -> ARTController:
        """Get the ARTController instance."""
        # Ensure art_controller is initialized before returning
        if self.art_controller is None:
            raise RuntimeError("ARTController must be initialized before use.")
        return self.art_controller

    def get_art_trainer(self) -> Optional[Any]:
        """Get the ArtTrainer instance."""
        return self.art_trainer

    def get_generative_art(self) -> Optional[Any]:
        """Get the GenerativeArt instance."""
        return self.generative_art

    def get_entropy_bridge(self) -> Optional[Any]:
        """Get the ArtEntropyBridge instance."""
        return self.entropy_bridge

    def get_rag_bridge(self) -> Optional[Any]:
        """Get the ARTRAGBridge instance."""
        return self.rag_bridge

    def get_blt_bridge(self) -> Optional[Any]:
        """Get the ARTBLTBridge instance."""
        return self.blt_bridge

    def get_pattern_analyzer(self) -> Optional[Any]:
        """Get the PatternAnalysis instance."""
        return self.pattern_analyzer

    def get_duplication_checker(self) -> Optional[Any]:
        """Get the DuplicationChecker instance."""
        return self.duplication_checker

    # === High-Level ART Operations ===

    def analyze_input(
        self, input_data: Any, analysis_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a single input data item using the full ART pipeline.

        Args:
            input_data: The input data to analyze.
            analysis_type: Optional hint for PatternAnalysis.

        Returns:
            A dictionary containing the analysis result from ARTController.
        """
        if not self.initialized:
            return {"error": "ARTManager not properly initialized"}

        self.logger.info(f"Analyzing input: {type(input_data)}")

        with self.lock:
            try:
                # 1. Check for duplicates if duplication checker is available
                if self.duplication_checker and self.pattern_analyzer:
                    raw_input_hash = self.pattern_analyzer.get_hashable_representation(
                        input_data
                    )
                    if self.duplication_checker.is_duplicate_pattern(raw_input_hash):
                        self.logger.warning(
                            f"Duplicate raw input detected (hash: {raw_input_hash[:10]}...)"
                        )

                # 2. Scan input using PatternAnalysis to get a numerical vector
                if self.pattern_analyzer:
                    numerical_pattern_vector = self.pattern_analyzer.scan(
                        input_data, analysis_type=analysis_type
                    )
                else:
                    # Fallback: if no pattern analyzer, try to use input directly if it's numeric
                    if isinstance(input_data, (list, tuple)) and all(
                        isinstance(x, (int, float)) for x in input_data
                    ):
                        numerical_pattern_vector = input_data
                    else:
                        self.logger.error(
                            "Pattern analyzer unavailable and input is not numeric"
                        )
                        return {
                            "error": "Pattern analysis failed",
                            "category_id": None,
                            "resonance": 0.0,
                        }
                    if numerical_pattern_vector is None:
                        self.logger.error("Pattern analysis failed to produce a vector")
                    return {
                        "error": "Pattern analysis failed",
                        "category_id": None,
                        "resonance": 0.0,
                    }
                # 3. Process with ARTController
                if isinstance(numerical_pattern_vector, tuple):
                    numerical_pattern_vector = list(numerical_pattern_vector)

                # Check if art_controller is available
                if not self.art_controller:
                    return {
                        "error": "ARTController not available",
                        "category_id": None,
                        "resonance": 0.0,
                    }

                if numerical_pattern_vector is None:
                    self.logger.error(
                        "Numerical pattern vector is None, cannot process input"
                    )
                    return {
                        "error": "Numerical pattern vector is None",
                        "category_id": None,
                        "resonance": 0.0,
                    }

                if not all(
                    isinstance(x, (float, int)) for x in numerical_pattern_vector
                ):
                    self.logger.error("Numerical pattern vector contains invalid types")
                    return {
                        "error": "Invalid numerical pattern vector",
                        "category_id": None,
                        "resonance": 0.0,
                    }

                numerical_pattern_vector = [
                    float(x) for x in numerical_pattern_vector
                ]  # Ensure all elements are floats
                analysis_result = self.art_controller.process(
                    numerical_pattern_vector, training=False
                )

                # 4. Log pattern trace
                self._log_pattern_trace(
                    pattern=input_data, result=analysis_result, context="analyze_input"
                )

                self.logger.info(
                    f"Input analysis complete. Category: {analysis_result.get('category_id')}, "
                    f"Resonance: {analysis_result.get('resonance')}"
                )
                # Wrap result for test compatibility
                return {
                    "category": {"id": analysis_result.get("category_id")},
                    "resonance": analysis_result.get("resonance"),
                    "is_novel_category": analysis_result.get("is_new_category"),
                }

            except Exception as e:
                self.logger.error(f"Error during input analysis: {e}", exc_info=True)
                return {
                    "error": f"Analysis exception: {e}",
                    "category_id": None,
                    "resonance": 0.0,
                }

    def train_on_input(
        self, input_data: Any, analysis_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trains the ART system on a single input.

        Args:
            input_data: The input data to train on.
            analysis_type: Optional hint for PatternAnalysis.

        Returns:
            A dictionary containing the training result.
        """
        if not self.initialized:
            return {"error": "ARTManager not properly initialized"}

        self.logger.info(f"Training on input: {type(input_data)}")

        with self.lock:
            try:
                # Use ArtTrainer if available, otherwise fallback to direct controller training
                if self.art_trainer:
                    return self.art_trainer.train_from_event(
                        input_data=input_data,
                        output_data=None,
                        metadata={"analysis_type": analysis_type}
                        if analysis_type
                        else None,
                    )
                else:
                    # Fallback to direct training
                    if self.pattern_analyzer:
                        numerical_pattern_vector = self.pattern_analyzer.scan(
                            input_data, analysis_type=analysis_type
                        )
                    else:
                        if isinstance(input_data, (list, tuple)) and all(
                            isinstance(x, (int, float)) for x in input_data
                        ):
                            numerical_pattern_vector = input_data
                        else:
                            return {
                                "error": "Pattern analyzer unavailable and input is not numeric"
                            }
                    if numerical_pattern_vector is None:
                        return {"error": "Pattern analysis failed"}

                    if isinstance(numerical_pattern_vector, tuple):
                        numerical_pattern_vector = list(numerical_pattern_vector)

                    # Check if art_controller is available
                    if not self.art_controller:
                        return {"error": "ARTController not available"}

                    train_result = self.art_controller.train(numerical_pattern_vector)
                    self._log_pattern_trace(
                        pattern=input_data, result=train_result, context="train_input"
                    )
                    return {"status": "success", "art_result": train_result}

            except Exception as e:
                self.logger.error(f"Error during training: {e}", exc_info=True)
                return {"error": f"Training exception: {e}"}

    def train_on_batch(
        self, batch_data: List[Any], analysis_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trains the ART module on a batch of data items.

        Args:
            batch_data: A list of data items for training.
            analysis_type: Optional hint for PatternAnalysis.

        Returns:
            A dictionary summarizing the batch training results.
        """
        if not self.initialized:
            return {"status": "error", "items_processed": 0}

        self.logger.info(f"Starting training on batch of {len(batch_data)} items")

        with self.lock:
            try:
                # Use ArtTrainer batch method if available
                if self.art_trainer:
                    batch_events = [
                        (
                            item,
                            None,
                            {"analysis_type": analysis_type} if analysis_type else {},
                        )
                        for item in batch_data
                    ]
                    results = self.art_trainer.train_batch(batch_events)

                    # Summarize results
                    successful = sum(1 for r in results if r.get("status") == "success")
                    return {"status": "success", "items_processed": len(batch_data)}
                else:
                    # Fallback to individual training
                    results = []
                    successful = 0
                    for item in batch_data:
                        result = self.train_on_input(item, analysis_type)
                        results.append(result)
                        if result.get("status") == "success":
                            successful += 1

                    return {"status": "success", "items_processed": len(batch_data)}

            except Exception as e:
                self.logger.error(f"Error during training: {e}", exc_info=True)
                return {"status": "error", "items_processed": 0}

    def generate_art(
        self, prompt: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate art using the GenerativeArt component.

        Args:
            prompt: Text prompt for art generation.
            metadata: Additional generation parameters.

        Returns:
            Dictionary containing generated art and metadata.
        """
        if not self.generative_art:
            return {"error": "GenerativeArt component not available"}

        try:
            return self.generative_art.generate(prompt=prompt, metadata=metadata)
        except Exception as e:
            self.logger.error(f"Error during art generation: {e}", exc_info=True)
            return {"error": f"Art generation exception: {e}"}

        # === Integration Methods ===    def set_entropy_guardian(self, guardian: Any) -> bool:
        """Attach an entropy guardian to the entropy bridge."""
        if not self.entropy_bridge:
            return False
        try:
            if not guardian:
                self.logger.warning(
                    "No guardian provided to set_entropy_guardian. Ensure a valid guardian is passed."
                )
                return False
            # Note: Ensure this method is called with a valid `guardian` before using `entropy_bridge`.
            if guardian is None:
                self.logger.warning(
                    "set_entropy_guardian called without a valid guardian."
                )
                return False
            self.entropy_bridge.entropy_guardian = guardian
            return True
        except Exception as e:
            self.logger.error(f"Failed to set entropy guardian: {e}")
            return False

    def enable_entropy_adaptation(self) -> bool:
        """Enable entropy-based adaptation if ArtEntropyBridge is available."""
        if not self.entropy_bridge:
            return False

        try:
            if hasattr(self.entropy_bridge, "activate"):
                self.entropy_bridge.activate()
                return True
        except Exception as e:
            self.logger.error(f"Error enabling entropy adaptation: {e}")
        return False

    def disable_entropy_adaptation(self) -> bool:
        """Disable entropy-based adaptation if ArtEntropyBridge is available."""
        if not self.entropy_bridge:
            return False

        try:
            if hasattr(self.entropy_bridge, "deactivate"):
                self.entropy_bridge.deactivate()
                return True
        except Exception as e:
            self.logger.error(f"Error disabling entropy adaptation: {e}")
        return False

    # === State Management ===

    def save_state(self, file_path: str) -> bool:
        """
        Saves the state of all ART components.

        Args:
            file_path: Path to save the state file.

        Returns:
            True if successful, False otherwise.
        """
        self.logger.info(f"ARTManager: Attempting to save state to {file_path}")

        with self.lock:
            try:
                state = {
                    "art_controller": None,
                    "art_trainer": None,
                    "generative_art": None,
                    "config": self.config,
                    "component_status": self.component_status,
                    "timestamp": time.time(),
                }

                # Save ARTController state
                if self.art_controller:
                    controller_state_path = f"{file_path}_controller.pkl"
                    if self.art_controller.save_state(controller_state_path):
                        state["art_controller"] = controller_state_path

                # Save ArtTrainer state if available
                if self.art_trainer and hasattr(self.art_trainer, "save_state"):
                    trainer_state_path = f"{file_path}_trainer.pkl"
                    if self.art_trainer.save_state(trainer_state_path):
                        state["art_trainer"] = trainer_state_path

                # Save GenerativeArt state if it has state to save
                if self.generative_art and hasattr(self.generative_art, "get_stats"):
                    state["generative_art"] = self.generative_art.get_stats()

                # Save main state file
                import pickle

                with open(file_path, "wb") as f:
                    pickle.dump(state, f)

                self.logger.info(f"ARTManager state saved to {file_path}")
                return True

            except Exception as e:
                self.logger.error(f"Error saving ARTManager state: {e}", exc_info=True)
                return False

    def load_state(self, file_path: str) -> bool:
        """
        Loads the state of all ART components.

        Args:
            file_path: Path to load the state file from.

        Returns:
            True if successful, False otherwise.
        """
        self.logger.info(f"ARTManager: Attempting to load state from {file_path}")

        with self.lock:
            try:
                import os
                import pickle

                if not os.path.exists(file_path):
                    self.logger.error(f"State file {file_path} not found")
                    return False

                with open(file_path, "rb") as f:
                    state = pickle.load(f)

                # Load ARTController state
                if state.get("art_controller") and self.art_controller:
                    if not self.art_controller.load_state(state["art_controller"]):
                        self.logger.warning("Failed to load ARTController state")

                # Load ArtTrainer state
                if (
                    state.get("art_trainer")
                    and self.art_trainer
                    and hasattr(self.art_trainer, "load_state")
                ):
                    if not self.art_trainer.load_state(state["art_trainer"]):
                        self.logger.warning("Failed to load ArtTrainer state")

                # Restore component status
                if state.get("component_status"):
                    self.component_status.update(state["component_status"])

                self.logger.info(f"ARTManager state loaded from {file_path}")
                return True

            except Exception as e:
                self.logger.error(f"Error loading ARTManager state: {e}", exc_info=True)
                return False

    # === Status and Configuration ===

    def status(self) -> Dict[str, Any]:
        """
        Returns a comprehensive status of the ARTManager and all its components.

        Returns:
            Dictionary containing status information.
        """
        with self.lock:
            status = {
                "initialized": self.initialized,
                "timestamp": time.time(),
                "component_status": self.component_status.copy(),
                "art_controller_status": self.art_controller.status()
                if self.art_controller
                else None,
                "art_trainer_stats": None,
                "generative_art_stats": None,
            }

            # Get ArtTrainer stats if available
            if self.art_trainer and hasattr(self.art_trainer, "get_training_stats"):
                try:
                    status["art_trainer_stats"] = self.art_trainer.get_training_stats()
                except Exception as e:
                    self.logger.warning(f"Error getting ArtTrainer stats: {e}")

            # Get GenerativeArt stats if available
            if self.generative_art and hasattr(self.generative_art, "get_stats"):
                try:
                    status["generative_art_stats"] = self.generative_art.get_stats()
                except Exception as e:
                    self.logger.warning(f"Error getting GenerativeArt stats: {e}")

            return status

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration. Note: This does not reinitialize components.

        Args:
            new_config: New configuration dictionary to merge.
        """
        with self.lock:
            self.config.update(new_config)
            self.logger.info("ARTManager configuration updated")

    # === Utility Methods ===

    def _log_pattern_trace(
        self, pattern: Any, result: Dict[str, Any], context: Optional[str] = None
    ) -> None:
        """
        Logs the trace of a pattern analysis or training event.

        Args:
            pattern: The input pattern that was processed.
            result: The result dictionary from processing.
            context: Optional context string.
        """
        trace_log = {
            "timestamp": time.time(),
            "context": context or "general_processing",
            "pattern_type": str(type(pattern)),
            "result_category_id": result.get("category_id"),
            "result_resonance": result.get("resonance"),
            "result_is_new_category": result.get("is_new_category"),
            "result_is_anomaly": result.get("is_anomaly"),
            "art_controller_stats_snapshot": self.art_controller.get_statistics()
            if self.art_controller
            else None,
        }
        self.logger.info("ART Event Trace", extra=trace_log)

    # --- Test compatibility helpers ---
    @property
    def controller(self) -> ARTController:
        """Alias to the underlying ARTController."""
        return self.art_controller

    @property
    def pattern_analysis(self) -> Any:
        """Alias to the PatternAnalysis component."""
        return self.pattern_analyzer

    def set_vigilance(self, value: float) -> bool:
        """Proxy to set_vigilance on ARTController for tests."""
        if not self.art_controller:
            return False
        try:
            self.art_controller.set_vigilance(value)
            return True
        except Exception:
            return False

    def set_learning_rate(self, value: float) -> bool:
        """Proxy to set_learning_rate on ARTController for tests."""
        if not self.art_controller:
            return False
        try:
            self.art_controller.set_learning_rate(value)
            return True
        except Exception:
            return False

    def log_pattern_trace(
        self,
        pattern_id: Any,
        source_data: Any,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Public method for pattern trace logging and returning a trace dict."""
        trace = {
            "pattern_id": pattern_id,
            "source_data": source_data,
            "context": context,
            "metadata": metadata,
        }
        try:
            self._log_pattern_trace(
                pattern=source_data,
                result={"category_id": pattern_id},
                context=str(context),
            )
        except Exception:
            self.logger.error("Failed to log pattern trace")
        return trace
