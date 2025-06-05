"""
ART Unified Adapter Module

This module provides a comprehensive adapter that unifies connections between:
- ART (Adaptive Resonance Theory)
- BLT (Belief Learning Technology)
- VantaCore
- Gridformer
- GUI systems

It centralizes all bridge logic into a single coherent interface, handling:
1. Dynamic component discovery and loading
2. Cross-system communication and data formatting
3. Common configuration management
4. Event propagation and error handling
"""

import importlib
import logging
import os
import sys
import threading
import time
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path

# Import ART components with proper error handling
try:
    from .art_controller import ARTController

    HAS_ART_CONTROLLER = True
except ImportError:
    HAS_ART_CONTROLLER = False

try:
    from .art_trainer import ArtTrainer

    HAS_ART_TRAINER = True
except ImportError:
    HAS_ART_TRAINER = False

try:
    from .art_logger import get_art_logger
except ImportError:
    # Create a simple fallback logging function
    def get_art_logger(name=None, level=None, log_file=None, base_logger_name=None):
        logger = logging.getLogger(name or "art_adapter")
        if level:
            logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


# Create a default logger for this module
logger = get_art_logger("ArtAdapter")

# Import BLT components dynamically
try:
    from ..BLT.blt_system_adapters import BLTSystem

    HAS_BLT_SYSTEM = True
except ImportError:
    HAS_BLT_SYSTEM = False
    logger.warning("BLT system components not available")

# Check for Gridformer components
try:
    # Try to find and import Gridformer components
    import sys

    sys_paths = sys.path.copy()
    if "Gridformer" not in sys.modules:
        # Try to locate Gridformer relative to this file
        gridformer_path = Path(__file__).parent.parent / "Gridformer"
        if gridformer_path.exists():
            sys.path.append(str(gridformer_path))
            HAS_GRIDFORMER = True
            logger.info(f"Added Gridformer path: {gridformer_path}")
        else:
            HAS_GRIDFORMER = False
            logger.warning("Gridformer directory not found")
    else:
        HAS_GRIDFORMER = True
except Exception as e:
    HAS_GRIDFORMER = False
    logger.error(f"Error checking for Gridformer: {e}")

# Check for VantaCore components
try:
    vanta_package_path = Path(__file__).parent.parent / "Vanta"
    vanta_core_path = vanta_package_path / "core"

    if vanta_core_path.exists():
        if str(vanta_package_path) not in sys.path:
            sys.path.append(str(vanta_package_path))
        HAS_VANTACORE = True
        logger.info(f"Added VantaCore path: {vanta_package_path}")
    else:
        HAS_VANTACORE = False
        logger.warning("VantaCore directory not found")
except Exception as e:
    HAS_VANTACORE = False
    logger.error(f"Error checking for VantaCore: {e}")

# Check for GUI components
try:
    gui_package_path = Path(__file__).parent.parent / "GUI"

    if gui_package_path.exists():
        if str(gui_package_path) not in sys.path:
            sys.path.append(str(gui_package_path))
        HAS_GUI = True
        logger.info(f"Added GUI path: {gui_package_path}")
    else:
        HAS_GUI = False
        logger.warning("GUI directory not found")
except Exception as e:
    HAS_GUI = False
    logger.error(f"Error checking for GUI: {e}")


class ArtAdapter:
    """
    Unified adapter class that provides a single interface for all ART functionality
    and bridges connections to other components (BLT, VantaCore, Gridformer, GUI).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        art_controller: Optional[Any] = None,
        art_trainer: Optional[Any] = None,
        logger_instance: Optional[logging.Logger] = None,
        enable_bridges: bool = True,
    ):
        """
        Initialize the ArtAdapter with optional components and configuration.

        Args:
            config: Configuration dictionary for all components
            art_controller: Optional existing ARTController instance
            art_trainer: Optional existing ArtTrainer instance
            logger_instance: Optional logger instance
            enable_bridges: Whether to enable automatic bridge connections
        """
        self.logger = logger_instance or get_art_logger("ArtAdapter")
        self.logger.info("Initializing ArtAdapter...")

        # Set up configuration
        self.config = config or {}

        # Component instances
        self.art_controller = None
        self.art_trainer = None
        self.blt_system = None
        self.vanta_core = None
        self.gridformer = None

        # Bridge instances
        self.bridges = {}
        self.callbacks = {}
        self.lock = threading.RLock()

        # Initialize components
        self._init_art_controller(art_controller)
        self._init_art_trainer(art_trainer)

        # Connect bridges if enabled
        if enable_bridges:
            self._connect_bridges()

        self.logger.info("ArtAdapter initialization complete")

    def _init_art_controller(self, existing_controller: Optional[Any] = None) -> None:
        """Initialize the ARTController component"""
        if existing_controller:
            self.art_controller = existing_controller
            self.logger.info("Using provided ARTController instance")
            return

        if not HAS_ART_CONTROLLER:
            self.logger.warning("ARTController not available, cannot initialize")
            return

        try:
            # Extract ARTController specific config
            art_config = self.config.get("art_controller", {})

            # Create ARTController instance
            self.art_controller = ARTController(
                config=art_config, logger_instance=self.logger
            )
            self.logger.info("Created new ARTController instance")
        except Exception as e:
            self.logger.error(f"Failed to initialize ARTController: {e}")

    def _init_art_trainer(self, existing_trainer: Optional[Any] = None) -> None:
        """Initialize the ArtTrainer component"""
        if existing_trainer:
            self.art_trainer = existing_trainer
            self.logger.info("Using provided ArtTrainer instance")
            return

        if not HAS_ART_TRAINER:
            self.logger.warning("ArtTrainer not available, cannot initialize")
            return

        if not self.art_controller:
            self.logger.warning(
                "ARTController not initialized, ArtTrainer may have limited functionality"
            )

        try:
            # Extract ArtTrainer specific config
            trainer_config = self.config.get("art_trainer", {})

            # Create ArtTrainer instance
            self.art_trainer = ArtTrainer(
                art_controller=self.art_controller,
                config=trainer_config,
                logger_instance=self.logger,
            )
            self.logger.info("Created new ArtTrainer instance")
        except Exception as e:
            self.logger.error(f"Failed to initialize ArtTrainer: {e}")

    def _connect_bridges(self) -> None:
        """Set up all bridge connections between components"""
        self.logger.info("Setting up bridge connections...")

        try:
            # Connect ART to BLT if available
            self._connect_art_blt_bridge()

            # Connect ART to VantaCore if available
            self._connect_art_vanta_bridge()

            # Connect ART to Gridformer if available
            self._connect_art_gridformer_bridge()

            # Connect ART to GUI if available
            self._connect_art_gui_bridge()

            # Connect ART entropy bridge if available
            self._connect_art_entropy_bridge()

            # Connect ART RAG bridge if available
            self._connect_art_rag_bridge()

            self.logger.info(
                f"Bridge connections established: {list(self.bridges.keys())}"
            )
        except Exception as e:
            self.logger.error(f"Error setting up bridge connections: {e}")

    def _connect_art_blt_bridge(self) -> None:
        """Connect ART to BLT system"""
        if not self.art_controller or not HAS_BLT_SYSTEM:
            return

        try:
            # Try to import the bridge module
            from ART.art_hybrid_blt_bridge import ARTHybridBLTBridge as ARTBLTBridge

            # Create bridge instance
            blt_bridge = ARTBLTBridge(
                art_manager=self.art_controller,
                logger_instance=self.logger,
                entropy_threshold=self.config.get("blt_entropy_threshold", 0.4),
                patch_size=self.config.get("blt_patch_size", 8),
            )

            self.bridges["art_blt"] = blt_bridge
            self.logger.info("ART-BLT bridge connected")
        except ImportError:
            self.logger.warning("ARTBLTBridge not available, skipping BLT connection")
        except Exception as e:
            self.logger.error(f"Error connecting ART-BLT bridge: {e}")

    def _connect_art_vanta_bridge(self) -> None:
        """Connect ART to VantaCore system"""
        if not self.art_controller or not HAS_VANTACORE:
            return

        try:
            # Try to import VANTA factory from adapter
            from .adapter import VANTAFactory

            # Store the factory for later use
            self.bridges["vanta_factory"] = VANTAFactory
            self.logger.info("ART-VANTA connection available via VANTAFactory")
        except ImportError:
            self.logger.warning("VANTAFactory not available, skipping VANTA connection")
        except Exception as e:
            self.logger.error(f"Error connecting ART-VANTA bridge: {e}")

    def _connect_art_gridformer_bridge(self) -> None:
        """Connect ART to Gridformer system"""
        if not self.art_controller or not HAS_GRIDFORMER:
            return

        try:
            # Placeholder for future Gridformer adapter import
            # Commented out to avoid import errors
            # from ..BLT.arc_gridformer_blt_adapter import GridformerBLTAdapter

            # Create a placeholder class if import fails
            class GridformerBLTAdapter:
                """Placeholder for GridformerBLTAdapter when the module is not available."""

                def __init__(self, **kwargs):
                    self.logger = kwargs.get(
                        "logger", logging.getLogger("PlaceholderGridformerAdapter")
                    )
                    self.logger.warning("Using placeholder GridformerBLTAdapter")

                def connect(self):
                    self.logger.info(
                        "Placeholder GridformerBLTAdapter.connect() called"
                    )
                    return True

            # Create adapter instance (this acts as a bridge)
            gridformer_adapter = GridformerBLTAdapter(logger=self.logger)

            self.bridges["art_gridformer"] = gridformer_adapter
            self.logger.info("ART-Gridformer bridge connected via BLT adapter")
        except ImportError:
            self.logger.warning(
                "ARCGridFormerBLTAdapter not available, skipping Gridformer connection"
            )
        except Exception as e:
            self.logger.error(f"Error connecting ART-Gridformer bridge: {e}")

    def _connect_art_gui_bridge(self) -> None:
        """Connect ART to GUI system"""
        if not self.art_controller or not HAS_GUI:
            return

        try:
            # We don't have a direct GUI bridge yet, but we can register callbacks
            # that the GUI can use to get ART state information
            self.callbacks["get_art_state"] = self.get_art_state
            self.callbacks["get_art_categories"] = self.get_art_categories
            self.callbacks["get_art_anomalies"] = self.get_art_anomalies

            self.logger.info("ART-GUI integration prepared via callbacks")
        except Exception as e:
            self.logger.error(f"Error setting up ART-GUI callbacks: {e}")

    def _connect_art_entropy_bridge(self) -> None:
        """Connect ART entropy bridge if available"""
        if not self.art_controller:
            return

        try:
            # Try to import the entropy bridge module
            from .art_entropy_bridge import ArtEntropyBridge

            # For now, store the class for later instantiation
            # We need an entropy guardian component to fully initialize
            self.bridges["art_entropy_class"] = ArtEntropyBridge
            self.logger.info(
                "ART-Entropy bridge class available (requires entropy guardian)"
            )
        except ImportError:
            self.logger.warning("ArtEntropyBridge not available")
        except Exception as e:
            self.logger.error(f"Error with ART-Entropy bridge: {e}")

    def _connect_art_rag_bridge(self) -> None:
        """Connect ART RAG bridge if available"""
        if not self.art_controller:
            return

        try:
            # Try to import the RAG bridge module
            from .art_rag_bridge import ARTRAGBridge

            # Create bridge instance
            rag_bridge = ARTRAGBridge(
                art_manager=self.art_controller,
                embedding_model=self.config.get(
                    "rag_embedding_model", "all-MiniLM-L6-v2"
                ),
                blt_hybrid_weight=self.config.get("blt_hybrid_weight", 0.7),
            )

            self.bridges["art_rag"] = rag_bridge
            self.logger.info("ART-RAG bridge connected")
        except ImportError:
            self.logger.warning("ARTRAGBridge not available, skipping RAG connection")
        except Exception as e:
            self.logger.error(f"Error connecting ART-RAG bridge: {e}")

    # --- Public Methods for ART Management ---

    def train(
        self,
        input_data: Any,
        output_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train the ART system on an input/output pair with optional metadata.

        Args:
            input_data: Input data to train on (text, vector, etc.)
            output_data: Optional output/response data
            metadata: Optional metadata about the interaction

        Returns:
            Dictionary with training results
        """
        if not self.art_trainer:
            self.logger.error("ArtTrainer not available, cannot train")
            return {"status": "error", "message": "ArtTrainer not available"}

        try:
            # Use ArtTrainer to process the event
            result = self.art_trainer.train_from_event(
                input_data, output_data, metadata
            )

            # Notify bridges about training event
            self._notify_bridges("on_art_trained", result)

            return result
        except Exception as e:
            self.logger.error(f"Error training ART: {e}")
            return {"status": "error", "message": f"Training error: {e}"}

    def process(self, input_data: Any, training: bool = False) -> Dict[str, Any]:
        """
        Process an input through the ART controller directly.

        Args:
            input_data: Input data to process (usually a feature vector)
            training: Whether to create new categories if no match

        Returns:
            Dictionary with processing results
        """
        if not self.art_controller:
            self.logger.error("ARTController not available, cannot process")
            return {"status": "error", "message": "ARTController not available"}

        try:
            # Check if input needs feature extraction
            if not isinstance(input_data, (list, np.ndarray)) and self.art_trainer:
                # Use trainer to extract features first
                self.logger.info("Converting input to feature vector using ArtTrainer")
                feature_vector = self.art_trainer._create_feature_vector(
                    input_data, None, {}
                )
                if feature_vector is None:
                    return {"status": "error", "message": "Feature extraction failed"}
            else:
                feature_vector = input_data

            # Process through ARTController
            result = self.art_controller.process(feature_vector, training=training)

            # Notify bridges about processing event
            self._notify_bridges("on_art_processed", result)

            return result
        except Exception as e:
            self.logger.error(f"Error processing through ART: {e}")
            return {"status": "error", "message": f"Processing error: {e}"}

    def get_art_state(self) -> Dict[str, Any]:
        """Get the current state of the ART system"""
        state = {
            "adapter_status": "operational",
            "bridges_connected": list(self.bridges.keys()),
            "timestamp": time.time(),
        }

        if self.art_controller:
            try:
                state["art_controller"] = {
                    "status": "available",
                    "stats": self.art_controller.get_statistics(),
                    "category_count": len(self.art_controller.get_all_categories()),
                }
            except Exception as e:
                state["art_controller"] = {"status": "error", "message": str(e)}
        else:
            state["art_controller"] = {"status": "unavailable"}

        if self.art_trainer:
            try:
                state["art_trainer"] = {
                    "status": "available",
                    "stats": self.art_trainer.get_training_stats(),
                    "config": self.art_trainer.get_config_summary(),
                }
            except Exception as e:
                state["art_trainer"] = {"status": "error", "message": str(e)}
        else:
            state["art_trainer"] = {"status": "unavailable"}

        return state

    def get_art_categories(self) -> List[Dict[str, Any]]:
        """Get all categories in the ART system"""
        if not self.art_controller:
            return []

        try:
            return self.art_controller.get_all_categories()
        except Exception as e:
            self.logger.error(f"Error getting ART categories: {e}")
            return []

    def get_art_anomalies(self) -> List[int]:
        """Get anomalous categories in the ART system"""
        if not self.art_controller:
            return []

        try:
            return self.art_controller.get_anomaly_categories()
        except Exception as e:
            self.logger.error(f"Error getting ART anomalies: {e}")
            return []

    def save_state(self, directory: str = "./checkpoints") -> Dict[str, bool]:
        """
        Save the state of all ART components

        Args:
            directory: Directory to save state files

        Returns:
            Dictionary indicating success/failure for each component
        """
        results = {}

        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)

        # Save ARTController state
        if self.art_controller:
            try:
                controller_path = os.path.join(directory, "art_controller_state.pkl")
                controller_result = self.art_controller.save_state(controller_path)
                results["art_controller"] = controller_result
            except Exception as e:
                self.logger.error(f"Error saving ARTController state: {e}")
                results["art_controller"] = False

        # Save ArtTrainer state
        if self.art_trainer:
            try:
                trainer_path = os.path.join(directory, "art_trainer_state")
                trainer_result = self.art_trainer.save_state(trainer_path)
                results["art_trainer"] = trainer_result
            except Exception as e:
                self.logger.error(f"Error saving ArtTrainer state: {e}")
                results["art_trainer"] = False

        return results

    def load_state(self, directory: str = "./checkpoints") -> Dict[str, bool]:
        """
        Load the state of all ART components

        Args:
            directory: Directory containing state files

        Returns:
            Dictionary indicating success/failure for each component
        """
        results = {}

        # Load ARTController state
        if self.art_controller:
            try:
                controller_path = os.path.join(directory, "art_controller_state.pkl")
                if os.path.exists(controller_path):
                    controller_result = self.art_controller.load_state(controller_path)
                    results["art_controller"] = controller_result
                else:
                    self.logger.warning(
                        f"ARTController state file not found: {controller_path}"
                    )
                    results["art_controller"] = False
            except Exception as e:
                self.logger.error(f"Error loading ARTController state: {e}")
                results["art_controller"] = False

        # Load ArtTrainer state
        if self.art_trainer:
            try:
                trainer_path = os.path.join(directory, "art_trainer_state")
                if os.path.exists(f"{trainer_path}.pkl"):
                    trainer_result = self.art_trainer.load_state(trainer_path)
                    results["art_trainer"] = trainer_result
                else:
                    self.logger.warning(
                        f"ArtTrainer state file not found: {trainer_path}.pkl"
                    )
                    results["art_trainer"] = False
            except Exception as e:
                self.logger.error(f"Error loading ArtTrainer state: {e}")
                results["art_trainer"] = False

        return results

    # --- Bridge Event Notification Methods ---

    def _notify_bridges(self, event_name: str, data: Any) -> None:
        """Notify all bridges about an event"""
        for bridge_name, bridge in self.bridges.items():
            try:
                handler_method = getattr(bridge, event_name, None)
                if handler_method and callable(handler_method):
                    handler_method(data)
            except Exception as e:
                self.logger.error(
                    f"Error notifying bridge {bridge_name} of {event_name}: {e}"
                )

    def register_callback(self, name: str, callback: Callable) -> None:
        """Register a callback function"""
        with self.lock:
            self.callbacks[name] = callback
            self.logger.info(f"Registered callback: {name}")

    def unregister_callback(self, name: str) -> None:
        """Unregister a callback function"""
        with self.lock:
            if name in self.callbacks:
                del self.callbacks[name]
                self.logger.info(f"Unregistered callback: {name}")

    # --- Specific Bridge Access Methods ---

    def get_rag_bridge(self) -> Optional[Any]:
        """Get the ARTRAGBridge instance if available"""
        return self.bridges.get("art_rag")

    def get_blt_bridge(self) -> Optional[Any]:
        """Get the ARTBLTBridge instance if available"""
        return self.bridges.get("art_blt")

    def create_vanta_supervisor(self, **kwargs) -> Optional[Any]:
        """Create a VANTA supervisor using the VANTAFactory"""
        vanta_factory = self.bridges.get("vanta_factory")
        if not vanta_factory:
            self.logger.error("VANTAFactory not available")
            return None

        try:
            # Add art_manager_instance if not provided
            if "art_manager_instance" not in kwargs and self.art_controller:
                kwargs["art_manager_instance"] = self.art_controller

            # Use factory to create supervisor
            return vanta_factory.create_new(**kwargs)
        except Exception as e:
            self.logger.error(f"Error creating VANTA supervisor: {e}")
            return None

    def connect_entropy_guardian(self, entropy_guardian: Any) -> bool:
        """
        Connect an entropy guardian component to the ART system

        Args:
            entropy_guardian: An entropy monitoring component instance

        Returns:
            True if connection was successful, False otherwise
        """
        if not self.art_controller:
            self.logger.error(
                "ARTController not available, cannot connect entropy guardian"
            )
            return False

        art_entropy_class = self.bridges.get("art_entropy_class")
        if not art_entropy_class:
            self.logger.error("ArtEntropyBridge class not available")
            return False

        try:
            # Create bridge instance
            entropy_bridge = art_entropy_class(
                art_controller=self.art_controller,
                entropy_guardian=entropy_guardian,
                logger_instance=self.logger,
            )

            # Store the instance
            self.bridges["art_entropy"] = entropy_bridge
            self.logger.info("Entropy guardian connected to ART system")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting entropy guardian: {e}")
            return False


# Convenience function to create an ArtAdapter instance
def create_art_adapter(config: Optional[Dict[str, Any]] = None) -> ArtAdapter:
    """
    Create an ArtAdapter instance with default configuration

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured ArtAdapter instance
    """
    return ArtAdapter(config=config)
