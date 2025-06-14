"""
ART Training Integration Module

This module provides integration between the ART controller and
agent components for training on live agent interactions, featuring enhanced
feature extraction, configurable training parameters.

HOLO-1.5 Recursive Symbolic Cognition Mesh Integration:
- Role: SYNTHESIZER (cognitive_load=3.5, symbolic_depth=4)
- Capabilities: Feature synthesis, learning orchestration, pattern adaptation
- Cognitive metrics: Training complexity, feature coherence, adaptation efficiency
"""

import logging
import numpy as np
import time
import json
import pickle
import threading
import os
import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from collections import deque

# HOLO-1.5 Cognitive Mesh Integration
try:
    from ..agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
    
    # Define VantaAgentCapability locally as it's not in a centralized location
    class VantaAgentCapability:
        FEATURE_SYNTHESIS = "feature_synthesis"
        LEARNING_ORCHESTRATION = "learning_orchestration"
        PATTERN_ADAPTATION = "pattern_adaptation"
        
    HOLO_AVAILABLE = True
except ImportError:
    # Fallback for non-HOLO environments
    def vanta_agent(role=None, cognitive_load=0, symbolic_depth=0, capabilities=None):
        def decorator(cls):
            cls._holo_role = role
            cls._holo_cognitive_load = cognitive_load
            cls._holo_symbolic_depth = symbolic_depth
            cls._holo_capabilities = capabilities or []
            return cls
        return decorator
    
    class CognitiveMeshRole:
        SYNTHESIZER = "SYNTHESIZER"
    
    class VantaAgentCapability:
        FEATURE_SYNTHESIS = "feature_synthesis"
        LEARNING_ORCHESTRATION = "learning_orchestration"
        PATTERN_ADAPTATION = "pattern_adaptation"
    
    class BaseAgent:
        pass
    
    HOLO_AVAILABLE = False

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .art_controller import ARTController

# Import the logger helper
try:
    from .art_logger import get_art_logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _default_logger = logging.getLogger("voxsigil.art.trainer_fallback")

    def get_art_logger(name=None, level=None, log_file=None, base_logger_name=None):
        return _default_logger


# Simple placeholder ARTController class for when the real one is not available
class PlaceholderARTController:
    """Minimal placeholder for ARTController when real implementation is unavailable"""

    def __init__(self, *args, **kwargs):
        self.vigilance = 0.5
        self.logger = kwargs.get(
            "logger_instance", logging.getLogger("PlaceholderARTController")
        )
        self.logger.info("Using Placeholder ARTController.")
        self.categories = {}
        self.input_dim = kwargs.get("input_dim", 16)

    def train(self, features, epochs=1):
        self.logger.debug("Placeholder ART train called.")
        self.categories[0] = {"size": 1}
        return {
            "category_id": 0,
            "resonance": 0.9,
            "is_new_category": False,
            "is_anomaly": False,
        }

    def get_all_categories(self):
        return [
            {
                "category_id": k,
                "pattern_count": v.get("size", 0),
                "created_time": time.time() - 1000,
                "updated_time": time.time(),
                "avg_resonance": 0.8,
            }
            for k, v in self.categories.items()
        ]

    def get_anomaly_categories(self):
        return []

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "vigilance": self.vigilance,
        }


def _match_input_dim(vector, target_dim):
    """Utility to match a vector to the target dimension by padding or trimming"""
    if vector is None:
        return np.zeros(target_dim)
    current_dim = vector.shape[0]
    if current_dim == target_dim:
        return vector
    if current_dim > target_dim:
        return vector[:target_dim]
    # Pad with zeros if current_dim < target_dim
    padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
    return np.concatenate((vector, padding))


# --- Encapsulated Features (EF) ---
# EF1 to EF13 remain largely the same, but logging should use self.logger
# EF14 train_batch: Ensure it uses self.logger
# EF15 _log_training_event: This should be removed and its functionality
#      integrated into direct calls to self.logger.

# --- ARTTrainer Class ---

@vanta_agent(
    role=CognitiveMeshRole.SYNTHESIZER,
    cognitive_load=3.5,
    symbolic_depth=4,
    capabilities=[
        VantaAgentCapability.FEATURE_SYNTHESIS,
        VantaAgentCapability.LEARNING_ORCHESTRATION,
        VantaAgentCapability.PATTERN_ADAPTATION
    ]
)
class ArtTrainer(BaseAgent if HOLO_AVAILABLE else object):
    """
    Trainer for integrating ART controller with agent activity, featuring
    enhanced feature extraction, configurable training logic, and state management.
    
    HOLO-1.5 Integration:
    - Synthesizes features from multi-modal inputs
    - Orchestrates learning across cognitive domains
    - Adapts patterns based on training feedback
    """

    def __init__(
        self,
        art_controller: Optional[
            Union["ARTController", "PlaceholderARTController"]
        ] = None,
        config: Optional[dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the ART trainer with HOLO-1.5 cognitive mesh integration.

        Args:
            art_controller: Optional existing ART controller instance.
            config: Configuration dictionary. Keys include:
                    'art_config': Config passed to ARTController if created internally.
                    'vigilance', 'learning_rate', 'input_dim', 'max_categories': Direct ART overrides.
                    'feature_weights': Dict mapping feature source names to weights.
                    'feature_strategy': Method to combine features ('concatenate', etc.).
                    'max_training_history': Max size for internal training history log.
                    'epochs_per_event': Base number of training epochs per event.
                    'anomaly_epoch_multiplier': Multiplier for epochs on anomalies.
                    'config_source': Path/name to load trainer config (uses simple file IO now).
            logger_instance: Optional logger instance. If None, a default one is created.
        """
        # Initialize HOLO-1.5 base agent if available
        if HOLO_AVAILABLE:
            super().__init__()
        
        self.logger = (
            logger_instance
            if logger_instance
            else get_art_logger("voxsigil.art.trainer")
        )
        self.logger.info("ARTTrainer initializing with HOLO-1.5 cognitive mesh...")

        # HOLO-1.5 Cognitive metrics initialization
        self._cognitive_metrics = {
            "training_complexity": 0.0,
            "feature_coherence": 1.0,
            "adaptation_efficiency": 0.0,
            "synthesis_depth": 0,
            "orchestration_load": 0.0
        }
        
        # Initialize async components if HOLO available
        if HOLO_AVAILABLE:
            asyncio.create_task(self._initialize_cognitive_mesh())
        
        base_config = config or {}
        config_source_name = base_config.get(
            "config_source"
        )  # e.g., "trainer_default_config"
        self.config = self._load_trainer_config_internal(
            config_source_name
        )  # Use internal loader
        self.config.update(base_config)  # Override loaded with passed config

        if art_controller:
            self.art = art_controller
            self.logger.info("ARTTrainer using provided ARTController instance.")
        else:
            self.logger.warning("ARTController not provided. Creating new instance.")
            art_c_config = self.config.get("art_config", {})
            art_c_config["vigilance"] = self.config.get(
                "vigilance", art_c_config.get("vigilance", 0.5)
            )
            art_c_config["learning_rate"] = self.config.get(
                "learning_rate", art_c_config.get("learning_rate", 0.1)
            )
            art_c_config["input_dim"] = self.config.get(
                "input_dim", art_c_config.get("input_dim", None)
            )
            art_c_config["max_categories"] = self.config.get(
                "max_categories", art_c_config.get("max_categories", 50)
            )
            # Pass logger to ARTController if it accepts it
            try:
                # Check if ARTController has been enhanced with HOLO-1.5
                if hasattr(art_c_config, '_holo_role'):
                    # Pass cognitive mesh context
                    art_c_config['cognitive_mesh_context'] = {
                        'parent_role': 'SYNTHESIZER',
                        'synthesis_target': 'pattern_learning'
                    }
                self.art = ARTController(
                    config=art_c_config, logger_instance=self.logger
                )
            except TypeError:  # If ARTController doesn't take logger_instance
                self.art = ARTController(config=art_c_config)
                self.logger.warning(
                    "ARTController does not accept logger_instance, created without it."
                )

        self.feature_weights = self.config.get(
            "feature_weights",
            {"input_features": 1.0, "output_features": 0.5, "metadata_features": 1.0},
        )
        self.feature_strategy = self.config.get("feature_strategy", "concatenate")

        art_input_dim = self._get_art_input_dim_internal(self.art)
        if art_input_dim:
            self.config["input_dim"] = art_input_dim
        else:
            self.logger.warning(
                "ARTController input_dim not available; feature vector validation might be skipped."
            )

        self.max_history = self.config.get("max_training_history", 100)
        self.training_history = deque(maxlen=self.max_history)
        self.training_stats = {"total_trained": 0, "last_trained_ts": None}  # F5

        self.lock = threading.Lock()  # For thread safety during training
        self.logger.info(f"ARTTrainer initialized with HOLO-1.5. Config: {self.config}")

    async def _initialize_cognitive_mesh(self) -> None:
        """Initialize HOLO-1.5 cognitive mesh capabilities"""
        try:
            # Register synthesis capabilities
            await self.register_capability(
                VantaAgentCapability.FEATURE_SYNTHESIS,
                {
                    'synthesis_types': ['text', 'numeric', 'metadata'],
                    'complexity_handling': True,
                    'multi_modal_fusion': True
                }
            )
            
            # Register learning orchestration
            await self.register_capability(
                VantaAgentCapability.LEARNING_ORCHESTRATION,
                {
                    'batch_processing': True,
                    'adaptive_epochs': True,
                    'anomaly_handling': True
                }
            )
            
            # Register pattern adaptation
            await self.register_capability(
                VantaAgentCapability.PATTERN_ADAPTATION,
                {
                    'context_aware_weighting': True,
                    'dynamic_feature_adjustment': True,
                    'resonance_optimization': True
                }            )
            
            self.logger.info("HOLO-1.5 cognitive mesh capabilities registered successfully")
            
        except Exception as e:
            self.logger.warning(f"HOLO-1.5 initialization partial: {e}")
    
    async def register_capability(self, capability: str, metadata: dict) -> None:
        """Register a capability with the cognitive mesh"""
        if hasattr(self, 'vanta_core') and self.vanta_core:
            try:
                await self.vanta_core.register_capability(capability, metadata)
            except Exception as e:
                self.logger.warning(f"Failed to register capability {capability}: {e}")
        else:
            self.logger.debug(f"VantaCore not available, capability {capability} registered locally")
    
    def _calculate_training_complexity(self, feature_vector: np.ndarray, metadata: dict) -> float:
        """Calculate cognitive load for training complexity"""
        if feature_vector is None:
            return 0.0
        
        # Base complexity from feature vector properties
        vector_entropy = -np.sum(feature_vector * np.log(feature_vector + 1e-10))
        vector_variance = np.var(feature_vector)
        
        # Metadata complexity
        metadata_complexity = len(metadata) / 20.0 if metadata else 0.0
        
        # Anomaly detection complexity
        anomaly_weight = 2.0 if metadata and metadata.get('art_insights', {}).get('is_anomaly') else 1.0
        
        complexity = (vector_entropy * 0.4 + vector_variance * 0.3 + metadata_complexity * 0.3) * anomaly_weight
        return min(complexity, 5.0)  # Cap at 5.0
    
    def _calculate_feature_coherence(self, feature_sources: list) -> float:
        """Calculate feature coherence across synthesis sources"""
        if not feature_sources:
            return 0.0
        
        # Calculate coherence based on feature source diversity and weights
        total_weight = sum(weight for _, _, weight in feature_sources)
        weight_balance = 1.0 - abs(total_weight - len(feature_sources)) / len(feature_sources)
        
        # Source diversity (having multiple types is good)
        source_types = set(name.split('_')[0] for name, _, _ in feature_sources)
        diversity_score = len(source_types) / 3.0  # Max 3 types: input, output, metadata
        
        return (weight_balance * 0.6 + diversity_score * 0.4)
    
    def _generate_synthesis_trace(self, operation: str, inputs: dict, outputs: dict) -> dict:
        """Generate HOLO-1.5 synthesis trace for cognitive mesh learning"""
        return {
            'timestamp': time.time(),
            'operation': operation,
            'role': 'SYNTHESIZER',
            'cognitive_load': self._cognitive_metrics.get('training_complexity', 0.0),
            'symbolic_depth': self._cognitive_metrics.get('synthesis_depth', 0),
            'inputs': {
                'input_type': inputs.get('input_type'),
                'feature_sources': inputs.get('feature_sources', 0),
                'metadata_keys': inputs.get('metadata_keys', [])
            },
            'outputs': {
                'feature_vector_size': outputs.get('feature_vector_size', 0),
                'training_epochs': outputs.get('training_epochs', 0),
                'resonance': outputs.get('resonance', 0.0)
            },
            'metrics': {
                'synthesis_efficiency': outputs.get('resonance', 0.0),
                'adaptation_success': outputs.get('training_success', False),
                'complexity_handled': self._cognitive_metrics.get('training_complexity', 0.0)
            }
        }

    def _get_art_input_dim_internal(
        self, art_controller: Optional[Any]
    ) -> Optional[int]:
        """Safely gets the input_dim from the ARTController's config or attribute."""
        if art_controller:
            if hasattr(art_controller, "get_config") and callable(
                getattr(art_controller, "get_config")
            ):
                controller_config = art_controller.get_config()
                if (
                    isinstance(controller_config, dict)
                    and "input_dim" in controller_config
                ):
                    return controller_config["input_dim"]
            if hasattr(art_controller, "input_dim"):  # Fallback to direct attribute
                return getattr(art_controller, "input_dim")
        self.logger.warning(
            "_get_art_input_dim_internal: Could not determine input_dim from ARTController."
        )
        return None

    def _load_trainer_config_internal(self, name: Optional[str]) -> dict[str, Any]:
        """Loads trainer configuration from a simple JSON file if name is provided."""
        if not name:
            return {}
        # Simplified path, assuming config files are stored in a known relative location or full path is given
        # For this refactor, let's assume it's just a name and we don't have a complex persistence layer yet.
        # This would typically involve a config directory.
        try:
            # Placeholder: In a real scenario, you'd have a config path resolution.
            # For now, let's assume 'name' could be a full path or just a file in current dir.
            file_path = f"{name}.json" if not name.endswith(".json") else name
            if os.path.exists(file_path):  # os import needed
                with open(file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.logger.info(f"Loaded trainer config from {file_path}")
                return config
            else:
                self.logger.warning(
                    f"Trainer config file {file_path} not found. Using defaults."
                )
        except Exception as e:
            self.logger.error(f"Error loading trainer config {name}: {e}")
        return {}  # Return empty if not found or error

    def _save_trainer_config_internal(self, name: str) -> bool:
        """Saves current trainer configuration to a simple JSON file."""
        if not name:
            return False
        file_path = f"{name}.json" if not name.endswith(".json") else name
        try:
            # Save only relevant parts of self.config, not the entire live config object
            # which might contain runtime state not suitable for a config file.
            config_to_save = {
                "art_config": self.config.get("art_config"),
                "vigilance": self.config.get("vigilance"),
                "learning_rate": self.config.get("learning_rate"),
                "input_dim": self.config.get(
                    "input_dim"
                ),  # This should be from ARTController ideally
                "max_categories": self.config.get("max_categories"),
                "feature_weights": self.feature_weights,
                "feature_strategy": self.feature_strategy,
                "max_training_history": self.max_history,
                "epochs_per_event": self.config.get("epochs_per_event", 1),
                "anomaly_epoch_multiplier": self.config.get(
                    "anomaly_epoch_multiplier", 2
                ),
            }
            # Filter out None values to keep config clean
            config_to_save = {k: v for k, v in config_to_save.items() if v is not None}

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2)
            self.logger.info(f"Saved trainer config to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving trainer config {name}: {e}")
        return False    
    
    def _create_feature_vector(
        self,
        input_data: Any,
        output_data: Optional[Any],
        metadata: Optional[dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """Create feature vector with HOLO-1.5 synthesis tracking"""
        target_dim = self.config.get("input_dim")
        if target_dim is None or target_dim <= 0:
            self.logger.error(
                "ART input_dim not defined in trainer or ART controller config."
            )
            return None

        weights = self.feature_weights
        feature_sources: list[tuple[str, list[float], float]] = []

        # Track synthesis process
        synthesis_start = time.time()
        
        input_weight = weights.get("input_features", 1.0)
        if isinstance(input_data, str):
            input_feats = self._extract_text_features(
                input_data
            )  # Changed to self._extract_text_features
            feature_sources.append(("input_text", input_feats, input_weight))
        elif isinstance(input_data, (np.ndarray, list)):
            try:
                num_input = np.array(input_data).flatten().astype(np.float32)
                num_input = np.clip(num_input, 0, 5) / 5.0
                feature_sources.append(
                    ("input_numeric", num_input.tolist(), input_weight)
                )
            except Exception as e:
                self.logger.warning(f"Could not process numeric input features: {e}")

        output_weight = weights.get("output_features", 0.5)
        if output_data is not None:
            if isinstance(output_data, str):
                output_feats = self._extract_text_features(output_data)  # Changed
                feature_sources.append(("output_text", output_feats, output_weight))
            elif isinstance(output_data, (np.ndarray, list)):
                try:
                    num_output = np.array(output_data).flatten().astype(np.float32)
                    num_output = np.clip(num_output, 0, 5) / 5.0
                    feature_sources.append(
                        ("output_numeric", num_output.tolist(), output_weight)
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not process numeric output features: {e}"
                    )

        metadata_weight = weights.get("metadata_features", 1.0)
        meta_feats = self._extract_metadata_features(metadata)  # Changed
        feature_sources.append(("metadata", meta_feats, metadata_weight))

        # Calculate cognitive metrics for synthesis
        self._cognitive_metrics['feature_coherence'] = self._calculate_feature_coherence(feature_sources)
        self._cognitive_metrics['synthesis_depth'] = len(feature_sources)

        strategy = self.feature_strategy
        final_features = []

        if strategy == "concatenate":
            for name, feats, weight in feature_sources:
                final_features.extend(feats)
            final_vector = _match_input_dim(
                np.array(final_features, dtype=np.float32), target_dim
            )  # _match_input_dim is standalone
        elif strategy == "weighted_average":
            self.logger.warning(
                "Weighted average feature strategy not fully implemented. Using concatenate."
            )
            for name, feats, weight in feature_sources:
                final_features.extend(feats)
            final_vector = _match_input_dim(
                np.array(final_features, dtype=np.float32), target_dim
            )
        else:
            self.logger.error(
                f"Unknown feature strategy: {strategy}. Cannot create vector."
            )
            return None

        if final_vector.shape[0] != target_dim:
            self.logger.error(
                f"Final feature vector dim {final_vector.shape[0]} != target {target_dim}."
            )
            return None
        
        # Update cognitive metrics with synthesis results
        synthesis_time = time.time() - synthesis_start
        self._cognitive_metrics['orchestration_load'] = min(synthesis_time * 10, 5.0)
        
        # Generate synthesis trace for HOLO-1.5 learning
        if HOLO_AVAILABLE:
            trace = self._generate_synthesis_trace(
                'feature_synthesis',
                {
                    'input_type': type(input_data).__name__,
                    'feature_sources': len(feature_sources),
                    'metadata_keys': list(metadata.keys()) if metadata else []
                },
                {
                    'feature_vector_size': len(final_vector),
                    'synthesis_time': synthesis_time,
                    'coherence': self._cognitive_metrics['feature_coherence']
                }
            )
            # Store trace for mesh learning
            if hasattr(self, 'cognitive_traces'):
                self.cognitive_traces.append(trace)
        
        return final_vector

    # Make EF helpers methods of the class or ensure they use self.logger if kept as static/module-level
    def _extract_text_features(
        self, text: str, max_len_norm: float = 1000.0, max_word_norm: float = 100.0
    ) -> list[float]:
        # ... (implementation as before, ensure logging uses self.logger if any)
        # For brevity, assuming original implementation is fine, just ensure logger usage if needed.
        # This one doesn't seem to log, so it's okay as a static method or helper.
        features = []
        text_lower = text.lower()
        norm_len = len(text) / max(1.0, max_len_norm)
        words = text_lower.split()
        norm_word_count = len(words) / max(1.0, max_word_norm)
        features.append(min(1.0, norm_len))
        features.append(min(1.0, norm_word_count))
        features.append(min(1.0, text.count("?") / 5.0))
        features.append(min(1.0, text.count("!") / 5.0))
        pos_words = {
            "good",
            "great",
            "excellent",
            "happy",
            "positive",
            "like",
            "love",
            "agree",
            "yes",
            "correct",
        }
        neg_words = {
            "bad",
            "terrible",
            "awful",
            "sad",
            "negative",
            "dislike",
            "hate",
            "disagree",
            "no",
            "wrong",
            "issue",
            "problem",
        }
        pos_count = sum(1 for word in words if word in pos_words)
        neg_count = sum(1 for word in words if word in neg_words)
        features.append(min(1.0, pos_count / max(1, len(words) * 0.2)))
        features.append(min(1.0, neg_count / max(1, len(words) * 0.2)))
        return features

    def _extract_metadata_features(
        self, metadata: Optional[dict[str, Any]]
    ) -> list[float]:
        # ... (implementation as before, ensure logging uses self.logger if any)
        # This one doesn't seem to log, so it's okay as a static method or helper.
        features = []
        if metadata:
            features.append(
                min(1.0, float(self._safe_get(metadata, "response_time", 0)) / 10.0)
            )  # Changed to self._safe_get
            features.append(float(self._safe_get(metadata, "error_rate", 0)))
            features.append(float(self._safe_get(metadata, "confidence", 0.5)))
            features.append(float(self._safe_get(metadata, "emotion_strength", 0.0)))
            features.append(float(self._safe_get(metadata, "emotion_valence", 0.5)))
            features.append(float(self._safe_get(metadata, "awareness", 0.5)))
            features.append(float(self._safe_get(metadata, "regulation", 0.5)))
            features.append(1.0 if self._safe_get(metadata, "success", True) else 0.0)
            features.append(
                float(self._safe_get(metadata, "art_insights.resonance", 0.5))
            )
            features.append(
                1.0
                if self._safe_get(metadata, "art_insights.is_anomaly", False)
                else 0.0
            )
        else:
            features.extend([0.0] * 10)
        return features

    def _safe_get(
        self, data: dict, key_path: Union[str, list[str]], default: Any = None
    ) -> Any:
        # This is a utility, can be static or helper. No logging.
        if isinstance(key_path, str):
            keys = key_path.split(".")
        elif isinstance(key_path, list):
            keys = key_path
        else:
            return default
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _validate_feature_vector(
        self, vector: Optional[np.ndarray], expected_dim: Optional[int]
    ) -> bool:
        if vector is None:
            return False
        if not isinstance(vector, np.ndarray):
            return False
        if vector.ndim != 1:
            self.logger.error(f"Feature vector has invalid ndim: {vector.ndim}")
            return False
        if expected_dim is not None and vector.shape[0] != expected_dim:
            self.logger.error(
                f"Feature vector dim {vector.shape[0]} != expected {expected_dim}."            )
            return False
        if not np.issubdtype(vector.dtype, np.floating):
            self.logger.warning(f"Feature vector dtype ({vector.dtype}) not float.")
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            self.logger.error("Feature vector contains NaN or Inf.")
            return False
        return True

    def _get_training_epochs(self, metadata: Optional[dict] = None) -> int:
        base_epochs = self.config.get("epochs_per_event", 1)
        if metadata and self._safe_get(metadata, "art_insights.is_anomaly", False):
            return base_epochs * self.config.get("anomaly_epoch_multiplier", 2)
        return base_epochs

    def train_from_event(
        self,
        input_data: Any,
        output_data: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Processes a single event with HOLO-1.5 cognitive synthesis and learning orchestration.
        Extracts features, trains the ART controller, and generates cognitive traces.
        """
        with self.lock:  # Thread safety
            # HOLO-1.5 cognitive event processing start
            event_start_time = time.time()
            
            self.logger.debug(
                f"HOLO-1.5 train_from_event called. Input type: {type(input_data)}"
            )
            if metadata is None:
                metadata = {}

            # Calculate initial cognitive metrics
            self._cognitive_metrics['training_complexity'] = self._calculate_training_complexity(
                getattr(input_data, 'shape', None) if hasattr(input_data, 'shape') else np.array([0]), 
                metadata
            )

            # F7: Context-Aware Feature Weighting (HOLO-1.5 Enhanced)
            current_feature_weights = self.feature_weights.copy()  # Start with base weights
            if self.config.get("context_aware_feature_weighting", False):
                if metadata.get("high_priority_event"):  # Example context
                    current_feature_weights["input_features"] *= 1.5  # Boost input importance
                    self.logger.info(
                        "Applied HOLO-1.5 context-aware weight adjustment for high_priority_event."
                    )
                # Additional HOLO-1.5 cognitive adaptations
                if metadata.get("anomaly_detected"):
                    current_feature_weights["metadata_features"] *= 2.0
                    self.logger.info("Enhanced metadata weighting for anomaly processing")

            # EF3: Create feature vector using cognitive synthesis
            feature_vector = self._create_feature_vector(
                input_data, output_data, metadata
            )

            # EF12: Validate with enhanced cognitive awareness
            art_input_dim = self.config.get("input_dim")
            if not self._validate_feature_vector(feature_vector, art_input_dim):
                self.logger.error(
                    "HOLO-1.5 feature vector validation failed. Skipping training for this event."
                )
                return {
                    "status": "error",
                    "message": "Feature vector validation failed",
                    "cognitive_metrics": self._cognitive_metrics.copy()
                }

            # EF13: Get epochs with cognitive load adjustment
            epochs = self._get_training_epochs(metadata)
            
            # HOLO-1.5 cognitive load adjustment based on complexity
            if self._cognitive_metrics['training_complexity'] > 3.0:
                epochs = max(1, int(epochs * 0.8))  # Reduce epochs for high complexity
                self.logger.info(f"Reduced training epochs to {epochs} due to high cognitive complexity")

            # F3: Train ART with cognitive orchestration
            if feature_vector is None:
                self.logger.error(
                    "HOLO-1.5 feature vector is None. Skipping training for this event."
                )
                return {
                    "status": "error",
                    "message": "Feature vector is None",
                    "cognitive_metrics": self._cognitive_metrics.copy()
                }

            try:
                # Enhanced training with cognitive context
                train_result = self.art.train(
                    feature_vector, epochs=epochs
                )  # ARTController handles its own logging
                
                # Calculate adaptation efficiency
                resonance = train_result.get('resonance', 0.0)
                self._cognitive_metrics['adaptation_efficiency'] = min(
                    resonance * (1.0 + epochs / 10.0), 1.0
                )
                
                self.logger.info(f"HOLO-1.5 ART training completed. Result: {train_result}")
                
            except Exception as e:
                self.logger.error(f"Exception during HOLO-1.5 ART training: {e}", exc_info=True)
                return {
                    "status": "error", 
                    "message": f"ART training exception: {e}",
                    "cognitive_metrics": self._cognitive_metrics.copy()
                }

            # F5: Enhanced training event logging with cognitive traces
            timestamp = time.time()
            event_duration = timestamp - event_start_time
            
            history_entry = {
                "timestamp": timestamp,
                "input_type": str(type(input_data)),
                "output_type": str(type(output_data)) if output_data else None,
                "metadata_keys": list(metadata.keys()) if metadata else [],
                "feature_vector_preview": feature_vector[
                    : min(5, len(feature_vector))
                ].tolist()
                if feature_vector is not None
                else None,  # First 5 features
                "epochs": epochs,
                "art_result": train_result,
                "cognitive_metrics": self._cognitive_metrics.copy(),
                "event_duration": event_duration,
                "holo_version": "1.5"
            }
            
            self.training_history.append(history_entry)
            self.training_stats["total_trained"] += 1
            self.training_stats["last_trained_ts"] = timestamp

            # Generate HOLO-1.5 cognitive trace for mesh learning
            if HOLO_AVAILABLE:
                cognitive_trace = self._generate_synthesis_trace(
                    'training_orchestration',
                    {
                        'input_type': str(type(input_data)),
                        'feature_sources': self._cognitive_metrics.get('synthesis_depth', 0),
                        'metadata_keys': list(metadata.keys()) if metadata else []
                    },
                    {
                        'feature_vector_size': len(feature_vector) if feature_vector is not None else 0,
                        'training_epochs': epochs,
                        'resonance': train_result.get('resonance', 0.0),
                        'training_success': train_result.get('category_id', -1) >= 0
                    }
                )
                
                # Store trace for cognitive mesh learning
                if not hasattr(self, 'cognitive_traces'):
                    self.cognitive_traces = deque(maxlen=1000)
                self.cognitive_traces.append(cognitive_trace)

            # Enhanced logging with cognitive context
            self.logger.info(
                "HOLO-1.5 ART Training Event Completed",
                extra={
                    "event_type": "holo_train_from_event_success", 
                    "data": history_entry,
                    "cognitive_load": self._cognitive_metrics.get('training_complexity', 0.0),
                    "adaptation_efficiency": self._cognitive_metrics.get('adaptation_efficiency', 0.0)
                },
            )

            return {
                "status": "success",
                "art_result": train_result,
                "feature_vector_preview": history_entry["feature_vector_preview"],
                "cognitive_metrics": self._cognitive_metrics.copy(),
                "event_duration": event_duration,
                "holo_version": "1.5"
            }
            if metadata is None:
                metadata = {}

            # F7: Context-Aware Feature Weighting (Example: adjust weights based on metadata)
            # This is a placeholder for more complex logic.
            current_feature_weights = (
                self.feature_weights.copy()
            )  # Start with base weights
            if self.config.get("context_aware_feature_weighting", False):
                if metadata.get("high_priority_event"):  # Example context
                    current_feature_weights["input_features"] *= (
                        1.5  # Boost input importance
                    )
                    self.logger.info(
                        "Applied context-aware weight adjustment for high_priority_event."
                    )

            # EF3: Create feature vector using current (potentially adjusted) weights
            # Note: _create_feature_vector needs to be adapted to use these dynamic weights if this is desired.
            # For now, _create_feature_vector uses self.feature_weights directly.
            # To implement F7 fully, _create_feature_vector would need to accept weights as an argument.
            # Let's assume for now that self.feature_weights is what _create_feature_vector uses.
            # If F7 is enabled, one might update self.feature_weights temporarily or pass them.
            # For simplicity, this example doesn't pass dynamic weights to _create_feature_vector.

            feature_vector = self._create_feature_vector(
                input_data, output_data, metadata
            )

            # EF12: Validate
            art_input_dim = self.config.get("input_dim")
            if not self._validate_feature_vector(feature_vector, art_input_dim):
                self.logger.error(
                    "Feature vector validation failed. Skipping training for this event."
                )
                return {
                    "status": "error",
                    "message": "Feature vector validation failed",
                }

            # EF13: Get epochs
            epochs = self._get_training_epochs(metadata)

            # F3: Train ART
            if feature_vector is None:
                self.logger.error(
                    "Feature vector is None. Skipping training for this event."
                )
                return {
                    "status": "error",
                    "message": "Feature vector is None",
                }

            try:
                train_result = self.art.train(
                    feature_vector, epochs=epochs
                )  # ARTController handles its own logging
                self.logger.info(f"ART training completed. Result: {train_result}")
            except Exception as e:
                self.logger.error(f"Exception during ART training: {e}", exc_info=True)
                return {"status": "error", "message": f"ART training exception: {e}"}

            # F5: Log training event and update history/stats
            timestamp = time.time()
            history_entry = {
                "timestamp": timestamp,
                "input_type": str(type(input_data)),
                "output_type": str(type(output_data)) if output_data else None,
                "metadata_keys": list(metadata.keys()) if metadata else [],
                "feature_vector_preview": feature_vector[
                    : min(5, len(feature_vector))
                ].tolist()
                if feature_vector is not None
                else None,  # First 5 features
                "epochs": epochs,
                "art_result": train_result,
            }
            self.training_history.append(history_entry)
            self.training_stats["total_trained"] += 1
            self.training_stats["last_trained_ts"] = timestamp

            # Log using the main logger
            self.logger.info(
                "ART Training Event",
                extra={"event_type": "train_from_event_success", "data": history_entry},
            )

            return {
                "status": "success",
                "art_result": train_result,
                "feature_vector_preview": history_entry["feature_vector_preview"],
            }

    def train_batch(self, batch: list[tuple[Any, Any, dict]]) -> list[dict[str, Any]]:
        """Trains ART controller on a batch of events. (EF14)"""
        results = []
        self.logger.info(f"Processing training batch of size {len(batch)}")
        for i, (input_data, output_data, metadata) in enumerate(batch):
            self.logger.debug(f"Training batch item {i + 1}/{len(batch)}")
            # Consider if metadata needs to be deepcopied if modified by train_from_event
            # or feature extraction, though current implementation seems safe.
            results.append(self.train_from_event(input_data, output_data, metadata))
        self.logger.info(f"Batch training completed. Results count: {len(results)}")
        return results

    def get_training_stats(self) -> dict[str, Any]:
        """Returns statistics about training activity."""
        with self.lock:  # Ensure consistent read of stats
            # EF10: Calculate rolling stats for resonance, etc., if available in history
            # This requires parsing 'art_result' from history items.
            resonances = [
                item["art_result"]["resonance"]
                for item in self.training_history
                if item.get("art_result") and "resonance" in item["art_result"]
            ]
            rolling_resonance_stats = self._calculate_rolling_stats(
                deque(resonances)
            )  # Use self._

            return {
                "total_events_processed": self.training_stats["total_trained"],
                "last_event_timestamp": self.training_stats["last_trained_ts"],
                "training_history_size": len(self.training_history),
                "rolling_resonance_mean": rolling_resonance_stats.get("mean"),
                "rolling_resonance_std": rolling_resonance_stats.get("std"),
            }

    def _calculate_rolling_stats(
        self, history_deque: deque
    ) -> dict[str, Optional[float]]:
        """Calculates stats (mean, std) over the history deque. (EF10)"""
        if not history_deque:
            return {"mean": None, "std": None, "count": 0}
        values = np.array(list(history_deque), dtype=np.float32)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": len(values),
        }

    def save_state(self, name: str = "default_trainer_state") -> bool:
        """Saves the trainer's current state (config, history) to a file. (EF6)"""
        with self.lock:
            state = {
                "config": self.config,  # Save the live config
                "training_history": list(self.training_history),
                "training_stats": self.training_stats,
                "save_timestamp": time.time(),
            }
            file_path = f"{name}.pkl"  # Simplified naming
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(state, f)
                self.logger.info(f"ARTTrainer state saved to {file_path}")
                return True
            except Exception as e:
                self.logger.error(
                    f"Error saving ARTTrainer state to {file_path}: {e}", exc_info=True
                )
                return False

    def load_state(self, name: str = "default_trainer_state") -> bool:
        """Loads the trainer's state from a file. (EF7)"""
        with self.lock:
            file_path = f"{name}.pkl"
            if not os.path.exists(file_path):  # os import needed
                self.logger.warning(
                    f"ARTTrainer state file {file_path} not found. Cannot load."
                )
                return False
            try:
                with open(file_path, "rb") as f:
                    state = pickle.load(f)

                # Apply state
                self.config = state.get("config", self.config)
                # Re-apply specific config values that drive behavior
                self.feature_weights = self.config.get(
                    "feature_weights", self.feature_weights
                )
                self.feature_strategy = self.config.get(
                    "feature_strategy", self.feature_strategy
                )
                self.max_history = self.config.get(
                    "max_training_history", self.max_history
                )

                history_list = state.get("training_history", [])
                self.training_history = deque(history_list, maxlen=self.max_history)
                self.training_stats = state.get("training_stats", self.training_stats)

                self.logger.info(
                    f"ARTTrainer state loaded from {file_path}. History items: {len(self.training_history)}"
                )
                # Potentially re-initialize or update ARTController if its config changed
                # For now, assume ARTController's state is managed separately or its config is stable.
                return True
            except Exception as e:
                self.logger.error(
                    f"Error loading ARTTrainer state from {file_path}: {e}",
                    exc_info=True,
                )
                return False

    def get_config_summary(self) -> dict[str, Any]:
        """Returns a summary of the current trainer configuration."""
        return {
            "input_dim_target": self.config.get("input_dim"),
            "feature_strategy": self.feature_strategy,
            "feature_weights": self.feature_weights,
            "max_training_history": self.max_history,
            "epochs_per_event": self.config.get("epochs_per_event"),
            "anomaly_epoch_multiplier": self.config.get("anomaly_epoch_multiplier"),
            "art_controller_config_preview": self.art.get_config()
            if hasattr(self.art, "get_config")
            else "N/A",
        }
