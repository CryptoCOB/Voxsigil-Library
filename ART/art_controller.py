"""
Adaptive Resonance Theory (ART) Controller

This module implements an enhanced ART neural network for unsupervised category
learning, pattern recognition, novelty detection, and adaptable resonance.
"""

import json
import logging  # Standard logging
import pickle
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

# --- VoxSigil ART Module Imports (relative imports for components within the same package) ---
from .art_logger import get_art_logger  # Use the new logger

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .art_trainer import ArtTrainer


# --- Locally Defined Math Utilities (formerly from MetaConsciousness SDK) ---
def calculate_ema(current_value: float, previous_ema: float, alpha: float) -> float:
    """Calculates the Exponential Moving Average."""
    return alpha * current_value + (1 - alpha) * previous_ema


def calculate_moving_average(
    data: list[float], window_size: Optional[int] = None
) -> float:
    """Calculates the moving average of a list of numbers."""
    if not data:
        return 0.0
    if window_size is None or window_size <= 0 or window_size > len(data):
        window_size = len(data)
    return float(np.mean(data[-window_size:])) if data else 0.0


# --- Encapsulated Features ---


# EF1: Input Vector Normalization (L2 Norm)
def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalizes a vector using L2 norm. Returns zero vector if norm is zero."""
    norm = np.linalg.norm(vector)
    if norm > 1e-9:  # Avoid division by zero
        return vector / norm
    else:
        # logger.debug("Input vector norm is zero, returning zero vector.") # Can be noisy
        return np.zeros_like(vector)


# EF2: Input Dimension Padding/Trimming
def _match_input_dim(input_vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Pads with zeros or trims the input vector to match the target dimension."""
    current_dim = input_vector.shape[0]
    if current_dim == target_dim:
        return input_vector
    elif current_dim < target_dim:
        # Pad with zeros
        padding = np.zeros(target_dim - current_dim, dtype=input_vector.dtype)
        return np.concatenate([input_vector, padding])
    else:  # current_dim > target_dim
        # Trim (take first elements)
        return input_vector[:target_dim]


# EF3: Vigilance Check Function
def _check_vigilance(match_score: float, vigilance_threshold: float) -> bool:
    """Performs the vigilance test."""
    return match_score >= vigilance_threshold


# EF4: ART Weight Update Rule (Fast Learning)
def _update_weights_fast(
    weights: np.ndarray, input_pattern: np.ndarray, learning_rate: float
) -> np.ndarray:
    """Applies the fast ART weight update rule (conjunction)."""
    # W_new = learning_rate * (Input AND W_old) + (1 - learning_rate) * W_old
    # For normalized binary/real ART, this simplifies if lr=1 (fast learning): W_new = Input AND W_old
    # Using element-wise minimum for AND operation with real values [0, 1]
    # Assume input_pattern and weights are normalized and >= 0
    if learning_rate == 1.0:  # Fast learning
        updated_weights = np.minimum(weights, input_pattern)
    else:  # Slow learning / averaging
        updated_weights = (1 - learning_rate) * weights + learning_rate * np.minimum(
            weights, input_pattern
        )
        # Re-normalize after update? Typically not done in standard ART update.
    return updated_weights


# EF5: Cosine Similarity Calculation
def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors, handling zero norms."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0  # Similarity is zero if either vector is zero
    # Normalize manually to ensure consistency
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    similarity = np.dot(vec1_norm, vec2_norm)
    return max(0.0, float(similarity))  # Ensure non-negative, return float


# EF6: Category Pruning Criteria (Example: Low Count)
def _check_pruning_criteria(
    category_info: dict[str, Any],
    min_count: int = 3,
    max_age_seconds: Optional[float] = None,
    current_logger: Optional[logging.Logger] = None,
) -> bool:  # Added logger
    """Checks if a category meets criteria for pruning (e.g., low count, old age)."""
    log_target = (
        current_logger if current_logger else logging
    )  # Fallback to default logging if no logger passed
    if category_info.get("pattern_count", 0) < min_count:
        log_target.debug(
            f"Category {category_info.get('category_id')} flagged for pruning (count < {min_count})."
        )
        return True
    if max_age_seconds is not None:
        age = time.time() - category_info.get("updated_time", 0)
        if age > max_age_seconds:
            log_target.debug(
                f"Category {category_info.get('category_id')} flagged for pruning (age > {max_age_seconds}s)."
            )
            return True
    return False


# EF7: Safe Dictionary Access Helper
def _safe_get(data: dict, key_path: Union[str, list[str]], default: Any = None) -> Any:
    """Safely gets a value from a nested dictionary using a dot-separated key path or list."""
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


# EF8: Load ART Config
def _load_art_config(
    config_source: Union[str, dict[str, Any]],
    current_logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:  # Added logger
    """Loads ART-specific configuration."""
    log_target = current_logger if current_logger else logging
    if isinstance(config_source, str):
        try:
            with open(config_source, "r") as f:
                config = json.load(f)
                log_target.info(f"Loaded ART config from {config_source}")
                return config
        except Exception as e:
            log_target.error(f"Failed load ART config from {config_source}: {e}")
            return {}
    elif isinstance(config_source, dict):
        return config_source.copy()
    else:
        log_target.warning("Invalid ART config_source type.")
        return {}


# EF9: Save ART Config
def _save_art_config(
    config: dict[str, Any],
    file_path: str,
    current_logger: Optional[logging.Logger] = None,
) -> bool:  # Added logger
    """Saves ART-specific configuration."""
    log_target = current_logger if current_logger else logging
    try:
        with open(file_path, "w") as f:
            json.dump(config, f, indent=4)
            log_target.info(f"Saved ART config to {file_path}")
            return True  # noqa
    except Exception as e:
        log_target.error(f"Failed save ART config to {file_path}: {e}")
        return False


# EF10: Serialize ART State
def _serialize_art_state(controller: "ARTController") -> dict[str, Any]:
    """Serializes the crucial state of the ART controller."""
    # Note: Weights can be large. Consider alternatives for very large networks.
    # Avoid saving the full stats dict, just core state needed to reconstruct.
    return {
        "config": controller.get_config(),  # Use get_config F5
        "weights": controller.weights.tolist()
        if controller.weights.size > 0
        else [],  # Convert weights to list
        "category_counts": controller.category_counts,
        "category_created": controller.category_created,
        "category_updated": controller.category_updated,
        "category_resonance": controller.category_resonance,
        "total_patterns": controller.total_patterns,
        # Don't save full stats, maybe just running avg resonance?
        "avg_resonance_running": controller.stats.get("avg_resonance", 0.0),
        "save_timestamp": time.time(),
    }


# EF11: Deserialize ART State
def _deserialize_art_state(
    controller: "ARTController", state: dict[str, Any]
) -> None:  # Logger will be controller.logger
    """Applies deserialized state to the ART controller."""
    # Load config first
    controller.config = state.get("config", controller.config)
    controller.vigilance = controller.config.get("vigilance", 0.5)
    controller.learning_rate = controller.config.get("learning_rate", 0.1)
    controller.input_dim = controller.config.get("input_dim", 128)
    controller.max_categories = controller.config.get("max_categories", 50)
    controller.anomaly_threshold = controller.config.get("anomaly_threshold", 0.3)
    controller.enable_pruning = controller.config.get("enable_pruning", False)  # F3
    controller.pruning_min_count = controller.config.get("pruning_min_count", 3)  # F3
    # Load state variables
    weights_list = state.get("weights", [])
    controller.weights = (
        np.array(weights_list, dtype=np.float64)
        if weights_list
        else np.zeros((0, controller.input_dim))
    )  # noqa
    controller.category_counts = state.get("category_counts", [])
    controller.category_created = state.get("category_created", [])
    controller.category_updated = state.get("category_updated", [])
    controller.category_resonance = state.get("category_resonance", [])
    controller.total_patterns = state.get("total_patterns", 0)
    # Reset runtime stats, optionally load running average
    controller._reset_runtime_stats()
    controller.stats["avg_resonance"] = state.get("avg_resonance_running", 0.0)
    # Ensure list lengths match category count derived from weights
    num_cats = controller.weights.shape[0]
    if len(controller.category_counts) != num_cats:
        controller.category_counts = [1] * num_cats  # Default if mismatch
    if len(controller.category_created) != num_cats:
        controller.category_created = [time.time()] * num_cats
    if len(controller.category_updated) != num_cats:
        controller.category_updated = [time.time()] * num_cats
    if len(controller.category_resonance) != num_cats:
        controller.category_resonance = [1.0] * num_cats
    controller.logger.info(
        f"ART state loaded. Categories: {num_cats}, Total Patterns: {controller.total_patterns}"
    )


# EF12: Recent Resonance Tracker
def _create_recent_resonance_tracker(size: int = 20) -> deque:
    """Creates a deque to track recent resonance values."""
    return deque(maxlen=size)


# EF13: Get Category Summary (Refined)
def _get_category_summary(
    controller: "ARTController",
) -> list[dict[str, Any]]:  # Logger will be controller.logger
    """Provides a summary list of category IDs and their counts/resonance."""
    summary = []
    with controller._lock:  # Added F8 Lock
        for i in range(controller.weights.shape[0]):
            summary.append(
                {
                    "category_id": i,
                    "pattern_count": controller.category_counts[i]
                    if i < len(controller.category_counts)
                    else 0,
                    "avg_resonance": controller.category_resonance[i]
                    if i < len(controller.category_resonance)
                    else 0.0,
                    "last_updated_ago_s": time.time() - controller.category_updated[i]
                    if i < len(controller.category_updated)
                    else -1,  # F10 Recency
                }
            )
    return summary


# EF14: Parameter Validation Helper for ART
def _validate_art_parameters(
    controller: "ARTController",
) -> None:  # Logger will be controller.logger
    """Validates key ART parameters are within sensible ranges."""
    if not (0 <= controller.vigilance <= 1):
        controller.logger.warning(
            f"Vigilance ({controller.vigilance}) out of range [0, 1]."
        )
    if not (0 <= controller.learning_rate <= 1):
        controller.logger.warning(
            f"Learning rate ({controller.learning_rate}) out of range [0, 1]."
        )
    if controller.input_dim is not None and controller.input_dim <= 0:
        controller.logger.error(f"Invalid input_dim: {controller.input_dim}")
    if controller.max_categories <= 0:
        controller.logger.warning(
            f"Max categories ({controller.max_categories}) should be positive."
        )
    if not (0 <= controller.anomaly_threshold <= 1):
        controller.logger.warning(
            f"Anomaly threshold ({controller.anomaly_threshold}) out of range [0, 1]."
        )


# EF15: Safe Array Indexing Helper
def _safe_list_get(lst: list, index: int, default: Any = None) -> Any:
    """Safely gets an element from a list by index."""
    try:
        return lst[index]
    except IndexError:
        return default


def _apply_bounds(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Apply bounds to a value
    """
    return max(min_val, min(max_val, value))


# --- ARTController Class ---


class ARTController:
    """
    Adaptive Resonance Theory (ART) neural network controller with enhancements.

    Implements ART-1 style learning with features like anomaly detection,
    category pruning, adaptive vigilance (placeholder hook), state persistence,
    and detailed statistics.
    """

    def __init__(
        self,
        vigilance: float = 0.5,
        learning_rate: float = 0.1,
        input_dim: Optional[int] = None,  # Allow None initially F1
        max_categories: int = 50,
        config: Optional[dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:  # Added logger_instance
        """
        Initialize the ART controller.

        Args:
            vigilance: Initial vigilance parameter (0-1).
            learning_rate: Initial rate of weight updates (0-1).
            input_dim: Dimension of input vectors. Can be None initially if dynamic input dim is enabled.
            max_categories: Maximum number of categories to learn.
            config: Optional configuration dictionary overriding other parameters.
            logger_instance: Optional logger instance. If None, a new one will be created.
        """
        self.logger = (
            logger_instance if logger_instance else get_art_logger("ARTController")
        )

        # Initialize recent_resonance as a deque instead of None to prevent AttributeError
        self.recent_resonance = _create_recent_resonance_tracker(
            20
        )  # Initialize with default size

        # Initialize key attributes first to prevent AttributeError
        base_config = config or {}
        config_source = base_config.get("config_source")
        # Pass self.logger to config loading function
        self.config = (
            _load_art_config(config_source, self.logger) if config_source else {}
        )
        self.config.update(base_config)  # Passed config overrides file/defaults

        # Initialize total_patterns before it's used in _reset_runtime_stats
        self.total_patterns = 0
        # Initialize weights with default shape
        self.weights = np.zeros((0, input_dim or 0))

        # Initialize category_resonance attribute before calling _reset_runtime_stats
        self.category_resonance = []

        # Now reset runtime stats which will call clear() on recent_resonance
        self._reset_runtime_stats()
        self.connected_to = []
        # Initialize parameters from config or args, validating ranges
        # Safe extraction of vigilance ensuring it's a float and not a dict
        if isinstance(self.config.get("vigilance"), dict):
            # If vigilance is a dictionary, extract a numerical value or use default
            config_vigilance = self.config.get("vigilance")
            if isinstance(config_vigilance, dict) and "value" in config_vigilance:
                self.vigilance = float(
                    _apply_bounds(config_vigilance.get("value", vigilance), 0.0, 1.0)
                )
            else:
                self.logger.warning(
                    f"Vigilance provided as dict without 'value': {config_vigilance}, using default {vigilance}"
                )
                self.vigilance = float(_apply_bounds(vigilance, 0.0, 1.0))
        else:
            # Normal case: vigilance is a scalar or not provided
            self.vigilance = float(
                _apply_bounds(self.config.get("vigilance", vigilance), 0.0, 1.0)
            )

        # Safe extraction of other parameters
        if isinstance(self.config.get("learning_rate"), dict):
            config_lr = self.config.get("learning_rate")
            if isinstance(config_lr, dict) and "value" in config_lr:
                self.learning_rate = float(
                    _apply_bounds(config_lr.get("value", learning_rate), 0.0, 1.0)
                )
            else:
                self.learning_rate = float(_apply_bounds(learning_rate, 0.0, 1.0))
        else:
            self.learning_rate = float(
                _apply_bounds(self.config.get("learning_rate", learning_rate), 0.0, 1.0)
            )

        # Safe extraction of max_categories
        if isinstance(self.config.get("max_categories"), dict):
            config_max = self.config.get("max_categories")
            if isinstance(config_max, dict) and "value" in config_max:
                self.max_categories = int(config_max.get("value", max_categories))
            else:
                self.max_categories = int(max_categories)
        else:
            self.max_categories = int(self.config.get("max_categories", max_categories))

        # Safe extraction of anomaly_threshold
        if isinstance(self.config.get("anomaly_threshold"), dict):
            config_threshold = self.config.get("anomaly_threshold")
            if isinstance(config_threshold, dict) and "value" in config_threshold:
                self.anomaly_threshold = float(
                    _apply_bounds(config_threshold.get("value", 0.3), 0.0, 1.0)
                )
            else:
                self.anomaly_threshold = float(_apply_bounds(0.3, 0.0, 1.0))
        else:
            self.anomaly_threshold = float(
                _apply_bounds(self.config.get("anomaly_threshold", 0.3), 0.0, 1.0)
            )

        # F1: Dynamic Input Dimension Handling
        if isinstance(self.config.get("input_dim"), dict):
            config_input_dim = self.config.get("input_dim")
            if isinstance(config_input_dim, dict) and "value" in config_input_dim:
                self.input_dim = (
                    int(config_input_dim.get("value", input_dim))
                    if input_dim is not None
                    else None
                )
            else:
                self.input_dim = input_dim
        else:
            self.input_dim = self.config.get("input_dim", input_dim)

        if self.input_dim is not None:
            self.input_dim = int(self.input_dim)
        if self.input_dim is not None and self.input_dim <= 0:
            self.logger.warning("input_dim must be positive, setting to default 128")
            self.input_dim = 128  # Set to a reasonable default instead of raising error

        self.dynamic_input_dim = self.config.get(
            "dynamic_input_dim", self.input_dim is None
        )  # Enable if not specified

        # F3: Category Pruning Config
        if isinstance(self.config.get("enable_pruning"), dict):
            config_pruning = self.config.get("enable_pruning")
            self.enable_pruning = (
                bool(config_pruning.get("value", False))
                if isinstance(config_pruning, dict)
                else False
            )
        else:
            self.enable_pruning = bool(self.config.get("enable_pruning", False))

        if isinstance(self.config.get("pruning_min_count"), dict):
            config_min_count = self.config.get("pruning_min_count")
            self.pruning_min_count = (
                int(config_min_count.get("value", 3))
                if isinstance(config_min_count, dict)
                else 3
            )
        else:
            self.pruning_min_count = int(self.config.get("pruning_min_count", 3))

        if isinstance(self.config.get("pruning_max_age_s"), dict):
            config_max_age = self.config.get("pruning_max_age_s")
            self.pruning_max_age_s = (
                float(config_max_age.get("value", 0.0))
                if isinstance(config_max_age, dict) and "value" in config_max_age
                else None
            )
        else:
            self.pruning_max_age_s = self.config.get("pruning_max_age_s", None)

        # State Initialization
        self._lock = threading.Lock()  # F8 Thread safety
        initial_weights = self.config.get("initial_weights")
        initial_cats = self.config.get("initial_categories")

        if initial_weights is not None and self.input_dim is not None:
            try:
                self.weights = np.array(initial_weights, dtype=np.float64)
            except Exception as e:
                self.logger.error(f"Invalid initial_weights: {e}. Initializing empty.")
                self._init_empty_weights(self.input_dim)
            if self.weights.ndim != 2 or self.weights.shape[1] != self.input_dim:
                self.logger.error(
                    f"Shape mismatch: initial_weights {self.weights.shape} vs input_dim {self.input_dim}. Initializing empty."
                )
                self._init_empty_weights(self.input_dim)  # noqa
        elif (
            initial_weights is not None
            and self.input_dim is None
            and self.dynamic_input_dim
        ):
            try:
                self.weights = np.array(initial_weights, dtype=np.float64)
                self.input_dim = self.weights.shape[1]
                self.logger.info(
                    f"Inferred input_dim={self.input_dim} from initial weights."
                )  # noqa
            except Exception as e:
                self.logger.error(f"Invalid initial_weights: {e}. Initializing empty.")
                self._init_empty_weights(0)  # Can't infer dim
        else:
            self._init_empty_weights(self.input_dim)

        # Initialize category metadata, attempting consistency check if state provided
        num_cats_init = self.weights.shape[0]
        self.category_counts = (
            list(initial_cats.get("counts", [1] * num_cats_init))
            if initial_cats
            else [1] * num_cats_init
        )
        self.category_created = (
            list(initial_cats.get("created", [time.time()] * num_cats_init))
            if initial_cats
            else [time.time()] * num_cats_init
        )
        self.category_updated = (
            list(initial_cats.get("updated", [time.time()] * num_cats_init))
            if initial_cats
            else [time.time()] * num_cats_init
        )
        self.category_resonance = (
            list(initial_cats.get("resonance", [1.0] * num_cats_init))
            if initial_cats
            else [1.0] * num_cats_init
        )
        self._ensure_metadata_consistency()  # F7 Ensure lists match weights

        # Statistics
        self.total_patterns = int(
            self.config.get("initial_total_patterns", sum(self.category_counts))
        )  # Start from initial counts
        self._reset_runtime_stats()  # EF11 Deserialization helper part
        self.stats["avg_resonance"] = (
            float(np.mean(self.category_resonance)) if self.category_resonance else 0.0
        )

        # Initialize recent_resonance tracker with proper size
        resonance_history_size = 20  # Default size
        if isinstance(self.config.get("resonance_history_size"), dict):
            config_history = self.config.get("resonance_history_size")
            if isinstance(config_history, dict) and "value" in config_history:
                resonance_history_size = int(config_history.get("value", 20))
        else:
            resonance_history_size = int(self.config.get("resonance_history_size", 20))

        self.recent_resonance = _create_recent_resonance_tracker(resonance_history_size)

        # Final validation
        _validate_art_parameters(self)  # EF14 (will use self.logger)

        self.logger.info(f"ART controller initialized. Config: {self.get_config()}")

    def _init_empty_weights(self, dim: Optional[int]) -> None:
        """Helper to initialize empty weights, handling None dimension."""
        if dim is None:  # Dynamic dim enabled, but no initial weights
            self.weights = np.zeros((0, 0))  # Placeholder until first input
            self.logger.warning(
                "ART initialized with dynamic input dimension and no initial weights. Dimension will be set by first input."
            )
        else:
            self.weights = np.zeros((0, dim), dtype=np.float64)

    def _reset_runtime_stats(self) -> None:
        """Resets runtime statistics (not core state like weights)."""
        self.stats = {
            "total_patterns": self.total_patterns,  # Keep total pattern count
            "total_categories": self.weights.shape[0],  # Derived from weights
            "anomalies_detected_session": 0,  # Track anomalies since last reset/start
            "last_resonance": 0.0,
            "avg_resonance": float(np.mean(self.category_resonance))
            if self.category_resonance
            else 0.0,  # Use current average
            "categories_created_session": 0,  # Track session creations F10
            "categories_pruned_session": 0,  # Track session prunes F10
        }
        self.recent_resonance.clear()  # Clear recent resonance history EF12

    def train(
        self, input_pattern: Union[list[float], np.ndarray], epochs: int = 1
    ) -> dict[str, Any]:
        """Train the network on a single input pattern."""
        start_time = time.monotonic()
        processed_input = self._prepare_input(input_pattern)  # F1 Helper
        if processed_input is None:
            self.logger.error("ART train: Invalid input format or dimension.")
            return {"error": "Invalid input format or dimension."}

        best_matching_result = None
        category_updates = set()

        for _ in range(epochs):
            result = self.process(processed_input, training=True)  # Use processed input
            if result.get("category_id") is not None:
                category_updates.add(result["category_id"])
            # Keep track of the best resonance achieved during epochs
            if best_matching_result is None or result.get(
                "resonance", 0
            ) > best_matching_result.get("resonance", 0):
                best_matching_result = result

        # --- Update statistics (thread-safe) ---
        duration = time.monotonic() - start_time
        with self._lock:  # F8 Lock for stats update
            self.total_patterns += 1  # Increment total patterns processed
            self.stats["total_patterns"] = self.total_patterns
            current_resonance = (
                best_matching_result.get("resonance", 0)
                if best_matching_result
                else 0.0
            )
            self.stats["last_resonance"] = current_resonance
            # Update running average resonance safely
            # Use EMA? Or simple average? Use EMA with low alpha for stability.
            alpha = 0.05
            current_avg = self.stats.get(
                "avg_resonance", current_resonance
            )  # Use current or last resonance if no avg yet
            self.stats["avg_resonance"] = calculate_ema(
                current_resonance, current_avg, alpha
            )  # EF15
            # EF12 Track recent resonance
            self.recent_resonance.append(current_resonance)

        final_result = best_matching_result or {}  # Ensure result dict exists
        final_result["duration_ms"] = duration * 1000
        final_result["epochs"] = epochs
        final_result["updated_categories"] = list(category_updates)

        self.logger.info(
            "ART train pattern", extra=final_result
        )  # Use extra for structured logging
        return final_result

    def process(
        self, input_pattern: Union[list[float], np.ndarray], training: bool = False
    ) -> dict[str, Any]:
        """Process an input pattern: find match or create category (if training)."""
        processed_input = self._prepare_input(input_pattern)  # F1 Helper
        if processed_input is None:
            self.logger.error("ART process: Invalid input.")
            return {
                "error": "Invalid input",
                "category_id": None,
                "resonance": 0.0,
                "is_new_category": False,
                "is_anomaly": True,
            }  # Return consistent error structure # noqa

        # Handle dynamic dimension setting on first valid input
        if (
            self.dynamic_input_dim and self.input_dim is None
        ):  # Corrected from self.input_dim == 0
            with self._lock:  # F8 Lock for setting input_dim and weights
                if self.input_dim == 0:  # Double check inside lock
                    self.input_dim = processed_input.shape[0]
                    self.weights = np.zeros(
                        (0, self.input_dim)
                    )  # Initialize weights with correct dimension
                    self.logger.info(
                        f"ART Controller input dimension dynamically set to {self.input_dim}."
                    )

        # Lock relevant state for read/potential write F8
        with self._lock:
            current_weights = self.weights.copy()  # Work on copy for read consistency
            current_max_categories = self.max_categories
            current_vigilance = self.vigilance  # Get current vigilance

            # If no categories exist
            if current_weights.shape[0] == 0:
                if training and current_weights.shape[0] < current_max_categories:
                    # Create category needs lock -> call helper that acquires lock internally?
                    # Refactor: Do creation outside initial read lock if possible, or handle nested locking carefully.
                    # Simple approach: Call helper directly assuming it handles lock or it's safe.
                    new_category_idx = self._create_new_category(processed_input)
                    self.logger.info(
                        f"ART process: New category {new_category_idx} created (no existing categories)."
                    )
                    return {
                        "category_id": int(new_category_idx),
                        "resonance": 1.0,
                        "is_new_category": True,
                        "is_anomaly": False,
                    }
                else:
                    self.logger.warning(
                        "ART process: No categories exist and not training or max categories reached."
                    )
                    return {
                        "category_id": None,
                        "resonance": 0.0,
                        "is_new_category": False,
                        "is_anomaly": True,
                    }

            # Calculate match scores (can be done outside lock if weights copied)
            match_scores = self._calculate_match_scores(
                processed_input, current_weights
            )  # Pass weights

            # --- Resonance Search Loop (F2 Optional) ---
            # Standard ART checks best match first. Resonance loop sorts and checks iteratively.
            # Simple version: Stick to best match first for now.
            sorted_indices = np.argsort(match_scores)[
                ::-1
            ]  # Indices sorted by score (high to low)
            best_match_idx = sorted_indices[0]
            best_match_score = match_scores[best_match_idx]

            vigilance_passed = _check_vigilance(
                best_match_score, current_vigilance
            )  # EF3

            if vigilance_passed:
                category_id = int(best_match_idx)
                resonance = float(best_match_score)
                is_anomaly = resonance < self.anomaly_threshold
                if is_anomaly:
                    self.stats["anomalies_detected_session"] += (
                        1  # Increment session anomaly count F10
                    )

                if training:
                    self._update_category(
                        category_id, processed_input
                    )  # Needs internal lock
                    # F3 Category pruning check after update? Or periodic? Periodic is safer.

                self.logger.debug(
                    f"ART process: Input matched category {category_id} with resonance {resonance:.4f}."
                )
                return {
                    "category_id": category_id,
                    "resonance": resonance,
                    "is_new_category": False,
                    "is_anomaly": is_anomaly,
                }

            # --- No resonant category found ---
            else:
                is_anomaly = True  # Not meeting vigilance is an anomaly
                self.stats["anomalies_detected_session"] += (
                    1  # Increment session anomaly count F10
                )
                self.logger.info(
                    f"ART process: No resonant category found for input. Best match score: {best_match_score:.4f}, Vigilance: {current_vigilance:.4f}."
                )

                if training and current_weights.shape[0] < current_max_categories:
                    # Create new category
                    new_category_idx = self._create_new_category(
                        processed_input
                    )  # Needs internal lock
                    self.logger.info(
                        f"ART process: New category {new_category_idx} created (no resonance)."
                    )
                    return {
                        "category_id": int(new_category_idx),
                        "resonance": 1.0,
                        "is_new_category": True,
                        "is_anomaly": False,
                    }
                else:
                    # Cannot create new category (max reached or not training)
                    # Return best mismatch info
                    self.logger.warning(
                        "ART process: Max categories reached or not training, cannot create new category."
                    )
                    return {
                        "category_id": None,
                        "resonance": float(best_match_score),
                        "is_new_category": False,
                        "is_anomaly": True,
                    }

    # F1: Input Preparation Helper
    def _prepare_input(
        self, input_pattern: Union[list[float], np.ndarray]
    ) -> Optional[np.ndarray]:
        """Validates, converts, reshapes, normalizes input. Handles dynamic dim check."""
        if isinstance(input_pattern, list):
            input_pattern = np.array(input_pattern, dtype=np.float64)
        if not isinstance(input_pattern, np.ndarray):
            self.logger.error("Input must be list or numpy array.")
            return None

        input_vector = input_pattern.flatten()  # Ensure 1D

        # Check dynamic dim OR match existing dim
        if (
            self.dynamic_input_dim and self.input_dim is None
        ):  # Corrected from self.input_dim == 0
            # Dimension not set yet, accept this input's dim IF no categories exist yet
            with self._lock:  # F8 Check weights safely
                if self.weights.shape[0] == 0:
                    self.input_dim = input_vector.shape[0]
                    self.weights = np.zeros((0, self.input_dim))
                    self.logger.info(
                        f"ART Controller input dimension set to {self.input_dim}"
                    )
                else:  # Dim already set, input must match
                    if input_vector.shape[0] != self.input_dim:
                        self.logger.error(
                            f"Input dim {input_vector.shape[0]} != network dim {self.input_dim}."
                        )
                        return None  # noqa
        elif self.input_dim is not None and input_vector.shape[0] != self.input_dim:
            # Fixed dim: Pad or trim if needed (EF2)
            self.logger.warning(
                f"Input dim {input_vector.shape[0]} != network dim {self.input_dim}. Padding/trimming."
            )
            input_vector = _match_input_dim(input_vector, self.input_dim)
        elif self.input_dim is None and not self.dynamic_input_dim:
            self.logger.error("ART input_dim not set and dynamic_input_dim is False.")
            return None

        # Normalize (EF1)
        normalized_vector = _normalize_vector(input_vector)
        return normalized_vector

    def _calculate_match_scores(
        self, input_pattern: np.ndarray, category_weights: np.ndarray
    ) -> np.ndarray:
        """Calculate match scores (cosine similarity)."""
        # EF5 Use cosine similarity helper
        return np.array(
            [_cosine_similarity(input_pattern, weight) for weight in category_weights]
        )

    # F7 Lock for state modification methods
    def _create_new_category(self, input_pattern: np.ndarray) -> int:
        """Create a new category thread-safely."""
        with self._lock:  # F8 Acquire lock
            if self.weights.shape[0] >= self.max_categories:
                self.logger.error(
                    "Maximum number of categories reached. Cannot create new category."
                )
                raise IndexError(
                    "Maximum number of categories reached."
                )  # Safety check
            # Handle initial empty weights state
            if self.weights.shape[1] == 0 and self.dynamic_input_dim:
                self.input_dim = input_pattern.shape[0]  # Set dimension
                self.weights = input_pattern.reshape(1, self.input_dim)
            elif self.weights.shape[0] == 0:
                self.weights = input_pattern.reshape(1, -1)
            else:
                self.weights = np.vstack([self.weights, input_pattern])

            now = time.time()
            self.category_counts.append(1)
            self.category_created.append(now)
            self.category_updated.append(now)
            self.category_resonance.append(1.0)  # Initial resonance is perfect match
            new_idx = self.weights.shape[0] - 1
            self.stats["total_categories"] = self.weights.shape[0]  # Update stats cache
            self.stats["categories_created_session"] += 1  # F10 Update session stat
        self.logger.info(
            "ART category created",
            extra={"category_id": new_idx, "total_categories": new_idx + 1},
        )
        return new_idx

    # F7 Lock for state modification methods
    def _update_category(self, category_idx: int, input_pattern: np.ndarray) -> None:
        """Update weights and metadata for a category thread-safely."""
        with self._lock:  # F8 Acquire lock
            if not (0 <= category_idx < self.weights.shape[0]):
                self.logger.error(f"Invalid category index {category_idx} for update.")
                return  # noqa
            # Update weights using fast rule EF4
            updated_weight = _update_weights_fast(
                self.weights[category_idx], input_pattern, self.learning_rate
            )
            # F9: Prevent weight decay to zero (optional stability check)
            if (
                self.config.get("prevent_weight_zeroing", True)
                and np.linalg.norm(updated_weight) < 1e-6
            ):
                self.logger.warning(
                    f"Weight update for cat {category_idx} resulted in near-zero norm. Reverting slightly."
                )
                # Option: Revert partially, or skip update? Skip update for simplicity.
                # Option 2: Small nudge back towards input?
                # updated_weight = 0.9 * self.weights[category_idx] + 0.1 * input_pattern # Nudge
            else:
                self.weights[category_idx] = updated_weight

            # Update metadata
            now = time.time()
            self.category_counts[category_idx] += 1
            self.category_updated[category_idx] = (
                now  # Update resonance EMA using similarity EF5
            )
            similarity = _cosine_similarity(input_pattern, self.weights[category_idx])
            old_resonance = self.category_resonance[category_idx]
            # count variable was assigned but never used - removing it
            # Use EMA for category resonance for smoother updates (like global avg)
            alpha_cat = 0.1  # Separate alpha for category EMA
            self.category_resonance[category_idx] = calculate_ema(
                similarity, old_resonance, alpha_cat
            )  # EF15

    # --- Parameter Setting (Thread-safe) ---
    def set_vigilance(self, vigilance: float) -> None:  # noqa
        """Set the vigilance parameter thread-safely."""
        new_vigilance = float(_apply_bounds(vigilance, 0.0, 1.0))  # EF14
        with self._lock:
            self.vigilance = new_vigilance  # F8
        self.logger.info(f"ART vigilance set to {self.vigilance}")

    def set_learning_rate(self, learning_rate: float) -> None:  # noqa
        """Set the learning rate thread-safely."""
        new_lr = float(_apply_bounds(learning_rate, 0.0, 1.0))  # EF14
        with self._lock:
            self.learning_rate = new_lr  # F8
        self.logger.info(f"ART learning rate set to {self.learning_rate}")

    # F2: Adaptive Vigilance Hook (Example Implementation)
    def adapt_vigilance(
        self,
        performance_metric: Optional[float] = None,
        meta_state: Optional[dict] = None,
    ) -> None:
        """Adapts vigilance based on performance or metacognitive state."""
        with self._lock:
            old_vigilance = self.vigilance  # F8 Get current safely
        new_vigilance = old_vigilance  # Start with current
        reason = "no_trigger"

        # Example Adaptation Logic:
        if performance_metric is not None:
            # Lower vigilance if performance is low (more generalization)
            if performance_metric < 0.5:
                new_vigilance = old_vigilance * 0.95
                reason = "low_performance"
            # Raise vigilance slightly if performance is very high (more specialization)
            elif performance_metric > 0.9:
                new_vigilance = old_vigilance * 1.05
                reason = "high_performance"

        # Example based on meta_state (awareness)
        elif meta_state and "awareness" in meta_state:
            awareness = meta_state["awareness"]
            # Lower vigilance if awareness is low (explore more broadly)
            if awareness < 0.4:
                new_vigilance = old_vigilance * 0.9
                reason = "low_awareness"
            # Raise vigilance if awareness is high (focus on specifics)
            elif awareness > 0.8:
                new_vigilance = old_vigilance * 1.1
                reason = "high_awareness"

        new_vigilance_clamped = _apply_bounds(new_vigilance, 0.1, 0.99)  # Clamp EF14
        if abs(new_vigilance_clamped - old_vigilance) > 0.01:
            self.set_vigilance(new_vigilance_clamped)  # Use thread-safe setter
            self.logger.info(
                "Vigilance adapted",
                extra={  # Use extra for structured logging
                    "old": old_vigilance,
                    "new": new_vigilance_clamped,
                    "reason": reason,
                    "perf": performance_metric,
                    "awareness": meta_state.get("awareness") if meta_state else None,
                },
            )
        else:
            self.logger.debug(
                f"Vigilance adaptation skipped (change too small or no trigger). Current: {old_vigilance:.3f}"
            )

    # --- Statistics and Information ---
    def get_statistics(self) -> dict[str, Any]:  # noqa
        """Get network runtime statistics thread-safely."""
        with self._lock:  # F8 Read stats safely
            stats_copy = self.stats.copy()
            # Add derived/static info
            stats_copy["vigilance"] = self.vigilance
            stats_copy["learning_rate"] = self.learning_rate
            stats_copy["input_dim"] = self.input_dim
            stats_copy["max_categories"] = self.max_categories
            stats_copy["anomaly_threshold"] = self.anomaly_threshold
            stats_copy["recent_resonance_avg"] = calculate_moving_average(
                list(self.recent_resonance), len(self.recent_resonance)
            )  # EF4/EF12
        return stats_copy

    def get_category_info(self, category_id: int) -> dict[str, Any]:  # noqa
        """Get information about a specific category thread-safely."""
        with self._lock:  # F8 Read safely
            if not (0 <= category_id < self.weights.shape[0]):
                return {"error": "Invalid category ID"}
            info = {
                "category_id": category_id,
                "pattern_count": _safe_list_get(self.category_counts, category_id, 0),
                "created_time": _safe_list_get(self.category_created, category_id),
                "updated_time": _safe_list_get(self.category_updated, category_id),
                "avg_resonance": _safe_list_get(
                    self.category_resonance, category_id, 0.0
                ),
                "weights": self.weights[category_id].tolist(),
            }  # noqa Use EF15
        return info

    def get_all_categories(self) -> list[dict[str, Any]]:  # noqa
        """Get summary information about all categories thread-safely."""
        # Use refined helper EF13
        return _get_category_summary(self)

    def get_anomaly_categories(self) -> list[int]:  # noqa
        """Get categories with low average resonance thread-safely."""
        anomalies = []
        with self._lock:  # F8 Read safely
            threshold = self.anomaly_threshold
            resonances = self.category_resonance  # Get ref inside lock
            for i, resonance in enumerate(resonances):
                if resonance < threshold:
                    anomalies.append(i)
        return anomalies

    # --- Control Methods ---
    def reset(self) -> None:  # noqa
        """Reset the network to initial state thread-safely."""
        with self._lock:  # F8 Lock for reset
            self.logger.info("Resetting ART controller state...")
            # Re-read initial config settings? Or just clear state? Clear state.
            # Re-initialize based on initial dimension (if fixed)
            self._init_empty_weights(
                self.config.get("input_dim") if not self.dynamic_input_dim else None
            )
            self.category_counts = []
            self.category_created = []
            self.category_updated = []
            self.category_resonance = []
            self.total_patterns = 0  # Reset total count too? Yes.
            self._reset_runtime_stats()  # Resets session stats
        self.logger.info("ART controller reset")

    # F3: Periodic Pruning Method
    def prune_categories(self) -> int:
        """Identifies and removes infrequent or old categories. Returns number pruned."""
        if not self.enable_pruning:
            self.logger.debug("Pruning disabled.")
            return 0  # noqa

        indices_to_prune = []
        pruned_count = 0
        with self._lock:  # F8 Lock for pruning
            if self.weights.shape[0] == 0:
                return 0  # Nothing to prune

            current_time = time.time()
            for i in range(self.weights.shape[0]):
                # Gather info safely using helpers
                info = {
                    "category_id": i,
                    "pattern_count": _safe_list_get(self.category_counts, i, 0),
                    "updated_time": _safe_list_get(
                        self.category_updated, i, current_time
                    ),
                }  # noqa
                if _check_pruning_criteria(
                    info, self.pruning_min_count, self.pruning_max_age_s, self.logger
                ):  # Pass logger to helper
                    indices_to_prune.append(i)

            if indices_to_prune:
                self.logger.info(
                    f"Pruning {len(indices_to_prune)} categories: {indices_to_prune}"
                )
                # Remove from weights and metadata lists in reverse order of index
                for idx in sorted(indices_to_prune, reverse=True):
                    self.weights = np.delete(self.weights, idx, axis=0)
                    del self.category_counts[idx]
                    del self.category_created[idx]
                    del self.category_updated[idx]
                    del self.category_resonance[idx]
                pruned_count = len(indices_to_prune)
                self.stats["total_categories"] = self.weights.shape[
                    0
                ]  # Update stats cache
                self.stats["categories_pruned_session"] += (
                    pruned_count  # F10 Update session stat
                )
                self.logger.info(
                    "Categories pruned",
                    extra={"count": pruned_count, "indices": indices_to_prune},
                )

        return pruned_count

    # F4: State Persistence Methods
    def save_state(self, file_path: str) -> bool:
        """Saves the ART controller's state."""
        self.logger.info(f"Attempting to save ART state to {file_path}")
        with self._lock:
            state_dict = _serialize_art_state(self)  # EF10 Get state safely
        try:
            # Use pickle for numpy array and floats
            serialized = pickle.dumps(state_dict)
            with open(file_path, "wb") as f:
                f.write(serialized)
            self.logger.info(f"ART state saved successfully to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save ART state to {file_path}: {e}")
            return False

    def load_state(self, file_path: str) -> bool:
        """Loads the ART controller's state. WARNING: Overwrites current state."""
        self.logger.info(f"Attempting to load ART state from {file_path}")
        try:
            with open(file_path, "rb") as f:
                serialized = f.read()
            state_dict = pickle.loads(serialized)
            with self._lock:  # F8 Lock for loading state
                _deserialize_art_state(
                    self, state_dict
                )  # EF11 Apply state safely (uses self.logger)
            return True
        except FileNotFoundError:
            self.logger.error(f"ART state file not found: {file_path}")
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to load ART state from {file_path}: {e}", exc_info=True
            )
            return False

    # F5: Configuration Export
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration dictionary."""
        # Collect relevant config attributes
        # Needs manual update if new config options added
        return {
            "vigilance": self.vigilance,
            "learning_rate": self.learning_rate,
            "input_dim": self.input_dim,
            "max_categories": self.max_categories,
            "anomaly_threshold": self.anomaly_threshold,
            "enable_pruning": self.enable_pruning,
            "pruning_min_count": self.pruning_min_count,
            "pruning_max_age_s": self.pruning_max_age_s,
            "resonance_history_size": self.recent_resonance.maxlen,
            "dynamic_input_dim": self.dynamic_input_dim,
            "prevent_weight_zeroing": self.config.get("prevent_weight_zeroing", True),
        }  # noqa

    def export_config(self, file_path: str) -> bool:
        """Exports the current configuration to a JSON file."""
        return _save_art_config(
            self.get_config(), file_path, self.logger
        )  # EF9, pass logger    # F10: Status Method

    def status(self) -> dict[str, Any]:
        """Returns a dictionary summarizing the current status and key stats."""
        return {
            "status": "operational",  # Could add 'degraded'/'error' based on internal checks
            "timestamp": time.time(),
            **self.get_statistics(),  # Include runtime stats
        }

    def _ensure_metadata_consistency(self) -> None:
        """
        Ensure metadata consistency for logging and tracing.
        """
        num_cats = self.weights.shape[0]
        if len(self.category_counts) != num_cats:
            self.category_counts = [1] * num_cats
        if len(self.category_created) != num_cats:
            self.category_created = [time.time()] * num_cats
        if len(self.category_updated) != num_cats:
            self.category_updated = [time.time()] * num_cats
        if len(self.category_resonance) != num_cats:
            self.category_resonance = [1.0] * num_cats

    @property
    def categories(self) -> dict[int, dict]:
        """Mapping of category_id to their data (currently weights only)."""
        return {i: {"weights": self.weights[i]} for i in range(self.weights.shape[0])}


# At end of file, expose ARTManager for tests
try:
    from .art_manager import ARTManager
except ImportError:
    ARTManager = None  # type: ignore
