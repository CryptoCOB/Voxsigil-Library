import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable  # Added Callable

# Preserve original imports exactly
from ..utils.log_event import log_event
from MetaConsciousness.core.context import SDKContext

logger = logging.getLogger("metaconsciousness.meta_learner")

# --- Encapsulated Features ---


# EncapsulatedFeature-1: Safe SDK Context Connection
def _safe_sdk_connect(component_name: str) -> Optional[Any]:
    """
    Safely attempts to get a component from the SDKContext.

    Args:
        component_name: The name of the component to retrieve.

    Returns:
        The component instance or None if connection fails. Logs errors internally.
    """
    try:
        component = SDKContext.get(component_name)
        if component:
            logger.debug(f"Successfully connected to SDK component: {component_name}")
            return component
        else:
            logger.warning(
                f"SDKContext.get returned None for component: {component_name}. Component might not be initialized."
            )
            return None
    except Exception as e:
        logger.error(
            f"Failed to connect to SDK component '{component_name}': {e}", exc_info=True
        )
        return None


# EncapsulatedFeature-2: Validate Performance Metric
def _validate_performance_metric(performance: Any) -> Optional[float]:
    """
    Validates if the performance metric is a float between 0.0 and 1.0.

    Args:
        performance: The performance value to validate.

    Returns:
        The validated float performance, or None if invalid.
    """
    if isinstance(performance, (int, float)):
        p_float = float(performance)
        if 0.0 <= p_float <= 1.0:
            return p_float
        else:
            logger.warning(
                f"Performance metric {p_float} is outside the valid range [0.0, 1.0]."
            )
            return None
    else:
        logger.warning(
            f"Invalid type for performance metric: {type(performance)}. Expected float or int."
        )
        return None


# EncapsulatedFeature-3: Calculate Weighted Average
def _calculate_weighted_average(
    values: List[float], weights: List[float]
) -> Optional[float]:
    """
    Calculates the weighted average of values. Handles empty lists and normalization.

    Args:
        values: List of numerical values.
        weights: List of corresponding weights (must be non-negative).

    Returns:
        The weighted average, or None if inputs are invalid or weights sum to zero.
    """
    if not values or not weights or len(values) != len(weights):
        logger.warning("Invalid input for weighted average calculation.")
        return None

    total_weight = sum(weights)
    if total_weight <= 0:
        logger.warning(
            "Total weight is zero or negative in weighted average calculation."
        )
        # Return simple average if weights are zero? Or None? Let's return None.
        return None

    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight


# EncapsulatedFeature-4: Apply Damping Update Rule
def _apply_damping(old_value: float, new_signal: float, damping_factor: float) -> float:
    """
    Applies a damping update rule: new_value = old * damping + signal * (1 - damping).

    Args:
        old_value: The current value.
        new_signal: The incoming signal or target value.
        damping_factor: The damping factor (0.0 to 1.0). 1.0 means no change, 0.0 means full adoption of new_signal.

    Returns:
        The updated value after applying damping.
    """
    damping_factor = max(0.0, min(1.0, damping_factor))  # Ensure factor is in [0, 1]
    return old_value * damping_factor + new_signal * (1.0 - damping_factor)


# EncapsulatedFeature-5: Trim History List
def _trim_history(history_list: List[Any], max_size: int) -> List[Any]:
    """
    Trims a list to maintain a maximum size by removing the oldest entries.

    Args:
        history_list: The list to trim.
        max_size: The maximum allowed size of the list.

    Returns:
        The trimmed list (might be the original list if size is within limit).
        Returns a new list slice if trimming occurs.
    """
    if max_size <= 0:
        logger.warning(
            f"Cannot trim history with max_size <= 0 ({max_size}). Returning empty list."
        )
        return []
    if len(history_list) > max_size:
        return history_list[-max_size:]
    else:
        return history_list


# EncapsulatedFeature-6: Get Recent Items Safely
def _get_recent_items(data_list: List[Any], count: int) -> List[Any]:
    """
    Safely gets the last 'count' items from a list.

    Args:
        data_list: The list to get items from.
        count: The number of items to retrieve from the end.

    Returns:
        A list containing the last 'count' items, or fewer if the list is shorter.
    """
    if not isinstance(data_list, list):
        logger.warning(f"Cannot get recent items from non-list type: {type(data_list)}")
        return []
    count = max(0, count)  # Ensure count is non-negative
    return data_list[-count:]


# EncapsulatedFeature-7: Calculate Trend Slope (Simple Linear Regression)
def _calculate_trend_slope(values: List[float]) -> float:
    """
    Calculates the slope of the trend line for a sequence of values using simple linear regression.

    Args:
        values: A list of numerical values.

    Returns:
        The calculated slope, or 0.0 if insufficient data (less than 2 points).
    """
    n = len(values)
    if n < 2:
        return 0.0

    x = np.arange(n)  # Use sequence index as x-coordinate
    y = np.array(values)

    # Using numpy's polyfit for simplicity and robustness
    try:
        # polyfit returns [slope, intercept] for degree 1
        slope, _ = np.polyfit(x, y, 1)
        # Handle potential NaN/Inf if polyfit encounters issues, although unlikely with simple sequence
        return float(slope) if np.isfinite(slope) else 0.0
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(
            f"Could not calculate trend slope due to numpy error: {e}. Returning 0.0"
        )
        return 0.0


# EncapsulatedFeature-8: Get Current Timestamp
def _timestamp_now() -> float:
    """Returns the current time as a Unix timestamp."""
    return time.time()


# EncapsulatedFeature-9: Get Nested Dictionary Value Safely
def _get_nested_value(
    data_dict: Dict[str, Any], key_path: List[str], default: Any = None
) -> Any:
    """
    Safely retrieves a value from a nested dictionary using a list of keys.

    Args:
        data_dict: The dictionary to access.
        key_path: A list of keys representing the path to the value.
        default: The value to return if any key is not found or path is invalid.

    Returns:
        The retrieved value or the default.
    """
    current_level = data_dict
    try:
        for key in key_path:
            if not isinstance(current_level, dict):
                return default
            current_level = current_level[key]
        return current_level
    except (KeyError, TypeError):
        return default


# EncapsulatedFeature-10: Calculate Variance Safely
def _calculate_variance(values: List[float]) -> Optional[float]:
    """
    Calculates the variance of a list of values using numpy. Handles insufficient data.

    Args:
        values: A list of numerical values.

    Returns:
        The calculated variance, or None if insufficient data (less than 2 points).
    """
    if len(values) < 2:
        logger.debug("Variance calculation requires at least 2 data points.")
        return None
    try:
        variance = np.var(values)
        return (
            float(variance) if np.isfinite(variance) else None
        )  # Handle potential NaN/Inf
    except Exception as e:
        logger.warning(f"Numpy variance calculation failed: {e}. Returning None.")
        return None


# EncapsulatedFeature-11: Group List Items by Key
def _group_by_key(
    items: List[Any], key_func: Callable[[Any], Any]
) -> Dict[Any, List[Any]]:
    """
    Groups items in a list based on the result of a key function.

    Args:
        items: The list of items to group.
        key_func: A function that takes an item and returns a key.

    Returns:
        A dictionary where keys are the result of key_func and values are lists of items with that key.
    """
    grouped_items: Dict[Any, List[Any]] = {}
    try:
        for item in items:
            key = key_func(item)
            if key not in grouped_items:
                grouped_items[key] = []
            grouped_items[key].append(item)
        return grouped_items
    except Exception as e:
        logger.error(f"Error during grouping by key: {e}", exc_info=True)
        return {}  # Return empty dict on error


# EncapsulatedFeature-12: Clamp Value within Bounds
def _clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamps a numerical value within the specified minimum and maximum bounds."""
    return max(min_value, min(value, max_value))


# EncapsulatedFeature-13: Normalize Dictionary Values (Numerical)
def _normalize_dict_values(data: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizes numerical values in a dictionary to sum to 1.0.
    Assumes values are non-negative.

    Args:
        data: Dictionary with string keys and numerical values.

    Returns:
        A new dictionary with normalized values, or the original if sum is zero.
    """
    if not data:
        return {}

    total = sum(data.values())
    if total <= 0:
        logger.warning("Cannot normalize dictionary values, sum is zero or negative.")
        # Return a dict with equal distribution if sum is zero? Or original? Let's return original/empty.
        return (
            data.copy() if total == 0 else {}
        )  # Or maybe return equal distribution? Let's return copy for 0 sum.

    normalized_data = {key: value / total for key, value in data.items()}
    return normalized_data


# EncapsulatedFeature-14: Generate Parameter Key String
def _generate_param_key(parameters: Dict[str, float], precision: int = 3) -> str:
    """
    Generates a consistent string key from a parameter dictionary for grouping/lookup.

    Args:
        parameters: Dictionary of parameter names and values.
        precision: The decimal precision to use for rounding float values.

    Returns:
        A standardized string key (e.g., "lr0.100_exr0.200").
    """
    sorted_items = sorted(parameters.items())
    parts = [
        f"{key[:3]}{round(value, precision):.{precision}f}"
        for key, value in sorted_items
        if isinstance(value, (int, float))
    ]
    return "_".join(parts)


# EncapsulatedFeature-15: Check if Task Exists
def _task_exists(task_id: str, learning_tasks: Dict[str, Any]) -> bool:
    """Checks if a task ID exists in the learning tasks dictionary."""
    exists = task_id in learning_tasks
    if not exists:
        logger.warning(f"Operation failed: Task ID '{task_id}' not found.")
    return exists


class AdvancedMetaLearner:
    """
    Advanced meta-learning system for cross-domain knowledge transfer.

    Enhanced system with:
    - Configurable settings & validation
    - Improved similarity metrics and transfer control
    - Robust parameter adaptation with bounds
    - State management (task removal, reset)
    - Performance decay and task pruning concepts
    - Detailed status reporting and performance analysis
    - Dynamic adjustment of meta-parameters like transfer strength
    """

    # Feature-5: Configurable Similarity Threshold & Defaults
    DEFAULT_CONFIG = {
        "enabled": True,
        "learning_rate": 0.1,
        "exploration_rate": 0.2,
        "transfer_strength": 0.5,
        "max_history": 100,
        "similarity_threshold": 0.3,  # Configurable threshold
        "param_bounds": {  # Feature-4: Parameter Bounds
            "learning_rate": (0.001, 1.0),
            "exploration_rate": (0.01, 0.8),  # Adjusted bounds slightly
        },
        "meta_param_damping": 0.7,  # Damping for meta-parameter optimization
        "task_ttl_seconds": None,  # Feature-8: Task Pruning (None = disabled)
        "performance_decay_factor": None,  # Feature-7: Performance Decay (None = disabled)
        "min_performance_points_for_adapt": 3,
        "min_performance_points_for_meta_opt": 5,
        "top_n_similar_tasks": 3,  # Control how many tasks influence transfer
        "allow_transfer_override": True,  # Feature-10 Control flag
        "similarity_weights": {  # Feature-6 Advanced Similarity Weights
            "domain": 1.0,
            "type": 1.0,
            "category": 0.8,
            # Numerical features could have weights too
        },
        "required_features": ["domain", "type"],  # Minimum features needed for a task
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the advanced meta-learner with enhanced configuration.

        Args:
            config: User-provided configuration dictionary to override defaults.
        """
        # Merge user config with defaults
        # Note: This is a simple update, a deep merge might be needed for nested dicts like param_bounds
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)  # User config overrides defaults

        self.enabled = self.config.get("enabled", True)
        if not self.enabled:
            logger.info("AdvancedMetaLearner is disabled by configuration.")
            # Initialize required attributes even if disabled, but keep them empty
            self.learning_tasks: Dict[str, Dict[str, Any]] = {}
            self.meta_parameters: Dict[str, float] = {}
            self.transfer_history: List[Dict[str, Any]] = []
            self.memory = None
            self.pattern_memory = None
            self.art_controller = None
            self.meta_cognitive = None
            return  # Skip rest of init if disabled

        self.learning_tasks = {}  # task_id -> {features, performances, parameters, created, last_updated, no_transfer}
        # Initialize meta parameters from config, ensuring keys exist
        self.meta_parameters = {
            "learning_rate": float(self.config.get("learning_rate", 0.1)),
            "exploration_rate": float(self.config.get("exploration_rate", 0.2)),
            "transfer_strength": float(self.config.get("transfer_strength", 0.5)),
        }
        self._validate_meta_parameters()  # Ensure initial meta params are valid

        # Performance tracking (Now primarily within learning_tasks) - Removing redundant self.task_performances
        # self.task_performances = {} # REMOVED - Use task["performances"] instead

        self.transfer_history = []
        # Use helper to get config value safely
        self.max_history = int(_get_nested_value(self.config, ["max_history"], 100))
        self.similarity_threshold = float(
            _get_nested_value(self.config, ["similarity_threshold"], 0.3)
        )
        self.top_n_similar_tasks = int(
            _get_nested_value(self.config, ["top_n_similar_tasks"], 3)
        )
        self.min_perf_points_adapt = int(
            _get_nested_value(self.config, ["min_performance_points_for_adapt"], 3)
        )
        self.connected_to = []
        # Connect to other systems using encapsulated helper
        self._connect_components()

        log_event(
            "AdvancedMetaLearner initialized", self.get_status()
        )  # Use get_status for richer log info
        logger.info(
            f"Advanced meta-learner initialized. Similarity Threshold: {self.similarity_threshold}, Top N Transfer: {self.top_n_similar_tasks}"
        )

    def _validate_meta_parameters(self) -> None:
        """Validate and clamp initial meta-parameters."""
        lr_bounds = self.config.get("param_bounds", {}).get(
            "learning_rate", (0.001, 1.0)
        )
        exr_bounds = self.config.get("param_bounds", {}).get(
            "exploration_rate", (0.01, 0.8)
        )
        ts_bounds = (0.0, 1.0)  # Transfer strength bounds

        self.meta_parameters["learning_rate"] = _clamp(
            self.meta_parameters["learning_rate"], lr_bounds[0], lr_bounds[1]
        )
        self.meta_parameters["exploration_rate"] = _clamp(
            self.meta_parameters["exploration_rate"], exr_bounds[0], exr_bounds[1]
        )
        self.meta_parameters["transfer_strength"] = _clamp(
            self.meta_parameters["transfer_strength"], ts_bounds[0], ts_bounds[1]
        )

    def _connect_components(self) -> None:
        """Connect to other SDK components using safe connect helper."""
        logger.info("Connecting to SDK components...")
        # Use EncapsulatedFeature-1
        self.memory = _safe_sdk_connect("memory")
        self.pattern_memory = _safe_sdk_connect("patterns")
        self.art_controller = _safe_sdk_connect("art_controller")
        # Debug: Original key was 'meta_learning.engine', check if correct
        self.meta_cognitive = _safe_sdk_connect(
            "meta_learning.engine"
        )  # Assuming this key is correct
        # Check connection results
        if not self.memory:
            logger.warning("MetaLearner: Memory component not connected.")
        if not self.pattern_memory:
            logger.warning("MetaLearner: Pattern Memory component not connected.")
        # Add checks for others if needed

    def register_task(
        self,
        task_id: str,
        task_features: Dict[str, Any],
        disable_transfer: bool = False,
    ) -> bool:
        """
        Register a new learning task, validate features, and handle knowledge transfer.

        Args:
            task_id: Task identifier (must be unique).
            task_features: Features describing the task (must contain required features).
            disable_transfer: If True, disable initial parameter transfer for this task.

        Returns:
            bool: True if registration was successful, False otherwise.
        """
        if not self.enabled:
            logger.debug("MetaLearner disabled, skipping task registration.")
            return False

        if not task_id or not isinstance(task_id, str):
            logger.error("Task registration failed: Invalid task_id provided.")
            return False

        if task_id in self.learning_tasks:
            logger.warning(
                f"Task '{task_id}' is already registered. Ignoring registration request."
            )
            return False  # Indicate it wasn't newly registered

        # Validate features using EncapsulatedFeature (Conceptual - Needs implementation)
        if not self._validate_task_features(task_features):
            logger.error(
                f"Task registration failed for '{task_id}': Invalid task features."
            )
            return False

        logger.info(f"Registering new task: {task_id}")
        task_created_time = _timestamp_now()  # Use EncapsulatedFeature-8

        # Initialize task record with default parameters from meta_parameters
        initial_params = {
            "learning_rate": self.meta_parameters["learning_rate"],
            "exploration_rate": self.meta_parameters["exploration_rate"],
        }

        self.learning_tasks[task_id] = {
            "features": task_features,
            "performances": [],  # List of {value, timestamp, parameters} dicts
            "parameters": initial_params.copy(),  # Start with meta defaults
            "created": task_created_time,
            "last_updated": task_created_time,
            # Feature-10: Knowledge Transfer Control per task
            "no_transfer": disable_transfer
            or not self.config.get("allow_transfer_override", True),
        }

        # Find similar tasks for knowledge transfer unless disabled for this task
        if not self.learning_tasks[task_id]["no_transfer"]:
            logger.debug(
                f"Finding similar tasks for potential parameter transfer to '{task_id}'..."
            )
            similar_tasks = self._find_similar_tasks(task_id, task_features)

            if similar_tasks:
                logger.info(
                    f"Found {len(similar_tasks)} similar tasks for '{task_id}'. Performing parameter transfer."
                )
                self._transfer_parameters(task_id, similar_tasks)
            else:
                logger.info(
                    f"No sufficiently similar tasks found for '{task_id}' based on threshold {self.similarity_threshold}. Using default parameters."
                )
        else:
            logger.info(
                f"Parameter transfer explicitly disabled for task '{task_id}'. Using default parameters."
            )

        log_event(
            "task_registered",
            {
                "task_id": task_id,
                "features": task_features,
                "initial_params": initial_params,
                "transfer_disabled": self.learning_tasks[task_id]["no_transfer"],
            },
        )
        return True

    # EncapsulatedFeature-16 (Placeholder): Validate Task Features
    def _validate_task_features(self, features: Dict[str, Any]) -> bool:
        """Basic validation for task features structure and required keys."""
        if not isinstance(features, dict):
            logger.warning("Task features must be a dictionary.")
            return False
        required = self.config.get("required_features", [])
        missing = [key for key in required if key not in features]
        if missing:
            logger.warning(f"Task features missing required keys: {missing}")
            return False
        # Add more checks? e.g., type checks for specific features
        return True

    # Feature-2: Task Removal Method
    def remove_task(self, task_id: str) -> bool:
        """
        Removes a task and its associated data from the meta-learner.

        Args:
            task_id: The ID of the task to remove.

        Returns:
            bool: True if the task was successfully removed, False otherwise.
        """
        if not self.enabled:
            logger.debug("MetaLearner disabled, skipping task removal.")
            return False

        if task_id in self.learning_tasks:
            del self.learning_tasks[task_id]
            # Remove from performance tracking if it was being used redundantly (it's not anymore)
            # self.task_performances.pop(task_id, None) # Keep line commented out
            logger.info(f"Removed task: {task_id}")
            log_event("task_removed", {"task_id": task_id})
            return True
        else:
            logger.warning(f"Attempted to remove non-existent task: {task_id}")
            return False

    # Feature-3: State Reset Method
    def reset_state(self, reset_meta_parameters: bool = True) -> None:
        """
        Clears all registered tasks, history, and optionally resets meta-parameters to config defaults.
        """
        if not self.enabled:
            logger.debug("MetaLearner disabled, skipping state reset.")
            return

        self.learning_tasks.clear()
        # self.task_performances.clear() # Removed redundancy
        self.transfer_history.clear()
        logger.info("Cleared all learning tasks and transfer history.")

        if reset_meta_parameters:
            self.meta_parameters = {
                "learning_rate": float(self.config.get("learning_rate", 0.1)),
                "exploration_rate": float(self.config.get("exploration_rate", 0.2)),
                "transfer_strength": float(self.config.get("transfer_strength", 0.5)),
            }
            self._validate_meta_parameters()  # Ensure reset params are valid
            logger.info("Reset meta-parameters to configuration defaults.")
            log_event(
                "meta_learner_reset",
                {
                    "meta_parameters_reset": True,
                    "final_meta_params": self.meta_parameters,
                },
            )
        else:
            log_event(
                "meta_learner_reset",
                {
                    "meta_parameters_reset": False,
                    "final_meta_params": self.meta_parameters,
                },
            )

    def _find_similar_tasks(
        self, current_task_id: str, task_features: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Find tasks that are similar based on weighted feature similarity.
        Uses configurable threshold and top_n limit.

        Args:
            current_task_id: Task to compare against.
            task_features: Features of the current task.

        Returns:
            List of (task_id, similarity) tuples, sorted by similarity DESC.
        """
        similarities = []

        # Feature-6: Advanced Similarity - Use weights from config
        feature_weights = self.config.get("similarity_weights", {})

        for task_id, task in self.learning_tasks.items():
            if task_id == current_task_id:
                continue

            # Pass weights to similarity calculation
            similarity = self._calculate_similarity(
                task_features, task["features"], feature_weights
            )

            # Use configurable threshold
            if similarity > self.similarity_threshold:
                similarities.append((task_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return only top N results (N is configurable)
        return similarities[: self.top_n_similar_tasks]

    def _calculate_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:  # Added weights arg
        """
        Calculate weighted similarity between two feature sets. Handles categorical,
        numerical, and potentially other types more robustly.

        Args:
            features1: First feature set.
            features2: Second feature set.
            weights: Optional dictionary mapping feature keys to similarity weights (default 1.0).

        Returns:
            Weighted similarity score (0-1 range approximately).
        """
        if weights is None:
            weights = {}
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        weighted_similarities = []
        total_weight = 0.0

        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            weight = weights.get(key, 1.0)  # Default weight is 1.0

            if weight <= 0:
                continue  # Skip features with no weight

            feature_similarity = 0.0  # Default similarity if type mismatch or unknown

            # Categorical (string types assumed categorical for now)
            if isinstance(val1, str) and isinstance(val2, str):
                feature_similarity = (
                    1.0 if val1.lower() == val2.lower() else 0.0
                )  # Case-insensitive compare
            # Numerical (int/float)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_abs_val = max(abs(float(val1)), abs(float(val2)))
                if max_abs_val > 1e-9:  # Avoid division by zero or near-zero
                    diff = abs(float(val1) - float(val2)) / max_abs_val
                    feature_similarity = 1.0 - _clamp(
                        diff, 0.0, 1.0
                    )  # Clamp ensures 0-1 range
                elif abs(float(val1) - float(val2)) < 1e-9:
                    feature_similarity = 1.0  # Both are zero or very close
                else:
                    feature_similarity = (
                        0.0  # One is zero, one isn't (significant diff)
                    )
            # Boolean
            elif isinstance(val1, bool) and isinstance(val2, bool):
                feature_similarity = 1.0 if val1 == val2 else 0.0
            # List/Set overlap (Jaccard Index) - Added for potential list features
            elif isinstance(val1, (list, set)) and isinstance(val2, (list, set)):
                set1 = set(val1)
                set2 = set(val2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                feature_similarity = (
                    intersection / union
                    if union > 0
                    else 1.0
                    if not set1 and not set2
                    else 0.0
                )
            else:
                # Log unsupported type comparison once per key?
                logger.debug(
                    f"Skipping similarity calculation for key '{key}': Unsupported type comparison ({type(val1)} vs {type(val2)})."
                )
                continue  # Skip this feature for similarity calculation

            weighted_similarities.append(feature_similarity * weight)
            total_weight += weight

        if total_weight > 0:
            return sum(weighted_similarities) / total_weight
        else:
            return 0.0  # No common, weighted features found

    def _transfer_parameters(
        self, target_task_id: str, similar_tasks: List[Tuple[str, float]]
    ) -> None:
        """
        Transfer parameters from similar tasks, respecting transfer strength and parameter bounds.

        Args:
            target_task_id: Task to transfer parameters to.
            similar_tasks: List of (task_id, similarity) tuples.
        """
        # Use EncapsulatedFeature-15 to check task exists
        if not _task_exists(target_task_id, self.learning_tasks):
            return

        target_task = self.learning_tasks[target_task_id]

        if not similar_tasks:
            logger.debug(
                f"No similar tasks provided for parameter transfer to '{target_task_id}'."
            )
            return

        # Extract values and weights for weighted average calculation
        param_values_sources: Dict[
            str, List[float]
        ] = {}  # param_name -> [val1, val2, ...]
        weights: List[float] = [sim for _, sim in similar_tasks]

        for task_id, similarity in similar_tasks:
            if not _task_exists(task_id, self.learning_tasks):
                continue  # Skip if source task somehow disappeared
            source_task = self.learning_tasks[task_id]
            for param_name, param_value in source_task["parameters"].items():
                if param_name not in param_values_sources:
                    param_values_sources[param_name] = []
                param_values_sources[param_name].append(param_value)

        # Calculate weighted average parameters using EncapsulatedFeature-3
        transferred_params: Dict[str, float] = {}
        for param_name, values in param_values_sources.items():
            # Ensure number of values matches number of weights (tasks)
            if len(values) == len(weights):
                avg_param = _calculate_weighted_average(values, weights)
                if avg_param is not None:
                    transferred_params[param_name] = avg_param
            else:
                logger.warning(
                    f"Mismatch between param values ({len(values)}) and weights ({len(weights)}) for '{param_name}' during transfer to '{target_task_id}'. Skipping param."
                )

        # Apply transfer strength and bounds using EncapsulatedFeature-4 and EncapsulatedFeature-12
        transfer_strength = self.meta_parameters["transfer_strength"]
        param_bounds = self.config.get("param_bounds", {})
        original_params = target_task["parameters"].copy()
        updated_params_applied: Dict[str, float] = {}

        for param_name, transferred_value in transferred_params.items():
            original_value = original_params.get(param_name)
            # Fallback if original value missing (shouldn't happen if init is correct)
            if original_value is None:
                original_value = self.meta_parameters.get(
                    param_name, 0.1
                )  # Fallback to meta-param
                logger.warning(
                    f"Original value for '{param_name}' missing in task '{target_task_id}' during transfer. Using meta default."
                )

            # Apply damping/transfer strength using EncapsulatedFeature-4
            new_value = _apply_damping(
                original_value, transferred_value, 1.0 - transfer_strength
            )  # damping = 1 - strength

            # Apply bounds using EncapsulatedFeature-12
            bounds = param_bounds.get(param_name)
            if bounds:
                new_value = _clamp(new_value, bounds[0], bounds[1])

            target_task["parameters"][param_name] = new_value
            updated_params_applied[param_name] = new_value

        if updated_params_applied:
            logger.info(
                f"Transferred parameters to task '{target_task_id}'. New params: {updated_params_applied}"
            )
            # Record transfer event
            transfer_record = {
                "timestamp": _timestamp_now(),  # Use EncapsulatedFeature-8
                "target_task": target_task_id,
                "source_tasks": [task_id for task_id, _ in similar_tasks],
                "similarities": [similarity for _, similarity in similar_tasks],
                "transfer_strength": transfer_strength,
                "original_params": original_params,
                "transferred_signal": transferred_params,  # Values before applying strength/bounds
                "final_params": target_task["parameters"].copy(),
            }
            self.transfer_history.append(transfer_record)
            # Trim history using EncapsulatedFeature-5
            self.transfer_history = _trim_history(
                self.transfer_history, self.max_history
            )
            log_event("parameter_transfer", transfer_record)
        else:
            logger.warning(
                f"Parameter transfer calculation resulted in no applicable parameter updates for task '{target_task_id}'."
            )

    def update_performance(self, task_id: str, performance: float) -> None:
        """
        Update performance metrics for a task, validate input, and trigger adaptation.

        Args:
            task_id: Task identifier.
            performance: Performance metric (validated to be float 0-1).
        """
        if not self.enabled:
            return
        # Use EncapsulatedFeature-15 to check task exists
        if not _task_exists(task_id, self.learning_tasks):
            return

        # Use EncapsulatedFeature-2 to validate performance metric
        validated_performance = _validate_performance_metric(performance)
        if validated_performance is None:
            logger.warning(
                f"Invalid performance metric '{performance}' provided for task '{task_id}'. Update skipped."
            )
            return

        task = self.learning_tasks[task_id]
        current_time = _timestamp_now()  # Use EncapsulatedFeature-8

        # Store performance with current parameters
        performance_record = {
            "value": validated_performance,
            "timestamp": current_time,
            "parameters": task[
                "parameters"
            ].copy(),  # Record params *at the time of* this performance
        }
        task["performances"].append(performance_record)

        # Limit performance history using EncapsulatedFeature-5
        task["performances"] = _trim_history(task["performances"], self.max_history)

        # Update task timestamp
        task["last_updated"] = current_time

        # Debug: Removed redundant global self.task_performances tracking

        log_event(
            "performance_updated",
            {
                "task_id": task_id,
                "performance": validated_performance,
                "params_at_update": performance_record["parameters"],
            },
        )
        logger.debug(
            f"Updated performance for task '{task_id}': {validated_performance:.4f}"
        )

        # Adapt parameters based on performance history
        self._adapt_parameters(task_id)

        # Feature-8: Trigger Task Pruning Check
        self._prune_inactive_tasks()

    def _adapt_parameters(self, task_id: str) -> None:
        """
        Adapt task parameters based on performance history, respecting bounds.
        Considers performance decay if enabled.

        Args:
            task_id: Task identifier.
        """
        # Use EncapsulatedFeature-15 to check task exists
        if not _task_exists(task_id, self.learning_tasks):
            return

        task = self.learning_tasks[task_id]
        performances = task["performances"]

        # Need sufficient performance points
        if len(performances) < self.min_perf_points_adapt:
            logger.debug(
                f"Skipping parameter adaptation for '{task_id}': requires {self.min_perf_points_adapt} points, has {len(performances)}."
            )
            return

        # Use EncapsulatedFeature-6 to get recent performances
        recent_performances = _get_recent_items(
            performances, self.min_perf_points_adapt
        )
        values = [p["value"] for p in recent_performances]

        # Feature-7: Performance Decay (Optional weighting for trend/variance)
        weights = None
        decay_factor = self.config.get("performance_decay_factor")
        if decay_factor is not None and 0.0 < decay_factor < 1.0:
            # Simple exponential decay weights (newest = 1, older = decay_factor, decay_factor^2, ...)
            weights = [decay_factor**i for i in range(len(values) - 1, -1, -1)]
            logger.debug(
                f"Applying performance decay weights for adaptation: {weights}"
            )

        # Calculate trend using EncapsulatedFeature-7 (can add weights later if needed)
        trend_slope = _calculate_trend_slope(values)

        # Get current parameters and bounds
        current_params = task["parameters"]
        param_bounds = self.config.get("param_bounds", {})
        lr_bounds = param_bounds.get("learning_rate", (0.001, 1.0))
        exr_bounds = param_bounds.get("exploration_rate", (0.01, 0.8))

        # Adapt learning rate based on trend
        # More robust adaptation: Adjust proportionally to slope magnitude?
        lr_adjustment_factor = 1.0
        if abs(trend_slope) < 0.005:  # Stable performance
            lr_adjustment_factor = (
                0.98  # Slightly decrease LR if stable but not maxed out?
            )
        elif trend_slope > 0:  # Improving
            lr_adjustment_factor = (
                1.0 + abs(trend_slope) * 2.0
            )  # Increase LR more if steep improvement (capped later)
        else:  # Declining
            lr_adjustment_factor = (
                1.0 - abs(trend_slope) * 5.0
            )  # Decrease LR more if steep decline

        new_lr = current_params["learning_rate"] * lr_adjustment_factor
        new_lr = _clamp(
            new_lr, lr_bounds[0], lr_bounds[1]
        )  # Use EncapsulatedFeature-12

        # Adapt exploration rate based on performance variance
        # Use EncapsulatedFeature-10 to calculate variance safely
        variance = _calculate_variance(values)  # Can add weighted variance later
        new_exr = current_params["exploration_rate"]  # Default to current

        if variance is not None:
            exr_adjustment_factor = 1.0
            if variance < 0.005:  # Very low variance - might be stuck in local optimum
                exr_adjustment_factor = 1.1  # Increase exploration
            elif variance > 0.05:  # High variance - learning might be unstable
                exr_adjustment_factor = 0.9  # Decrease exploration

            new_exr = current_params["exploration_rate"] * exr_adjustment_factor
            new_exr = _clamp(
                new_exr, exr_bounds[0], exr_bounds[1]
            )  # Use EncapsulatedFeature-12

        # Apply changes if they are significant enough
        significant_change = False
        if abs(new_lr - current_params["learning_rate"]) > 1e-5:
            task["parameters"]["learning_rate"] = new_lr
            significant_change = True
        if abs(new_exr - current_params["exploration_rate"]) > 1e-5:
            task["parameters"]["exploration_rate"] = new_exr
            significant_change = True

        if significant_change:
            logger.info(
                f"Adapted parameters for task '{task_id}': LR={new_lr:.4f}, EXR={new_exr:.4f} (Slope: {trend_slope:.4f}, Variance: {variance if variance is not None else 'N/A'})"
            )
            log_event(
                "parameter_adaptation",
                {
                    "task_id": task_id,
                    "new_parameters": task["parameters"].copy(),
                    "trend_slope": trend_slope,
                    "performance_variance": variance,
                },
            )
        else:
            logger.debug(
                f"Parameter adaptations for task '{task_id}' were not significant enough to apply."
            )

    # Feature-9: Manual Parameter Override
    def set_task_parameters(
        self, task_id: str, parameters: Dict[str, float], temporary: bool = False
    ) -> bool:
        """
        Manually sets parameters for a specific task, overriding adaptation.

        Args:
            task_id: The ID of the task to modify.
            parameters: A dictionary containing the parameters to set (e.g., {"learning_rate": 0.5}).
            temporary: If True, these parameters might be overwritten by adaptation later.
                       If False, potentially mark the task to stop adapting? (Needs flag). For now, just sets them.

        Returns:
            bool: True if parameters were successfully set, False otherwise.
        """
        if not self.enabled:
            return False
        if not _task_exists(task_id, self.learning_tasks):
            return False

        task = self.learning_tasks[task_id]
        param_bounds = self.config.get("param_bounds", {})
        updated = False
        new_params = task["parameters"].copy()

        for key, value in parameters.items():
            if key in task["parameters"]:  # Only override existing parameter types
                if isinstance(value, (int, float)):
                    bounds = param_bounds.get(key)
                    clamped_value = float(value)
                    if bounds:
                        clamped_value = _clamp(clamped_value, bounds[0], bounds[1])
                    if (
                        abs(new_params[key] - clamped_value) > 1e-6
                    ):  # Check if value actually changes
                        new_params[key] = clamped_value
                        updated = True
                else:
                    logger.warning(
                        f"Invalid value type '{type(value)}' provided for parameter '{key}' in set_task_parameters. Skipping."
                    )
            else:
                logger.warning(
                    f"Parameter '{key}' not recognized for task '{task_id}' in set_task_parameters. Skipping."
                )

        if updated:
            task["parameters"] = new_params
            # Add a flag or state if permanent override is needed? Complexity tradeoff.
            # For now, adaptation might overwrite these later unless it checks for a manual override flag.
            task["last_updated"] = _timestamp_now()  # Mark as updated
            logger.info(f"Manually set parameters for task '{task_id}': {new_params}")
            log_event(
                "manual_parameter_override",
                {
                    "task_id": task_id,
                    "set_parameters": new_params,
                    "temporary": temporary,
                },
            )
            return True
        else:
            logger.info(
                f"Manual parameter set for task '{task_id}' resulted in no changes."
            )
            return False

    def get_task_parameters(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current parameters for a task. Returns None if task doesn't exist.

        Args:
            task_id: Task identifier.

        Returns:
            Dict with task parameters or None if task not found.
        """
        if not self.enabled:
            return None
        if _task_exists(task_id, self.learning_tasks):
            # Return a copy to prevent external modification
            return self.learning_tasks[task_id]["parameters"].copy()
        else:
            # Return None instead of default params to clearly indicate task not found
            return None

    def analyze_task_performance(self, task_id: str) -> Dict[str, Any]:
        """
        Analyze task performance over time, including trend and basic statistics.

        Args:
            task_id: Task identifier.

        Returns:
            Dict with performance analysis, or an error dict if task not found/no data.
        """
        if not self.enabled:
            return {"error": "MetaLearner is disabled"}
        if not _task_exists(task_id, self.learning_tasks):
            return {"error": "Task not found"}

        task = self.learning_tasks[task_id]
        performances = task["performances"]

        if not performances:
            return {"error": "No performance data available for this task"}

        # Extract performance values and timestamps
        values = [p["value"] for p in performances]
        timestamps = [p["timestamp"] for p in performances]

        # Calculate basic statistics
        avg_performance = np.mean(values)
        max_performance = np.max(values)
        min_performance = np.min(values)

        # Calculate trend using EncapsulatedFeature-7
        slope = _calculate_trend_slope(values)

        # Determine trend description
        if abs(slope) < 0.001:
            trend = "stable"  # More sensitive threshold?
        elif slope > 0:
            trend = "improving"
        else:
            trend = "declining"

        # Calculate variance using EncapsulatedFeature-10
        variance = _calculate_variance(values)

        analysis = {
            "task_id": task_id,
            "performance_count": len(values),
            "avg_performance": float(avg_performance)
            if np.isfinite(avg_performance)
            else None,
            "max_performance": float(max_performance)
            if np.isfinite(max_performance)
            else None,
            "min_performance": float(min_performance)
            if np.isfinite(min_performance)
            else None,
            "latest_performance": values[-1],
            "trend": trend,
            "trend_slope": float(slope) if np.isfinite(slope) else None,
            "variance": float(variance)
            if variance is not None and np.isfinite(variance)
            else None,
            "first_update_time": timestamps[0],
            "last_update_time": timestamps[-1],
            "time_elapsed_seconds": timestamps[-1] - timestamps[0]
            if len(timestamps) > 1
            else 0.0,
        }
        log_event("task_performance_analysis", analysis)
        return analysis

    def get_global_performance(self) -> Dict[str, Any]:
        """
        Get global performance metrics across all registered tasks.

        Returns:
            Dict with global performance metrics.
        """
        if not self.enabled:
            return {"tasks": 0, "performances": 0, "status": "Disabled"}

        all_task_performances = [
            (task_id, p["value"])
            for task_id, task_data in self.learning_tasks.items()
            for p in task_data.get("performances", [])
        ]

        if not all_task_performances:
            return {
                "tasks_registered": len(self.learning_tasks),
                "tasks_with_performance_data": 0,
                "total_performance_updates": 0,
                "global_avg_performance": None,
            }

        # Group performances by task ID
        perf_by_task: Dict[str, List[float]] = {}
        for task_id, perf_value in all_task_performances:
            if task_id not in perf_by_task:
                perf_by_task[task_id] = []
            perf_by_task[task_id].append(perf_value)

        # Calculate per-task metrics
        task_metrics = {}
        valid_tasks_count = 0
        for task_id, performances in perf_by_task.items():
            if performances:
                valid_tasks_count += 1
                task_metrics[task_id] = {
                    "avg": float(np.mean(performances)),
                    "max": float(np.max(performances)),
                    "min": float(np.min(performances)),
                    "count": len(performances),
                }

        # Calculate global metrics
        all_values = [p for _, p in all_task_performances]
        global_avg = float(np.mean(all_values)) if all_values else None

        return {
            "tasks_registered": len(self.learning_tasks),
            "tasks_with_performance_data": valid_tasks_count,
            "total_performance_updates": len(all_values),
            "global_avg_performance": global_avg,
            "task_metrics_summary": task_metrics,  # Contains detailed stats per task
        }

    def optimize_meta_parameters(self) -> Dict[str, Any]:
        """
        Optimize global meta-parameters (learning rate, exploration rate, transfer strength)
        based on historical effectiveness across tasks.

        Returns:
            Dict with optimization results.
        """
        if not self.enabled:
            return {"optimized": False, "reason": "MetaLearner is disabled"}

        min_points_meta = int(self.config.get("min_performance_points_for_meta_opt", 5))

        # Gather tasks with sufficient data
        tasks_for_optimization: List[str] = [
            task_id
            for task_id, task_data in self.learning_tasks.items()
            if len(task_data.get("performances", [])) >= min_points_meta
        ]

        if len(tasks_for_optimization) < 1:  # Need at least one task, ideally more
            return {
                "optimized": False,
                "reason": f"Insufficient tasks with >= {min_points_meta} performance points ({len(tasks_for_optimization)} found).",
            }

        # Analyze parameter effectiveness for each task
        task_param_effectiveness: Dict[str, Dict[str, Any]] = {}

        for task_id in tasks_for_optimization:
            task = self.learning_tasks[task_id]
            performances = task[
                "performances"
            ]  # List of {value, timestamp, parameters}

            # Group performances by the parameters used *at that time*
            # Use EncapsulatedFeature-11 and EncapsulatedFeature-14
            grouped_perf = _group_by_key(
                performances, lambda p: _generate_param_key(p["parameters"])
            )

            best_avg_perf = -1.0  # Start below valid performance range
            best_param_key = None
            best_params_dict = None
            evaluated_groups = 0

            for param_key, perfs in grouped_perf.items():
                # Require a minimum number of data points for this parameter set? e.g. 2?
                if len(perfs) < 2:
                    continue

                evaluated_groups += 1
                avg_perf = float(np.mean([p["value"] for p in perfs]))

                if avg_perf > best_avg_perf:
                    best_avg_perf = avg_perf
                    best_param_key = param_key
                    # Store the actual dict that generated the key
                    best_params_dict = perfs[0][
                        "parameters"
                    ].copy()  # Get params from first record in group

            if best_params_dict:
                task_param_effectiveness[task_id] = {
                    "best_avg_performance": best_avg_perf,
                    "best_params_config": best_params_dict,  # The specific params that worked best
                    "param_key": best_param_key,
                    "groups_evaluated": evaluated_groups,
                }

        # Aggregate best parameters across tasks
        if not task_param_effectiveness:
            return {
                "optimized": False,
                "reason": "Could not identify effective parameters across tasks.",
            }

        # Calculate weighted average of the best parameters found, weighted by the performance achieved
        # Example: Task A did best with LR=0.5 (achieved 0.9), Task B with LR=0.2 (achieved 0.7)
        # Weighted LR = (0.5 * 0.9 + 0.2 * 0.7) / (0.9 + 0.7)
        aggregated_params: Dict[str, List[float]] = {}
        perf_weights: List[float] = []

        for task_id, info in task_param_effectiveness.items():
            perf_weights.append(info["best_avg_performance"])
            best_p = info["best_params_config"]
            for param_name, value in best_p.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = []
                aggregated_params[param_name].append(value)

        # Calculate final weighted average signal using EncapsulatedFeature-3
        new_meta_signal: Dict[str, float] = {}
        for param_name, values_list in aggregated_params.items():
            # Ensure lists align (should if data is consistent)
            if len(values_list) == len(perf_weights):
                avg_val = _calculate_weighted_average(values_list, perf_weights)
                if avg_val is not None:
                    new_meta_signal[param_name] = avg_val
            else:
                logger.warning(
                    f"Mismatch calculating weighted average for meta-param '{param_name}'. Skipping."
                )

        # Feature-11: Dynamic Transfer Strength (Example) - If overall avg performance low, maybe increase transfer?
        global_avg = self.get_global_performance().get("global_avg_performance")
        if global_avg is not None and global_avg < 0.5:  # Example threshold
            new_ts_signal = min(
                1.0, self.meta_parameters["transfer_strength"] * 1.1
            )  # Increase transfer strength signal
            logger.info(
                f"Low global avg performance ({global_avg:.3f}), potentially increasing transfer strength."
            )
        else:
            new_ts_signal = self.meta_parameters[
                "transfer_strength"
            ]  # Keep current as signal

        # Update meta-parameters using damping (EncapsulatedFeature-4)
        old_params = self.meta_parameters.copy()
        damping = float(self.config.get("meta_param_damping", 0.7))
        updated_count = 0

        for param_name, signal_value in new_meta_signal.items():
            if param_name in self.meta_parameters:
                current_value = self.meta_parameters[param_name]
                new_value = _apply_damping(current_value, signal_value, damping)
                # Apply bounds relevant to meta-parameters
                bounds = self.config.get("param_bounds", {}).get(param_name)
                if bounds:
                    new_value = _clamp(new_value, bounds[0], bounds[1])
                # Only update if change is significant
                if abs(new_value - current_value) > 1e-5:
                    self.meta_parameters[param_name] = new_value
                    updated_count += 1

        # Update transfer strength separately
        current_ts = self.meta_parameters["transfer_strength"]
        new_ts = _apply_damping(current_ts, new_ts_signal, damping)
        new_ts = _clamp(new_ts, 0.0, 1.0)
        if abs(new_ts - current_ts) > 1e-5:
            self.meta_parameters["transfer_strength"] = new_ts
            updated_count += 1

        if updated_count > 0:
            logger.info(
                f"Optimized meta-parameters. Old: {old_params}, New: {self.meta_parameters}"
            )
            result = {
                "optimized": True,
                "old_params": old_params,
                "new_params": self.meta_parameters.copy(),
                "tasks_analyzed": len(task_param_effectiveness),
                "param_signal": new_meta_signal,  # Include the aggregated target signal before damping
            }
            log_event("meta_parameters_optimized", result)
            return result
        else:
            logger.info(
                "Meta-parameter optimization resulted in no significant changes."
            )
            return {
                "optimized": False,
                "reason": "No significant parameter changes identified.",
            }

    # Feature-8: Task Pruning (Inactive Tasks)
    def _prune_inactive_tasks(self) -> int:
        """
        Removes tasks that haven't been updated within the configured TTL.
        Triggered periodically or after performance updates.

        Returns:
            int: Number of tasks pruned.
        """
        task_ttl = self.config.get("task_ttl_seconds")
        if task_ttl is None or task_ttl <= 0:
            return 0  # Pruning disabled

        current_time = _timestamp_now()
        pruned_count = 0
        tasks_to_prune = []

        for task_id, task_data in self.learning_tasks.items():
            last_updated = task_data.get("last_updated", 0)
            if (current_time - last_updated) > task_ttl:
                tasks_to_prune.append(task_id)

        for task_id in tasks_to_prune:
            if self.remove_task(task_id):  # Use the existing removal method
                pruned_count += 1

        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} inactive tasks (TTL: {task_ttl}s).")
            log_event("tasks_pruned", {"count": pruned_count, "ttl_seconds": task_ttl})

        return pruned_count

    # Feature-12: Export State (For external persistence/visualization)
    def export_state(self) -> Dict[str, Any]:
        """
        Exports the current state of the meta-learner as a dictionary.
        Suitable for saving to JSON or other formats externally.

        Returns:
            Dict containing the current state.
        """
        state = {
            "config": self.config,  # Include config used at init
            "meta_parameters": self.meta_parameters,
            "learning_tasks": self.learning_tasks,  # Contains features, params, performance history
            "transfer_history": self.transfer_history,
            "export_timestamp": _timestamp_now(),
        }
        logger.info(
            f"Exported MetaLearner state ({len(self.learning_tasks)} tasks, {len(self.transfer_history)} transfer records)."
        )
        return state

    # Feature-13: Import State (Complementary to Export)
    def import_state(self, state: Dict[str, Any]) -> bool:
        """
        Imports state from a dictionary, overwriting the current state.
        Performs basic validation.

        Args:
            state: A dictionary containing the state to load (matching the export format).

        Returns:
            True if import was successful, False otherwise.
        """
        if not isinstance(state, dict):
            logger.error("Import state failed: Input must be a dictionary.")
            return False

        required_keys = ["meta_parameters", "learning_tasks", "transfer_history"]
        if not all(key in state for key in required_keys):
            logger.error(
                f"Import state failed: Input dictionary missing required keys ({required_keys})."
            )
            return False

        try:
            # Optionally re-apply config from imported state? Or keep current? Keep current for now.
            # self.config = state.get("config", self.config) # Decide if config should be restored

            self.meta_parameters = state["meta_parameters"]
            self.learning_tasks = state["learning_tasks"]
            self.transfer_history = state["transfer_history"]

            # Validate structure/types after loading (optional but recommended)
            # e.g., check if tasks have required nested keys
            self._validate_meta_parameters()  # Re-validate loaded meta params

            logger.info(
                f"Successfully imported MetaLearner state ({len(self.learning_tasks)} tasks, {len(self.transfer_history)} transfer records)."
            )
            log_event(
                "meta_learner_state_imported",
                {
                    "tasks": len(self.learning_tasks),
                    "transfers": len(self.transfer_history),
                },
            )
            return True

        except Exception as e:
            logger.error(f"Error during state import: {e}", exc_info=True)
            # Optionally reset to a clean state on import failure?
            # self.reset_state()
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive meta-learner status, including connection checks.

        Returns:
            Dict with status information.
        """
        # Perform quick connection check
        memory_ok = self.memory is not None
        patterns_ok = self.pattern_memory is not None
        # Add others if needed

        status = {
            "enabled": self.enabled,
            "task_count": len(self.learning_tasks),
            "transfer_history_count": len(self.transfer_history),
            "meta_parameters": self.meta_parameters.copy(),
            "config_summary": {  # Include key config values
                "similarity_threshold": self.similarity_threshold,
                "transfer_strength": self.meta_parameters.get(
                    "transfer_strength"
                ),  # From actual params
                "max_history": self.max_history,
                "task_ttl_seconds": self.config.get("task_ttl_seconds"),
                "performance_decay_factor": self.config.get("performance_decay_factor"),
            },
            "sdk_connections": {
                "memory": memory_ok,
                "pattern_memory": patterns_ok,
                "art_controller": self.art_controller is not None,
                "meta_cognitive_engine": self.meta_cognitive is not None,
            },
        }
        logger.debug(f"MetaLearner status requested: {status}")
        return status


"""
Meta-Cognitive Module

Provides higher-order cognitive capabilities for the MetaConsciousness system.
This includes:
1. Self-reflection and introspection capabilities
2. Meta-reasoning about beliefs and memories
3. Cognitive state monitoring and regulation
4. Integration of ART and Omega3 feedback loops
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger("metaconsciousness.meta_cognitive")


class MetaCognitiveComponent:
    """
    Core component for meta-cognitive capabilities.

    This component provides the ability for the system to reason about its
    own cognitive processes, monitor its own thinking, and regulate its
    cognitive strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the meta-cognitive component.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.reflection_depth = self.config.get("reflection_depth", 2)
        self.last_reflection = 0.0
        self.reflection_interval = self.config.get(
            "reflection_interval_s", 300
        )  # 5 minutes
        self.reflection_history = []
        self.cognitive_metrics = {
            "certainty": 0.0,
            "coherence": 0.0,
            "flexibility": 0.0,
            "complexity": 0.0,
        }
        self.connected_to = []
        # Components
        self.metaconscious_agent = None
        self.art_controller = None
        self.memory_cluster = None
        self.omega3_agent = None
        self.belief_registry = None

        # Initialize connections
        self._connect_components()

        logger.info("MetaCognitiveComponent initialized")

    def _connect_components(self) -> None:
        """Connect to required components via SDKContext."""
        try:
            from MetaConsciousness.core.context import SDKContext

            # Try to connect to components
            self.metaconscious_agent = SDKContext.get("metaconscious_agent")
            self.art_controller = SDKContext.get("art_controller")
            self.memory_cluster = SDKContext.get("memory_cluster")
            self.omega3_agent = SDKContext.get("omega3_agent")
            self.belief_registry = SDKContext.get("belief_registry")

            # Register self
            SDKContext.register("meta_cognitive", self)
            logger.info("MetaCognitiveComponent registered with SDKContext")

            # Log connected components
            connected = []
            if self.metaconscious_agent:
                connected.append("metaconscious_agent")
            if self.art_controller:
                connected.append("art_controller")
            if self.memory_cluster:
                connected.append("memory_cluster")
            if self.omega3_agent:
                connected.append("omega3_agent")
            if self.belief_registry:
                connected.append("belief_registry")

            if connected:
                logger.info(f"Connected to components: {', '.join(connected)}")
            else:
                logger.warning(
                    "No components connected. Functionality will be limited."
                )

        except ImportError:
            logger.warning("SDKContext not available. Running in standalone mode.")
        except Exception as e:
            logger.error(f"Error connecting components: {e}")

    def trigger_reflection(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trigger a meta-cognitive reflection cycle.

        Args:
            context: Additional context for reflection

        Returns:
            Reflection result
        """
        if not self.enabled:
            logger.debug("Meta-cognitive reflection disabled")
            return {"status": "disabled"}

        # Check if enough time has passed since last reflection
        current_time = time.time()
        if current_time - self.last_reflection < self.reflection_interval:
            logger.debug("Skipping reflection due to rate limiting")
            return {"status": "rate_limited"}

        # Update last reflection time
        self.last_reflection = current_time

        # Combine provided context with system state
        full_context = self._gather_reflection_context()
        if context:
            full_context.update(context)

        # Perform the reflection
        result = self._reflect(full_context)

        # Store reflection in history
        self.reflection_history.append(
            {"timestamp": current_time, "context": full_context, "result": result}
        )

        # Limit history size
        max_history = self.config.get("max_reflection_history", 10)
        if len(self.reflection_history) > max_history:
            self.reflection_history = self.reflection_history[-max_history:]

        # Log reflection event
        logger.info(
            f"Meta-cognitive reflection completed: {result.get('summary', 'No summary')}"
        )

        return result

    def _gather_reflection_context(self) -> Dict[str, Any]:
        """
        Gather context for reflection from various components.

        Returns:
            Reflection context
        """
        context = {
            "timestamp": time.time(),
            "cognitive_metrics": self.cognitive_metrics.copy(),
        }

        # Get info from belief registry
        if self.belief_registry:
            try:
                if hasattr(self.belief_registry, "get_status"):
                    context["belief_status"] = self.belief_registry.get_status()

                if hasattr(self.belief_registry, "generate_insight"):
                    context["belief_insight"] = self.belief_registry.generate_insight()
            except Exception as e:
                logger.error(f"Error getting belief registry info: {e}")

        # Get info from memory cluster
        if self.memory_cluster:
            try:
                memory_info = {}

                if hasattr(self.memory_cluster, "get_stats"):
                    memory_info["stats"] = self.memory_cluster.get_stats()

                # Get episodic memory stats if available
                episodic = getattr(self.memory_cluster, "episodic", None)
                if episodic and hasattr(episodic, "get_stats"):
                    memory_info["episodic"] = episodic.get_stats()

                context["memory"] = memory_info
            except Exception as e:
                logger.error(f"Error getting memory cluster info: {e}")

        # Get info from ART controller
        if self.art_controller:
            try:
                if hasattr(self.art_controller, "get_status"):
                    context["art_status"] = self.art_controller.get_status()

                if hasattr(self.art_controller, "get_resonance_history"):
                    # Get limited history to avoid too much data
                    art_history = self.art_controller.get_resonance_history(5)
                    context["art_history"] = art_history
            except Exception as e:
                logger.error(f"Error getting ART controller info: {e}")

        # Get info from Omega3 agent
        if self.omega3_agent:
            try:
                if hasattr(self.omega3_agent, "get_status"):
                    context["omega3_status"] = self.omega3_agent.get_status()
            except Exception as e:
                logger.error(f"Error getting Omega3 agent info: {e}")

        return context

    def _reflect(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform meta-cognitive reflection.

        Args:
            context: Reflection context

        Returns:
            Reflection result
        """
        # This is a placeholder implementation
        # A full implementation would use the LLM to perform deeper reflections

        # Simple rule-based reflection for now
        result = {
            "timestamp": time.time(),
            "meta_level": self.reflection_depth,
            "insights": [],
            "adaptations": [],
            "status": "completed",
        }

        # Example insights based on memory
        if "memory" in context:
            memory_stats = context["memory"].get("stats", {})
            episodic_stats = context["memory"].get("episodic", {})

            # Check if episodic memory is getting full
            if episodic_stats:
                capacity = episodic_stats.get("capacity", 0)
                count = episodic_stats.get("total_events", 0)
                if capacity > 0 and count / capacity > 0.8:
                    result["insights"].append(
                        {
                            "type": "memory_capacity",
                            "message": "Episodic memory is approaching capacity.",
                            "data": {"usage": count / capacity},
                        }
                    )
                    result["adaptations"].append(
                        {
                            "type": "memory_management",
                            "action": "compress_episodic",
                            "reason": "Memory approaching capacity",
                        }
                    )

        # Example insights based on beliefs
        if "belief_insight" in context:
            insight = context["belief_insight"]
            if insight:
                result["insights"].append(
                    {
                        "type": "belief_insight",
                        "message": insight.get("message", "Belief insight generated."),
                        "data": insight,
                    }
                )

        # Example insights based on art status
        if "art_status" in context:
            art_status = context["art_status"]
            category_count = art_status.get("category_count", 0)
            max_categories = art_status.get("max_categories", 50)

            if max_categories > 0 and category_count / max_categories > 0.7:
                result["insights"].append(
                    {
                        "type": "art_categories",
                        "message": "ART network is developing many categories.",
                        "data": {"usage": category_count / max_categories},
                    }
                )

        # Set summary based on insights
        if result["insights"]:
            # Take the most important insight as summary
            result["summary"] = result["insights"][0]["message"]
        else:
            result["summary"] = "No significant insights generated."

        # Update cognitive metrics based on reflection
        self._update_cognitive_metrics(context, result)

        return result

    def _update_cognitive_metrics(
        self, context: Dict[str, Any], reflection_result: Dict[str, Any]
    ) -> None:
        """
        Update cognitive metrics based on reflection.

        Args:
            context: Reflection context
            reflection_result: Reflection result
        """
        # Example metric updates based on reflection insights

        # Certainty: Based on belief confidence
        if "belief_status" in context:
            avg_confidence = context["belief_status"].get("avg_confidence", 0.5)
            self.cognitive_metrics["certainty"] = avg_confidence

        # Coherence: Based on contradictions in beliefs
        if "belief_insight" in context:
            insight_type = context["belief_insight"].get("type")
            if insight_type == "contradictions":
                # Lower coherence if contradictions found
                self.cognitive_metrics["coherence"] = max(
                    0.0, self.cognitive_metrics["coherence"] - 0.1
                )
            else:
                # Slowly increase coherence otherwise
                self.cognitive_metrics["coherence"] = min(
                    1.0, self.cognitive_metrics["coherence"] + 0.05
                )

        # Flexibility: Based on ART network adaptations
        if "art_status" in context:
            recent_adaptations = context["art_status"].get("recent_adaptations", 0)
            if recent_adaptations > 0:
                self.cognitive_metrics["flexibility"] = min(
                    1.0, self.cognitive_metrics["flexibility"] + 0.1
                )
            else:
                # Slowly decrease flexibility if no recent adaptations
                self.cognitive_metrics["flexibility"] = max(
                    0.0, self.cognitive_metrics["flexibility"] - 0.05
                )

        # Complexity: Based on number of insights and adaptations
        insight_count = len(reflection_result.get("insights", []))
        adaptation_count = len(reflection_result.get("adaptations", []))
        complexity_factor = (
            insight_count + adaptation_count
        ) / 10  # Normalize to 0-1 range
        # Blend current with new value
        self.cognitive_metrics["complexity"] = (
            0.7 * self.cognitive_metrics["complexity"] + 0.3 * complexity_factor
        )

    def get_cognitive_metrics(self) -> Dict[str, float]:
        """
        Get current cognitive metrics.

        Returns:
            Dictionary of cognitive metrics
        """
        return self.cognitive_metrics.copy()

    def get_health(self) -> Dict[str, Any]:
        """
        Get component health status.

        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "enabled": self.enabled,
            "reflection_depth": self.reflection_depth,
            "last_reflection_age": time.time() - self.last_reflection,
            "reflection_count": len(self.reflection_history),
            "connected_components": {
                "metaconscious_agent": self.metaconscious_agent is not None,
                "art_controller": self.art_controller is not None,
                "memory_cluster": self.memory_cluster is not None,
                "omega3_agent": self.omega3_agent is not None,
                "belief_registry": self.belief_registry is not None,
            },
        }

    def process_external_event(
        self, event_type: str, event_data: Dict[str, Any]
    ) -> None:
        """
        Process an external event.

        Args:
            event_type: Type of event
            event_data: Event data
        """
        # This method would be called by other components to trigger
        # meta-cognitive processing of significant events

        if not self.enabled:
            return

        # Determine if event warrants reflection
        should_reflect = False

        # Events that always trigger reflection
        if event_type in [
            "anomaly_detected",
            "belief_contradiction",
            "significant_insight",
        ]:
            should_reflect = True

        # Other events based on significance
        if "significance" in event_data and event_data["significance"] > 0.7:
            should_reflect = True

        # Optional rate limiting for reflections
        if should_reflect:
            # Create context with event information
            context = {
                "event_type": event_type,
                "event_data": event_data,
                "is_triggered": True,
            }
            self.trigger_reflection(context)


# Create a default instance
def create_metacognitive(
    config: Optional[Dict[str, Any]] = None,
) -> MetaCognitiveComponent:
    """
    Create a MetaCognitiveComponent instance.

    Args:
        config: Optional configuration

    Returns:
        MetaCognitiveComponent instance
    """
    return MetaCognitiveComponent(config)
