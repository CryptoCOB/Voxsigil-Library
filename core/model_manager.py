# File: vanta_model_manager.py
"""
Neural model management for VantaCore ecosystem.

This module handles:
1. Loading and managing neural models (local and registered VantaCore services)
2. Generating embeddings
3. Managing model inference (text generation)
4. Vectorization and similarity search (conceptual, via embedding + np.dot)
5. Task tracking for model operations.
"""

import json
import logging
import os
import threading
import time
import uuid  # For generate_id stub
from typing import (  # Added Union, Callable, and Optional
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

# HOLO-1.5 imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# Handle numpy import gracefully to avoid circular import issues
try:
    import numpy as np

    # When numpy is available, ndarray is np.ndarray
    ndarray = np.ndarray  # type: ignore
    NUMPY_AVAILABLE = True
except (ImportError, AttributeError):
    NUMPY_AVAILABLE = False

    # Create minimal numpy stub for basic functionality
    class ndarray:  # This is the stub ndarray
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]
            if (
                isinstance(self.data, list)
                and self.data
                and isinstance(self.data[0], list)
            ):
                self.shape = (len(self.data), len(self.data[0]))
            elif isinstance(self.data, list):
                self.shape = (len(self.data),)
            else:
                self.shape = (1,)
            self.size = sum(s for s in self.shape if isinstance(s, int))

        @property
        def ndim(self):
            return len(self.shape)

        def flatten(self):
            result = []

            def _flatten(item):
                if isinstance(item, (list, tuple)):
                    for subitem in item:
                        _flatten(subitem)
                else:
                    result.append(item)

            _flatten(self.data)
            return ndarray(result)

        def __getitem__(self, key):
            item = self.data[key]
            return ndarray(item) if isinstance(item, list) else item

        def item(self):
            """Convert array to Python scalar - enhanced with validation"""
            if not hasattr(self, 'data'):
                raise ValueError("ndarray stub has no data attribute")
            
            flat_data = []

            def _recursive_flatten(item_list):
                if isinstance(item_list, (int, float)):
                    flat_data.append(item_list)
                    return
                
                if not isinstance(item_list, (list, tuple)):
                    # Try to convert other types to numeric if possible
                    try:
                        flat_data.append(float(item_list))
                        return
                    except (ValueError, TypeError):
                        raise ValueError(f"Cannot convert {type(item_list)} to numeric value")
                
                for i in item_list:
                    _recursive_flatten(i)

            try:
                _recursive_flatten(self.data)
            except (TypeError, AttributeError) as e:
                raise ValueError(f"Error flattening array data: {e}")
            
            if len(flat_data) == 0:
                raise ValueError("cannot convert empty array to scalar")
            elif len(flat_data) == 1:
                return flat_data[0]
            else:
                raise ValueError("can only convert an array of size 1 to a Python scalar")

        def __len__(self):
            """Return the length of the array"""
            if not hasattr(self, 'shape') or not self.shape:
                return 0
            return self.shape[0] if self.shape else 0
        
        def __str__(self):
            """String representation for debugging"""
            return f"ndarray_stub(shape={getattr(self, 'shape', 'unknown')}, data={str(self.data)[:50]}...)"
    class NumpyStub:
        ndarray = ndarray

        class linalg:
            @staticmethod
            def norm(data_obj):
                data_list = []
                current_data = []
                if hasattr(data_obj, "data") and isinstance(data_obj.data, list):  # type: ignore
                    current_data = data_obj.data  # type: ignore
                elif isinstance(data_obj, (list, tuple)):
                    current_data = list(data_obj)
                else:
                    return 0.0

                def _flatten_for_norm(item):
                    if isinstance(item, (list, tuple)):
                        for subitem in item:
                            _flatten_for_norm(subitem)
                    elif isinstance(item, (int, float)):
                        data_list.append(item)

                _flatten_for_norm(current_data)
                if data_list:
                    return sum(x * x for x in data_list) ** 0.5
                return 0.0

        @staticmethod
        def array(data, dtype=None):
            return ndarray(data)

        @staticmethod
        def asarray(data_obj, dtype=None):
            if isinstance(data_obj, ndarray):
                return data_obj
            return ndarray(data_obj)        @staticmethod
        def dot(a_obj, b_obj):
            """Enhanced dot product with validation"""
            # Validate inputs
            if a_obj is None or b_obj is None:
                logger.warning("NumpyStub.dot: received None input")
                return 0.0
            
            a_data_orig = a_obj.data if hasattr(a_obj, "data") else a_obj
            b_data_orig = b_obj.data if hasattr(b_obj, "data") else b_obj
            a_flat = []

            def _flatten_dot_a(item):
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        _flatten_dot_a(sub_item)
                elif isinstance(item, (int, float)):
                    a_flat.append(float(item))  # Ensure float type
                else:
                    logger.warning(f"NumpyStub.dot: non-numeric value in a: {item} ({type(item)})")

            _flatten_dot_a(a_data_orig)
            b_flat = []

            def _flatten_dot_b(item):
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        _flatten_dot_b(sub_item)
                elif isinstance(item, (int, float)):
                    b_flat.append(float(item))  # Ensure float type
                else:
                    logger.warning(f"NumpyStub.dot: non-numeric value in b: {item} ({type(item)})")

            _flatten_dot_b(b_data_orig)
            
            if not a_flat and not b_flat:
                return 0.0
            if not a_flat or not b_flat:
                logger.warning(f"NumpyStub.dot: one array is empty (a: {len(a_flat)}, b: {len(b_flat)})")
                return 0.0
            if len(a_flat) != len(b_flat):
                logger.warning(
                    f"NumpyStub.dot: shape mismatch {len(a_flat)} vs {len(b_flat)}, truncating to shorter length"
                )
                # Truncate to shorter length instead of returning 0
                min_len = min(len(a_flat), len(b_flat))
                a_flat = a_flat[:min_len]
                b_flat = b_flat[:min_len]
            
            try:
                result = sum(x * y for x, y in zip(a_flat, b_flat))
                return float(result)
            except (TypeError, ValueError) as e:
                logger.error(f"NumpyStub.dot: computation error: {e}")
                return 0.0

        @staticmethod
        def clip(val_obj, min_val, max_val):
            val_item = val_obj
            if hasattr(val_obj, "item"):  # If it's our ndarray stub
                val_item = val_obj.item()  # type: ignore
            elif (
                isinstance(val_obj, list) and len(val_obj) == 1
            ):  # If it's a list with one item
                val_item = val_obj[0]

            if not isinstance(val_item, (int, float)):
                logger.warning(
                    f"NumpyStub.clip: received non-numeric value {val_item} type {type(val_item)}"
                )
                # Attempt to convert if it's a single-element list from bad flatten
                if (
                    isinstance(val_item, list)
                    and len(val_item) == 1
                    and isinstance(val_item[0], (int, float))
                ):
                    val_item = val_item[0]
                else:
                    return float(min_val)  # Ensure return type consistency
            return float(max(min_val, min(max_val, val_item)))

        @staticmethod
        def _get_flat_numeric_list(data_obj):
            data_list_orig = data_obj.data if hasattr(data_obj, "data") else data_obj
            flat_list = []

            def _flatten(item):
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        _flatten(sub_item)
                elif isinstance(item, (int, float)):
                    flat_list.append(item)

            _flatten(data_list_orig)
            return flat_list

        @staticmethod
        def mean(data_obj, axis=None):  # Added axis for compatibility
            numeric_data = NumpyStub._get_flat_numeric_list(data_obj)
            if numeric_data:
                return sum(numeric_data) / len(numeric_data)
            return 0.0

        @staticmethod
        def std(data_obj, axis=None):  # Added axis for compatibility
            numeric_data = NumpyStub._get_flat_numeric_list(data_obj)
            if len(numeric_data) > 0:
                mu = NumpyStub.mean(numeric_data)  # type: ignore
                variance = sum((x - mu) ** 2 for x in numeric_data) / len(numeric_data)
                return variance**0.5
            return 0.0

        @staticmethod
        def max(data_obj, axis=None):  # Added axis for compatibility
            numeric_data = NumpyStub._get_flat_numeric_list(data_obj)
            if numeric_data:
                return float(max(numeric_data))
            return 0.0

        @staticmethod
        def min(data_obj, axis=None):  # Added axis for compatibility
            numeric_data = NumpyStub._get_flat_numeric_list(data_obj)
            if numeric_data:
                return float(min(numeric_data))
            return 0.0

    if not NUMPY_AVAILABLE:
        np = NumpyStub()  # type: ignore
# Assuming vanta_core.py is accessible
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore
    from Vanta.core.UnifiedVantaCore import trace_event as vanta_trace_event_real

    VantaCore = UnifiedVantaCore  # type: ignore
    trace_event = vanta_trace_event_real  # type: ignore
    HAVE_VANTA_CORE_INSTALLED = True
except ImportError:
    HAVE_VANTA_CORE_INSTALLED = False

    class VantaCore:  # type: ignore
        def get_component(self, name: str, default: Any = None) -> Any:
            logger.warning(
                f"[VantaCoreStub] get_component for '{name}' returning default."
            )
            return default

        def register_component(
            self, name: str, comp: Any, meta: Dict[str, Any] | None = None
        ) -> None:
            logger.debug(f"[VantaCoreStub] register_component called for '{name}'")

        def publish_event(
            self,
            etype: str,
            data: Dict[str, Any] | None = None,
            source: str | None = None,
        ) -> None:
            logger.debug(
                f"[VantaCoreStub] publish_event: {etype}, Data: {data}, Src: {source}"
            )

    def trace_event(  # type: ignore
        event_type: str,
        metadata: Dict[str, Any] | None = None,
        source_override: str | None = None,
    ) -> None:
        logger.debug(
            f"[TraceEventStub] Type:{event_type}, Meta:{metadata}, Src:{source_override}"
        )


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("VantaCore.ModelManager")


# --- Stubs for .common imports (TaskState, generate_id) ---
class TaskState:
    """Minimal stub for TaskState if .common is not available."""

    def __init__(self, name: str, metadata: Dict[str, Any] | None = None):
        self.task_id: str = generate_id()
        self.name: str = name
        self.metadata: Dict[str, Any] = metadata or {}
        self.status: str = "pending"
        self.progress: float = 0.0
        self.created: float = time.time()
        self.last_updated: float = time.time()
        self.completed: float | None = None
        self.result: Any = None
        self.error: str | None = None
        logger.debug(f"TaskState stub created: {self.task_id} ({self.name})")

    def update_progress(self, progress: float):
        self.progress = progress
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def generate_id() -> str:
    """Minimal stub for generate_id."""
    return uuid.uuid4().hex


class ModelManagerConfig:
    def __init__(
        self,
        log_level: str = "INFO",
        default_embedding_model: str = "all-MiniLM-L6-v2",
        default_reasoning_model: str = "default_reasoning_model_key",
        max_embedding_cache_size: int = 1000,
        embedding_cache_ttl_s: int = 3600,
        local_models_config: Dict[str, Any] | None = None,
        vanta_embedding_service_name: str = "embedding_service",
        vanta_language_model_service_name: str = "language_model_service",
        vanta_reasoning_engine_service_name: str = "reasoning_engine_service",
        vanta_lmstudio_adapter_name: str = "lmstudio_adapter_service",
        allow_sdk_fallback: bool = False,
    ):
        self.log_level = log_level
        self.default_embedding_model = default_embedding_model
        self.default_reasoning_model = default_reasoning_model
        self.max_embedding_cache_size = max_embedding_cache_size
        self.embedding_cache_ttl_s = embedding_cache_ttl_s
        self.local_models_config = local_models_config or {}
        self.vanta_embedding_service_name = vanta_embedding_service_name
        self.vanta_language_model_service_name = vanta_language_model_service_name
        self.vanta_reasoning_engine_service_name = vanta_reasoning_engine_service_name
        self.vanta_lmstudio_adapter_name = vanta_lmstudio_adapter_name
        self.allow_sdk_fallback = allow_sdk_fallback

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


@vanta_core_module(
    name="model_manager",
    subsystem="model_management", 
    mesh_role=CognitiveMeshRole.MANAGER,
    description="Neural model management for VantaCore ecosystem with comprehensive model orchestration and HOLO-1.5 integration",
    capabilities=["model_loading", "inference_generation", "embedding_computation", "vectorization", "similarity_search", "model_orchestration", "task_management"],
    cognitive_load=2.8,
    symbolic_depth=2,
    collaboration_patterns=["model_serving", "cognitive_inference", "knowledge_vectorization"]
)
class ModelManager(BaseCore):
    COMPONENT_NAME = "model_manager"

    def set_activation_hook(self, hook_function: Callable) -> bool:
        logger.info("ModelManager: Activation hook set.")
        self.activation_hook = hook_function
        return True

    def __init__(self, vanta_core: VantaCore, config: ModelManagerConfig):
        # Initialize BaseCore with HOLO-1.5 mesh capabilities
        super().__init__(vanta_core, config.__dict__ if hasattr(config, '__dict__') else {})
        
        self.vanta_core = vanta_core
        self.config = config
        self._available_models: Dict[str, Any] = {}  # Stores config and metadata
        self._loaded_model_instances: Dict[str, Any] = {}  # Stores actual model objects
        self.tasks: Dict[str, TaskState] = {}
        self.embedding_cache: Dict[
            str, Any
        ] = {}  # Using Any for cache value for flexibility
        self.embedding_cache_lru: List[str] = []
        self.cache_lock = threading.Lock()
        self.max_cache_size = config.max_embedding_cache_size

        # Initialize models based on config
        self._initialize_models()
        self.models: Dict[str, Dict[str, Any]] = {}  # type: ignore
        self.default_embedding_model = config.default_embedding_model
        self.default_reasoning_model = config.default_reasoning_model
        self.cach_ttl_s = config.embedding_cache_ttl_s

        logger.info(
            f"Base ModelManager initialized. Default Embed: {self.default_embedding_model}, Default Reason: {self.default_reasoning_model}"
        )

    async def initialize(self) -> bool:
        """Initialize the ModelManager for BaseCore compliance."""
        try:
            # Ensure models are loaded and ready
            logger.info("ModelManager initialized successfully with HOLO-1.5 enhancement")
            return True
        except Exception as e:
            logger.error(f"Error initializing ModelManager: {e}")
            return False

    def _initialize_models(self):
        """Registers models from config for lazy loading and attempts to get shared VantaCore services."""
        try:
            models_from_vanta_registry = self._get_models_from_vanta_registry()
            if models_from_vanta_registry:
                logger.info(
                    f"Found {len(models_from_vanta_registry)} model services in VantaCore registry."
                )
                self.models.update(models_from_vanta_registry)

            # Register models from this manager's own configuration (local_models_config)
            # This replaces the old `config.get("models", {})` part
            if self.config.local_models_config:
                logger.info(
                    f"Registering {len(self.config.local_models_config)} local models from ModelManagerConfig."
                )
            for model_name, model_info in self.config.local_models_config.items():
                try:
                    model_type = model_info.get("type", "unknown").lower()
                    model_path = model_info.get("path")
                    if (
                        model_name in self.models
                    ):  # Already found from VantaCore, skip re-registration
                        logger.info(
                            f"Model '{model_name}' already known from VantaCore registry, skipping local config registration."
                        )
                        continue

                    if model_type == "embedding":
                        self._register_embedding_model_for_loading(
                            model_name, model_path, model_info
                        )
                    elif model_type == "reasoning":
                        self._register_reasoning_model_for_loading(
                            model_name, model_path, model_info
                        )
                    else:
                        # For VantaRuntimeModelManager, 'chat' or other types might be handled by its specialized discovery
                        if not isinstance(
                            self, VantaRuntimeModelManager
                        ):  # Only log warning if base ModelManager
                            logger.warning(
                                f"Unknown model type: '{model_type}' for '{model_name}' in local_models_config."
                            )
                except Exception as e_reg:
                    logger.error(
                        f"Error registering local model '{model_name}' for loading: {e_reg}"
                    )

            if (
                not self.models
            ):  # If still no models after Vanta registry and local config
                logger.warning(
                    "No models were found in VantaCore registry or specified in local config."
                )
                self.health_state = "degraded_no_models"
            else:
                self.health_state = "ok_models_registered"
                logger.info(
                    f"Total {len(self.models)} models/services registered (VantaCore registry + local). Actual loading is lazy."
                )

        except Exception as e_init:
            logger.exception(f"Error during ModelManager _initialize_models: {e_init}")
            self.health_state = "error_initialization"

    def _get_models_from_vanta_registry(self) -> Dict[str, Any]:
        """Tries to get pre-configured/shared model services from VantaCore registry."""
        models_via_vanta: Dict[str, Any] = {}
        service_map = {
            self.config.vanta_embedding_service_name: "embedding",
            self.config.vanta_language_model_service_name: "reasoning",  # Assuming "language_model" maps to "reasoning" type
            self.config.vanta_reasoning_engine_service_name: "reasoning",
            # self.config.vanta_lmstudio_adapter_name: "reasoning", # LMStudio adapter is special, handled by VantaRuntimeModelManager
        }
        for service_registry_name, model_type in service_map.items():
            service_instance = self.vanta_core.get_component(service_registry_name)
            if service_instance:
                # Use a consistent key for self.models, e.g., "vanta_shared_embedding"
                model_key = (
                    f"vanta_shared_{service_registry_name.replace('_service', '')}"
                )
                models_via_vanta[model_key] = {
                    "model": service_instance,
                    "type": model_type,
                    "source": "vanta_registry",
                    "loaded": True,  # Assume VantaCore components are ready
                    "name_in_registry": service_registry_name,
                }
                logger.info(
                    f"Using '{service_registry_name}' (type: {model_type}) from VantaCore registry as '{model_key}'."
                )
        return models_via_vanta

    def _register_embedding_model_for_loading(
        self, name: str, path: str | None, model_cfg_info: Dict[str, Any]
    ):
        if not path:
            logger.warning(f"Embedding model {name} has no path. Cannot lazy-load.")
            return
        self.models[name] = {
            "type": "embedding",
            "path": path,
            "config": model_cfg_info,
            "model": None,
            "loaded": False,
            "source": "local_mm_config",  # Source is ModelManager's own config
        }
        logger.info(
            f"Registered local embedding model '{name}' for lazy loading from: {path}"
        )

    def _register_reasoning_model_for_loading(
        self, name: str, path: str | None, model_cfg_info: Dict[str, Any]
    ):
        if not path:
            logger.warning(f"Reasoning model {name} has no path. Cannot lazy-load.")
            return
        self.models[name] = {
            "type": "reasoning",
            "path": path,
            "config": model_cfg_info,
            "model": None,
            "loaded": False,
            "source": "local_mm_config",
        }
        logger.info(
            f"Registered local reasoning model '{name}' for lazy loading from: {path}"
        )

    def _attempt_load_embedding_model_instance(
        self, model_name: str, model_info: Dict[str, Any]
    ) -> Any:
        path = model_info.get("path")
        logger.info(f"Lazy loading embedding model: '{model_name}' from path: {path}")
        try:
            from sentence_transformers import (
                SentenceTransformer,  # Requires pip install sentence-transformers
            )

            instance = SentenceTransformer(path)  # Path can be HF name or local dir
            self.models[model_name].update({"model": instance, "loaded": True})
            logger.info(f"Successfully lazy-loaded embedding model: '{model_name}'")
            return instance
        except ImportError:
            logger.error(
                "SentenceTransformers library not found for embedding. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(
                f"Error lazy-loading embedding model '{model_name}' from '{path}': {e}"
            )
        self.models[model_name]["loaded"] = False
        return None

    def _attempt_load_reasoning_model_instance(
        self, model_name: str, model_info: Dict[str, Any]
    ) -> Any:
        path = model_info.get("path")
        model_specific_config = model_info.get("config", {})
        loader_hint = model_specific_config.get("loader", "transformers").lower()
        logger.info(
            f"Lazy loading reasoning model: '{model_name}' from path: '{path}' using loader: {loader_hint}"
        )
        try:
            instance = None
            if loader_hint == "transformers":
                from transformers.models.auto.modeling_auto import AutoModelForCausalLM
                from transformers.models.auto.tokenization_auto import (
                    AutoTokenizer,  # Requires pip install transformers torch (or tensorflow)
                )

                if not path:
                    logger.error(f"Path missing for Transformers model '{model_name}'")
                    return None
                tokenizer = AutoTokenizer.from_pretrained(path)
                instance = AutoModelForCausalLM.from_pretrained(path)
                self.models[model_name].update(
                    {"tokenizer": tokenizer, "model": instance, "loaded": True}
                )
            elif loader_hint == "llama.cpp":
                from llama_cpp import Llama  # Requires pip install llama-cpp-python

                if not path or not os.path.isfile(path):
                    logger.error(
                        f"GGUF file path invalid for Llama.cpp model '{model_name}': {path}"
                    )
                    return None
                llm_params = {
                    "model_path": path,
                    "n_ctx": model_specific_config.get("n_ctx", 2048),
                    "n_gpu_layers": model_specific_config.get("n_gpu_layers", 0),
                }
                instance = Llama(**llm_params)
                self.models[model_name].update({"model": instance, "loaded": True})
            else:
                logger.error(
                    f"Unsupported loader '{loader_hint}' for reasoning model '{model_name}'"
                )
                return None

            if instance:
                logger.info(
                    f"Successfully lazy-loaded reasoning model ({loader_hint}): '{model_name}'"
                )
            return instance
        except ImportError as ie:
            logger.error(
                f"Lib missing for loader '{loader_hint}' for model '{model_name}'. Install required packages. Details: {ie}"
            )
        except Exception as e:
            logger.error(
                f"Error lazy-loading reasoning model '{model_name}' from '{path}': {e}"
            )
        self.models[model_name]["loaded"] = False
        return None

    def _get_model_instance(
        self, model_key_or_name: str, expected_type: str
    ) -> Any | None:
        """Internal helper to get a model instance, handling aliases and lazy loading."""
        model_info = self.models.get(model_key_or_name)

        # If VantaRuntimeModelManager, it might have more sophisticated alias resolution
        if not model_info and isinstance(self, VantaRuntimeModelManager):
            model_key_or_name = self._resolve_alias_to_key(
                model_key_or_name, expected_type
            )  # type: ignore
            if model_key_or_name:
                model_info = self.models.get(model_key_or_name)

        if not model_info:
            logger.warning(
                f"Model key/name '{model_key_or_name}' not registered with ModelManager."
            )
            return None
        if model_info.get("type") != expected_type:
            logger.warning(
                f"Model '{model_key_or_name}' is type '{model_info.get('type')}', expected '{expected_type}'."
            )
            return None

        if model_info.get("source") == "vanta_registry" and model_info.get(
            "loaded"
        ):  # From VantaCore registry
            return model_info.get("model")

        # For locally configured models (needs loading)
        if not model_info.get("loaded", False) and model_info.get("path"):
            logger.info(
                f"Triggering lazy load for {expected_type} model: {model_key_or_name}"
            )
            if expected_type == "embedding":
                self._attempt_load_embedding_model_instance(
                    model_key_or_name, model_info
                )
            elif expected_type == "reasoning":
                self._attempt_load_reasoning_model_instance(
                    model_key_or_name, model_info
                )

        return model_info.get("model") if model_info.get("loaded") else None

    def get_embedding(
        self,
        text: Union[str, List[str]],
        model_name_or_alias: str | None = None,
        use_cache: bool = True,
    ) -> Union[Any, List[Any], None]:  # Changed np.ndarray to Any
        effective_model_key = model_name_or_alias or self.default_embedding_model
        instance_key_to_use = effective_model_key

        # Cache logic
        cache_hit_value = None
        if use_cache and isinstance(text, str):
            cache_key = f"{effective_model_key}:{text}"  # Key should be based on actual model key/name used
            with self.cache_lock:
                if cache_key in self.embedding_cache:
                    entry = self.embedding_cache[cache_key]
                    if time.time() - entry["timestamp"] < self.cach_ttl_s:
                        logger.debug(
                            f"Embedding cache HIT for model '{effective_model_key}' key_prefix '{text[:20]}...'"
                        )
                        cache_hit_value = entry["embedding"]
        if cache_hit_value is not None:
            return cache_hit_value

        try:
            # Resolve alias to actual model key if VantaRuntimeModelManager, then get instance
            if isinstance(self, VantaRuntimeModelManager):
                resolved_key = self._resolve_alias_to_key(
                    effective_model_key, "embedding"
                )  # type: ignore
                if resolved_key:
                    instance_key_to_use = resolved_key

            model_instance = self._get_model_instance(instance_key_to_use, "embedding")
            if not model_instance:
                logger.error(
                    f"Embedding model for key/alias '{effective_model_key}' (resolved to '{instance_key_to_use}') failed to load or not found."
                )
                return None  # Return None if model instance is not found/loaded

            start_time = time.time()
            embeddings: Any = None
            if hasattr(model_instance, "encode"):
                embeddings = model_instance.encode(text)  # type: ignore
            elif hasattr(model_instance, "embed"):
                embeddings = model_instance.embed(text)  # type: ignore
            else:
                logger.error(
                    f"Model for '{instance_key_to_use}' has no 'encode' or 'embed' method."
                )
                return None
            if embeddings is None:
                logger.error(f"Embedding for '{instance_key_to_use}' returned None.")
                return None

            elapsed = time.time() - start_time
            if isinstance(text, str):
                input_s = len(text)
                b_size = 1
            else:
                input_s = sum(len(t) for t in text)
                b_size = len(text)
            trace_event(
                f"{self.COMPONENT_NAME}.embedding_generated",
                {
                    "model": instance_key_to_use,
                    "input_chars": input_s,
                    "batch": b_size,
                    "time_ms": elapsed * 1000,
                },
                source_override=self.COMPONENT_NAME,
            )
            input_s = len(text) if isinstance(text, str) else sum(len(t) for t in text)
            b_size = 1 if isinstance(text, str) else len(text)
            trace_event(
                f"{self.COMPONENT_NAME}.embedding_generated",
                {
                    "model": instance_key_to_use,
                    "input_chars": input_s,
                    "batch": b_size,
                    "time_ms": elapsed * 1000,
                },
                source_override=self.COMPONENT_NAME,
            )

            # Ensure numpy array output, handle SentenceTransformer list of arrays for batch
            if isinstance(embeddings, list) and isinstance(
                text, list
            ):  # Batch input, batch output
                embeddings = np.array(embeddings)
            elif not isinstance(embeddings, np.ndarray) and isinstance(
                text, str
            ):  # Single input, ensure ndarray
                embeddings = np.array(embeddings)

            if use_cache and isinstance(text, str) and embeddings is not None:
                with self.cache_lock:
                    # Use the key that led to this instance for caching (instance_key_to_use)
                    final_cache_key = f"{instance_key_to_use}:{text}"
                    self.embedding_cache[final_cache_key] = {
                        "embedding": embeddings,
                        "timestamp": time.time(),
                    }
                    self._prune_embedding_cache()
            return embeddings
        except Exception as e:
            logger.error(
                f"Error in get_embedding (model key/alias: '{effective_model_key}'): {e}",
                exc_info=True,
            )
            return None

    def _prune_embedding_cache(self):
        """Prunes the embedding cache if it exceeds max_cache_size using LRU."""
        if len(self.embedding_cache) > self.max_cache_size:
            num_to_prune = (
                len(self.embedding_cache)
                - self.max_cache_size
                + (self.max_cache_size // 10)
            )  # Prune a bit more
            sorted_cache = sorted(
                self.embedding_cache.items(), key=lambda item: item[1]["timestamp"]
            )
            for i in range(min(num_to_prune, len(sorted_cache))):
                del self.embedding_cache[sorted_cache[i][0]]
            logger.info(
                f"Pruned {min(num_to_prune, len(sorted_cache))} items from embedding cache. Size now: {len(self.embedding_cache)}"
            )

    def compute_similarity(
        self, emb1: Any, emb2: Any
    ) -> float:  # Changed np.ndarray to Any
        if emb1 is None or emb2 is None:
            logger.debug("compute_similarity: one or both embeddings are None.")
            return 0.0

        # If numpy is not available, convert to NumpyStub.ndarray if they aren\'t already
        if not NUMPY_AVAILABLE:
            if not isinstance(emb1, NumpyStub.ndarray):  # type: ignore
                emb1 = NumpyStub.array(emb1)  # type: ignore
            if not isinstance(emb2, NumpyStub.ndarray):  # type: ignore
                emb2 = NumpyStub.array(emb2)  # type: ignore
            # Check if data exists after conversion
            if not emb1.data or not emb2.data or not emb1.data[0] or not emb2.data[0]:  # type: ignore
                logger.debug(
                    "compute_similarity (stub): one or both embeddings have no data after conversion."
                )
                return 0.0
        elif NUMPY_AVAILABLE:  # Real numpy operations
            if not isinstance(emb1, np.ndarray):  # type: ignore
                emb1 = np.array(emb1)  # type: ignore
            if not isinstance(emb2, np.ndarray):  # type: ignore
                emb2 = np.array(emb2)  # type: ignore
            if emb1.size == 0 or emb2.size == 0:  # type: ignore
                logger.debug(
                    "compute_similarity (numpy): one or both embeddings are empty."
                )
                return 0.0

        # Proceed with calculation
        if NUMPY_AVAILABLE:
            n1 = np.linalg.norm(emb1)  # type: ignore
            n2 = np.linalg.norm(emb2)  # type: ignore
            if n1 == 0 or n2 == 0:
                return 0.0
            dot_product = np.dot(emb1, emb2)  # type: ignore
            similarity = dot_product / (n1 * n2)
            # np.clip can return ndarray, ensure .item() is called on a 0-d array
            clipped_similarity = np.clip(similarity, -1.0, 1.0)  # type: ignore
            return float(clipped_similarity)
        else:  # NumpyStub operations
            n1 = np.linalg.norm(emb1)  # type: ignore
            n2 = np.linalg.norm(emb2)  # type: ignore
            if n1 == 0 or n2 == 0:
                return 0.0
            dot_product = np.dot(emb1, emb2)  # type: ignore
            if n1 * n2 == 0:  # Avoid division by zero if norms were zero but not caught
                return 0.0
            similarity = dot_product / (n1 * n2)
            return np.clip(similarity, -1.0, 1.0)  # type: ignore

    def search_texts(
        self,
        query_text: str,
        texts_to_search: List[str],
        model_name_or_alias: str | None = None,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> List[tuple[int, str, float]]:
        if not query_text or not texts_to_search:
            return []

        q_emb = self.get_embedding(
            query_text, model_name_or_alias=model_name_or_alias, use_cache=use_cache
        )
        if q_emb is None:
            logger.error(f"Could not generate embedding for query: {query_text[:100]}")
            return []

        doc_embs = self.get_embedding(
            texts_to_search,
            model_name_or_alias=model_name_or_alias,
            use_cache=use_cache,
        )
        if doc_embs is None or (
            isinstance(doc_embs, list) and not all(doc_embs)
        ):  # check for list of Nones
            logger.error("Could not generate embeddings for some documents.")
            # Filter out None embeddings if it's a list
            if isinstance(doc_embs, list) and isinstance(texts_to_search, list):
                valid_docs_embs = []
                valid_texts = []
                for i, emb in enumerate(doc_embs):
                    if emb is not None:
                        valid_docs_embs.append(emb)
                        valid_texts.append(texts_to_search[i])
                doc_embs = valid_docs_embs
                texts_to_search = valid_texts
                if not doc_embs:
                    return []  # All failed
            else:  # If doc_embs is None or not a list when it should be
                return []

        # Ensure doc_embs is a list for iteration
        # Handled by making get_embedding return List[Any] or None
        # Further checks:
        if doc_embs is None:  # This check was already there, good.
            logger.error(
                "Could not generate embeddings for documents (doc_embs is None)."
            )
            return []

        if not isinstance(doc_embs, list):
            # If a single text was passed to get_embedding, it might return a single embedding, not a list.
            # If multiple texts were passed, it should return a list.
            if isinstance(texts_to_search, str):  # Single query text, single doc text
                doc_embs = [doc_embs]
            else:  # Multiple doc texts, but doc_embs is not a list - this is an issue.
                logger.error(
                    f"doc_embs is not a list as expected when searching multiple texts. Type: {type(doc_embs)}"
                )
                return []

        if len(doc_embs) != len(
            texts_to_search if isinstance(texts_to_search, list) else [texts_to_search]
        ):
            logger.error(
                f"Number of document embeddings ({len(doc_embs)}) does not match number of texts ({len(texts_to_search if isinstance(texts_to_search, list) else [texts_to_search])})."
            )
            return []

        scores = []
        for i, doc_emb_item in enumerate(doc_embs):
            if doc_emb_item is None:
                logger.warning(
                    f"Skipping document index {i} ('{texts_to_search[i][:50]}...') due to None embedding."
                )
                continue

            current_q_emb = q_emb
            current_doc_emb = doc_emb_item

            # Ensure types are consistent for compute_similarity
            if NUMPY_AVAILABLE:
                if not isinstance(current_q_emb, np.ndarray):  # type: ignore
                    current_q_emb = np.array(current_q_emb)  # type: ignore
                if not isinstance(current_doc_emb, np.ndarray):  # type: ignore
                    current_doc_emb = np.array(current_doc_emb)  # type: ignore
            else:
                if not isinstance(current_q_emb, NumpyStub.ndarray):  # type: ignore
                    current_q_emb = NumpyStub.array(current_q_emb)  # type: ignore
                if not isinstance(current_doc_emb, NumpyStub.ndarray):  # type: ignore
                    current_doc_emb = NumpyStub.array(current_doc_emb)  # type: ignore

            similarity = self.compute_similarity(current_q_emb, current_doc_emb)
            scores.append((i, texts_to_search[i], similarity))

        # Sort by score descending, then take top_k
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def generate_text(
        self,
        prompt: str,
        model_name_or_alias: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str | None:
        effective_model_key = model_name_or_alias or self.default_reasoning_model
        try:
            instance_key_to_use = effective_model_key
            if isinstance(self, VantaRuntimeModelManager):  # Resolve alias
                resolved_key = self._resolve_alias_to_key(
                    effective_model_key, "reasoning"
                )  # type: ignore
                if resolved_key:
                    instance_key_to_use = resolved_key

            model_instance = self._get_model_instance(instance_key_to_use, "reasoning")
            if not model_instance:
                logger.error(
                    f"Reasoning model for key/alias '{effective_model_key}' (-> '{instance_key_to_use}') failed to load/not found."
                )
                return None  # Ensure return

            start_tm = time.time()
            result_text: str | None = None
            model_entry = self.models.get(
                instance_key_to_use, {}
            )  # Get the entry from self.models
            gen_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            # Handle different model types / loaders
            if (
                model_entry.get("source") == "vanta_registry"
            ):  # Shared VantaCore service
                if hasattr(model_instance, "generate_text"):
                    result_text = model_instance.generate_text(
                        prompt, **gen_params
                    )  # Assumed method
                elif hasattr(model_instance, "generate"):
                    result_text = model_instance.generate(prompt, **gen_params)
                elif hasattr(model_instance, "complete"):
                    result_text = model_instance.complete(prompt, **gen_params)
            elif model_entry.get("source") in [
                "local_mm_config",
                "vanta_config_local",
            ]:  # Locally loaded
                loader_hint = model_entry.get("config", {}).get(
                    "loader", "transformers"
                )
                if loader_hint == "transformers":
                    tokenizer = model_entry.get("tokenizer")
                    hf_model = model_entry.get(
                        "model"
                    )  # This is the actual HF model instance
                    if (
                        not tokenizer
                        or not hf_model
                        or not hasattr(hf_model, "generate")
                    ):
                        return None
                    inputs = tokenizer(prompt, return_tensors="pt")
                    # Add device if GPU is used by hf_model: .to(hf_model.device)
                    outputs = hf_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        **kwargs,
                    )
                    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elif loader_hint == "llama.cpp":
                    llm_cpp_instance = model_entry.get("model")
                    if not llm_cpp_instance or not hasattr(
                        llm_cpp_instance, "create_completion"
                    ):
                        return None
                    comp = llm_cpp_instance.create_completion(
                        prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
                    )
                    if comp and comp.get("choices"):
                        result_text = comp["choices"][0].get("text", "")

            if result_text is None:
                logger.error(
                    f"Model '{instance_key_to_use}' has no recognized generation method or failed."
                )
                return None

            elapsed_tm = time.time() - start_tm
            trace_event(
                f"{self.COMPONENT_NAME}.text_generated",
                {
                    "model": instance_key_to_use,
                    "prompt_len": len(prompt),
                    "result_len": len(result_text),
                    "temp": temperature,
                    "max_tok": max_tokens,
                    "time_s": elapsed_tm,
                },
                source_override=self.COMPONENT_NAME,
            )
            return result_text
        except Exception as e:
            logger.error(
                f"Error generating text (model key/alias: '{effective_model_key}'): {e}",
                exc_info=True,
            )
            return None  # Task management methods (kept as stubs, assuming VantaCore might have its own task manager)

    def start_task(self, name: str, meta: dict | None = None) -> str:
        task = TaskState(name, meta)
        self.tasks[task.task_id] = task
        trace_event(
            f"{self.COMPONENT_NAME}.task_started",
            {"id": task.task_id, "name": name},
            source_override=self.COMPONENT_NAME,
        )
        return task.task_id

    def update_task(
        self,
        tid: str,
        prog: float | None = None,
        stat: str | None = None,
        res: Any = None,
        err: str | None = None,
    ) -> bool:
        if tid not in self.tasks:
            return False
        task = self.tasks[tid]  # Renamed variable to avoid 't' undefined error
        if prog is not None:
            task.progress = prog
        if stat:
            task.status = stat
        if res is not None:
            task.result = res
        if err:
            task.error = err
        task.last_updated = time.time()
        if stat in ["completed", "failed"] and task.completed is None:
            task.completed = time.time()
            trace_event(
                f"{self.COMPONENT_NAME}.task_ended",
                {
                    "id": tid,
                    "name": task.name,
                    "status": stat,
                    "dur_s": task.completed - task.created if task.completed else -1,
                },
                source_override=self.COMPONENT_NAME,
            )
        return True

    def get_task(self, tid: str) -> TaskState | None:
        return self.tasks.get(tid)

    def get_all_tasks(self) -> dict[str, TaskState]:
        return self.tasks.copy()

    def cleanup_tasks(
        self, max_age_s: int = 3600
    ) -> int:  # Kept max_age_s optional to match original
        now = time.time()
        to_del = [
            tid
            for tid, t in self.tasks.items()
            if t.completed and (now - t.completed > max_age_s)
        ]
        for tid in to_del:
            del self.tasks[tid]
        if to_del:
            logger.info(f"Cleaned up {len(to_del)} old tasks.")
        return len(to_del)

    def clear_embedding_cache(self) -> int:  # Renamed for clarity
        with self.cache_lock:
            count = len(self.embedding_cache)
            self.embedding_cache.clear()
            logger.info(f"Cleared {count} items from embedding cache.")
        return count

    def get_model_info_summary(
        self,
    ) -> dict[str, dict[str, Any]]:  # Renamed for clarity
        res = {}
        for name, info_dict in self.models.items():
            cfg = {
                k: v
                for k, v in info_dict.get("config", {}).items()
                if k not in ["model", "tokenizer"] and not callable(v)
            }
            res[name] = {
                "type": info_dict.get("type", "?"),
                "loaded": info_dict.get("loaded", False),
                "source": info_dict.get("source", "?"),
                "path": info_dict.get("path"),
                "config_summary": cfg,
            }
        return res

    def get_health_status(self) -> str:  # Renamed for clarity
        # Simple health: ok if models are registered, degraded if not. Can be more complex.
        if (
            not self.models and self.health_state == "initializing"
        ):  # Check if init failed to find any
            self.health_state = "degraded_no_models_found"
        elif (
            self.health_state == "initializing" and self.models
        ):  # Init finished, models registered
            self.health_state = "ok_models_registered"
        # Real health check might try to load default models.
        return self.health_state

    def get_status(self) -> Dict[str, Any]:
        """Provide a basic status summary for ModelManager."""
        return {
            "health_status": self.get_health_status(),
            "models_registered": len(self.models),
            "tasks_count": len(self.tasks),
            "embedding_cache_size": len(self.embedding_cache),
        }

    def get_model_instance_by_name(
        self, model_name_or_alias: str
    ) -> Any | None:  # Renamed
        # This becomes the primary way to get a model, using the unified _get_model_instance
        # It tries to determine type based on registered info or typical use.
        model_info = self.models.get(model_name_or_alias)
        if isinstance(self, VantaRuntimeModelManager) and not model_info:  # type: ignore
            # VantaRuntimeModelManager might have _available_models with more info
            resolved_key_from_alias = self._resolve_alias_to_key(
                model_name_or_alias, None
            )  # type: ignore # Try resolving alias without type constraint
            if resolved_key_from_alias:
                model_info = self.models.get(resolved_key_from_alias)
                if model_info:
                    model_name_or_alias = (
                        resolved_key_from_alias  # Use the resolved key
                    )

        if not model_info:
            logger.warning(
                f"get_model_instance_by_name: Model '{model_name_or_alias}' not registered."
            )
            return None

        model_type = model_info.get("type")
        if model_type not in ["embedding", "reasoning"]:
            logger.warning(
                f"Model '{model_name_or_alias}' has unknown type '{model_type}'. Cannot determine access method."
            )
            return None

        return self._get_model_instance(model_name_or_alias, model_type)

    def training_job_active(self) -> bool:
        """
        Check if any model training jobs are currently active.

        Returns:
            bool: True if at least one training job is active, False otherwise.
        """
        logger.debug("Checking for active training jobs...")

        # Check if there are any tasks with training-related names that are not completed
        active_training_tasks = [
            task
            for task in self.tasks.values()
            if (
                task.status not in ["completed", "failed", "canceled"]
                and any(
                    kw in task.name.lower()
                    for kw in ["train", "finetune", "tuning", "adaptation"]
                )
            )
        ]

        is_active = len(active_training_tasks) > 0
        if is_active:
            task_names = [task.name for task in active_training_tasks]
            logger.info(
                f"Found {len(active_training_tasks)} active training tasks: {task_names}"
            )

        return is_active

    # --- Protocol compatibility methods for LearningManager ---
    def get_available_models(self) -> list[str]:
        """Return a list of available model names."""
        return list(self.models.keys()) if hasattr(self, "models") else []

    def get_model_status(self, model_name: str) -> dict:
        """Return status information for a specific model."""
        if hasattr(self, "models") and model_name in self.models:
            return {"status": "available", "name": model_name}
        return {"status": "unknown", "name": model_name}

    def request_model_tuning(
        self, model_name: str, tuning_params: Optional[dict] = None
    ) -> bool:
        """Stub for requesting model tuning."""
        # Implement actual tuning logic if available
        return False

    def update_model_weights(self, model_name: str, weights: dict) -> bool:
        """Stub for updating model weights."""
        # Implement actual weight update logic if available
        return False


class VantaRuntimeModelManager(ModelManager):
    """
    VantaRuntimeModelManager extends ModelManager with Vanta-specific discovery and registration.
    """

    COMPONENT_NAME_VRMM = (
        "vanta_runtime_model_manager"  # Different name for VantaCore registry
    )

    def __init__(
        self,
        vanta_core: VantaCore,
        config: ModelManagerConfig,
        register_to_vanta_core: bool = True,
    ):
        super().__init__(vanta_core, config)  # Pass vanta_core to parent
        self.register_to_vanta_core = register_to_vanta_core

        # _available_models: store detailed info about ALL models found, including those not managed by self.models directly (e.g. pure VantaCore services)
        self._available_models: Dict[str, Dict[str, Any]] = {}

        self._discover_and_register_all_models()  # Renamed for clarity

        if self._available_models:
            model_names_summary = ", ".join(
                sorted(
                    [
                        m_info.get("name", m_id)
                        for m_id, m_info in self._available_models.items()
                    ]
                )
            )
            logger.info(
                f"VantaRuntimeModelManager: Discovered/Processed {len(self._available_models)} total models/services: [{model_names_summary[:200]}...]"
            )
        else:
            logger.warning(
                "VantaRuntimeModelManager: No models found from any source (config, VantaCore registry, etc.)."
            )

        if self.register_to_vanta_core:
            self.vanta_core.register_component(
                self.COMPONENT_NAME_VRMM,
                self,
                {"type": "model_management_service_vanta"},
            )
            logger.info(
                f"VantaRuntimeModelManager registered with VantaCore as '{self.COMPONENT_NAME_VRMM}'"
            )

    def _discover_and_register_all_models(self):
        """Discovers from VantaCore registry, local config, and potentially other Vanta specific sources."""
        # 1. Models from parent's _initialize_models (which includes its local_config and VantaCore registry shared services)
        # These are already in self.models. We need to populate self._available_models from them.
        for model_key, model_info_dict in self.models.items():
            self._available_models[model_key] = {
                "source": model_info_dict.get("source", "unknown"),
                "type": model_info_dict.get("type"),
                "name": model_info_dict.get("config", {}).get(
                    "model_name", model_key
                ),  # Prefer configured name
                "vanta_key": model_key,  # The key used in self.models
                "loaded": model_info_dict.get("loaded", False),
            }
            if model_info_dict.get("source") == "vanta_registry":
                self._available_models[model_key]["provider"] = model_info_dict.get(
                    "name_in_registry"
                )

        # 2. Models from specific VantaCore services (e.g., an LMStudio Adapter if VantaCore provides one)
        lmstudio_adapter = self.vanta_core.get_component(
            self.config.vanta_lmstudio_adapter_name
        )
        if lmstudio_adapter and hasattr(
            lmstudio_adapter, "get_available_models_info"
        ):  # Assume adapter has this method
            try:
                lm_studio_model_list = (
                    lmstudio_adapter.get_available_models_info()
                )  # Expected: list of dicts
                for lm_model_data in lm_studio_model_list:
                    # lm_model_data might be like {"id": "ollama_model_name", "name": "Friendly Name", "provider_path": "http://..." }
                    lm_id = lm_model_data.get("id")  # This is the ID the adapter uses
                    if not lm_id:
                        continue

                    # Create a unique key for _available_models
                    available_key = f"lmstudio_svc_{lm_id.replace('/', '_')}"
                    self._available_models[available_key] = {
                        "source": "vanta_lmstudio_adapter",
                        "type": lm_model_data.get("type", "reasoning"),
                        "name": lm_model_data.get("name", lm_id),
                        "provider": self.config.vanta_lmstudio_adapter_name,
                        "adapter_model_id": lm_id,  # Store the ID used by the adapter
                        "loaded": True,  # Assume adapter handles connection state
                    }
                    # Note: These models are accessed via the adapter. _get_model_instance needs to know this.
                logger.info(
                    f"Processed {len(lm_studio_model_list)} models from VantaCore LMStudio adapter '{self.config.vanta_lmstudio_adapter_name}'."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get models from VantaCore LMStudio adapter '{self.config.vanta_lmstudio_adapter_name}': {e}"
                )

        # 3. Re-evaluate default models after full discovery
        self._update_default_model_selections()

        # Optionally, register the flat list of available models with VantaCore for other components' discovery
        self.vanta_core.register_component(
            "all_discovered_models_summary",
            self.get_available_models_summary_list(),
            meta={"type": "model_summary_list"},
        )

    def _update_default_model_selections(self):
        """Sets default_embedding_model and default_reasoning_model based on available and configured models."""

        # Reasoning Model
        reasoning_candidates: list[tuple[str, dict[str, Any]]] = []  # (key, info_dict)
        for key, info in self._available_models.items():
            if info.get("type") == "reasoning":
                reasoning_candidates.append((key, info))

        # Try configured default first
        cfg_def_reason = self.config.default_reasoning_model
        if cfg_def_reason:
            # Is it a direct key?
            if (
                cfg_def_reason in self._available_models
                and self._available_models[cfg_def_reason].get("type") == "reasoning"
            ):
                self.default_reasoning_model = cfg_def_reason
            else:  # Is it a "name" of an available model?
                found = next(
                    (
                        k
                        for k, i in reasoning_candidates
                        if i.get("name") == cfg_def_reason
                    ),
                    None,
                )
                if found:
                    self.default_reasoning_model = found
                else:
                    logger.warning(
                        f"Configured default reasoning model '{cfg_def_reason}' not found in available models. Picking first available."
                    )

        if (
            self.default_reasoning_model == self.config.default_reasoning_model
            and not self._is_valid_available_model_key(
                self.default_reasoning_model, "reasoning"
            )
        ):
            if reasoning_candidates:  # Pick the first available if current default is invalid or still the placeholder
                # Simple pick: first one. Could be more sophisticated (e.g. highest tier, local pref).
                self.default_reasoning_model = reasoning_candidates[0][0]
            else:
                logger.error("No reasoning models available to set as default.")
        logger.info(
            f"VantaRuntime: Default reasoning model finalized to: '{self.default_reasoning_model}'"
        )

        # Embedding Model (similar logic)
        embedding_candidates: list[tuple[str, dict[str, Any]]] = []
        for key, info in self._available_models.items():
            if info.get("type") == "embedding":
                embedding_candidates.append((key, info))

        cfg_def_embed = self.config.default_embedding_model
        if cfg_def_embed:
            if (
                cfg_def_embed in self._available_models
                and self._available_models[cfg_def_embed].get("type") == "embedding"
            ):
                self.default_embedding_model = cfg_def_embed
            else:
                found = next(
                    (
                        k
                        for k, i in embedding_candidates
                        if i.get("name") == cfg_def_embed
                    ),
                    None,
                )
                if found:
                    self.default_embedding_model = found
                else:
                    logger.warning(
                        f"Configured default embedding model '{cfg_def_embed}' not found. Picking first available."
                    )

        if (
            self.default_embedding_model == self.config.default_embedding_model
            and not self._is_valid_available_model_key(
                self.default_embedding_model, "embedding"
            )
        ):
            if embedding_candidates:
                self.default_embedding_model = embedding_candidates[0][0]
            else:
                logger.error("No embedding models available to set as default.")
        logger.info(
            f"VantaRuntime: Default embedding model finalized to: '{self.default_embedding_model}'"
        )

    def _is_valid_available_model_key(self, key: str, model_type: str) -> bool:
        """Checks if a key is in _available_models and matches type."""
        info = self._available_models.get(key)
        return info is not None and info.get("type") == model_type

    def _resolve_alias_to_key(
        self, name_or_alias: str, expected_type: str | None
    ) -> str | None:
        """Resolves a friendly name or alias to an actual model key in self.models or _available_models."""
        if name_or_alias in self.models and (
            not expected_type or self.models[name_or_alias].get("type") == expected_type
        ):
            return name_or_alias  # It's already a direct key in self.models
        if name_or_alias in self._available_models and (
            not expected_type
            or self._available_models[name_or_alias].get("type") == expected_type
        ):
            # If it's a key in _available_models, we need to check if it's also a key in self.models (for locally loaded)
            # or if it's externally managed (e.g. vanta_registry, lmstudio_svc)
            avail_info = self._available_models[name_or_alias]
            if avail_info.get("source") in [
                "vanta_config_local",
                "local_mm_config",
            ]:  # These are keys for self.models
                return name_or_alias
            elif avail_info.get(
                "vanta_key"
            ):  # If _available_models stores the self.models key
                return avail_info["vanta_key"]
            # For purely external services (like an LMStudio model not directly in self.models),
            # the key from _available_models is what we use to identify it when talking to its provider.
            return name_or_alias

        # Try to find by "name" field in _available_models
        for key, info in self._available_models.items():
            if info.get("name") == name_or_alias and (
                not expected_type or info.get("type") == expected_type
            ):
                # Similar logic as above: is this key for self.models or for an external provider?
                if info.get("source") in [
                    "vanta_config_local",
                    "local_mm_config",
                    "vanta_registry",
                    "sdk",
                ]:  # "sdk" source might still be in self.models
                    return key  # This key is likely in self.models
                elif info.get("vanta_key"):
                    return info["vanta_key"]
                return key  # It's an identifier for an externally managed model in _available_models
        logger.debug(
            f"Could not resolve alias '{name_or_alias}' (type: {expected_type}) to a known model key."
        )
        return None

    # Override _get_model_instance to be aware of how VantaRuntimeModelManager handles externally managed models
    def _get_model_instance(
        self, model_key_or_name: str, expected_type: str
    ) -> Any | None:
        # 1. Try resolving alias/name to a definitive key
        actual_key_to_use = self._resolve_alias_to_key(model_key_or_name, expected_type)
        if not actual_key_to_use:
            logger.warning(
                f"_get_model_instance: Could not resolve '{model_key_or_name}' (type: {expected_type})."
            )
            return None

        # 2. Check if this key corresponds to a model in self.models (already loaded or lazy-loadable by ModelManager)
        if actual_key_to_use in self.models:
            return super()._get_model_instance(
                actual_key_to_use, expected_type
            )  # Use parent's loading logic

        # 3. If not in self.models, it might be an externally managed model described in self._available_models
        #    (e.g., from LMStudio adapter, or a VantaCore registered service not directly added to self.models)
        available_info = self._available_models.get(actual_key_to_use)
        if available_info:
            source = available_info.get("source")
            if source == "vanta_lmstudio_adapter":
                adapter = self.vanta_core.get_component(
                    self.config.vanta_lmstudio_adapter_name
                )
                if adapter and hasattr(adapter, "get_model_instance_by_adapter_id"):
                    # We stored the adapter's internal ID for the model in "adapter_model_id"
                    adapter_model_id = available_info.get("adapter_model_id")
                    if adapter_model_id:
                        instance = adapter.get_model_instance_by_adapter_id(
                            adapter_model_id
                        )
                        if instance:
                            return instance
                        else:
                            logger.warning(
                                f"LMStudio adapter '{self.config.vanta_lmstudio_adapter_name}' failed to provide instance for adapter_id '{adapter_model_id}'."
                            )
                    else:
                        logger.warning(
                            f"No 'adapter_model_id' for LMStudio model key '{actual_key_to_use}'."
                        )
                else:
                    logger.warning(
                        f"LMStudio adapter '{self.config.vanta_lmstudio_adapter_name}' not found or lacks 'get_model_instance_by_adapter_id'."
                    )

            elif (
                source == "vanta_registry"
            ):  # This was handled by parent, but as a fallback
                reg_name = available_info.get(
                    "provider"
                )  # Should be "name_in_registry"
                if reg_name:
                    instance = self.vanta_core.get_component(reg_name)
                    if instance:
                        return instance  # Assume it matches expected_type

            # Add other external provider handling here if needed.

        logger.error(
            f"VantaRuntimeModelManager: Unable to get or load instance for model key/alias '{model_key_or_name}' (resolved to '{actual_key_to_use}', type '{expected_type}')."
        )
        return None

    def get_available_models_summary_list(self) -> list[dict[str, Any]]:
        """Returns a list of summaries for all discovered available models."""
        if not self._available_models:
            self._discover_and_register_all_models()  # Ensure populated
        summary = []
        for key, info in self._available_models.items():
            s = {
                "key_or_id": key,
                "name": info.get("name", key),
                "type": info.get("type"),
                "loaded": info.get("loaded", False),
                "path": info.get("path"),
                "config": info.get("config", {}),
                "source": info.get("source"),
                "provider_info": info.get(
                    "provider"
                ),  # e.g. "vanta_config", "lmstudio_adapter"
                "is_locally_managed_by_mm": key in self.models,
                "is_loaded_if_local": self.models.get(key, {}).get("loaded", False)
                if key in self.models
                else (
                    info.get("source") != "vanta_config_local"
                ),  # True if not local path model
            }
            summary.append(s)
        return summary

    def ensure_required_models_available(self, required_model_types: list[str]) -> bool:
        for model_type in required_model_types:
            if model_type == "embedding":
                if not self._get_model_instance(
                    self.default_embedding_model, "embedding"
                ):
                    logger.error("Required embedding model is not available.")
                    return False
            elif model_type == "reasoning":
                if not self._get_model_instance(
                    self.default_reasoning_model, "reasoning"
                ):
                    logger.error("Required reasoning model is not available.")
                    return False
            else:
                logger.warning(f"Unknown required model type: {model_type}")
                return False
        return True

    def get_status(self) -> Dict[str, Any]:
        parent_status = super().get_status()
        parent_status.update(
            {
                "vanta_available_models_count": len(self._available_models),
                "is_registered_with_vanta_core": self.register_to_vanta_core,
                "default_embedding_model_resolved": self.default_embedding_model,  # Shows resolved key
                "default_reasoning_model_resolved": self.default_reasoning_model,  # Shows resolved key
            }
        )
        return parent_status

    def set_activation_hook(
        self, hook_function: Callable[[], None]
    ) -> bool:  # Type hint adjusted
        # (Parent implementation is okay, but a VantaRuntime version might need to
        #  additionally iterate over models managed by adapters if they support hooks)
        logger.info("VantaRuntimeModelManager: Setting activation hook.")
        parent_success = super().set_activation_hook(hook_function)

        adapter_hook_count = 0
        # Example: If LMStudio adapter supports hooks on its models
        lmstudio_adapter = self.vanta_core.get_component(
            self.config.vanta_lmstudio_adapter_name
        )
        if lmstudio_adapter and hasattr(
            lmstudio_adapter, "set_model_activation_hook_for_all"
        ):
            try:
                lmstudio_adapter.set_model_activation_hook_for_all(hook_function)
                adapter_hook_count += 1  # Or count of models in adapter
                logger.info("Registered activation hook with LMStudio adapter.")
            except Exception as e:
                logger.warning(f"Failed to register hook with LMStudio adapter: {e}")

        return parent_success or adapter_hook_count > 0


# --- Example Usage (Adapted for VantaCore) ---
if __name__ == "__main__":
    main_logger_mm = logging.getLogger("ModelManagerExample")
    main_logger_mm.setLevel(logging.DEBUG)

    main_logger_mm.info("--- Starting Vanta Model Manager Example ---")
    vanta_sys = VantaCore()  # Get VantaCore instance

    # Example local models config (would typically come from a file or env)
    local_models_for_config = {
        "local_mini_lm": {
            "type": "embedding",
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "loader": "sentence-transformers",
        },
        "local_phi3_mini": {  # Requires transformers and PyTorch/TF. Can be large download.
            "type": "reasoning",
            "path": "microsoft/Phi-3-mini-4k-instruct",
            "loader": "transformers",
            "n_ctx": 2048,  # Example model-specific config
        },
        # Example for a GGUF model if llama-cpp-python is installed
        # "local_phi3_gguf": {
        #     "type": "reasoning", "path": "/path/to/your/phi-3-mini-4k-instruct.Q4_K_M.gguf",
        #     "loader": "llama.cpp", "n_gpu_layers": 0 # Or -1 for all
        # }
    }

    mm_config = ModelManagerConfig(
        log_level="DEBUG",
        default_embedding_model="local_mini_lm",  # Reference a model defined in local_models_config
        default_reasoning_model="local_phi3_mini",
        local_models_config=local_models_for_config,
        vanta_lmstudio_adapter_name="my_lmstudio_service",  # Name it expects in Vanta registry
    )

    # Example: Register a mock LMStudio adapter service with VantaCore
    class MockLMStudioAdapter:
        def get_available_models_info(self):
            return [
                {
                    "id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                    "name": "Mistral 7B Instruct GGUF (LMStudio)",
                    "type": "reasoning",
                }
            ]

        def get_model_instance_by_adapter_id(self, adapter_model_id):
            if "Mistral-7B" in adapter_model_id:
                logger.info(
                    f"[MockLMStudio] Providing mock instance for {adapter_model_id}"
                )

                # Return a dummy object that has a 'generate' or 'create_completion' method
                class MockLMStudioModel:
                    def create_completion(self, prompt, **kwargs):
                        return {
                            "choices": [
                                {"text": f"LMStudio mock response to: {prompt[:20]}"}
                            ]
                        }

                return MockLMStudioModel()
            return None

    vanta_sys.register_component("my_lmstudio_service", MockLMStudioAdapter())

    # Instantiate VantaRuntimeModelManager
    vrmm = VantaRuntimeModelManager(
        vanta_core=vanta_sys, config=mm_config, register_to_vanta_core=True
    )

    main_logger_mm.info("\n--- VantaRuntimeModelManager Status ---")
    main_logger_mm.info(json.dumps(vrmm.get_status(), indent=2, default=str))

    main_logger_mm.info("\n--- Testing Embedding Generation (default model) ---")
    test_text_for_embed = "This is a test sentence for VantaCore model manager."
    embedding = vrmm.get_embedding(test_text_for_embed)
    if embedding is not None:
        main_logger_mm.info(
            f"Embedding shape: {embedding.shape if isinstance(embedding, np.ndarray) else type(embedding)}"
        )
        main_logger_mm.info(
            f"Embedding (first 3): {embedding[:3] if isinstance(embedding, np.ndarray) else str(embedding)[:50]}"
        )
    else:
        main_logger_mm.error("Failed to generate embedding using default model.")

    main_logger_mm.info("\n--- Testing Text Generation (default model) ---")
    test_prompt_for_gen = "Explain the concept of lazy loading in Python."
    generated_text = vrmm.generate_text(test_prompt_for_gen, max_tokens=50)
    if generated_text:
        main_logger_mm.info(
            f"Generated text (default model '{vrmm.default_reasoning_model}'):\n{generated_text}"
        )
    else:
        main_logger_mm.error(
            f"Failed to generate text using default model '{vrmm.default_reasoning_model}'. Check model path and dependencies."
        )

    main_logger_mm.info(
        "\n--- Testing Text Generation (LMStudio discovered model, if any) ---"
    )
    lm_studio_model_key = None
    for (
        key,
        info,
    ) in vrmm.get_available_models_summary_list():  # Use new method to get summary
        # Add a type check for info before calling .get()
        if isinstance(info, dict):
            if info.get("source") == "vanta_lmstudio_adapter":
                lm_studio_model_key = key
                main_logger_mm.info(f"Found LM Studio model: {key}")
        elif isinstance(info, str):
            main_logger_mm.warning(f"Encountered string info for key {key}: {info}")
        else:
            main_logger_mm.warning(
                f"Encountered unexpected info type for key {key}: {type(info)}"
            )

    if lm_studio_model_key:
        main_logger_mm.info(
            f"Attempting generation with discovered LMStudio model: '{lm_studio_model_key}'"
        )
        lm_generated_text = vrmm.generate_text(
            test_prompt_for_gen, model_name_or_alias=lm_studio_model_key, max_tokens=30
        )
        if lm_generated_text:
            main_logger_mm.info(
                f"Generated text (LMStudio model '{lm_studio_model_key}'):\n{lm_generated_text}"
            )
        else:
            main_logger_mm.error(
                f"Failed to generate text using LMStudio model '{lm_studio_model_key}'."
            )
    else:
        main_logger_mm.info(
            "No LMStudio models discovered/configured for generation test."
        )

    main_logger_mm.info("\n--- Testing Similarity Search ---")
    docs_for_search = [
        "Lazy loading defers initialization of an object until the point at which it is needed.",
        "Eager loading initializes an object immediately upon creation.",
        "Python is a dynamically typed language.",
    ]
    search_results = vrmm.search_texts(test_prompt_for_gen, docs_for_search, top_k=2)
    main_logger_mm.info("Search Results:")
    for idx, text, score in search_results:
        main_logger_mm.info(f"  Score: {score:.4f} (Index: {idx}) - '{text[:50]}...'")

    main_logger_mm.info("\n--- Model Manager VantaCore Example Finished ---")
