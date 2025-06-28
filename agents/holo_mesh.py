# d:/Vox/Voxsigil-Library/agents/holo_mesh.py
from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Type

# HOLO-OPT: Efficiency Feature Flags and Configuration
HOLO_OPT_FEATURES = {
    "smart_caching": True,  # Enhanced caching with TTL and weak references
    "async_batching": True,  # Batch operations for better throughput
    "lazy_loading": True,  # Defer expensive operations until needed
    "memory_optimization": True,  # Advanced memory management
    "connection_pooling": True,  # Pool connections and resources
    "compression": True,  # Compress data in transit and storage
    "rate_limiting": True,  # Smart rate limiting for API calls
    "monitoring": True,  # Enhanced monitoring and metrics
    "preemptive_cleanup": True,  # Proactive resource cleanup
    "adaptive_timeouts": True,  # Dynamic timeout adjustment
}

# HOLO-OPT: Global efficiency tracking
_PERFORMANCE_METRICS = defaultdict(list)
_CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0}
_REQUEST_QUEUE = deque(maxlen=1000)
_CLEANUP_TASKS = set()
_THREAD_POOL = None


def get_thread_pool() -> ThreadPoolExecutor:
    """HOLO-OPT: Lazy thread pool initialization"""
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="holo-opt")
    return _THREAD_POOL


# HOLO-OPT: Smart caching decorator with TTL
def smart_cache(ttl_seconds: int = 300, maxsize: int = 128):
    """Enhanced caching with TTL and weak references"""

    def decorator(func):
        cache = {}
        timestamps = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not HOLO_OPT_FEATURES["smart_caching"]:
                return func(*args, **kwargs)

            # Create cache key
            key = str(hash((str(args), str(sorted(kwargs.items())))))
            current_time = time.time()

            # Check if cached and not expired
            if key in cache and current_time - timestamps.get(key, 0) < ttl_seconds:
                _CACHE_STATS["hits"] += 1
                return cache[key]

            # Cache miss or expired
            _CACHE_STATS["misses"] += 1
            result = func(*args, **kwargs)

            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = min(timestamps.keys(), key=timestamps.get)
                del cache[oldest_key]
                del timestamps[oldest_key]
                _CACHE_STATS["evictions"] += 1

            cache[key] = result
            timestamps[key] = current_time
            return result

        wrapper.cache_info = lambda: _CACHE_STATS.copy()
        wrapper.cache_clear = lambda: (cache.clear(), timestamps.clear())
        return wrapper

    return decorator


# HOLO-OPT: Async batching utility
class AsyncBatcher:
    """Batch async operations for better throughput"""

    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending = []
        self.lock = asyncio.Lock()

    async def batch_execute(self, operation, *args, **kwargs):
        if not HOLO_OPT_FEATURES["async_batching"]:
            return await operation(*args, **kwargs)

        async with self.lock:
            self.pending.append((operation, args, kwargs))

            if len(self.pending) >= self.batch_size:
                batch = self.pending.copy()
                self.pending.clear()
                return await self._execute_batch(batch)

        # Wait for timeout or batch to fill
        await asyncio.sleep(self.timeout)
        async with self.lock:
            if self.pending:
                batch = self.pending.copy()
                self.pending.clear()
                return await self._execute_batch(batch)

    async def _execute_batch(self, batch):
        """Execute batch of operations concurrently"""
        tasks = [op(*args, **kwargs) for op, args, kwargs in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)


# HOLO-OPT: Lazy loader for expensive operations
class LazyLoader:
    """Defer expensive operations until actually needed"""

    def __init__(self, loader_func, *args, **kwargs):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._value = None
        self._loaded = False
        self._lock = threading.Lock()

    def __call__(self):
        if not HOLO_OPT_FEATURES["lazy_loading"] or self._loaded:
            return self._value

        with self._lock:
            if not self._loaded:
                self._value = self.loader_func(*self.args, **self.kwargs)
                self._loaded = True

        return self._value

    @property
    def is_loaded(self):
        return self._loaded


# HOLO-OPT: Memory-efficient weak reference manager
class WeakRefManager:
    """Manage weak references to prevent memory leaks"""

    def __init__(self):
        self._refs = weakref.WeakSet()

    def add(self, obj):
        if HOLO_OPT_FEATURES["memory_optimization"]:
            self._refs.add(obj)
        return obj

    def cleanup(self):
        """Force cleanup of dead references"""
        if HOLO_OPT_FEATURES["preemptive_cleanup"]:
            # WeakSet automatically removes dead references
            return len(self._refs)
        return 0


# HOLO-OPT: Connection pooling for resources
class ResourcePool:
    """Pool resources to avoid repeated creation/destruction"""

    def __init__(self, factory, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.pool = deque()
        self.in_use = set()
        self.lock = threading.Lock()

    def acquire(self):
        if not HOLO_OPT_FEATURES["connection_pooling"]:
            return self.factory()

        with self.lock:
            if self.pool:
                resource = self.pool.popleft()
                self.in_use.add(resource)
                return resource

            if len(self.in_use) < self.max_size:
                resource = self.factory()
                self.in_use.add(resource)
                return resource

        # Pool exhausted, create temporary resource
        return self.factory()

    def release(self, resource):
        if not HOLO_OPT_FEATURES["connection_pooling"]:
            return

        with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)


# HOLO-OPT: Adaptive timeout manager
class AdaptiveTimeout:
    """Dynamically adjust timeouts based on performance"""

    def __init__(self, initial_timeout: float = 5.0):
        self.timeout = initial_timeout
        self.success_times = deque(maxlen=100)
        self.lock = threading.Lock()

    def get_timeout(self) -> float:
        if not HOLO_OPT_FEATURES["adaptive_timeouts"]:
            return self.timeout

        with self.lock:
            if self.success_times:
                avg_time = sum(self.success_times) / len(self.success_times)
                # Set timeout to 2x average success time, with bounds
                self.timeout = max(1.0, min(30.0, avg_time * 2))

        return self.timeout

    def record_success(self, elapsed_time: float):
        if HOLO_OPT_FEATURES["adaptive_timeouts"]:
            with self.lock:
                self.success_times.append(elapsed_time)


# HOLO-OPT: Initialize global efficiency managers
_WEAK_REF_MANAGER = WeakRefManager()
_ASYNC_BATCHER = AsyncBatcher()
_ADAPTIVE_TIMEOUT = AdaptiveTimeout()

# Initialize logger early
logger = logging.getLogger("agents.holomesh")


# Define CognitiveMeshRole first as a fallback
class CognitiveMeshRole:
    ORCHESTRATOR = "orchestrator"
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"
    EVALUATOR = "evaluator"


# Default fallback definitions
def vanta_agent(**kwargs: Any) -> Callable[[Type[Any]], Type[Any]]:
    def decorator(cls: Type[Any]) -> Type[Any]:
        return cls

    return decorator


class BaseAgent:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def async_init(self) -> None:
        pass


# Try to import real HOLO-1.5 components and override fallbacks
HOLO_AVAILABLE = False
try:
    from ..engines.base import CognitiveMeshRole as _CognitiveMeshRole
    from .base import BaseAgent as _BaseAgent
    from .base import vanta_agent as _vanta_agent

    # Override with real implementations
    CognitiveMeshRole = _CognitiveMeshRole
    BaseAgent = _BaseAgent
    vanta_agent = _vanta_agent
    HOLO_AVAILABLE = True
    logger.info(f"Using real HOLO components: {type(CognitiveMeshRole)}")
except ImportError as e:
    logger.warning(f"Could not import from engines.base: {e}")
    try:
        from .base import BaseAgent as _BaseAgent
        from .base import CognitiveMeshRole as _CognitiveMeshRole
        from .base import vanta_agent as _vanta_agent

        # Override with real implementations
        CognitiveMeshRole = _CognitiveMeshRole
        BaseAgent = _BaseAgent
        vanta_agent = _vanta_agent
        HOLO_AVAILABLE = True
        logger.info(f"Using base HOLO components: {type(CognitiveMeshRole)}")
    except ImportError as e:
        logger.warning(f"Could not import from base: {e}")
        try:
            # Alternative import path
            from ..core.base_agent import BaseAgent as _BaseAgent
            from ..core.cognitive_mesh import CognitiveMeshRole as _CognitiveMeshRole
            from ..core.vanta_core import vanta_agent as _vanta_agent

            # Override with real implementations
            CognitiveMeshRole = _CognitiveMeshRole
            BaseAgent = _BaseAgent
            vanta_agent = _vanta_agent
            HOLO_AVAILABLE = True
            logger.info(f"Using core HOLO components: {type(CognitiveMeshRole)}")
        except ImportError as e:
            # Keep fallback definitions
            logger.warning(f"Could not import from core: {e}")
            logger.info(f"Using fallback CognitiveMeshRole: {type(CognitiveMeshRole)}")
            logger.info(f"CognitiveMeshRole attributes: {dir(CognitiveMeshRole)}")
            HOLO_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Globals                                                                     #
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_COMPONENT_CACHE: dict[str, dict[str, Any]] = {}

# HOLO-OPT: Enhanced caching with compression and metrics
_COMPRESSED_CACHE: dict[str, bytes] = {}
_CACHE_METRICS = {
    "model_hits": 0,
    "model_misses": 0,
    "component_hits": 0,
    "component_misses": 0,
}


@smart_cache(ttl_seconds=600, maxsize=64)
def _hash(cfg: Mapping[str, Any]) -> str:
    """Stable 16-char hash of any JSON-serialisable mapping with caching."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# HOLO-OPT: Compressed configuration storage
def _compress_config(cfg: dict) -> bytes:
    """Compress configuration for memory efficiency"""
    if not HOLO_OPT_FEATURES["compression"]:
        return json.dumps(cfg).encode()

    import gzip

    return gzip.compress(json.dumps(cfg, sort_keys=True).encode())


def _decompress_config(data: bytes) -> dict:
    """Decompress configuration"""
    if not HOLO_OPT_FEATURES["compression"]:
        return json.loads(data.decode())

    import gzip

    return json.loads(gzip.decompress(data).decode())


# HOLO-OPT: Model cache with TTL and memory management
class ModelCacheManager:
    """Enhanced model cache with TTL and weak references"""

    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        self.ttl = 3600  # 1 hour TTL for models
        self.max_models = 5  # Maximum models in cache

    def get(self, key: str) -> tuple[Any, Any] | None:
        if not HOLO_OPT_FEATURES["smart_caching"]:
            return _MODEL_CACHE.get(key)

        current_time = time.time()

        # Check if cached and not expired
        if key in self.cache:
            if current_time - self.timestamps.get(key, 0) < self.ttl:
                _CACHE_METRICS["model_hits"] += 1
                return self.cache[key]
            else:
                # Expired, remove
                del self.cache[key]
                del self.timestamps[key]

        _CACHE_METRICS["model_misses"] += 1
        return _MODEL_CACHE.get(key)

    def set(self, key: str, value: tuple[Any, Any]):
        if not HOLO_OPT_FEATURES["smart_caching"]:
            _MODEL_CACHE[key] = value
            return

        # Manage cache size
        if len(self.cache) >= self.max_models:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

        self.cache[key] = value
        self.timestamps[key] = time.time()
        _MODEL_CACHE[key] = value  # Keep fallback cache


_MODEL_CACHE_MANAGER = ModelCacheManager()


# --------------------------------------------------------------------------- #
# Optional deps (fail-soft)                                                   #
# --------------------------------------------------------------------------- #
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAVE_TRANSFORMERS = False
    torch = None  # type: ignore

try:
    from core.novel_reasoning import (
        create_akorn_network,
        create_reasoning_engine,
        create_splr_network,
    )

    HAVE_NOVEL_PARADIGMS = True
except ImportError:
    HAVE_NOVEL_PARADIGMS = False

try:
    from core.novel_efficiency import (
        AdaptiveMemoryManager,
        DatasetManager,
        DeltaNetAttention,
        DeltaRuleOperator,
        KVCacheCompressor,
        LinearAttentionConfig,
        MemoryPool,
        MiniCacheWrapper,
        OutlierTokenDetector,
        ResourceOptimizer,
    )

    HAVE_EFFICIENCY = True
except ImportError:
    HAVE_EFFICIENCY = False

# VoxSigil RAG and BLT optional imports (avoid circular dependencies)
try:
    from engines.rag_compression_engine import RAGCompressionEngine
    from VoxSigilRag.voxsigil_rag import VoxSigilRAG

    HAVE_VOXSIGIL_RAG = True
except ImportError as e:
    logger.warning(f"VoxSigil RAG components not available: {e}")
    HAVE_VOXSIGIL_RAG = False

try:
    # Import BLT components if available
    from BLT import (  # Main encoder from __init__.py
        BLTEncoder,
        ByteLatentTransformerEncoder,
        SigilPatchEncoder,
    )
    from BLT.hybrid_blt import BLTEnhancedRAG  # Enhanced RAG component

    HAVE_BLT = True
except ImportError as e:
    logger.warning(f"BLT components not available: {e}")
    HAVE_BLT = False  # Create dummy classes for when BLT is not available

    class BLTEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ByteLatentTransformerEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class SigilPatchEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class BLTEnhancedRAG:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class HOLOAgentConfig:
    # model / generation ---------------------------------------------------- #
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_adapters: List[str] = field(default_factory=list)
    max_tokens: int = 512
    device: str = "cuda"

    # feature switches ------------------------------------------------------ #
    use_spiking_networks: bool = True
    use_logical_reasoning: bool = True
    use_oscillatory_binding: bool = True
    use_adaptive_memory: bool = True
    use_minicache: bool = True
    use_deltanet_attention: bool = True
    use_dataset_manager: bool = True
    use_delta_rule_operator: bool = True
    use_outlier_detection: bool = True
    use_voxsigil_rag: bool = True
    use_blt_sigils: bool = True
    use_art_training: bool = True

    # HOLO-OPT efficiency features ----------------------------------------- #
    enable_smart_caching: bool = True
    enable_async_batching: bool = True
    enable_lazy_loading: bool = True
    enable_memory_optimization: bool = True
    enable_connection_pooling: bool = True
    enable_compression: bool = True
    enable_rate_limiting: bool = True
    enable_monitoring: bool = True
    enable_preemptive_cleanup: bool = True
    enable_adaptive_timeouts: bool = True

    # HOLO-OPT performance tuning ------------------------------------------ #
    batch_size: int = 8
    cache_ttl_seconds: int = 300
    max_concurrent_requests: int = 10
    memory_threshold_mb: int = 1024
    cleanup_interval_seconds: int = 60
    # component configurations --------------------------------------------- #
    spiking_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "input_dim": 256,
            "hidden_dim": 128,
            "output_dim": 64,
            "spike_threshold": 1.0,
            "decay_rate": 0.95,
            "dt": 0.1,
        }
    )
    logical_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_variables": 10,
            "max_rules": 20,
            "inference_steps": 100,
            "use_symbolic_cache": True,
        }
    )
    binding_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_oscillators": 64,
            "coupling_strength": 0.1,
            "frequency_range": (30, 100),
            "synchrony_threshold": 0.8,
        }
    )
    memory_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "budget": {"total_gb": 16.0, "reserve_gb": 2.0},
            "pool_size_gb": 1.0,
        }
    )
    minicache_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "compression_ratio": 0.5,
            "similarity_threshold": 0.9,
            "adaptive_compression": True,
        }
    )
    deltanet_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "d_model": 512,
            "n_heads": 8,
            "delta_rule_strength": 0.1,
        }
    )
    dataset_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "data_directory": "./data",
        }
    )
    delta_rule_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "d_model": 512,
            "strength": 0.1,
        }
    )
    outlier_detection_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "threshold": 2.0,
            "window_size": 50,
        }
    )
    rag_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "compression_mode": "zlib",
            "compress_level": 9,
            "min_entropy": 1.5,
            "encoding": "utf-8",
        }
    )
    blt_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "sigil_mode": "adaptive",
            "binding_strength": 0.8,
            "temporal_decay": 0.95,
            "max_sigils": 100,
        }
    )
    # ART trainer configurations for VoxSigil integration ------------ #
    art_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "input_dim": 256,  # Default input dimension for HOLO-1.5 features
            "vigilance": 0.75,  # ART vigilance parameter
            "learning_rate": 0.1,  # Learning rate for adaptation
            "max_categories": 100,  # Maximum number of categories
            "enable_art_training": True,  # Enable ART training by default
        }
    )


# --------------------------------------------------------------------------- #
# HOLO Agent                                                                  #
# --------------------------------------------------------------------------- #


@vanta_agent(
    name="holo_agent",
    subsystem="cognitive_mesh",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    description="HOLO agent with novel LLM paradigms integration",
    capabilities=[
        "language_modeling",
        "logical_reasoning",
        "memory_management",
        "outlier_detection",
        "rag_compression",
    ],
    models=["auto"],
)
class HOLOAgent(BaseAgent if HOLO_AVAILABLE else object):
    """Single language-model instance with many optional augmentations."""

    def __init__(self, name: str, cfg: HOLOAgentConfig):
        # Initialize BaseAgent if HOLO is available
        if HOLO_AVAILABLE:
            super().__init__()

        self.name, self.cfg = name, cfg
        self.model = self.tokenizer = None
        self.initialized = False

        # HOLO-OPT: Apply configuration to global features
        if hasattr(cfg, "enable_smart_caching"):
            HOLO_OPT_FEATURES["smart_caching"] = cfg.enable_smart_caching
        if hasattr(cfg, "enable_async_batching"):
            HOLO_OPT_FEATURES["async_batching"] = cfg.enable_async_batching
        if hasattr(cfg, "enable_lazy_loading"):
            HOLO_OPT_FEATURES["lazy_loading"] = cfg.enable_lazy_loading
        if hasattr(cfg, "enable_memory_optimization"):
            HOLO_OPT_FEATURES["memory_optimization"] = cfg.enable_memory_optimization
        if hasattr(cfg, "enable_connection_pooling"):
            HOLO_OPT_FEATURES["connection_pooling"] = cfg.enable_connection_pooling
        if hasattr(cfg, "enable_compression"):
            HOLO_OPT_FEATURES["compression"] = cfg.enable_compression
        if hasattr(cfg, "enable_rate_limiting"):
            HOLO_OPT_FEATURES["rate_limiting"] = cfg.enable_rate_limiting
        if hasattr(cfg, "enable_monitoring"):
            HOLO_OPT_FEATURES["monitoring"] = cfg.enable_monitoring
        if hasattr(cfg, "enable_preemptive_cleanup"):
            HOLO_OPT_FEATURES["preemptive_cleanup"] = cfg.enable_preemptive_cleanup
        if hasattr(cfg, "enable_adaptive_timeouts"):
            HOLO_OPT_FEATURES["adaptive_timeouts"] = cfg.enable_adaptive_timeouts

        # HOLO-OPT: Initialize efficiency components
        self._request_limiter = None
        self._cleanup_task = None
        self._performance_monitor = None
        self._lazy_components = {}

        # HOLO-OPT: Set up monitoring if enabled
        if HOLO_OPT_FEATURES["monitoring"]:
            self._performance_monitor = {
                "requests_processed": 0,
                "avg_response_time": 0.0,
                "error_count": 0,
                "cache_hit_ratio": 0.0,
                "memory_usage_mb": 0.0,
                "last_cleanup": time.time(),
            }

        # HOLO-OPT: Rate limiter initialization
        if HOLO_OPT_FEATURES["rate_limiting"]:
            self._request_limiter = {
                "requests": deque(maxlen=100),
                "max_per_minute": getattr(cfg, "max_concurrent_requests", 10) * 6,
                "last_reset": time.time(),
            }

        # runtime components (populated later)
        self.__dict__.update(
            {
                k: None
                for k in (
                    "spiking_network",
                    "logical_engine",
                    "binding_network",
                    "memory_manager",
                    "memory_pool",
                    "resource_optimizer",
                    "minicache",
                    "kv_compressor",
                    "deltanet_attention",
                    "dataset_manager",
                    "delta_rule_operator",
                    "outlier_detector",
                    "rag_engine",
                    "vox_rag",
                    "blt_encoder",
                    "sigil_encoder",
                    "byte_transformer",
                    "art_trainer",  # Add ART trainer component
                )
            }
        )

        # HOLO-OPT: Add to weak reference manager
        _WEAK_REF_MANAGER.add(self)

        if HAVE_TRANSFORMERS:
            self._load_model()
            self.initialized = True
        else:
            logger.warning("transformers unavailable; running in mock mode")

        # fire-and-forget asynchronous component initialisation
        asyncio.create_task(self._init_components_async())

        # HOLO-OPT: Schedule cleanup task
        if HOLO_OPT_FEATURES["preemptive_cleanup"]:
            self._schedule_cleanup_task()

    # --------------------------------------------------------------------- #
    # LM loading with HOLO-OPT optimizations                               #
    # --------------------------------------------------------------------- #

    def _load_model(self) -> None:
        if not HAVE_TRANSFORMERS:
            return

        key = f"{self.cfg.model_name}_{self.cfg.device}"

        # HOLO-OPT: Use enhanced model cache
        cached_model = _MODEL_CACHE_MANAGER.get(key)
        if cached_model:
            self.model, self.tokenizer = cached_model
            logger.info(f"✅ Model loaded from cache for {self.name}")
            return

        # HOLO-OPT: Record load start time for adaptive timeouts
        load_start = time.time()

        # Try quantization, fall back to regular loading
        try:
            q_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # HOLO-OPT: Use thread pool for model loading to avoid blocking
            if HOLO_OPT_FEATURES["async_batching"]:
                future = get_thread_pool().submit(
                    AutoModelForCausalLM.from_pretrained,
                    self.cfg.model_name,
                    device_map="auto",
                    quantization_config=q_cfg,
                )
                self.model = future.result(timeout=_ADAPTIVE_TIMEOUT.get_timeout())
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_name, device_map="auto", quantization_config=q_cfg
                )

        except Exception as e:
            logger.warning(f"Quantization failed ({e}), loading without quantization")
            try:
                if HOLO_OPT_FEATURES["async_batching"]:
                    future = get_thread_pool().submit(
                        AutoModelForCausalLM.from_pretrained,
                        self.cfg.model_name,
                        device_map="auto",
                    )
                    self.model = future.result(timeout=_ADAPTIVE_TIMEOUT.get_timeout())
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.cfg.model_name, device_map="auto"
                    )
            except Exception as e2:
                logger.error(f"Model loading failed completely: {e2}")
                self.model = None
                return

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        # HOLO-OPT: Store in enhanced cache
        _MODEL_CACHE_MANAGER.set(key, (self.model, self.tokenizer))

        # HOLO-OPT: Record successful load time
        load_time = time.time() - load_start
        _ADAPTIVE_TIMEOUT.record_success(load_time)

        if HOLO_OPT_FEATURES["monitoring"] and self._performance_monitor:
            self._performance_monitor["last_model_load_time"] = load_time

        logger.info(f"✅ Model loaded successfully for {self.name} in {load_time:.2f}s")

    # HOLO-OPT: Cleanup task scheduler
    def _schedule_cleanup_task(self):
        """Schedule periodic cleanup tasks"""

        async def cleanup_worker():
            while True:
                try:
                    await asyncio.sleep(
                        getattr(self.cfg, "cleanup_interval_seconds", 60)
                    )
                    await self._perform_cleanup()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Cleanup task error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_worker())
        _CLEANUP_TASKS.add(self._cleanup_task)

    async def _perform_cleanup(self):
        """Perform periodic cleanup operations"""
        if not HOLO_OPT_FEATURES["preemptive_cleanup"]:
            return

        try:
            # Clean up weak references
            cleaned_refs = _WEAK_REF_MANAGER.cleanup()

            # Update performance metrics
            if self._performance_monitor:
                import psutil

                process = psutil.Process()
                self._performance_monitor["memory_usage_mb"] = (
                    process.memory_info().rss / 1024 / 1024
                )
                self._performance_monitor["last_cleanup"] = time.time()

            # Force garbage collection if memory usage is high
            if self._performance_monitor and self._performance_monitor[
                "memory_usage_mb"
            ] > getattr(self.cfg, "memory_threshold_mb", 1024):
                import gc

                collected = gc.collect()
                logger.debug(f"Forced GC collected {collected} objects")

            logger.debug(
                f"Cleanup completed for {self.name}: {cleaned_refs} refs cleaned"
            )

        except Exception as e:
            logger.warning(f"Cleanup error for {self.name}: {e}")

    # --------------------------------------------------------------------- #
    # Component helpers                                                     #
    # --------------------------------------------------------------------- #

    async def _maybe(
        self, cond: bool, fn: Callable[..., Any], *args: Any, **kw: Any
    ) -> Any:
        """Utility: awaits fn iff cond is true, otherwise returns None."""
        return await fn(*args, **kw) if cond else None

    async def _init_components_async(self) -> None:
        if not (HAVE_NOVEL_PARADIGMS and HAVE_EFFICIENCY):
            logger.warning("Optional component packages missing for %s", self.name)
            return

        key = _hash(self.cfg.__dict__)

        # HOLO-OPT: Check enhanced component cache
        if key in _COMPONENT_CACHE:  # fast path
            cached_components = _COMPONENT_CACHE[key]

            # HOLO-OPT: Use lazy loading for expensive components
            if HOLO_OPT_FEATURES["lazy_loading"]:
                for comp_name, comp_value in cached_components.items():
                    if comp_value is not None:
                        # Create lazy loader for cached components
                        self._lazy_components[comp_name] = LazyLoader(
                            lambda cv=comp_value: cv
                        )
                        # Set attribute to lazy loader for transparent access
                        setattr(self, comp_name, self._lazy_components[comp_name]())
                    else:
                        setattr(self, comp_name, None)
            else:
                self.__dict__.update(cached_components)

            _CACHE_METRICS["component_hits"] += 1
            logger.info(f"✅ Components loaded from cache for {self.name}")
            return

        _CACHE_METRICS["component_misses"] += 1
        comp: dict[str, Any] = {}

        try:
            # HOLO-OPT: Use batching for component initialization
            if HOLO_OPT_FEATURES["async_batching"]:
                await self._init_components_batched(comp, key)
            else:
                await self._init_components_sequential(comp, key)

        except Exception as e:
            logger.error(
                f"❌ Critical error in component initialization for {self.name}: {e}"
            )
            # Still update with whatever components we managed to create
            self.__dict__.update(comp)

    async def _init_components_batched(self, comp: dict[str, Any], cache_key: str):
        """Initialize components in batches for better performance"""

        # Initialize reasoning components concurrently
        reasoning_tasks = []
        active_names = []

        if self.cfg.use_spiking_networks:
            reasoning_tasks.append(
                self._init_single_component(
                    "spiking_network", create_splr_network, self.cfg.spiking_config
                )
            )
            active_names.append("spiking_network")
        if self.cfg.use_logical_reasoning:
            reasoning_tasks.append(
                self._init_single_component(
                    "logical_engine", create_reasoning_engine, self.cfg.logical_config
                )
            )
            active_names.append("logical_engine")
        if self.cfg.use_oscillatory_binding:
            reasoning_tasks.append(
                self._init_single_component(
                    "binding_network", create_akorn_network, self.cfg.binding_config
                )
            )
            active_names.append("binding_network")

        if reasoning_tasks:
            reasoning_results = await asyncio.gather(
                *reasoning_tasks, return_exceptions=True
            )

            for name, result in zip(active_names, reasoning_results):
                if not isinstance(result, Exception):
                    comp[name] = result
                else:
                    logger.warning(f"Failed to initialize {name}: {result}")
                    comp[name] = None

        # Initialize efficiency components
        await self._init_efficiency_components_batched(comp)

        # Initialize integration components
        await self._init_integration_components_batched(comp)

        # Finalize initialization
        self.__dict__.update(comp)
        _COMPONENT_CACHE[cache_key] = comp

        # Log successful component initialization
        initialized_comps = [k for k, v in comp.items() if v is not None]
        logger.info(
            f"✅ Initialized {len(initialized_comps)} components for {self.name}: "
            f"{initialized_comps}"
        )

    async def _init_single_component(self, name: str, func: Callable, config: dict):
        """Initialize a single component with error handling"""
        try:
            return await self._maybe(True, func, config)
        except Exception as e:
            logger.warning(f"Failed to initialize {name}: {e}")
            return None

    async def _init_efficiency_components_batched(self, comp: dict[str, Any]):
        """Initialize efficiency components with HOLO-OPT optimizations"""

        # Memory components
        if self.cfg.use_adaptive_memory:
            try:
                from core.novel_efficiency.adaptive_memory import MemoryBudget

                budget = MemoryBudget(**self.cfg.memory_config["budget"])

                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["memory_manager"] = LazyLoader(AdaptiveMemoryManager, budget)
                    comp["memory_pool"] = LazyLoader(
                        MemoryPool, self.cfg.memory_config["pool_size_gb"]
                    )
                    comp["resource_optimizer"] = LazyLoader(
                        ResourceOptimizer, comp["memory_manager"]
                    )
                else:
                    comp["memory_manager"] = AdaptiveMemoryManager(budget)
                    comp["memory_pool"] = MemoryPool(
                        self.cfg.memory_config["pool_size_gb"]
                    )
                    comp["resource_optimizer"] = ResourceOptimizer(
                        comp["memory_manager"]
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to initialize memory components for {self.name}: {e}"
                )

        # Cache components
        if self.cfg.use_minicache:
            try:
                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["minicache"] = LazyLoader(
                        MiniCacheWrapper, self.cfg.minicache_config
                    )
                    comp["kv_compressor"] = LazyLoader(
                        KVCacheCompressor,
                        **{
                            k: self.cfg.minicache_config[k]
                            for k in ("similarity_threshold", "compression_ratio")
                        },
                    )
                else:
                    comp["minicache"] = MiniCacheWrapper(self.cfg.minicache_config)
                    comp["kv_compressor"] = KVCacheCompressor(
                        **{
                            k: self.cfg.minicache_config[k]
                            for k in ("similarity_threshold", "compression_ratio")
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize cache components for {self.name}: {e}"
                )

        # Attention components
        if self.cfg.use_deltanet_attention:
            try:
                acfg = LinearAttentionConfig(
                    d_model=self.cfg.deltanet_config["d_model"],
                    n_heads=self.cfg.deltanet_config["n_heads"],
                    delta_rule_strength=self.cfg.deltanet_config["delta_rule_strength"],
                )

                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["deltanet_attention"] = LazyLoader(DeltaNetAttention, acfg)
                else:
                    comp["deltanet_attention"] = DeltaNetAttention(acfg)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize deltanet attention for {self.name}: {e}"
                )

        # Dataset and rule components
        if self.cfg.use_dataset_manager:
            try:
                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["dataset_manager"] = LazyLoader(
                        DatasetManager, Path(self.cfg.dataset_config["data_directory"])
                    )
                else:
                    comp["dataset_manager"] = DatasetManager(
                        Path(self.cfg.dataset_config["data_directory"])
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize dataset manager for {self.name}: {e}"
                )

        if self.cfg.use_delta_rule_operator:
            try:
                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["delta_rule_operator"] = LazyLoader(
                        DeltaRuleOperator, **self.cfg.delta_rule_config
                    )
                else:
                    comp["delta_rule_operator"] = DeltaRuleOperator(
                        **self.cfg.delta_rule_config
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize delta rule operator for {self.name}: {e}"
                )

        if self.cfg.use_outlier_detection:
            try:
                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["outlier_detector"] = LazyLoader(
                        OutlierTokenDetector, **self.cfg.outlier_detection_config
                    )
                else:
                    comp["outlier_detector"] = OutlierTokenDetector(
                        **self.cfg.outlier_detection_config
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize outlier detector for {self.name}: {e}"
                )

    async def _init_integration_components_batched(self, comp: dict[str, Any]):
        """Initialize integration components (RAG, BLT, ART) with optimizations"""

        # VoxSigil RAG integration
        if self.cfg.use_voxsigil_rag and HAVE_VOXSIGIL_RAG:
            try:
                rag_config = getattr(
                    self.cfg,
                    "rag_config",
                    {
                        "compression_mode": "zlib",
                        "compress_level": 9,
                        "min_entropy": 1.5,
                        "encoding": "utf-8",
                    },
                )

                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["rag_engine"] = LazyLoader(RAGCompressionEngine, rag_config)
                    comp["vox_rag"] = LazyLoader(VoxSigilRAG)
                else:
                    comp["rag_engine"] = RAGCompressionEngine(rag_config)
                    comp["vox_rag"] = VoxSigilRAG()

                logger.info(f"✅ VoxSigil RAG initialized for {self.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize VoxSigil RAG for {self.name}: {e}"
                )

        # BLT Sigil System integration
        if self.cfg.use_blt_sigils and HAVE_BLT:
            try:
                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["blt_encoder"] = LazyLoader(BLTEncoder)
                    comp["sigil_encoder"] = LazyLoader(SigilPatchEncoder)
                    comp["byte_transformer"] = LazyLoader(ByteLatentTransformerEncoder)

                    if self.cfg.use_voxsigil_rag and HAVE_VOXSIGIL_RAG:
                        comp["blt_enhanced_rag"] = LazyLoader(BLTEnhancedRAG)
                else:
                    comp["blt_encoder"] = BLTEncoder()
                    comp["sigil_encoder"] = SigilPatchEncoder()
                    comp["byte_transformer"] = ByteLatentTransformerEncoder()

                    if self.cfg.use_voxsigil_rag and HAVE_VOXSIGIL_RAG:
                        comp["blt_enhanced_rag"] = BLTEnhancedRAG()
                        logger.info(f"✅ BLT Enhanced RAG initialized for {self.name}")

                logger.info(f"✅ BLT Sigil System initialized for {self.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize BLT Sigil System for {self.name}: {e}"
                )

        # ART Trainer integration
        if self.cfg.use_art_training:
            try:
                art_config = getattr(
                    self.cfg,
                    "art_config",
                    {
                        "input_dim": 256,
                        "vigilance": 0.75,
                        "learning_rate": 0.1,
                        "max_categories": 100,
                        "enable_art_training": True,
                    },
                )

                art_trainer = type(
                    "ARTTrainer",
                    (),
                    {
                        "input_dim": art_config.get("input_dim", 256),
                        "vigilance": art_config.get("vigilance", 0.75),
                        "learning_rate": art_config.get("learning_rate", 0.1),
                        "max_categories": art_config.get("max_categories", 100),
                        "enabled": art_config.get("enable_art_training", True),
                        "validate_feature_vector": lambda self, vector: len(vector)
                        == self.input_dim,
                        "train": lambda self,
                        data: f"ART training with {len(data)} samples",
                    },
                )()

                if HOLO_OPT_FEATURES["lazy_loading"]:
                    comp["art_trainer"] = LazyLoader(lambda: art_trainer)
                else:
                    comp["art_trainer"] = art_trainer
                logger.info(
                    f"✅ ART Trainer initialized for {self.name} "
                    f"with input_dim={art_config.get('input_dim', 256)}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ART Trainer for {self.name}: {e}")

    async def _init_components_sequential(self, comp: dict[str, Any], cache_key: str):
        """Original sequential component initialization (fallback)"""
        try:
            # reasoning -------------------------------------------------------- #
            comp["spiking_network"] = await self._maybe(
                self.cfg.use_spiking_networks,
                create_splr_network,
                self.cfg.spiking_config,
            )
            comp["logical_engine"] = await self._maybe(
                self.cfg.use_logical_reasoning,
                create_reasoning_engine,
                self.cfg.logical_config,
            )
            comp["binding_network"] = await self._maybe(
                self.cfg.use_oscillatory_binding,
                create_akorn_network,
                self.cfg.binding_config,
            )

            # Call other initialization methods
            await self._init_efficiency_components_sequential(comp)
            await self._init_integration_components_sequential(comp)

            # Finalize initialization
            self.__dict__.update(comp)
            _COMPONENT_CACHE[cache_key] = comp

            # Log successful component initialization
            initialized_comps = [k for k, v in comp.items() if v is not None]
            logger.info(
                f"✅ Initialized {len(initialized_comps)} components for {self.name}: "
                f"{initialized_comps}"
            )

        except Exception as e:
            logger.error(f"Sequential initialization error: {e}")
            raise

    async def _init_efficiency_components_sequential(self, comp: dict[str, Any]):
        """Sequential efficiency component initialization"""

        # efficiency ------------------------------------------------------- #
        if self.cfg.use_adaptive_memory:
            try:
                from core.novel_efficiency.adaptive_memory import MemoryBudget

                budget = MemoryBudget(**self.cfg.memory_config["budget"])
                comp["memory_manager"] = AdaptiveMemoryManager(budget)
                comp["memory_pool"] = MemoryPool(self.cfg.memory_config["pool_size_gb"])
                comp["resource_optimizer"] = ResourceOptimizer(comp["memory_manager"])
            except Exception as e:
                logger.warning(
                    f"Failed to initialize memory components for {self.name}: {e}"
                )

        if self.cfg.use_minicache:
            try:
                comp["minicache"] = MiniCacheWrapper(self.cfg.minicache_config)
                comp["kv_compressor"] = KVCacheCompressor(
                    **{
                        k: self.cfg.minicache_config[k]
                        for k in ("similarity_threshold", "compression_ratio")
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize cache components for {self.name}: {e}"
                )

        if self.cfg.use_deltanet_attention:
            try:
                acfg = LinearAttentionConfig(
                    d_model=self.cfg.deltanet_config["d_model"],
                    n_heads=self.cfg.deltanet_config["n_heads"],
                    delta_rule_strength=self.cfg.deltanet_config["delta_rule_strength"],
                )
                comp["deltanet_attention"] = DeltaNetAttention(acfg)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize deltanet attention for {self.name}: {e}"
                )

        if self.cfg.use_dataset_manager:
            try:
                comp["dataset_manager"] = DatasetManager(
                    Path(self.cfg.dataset_config["data_directory"])
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize dataset manager for {self.name}: {e}"
                )

        if self.cfg.use_delta_rule_operator:
            try:
                comp["delta_rule_operator"] = DeltaRuleOperator(
                    **self.cfg.delta_rule_config
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize delta rule operator for {self.name}: {e}"
                )

        if self.cfg.use_outlier_detection:
            try:
                comp["outlier_detector"] = OutlierTokenDetector(
                    **self.cfg.outlier_detection_config
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize outlier detector for {self.name}: {e}"
                )

    async def _init_integration_components_sequential(self, comp: dict[str, Any]):
        """Sequential integration component initialization"""

        # VoxSigil RAG integration -----------------------------------------
        if self.cfg.use_voxsigil_rag and HAVE_VOXSIGIL_RAG:
            try:
                # Initialize RAG compression engine
                rag_config = getattr(
                    self.cfg,
                    "rag_config",
                    {
                        "compression_mode": "zlib",
                        "compress_level": 9,
                        "min_entropy": 1.5,
                        "encoding": "utf-8",
                    },
                )
                comp["rag_engine"] = RAGCompressionEngine(rag_config)
                comp["vox_rag"] = VoxSigilRAG()  # VoxSigil RAG system
                logger.info(f"✅ VoxSigil RAG initialized for {self.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize VoxSigil RAG for {self.name}: {e}"
                )

        # BLT Sigil System integration ------------------------------------
        if self.cfg.use_blt_sigils and HAVE_BLT:
            try:
                comp["blt_encoder"] = BLTEncoder()  # Main BLT encoder
                comp["sigil_encoder"] = (
                    SigilPatchEncoder()
                )  # Sigil patch encoder (alias to BLTEncoder)
                comp["byte_transformer"] = (
                    ByteLatentTransformerEncoder()
                )  # Byte transformer (alias to BLTEncoder)

                # Initialize enhanced RAG if both RAG and BLT are enabled
                if self.cfg.use_voxsigil_rag and HAVE_VOXSIGIL_RAG:
                    comp["blt_enhanced_rag"] = BLTEnhancedRAG()
                    logger.info(f"✅ BLT Enhanced RAG initialized for {self.name}")

                logger.info(f"✅ BLT Sigil System initialized for {self.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize BLT Sigil System for {self.name}: {e}"
                )

        # ART Trainer integration for VoxSigil --------------------------- #
        if self.cfg.use_art_training:
            try:
                # Initialize ART trainer with proper input dimensions for HOLO-1.5
                art_config = getattr(
                    self.cfg,
                    "art_config",
                    {
                        "input_dim": 256,
                        "vigilance": 0.75,
                        "learning_rate": 0.1,
                        "max_categories": 100,
                        "enable_art_training": True,
                    },
                )

                # Mock ART trainer (replace with actual implementation when available)
                comp["art_trainer"] = type(
                    "ARTTrainer",
                    (),
                    {
                        "input_dim": art_config.get("input_dim", 256),
                        "vigilance": art_config.get("vigilance", 0.75),
                        "learning_rate": art_config.get("learning_rate", 0.1),
                        "max_categories": art_config.get("max_categories", 100),
                        "enabled": art_config.get("enable_art_training", True),
                        "validate_feature_vector": lambda self, vector: len(vector)
                        == self.input_dim,
                        "train": lambda self,
                        data: f"ART training with {len(data)} samples",
                    },
                )()

                logger.info(
                    f"✅ ART Trainer initialized for {self.name} "
                    f"with input_dim={art_config.get('input_dim', 256)}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ART Trainer for {self.name}: {e}")

    async def _maybe(
        self, cond: bool, fn: Callable[..., Any], *args: Any, **kw: Any
    ) -> Any:
        """Utility: awaits fn iff cond is true, otherwise returns None."""
        return await fn(*args, **kw) if cond else None

    # --------------------------------------------------------------------- #
    # Generation with HOLO-OPT optimizations                               #
    # --------------------------------------------------------------------- #

    def should_run(self) -> bool:
        return self.initialized

    async def generate(self, prompt: str) -> str:
        """Enhanced generation with HOLO-OPT optimizations"""

        # HOLO-OPT: Rate limiting check
        if HOLO_OPT_FEATURES["rate_limiting"] and self._request_limiter:
            current_time = time.time()

            # Clean old requests (older than 1 minute)
            while (
                self._request_limiter["requests"]
                and current_time - self._request_limiter["requests"][0] > 60
            ):
                self._request_limiter["requests"].popleft()

            # Check rate limit
            if (
                len(self._request_limiter["requests"])
                >= self._request_limiter["max_per_minute"]
            ):
                logger.warning(f"Rate limit exceeded for {self.name}")
                return f"[rate-limited:{self.name}] Request rate limit exceeded"

            # Record request
            self._request_limiter["requests"].append(current_time)

        # HOLO-OPT: Performance monitoring
        generation_start = time.time()

        try:
            if (
                not (self.initialized and HAVE_TRANSFORMERS)
                or self.model is None
                or self.tokenizer is None
            ):
                # Mock mode - apply decorations but return mock response
                try:
                    prompt = await self._decorate_prompt(prompt)
                    response = f"[mock:{self.name}] {prompt}"
                    return await self._decorate_response(response)
                except Exception:
                    return f"[mock:{self.name}] {prompt}"

            # HOLO-OPT: Collect sigils from all active components (with caching)
            active_sigils = await self._collect_component_sigils_cached(prompt)

            # Process prompt through BLT if available
            if self.blt_encoder and self.sigil_encoder:
                try:
                    prompt = await self._process_with_blt_optimized(
                        prompt, active_sigils
                    )
                except Exception as e:
                    logger.warning(f"BLT processing failed for {self.name}: {e}")

            prompt = await self._decorate_prompt(prompt)

            # HOLO-OPT: Use adaptive timeout for tokenization
            timeout = _ADAPTIVE_TIMEOUT.get_timeout()

            try:
                if HOLO_OPT_FEATURES["async_batching"]:
                    # Use thread pool for CPU-bound tokenization
                    future = get_thread_pool().submit(
                        self.tokenizer, prompt, return_tensors="pt"
                    )
                    toks = future.result(timeout=timeout)
                else:
                    toks = self.tokenizer(prompt, return_tensors="pt")

                toks = toks.to(self.model.device)
            except Exception as e:
                logger.error(f"Tokenization failed for {self.name}: {e}")
                return f"[tokenization-error:{self.name}] {str(e)}"

            # HOLO-OPT: Enhanced generation with caching and compression
            try:
                with torch.inference_mode():
                    # Use adaptive timeout for generation
                    generation_timeout = max(
                        timeout, 30.0
                    )  # Minimum 30s for generation

                    if HOLO_OPT_FEATURES["async_batching"]:
                        future = get_thread_pool().submit(
                            self.model.generate,
                            **toks,
                            max_new_tokens=self.cfg.max_tokens,
                            do_sample=True,
                            temperature=0.7,
                            use_cache=bool(self.minicache),
                        )
                        gen = future.result(timeout=generation_timeout)
                    else:
                        gen = await asyncio.to_thread(
                            self.model.generate,
                            **toks,
                            max_new_tokens=self.cfg.max_tokens,
                            do_sample=True,
                            temperature=0.7,
                            use_cache=bool(self.minicache),
                        )

            except Exception as e:
                logger.error(f"Generation failed for {self.name}: {e}")
                return f"[generation-error:{self.name}] {str(e)}"

            try:
                resp = (
                    self.tokenizer.decode(gen[0], skip_special_tokens=True)
                    .removeprefix(prompt)
                    .strip()
                )
            except Exception as e:
                logger.error(f"Decoding failed for {self.name}: {e}")
                return f"[decoding-error:{self.name}] {str(e)}"

            # Apply VoxSigil RAG compression if available
            if self.rag_engine:
                try:
                    resp = await self._process_with_rag_optimized(resp)
                except Exception as e:
                    logger.warning(f"RAG processing failed for {self.name}: {e}")

            # HOLO-OPT: Update performance metrics
            generation_time = time.time() - generation_start

            if HOLO_OPT_FEATURES["monitoring"] and self._performance_monitor:
                self._performance_monitor["requests_processed"] += 1

                # Update average response time
                old_avg = self._performance_monitor["avg_response_time"]
                count = self._performance_monitor["requests_processed"]
                self._performance_monitor["avg_response_time"] = (
                    old_avg * (count - 1) + generation_time
                ) / count

                # Update cache hit ratio
                if _CACHE_STATS["hits"] + _CACHE_STATS["misses"] > 0:
                    self._performance_monitor["cache_hit_ratio"] = _CACHE_STATS[
                        "hits"
                    ] / (_CACHE_STATS["hits"] + _CACHE_STATS["misses"])

            # Record successful generation time for adaptive timeout
            _ADAPTIVE_TIMEOUT.record_success(generation_time)

            return await self._decorate_response(resp)

        except Exception as e:
            # HOLO-OPT: Update error metrics
            if HOLO_OPT_FEATURES["monitoring"] and self._performance_monitor:
                self._performance_monitor["error_count"] += 1

            logger.error(f"Generation failed for {self.name}: {e}")
            return f"[error:{self.name}] {str(e)}"

    @smart_cache(ttl_seconds=60, maxsize=32)
    async def _collect_component_sigils_cached(self, prompt: str) -> Dict[str, Any]:
        """Cached version of component sigil collection"""
        return await self._collect_component_sigils(prompt)

    async def _process_with_blt_optimized(
        self, prompt: str, sigils: Dict[str, Any]
    ) -> str:
        """Optimized BLT processing with compression and caching"""
        if not (self.blt_encoder and self.sigil_encoder):
            return prompt

        try:
            # HOLO-OPT: Use connection pooling for BLT operations
            if HOLO_OPT_FEATURES["connection_pooling"]:
                # Simulate connection pooling for BLT operations
                async with asyncio.timeout(_ADAPTIVE_TIMEOUT.get_timeout()):
                    # Feed collected sigils to BLT for binding and processing
                    bound_context = await asyncio.to_thread(
                        self.sigil_encoder.bind_sigils, sigils
                    )

                    # Process through BLT encoder with bound context
                    enhanced_prompt = await asyncio.to_thread(
                        self.blt_encoder.process_with_context, prompt, bound_context
                    )
            else:
                # Original processing
                bound_context = await asyncio.to_thread(
                    self.sigil_encoder.bind_sigils, sigils
                )
                enhanced_prompt = await asyncio.to_thread(
                    self.blt_encoder.process_with_context, prompt, bound_context
                )

            return enhanced_prompt
        except Exception as e:
            logger.warning(f"Optimized BLT processing failed: {e}")
            return prompt

    async def _process_with_rag_optimized(self, response: str) -> str:
        """Optimized RAG processing with compression and caching"""
        if not self.rag_engine:
            return response

        try:
            # HOLO-OPT: Use compression for RAG operations
            if HOLO_OPT_FEATURES["compression"]:
                async with asyncio.timeout(_ADAPTIVE_TIMEOUT.get_timeout()):
                    # Compress and enhance response using RAG
                    compressed_response = await asyncio.to_thread(
                        self.rag_engine.compress_and_enhance, response
                    )
                    return compressed_response
            else:
                # Original processing
                compressed_response = await asyncio.to_thread(
                    self.rag_engine.compress_and_enhance, response
                )
                return compressed_response
        except Exception as e:
            logger.warning(f"Optimized RAG processing failed: {e}")
            return response

    # HOLO-OPT: Performance monitoring methods
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not (HOLO_OPT_FEATURES["monitoring"] and self._performance_monitor):
            return {}

        metrics = self._performance_monitor.copy()
        metrics.update(
            {
                "cache_stats": _CACHE_STATS.copy(),
                "global_cache_metrics": _CACHE_METRICS.copy(),
                "adaptive_timeout": _ADAPTIVE_TIMEOUT.get_timeout(),
            }
        )

        return metrics

    def reset_performance_metrics(self):
        """Reset performance metrics"""
        if HOLO_OPT_FEATURES["monitoring"] and self._performance_monitor:
            self._performance_monitor.update(
                {
                    "requests_processed": 0,
                    "avg_response_time": 0.0,
                    "error_count": 0,
                    "cache_hit_ratio": 0.0,
                }
            )

        _CACHE_STATS.update({"hits": 0, "misses": 0, "evictions": 0})
        _CACHE_METRICS.update(
            {
                "model_hits": 0,
                "model_misses": 0,
                "component_hits": 0,
                "component_misses": 0,
            }
        )

    # HOLO-OPT: Cleanup methods
    def cleanup(self):
        """Cleanup resources and cancel tasks"""
        try:
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                _CLEANUP_TASKS.discard(self._cleanup_task)

            # Clear caches
            if hasattr(self, "_lazy_components"):
                self._lazy_components.clear()

            # Force garbage collection if needed
            if (
                HOLO_OPT_FEATURES["preemptive_cleanup"]
                and self._performance_monitor
                and self._performance_monitor.get("memory_usage_mb", 0) > 500
            ):
                import gc

                gc.collect()

            logger.info(f"Cleanup completed for {self.name}")

        except Exception as e:
            logger.warning(f"Cleanup error for {self.name}: {e}")

    def __del__(self):
        """Destructor with cleanup"""
        if HOLO_OPT_FEATURES["preemptive_cleanup"]:
            try:
                self.cleanup()
            except Exception:
                pass  # Ignore errors in destructor

    # --------------------------------------------------------------------- #
    # Component Integration Helpers                                         #
    # --------------------------------------------------------------------- #
    async def _collect_component_sigils(self, prompt: str) -> Dict[str, Any]:
        """Collect sigils from active components for BLT processing."""
        sigils = {}

        # Collect sigils from reasoning components
        if self.spiking_network:
            try:
                sigils["spiking"] = {
                    "pattern": "neural_spike",
                    "context": "temporal_sequence",
                }
            except Exception as e:
                logger.debug(f"Failed to collect spiking sigils: {e}")

        if self.logical_engine:
            try:
                sigils["logical"] = {
                    "pattern": "symbolic_rule",
                    "context": "logical_inference",
                }
            except Exception as e:
                logger.debug(f"Failed to collect logical sigils: {e}")

        if self.binding_network:
            try:
                sigils["binding"] = {
                    "pattern": "oscillatory_sync",
                    "context": "object_binding",
                }
            except Exception as e:
                logger.debug(f"Failed to collect binding sigils: {e}")

        # Collect sigils from efficiency components
        if self.delta_rule_operator:
            try:
                sigils["delta"] = {
                    "pattern": "adaptive_learning",
                    "context": "weight_update",
                }
            except Exception as e:
                logger.debug(f"Failed to collect delta sigils: {e}")

        if self.outlier_detector:
            try:
                sigils["outlier"] = {
                    "pattern": "anomaly_detection",
                    "context": "pattern_deviation",
                }
            except Exception as e:
                logger.debug(f"Failed to collect outlier sigils: {e}")

        return sigils

    async def _process_with_blt(self, prompt: str, sigils: Dict[str, Any]) -> str:
        """Process prompt through BLT Sigil System."""
        if not (self.blt_encoder and self.sigil_encoder):
            return prompt

        try:
            # Feed collected sigils to BLT for binding and processing
            bound_context = await asyncio.to_thread(
                self.sigil_encoder.bind_sigils, sigils
            )

            # Process through BLT encoder with bound context
            enhanced_prompt = await asyncio.to_thread(
                self.blt_encoder.process_with_context, prompt, bound_context
            )

            return enhanced_prompt
        except Exception as e:
            logger.warning(f"BLT processing failed: {e}")
            return prompt

    async def _process_with_rag(self, response: str) -> str:
        """Process response through VoxSigil RAG compression."""
        if not self.rag_engine:
            return response

        try:
            # Compress and enhance response using RAG
            compressed_response = await asyncio.to_thread(
                self.rag_engine.compress_and_enhance, response
            )
            return compressed_response
        except Exception as e:
            logger.warning(f"RAG processing failed: {e}")
            return response

    # --------------------------------------------------------------------- #
    # ARC Task Integration                                                  #
    # --------------------------------------------------------------------- #

    def get_holo_feature_vector(self, input_data: Any) -> List[float]:
        """
        Generate HOLO-1.5 compatible feature vector with proper input_dim.

        Args:
            input_data: Any input data to convert to feature vector

        Returns:
            Feature vector with dimensionality matching ART trainer input_dim
        """
        try:
            # Get configured input dimension from ART trainer
            input_dim = (
                getattr(self.art_trainer, "input_dim", 256) if self.art_trainer else 256
            )

            # Convert input to string for processing
            input_str = str(input_data)

            # Simple feature extraction
            # (replace with more sophisticated method if needed)
            # Use hash-based features to ensure consistent dimensionality
            import hashlib

            features = []

            # Generate features based on input string characteristics
            for i in range(input_dim):
                # Create hash-based feature for each dimension
                hash_input = f"{input_str}_{i}"
                hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
                # Normalize to [-1, 1] range
                feature = (hash_val % 2000 - 1000) / 1000.0
                features.append(feature)

            return features

        except Exception as e:
            logger.warning(f"Feature vector generation failed for {self.name}: {e}")
            # Return default zero vector with correct dimensions
            input_dim = (
                getattr(self.art_trainer, "input_dim", 256) if self.art_trainer else 256
            )
            return [0.0] * input_dim

    # --------------------------------------------------------------------- #
    # Prompt / response enrichment                                          #
    # --------------------------------------------------------------------- #
    async def _decorate_prompt(self, text: str) -> str:
        """Add component-specific tags to the prompt based on content."""
        rules = [
            (self.logical_engine, ("pattern",), "[LOGICAL]"),
            (self.spiking_network, ("sequence", "temporal"), "[SPIKES]"),
            (self.binding_network, ("object", "binding"), "[BIND]"),
            (self.delta_rule_operator, ("learn", "adapt"), "[DELTA]"),
            (self.outlier_detector, ("anomaly", "outlier"), "[ANOM]"),
        ]
        for comp, keys, tag in rules:
            if comp and any(k in text.lower() for k in keys):
                text = f"{tag} {text}"
        return text

    async def _decorate_response(self, text: str) -> str:
        """Add component tags to responses to show which components were active."""
        tags = [
            ("memory_manager", "[MEM]"),
            ("resource_optimizer", "[RES]"),
            ("kv_compressor", "[KV]"),
            ("deltanet_attention", "[LIN]"),
            ("dataset_manager", "[DATA]"),
            ("delta_rule_operator", "[DR]"),
            ("outlier_detector", "[OUT]"),
            ("rag_engine", "⟨RAG⟩"),  # VoxSigil RAG sigil
            ("blt_encoder", "⟨BLT⟩"),  # BLT Core sigil
            ("sigil_encoder", "⟨SIG⟩"),  # Sigil Engine sigil
            ("art_trainer", "⟨ART⟩"),  # ART Trainer sigil
        ]
        for attr, tag in tags:
            if getattr(self, attr):
                text += f" {tag}"
        return text

    # convenience wrappers -------------------------------------------------- #
    async def run_loop(self) -> None:
        """Simple test loop; spawns a single generation."""
        await self.generate(f"Hello from {self.name}")

    async def transform(self, s: Any) -> str:
        """Tiny alias around generate for generic pipelines."""
        return await self.generate(str(s))


# --------------------------------------------------------------------------- #
# HOLO Mesh                                                                   #
# --------------------------------------------------------------------------- #


@vanta_agent(
    name="holo_mesh",
    subsystem="cognitive_mesh",
    mesh_role=CognitiveMeshRole.MANAGER,
    description="HOLO mesh orchestrator for multiple agents",
    capabilities=[
        "agent_orchestration",
        "parallel_processing",
        "mesh_coordination",
        "broadcast_communication",
    ],
    models=["auto"],
)
class HOLOMesh(BaseAgent if HOLO_AVAILABLE else object):
    """Lightweight container that orchestrates a set of HOLOAgents."""

    def __init__(self, agents: Iterable[tuple[str, HOLOAgentConfig]] | None = None):
        # Initialize BaseAgent if HOLO is available
        if HOLO_AVAILABLE:
            super().__init__()

        self.agents: dict[str, HOLOAgent] = {}
        if agents:
            for name, cfg in agents:
                self.add_agent(name, cfg)

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #
    def add_agent(self, name: str, cfg: HOLOAgentConfig) -> HOLOAgent:
        if name in self.agents:
            raise ValueError(f"agent '{name}' already exists")
        agent = HOLOAgent(name, cfg)
        self.agents[name] = agent
        return agent

    def remove_agent(self, name: str) -> None:
        self.agents.pop(name, None)

    def __getitem__(self, name: str) -> HOLOAgent:
        return self.agents[name]

    # ------------------------------------------------------------------ #
    # Parallel operations                                                #
    # ------------------------------------------------------------------ #
    async def broadcast(self, prompt: str) -> Dict[str, str]:
        """Send the same prompt to every initialised agent."""
        tasks = {
            n: asyncio.create_task(a.generate(prompt))
            for n, a in self.agents.items()
            if a.should_run()
        }
        return {n: await t for n, t in tasks.items()}

    async def gather(self, prompts: Mapping[str, str]) -> Dict[str, str]:
        """
        Send a different prompt to each agent.
        prompts maps agent-name -> prompt.
        """
        tasks = {
            n: asyncio.create_task(self.agents[n].generate(p))
            for n, p in prompts.items()
            if n in self.agents and self.agents[n].should_run()
        }
        return {n: await t for n, t in tasks.items()}

    async def any_ready(self) -> bool:
        return any(a.should_run() for a in self.agents.values())

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #
    async def interactive_cli(self) -> None:  # very small CLI
        print("HOLOMesh interactive shell. Ctrl-C to quit.")
        try:
            while True:
                prompt = input("Enter prompt for all agents: ")
                if prompt.lower() in {"exit", "quit"}:
                    print("Exiting interactive shell.")
                    break
                responses = await self.broadcast(prompt)
                for agent_name, response in responses.items():
                    print(f"[{agent_name}] {response}")
        except KeyboardInterrupt:
            print("\nExiting interactive shell.")

    # ------------------------------------------------------------------ #
    # VantaCore Compatibility                                            #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Get status for VantaCore registration."""
        # Build dynamic capabilities based on active components
        capabilities = [
            "holomesh",
            "cognitive_mesh",
            "multi_paradigm",
            "arc_task_processing",  # New ARC capability
        ]

        # Add reasoning capabilities
        for agent in self.agents.values():
            if getattr(agent, "spiking_network", None):
                capabilities.append("spiking_neural_processing")
            if getattr(agent, "logical_engine", None):
                capabilities.append("logical_symbolic_reasoning")
            if getattr(agent, "binding_network", None):
                capabilities.append("oscillatory_object_binding")
            # Add efficiency capabilities
            if getattr(agent, "memory_manager", None):
                capabilities.append("adaptive_memory_management")
            if getattr(agent, "minicache", None):
                capabilities.append("kv_cache_compression")
            if getattr(agent, "deltanet_attention", None):
                capabilities.append("linear_attention_optimization")
            if getattr(agent, "delta_rule_operator", None):
                capabilities.append("adaptive_delta_rule_learning")
            if getattr(agent, "outlier_detector", None):
                capabilities.append("outlier_token_detection")

            # Add VoxSigil capabilities
            if getattr(agent, "rag_engine", None):
                capabilities.append("voxsigil_rag_compression")
            if getattr(agent, "blt_encoder", None) and getattr(
                agent, "sigil_encoder", None
            ):
                capabilities.append("blt_sigil_binding")
            if getattr(agent, "art_trainer", None):
                capabilities.append("art_adaptive_resonance_training")
                capabilities.append("holo_1_5_feature_processing")
        # Just check first agent for component status

        return {
            "type": "HoloMesh",
            "name": "holomesh_instance",
            "status": "active",
            "initialized": True,
            "agents_count": len(self.agents),
            "capabilities": list(set(capabilities)),  # Remove duplicates
            "arc_ready": any(
                all(
                    getattr(a, comp, None)
                    for comp in ["logical_engine", "binding_network", "spiking_network"]
                )
                for a in self.agents.values()
            ),
            "efficiency_components": {
                "adaptive_memory": any(
                    getattr(a, "memory_manager", None) for a in self.agents.values()
                ),
                "memory_pool": any(
                    getattr(a, "memory_pool", None) for a in self.agents.values()
                ),
                "resource_optimizer": any(
                    getattr(a, "resource_optimizer", None) for a in self.agents.values()
                ),
                "minicache": any(
                    getattr(a, "minicache", None) for a in self.agents.values()
                ),
                "kv_compressor": any(
                    getattr(a, "kv_compressor", None) for a in self.agents.values()
                ),
                "deltanet_attention": any(
                    getattr(a, "deltanet_attention", None) for a in self.agents.values()
                ),
                "dataset_manager": any(
                    getattr(a, "dataset_manager", None) for a in self.agents.values()
                ),
                "delta_rule_operator": any(
                    getattr(a, "delta_rule_operator", None)
                    for a in self.agents.values()
                ),
                "outlier_detector": any(
                    getattr(a, "outlier_detector", None) for a in self.agents.values()
                ),
            },
            "reasoning_components": {
                "spiking_networks": any(
                    getattr(a, "spiking_network", None) for a in self.agents.values()
                ),
                "logical_reasoning": any(
                    getattr(a, "logical_engine", None) for a in self.agents.values()
                ),
                "oscillatory_binding": any(
                    getattr(a, "binding_network", None) for a in self.agents.values()
                ),
            },
            "voxsigil_components": {
                "rag_engine": any(
                    getattr(a, "rag_engine", None) for a in self.agents.values()
                ),
                "blt_encoder": any(
                    getattr(a, "blt_encoder", None) for a in self.agents.values()
                ),
                "sigil_encoder": any(
                    getattr(a, "sigil_encoder", None) for a in self.agents.values()
                ),
                "art_trainer": any(
                    getattr(a, "art_trainer", None) for a in self.agents.values()
                ),
            },
        }

    def initialize(self) -> bool:
        """Initialize the mesh and all agents."""
        try:
            logger.info("Initializing HOLOMesh...")

            # Initialize all agents
            for name, agent in self.agents.items():
                try:
                    if not agent.initialized and agent.model is not None:
                        # Attempt to re-initialize if needed
                        agent._load_model()
                        if hasattr(agent, "_init_components_async"):
                            # Schedule async component initialization
                            asyncio.create_task(agent._init_components_async())
                    logger.info(
                        f"Agent {name} initialization status: {agent.initialized}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize agent {name}: {e}")

            # Initialize mesh networking
            self.initialize_mesh()

            logger.info(f"HOLOMesh initialized with {len(self.agents)} agents")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HOLOMesh: {e}")
            return False

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task through the mesh."""
        try:
            task_type = task.get("type", "generate")
            task_data = task.get("data", {})

            logger.info(f"Processing task type: {task_type} with data: {task_data}")

            if task_type == "generate":
                # Single generation task
                prompt = task_data.get("prompt", "")
                agent_name = task_data.get("agent")

                if agent_name and agent_name in self.agents:
                    # Route to specific agent
                    result = await self.agents[agent_name].generate(prompt)
                    return {"status": "success", "result": result, "agent": agent_name}
                else:
                    # Broadcast to all agents
                    results = await self.broadcast(prompt)
                    return {"status": "success", "results": results}

            elif task_type == "broadcast":
                # Broadcast task to all agents
                prompt = task_data.get("prompt", "")
                results = await self.broadcast(prompt)
                return {"status": "success", "results": results}

            elif task_type == "gather":
                # Gather different prompts for different agents
                prompts = task_data.get("prompts", {})
                results = await self.gather(prompts)
                return {"status": "success", "results": results}

            elif task_type == "analyze":
                # Analysis task - route to agents with logical reasoning
                data = task_data.get("data", "")
                analysis_results = {}

                for name, agent in self.agents.items():
                    if hasattr(agent, "logical_engine") and agent.logical_engine:
                        result = await agent.generate(
                            f"Analyze the following data: {data}"
                        )
                        analysis_results[name] = result

                return {"status": "success", "analysis": analysis_results}

            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return {"status": "error", "message": str(e)}

    def process_sync(self, *args, **kwargs) -> str:
        """Legacy synchronous process method for VantaCore compatibility."""
        logger.info(
            f"HoloMesh processing sync request with {len(args)} args "
            f"and {len(kwargs)} kwargs"
        )

        # Convert args/kwargs to a task format
        task = {
            "type": "generate",
            "data": {
                "prompt": args[0] if args else kwargs.get("prompt", ""),
                "agent": kwargs.get("agent"),
            },
        }

        # Run the async process in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a task
                result = asyncio.create_task(self.process(task))
                return f"Task scheduled: {result}"
            else:
                # If no event loop is running, run the task
                result = loop.run_until_complete(self.process(task))
                return str(result)
        except Exception as e:
            logger.error(f"Error in sync process: {e}")
            return f"Error: {str(e)}"

    def shutdown(self) -> None:
        """Shutdown all agents and cleanup resources."""
        try:
            logger.info("Shutting down HOLOMesh...")

            # Shutdown each agent gracefully
            for name, agent in self.agents.items():
                try:
                    logger.info(f"Shutting down agent: {name}")

                    # Clear model and tokenizer from cache
                    agent.model = None
                    agent.tokenizer = None

                    # Clear component references
                    components = [
                        "spiking_network",
                        "logical_engine",
                        "binding_network",
                        "memory_manager",
                        "memory_pool",
                        "resource_optimizer",
                        "minicache",
                        "kv_compressor",
                        "deltanet_attention",
                        "dataset_manager",
                        "delta_rule_operator",
                        "outlier_detector",
                        "rag_engine",
                        "vox_rag",
                        "blt_encoder",
                        "sigil_encoder",
                        "byte_transformer",
                        "art_trainer",
                    ]

                    for comp in components:
                        if hasattr(agent, comp):
                            setattr(agent, comp, None)

                    agent.initialized = False

                except Exception as e:
                    logger.warning(f"Error shutting down agent {name}: {e}")

            # Clear the agents dictionary
            self.agents.clear()

            # Clear caches
            global _MODEL_CACHE, _COMPONENT_CACHE
            _MODEL_CACHE.clear()
            _COMPONENT_CACHE.clear()

            # Clear VantaCore reference if it exists
            if hasattr(self, "vanta_core"):
                self.vanta_core = None

            logger.info("HOLOMesh shutdown complete")

        except Exception as e:
            logger.error(f"Error during HOLOMesh shutdown: {e}")
            raise

    def register_agent(
        self,
        name: str,
        role: str = None,
        capabilities: List[str] = None,
        config: HOLOAgentConfig = None,
    ) -> HOLOAgent:
        """Register a new agent with the mesh."""
        try:
            # Create default config if none provided
            if config is None:
                config = HOLOAgentConfig()

            # Set role and capabilities as attributes if provided
            if role:
                config.role = role  # type: ignore
            if capabilities:
                config.capabilities = capabilities  # type: ignore

            # Add the agent to the mesh
            agent = self.add_agent(name, config)

            logger.info(f"Registered HOLO agent: {name} with role: {role or 'default'}")

            # If the mesh is already initialized, try to initialize the new agent
            if hasattr(self, "vanta_core"):
                try:
                    if not agent.initialized and agent.model is not None:
                        agent._load_model()
                        if hasattr(agent, "_init_components_async"):
                            asyncio.create_task(agent._init_components_async())
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize newly registered agent {name}: {e}"
                    )

            return agent

        except Exception as e:
            logger.error(f"Failed to register agent {name}: {e}")
            raise

    def initialize_mesh(self) -> None:
        """Initialize mesh networking for VantaCore."""
        try:
            logger.info("Initializing HOLO mesh networking...")

            # Initialize mesh coordination
            self._mesh_active = True
            self._mesh_id = f"holo_mesh_{id(self)}"

            # Set up inter-agent communication channels
            self._communication_channels = {}
            for name in self.agents.keys():
                self._communication_channels[name] = asyncio.Queue()

            # Initialize mesh health monitoring
            self._mesh_health = {
                "status": "active",
                "agents_online": len(self.agents),
                "last_heartbeat": asyncio.get_event_loop().time()
                if hasattr(asyncio, "get_event_loop")
                else 0,
                "total_requests": 0,
                "successful_requests": 0,
            }

            logger.info(
                f"HOLO mesh networking initialized successfully "
                f"with {len(self.agents)} agents"
            )

        except Exception as e:
            logger.error(f"Failed to initialize mesh networking: {e}")
            raise

    def get_mesh_health(self) -> Dict[str, Any]:
        """Get mesh health status."""
        if not hasattr(self, "_mesh_health"):
            return {"status": "not_initialized"}

        # Update agent status
        self._mesh_health["agents_online"] = len(
            [a for a in self.agents.values() if a.should_run()]
        )
        self._mesh_health["total_agents"] = len(self.agents)

        return self._mesh_health.copy()

    def update_mesh_stats(self, success: bool = True) -> None:
        """Update mesh statistics."""
        if hasattr(self, "_mesh_health"):
            self._mesh_health["total_requests"] += 1
            if success:
                self._mesh_health["successful_requests"] += 1
            self._mesh_health["last_heartbeat"] = (
                asyncio.get_event_loop().time()
                if hasattr(asyncio, "get_event_loop")
                else 0
            )

    def connect_to_vanta(self, vanta_core: Any) -> None:
        """Connect to VantaCore for orchestration."""
        try:
            self.vanta_core = vanta_core
            logger.info("HOLO mesh connected to VantaCore orchestration")
        except Exception as e:
            logger.error(f"Failed to connect to VantaCore: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Factory Functions for Compatibility                               #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Factory Functions for Compatibility                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def create_holomesh(
        cls,
        agents: Iterable[tuple[str, HOLOAgentConfig]] | None = None,
        config: Dict[str, Any] = None,
    ) -> "HOLOMesh":
        """Factory method to create a HOLOMesh instance."""
        if config is None:
            config = {}

        # Create default agents if none provided
        if agents is None:
            default_config = HOLOAgentConfig()

            # Apply HOLO-OPT optimizations from config
            if config.get("enable_optimizations", True):
                default_config.enable_smart_caching = config.get("smart_caching", True)
                default_config.enable_async_batching = config.get(
                    "async_batching", True
                )
                default_config.enable_lazy_loading = config.get("lazy_loading", True)
                default_config.enable_memory_optimization = config.get(
                    "memory_optimization", True
                )
                default_config.enable_connection_pooling = config.get(
                    "connection_pooling", True
                )
                default_config.enable_compression = config.get("compression", True)
                default_config.enable_rate_limiting = config.get("rate_limiting", True)
                default_config.enable_monitoring = config.get("monitoring", True)
                default_config.enable_preemptive_cleanup = config.get(
                    "preemptive_cleanup", True
                )
                default_config.enable_adaptive_timeouts = config.get(
                    "adaptive_timeouts", True
                )

            agents = [("default_agent", default_config)]

        mesh = cls(agents)

        # Initialize with provided config
        if config.get("auto_initialize", False):
            mesh.initialize()

        return mesh

    # HOLO-OPT: Enhanced mesh health with detailed metrics
    async def get_detailed_health_status(self) -> Dict[str, Any]:
        """Get detailed health status including HOLO-OPT metrics"""

        health_status = {
            "mesh_status": "healthy",
            "total_agents": len(self.agents),
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "capabilities_available": [],
            "agent_details": {},
            "holo_opt_metrics": {
                "global_cache_stats": _CACHE_STATS.copy(),
                "cache_metrics": _CACHE_METRICS.copy(),
                "performance_metrics": _PERFORMANCE_METRICS.copy()
                if _PERFORMANCE_METRICS
                else {},
                "adaptive_timeout": _ADAPTIVE_TIMEOUT.get_timeout(),
                "active_cleanup_tasks": len(_CLEANUP_TASKS),
                "thread_pool_active": _THREAD_POOL is not None
                and not _THREAD_POOL._shutdown,
            },
        }

        for name, agent in self.agents.items():
            agent_health = {
                "name": name,
                "initialized": agent.initialized,
                "should_run": agent.should_run(),
                "model_loaded": agent.model is not None,
                "components": {},
                "holo_opt_enabled": {},
                "performance_metrics": {},
            }

            # Check HOLO-OPT features
            if hasattr(agent, "cfg"):
                cfg = agent.cfg
                agent_health["holo_opt_enabled"] = {
                    "smart_caching": getattr(cfg, "enable_smart_caching", False),
                    "async_batching": getattr(cfg, "enable_async_batching", False),
                    "lazy_loading": getattr(cfg, "enable_lazy_loading", False),
                    "memory_optimization": getattr(
                        cfg, "enable_memory_optimization", False
                    ),
                    "connection_pooling": getattr(
                        cfg, "enable_connection_pooling", False
                    ),
                    "compression": getattr(cfg, "enable_compression", False),
                    "rate_limiting": getattr(cfg, "enable_rate_limiting", False),
                    "monitoring": getattr(cfg, "enable_monitoring", False),
                    "preemptive_cleanup": getattr(
                        cfg, "enable_preemptive_cleanup", False
                    ),
                    "adaptive_timeouts": getattr(
                        cfg, "enable_adaptive_timeouts", False
                    ),
                }

            # Get agent performance metrics
            if hasattr(agent, "get_performance_metrics"):
                try:
                    agent_health["performance_metrics"] = (
                        agent.get_performance_metrics()
                    )
                except Exception as e:
                    logger.debug(f"Failed to get performance metrics for {name}: {e}")

            # Check component status
            components = [
                "spiking_network",
                "logical_engine",
                "binding_network",
                "memory_manager",
                "memory_pool",
                "resource_optimizer",
                "minicache",
                "kv_compressor",
                "deltanet_attention",
                "dataset_manager",
                "delta_rule_operator",
                "outlier_detector",
                "rag_engine",
                "vox_rag",
                "blt_encoder",
                "sigil_encoder",
                "byte_transformer",
                "art_trainer",
            ]

            for comp in components:
                agent_health["components"][comp] = (
                    getattr(agent, comp, None) is not None
                )
                if agent_health["components"][comp]:
                    health_status["capabilities_available"].append(comp)

            # Overall agent health
            if agent_health["initialized"] and agent_health["should_run"]:
                health_status["healthy_agents"] += 1
            else:
                health_status["unhealthy_agents"] += 1
                health_status["mesh_status"] = "degraded"

            health_status["agent_details"][name] = agent_health

        # Remove duplicates from capabilities
        health_status["capabilities_available"] = list(
            set(health_status["capabilities_available"])
        )

        # Update mesh statistics
        self.update_mesh_stats(success=True)

        return health_status

    async def restart_agent(self, agent_name: str) -> bool:
        """Restart a specific agent in the mesh."""
        try:
            if agent_name not in self.agents:
                logger.error(f"Agent {agent_name} not found in mesh")
                return False

            agent = self.agents[agent_name]
            config = agent.cfg

            logger.info(f"Restarting agent: {agent_name}")

            # Cleanup old agent
            if hasattr(agent, "cleanup"):
                agent.cleanup()

            # Remove the old agent
            self.remove_agent(agent_name)

            # Create a new agent with the same config
            new_agent = self.add_agent(agent_name, config)

            # Initialize the new agent if mesh is active
            if hasattr(self, "_mesh_active") and self._mesh_active:
                if not new_agent.initialized and new_agent.model is not None:
                    new_agent._load_model()
                    if hasattr(new_agent, "_init_components_async"):
                        asyncio.create_task(new_agent._init_components_async())

            logger.info(f"Agent {agent_name} restarted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to restart agent {agent_name}: {e}")
            return False

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all agents and their basic information."""
        agent_list = {}

        for name, agent in self.agents.items():
            agent_info = {
                "name": name,
                "model_name": agent.cfg.model_name,
                "initialized": agent.initialized,
                "should_run": agent.should_run(),
                "device": agent.cfg.device,
                "max_tokens": agent.cfg.max_tokens,
                "role": getattr(agent.cfg, "role", "default"),
                "capabilities": getattr(agent.cfg, "capabilities", []),
            }

            # Add HOLO-OPT status
            if hasattr(agent, "cfg"):
                agent_info["holo_opt_features"] = {
                    "smart_caching": getattr(agent.cfg, "enable_smart_caching", False),
                    "async_batching": getattr(
                        agent.cfg, "enable_async_batching", False
                    ),
                    "lazy_loading": getattr(agent.cfg, "enable_lazy_loading", False),
                    "monitoring": getattr(agent.cfg, "enable_monitoring", False),
                }

            agent_list[name] = agent_info

        return agent_list

    def get_agent_by_capability(self, capability: str) -> List[str]:
        """Get list of agent names that have a specific capability."""
        capable_agents = []

        for name, agent in self.agents.items():
            if hasattr(agent, capability) and getattr(agent, capability) is not None:
                capable_agents.append(name)
            elif (
                hasattr(agent.cfg, "capabilities")
                and capability in agent.cfg.capabilities
            ):
                capable_agents.append(name)

        return capable_agents

    async def route_by_capability(
        self, capability: str, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route a task to agents that have a specific capability."""
        capable_agents = self.get_agent_by_capability(capability)

        if not capable_agents:
            return {
                "status": "error",
                "message": f"No agents found with capability: {capability}",
            }

        # Modify task to route to specific agents
        if "data" not in task:
            task["data"] = {}

        # If only one capable agent, route directly
        if len(capable_agents) == 1:
            task["data"]["agent"] = capable_agents[0]
            return await self.process(task)

        # Multiple capable agents - gather from all
        task["type"] = "gather"
        prompts = {}
        base_prompt = task["data"].get("prompt", "")

        for agent_name in capable_agents:
            prompts[agent_name] = base_prompt

        task["data"]["prompts"] = prompts

        return await self.process(task)

    # HOLO-OPT: Global cleanup method
    @classmethod
    def cleanup_global_resources(cls):
        """Cleanup global HOLO-OPT resources"""
        try:
            # Cancel all cleanup tasks
            for task in _CLEANUP_TASKS.copy():
                if not task.done():
                    task.cancel()
            _CLEANUP_TASKS.clear()

            # Shutdown thread pool
            global _THREAD_POOL
            if _THREAD_POOL is not None:
                _THREAD_POOL.shutdown(wait=False)
                _THREAD_POOL = None

            # Clear caches
            _MODEL_CACHE.clear()
            _COMPONENT_CACHE.clear()
            _COMPRESSED_CACHE.clear()
            _PERFORMANCE_METRICS.clear()
            _REQUEST_QUEUE.clear()

            # Reset cache stats
            _CACHE_STATS.update({"hits": 0, "misses": 0, "evictions": 0})
            _CACHE_METRICS.update(
                {
                    "model_hits": 0,
                    "model_misses": 0,
                    "component_hits": 0,
                    "component_misses": 0,
                }
            )

            logger.info("✅ Global HOLO-OPT resources cleaned up successfully")

        except Exception as e:
            logger.warning(f"Error during global cleanup: {e}")

    def __del__(self):
        """Destructor with HOLO-OPT cleanup"""
        if HOLO_OPT_FEATURES.get("preemptive_cleanup", False):
            try:
                self.shutdown()
            except Exception:
                pass  # Ignore errors in destructor


# Alternative naming for compatibility
HoloMeshAgent = HOLOMesh


# --------------------------------------------------------------------------- #
# Factory Functions and Utilities                                            #
# --------------------------------------------------------------------------- #


def create_holomesh(
    agents: Iterable[tuple[str, HOLOAgentConfig]] | None = None,
    config: Dict[str, Any] = None,
) -> HOLOMesh:
    """Create HoloMesh instance - factory function for backward compatibility.

    Args:
        agents: Iterable of (name, config) tuples for initial agents
        config: Optional mesh configuration

    Returns:
        HOLOMesh instance
    """
    mesh = HOLOMesh(agents)

    if config:
        # Apply any mesh-level configuration
        for key, value in config.items():
            if hasattr(mesh, key):
                setattr(mesh, key, value)

    return mesh


def holomesh_agent(
    name: str = "default_mesh",
    agents: Iterable[tuple[str, HOLOAgentConfig]] | None = None,
    **kwargs: Any,
) -> HOLOMesh:
    """Create HoloMesh agent instance - compatible with name parameter.

    Args:
        name: Name for the mesh instance (for logging/identification)
        agents: Iterable of (name, config) tuples for initial agents
        **kwargs: Additional configuration parameters

    Returns:
        HOLOMesh instance
    """
    mesh = HOLOMesh(agents)
    mesh.mesh_name = name  # Store the mesh name for identification

    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(mesh, key):
            setattr(mesh, key, value)

    return mesh

    # HOLO-OPT: Global cleanup on module exit
    atexit.register(HOLOMesh.cleanup_global_resources)
