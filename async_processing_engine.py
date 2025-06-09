"""
Async Processing and Inference Engine for Vanta
Handles model loading, inference, and processing tasks asynchronously
"""

import asyncio
import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from Vanta.core.UnifiedAsyncBus import (
    MessageType,  # Import MessageType for async bus integration
)

logger = logging.getLogger("Vanta.AsyncProcessor")

# ML Dependencies
try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None  # type: ignore

try:
    import transformers

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    transformers = None  # type: ignore


@dataclass
class ProcessorConfig:
    max_workers: int = 4
    device: str = "auto"  # auto, cpu, cuda, mps
    model_cache_dir: str = "./models"
    max_memory_gb: float = (
        8.0  # Currently informational, not strictly enforced by simple offloading
    )
    inference_timeout: float = 30.0  # Default timeout for inference tasks
    enable_model_offloading: bool = True
    # Added by RefactorSynthesizer (FE006, FE009)
    max_loaded_models: Optional[int] = (
        None  # Max models in LRU cache if offloading is enabled
    )
    default_torch_dtype: Optional[str] = (
        "float16"  # e.g., "float16", "bfloat16", "float32", None
    )


class AsyncProcessingEngine:
    """Async Processing and Inference Engine"""

    COMPONENT_NAME = "async_processing_engine"

    def __init__(self, vanta_core: Any, config: ProcessorConfig):
        self.vanta_core = vanta_core
        self.config = config  # Processed by Pydantic if FE001 is active

        # FE001: Use Pydantic for config validation (conceptual, actual Pydantic model not shown here for brevity)
        # For now, we'll assume config is validated externally or by direct attribute access checks
        if not isinstance(config, ProcessorConfig):  # Basic type check
            raise TypeError("Config must be an instance of ProcessorConfig.")

        # Ensure model_cache_dir exists (moved from initialize for early setup)
        self.model_cache_path = Path(self.config.model_cache_dir).resolve()
        self.model_cache_path.mkdir(parents=True, exist_ok=True)

        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.device = self._determine_device()

        # FE006: Model Offloading (LRU)
        self.loaded_models: Dict[
            str, Any
        ] = {}  # Potentially an OrderedDict if LRU is fully active
        self.model_lru_order: List[str] = []  # For LRU tracking

        self.model_lock = threading.RLock()  # RLock for re-entrant safety
        self.is_initialized = False
        self.shutting_down = False  # FE007: For graceful shutdown

        # FE005: Task Prioritization
        self.processing_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.worker_tasks: List[asyncio.Task] = []

        # FE010: Detailed Statistics
        self._stats: Dict[str, Any] = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_timed_out": 0,
            "task_details": {},  # type: Dict[str, Dict[str, Any]]
            "model_load_events": [],  # type: List[Dict[str, Any]]
        }
        self._stats_lock = threading.Lock()

        # Register with VantaCore
        self.vanta_core.register_component(
            self.COMPONENT_NAME,
            self,
            {"type": "async_processor", "model_cache_dir": str(self.model_cache_path)},
        )
        logger.info(f"{self.COMPONENT_NAME} registered with VantaCore")

        # === Unified Async Bus Integration ===
        # Register with UnifiedVantaCore's async bus and subscribe handler
        if hasattr(self.vanta_core, "async_bus"):
            self.vanta_core.async_bus.register_component("processing_engine")
            self.vanta_core.async_bus.subscribe(
                "processing_engine",
                MessageType.PROCESSING_REQUEST,
                self.handle_processing_request,
            )
            logger.info(
                "processing_engine registered and subscribed to async bus (PROCESSING_REQUEST)"
            )
        else:
            logger.warning(
                "UnifiedVantaCore async bus not available; async bus integration skipped."
            )

    async def handle_processing_request(self, message):
        """
        Async bus handler for PROCESSING_REQUEST messages.
        Use this method for all async processing requests via UnifiedVantaCore's async bus.
        Args:
            message: AsyncMessage instance with processing task and metadata
        Returns:
            dict: Processing result
        """
        # Example: message.content should contain processing task data
        # Implement actual processing logic here
        # ...
        return {"error": "Not implemented", "success": False}

    def _determine_device(self) -> str:
        """Determine the best device for processing"""
        if self.config.device != "auto":
            if self.config.device == "cuda" and (
                not HAVE_TORCH or (torch and not torch.cuda.is_available())
            ):
                logger.warning("CUDA specified but not available. Falling back to CPU.")
                return "cpu"
            if self.config.device == "mps" and (
                not HAVE_TORCH
                or not (
                    torch
                    and hasattr(torch.backends, "mps")
                    and torch.device("mps").type == "mps"
                )
            ):
                logger.warning("MPS specified but not available. Falling back to CPU.")
                return "cpu"
            return self.config.device

        if HAVE_TORCH and torch:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.device("mps").type == "mps":
                logger.info(
                    "MPS device detected. Note: MPS support can be experimental."
                )
                return "mps"
        return "cpu"

    async def initialize(self) -> bool:
        """Initialize the processing engine"""
        if self.is_initialized:
            logger.warning("Processing Engine already initialized.")
            return True
        try:
            logger.info(f"Initializing Processing Engine on device: {self.device}")
            logger.info(f"Model cache directory: {self.model_cache_path}")

            if (
                self.config.enable_model_offloading
                and self.config.max_loaded_models is not None
            ):
                logger.info(
                    f"Model offloading enabled. LRU cache size: {self.config.max_loaded_models}"
                )
            elif self.config.enable_model_offloading:
                logger.warning(
                    "Model offloading enabled but max_loaded_models not set. Offloading will not be active."
                )

            # Start worker tasks
            for i in range(self.config.max_workers):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self.worker_tasks.append(task)

            self.is_initialized = True
            self.shutting_down = False

            self.vanta_core.publish_event(
                "processor.engine.initialized",
                {
                    "device": self.device,
                    "max_workers": self.config.max_workers,
                    "model_cache_dir": str(self.model_cache_path),
                    "torch_available": HAVE_TORCH,
                    "transformers_available": HAVE_TRANSFORMERS,
                },
                source="AsyncProcessingEngine",
            )
            logger.info("Processing Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Processing Engine: {e}", exc_info=True)
            self.is_initialized = False
            return False

    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing tasks"""
        logger.info(f"Processing worker {worker_id} started")

        while not self.shutting_down:
            try:
                # FE005: Get from priority queue
                # FE007: Shorter timeout during shutdown to allow faster exit
                queue_timeout = (
                    0.1 if self.shutting_down and self.processing_queue.empty() else 1.0
                )
                priority, task_data = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=queue_timeout
                )

                logger.debug(
                    f"Worker {worker_id} processing task: {task_data['type']} (Priority: {priority})"
                )

                # FE008: Per-task timeout
                task_timeout = task_data.get(
                    "timeout",
                    self.config.inference_timeout
                    if task_data["type"] == "inference"
                    else None,
                )

                if task_timeout is not None:
                    try:
                        await asyncio.wait_for(
                            self._execute_task(task_data, worker_id),
                            timeout=task_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Task {task_data['type']} (ID: {task_data.get('task_id', 'N/A')}) timed out after {task_timeout}s on worker {worker_id}"
                        )
                        self._update_task_stats(
                            task_data.get("type", "unknown"), "timed_out"
                        )
                        if "callback" in task_data:
                            await self._run_callback(
                                task_data["callback"],
                                None,
                                asyncio.TimeoutError(
                                    f"Task exceeded timeout of {task_timeout}s"
                                ),
                            )
                        # Fall through to task_done
                else:
                    await self._execute_task(task_data, worker_id)

                self.processing_queue.task_done()

            except asyncio.TimeoutError:  # Timeout for queue.get()
                if self.shutting_down and self.processing_queue.empty():
                    break  # Exit loop if shutting down and queue is empty
                continue  # Normal timeout, continue loop
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                # Potentially mark associated task as failed if identifiable
                await asyncio.sleep(1)  # Prevent rapid spinning on persistent error

        logger.info(f"Processing worker {worker_id} stopped")

    async def _execute_task(self, task_data: Dict, worker_id: str):
        """Execute a processing task"""
        task_type = task_data.get("type", "unknown")
        start_time = time.monotonic()
        result = None
        error = None

        try:
            if task_type == "load_model":
                result = await self._load_model_task(task_data)
            elif task_type == "unload_model":  # New task type for explicit unload
                result = await self._unload_model_task(task_data)
            elif task_type == "inference":
                result = await self._inference_task(task_data)
            elif task_type == "text_generation":  # FE003
                result = await self._text_generation_task(task_data)
            elif task_type == "embedding":  # FE004
                result = await self._embedding_task(task_data)
            elif task_type == "custom":
                result = await self._custom_task(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self._update_task_stats(
                task_type, "completed", time.monotonic() - start_time
            )

        except Exception as e:
            logger.error(
                f"Task execution error for type {task_type} (ID: {task_data.get('task_id', 'N/A')}): {e}",
                exc_info=True,
            )
            error = e
            self._update_task_stats(task_type, "failed", time.monotonic() - start_time)

        finally:
            processing_time = time.monotonic() - start_time
            if "callback" in task_data:
                await self._run_callback(task_data["callback"], result, error)

            event_name = (
                "processor.task.completed" if error is None else "processor.task.failed"
            )
            event_data = {
                "task_type": task_type,
                "task_id": task_data.get("task_id", "N/A"),
                "worker_id": worker_id,
                "processing_time": processing_time,
                "success": error is None,
            }
            if error:
                event_data["error"] = str(error)

            self.vanta_core.publish_event(
                event_name, event_data, source="AsyncProcessingEngine"
            )

    async def _run_callback(
        self, callback: Callable, result: Any, error: Optional[Exception]
    ):
        """Helper to run sync or async callbacks."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result, error)
            else:
                # Run synchronous callback in the event loop's default executor
                # to avoid blocking the worker's async handling.
                await asyncio.get_running_loop().run_in_executor(
                    None, callback, result, error
                )
        except Exception as e:
            logger.error(f"Error in task callback execution: {e}", exc_info=True)

    def _get_torch_dtype(self, requested_dtype: Optional[str] = None) -> Optional[Any]:
        """FE009: Get torch.dtype object from string."""
        if not HAVE_TORCH or torch is None:
            return None

        dtype_str = (
            requested_dtype
            if requested_dtype is not None
            else self.config.default_torch_dtype
        )
        if dtype_str is None or dtype_str.lower() == "none":
            return None  # Let transformers/torch decide

        if not isinstance(
            dtype_str, str
        ):  # Should be caught by Pydantic if used for config
            logger.warning(
                f"Invalid torch_dtype '{dtype_str}', expected a string. Using None."
            )
            return None

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            # Add other dtypes as needed
        }
        selected_dtype = dtype_map.get(dtype_str.lower())
        if selected_dtype is None:
            logger.warning(f"Unsupported torch_dtype: {dtype_str}. Defaulting to None.")
        return selected_dtype

    # FE006: Model Offloading (LRU) & FE009: Configurable Dtype
    async def _load_model_task(self, task_data: Dict) -> Any:
        model_name = task_data["model_name"]
        model_type = task_data.get(
            "model_type", "auto"
        )  # e.g., "auto", "transformers", "custom"
        pipeline_task = task_data.get(
            "pipeline_task"
        )  # For loading transformers.pipeline
        custom_loader = task_data.get("custom_loader")
        torch_dtype_str = task_data.get("torch_dtype")  # Per-task override

        with self.model_lock:
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded. Updating LRU order.")
                if (
                    self.config.enable_model_offloading
                    and self.config.max_loaded_models is not None
                ):
                    if model_name in self.model_lru_order:
                        self.model_lru_order.remove(model_name)
                    self.model_lru_order.append(model_name)
                return self.loaded_models[model_name]

            # FE006: LRU eviction logic
            if (
                self.config.enable_model_offloading
                and self.config.max_loaded_models is not None
                and len(self.loaded_models) >= self.config.max_loaded_models
            ):
                if not self.model_lru_order:  # Should not happen if logic is correct
                    logger.error(
                        "LRU offload enabled, cache full, but LRU order is empty. Cannot evict."
                    )
                else:
                    model_to_evict = self.model_lru_order.pop(
                        0
                    )  # Evict least recently used
                    logger.info(
                        f"Max loaded models ({self.config.max_loaded_models}) reached. Evicting {model_to_evict} (LRU)."
                    )
                    if model_to_evict in self.loaded_models:
                        del self.loaded_models[
                            model_to_evict
                        ]  # Actual offload (simple del for now)
                        if (
                            HAVE_TORCH
                            and torch is not None
                            and torch.cuda.is_available()
                        ):  # Basic cleanup
                            if torch is not None:
                                torch.cuda.empty_cache()
                        self._update_model_stats(model_to_evict, "offloaded_lru")

        logger.info(
            f"Loading model: {model_name} (Type: {model_type}, Pipeline Task: {pipeline_task})"
        )
        load_start_time = time.monotonic()
        loop = asyncio.get_running_loop()
        model = None

        try:
            if (
                HAVE_TRANSFORMERS
                and transformers is not None
                and model_type in ["auto", "transformers"]
            ):
                actual_torch_dtype = self._get_torch_dtype(torch_dtype_str)

                if pipeline_task:  # e.g. "text-generation", "sentence-similarity"
                    model = await loop.run_in_executor(
                        self.executor,
                        lambda: transformers.pipeline(  # type: ignore
                            task=pipeline_task,
                            model=model_name,
                            device=self.device
                            if self.device != "mps"
                            else -1,  # pipeline handles mps differently or may not
                            torch_dtype=actual_torch_dtype
                            if self.device == "cuda"
                            else None,  # dtype often for CUDA
                            cache_dir=str(self.model_cache_path),
                        ),
                    )
                else:  # Load model and tokenizer separately if needed, or just model
                    # For embedding models, usually AutoModel and AutoTokenizer
                    tokenizer = await loop.run_in_executor(
                        self.executor,
                        lambda: transformers.AutoTokenizer.from_pretrained(  # type: ignore
                            model_name, cache_dir=str(self.model_cache_path)
                        ),
                    )
                    _model = await loop.run_in_executor(
                        self.executor,
                        lambda: transformers.AutoModel.from_pretrained(  # type: ignore
                            model_name,
                            cache_dir=str(self.model_cache_path),
                            torch_dtype=actual_torch_dtype,
                        ),
                    )
                    if (
                        HAVE_TORCH and torch is not None and _model is not None
                    ):  # Move to device after loading
                        _model = await loop.run_in_executor(
                            self.executor, lambda: _model.to(self.device)
                        )  # type: ignore
                    model = {
                        "model": _model,
                        "tokenizer": tokenizer,
                    }  # Store both for embedding/feature extraction

            elif custom_loader:
                model = await loop.run_in_executor(
                    self.executor,
                    custom_loader,
                    model_name,
                    str(self.model_cache_path),
                    self.device,
                )
            else:
                raise ValueError(
                    f"Cannot load model {model_name}: No suitable loader (Transformers not available/specified, or no custom_loader)."
                )

            with self.model_lock:
                self.loaded_models[model_name] = model
                if (
                    self.config.enable_model_offloading
                    and self.config.max_loaded_models is not None
                ):
                    if model_name in self.model_lru_order:
                        self.model_lru_order.remove(model_name)  # Should not be if new
                    self.model_lru_order.append(model_name)

            load_time = time.monotonic() - load_start_time
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s.")
            self._update_model_stats(model_name, "loaded", load_time)
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            self._update_model_stats(
                model_name, "load_failed", time.monotonic() - load_start_time, str(e)
            )
            raise  # Re-raise to be caught by _execute_task

    async def _unload_model_task(self, task_data: Dict) -> Dict[str, Any]:
        """Unload a model explicitly."""
        model_name = task_data["model_name"]
        status = False
        message = ""
        with self.model_lock:
            if model_name in self.loaded_models:
                logger.info(f"Unloading model: {model_name}")
                del self.loaded_models[model_name]
                if model_name in self.model_lru_order:
                    self.model_lru_order.remove(model_name)

                # Basic cleanup, especially if it was on GPU
                if HAVE_TORCH and torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                status = True
                message = f"Model {model_name} unloaded successfully."
                self._update_model_stats(model_name, "unloaded_explicitly")
            else:
                logger.warning(
                    f"Attempted to unload model {model_name}, but it was not found."
                )
                message = f"Model {model_name} not found for unloading."

        return {"model_name": model_name, "unloaded": status, "message": message}

    async def _ensure_model_loaded(self, model_name: str, model_load_params: Dict):
        """Helper to load a model if not already loaded, for use in other tasks."""
        if model_name not in self.loaded_models:
            logger.info(f"Model {model_name} not loaded for task. Attempting to load.")
            # Construct a load_model task_data
            load_task_data = {
                "type": "load_model",
                "model_name": model_name,
                **model_load_params,  # e.g., model_type, pipeline_task, torch_dtype
            }
            await self._load_model_task(load_task_data)  # Await direct internal call

        with self.model_lock:  # Update LRU on access
            if (
                self.config.enable_model_offloading
                and self.config.max_loaded_models is not None
            ):
                if model_name in self.model_lru_order:
                    self.model_lru_order.remove(model_name)
                if model_name in self.loaded_models:
                    self.model_lru_order.append(model_name)  # ensure it's loaded
        return self.loaded_models[model_name]

    async def _inference_task(self, task_data: Dict) -> Any:
        model_name = task_data["model_name"]
        inputs = task_data["inputs"]
        # Default model load params for generic inference, can be overridden by task_data
        model_load_params = {
            "model_type": task_data.get("model_type", "auto"),
            "torch_dtype": task_data.get(
                "torch_dtype"
            ),  # Allow task to specify dtype for loading
        }
        model_artifact = await self._ensure_model_loaded(model_name, model_load_params)

        # Determine if it's a pipeline or a model/tokenizer dict
        actual_model = model_artifact
        if isinstance(model_artifact, dict) and "model" in model_artifact:
            actual_model = model_artifact[
                "model"
            ]  # If it's our {'model': ..., 'tokenizer': ...} structure

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, lambda: self._run_inference(actual_model, inputs, task_data)
        )

    def _run_inference(self, model: Any, inputs: Any, task_data: Dict) -> Any:
        """Run inference (blocking, called in thread pool)"""
        try:
            custom_inference_func = task_data.get("custom_inference")
            if custom_inference_func:
                return custom_inference_func(model, inputs)

            # Standard inference for Hugging Face models/pipelines
            if HAVE_TORCH and torch is not None and hasattr(model, "__call__"):
                with torch.no_grad():  # Essential for inference
                    # Handle pipeline objects
                    if (
                        HAVE_TRANSFORMERS
                        and transformers
                        and isinstance(model, transformers.pipelines.base.Pipeline)
                    ):
                        # Pipelines expect direct inputs, often not kwargs for simple cases
                        if (
                            isinstance(inputs, list)
                            or isinstance(inputs, str)
                            or isinstance(inputs, dict)
                        ):
                            return (
                                model(inputs)
                                if not isinstance(inputs, dict)
                                else model(**inputs)
                            )
                        else:
                            raise ValueError(
                                f"Unsupported input type for Hugging Face pipeline: {type(inputs)}"
                            )

                    # Handle raw models (e.g., AutoModel output) - usually needs tokenized input
                    # This part is tricky as "inputs" would need to be pre-tokenized and moved to device
                    # For now, assume inputs are correctly prepared if it's not a pipeline
                    if isinstance(inputs, dict):
                        # Move tensor inputs to device if model is a torch Module
                        if isinstance(model, torch.nn.Module):
                            inputs_on_device = {
                                k: v.to(self.device)
                                if isinstance(v, torch.Tensor)
                                else v
                                for k, v in inputs.items()
                            }
                            return model(**inputs_on_device)
                        return model(**inputs)
                    else:  # Single tensor or other direct input
                        if isinstance(model, torch.nn.Module) and isinstance(
                            inputs, torch.Tensor
                        ):
                            inputs_on_device = inputs.to(self.device)
                            return model(inputs_on_device)
                        return model(inputs)
            elif hasattr(model, "__call__"):  # Non-torch callable model
                return (
                    model(inputs) if not isinstance(inputs, dict) else model(**inputs)
                )
            else:
                raise TypeError(
                    f"Model {task_data['model_name']} is not callable or supported for inference."
                )

        except Exception as e:
            logger.error(
                f"Inference error with model {task_data['model_name']}: {e}",
                exc_info=True,
            )
            raise

    # FE003: Concrete Text Generation Task Implementation
    async def _text_generation_task(
        self, task_data: Dict
    ) -> Any:  # Return type can be str or List[Dict[str,str]]
        if not (HAVE_TRANSFORMERS and HAVE_TORCH):
            raise RuntimeError(
                "Text generation requires PyTorch and Transformers to be installed."
            )

        model_name = task_data["model_name"]
        prompt = task_data["prompt"]
        # Pipeline args:
        generation_kwargs = task_data.get("generation_kwargs", {})
        generation_kwargs.setdefault("max_length", task_data.get("max_length", 100))
        generation_kwargs.setdefault("temperature", task_data.get("temperature", 0.7))
        # other common args: num_return_sequences, top_k, top_p, do_sample

        # Ensure the text generation pipeline is loaded
        model_load_params = {
            "model_type": "transformers",  # Explicitly use transformers
            "pipeline_task": "text-generation",
            "torch_dtype": task_data.get("torch_dtype"),
        }
        pipeline = await self._ensure_model_loaded(model_name, model_load_params)

        if (
            transformers is None
            or not isinstance(pipeline, transformers.pipelines.base.Pipeline)
            or pipeline.task != "text-generation"
        ):
            raise ValueError(
                f"Model {model_name} is not a valid text-generation pipeline."
            )

        loop = asyncio.get_running_loop()
        # Pipelines are callable
        # Note: some pipelines return List[Dict[str, str]], e.g. [{'generated_text': '...'} ]
        # Others might return str directly. Standardize or document.
        try:
            # Run in executor as pipeline call can be blocking
            generated_output = await loop.run_in_executor(
                self.executor, lambda: pipeline(prompt, **generation_kwargs)
            )
            logger.debug(
                f"Text generation output for prompt '{prompt[:50]}...': {generated_output}"
            )
            return generated_output
        except Exception as e:
            logger.error(
                f"Text generation error for model {model_name}: {e}", exc_info=True
            )
            raise

    # FE004: Concrete Embedding Task Implementation
    async def _embedding_task(
        self, task_data: Dict
    ) -> List[List[float]]:  # Return list of embeddings (list of floats)
        if not (HAVE_TRANSFORMERS and HAVE_TORCH and torch is not None):
            raise RuntimeError(
                "Embedding generation requires PyTorch and Transformers to be installed."
            )

        model_name = task_data["model_name"]
        texts = task_data["texts"]  # Expect a list of strings or a single string
        pooling_strategy = task_data.get(
            "pooling_strategy", "mean"
        ).lower()  # mean, cls, etc.
        normalize_embeddings = task_data.get("normalize_embeddings", True)

        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError(
                "Input for embedding task must be a string or list of strings."
            )

        # Ensure model (AutoModel + AutoTokenizer) is loaded
        model_load_params = {
            "model_type": "transformers",  # Not a pipeline task, but raw model/tokenizer
            "torch_dtype": task_data.get("torch_dtype"),
        }
        model_artifacts = await self._ensure_model_loaded(model_name, model_load_params)

        if not (
            isinstance(model_artifacts, dict)
            and "model" in model_artifacts
            and "tokenizer" in model_artifacts
        ):
            raise ValueError(
                f"Model {model_name} did not load correctly as model/tokenizer pair for embeddings."
            )

        model = model_artifacts["model"]
        tokenizer = model_artifacts["tokenizer"]

        loop = asyncio.get_running_loop()

        try:
            # This whole block is CPU/GPU bound, so run in executor
            def _generate_embeddings_blocking():
                # Tokenize
                # Consider tokenizer kwargs: padding, truncation, return_tensors, max_length
                encoded_input = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

                # Compute token embeddings
                with torch.no_grad():  # type: ignore
                    model_output = model(**encoded_input)

                # Perform pooling
                # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded_input[
                    "attention_mask"
                ]  # (batch_size, sequence_length)

                if pooling_strategy == "mean":
                    if attention_mask is None:
                        raise ValueError(
                            "attention_mask is None; cannot perform mean pooling."
                        )
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    # Defensive: ensure torch is available
                    if torch is None:
                        raise RuntimeError(
                            "PyTorch (torch) is None; cannot perform sum for pooling."
                        )
                    sum_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, 1
                    )  # type: ignore
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # type: ignore
                    sentence_embeddings = sum_embeddings / sum_mask
                elif pooling_strategy == "cls":
                    sentence_embeddings = token_embeddings[:, 0]  # CLS token embedding
                else:
                    raise ValueError(
                        f"Unsupported pooling strategy: {pooling_strategy}"
                    )

                if normalize_embeddings:
                    if torch is None or getattr(torch, "nn", None) is None:
                        raise RuntimeError(
                            "PyTorch (torch.nn) is None; cannot normalize embeddings."
                        )
                    sentence_embeddings = torch.nn.functional.normalize(
                        sentence_embeddings, p=2, dim=1
                    )  # type: ignore

                return (
                    sentence_embeddings.cpu().tolist()
                )  # Return as list of lists of floats

            list_of_embeddings = await loop.run_in_executor(
                self.executor, _generate_embeddings_blocking
            )
            logger.debug(
                f"Generated {len(list_of_embeddings)} embeddings using model {model_name}."
            )
            return list_of_embeddings

        except Exception as e:
            logger.error(
                f"Embedding generation error for model {model_name}: {e}", exc_info=True
            )
            raise

    async def _custom_task(self, task_data: Dict) -> Any:
        """Execute custom processing task"""
        custom_function = task_data["function"]
        args = task_data.get("args", [])
        kwargs = task_data.get("kwargs", {})

        loop = asyncio.get_running_loop()

        if asyncio.iscoroutinefunction(custom_function):
            # Directly await if the custom function is already async
            return await custom_function(*args, **kwargs)
        else:
            # Run sync custom function in the thread pool executor
            return await loop.run_in_executor(
                self.executor, lambda: custom_function(*args, **kwargs)
            )

    # FE002: Refactor Repetitive asyncio.Future Logic - Helper
    async def _submit_task_and_await_future(self, task_type: str, **task_kwargs) -> Any:
        """Helper to submit a task and await its result via an asyncio.Future."""
        if not self.is_initialized:
            raise RuntimeError(
                "Processing engine not initialized. Call initialize() first."
            )
        if self.shutting_down:
            raise RuntimeError(
                "Processing engine is shutting down. No new tasks accepted."
            )

        result_future = asyncio.Future()
        task_id = f"{task_type}-{time.monotonic_ns()}"  # Simple unique ID

        def callback(result, error):
            if error:
                # Ensure the exception is set in the context of the future's loop
                if result_future.get_loop().is_running():
                    result_future.set_exception(error)
                else:  # Fallback if loop is not running (e.g. during shutdown sequence)
                    logger.error(
                        f"Event loop for future not running when setting exception for task {task_id}"
                    )

            else:
                if result_future.get_loop().is_running():
                    result_future.set_result(result)
                else:
                    logger.error(
                        f"Event loop for future not running when setting result for task {task_id}"
                    )

        await self.submit_task(
            task_type, callback=callback, task_id=task_id, **task_kwargs
        )
        return await result_future

    async def submit_task(
        self,
        task_type: str,
        callback: Optional[Callable] = None,
        priority: int = 5,
        **task_kwargs,
    ) -> None:
        """
        Submit a task for processing.
        Priority: Lower numbers are higher priority (e.g., 1 is higher than 5).
        """
        if not self.is_initialized:
            # Allow task submission even if not fully initialized, they will queue.
            # However, workers might not be running yet. Initialize() should be called.
            logger.warning(
                "Processing engine not fully initialized. Task submitted to queue, but ensure initialize() is called."
            )
        if self.shutting_down:
            logger.error(
                "Attempted to submit task while engine is shutting down. Task rejected."
            )
            if callback:  # Notify callback about rejection if possible
                await self._run_callback(
                    callback, None, RuntimeError("Engine shutting down, task rejected.")
                )
            return

        task_id = task_kwargs.pop("task_id", f"{task_type}-{time.monotonic_ns()}")
        task_data = {
            "type": task_type,
            "task_id": task_id,
            "callback": callback,
            **task_kwargs,
        }

        # FE005: Use priority queue - tasks are (priority, data)
        await self.processing_queue.put((priority, task_data))
        self._update_task_stats(task_type, "submitted")

        self.vanta_core.publish_event(
            "processor.task.submitted",
            {"task_type": task_type, "task_id": task_id, "priority": priority},
            source="AsyncProcessingEngine",
        )

    # Public methods refactored to use _submit_task_and_await_future (FE002)
    async def load_model(
        self,
        model_name: str,
        model_type: str = "auto",
        custom_loader: Optional[Callable] = None,
        pipeline_task: Optional[str] = None,  # For loading HF pipelines
        torch_dtype: Optional[str] = None,  # Per-call dtype override
        priority: int = 2,  # Higher priority for model loading
    ) -> Any:
        """Load a model and return when complete"""
        return await self._submit_task_and_await_future(
            "load_model",
            model_name=model_name,
            model_type=model_type,
            custom_loader=custom_loader,
            pipeline_task=pipeline_task,
            torch_dtype=torch_dtype,
            priority=priority,
        )

    async def unload_model(self, model_name: str, priority: int = 2) -> Dict[str, Any]:
        """Explicitly unload a model."""
        return await self._submit_task_and_await_future(
            "unload_model",
            model_name=model_name,
            priority=priority,
        )

    async def run_inference(
        self,
        model_name: str,
        inputs: Any,
        model_type: str = "auto",  # If model needs loading
        custom_inference: Optional[Callable] = None,
        torch_dtype: Optional[str] = None,  # For model loading if needed
        timeout: Optional[float] = None,  # Per-task timeout FE008
        priority: int = 5,
    ) -> Any:
        """Run inference and return result"""
        return await self._submit_task_and_await_future(
            "inference",
            model_name=model_name,
            inputs=inputs,
            model_type=model_type,
            custom_inference=custom_inference,
            torch_dtype=torch_dtype,
            timeout=timeout,
            priority=priority,
        )

    async def generate_text(
        self,
        model_name: str,
        prompt: Union[str, List[str]],
        generation_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # Replaces max_length, temperature etc.
        torch_dtype: Optional[str] = None,  # For model loading if needed
        timeout: Optional[float] = None,
        priority: int = 5,
    ) -> Any:  # FE003
        """Generate text response. `generation_kwargs` are passed to the HF pipeline."""
        _generation_kwargs = generation_kwargs or {}
        return await self._submit_task_and_await_future(
            "text_generation",
            model_name=model_name,
            prompt=prompt,
            generation_kwargs=_generation_kwargs,
            torch_dtype=torch_dtype,
            timeout=timeout,
            priority=priority,
        )

    async def get_embeddings(
        self,
        model_name: str,
        texts: Union[str, List[str]],
        pooling_strategy: str = "mean",
        normalize_embeddings: bool = True,
        torch_dtype: Optional[str] = None,  # For model loading if needed
        timeout: Optional[float] = None,
        priority: int = 6,  # Typically lower priority
    ) -> List[List[float]]:  # FE004
        """Get text embeddings."""
        return await self._submit_task_and_await_future(
            "embedding",
            model_name=model_name,
            texts=texts,
            pooling_strategy=pooling_strategy,
            normalize_embeddings=normalize_embeddings,
            torch_dtype=torch_dtype,
            timeout=timeout,
            priority=priority,
        )

    async def execute_custom(
        self,
        function: Callable,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        priority: int = 5,
    ) -> Any:
        """Execute custom function asynchronously."""
        return await self._submit_task_and_await_future(
            "custom",
            function=function,
            args=args or [],
            kwargs=kwargs or {},
            timeout=timeout,
            priority=priority,
        )

    # FE010: Detailed Statistics
    def _update_task_stats(
        self, task_type: str, status: str, duration: Optional[float] = None
    ):
        with self._stats_lock:
            self._stats[f"tasks_{status}"] = self._stats.get(f"tasks_{status}", 0) + 1

            if task_type not in self._stats["task_details"]:
                self._stats["task_details"][task_type] = {
                    "submitted": 0,
                    "completed": 0,
                    "failed": 0,
                    "timed_out": 0,
                    "total_processing_time": 0.0,
                    "count_for_avg_time": 0,
                }

            detail = self._stats["task_details"][task_type]
            detail[status] = detail.get(status, 0) + 1

            if (
                status in ["completed", "failed"] and duration is not None
            ):  # Timed_out duration is task_timeout itself
                detail["total_processing_time"] += duration
                detail["count_for_avg_time"] += 1
            elif (
                status == "timed_out" and duration is not None
            ):  # duration here is the timeout value
                # We don't add timeout value to processing time, as it didn't complete
                pass

    def _update_model_stats(
        self,
        model_name: str,
        event: str,
        duration: Optional[float] = None,
        error: Optional[str] = None,
    ):
        with self._stats_lock:
            event_data = {
                "model_name": model_name,
                "event": event,
                "timestamp": time.time(),
            }
            if duration is not None:
                event_data["duration_seconds"] = duration
            if error is not None:
                event_data["error"] = error
            self._stats["model_load_events"].append(event_data)

    async def get_stats(self) -> Dict:
        """Get processing engine statistics (FE010 enhanced)"""
        with self._stats_lock:  # Ensure consistent read of stats
            # Calculate average times
            for task_type, details in self._stats["task_details"].items():
                if details["count_for_avg_time"] > 0:
                    details["avg_processing_time_seconds"] = (
                        details["total_processing_time"] / details["count_for_avg_time"]
                    )
                else:
                    details["avg_processing_time_seconds"] = 0

            # For priority queue, qsize is not always reflective of actual items if complex objects are stored directly
            # and tasks are tuples (priority, data). The count is accurate though.
            # Create a copy to avoid returning internal mutable object
            current_stats = {
                "device": self.device,
                "loaded_models_count": len(self.loaded_models),
                "loaded_models_lru_order": list(self.model_lru_order),  # List copy
                "queue_size": self.processing_queue.qsize(),
                "worker_count": len(self.worker_tasks),
                "active_workers": sum(1 for t in self.worker_tasks if not t.done()),
                "is_initialized": self.is_initialized,
                "is_shutting_down": self.shutting_down,
                "torch_available": HAVE_TORCH,
                "transformers_available": HAVE_TRANSFORMERS,
                "config": self.config.__dict__,  # Or use Pydantic's .model_dump()
                **{
                    k: v for k, v in self._stats.items() if k != "model_load_events"
                },  # Exclude raw events list by default
            }
            # Make a deepcopy of task_details if it's to be modified by caller, or return as is for read-only.
            current_stats["task_details"] = copy.deepcopy(self._stats["task_details"])
        return current_stats

    async def get_model_load_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent model loading/unloading events."""
        with self._stats_lock:
            return list(self._stats["model_load_events"][-limit:])

    # FE007: Improved Graceful Shutdown
    async def shutdown(self, graceful_timeout: float = 10.0):
        """Shutdown the processing engine gracefully."""
        if not self.is_initialized and not self.shutting_down:
            logger.info(
                "Processing Engine not initialized or already shut down. Nothing to do."
            )
            return
        if self.shutting_down:
            logger.warning("Processing Engine shutdown already in progress.")
            return

        logger.info(
            f"Shutting down Processing Engine (graceful timeout: {graceful_timeout}s)..."
        )
        self.shutting_down = (
            True  # Signal workers to stop accepting new tasks from queue perspective
        )
        # and to exit their loops once current work + queue is clear (or timeout)

        # 1. Stop accepting new tasks (already handled by self.shutting_down flag in submit_task)

        # 2. Wait for existing tasks in the queue to be processed by workers
        #    or for a timeout.
        if not self.processing_queue.empty():
            logger.info(
                f"Waiting for {self.processing_queue.qsize()} tasks in queue to be processed..."
            )
            try:
                await asyncio.wait_for(
                    self.processing_queue.join(), timeout=graceful_timeout
                )
                logger.info("All tasks from queue processed.")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout waiting for queue to empty. {self.processing_queue.qsize()} tasks may remain unprocessed."
                )
                # Tasks remaining in queue will be lost as workers will be cancelled next.
            except Exception as e:
                logger.error(f"Error joining processing queue: {e}", exc_info=True)
        else:
            logger.info("Processing queue is empty.")

        # 3. Cancel worker tasks. This will interrupt them even if they are in the middle of _execute_task.
        #    The _execute_task itself should ideally handle asyncio.CancelledError if it has long internal awaits.
        logger.info(f"Cancelling {len(self.worker_tasks)} worker tasks...")
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()

        # 4. Wait for worker tasks to complete (i.e., acknowledge cancellation and exit).
        if self.worker_tasks:
            # Setting return_exceptions=True ensures that if a worker raises an error
            # (other than CancelledError) during its shutdown, gather doesn't immediately fail.
            results = await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, asyncio.CancelledError):
                    logger.info(f"Worker task {i} cancelled during shutdown.")
                elif isinstance(res, Exception):
                    logger.error(
                        f"Worker task {i} raised exception during shutdown: {res}"
                    )
                else:
                    logger.info(f"Worker task {i} exited cleanly.")
        self.worker_tasks.clear()

        # 5. Shutdown thread pool executor
        logger.info("Shutting down thread pool executor...")
        self.executor.shutdown(wait=True)
        logger.info("Thread pool executor shut down.")

        # 6. Clear loaded models (optional, depends on desired state post-shutdown)
        with self.model_lock:
            if self.loaded_models:
                logger.info(
                    f"Clearing {len(self.loaded_models)} loaded models from memory."
                )
                self.loaded_models.clear()
                self.model_lru_order.clear()

        self.is_initialized = False  # Mark as fully shut down
        # self.shutting_down remains True

        self.vanta_core.publish_event(
            "processor.engine.shutdown",
            {
                "graceful_timeout_hit": graceful_timeout <= 0
                if "await asyncio.wait_for" in locals() and "TimeoutError" in locals()
                else False
            },
            source="AsyncProcessingEngine",
        )
        logger.info("Processing Engine shutdown complete.")

    def get_health_status(self) -> dict:
        """Return health status of the processing engine."""
        return {
            "component": self.COMPONENT_NAME,
            "status": "healthy"
            if self.is_initialized and not self.shutting_down
            else "degraded",
            "device": str(self.device),
            "stats": self._stats.copy(),
        }


# Example Usage (Conceptual - requires a mock vanta_core and actual models)
async def main():
    # Mock VantaCore for event publishing
    class MockVantaCore:
        def publish_event(self, event_name: str, data: Dict, source: str):
            print(f"EVENT: [{source}] {event_name} -> {data}")

    vanta_core_mock = MockVantaCore()

    # Configure logger for Vanta.AsyncProcessor
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # You might want to set a more specific logger for the engine if it's noisy at INFO
    # logging.getLogger("Vanta.AsyncProcessor").setLevel(logging.DEBUG)

    # FE001: Pydantic would validate here. For now, direct ProcessorConfig.
    config = ProcessorConfig(
        max_workers=2,
        device="auto",  # Let it pick cuda, mps, or cpu
        model_cache_dir="./vanta_models_cache",
        inference_timeout=25.0,
        enable_model_offloading=True,
        max_loaded_models=1,  # Test LRU with a small cache
        default_torch_dtype="float16"
        if HAVE_TORCH and torch is not None and torch.cuda.is_available()
        else None,
    )

    engine = AsyncProcessingEngine(vanta_core_mock, config)

    if not await engine.initialize():
        print("Failed to initialize engine. Exiting.")
        return

    # --- Example Tasks (some require specific models to be available) ---
    try:
        # 1. Load a sentence transformer model for embeddings (FE004)
        # Replace with a small, fast model for testing if possible
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if HAVE_TRANSFORMERS and HAVE_TORCH:
            print(f"\nAttempting to load embedding model: {embedding_model_name}")
            try:
                await engine.load_model(
                    embedding_model_name, model_type="transformers", priority=1
                )
                print(f"'{embedding_model_name}' loaded.")

                # Get embeddings
                embeddings = await engine.get_embeddings(
                    embedding_model_name,
                    texts=["Hello world from Vanta!", "Async processing is cool."],
                    priority=3,
                )
                print(
                    f"Embeddings for '{embedding_model_name}' (first vector, first 5 dims): {[f'{x:.4f}' for x in embeddings[0][:5]]}..."
                )
                print(f"Number of embedding vectors: {len(embeddings)}")

            except Exception as e:
                print(f"Error with embedding model '{embedding_model_name}': {e}")
        else:
            print("Skipping embedding task: Transformers or PyTorch not available.")

        # 2. Load a text generation model (FE003)
        # Replace with a very small/fast text generation model for testing
        # e.g., "sshleifer/tiny-gpt2" or "distilgpt2" might be too big for quick CI tests
        # For a truly tiny model, one might need to be created/mocked.
        # Using a placeholder name that likely won't download for now if not present.
        text_gen_model_name = "gpt2"  # "sshleifer/tiny-gpt2" is a small alternative
        if HAVE_TRANSFORMERS and HAVE_TORCH:
            print(
                f"\nAttempting to load text generation model: {text_gen_model_name} (will trigger LRU if cache is full)"
            )
            try:
                # This load might trigger eviction of the embedding model if max_loaded_models=1
                await engine.load_model(
                    text_gen_model_name,
                    model_type="transformers",
                    pipeline_task="text-generation",
                    priority=1,
                )
                print(f"'{text_gen_model_name}' loaded.")

                generated_texts = await engine.generate_text(
                    text_gen_model_name,
                    prompt="Once upon a time in an async world",
                    generation_kwargs={
                        "max_new_tokens": 20,
                        "num_return_sequences": 1,
                    },  # Use max_new_tokens for pipelines
                    priority=2,
                )
                print(
                    f"Generated text from '{text_gen_model_name}': {generated_texts[0]['generated_text'] if generated_texts else 'N/A'}"
                )

                # Test LRU: try to load the first model again
                if (
                    config.max_loaded_models == 1
                    and embedding_model_name != text_gen_model_name
                ):
                    print(
                        f"\nAttempting to reload embedding model '{embedding_model_name}' to test LRU..."
                    )
                    await engine.load_model(
                        embedding_model_name, model_type="transformers", priority=1
                    )
                    print(
                        f"'{embedding_model_name}' reloaded (text-gen model should have been evicted)."
                    )

            except Exception as e:
                print(f"Error with text generation model '{text_gen_model_name}': {e}")
        else:
            print(
                "Skipping text generation task: Transformers or PyTorch not available."
            )

        # 3. Custom task example
        def my_sync_task(x, y):
            logger.info(f"Custom sync task running with {x}, {y}")
            time.sleep(0.1)  # Simulate work
            return x + y

        print("\nSubmitting custom synchronous task...")
        custom_result = await engine.execute_custom(
            my_sync_task, args=[10, 20], priority=4
        )
        print(f"Custom task result: {custom_result}")

        async def my_async_task(name):
            logger.info(f"Custom async task running for {name}")
            await asyncio.sleep(0.05)  # Simulate async work
            return f"Hello, {name} from async custom task!"

        print("\nSubmitting custom asynchronous task...")
        custom_async_result = await engine.execute_custom(
            my_async_task, args=["Async Vanta"], priority=4
        )
        print(f"Custom async task result: {custom_async_result}")

        # 4. Test per-task timeout (FE008)
        print("\nSubmitting a task designed to timeout...")
        try:
            await engine.execute_custom(
                time.sleep, args=[2], timeout=0.5, priority=1
            )  # Should timeout
            print("Timeout task completed without error (UNEXPECTED)")
        except asyncio.TimeoutError:
            print(
                "Timeout task correctly raised asyncio.TimeoutError through the future."
            )
        except Exception as e:
            print(f"Timeout task failed with unexpected error: {e}")

    except RuntimeError as e:
        print(f"Runtime Error during task submission or execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Display stats
        print("\nFinal Engine Stats:")
        stats = await engine.get_stats()
        # Pretty print stats or relevant parts
        print(f"  Queue Size: {stats['queue_size']}")
        print(f"  Tasks Submitted: {stats['tasks_submitted']}")
        print(f"  Tasks Completed: {stats['tasks_completed']}")
        print(f"  Tasks Failed: {stats['tasks_failed']}")
        print(f"  Tasks Timed Out: {stats['tasks_timed_out']}")
        print(f"  Loaded Models Count: {stats['loaded_models_count']}")
        print(f"  LRU Order: {stats['loaded_models_lru_order']}")
        # print(f"  Task Details: {stats['task_details']}") # Can be verbose

        model_history = await engine.get_model_load_history(5)
        print(f"  Recent Model History (last 5): {model_history}")

        # Shutdown
        print("\nShutting down engine...")
        await engine.shutdown(
            graceful_timeout=5.0
        )  # Allow 5s for graceful queue processing
        print("Engine shutdown process initiated.")


if __name__ == "__main__":
    # Note: Running Hugging Face model downloads/inference directly like this
    # might be slow on first run or without appropriate small models.
    # This main() is for demonstration; real usage would involve a persistent VantaCore.

    # To suppress voluminous Hugging Face downloader logs for cleaner output during demo:
    # from transformers.utils import logging as hf_logging
    # hf_logging.set_verbosity_error() # or WARNING

    # Python 3.7+ for asyncio.create_task, 3.8+ for asyncio.get_running_loop often
    # This script uses features generally compatible with Python 3.8+

    # FE001 (Pydantic) is not fully integrated in this generated script's ProcessorConfig
    # but its principles (validation, defaults) are partially emulated or noted.
    # A full Pydantic integration would involve defining ProcessorConfig as a BaseModel.

    asyncio.run(main())
