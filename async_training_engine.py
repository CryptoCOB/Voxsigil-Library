# enhanced_training_script.py
"""Async Training Engine for Vanta (Enhanced & Fixed).

Handles model training, fine-tuning, and learning tasks asynchronously.
"""

# pylint: disable=import-error

import asyncio
import logging
import threading
import time
import traceback  # TRFE008
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union
import uuid

# ML Dependencies


from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import confloat, conint
import torch
import torch.optim as optim
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import confloat, conint
from torch.cuda.amp import GradScaler, autocast  # TRFE004
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TRFE009


logger = logging.getLogger("Vanta.AsyncTraining")

# Use Pydantic v2 constraints
Gt = lambda x: confloat(gt=x)
Ge = lambda x: conint(ge=x)


# Attempt to import transformers and datasets (TRFE001, TRFE002)
try:
    import copy
    import json

    import transformers
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import (
        AutoModelForSequenceClassification,
    )
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.optimization import (
        get_cosine_schedule_with_warmup as hf_get_cosine_schedule_with_warmup,
    )
    from transformers.optimization import (
        get_linear_schedule_with_warmup as hf_get_linear_schedule_with_warmup,
    )

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    transformers = None
    AutoConfig = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    if transformers is not None:
        from transformers.models.auto.modeling_auto import (
            AutoModelForSequenceClassification,
        )
    if transformers is not None:
        from transformers.models.auto.modeling_auto import (
            AutoModelForSequenceClassification,
        )
    AutoTokenizer = None  # type: ignore
    hf_get_linear_schedule_with_warmup = None  # type: ignore
    hf_get_cosine_schedule_with_warmup = None  # type: ignore

# Make schedulers accessible via a consistent name, falling back to direct transformers access if needed
if hf_get_linear_schedule_with_warmup is not None:
    get_linear_schedule_with_warmup = hf_get_linear_schedule_with_warmup
elif transformers is not None and hasattr(
    transformers, "get_linear_schedule_with_warmup"
):
    get_linear_schedule_with_warmup = transformers.get_linear_schedule_with_warmup  # type: ignore
else:
    get_linear_schedule_with_warmup = None

if hf_get_cosine_schedule_with_warmup is not None:
    get_cosine_schedule_with_warmup = hf_get_cosine_schedule_with_warmup
elif transformers is not None and hasattr(
    transformers, "get_cosine_schedule_with_warmup"
):
    get_cosine_schedule_with_warmup = transformers.get_cosine_schedule_with_warmup  # type: ignore
else:
    get_cosine_schedule_with_warmup = None


try:
    import datasets as hf_datasets  # TRFE002
    from datasets import Dataset as HFDataset  # For isinstance check
    from datasets.dataset_dict import (
        DatasetDict as HFDatasetDict,  # For isinstance check
    )
    from datasets.iterable_dataset import (
        IterableDataset as HFIterableDataset,  # For isinstance check
    )

    HAVE_HF_DATASETS = True
except ImportError:
    HAVE_HF_DATASETS = False
    hf_datasets = None  # type: ignore
    HFIterableDataset = None  # type: ignore
    HFDatasetDict = None  # type: ignore
    HFDataset = None  # type: ignore


# TRFE003: Pydantic Models
class TrainingConfig(BaseModel):
    output_dir: Path = Field(default_factory=lambda: Path("./training_outputs"))
    checkpoint_dir: Path = Field(default_factory=lambda: Path("./checkpoints"))
    max_epochs: Annotated[int, Gt(0)] = Field(default=10)
    batch_size: Annotated[int, Gt(0)] = Field(default=8)
    learning_rate: Annotated[float, Gt(0)] = Field(default=1e-4)

    save_steps: int = Field(
        default=500, ge=0
    )  # 0 means save only at end of epoch potentially
    eval_steps: int = Field(default=100, ge=0)  # 0 means eval only at end of epoch

    warmup_steps: int = Field(default=100, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: Annotated[float, Gt(0)] = Field(default=1.0)
    device: str = Field(default="auto")
    dataset_text_column: str = Field(default="text")
    dataset_label_column: str = Field(default="label")
    dataset_train_split_name: str = Field(default="train")
    dataset_eval_split_name: Optional[str] = Field(default="validation")  # For TRFE006
    max_seq_length: Annotated[int, Gt(0)] = Field(default=128)

    # TRFE005 related
    scheduler_type: str = Field(
        default="linear", pattern="^(linear|cosine|none)$"
    )  # Example schedulers
    scheduler_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # TRFE007 related
    resume_from_checkpoint: Optional[Union[str, Path]] = Field(default=None)

    # TRFE009 related
    tensorboard_log_dir: Optional[Path] = Field(default=None)

    # New fields
    mixed_precision: bool = Field(
        default=False, description="Enable mixed precision training (AMP)"
    )
    dataloader_num_workers: int = Field(
        default=0, ge=0, description="Number of DataLoader workers"
    )

    @field_validator(
        "output_dir", "checkpoint_dir", "tensorboard_log_dir", mode="before"
    )
    @classmethod
    def _resolve_path(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).resolve()

    @model_validator(mode="after")
    def _validate_paths_exist_or_creatable(self) -> "TrainingConfig":
        for path_field in ["output_dir", "checkpoint_dir"]:
            path_val = getattr(self, path_field)
            if path_val:
                pass  # Could add writability checks here
        if self.tensorboard_log_dir is None and self.output_dir:
            self.tensorboard_log_dir = self.output_dir / "runs"
        return self

    def model_dump(self, *args, **kwargs):
        """Dump the model to a dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "device": self.device,
            "dataset_text_column": self.dataset_text_column,
            "dataset_label_column": self.dataset_label_column,
            "dataset_train_split_name": self.dataset_train_split_name,
            "dataset_eval_split_name": self.dataset_eval_split_name,
            "max_seq_length": self.max_seq_length,
            "scheduler_type": self.scheduler_type,
            "scheduler_kwargs": self.scheduler_kwargs,
            "resume_from_checkpoint": str(self.resume_from_checkpoint)
            if self.resume_from_checkpoint
            else None,
            "tensorboard_log_dir": str(self.tensorboard_log_dir)
            if self.tensorboard_log_dir
            else None,
            "mixed_precision": self.mixed_precision,
            "dataloader_num_workers": self.dataloader_num_workers,
        }

    def model_copy(self, deep=True):
        """Create a copy of the model."""
        if deep:
            return TrainingConfig(**copy.deepcopy(self.model_dump()))
        else:
            return TrainingConfig(**self.model_dump())

    def model_dump_json(self, *args, **kwargs):
        """Dump the model to a JSON string."""
        indent = kwargs.get("indent", None)
        return json.dumps(self.model_dump(), indent=indent)


class TrainingJob(BaseModel):  # TRFE003
    job_id: str
    model_name_or_path: str  # TRFE001: Changed from model_name
    dataset_name_or_path: str  # TRFE002: Changed from dataset_path
    config: TrainingConfig  # Uses the Pydantic TrainingConfig

    status: str = (
        "pending"  # pending, running, completed, failed, paused, stopping, stopped
    )
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_epoch: int = Field(0, ge=0)
    total_steps: int = Field(0, ge=0)  # Will be calculated
    current_step: int = Field(0, ge=0)
    loss: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[Dict[str, str]] = None  # TRFE008: For structured error

    # Use model_config for private attributes instead of field definitions
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize private attributes after model creation
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._optimizer: Optional[Any] = None
        self._scheduler: Optional[Any] = None
        self._train_dataloader: Optional[Any] = None
        self._eval_dataloader: Optional[Any] = None
        self._grad_scaler: Optional[Any] = None
        self._summary_writer: Optional[Any] = None
        self._loaded_checkpoint_path: Optional[Path] = None

    def model_dump(self, *args, **kwargs):
        raise NotImplementedError


class AsyncTrainingEngine:
    """Async Training Engine for model training and fine-tuning (Enhanced)"""

    COMPONENT_NAME = "async_training_engine"

    def __init__(
        self, vanta_core: Any, default_engine_config: Optional[TrainingConfig] = None
    ):
        self.vanta_core = vanta_core
        self.default_engine_config = default_engine_config or TrainingConfig()
        self.device = self._determine_device(self.default_engine_config.device)

        self.training_jobs: Dict[str, TrainingJob] = {}
        self.active_job_id: Optional[str] = None
        self.job_lock = threading.RLock()
        self.is_initialized = False
        self.training_task: Optional[asyncio.Task] = None

        if hasattr(self.vanta_core, "register_component"):
            self.vanta_core.register_component(
                self.COMPONENT_NAME,
                self,
                {
                    "type": "async_trainer",
                    "output_dir": str(self.default_engine_config.output_dir),
                },
            )
            logger.info(f"{self.COMPONENT_NAME} registered with VantaCore")
        else:
            logger.warning(
                "VantaCore does not have register_component method. Skipping registration."
            )

        if hasattr(self.vanta_core, "async_bus"):
            self.vanta_core.async_bus.register_component("training_engine")
            self.vanta_core.async_bus.subscribe(
                "training_engine",
                MessageType.PROCESSING_REQUEST,
                self.handle_training_request,
            )
            logger.info(
                "training_engine registered and subscribed to async bus (PROCESSING_REQUEST)"
            )

    def _determine_device(self, requested_device: str) -> str:
        logger.info(
            f"Device detection: requested_device={requested_device}, HAVE_TORCH={HAVE_TORCH}, torch={torch}"
        )

        if not HAVE_TORCH or torch is None:
            logger.warning("Torch not available. Falling back to CPU.")
            return "cpu"
        if requested_device != "auto":
            if requested_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA specified but not available. Falling back to CPU.")
                return "cpu"
            if requested_device == "mps":
                mps_available = False
                if hasattr(torch.backends, "mps"):
                    mps_backend = getattr(torch.backends, "mps", None)
                    if (
                        mps_backend is not None
                        and hasattr(mps_backend, "is_available")
                        and callable(mps_backend.is_available)  # type: ignore[truthy-function]
                    ):
                        mps_available = mps_backend.is_available()  # type: ignore[no-untyped-call]
                if not mps_available:
                    logger.warning(
                        "MPS specified but not available. Falling back to CPU for training (MPS support varies)."
                    )
                    return "cpu"
            return requested_device

        if torch.cuda.is_available():
            logger.info("CUDA detected and available for training")
            return "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and getattr(torch.backends.mps, "is_available", lambda: False)()  # type: ignore[no-untyped-call]
        ):
            logger.info(
                "MPS device detected. Using MPS for training can be experimental."
            )
            return "mps"
        return "cpu"

    async def initialize(self) -> bool:
        if self.is_initialized:
            logger.warning("Training Engine already initialized.")
            return True
        try:
            effective_config = self.default_engine_config
            logger.info(f"Initializing Training Engine on device: {self.device}")

            effective_config.output_dir.mkdir(parents=True, exist_ok=True)
            effective_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if effective_config.tensorboard_log_dir:
                effective_config.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

            self.is_initialized = True
            self.vanta_core.publish_event(
                "training.engine.initialized",
                {
                    "device": self.device,
                    "default_output_dir": str(effective_config.output_dir),
                    "torch_available": HAVE_TORCH,
                    "transformers_available": HAVE_TRANSFORMERS,
                    "hf_datasets_available": HAVE_HF_DATASETS,
                    "default_mixed_precision": effective_config.mixed_precision,
                },
                source="AsyncTrainingEngine",
            )
            logger.info("Training Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Training Engine: {e}", exc_info=True)
            self.is_initialized = False
            return False

    async def create_training_job(
        self,
        job_id: str,
        model_name_or_path: str,
        dataset_name_or_path: str,
        job_specific_config_dict: Optional[Dict[str, Any]] = None,
    ) -> TrainingJob:
        merged_config_data = self.default_engine_config.model_copy(
            deep=True
        ).model_dump()
        if job_specific_config_dict:
            merged_config_data.update(job_specific_config_dict)
        final_config = TrainingConfig(**merged_config_data)

        # Ensure job-specific paths are subdirectories of the main output/checkpoint dirs
        # and use the job_id to make them unique.
        final_config.output_dir = self.default_engine_config.output_dir / job_id
        final_config.output_dir.mkdir(parents=True, exist_ok=True)

        final_config.checkpoint_dir = self.default_engine_config.checkpoint_dir / job_id
        final_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if (
            final_config.tensorboard_log_dir
        ):  # If user specified one, make it job-specific
            # If the default (based on output_dir) was used, it will be updated correctly below
            # This assumes tensorboard_log_dir from config might be a general path
            final_config.tensorboard_log_dir = (
                final_config.tensorboard_log_dir / job_id
            )  # This line might need adjustment if tensorboard_log_dir is absolute
        elif (
            self.default_engine_config.tensorboard_log_dir
        ):  # If default engine config had one
            final_config.tensorboard_log_dir = (
                self.default_engine_config.tensorboard_log_dir / job_id
            )
        else:  # Fallback if no tensorboard log dir defined anywhere
            final_config.tensorboard_log_dir = (
                final_config.output_dir / "runs"
            )  # Default to job's output dir

        final_config.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        job = TrainingJob(
            job_id=job_id,
            model_name_or_path=model_name_or_path,
            dataset_name_or_path=dataset_name_or_path,
            config=final_config,
            progress=0.0,
            current_epoch=0,
            total_steps=0,
            current_step=0,
        )

        with self.job_lock:
            if job_id in self.training_jobs:
                raise ValueError(f"Job ID {job_id} already exists.")
            self.training_jobs[job_id] = job

        self.vanta_core.publish_event(
            "training.job.created",
            {
                "job_id": job_id,
                "model_name_or_path": model_name_or_path,
                "dataset_name_or_path": dataset_name_or_path,
                "config": final_config.model_dump(),
            },
            source="AsyncTrainingEngine",
        )
        logger.info(
            f"Created training job: {job_id} with config: {final_config.model_dump_json(indent=2)}"
        )
        return job

    async def handle_training_request(self, message):
        """Async bus handler to start a training job."""
        try:
            cfg = message.content.get("config") if hasattr(message, "content") else None
            model = message.content.get("model") if hasattr(message, "content") else None
            dataset = message.content.get("dataset") if hasattr(message, "content") else None
            job = await self.create_training_job(
                job_id=str(uuid.uuid4())[:8],
                model_name_or_path=model or "model",
                dataset_name_or_path=dataset or "dataset",
                job_specific_config_dict=cfg,
            )
            await self.start_training_job(job.job_id)
            await self.vanta_core.async_bus.publish(
                AsyncMessage(
                    MessageType.PROCESSING_RESPONSE,
                    self.COMPONENT_NAME,
                    {"job_id": job.job_id},
                    target_ids=[message.sender_id],
                )
            )
        except Exception as e:
            logger.error(f"handle_training_request error: {e}")

    async def start_training_job(self, job_id: str) -> bool:
        if not self.is_initialized:
            logger.error("Training engine not initialized")
            return False  # Return bool as per type hint, not raise

        with self.job_lock:
            if job_id not in self.training_jobs:
                logger.error(f"Training job {job_id} not found")
                return False

            job = self.training_jobs[job_id]
            if job.status not in ["pending", "failed", "completed", "stopped"]:
                logger.error(
                    f"Job {job_id} is in status '{job.status}' and cannot be started."
                )
                return False

            if self.active_job_id is not None and self.active_job_id != job_id:
                current_active_job = self.training_jobs.get(self.active_job_id)
                if current_active_job and current_active_job.status == "running":
                    logger.error(
                        f"Another training job {self.active_job_id} is already running."
                    )
                    return False

            job.status = "pending"
            job.progress = 0.0
            job.current_epoch = 0
            job.current_step = 0
            job.total_steps = 0  # Recalculated in _prepare_training_artifacts
            job.loss = None
            job.metrics = {}
            job.error = None
            job.start_time = time.time()
            job.end_time = None
            job._loaded_checkpoint_path = None  # Reset this

            self.active_job_id = job_id

        if self.training_task and not self.training_task.done():
            logger.info("Cancelling previous training task.")
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                logger.info("Previous training task cancelled.")
            except Exception as e:
                logger.warning(f"Error awaiting previous task cancellation: {e}")
            self.training_task = None

        self.training_task = asyncio.create_task(self._run_training_job(job))
        logger.info(f"Training job {job_id} submitted to run.")
        return True

    async def _run_training_job(self, job: TrainingJob):
        loop = asyncio.get_event_loop()
        try:
            with self.job_lock:
                job.status = "running"

            self.vanta_core.publish_event(
                "training.job.started",
                {"job_id": job.job_id},
                source="AsyncTrainingEngine",
            )
            logger.info(f"Running training job: {job.job_id} on device: {self.device}")

            if (
                HAVE_TORCH
                and SummaryWriter is not None
                and job.config.tensorboard_log_dir
            ):
                try:
                    job._summary_writer = await loop.run_in_executor(
                        None,
                        lambda: SummaryWriter(
                            log_dir=str(job.config.tensorboard_log_dir)
                        )
                        if SummaryWriter is not None
                        else None,
                    )
                    if job._summary_writer:
                        logger.info(
                            f"TensorBoard logging to: {job.config.tensorboard_log_dir}"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize TensorBoard SummaryWriter: {e}",
                        exc_info=True,
                    )
                    job._summary_writer = None

            if (
                job.config.mixed_precision
                and self.device == "cuda"
                and HAVE_TORCH
                and GradScaler
            ):
                job._grad_scaler = await loop.run_in_executor(None, GradScaler)
                if job._grad_scaler:
                    logger.info("Mixed precision (AMP) enabled with GradScaler.")

            await self._prepare_training_artifacts(job)
            await self._setup_optimizer_and_scheduler(job)
            await self._training_loop(job)

            with self.job_lock:
                if job.status == "running":
                    job.status = "completed"
                job.end_time = time.time()
                job.progress = 1.0
                if self.active_job_id == job.job_id:
                    self.active_job_id = None

            self.vanta_core.publish_event(
                "training.job.completed",
                {
                    "job_id": job.job_id,
                    "final_loss": job.loss,
                    "metrics": job.metrics,
                    "duration": (job.end_time - job.start_time)
                    if job.start_time and job.end_time
                    else None,
                },
                source="AsyncTrainingEngine",
            )
            logger.info(f"Training job completed: {job.job_id}")

        except asyncio.CancelledError:
            logger.info(f"Training job {job.job_id} was cancelled.")
            with self.job_lock:
                job.status = "stopped"
                job.error = {
                    "type": "CancelledError",
                    "message": "Job was cancelled.",
                    "traceback": "",
                }
                job.end_time = time.time()
                if self.active_job_id == job.job_id:
                    self.active_job_id = None
            self.vanta_core.publish_event(
                "training.job.stopped",
                {"job_id": job.job_id, "reason": "cancelled"},
                source="AsyncTrainingEngine",
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Training job failed: {job.job_id}, error: {e}\n{tb_str}")
            with self.job_lock:
                job.status = "failed"
                job.error = {
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": tb_str,
                }
                job.end_time = time.time()
                if self.active_job_id == job.job_id:
                    self.active_job_id = None
            self.vanta_core.publish_event(
                "training.job.failed",
                {
                    "job_id": job.job_id,
                    "error_type": e.__class__.__name__,
                    "error_message": str(e),
                },
                source="AsyncTrainingEngine",
            )
        finally:
            if job._summary_writer:
                await loop.run_in_executor(None, job._summary_writer.close)
            job._model = None
            job._tokenizer = None
            job._optimizer = None
            job._scheduler = None
            job._train_dataloader = None
            job._eval_dataloader = None
            job._grad_scaler = None
            if HAVE_TORCH and torch and self.device == "cuda":
                try:
                    await loop.run_in_executor(None, torch.cuda.empty_cache)
                except Exception:
                    pass

    async def _prepare_training_artifacts(self, job: TrainingJob):
        logger.info(f"Preparing training artifacts for job: {job.job_id}")
        loop = asyncio.get_event_loop()

        if not (HAVE_TORCH and HAVE_TRANSFORMERS and HAVE_HF_DATASETS):
            raise RuntimeError("PyTorch, Transformers, and HF Datasets are required.")

        if AutoTokenizer is None:  # Explicit check for Pydantic
            raise RuntimeError("AutoTokenizer not available from Transformers library.")

        logger.info(f"Loading tokenizer: {job.model_name_or_path}")
        job._tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(job.model_name_or_path),  # type: ignore[no-untyped-call]
        )

        if hf_datasets is None:  # Explicit check for Pydantic
            raise RuntimeError(
                "HuggingFace datasets library (hf_datasets) not available."
            )

        logger.info(f"Loading dataset: {job.dataset_name_or_path}")
        raw_datasets = await loop.run_in_executor(
            None,
            lambda: hf_datasets.load_dataset(  # type: ignore[union-attr]
                job.dataset_name_or_path,
                split=None,
                cache_dir=str(job.config.output_dir / "datasets_cache"),  # L588 Fix
            ),
        )

        if raw_datasets is None:
            raise RuntimeError(f"Failed to load dataset: {job.dataset_name_or_path}")

        def preprocess_function(examples):
            if job._tokenizer is None:
                raise ValueError("Tokenizer is None. Ensure it's properly initialized.")
            # Ensure examples[job.config.dataset_text_column] is correctly formatted for tokenizer
            # E.g., it might be a list of strings, or a dict of lists if dataset is structured.
            # For simplicity, assuming it's directly passable.
            texts_to_tokenize = examples[job.config.dataset_text_column]

            result = job._tokenizer(
                texts_to_tokenize,  # Directly pass the content
                padding="max_length",
                max_length=job.config.max_seq_length,
                truncation=True,
            )
            return result

        logger.info(
            "Preprocessing datasets..."
        )  # L607 Syntax fix: moved from previous line
        tokenized_datasets = await loop.run_in_executor(
            None,
            lambda: raw_datasets.map(preprocess_function, batched=True),  # type: ignore[union-attr]
        )

        train_dataset = None
        eval_dataset = None

        # Robust split access
        if HFDatasetDict is not None and isinstance(tokenized_datasets, HFDatasetDict):
            if job.config.dataset_train_split_name in tokenized_datasets:
                train_dataset = tokenized_datasets[job.config.dataset_train_split_name]
            if (
                job.config.dataset_eval_split_name
                and job.config.dataset_eval_split_name in tokenized_datasets
            ):
                eval_dataset = tokenized_datasets[job.config.dataset_eval_split_name]
        elif HFDataset is not None and isinstance(
            tokenized_datasets, HFDataset
        ):  # If load_dataset returned a single split
            logger.warning(
                f"Loaded dataset is a single Dataset, not DatasetDict. Assuming it's for training: {job.config.dataset_train_split_name}"
            )
            train_dataset = tokenized_datasets
            # Try to split if eval name is provided and different
            if (
                job.config.dataset_eval_split_name
                and job.config.dataset_eval_split_name
                != job.config.dataset_train_split_name
            ):
                logger.warning(
                    "Cannot get eval_split from a single Dataset. Eval will be skipped unless dataset is split manually."
                )
        elif HFIterableDataset is not None and isinstance(
            tokenized_datasets, HFIterableDataset
        ):
            logger.warning(
                "Loaded dataset is an IterableDataset. Dictionary/attribute access for splits may not apply."
            )
            train_dataset = tokenized_datasets  # Assume it's the train split
        else:  # Fallback to attribute access for other types or if checks fail
            if hasattr(tokenized_datasets, job.config.dataset_train_split_name):
                train_dataset = getattr(
                    tokenized_datasets, job.config.dataset_train_split_name
                )
            if job.config.dataset_eval_split_name and hasattr(
                tokenized_datasets, job.config.dataset_eval_split_name
            ):
                eval_dataset = getattr(
                    tokenized_datasets, job.config.dataset_eval_split_name
                )

        if train_dataset is None:
            raise RuntimeError(
                f"Failed to access training split '{job.config.dataset_train_split_name}'"
            )
        logger.info(
            f"Successfully loaded training split: {job.config.dataset_train_split_name}"
        )
        if eval_dataset:
            logger.info(
                f"Successfully loaded evaluation split: {job.config.dataset_eval_split_name}"
            )
        else:
            logger.warning(
                f"No evaluation split found or accessible for '{job.config.dataset_eval_split_name}'"
            )

        if hasattr(train_dataset, "with_format") and callable(
            getattr(train_dataset, "with_format")
        ):
            train_dataset = await loop.run_in_executor(
                None,
                lambda: train_dataset.with_format("torch") if train_dataset else None,
            )  # type: ignore[union-attr, attr-defined]
        if (
            eval_dataset
            and hasattr(eval_dataset, "with_format")
            and callable(getattr(eval_dataset, "with_format"))
        ):
            eval_dataset = await loop.run_in_executor(
                None,
                lambda: eval_dataset.with_format("torch") if eval_dataset else None,
            )  # type: ignore[union-attr, attr-defined]

        if DataLoader is None:
            raise RuntimeError("PyTorch DataLoader not available.")

        job._train_dataloader = await loop.run_in_executor(
            None,
            lambda: DataLoader(
                train_dataset,  # type: ignore[arg-type]
                shuffle=not isinstance(
                    train_dataset,
                    HFIterableDataset if HFIterableDataset is not None else (),
                ),  # Cannot shuffle IterableDataset here
                batch_size=job.config.batch_size,
                num_workers=job.config.dataloader_num_workers,
            )
            if train_dataset
            else None,
        )
        if eval_dataset:
            job._eval_dataloader = await loop.run_in_executor(
                None,
                lambda: DataLoader(
                    eval_dataset,  # type: ignore[arg-type]
                    batch_size=job.config.batch_size,
                    num_workers=job.config.dataloader_num_workers,
                )
                if eval_dataset
                else None,
            )

        if job._train_dataloader and hasattr(job._train_dataloader, "__len__"):
            try:
                dataloader_length = len(job._train_dataloader)
                job.total_steps = (
                    dataloader_length * job.config.max_epochs
                ) // job.config.gradient_accumulation_steps
                logger.info(f"Total training steps: {job.total_steps}")
            except TypeError:  # IterableDataset may not have __len__
                logger.warning(
                    "Train DataLoader does not support __len__. Total steps cannot be pre-calculated accurately."
                )
                job.total_steps = -1  # Indicate unknown total steps
        else:
            logger.warning(
                "Train DataLoader not available or does not support __len__."
            )
            job.total_steps = -1

        num_labels = 2
        try:  # Determine num_labels carefully
            if (
                train_dataset is not None
                and hasattr(train_dataset, "features")
                and isinstance(getattr(train_dataset, "features", None), dict)
                and train_dataset.features is not None
                and job.config.dataset_label_column in train_dataset.features
            ):  # type: ignore[union-attr, operator]
                label_feature = train_dataset.features[job.config.dataset_label_column]  # type: ignore[union-attr, operator]
                if hasattr(label_feature, "num_classes"):
                    num_labels = label_feature.num_classes
                elif (
                    HFIterableDataset is not None
                    and not isinstance(train_dataset, HFIterableDataset)
                    and hasattr(train_dataset, "unique")
                    and callable(getattr(train_dataset, "unique"))
                ):  # L736
                    unique_labels = await loop.run_in_executor(
                        None,
                        lambda: train_dataset.unique(job.config.dataset_label_column),
                    )  # type: ignore[union-attr, attr-defined]
                    if unique_labels is not None:
                        num_labels = len(unique_labels)
            logger.info(f"Determined {num_labels} labels for classification task")
        except Exception as e:
            logger.warning(
                f"Could not determine num_labels from dataset, using default of {num_labels}: {e}"
            )

        if AutoConfig is None:
            raise RuntimeError("AutoConfig not available.")
        logger.info(f"Loading model config: {job.model_name_or_path}")
        model_config = await loop.run_in_executor(
            None,
            lambda: AutoConfig.from_pretrained(
                job.model_name_or_path, num_labels=num_labels
            )
            if AutoConfig is not None
            else (_ for _ in ()).throw(
                RuntimeError(
                    "AutoConfig is not available. Ensure the transformers library is installed."
                )
            ),
        )

        if AutoModelForSequenceClassification is None:
            raise RuntimeError("AutoModelForSequenceClassification not available.")
        logger.info(f"Loading model: {job.model_name_or_path}")
        job._model = await loop.run_in_executor(
            None,
            lambda: AutoModelForSequenceClassification.from_pretrained(
                job.model_name_or_path, config=model_config
            )
            if AutoModelForSequenceClassification is not None
            else (_ for _ in ()).throw(
                RuntimeError(
                    "AutoModelForSequenceClassification is not available. Ensure the transformers library is installed."
                )
            ),  # type: ignore[no-untyped-call]
        )

        # Corrected indentation for this block
        if not job._model:
            raise RuntimeError("Failed to load model: model is None")
        if not torch:
            raise RuntimeError("PyTorch not available. Install PyTorch.")

        if hasattr(job._model, "to"):
            await loop.run_in_executor(None, lambda: job._model.to(self.device))  # type: ignore[union-attr]
            logger.info(f"Model successfully loaded and moved to device: {self.device}")
        else:
            logger.warning("Model doesn't have 'to' method, cannot move to device")
        # The above try-except for model loading was not shown in user snippet, but should be here. Added.

        checkpoint_to_load_path: Optional[Path] = None
        if job.config.resume_from_checkpoint:
            if job.config.resume_from_checkpoint == "latest":
                chkpt_dir = Path(job.config.checkpoint_dir)
                # Ensure correct sorting for steps, e.g., step_9 vs step_100
                checkpoints = sorted(
                    [
                        p
                        for p in chkpt_dir.glob(f"{job.job_id}_step_*.pt")
                        if p.stem.split("_")[-1].isdigit()
                    ],
                    key=lambda f: int(f.stem.split("_")[-1]),
                    reverse=True,
                )
                if checkpoints:
                    checkpoint_to_load_path = checkpoints[0]
            else:
                checkpoint_to_load_path = Path(job.config.resume_from_checkpoint)

            if checkpoint_to_load_path and not checkpoint_to_load_path.exists():
                logger.warning(
                    f"Specified checkpoint {checkpoint_to_load_path} not found. Starting from scratch."
                )
                checkpoint_to_load_path = None

        if checkpoint_to_load_path and job._model:
            logger.info(
                f"Loading model state from checkpoint: {checkpoint_to_load_path}"
            )
            try:
                if torch is None:
                    raise RuntimeError(
                        "PyTorch (torch) is None, cannot load checkpoint."
                    )
                loaded_chkpt = await loop.run_in_executor(
                    None,
                    lambda: torch.load(
                        checkpoint_to_load_path, map_location=self.device
                    )
                    if HAVE_TORCH and torch and hasattr(torch, "load")
                    else (_ for _ in ()).throw(
                        RuntimeError(
                            "PyTorch (torch) is None or missing 'load', cannot load checkpoint."
                        )
                    ),  # type: ignore[union-attr]
                )

                if hasattr(job._model, "load_state_dict"):
                    await loop.run_in_executor(
                        None,
                        lambda: job._model.load_state_dict(
                            loaded_chkpt["model_state_dict"]
                        )
                        if job._model is not None
                        else None,
                    )  # type: ignore[union-attr]

                job.current_epoch = loaded_chkpt.get("epoch", 0)
                job.current_step = loaded_chkpt.get(
                    "global_step", loaded_chkpt.get("step", 0)
                )
                job.loss = loaded_chkpt.get("loss")
                job.metrics = loaded_chkpt.get("metrics", {})
                job._loaded_checkpoint_path = (
                    checkpoint_to_load_path  # Store for optimizer/scheduler
                )
                logger.info(
                    f"Resuming from epoch {job.current_epoch}, step {job.current_step}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load model from checkpoint {checkpoint_to_load_path}: {e}. Training from scratch.",
                    exc_info=True,
                )
                job._loaded_checkpoint_path = None

    async def _setup_optimizer_and_scheduler(self, job: TrainingJob):
        if not (HAVE_TORCH and job._model):
            return
        loop = asyncio.get_event_loop()

        if optim is None:
            raise RuntimeError("PyTorch optim is None.")
        if job._model is None:
            raise RuntimeError("job._model is None, cannot create optimizer.")

        job._optimizer = await loop.run_in_executor(
            None,
            lambda: optim.AdamW(job._model.parameters(), lr=job.config.learning_rate),  # type: ignore[union-attr]
        )

        num_training_steps = (
            job.total_steps if job.total_steps > 0 else 10000
        )  # Fallback if total_steps unknown

        if job.config.scheduler_type == "linear":
            if (
                get_linear_schedule_with_warmup is not None
                and job._optimizer is not None
            ):
                job._scheduler = await loop.run_in_executor(
                    None,
                    lambda: get_linear_schedule_with_warmup(  # type: ignore[operator]
                        job._optimizer
                        if job._optimizer is not None
                        else (_ for _ in ()).throw(
                            ValueError("Optimizer is not initialized")
                        ),
                        num_warmup_steps=job.config.warmup_steps,
                        num_training_steps=num_training_steps,
                        **job.config.scheduler_kwargs,
                    ),
                )
            else:
                logger.warning(
                    "Linear scheduler or optimizer not available. Continuing without scheduler."
                )
        elif job.config.scheduler_type == "cosine":
            if (
                get_cosine_schedule_with_warmup is not None
                and job._optimizer is not None
            ):
                job._scheduler = await loop.run_in_executor(
                    None,
                    lambda: get_cosine_schedule_with_warmup(  # type: ignore[operator]
                        job._optimizer
                        if job._optimizer is not None
                        else (_ for _ in ()).throw(
                            ValueError("Optimizer is not initialized")
                        ),
                        num_warmup_steps=job.config.warmup_steps,
                        num_training_steps=num_training_steps,
                        **job.config.scheduler_kwargs,
                    ),
                )
            else:
                logger.warning(
                    "Cosine scheduler or optimizer not available. Continuing without scheduler."
                )

        if job._loaded_checkpoint_path:
            logger.info(
                f"Loading optimizer/scheduler state from checkpoint: {job._loaded_checkpoint_path}"
            )
            try:
                if torch is None:
                    raise RuntimeError("PyTorch (torch) is None.")
                loaded_chkpt = await loop.run_in_executor(
                    None,
                    lambda: torch.load(  # type: ignore
                        job._loaded_checkpoint_path
                        if job._loaded_checkpoint_path is not None
                        else (_ for _ in ()).throw(
                            ValueError("Checkpoint path is None")
                        ),
                        map_location=self.device,
                    ),  # type: ignore[union-attr]
                )
                if job._optimizer and "optimizer_state_dict" in loaded_chkpt:
                    await loop.run_in_executor(
                        None,
                        lambda: job._optimizer.load_state_dict(
                            loaded_chkpt["optimizer_state_dict"]
                        )
                        if job._optimizer is not None
                        else None,
                    )  # type: ignore[union-attr]
                if job._scheduler and "scheduler_state_dict" in loaded_chkpt:
                    await loop.run_in_executor(
                        None,
                        lambda: job._scheduler.load_state_dict(
                            loaded_chkpt["scheduler_state_dict"]
                        )
                        if job._scheduler is not None
                        else None,
                    )  # type: ignore[union-attr]
                if job._grad_scaler and "grad_scaler_state_dict" in loaded_chkpt:
                    await loop.run_in_executor(
                        None,
                        lambda: job._grad_scaler.load_state_dict(
                            loaded_chkpt["grad_scaler_state_dict"]
                        )
                        if job._grad_scaler is not None
                        else None,
                    )  # type: ignore[union-attr]
                logger.info(
                    "Optimizer, scheduler, and GradScaler states loaded from checkpoint."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load states from {job._loaded_checkpoint_path}: {e}.",
                    exc_info=True,
                )
            finally:
                job._loaded_checkpoint_path = None

    async def _training_loop(self, job: TrainingJob):
        logger.info(
            f"Starting training loop for job: {job.job_id}. Total steps: {job.total_steps}"
        )
        loop = asyncio.get_event_loop()

        if not (HAVE_TORCH and job._model and job._optimizer and job._train_dataloader):
            if not HAVE_TORCH:  # Original fallback for non-torch env
                logger.warning("Torch not available. Running dummy training loop.")
                for step_idx in range(100):
                    await asyncio.sleep(0.01)
                    with self.job_lock:
                        if job.status != "running":
                            break
                        job.current_step = step_idx + 1
                        job.progress = job.current_step / 100.0
                        job.loss = 1.0 / (job.current_step + 1e-5)
                    if job.current_step % 10 == 0:
                        await self._log_training_progress(
                            job
                        )  # Fixed call to correctly named method
                return
            raise RuntimeError("Missing critical components for PyTorch training loop.")

        model = job._model
        optimizer = job._optimizer
        scheduler = job._scheduler
        train_dataloader = job._train_dataloader
        grad_scaler = job._grad_scaler
        summary_writer = job._summary_writer

        global_step = job.current_step

        for epoch in range(job.current_epoch, job.config.max_epochs):
            if job.status != "running":
                break
            job.current_epoch = epoch
            if hasattr(model, "train"):
                model.train()
            epoch_loss = 0.0
            logger.info(
                f"Starting Epoch {epoch + 1}/{job.config.max_epochs} for job {job.job_id}"
            )

            for batch_idx, batch in enumerate(train_dataloader):
                if job.status != "running":
                    break
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if hasattr(v, "to")
                }

                use_amp = grad_scaler is not None and autocast is not None
                with (
                    autocast()
                    if use_amp and callable(autocast)
                    else (
                        lambda: (_ for _ in ()).throw(
                            Exception("autocast not callable")
                        )
                        if use_amp
                        else asyncio.sleep(0)
                    )
                ):  # type: ignore
                    outputs = model(**batch)  # type: ignore[operator]
                    loss = outputs.loss
                    if job.config.gradient_accumulation_steps > 1:
                        loss = loss / job.config.gradient_accumulation_steps

                if use_amp and grad_scaler is not None:
                    await loop.run_in_executor(
                        None, lambda: grad_scaler.scale(loss).backward()
                    )  # type: ignore[union-attr]
                else:
                    await loop.run_in_executor(None, lambda: loss.backward())

                epoch_loss += loss.item() * job.config.gradient_accumulation_steps

                if (batch_idx + 1) % job.config.gradient_accumulation_steps == 0:
                    if (
                        job.config.max_grad_norm is not None
                        and HAVE_TORCH
                        and torch is not None
                        and torch.nn is not None
                    ):  # L1069 Fix
                        if use_amp and grad_scaler is not None:
                            await loop.run_in_executor(
                                None, lambda: grad_scaler.unscale_(optimizer)
                            )  # type: ignore[union-attr]
                        await loop.run_in_executor(
                            None,
                            lambda: torch.nn.utils.clip_grad_norm_(
                                model.parameters(), job.config.max_grad_norm
                            )
                            if torch and torch.nn
                            else None,  # Ensure torch and torch.nn are available
                        )

                    if use_amp and grad_scaler is not None:
                        await loop.run_in_executor(
                            None, lambda: grad_scaler.step(optimizer)
                        )  # type: ignore[union-attr]
                        await loop.run_in_executor(None, grad_scaler.update)  # type: ignore[union-attr]
                    else:
                        await loop.run_in_executor(None, optimizer.step)  # type: ignore[union-attr]

                    if scheduler:
                        await loop.run_in_executor(None, scheduler.step)  # type: ignore[union-attr]
                    await loop.run_in_executor(None, optimizer.zero_grad)  # type: ignore[union-attr]

                    global_step += 1
                    job.current_step = global_step
                    job.loss = loss.item() * job.config.gradient_accumulation_steps
                    if job.total_steps > 0:
                        job.progress = min(1.0, global_step / job.total_steps)

                    if summary_writer:
                        await loop.run_in_executor(
                            None,
                            lambda: summary_writer.add_scalar(
                                "Loss/train",
                                job.loss if job.loss is not None else 0.0,
                                global_step,
                            )
                            if summary_writer
                            else None,
                        )  # type: ignore[union-attr]
                        if scheduler:
                            lr = (
                                scheduler.get_last_lr()[0]
                                if hasattr(scheduler, "get_last_lr")
                                else 0.0
                            )  # type: ignore[union-attr]
                            await loop.run_in_executor(
                                None,
                                lambda: summary_writer.add_scalar(
                                    "LearningRate", lr, global_step
                                )
                                if summary_writer
                                else None,
                            )  # type: ignore[union-attr]

                    if (
                        job.config.save_steps > 0
                        and global_step % job.config.save_steps == 0
                    ):
                        await self._save_checkpoint(job, global_step)
                    if (
                        job.config.eval_steps > 0
                        and global_step % job.config.eval_steps == 0
                        and job._eval_dataloader
                    ):
                        await self._evaluation_loop(job, global_step)
                        if hasattr(model, "train"):
                            model.train()

                if (batch_idx + 1) % (10 * job.config.gradient_accumulation_steps) == 0:
                    await self._log_training_progress(job)  # Fixed call
                await asyncio.sleep(0)

            avg_epoch_loss = epoch_loss / (len(train_dataloader) or 1)
            job.metrics[f"epoch_{epoch + 1}_train_loss"] = avg_epoch_loss
            logger.info(
                f"Epoch {epoch + 1} for job {job.job_id}. Avg Train Loss: {avg_epoch_loss:.4f}"
            )
            if summary_writer:
                await loop.run_in_executor(
                    None,
                    lambda: summary_writer.add_scalar(
                        "Loss/epoch_train", avg_epoch_loss, epoch + 1
                    )
                    if summary_writer
                    else None,
                )  # type: ignore[union-attr]

            if job.config.save_steps == 0 or (
                job.config.save_steps > 0 and global_step % job.config.save_steps != 0
            ):
                await self._save_checkpoint(job, global_step, epoch_end=True)
            if job.config.eval_steps == 0 or (
                job.config.eval_steps > 0 and global_step % job.config.eval_steps != 0
            ):
                if job._eval_dataloader:
                    await self._evaluation_loop(job, global_step, epoch_end_eval=True)
                    if hasattr(model, "train"):
                        model.train()
            if job.status != "running":
                break

    async def _evaluation_loop(
        self, job: TrainingJob, global_step: int, epoch_end_eval: bool = False
    ):
        if not (
            job._eval_dataloader and job._model and HAVE_TORCH and torch is not None
        ):
            logger.warning(f"Eval skipped for job {job.job_id}: missing components.")
            return

        logger.info(
            f"Starting evaluation for job {job.job_id} at global_step {global_step}..."
        )
        loop = asyncio.get_event_loop()
        model = job._model
        if hasattr(model, "eval"):
            model.eval()

        total_eval_loss = 0
        all_preds: List[Any] = []
        all_labels: List[Any] = []

        # Determine if autocast should be used for evaluation
        # Typically, if AMP was used for training (grad_scaler exists), use it for eval for consistency,
        # though for pure inference torch.no_grad() is the primary context manager.
        use_amp_eval = job._grad_scaler is not None and autocast is not None

        for batch in job._eval_dataloader:
            if job.status != "running":
                logger.info(f"Evaluation for job {job.job_id} interrupted.")
                if hasattr(model, "train"):
                    model.train()
                return

            batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, "to")}
            with torch.no_grad():  # type: ignore[union-attr]
                with (
                    autocast()
                    if use_amp_eval and callable(autocast)
                    else (
                        lambda: (_ for _ in ()).throw(
                            Exception("autocast not callable")
                        )
                        if use_amp_eval
                        else asyncio.sleep(0)
                    )
                ):  # type: ignore
                    outputs = model(**batch)  # type: ignore[operator]

            loss = outputs.loss
            total_eval_loss += loss.item()
            logits = outputs.logits
            preds = await loop.run_in_executor(
                None,
                lambda: torch.argmax(logits, dim=-1).cpu().tolist()
                if torch is not None and logits is not None
                else [],
            )  # type: ignore[union-attr]
            labels = await loop.run_in_executor(
                None, lambda: batch["labels"].cpu().tolist()
            )
            all_preds.extend(preds)
            all_labels.extend(labels)
            await asyncio.sleep(0)

        avg_eval_loss = total_eval_loss / (len(job._eval_dataloader) or 1)
        accuracy = (
            sum(p == label for p, label in zip(all_preds, all_labels)) / len(all_labels)
            if all_labels
            else 0.0
        )

        eval_metric_key_loss = (
            f"eval/loss_step_{global_step}" if not epoch_end_eval else "eval/epoch_loss"
        )
        eval_metric_key_acc = (
            f"eval/accuracy_step_{global_step}"
            if not epoch_end_eval
            else "eval/epoch_accuracy"
        )
        job.metrics[eval_metric_key_loss] = avg_eval_loss
        job.metrics[eval_metric_key_acc] = accuracy
        logger.info(
            f"Eval for job {job.job_id} at step {global_step}. Loss: {avg_eval_loss:.4f}, Acc: {accuracy:.4f}"
        )

        if job._summary_writer:
            log_step = global_step if not epoch_end_eval else job.current_epoch + 1
            metric_name_loss = eval_metric_key_loss.replace("_step_", "/").replace(
                "epoch_", "epoch/"
            )
            metric_name_acc = eval_metric_key_acc.replace("_step_", "/").replace(
                "epoch_", "epoch/"
            )

            await loop.run_in_executor(
                None,
                lambda: job._summary_writer.add_scalar(
                    metric_name_loss, avg_eval_loss, log_step
                )
                if job._summary_writer
                else None,
            )  # type: ignore[union-attr]
            await loop.run_in_executor(
                None,
                lambda: job._summary_writer.add_scalar(
                    metric_name_acc, accuracy, log_step
                )
                if job._summary_writer
                else None,
            )  # type: ignore[union-attr]

        await self._log_training_progress(job)  # Fixed call
        if hasattr(model, "train"):
            model.train()

    async def _log_training_progress(
        self, job: TrainingJob
    ):  # Fixed method definition (L1274 area)
        """Log training progress"""
        self.vanta_core.publish_event(
            "training.progress",
            {
                "job_id": job.job_id,
                "epoch": job.current_epoch + 1,  # epochs are 0-indexed internally
                "step": job.current_step,
                "total_steps": job.total_steps,
                "progress": job.progress,
                "loss": job.loss,
                "metrics": job.metrics,
                "status": job.status,
            },
            source="AsyncTrainingEngine",
        )

    async def _save_checkpoint(
        self, job: TrainingJob, current_global_step: int, epoch_end: bool = False
    ):
        if not (HAVE_TORCH and torch is not None and job._model and job._optimizer):
            logger.warning(
                f"Cannot save checkpoint for job {job.job_id}: missing components."
            )
            return

        loop = asyncio.get_event_loop()
        checkpoint_name_stem = f"{job.job_id}_step_{current_global_step}"
        if epoch_end:
            checkpoint_name_stem = (
                f"{job.job_id}_epoch_{job.current_epoch + 1}_step_{current_global_step}"
            )

        checkpoint_path = Path(job.config.checkpoint_dir) / f"{checkpoint_name_stem}.pt"
        latest_checkpoint_path = (
            Path(job.config.checkpoint_dir) / f"{job.job_id}_latest.pt"
        )

        state_dict: Dict[str, Any] = {  # Ensure type for state_dict
            "epoch": job.current_epoch,
            "global_step": current_global_step,
            "model_state_dict": job._model.state_dict(),  # type: ignore[union-attr]
            "optimizer_state_dict": job._optimizer.state_dict(),  # type: ignore[union-attr]
            "loss": job.loss,
            "metrics": job.metrics,
            "config": job.config.model_dump(),
        }
        if job._scheduler:
            state_dict["scheduler_state_dict"] = job._scheduler.state_dict()  # type: ignore[union-attr]
        if job._grad_scaler:
            state_dict["grad_scaler_state_dict"] = job._grad_scaler.state_dict()  # type: ignore[union-attr]

        try:
            await loop.run_in_executor(
                None,
                lambda: torch.save(state_dict, checkpoint_path)
                if torch is not None
                else (_ for _ in ()).throw(
                    RuntimeError("PyTorch (torch) is None, cannot save checkpoint.")
                ),
            )  # type: ignore[union-attr]
            logger.info(f"Checkpoint saved for job {job.job_id}: {checkpoint_path}")
            await loop.run_in_executor(
                None,
                lambda: torch.save(state_dict, latest_checkpoint_path)
                if torch is not None
                else (_ for _ in ()).throw(
                    RuntimeError("PyTorch (torch) is None, cannot save checkpoint.")
                ),
            )  # type: ignore[union-attr]
            logger.info(
                f"Updated latest checkpoint for job {job.job_id} to {latest_checkpoint_path}"
            )
            self.vanta_core.publish_event(
                "training.checkpoint.saved",
                {"job_id": job.job_id, "path": str(checkpoint_path)},
                source="AsyncTrainingEngine",
            )
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint for job {job.job_id} to {checkpoint_path}: {e}",
                exc_info=True,
            )

    async def pause_training_job(self, job_id: str) -> bool:
        with self.job_lock:
            if job_id not in self.training_jobs:
                logger.warning(f"Pause failed: Job ID {job_id} not found.")
                return False
            job = self.training_jobs[job_id]
            if job.status == "running":
                job.status = "paused"
                self.vanta_core.publish_event(
                    "training.job.paused",
                    {"job_id": job_id},
                    source="AsyncTrainingEngine",
                )
                logger.info(f"Training job paused: {job_id}")
                return True
            logger.warning(
                f"Pause failed: Job {job_id} not in 'running' state (current: {job.status})."
            )
        return False

    async def resume_training_job(self, job_id: str) -> bool:
        with self.job_lock:
            if job_id not in self.training_jobs:
                logger.warning(f"Resume failed: Job ID {job_id} not found.")
                return False
            job = self.training_jobs[job_id]
            if job.status == "paused":
                job.status = "running"
                self.vanta_core.publish_event(
                    "training.job.resumed",
                    {"job_id": job_id},
                    source="AsyncTrainingEngine",
                )
                logger.info(f"Training job resumed: {job_id}")
                # The training loop, if still active (self.training_task exists), will pick up the "running" status.
                # If the task had completed/failed while paused (unlikely if pause works as expected),
                # then start_training_job would be needed again. This basic resume assumes the task is poll-waiting.
                return True
            logger.warning(
                f"Resume failed: Job {job_id} not in 'paused' state (current: {job.status})."
            )
        return False

    async def stop_training_job(
        self, job_id: str, reason: str = "Stopped by user"
    ) -> bool:
        job_was_active_or_pausable = False
        original_status = ""
        with self.job_lock:
            if job_id not in self.training_jobs:
                logger.warning(f"Stop failed: Job ID {job_id} not found.")
                return False
            job = self.training_jobs[job_id]
            original_status = job.status
            if job.status in [
                "running",
                "paused",
                "pending",
            ]:  # Can stop pending jobs before task created
                job.status = "stopping"
                job.error = {"type": "UserStop", "message": reason, "traceback": ""}
                job_was_active_or_pausable = True
                logger.info(
                    f"Stopping training job: {job_id} (original status: {original_status})..."
                )
            else:
                logger.warning(
                    f"Stop failed: Job {job_id} not in a stoppable state (current: {original_status})."
                )
                return False

        if job_was_active_or_pausable:
            if (
                self.active_job_id == job_id
                and self.training_task
                and not self.training_task.done()
            ):
                logger.info(f"Cancelling asyncio task for job {job_id}.")
                self.training_task.cancel()
                # _run_training_job's except/finally will handle final status update and active_job_id clearing
                return True
            else:  # Job was pending, or paused and task already done, or no active task
                with self.job_lock:
                    job.status = "stopped"  # Finalize status directly
                    job.end_time = time.time()
                self.vanta_core.publish_event(
                    "training.job.stopped",
                    {"job_id": job_id, "reason": reason},
                    source="AsyncTrainingEngine",
                )
                logger.info(
                    f"Job {job_id} (original status: {original_status}) marked as stopped directly."
                )
                if (
                    self.active_job_id == job_id
                ):  # If it was the active job but task was None/done
                    self.active_job_id = None
                return True
        return False

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.job_lock:
            job = self.training_jobs.get(job_id)
            if not job:
                return None
            # Using model_dump ensures Pydantic serialization logic is applied
            return job.model_dump(
                exclude={
                    "_model",
                    "_tokenizer",
                    "_optimizer",
                    "_scheduler",
                    "_train_dataloader",
                    "_eval_dataloader",
                    "_grad_scaler",
                    "_summary_writer",
                    "_loaded_checkpoint_path",
                }
            )

    async def list_jobs(self) -> List[Dict[str, Any]]:
        # Changed to run get_job_status sequentially as it's quick and involves a lock.
        # Concurrency with asyncio.gather here doesn't offer much benefit and adds complexity.
        statuses: List[Dict[str, Any]] = []
        job_ids_copy = []
        with self.job_lock:  # Get a snapshot of job_ids under lock
            job_ids_copy = list(self.training_jobs.keys())

        for job_id in job_ids_copy:
            status = await self.get_job_status(
                job_id
            )  # This method already uses the lock
            if status:
                statuses.append(status)
        return statuses

    async def get_training_stats(self) -> Dict[str, Any]:
        with self.job_lock:
            active_job_details = None
            if self.active_job_id and self.active_job_id in self.training_jobs:
                active_job = self.training_jobs[self.active_job_id]
                # Ensure the fields exist before accessing, or use .get() for Pydantic models
                active_job_details = {
                    "job_id": active_job.job_id,
                    "status": active_job.status,
                    "progress": active_job.progress,
                    "current_epoch": active_job.current_epoch,
                    "current_step": active_job.current_step,
                }

            return {
                "device": self.device,
                "total_jobs_managed": len(self.training_jobs),
                "active_job_id": self.active_job_id,
                "active_job_details": active_job_details,
                "is_initialized": self.is_initialized,
                "default_engine_config": self.default_engine_config.model_dump(),
            }

    async def shutdown(self, graceful_timeout: float = 10.0):
        logger.info(
            f"Shutting down Training Engine (graceful_timeout: {graceful_timeout}s)..."
        )
        active_job_to_stop = None
        with self.job_lock:
            if self.active_job_id:
                active_job_to_stop = self.active_job_id

        if active_job_to_stop:
            logger.info(f"Requesting stop for active job: {active_job_to_stop}")
            await self.stop_training_job(active_job_to_stop, reason="Engine Shutdown")
            if self.training_task and not self.training_task.done():
                try:
                    await asyncio.wait_for(self.training_task, timeout=graceful_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout waiting for task of job {active_job_to_stop} to complete. Forcing cancel."
                    )
                    if not self.training_task.done():
                        self.training_task.cancel()
                    try:
                        await self.training_task
                    except asyncio.CancelledError:
                        pass
                except asyncio.CancelledError:
                    pass  # Already cancelled by stop_training_job

        self.is_initialized = False
        self.vanta_core.publish_event(
            "training.engine.shutdown", {}, source="AsyncTrainingEngine"
        )
        logger.info("Training Engine shutdown complete.")

    def get_health_status(self) -> dict:
        """Return health status of the training engine."""
        return {
            "component": self.COMPONENT_NAME,
            "status": "healthy" if self.is_initialized else "degraded",
            "active_job_id": self.active_job_id,
            "num_jobs": len(self.training_jobs),
        }
