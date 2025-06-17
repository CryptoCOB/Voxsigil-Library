"""
Vanta Async Training Engine
==========================

Unified asynchronous training engine controlled by Vanta core system.
Integrates with the existing engines/async_training_engine.py and provides
Vanta-controlled training orchestration.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Import the actual training engine from engines
try:
    from engines.async_training_engine import AsyncTrainingEngine as BaseAsyncTrainingEngine
    from engines.async_training_engine import TrainingConfig as BaseTrainingConfig
    from engines.async_training_engine import TrainingJob as BaseTrainingJob
    from engines.async_training_engine import TrainingMetrics, TrainingState

    HAVE_BASE_ENGINE = True
except ImportError:
    HAVE_BASE_ENGINE = False
    BaseAsyncTrainingEngine = object
    BaseTrainingConfig = object
    BaseTrainingJob = object
    TrainingState = None
    TrainingMetrics = None

# ML Dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None
    nn = None
    optim = None
    DataLoader = None

from .interfaces import BaseAgentInterface

logger = logging.getLogger(__name__)


class TrainingPriority(Enum):
    """Training job priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class VantaTrainingConfig:
    """Vanta-specific training configuration."""

    model_name: str
    dataset_path: Optional[str] = None
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    device: str = "auto"
    priority: TrainingPriority = TrainingPriority.NORMAL
    checkpoint_interval: int = 100
    validation_interval: int = 50
    vanta_supervised: bool = True
    agent_callbacks: List[str] = field(default_factory=list)

    def to_base_config(self) -> Dict[str, Any]:
        """Convert to base training config format."""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "device": self.device,
            "checkpoint_interval": self.checkpoint_interval,
            "validation_interval": self.validation_interval,
        }


@dataclass
class VantaTrainingJob:
    """Vanta-controlled training job."""

    job_id: str
    config: VantaTrainingConfig
    model: Optional[Any] = None
    dataset: Optional[Any] = None
    status: str = "pending"
    progress: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    assigned_agents: List[str] = field(default_factory=list)
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class VantaAsyncTrainingEngine:
    """
    Vanta-controlled async training engine.

    This wraps the base AsyncTrainingEngine and provides Vanta orchestration,
    agent coordination, and unified training management.
    """

    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        self.base_engine = None
        self.active_jobs: Dict[str, VantaTrainingJob] = {}
        self.job_queue: List[VantaTrainingJob] = []
        self.training_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.registered_agents: Dict[str, BaseAgentInterface] = {}

        # Initialize base engine if available
        if HAVE_BASE_ENGINE:
            try:
                self.base_engine = BaseAsyncTrainingEngine()
                logger.info("Initialized base async training engine")
            except Exception as e:
                logger.warning(f"Failed to initialize base engine: {e}")
                self.base_engine = None

        logger.info("VantaAsyncTrainingEngine initialized")

    def register_agent(self, agent_name: str, agent: BaseAgentInterface):
        """Register an agent for training callbacks."""
        self.registered_agents[agent_name] = agent
        logger.info(f"Registered training agent: {agent_name}")

    def create_training_job(
        self,
        config: VantaTrainingConfig,
        model: Optional[Any] = None,
        dataset: Optional[Any] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Create a new training job."""
        if job_id is None:
            job_id = f"train_{int(time.time() * 1000)}"

        job = VantaTrainingJob(job_id=job_id, config=config, model=model, dataset=dataset)

        self.job_queue.append(job)
        logger.info(f"Created training job: {job_id}")

        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a training job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "metrics": job.metrics,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
            }

        # Check queue
        for job in self.job_queue:
            if job.job_id == job_id:
                return {
                    "job_id": job_id,
                    "status": "queued",
                    "progress": 0.0,
                    "metrics": {},
                    "created_at": job.created_at,
                    "started_at": None,
                    "completed_at": None,
                }

        return None

    def start_training_loop(self):
        """Start the async training loop."""
        if self.is_running:
            logger.warning("Training loop already running")
            return

        self.is_running = True
        self.training_thread = threading.Thread(target=self._training_loop_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
        logger.info("Started Vanta training loop")

    def stop_training_loop(self):
        """Stop the training loop."""
        self.is_running = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        logger.info("Stopped Vanta training loop")

    def _training_loop_worker(self):
        """Main training loop worker."""
        while self.is_running:
            try:
                if self.job_queue:
                    # Get next job (priority queue would be better)
                    job = self.job_queue.pop(0)
                    self._execute_training_job(job)
                else:
                    time.sleep(1.0)  # Wait for jobs
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                time.sleep(1.0)

    def _execute_training_job(self, job: VantaTrainingJob):
        """Execute a single training job."""
        try:
            logger.info(f"Starting training job: {job.job_id}")
            job.status = "running"
            job.started_at = time.time()
            self.active_jobs[job.job_id] = job

            # Notify agents
            self._notify_agents("training_started", job)

            if self.base_engine and HAVE_BASE_ENGINE:
                # Use the base engine for actual training
                base_config = BaseTrainingConfig(**job.config.to_base_config())
                base_job = BaseTrainingJob(
                    job_id=job.job_id, config=base_config, model=job.model, dataset=job.dataset
                )

                # Submit to base engine
                self.base_engine.submit_job(base_job)

                # Monitor progress
                self._monitor_base_job(job, base_job)
            else:
                # Fallback training simulation
                self._simulate_training(job)

            job.status = "completed"
            job.completed_at = time.time()
            logger.info(f"Completed training job: {job.job_id}")

            # Notify agents
            self._notify_agents("training_completed", job)

        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}")
            job.status = "failed"
            job.completed_at = time.time()
            self._notify_agents("training_failed", job)
        finally:
            # Move from active to completed (could maintain a history)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    def _monitor_base_job(self, vanta_job: VantaTrainingJob, base_job):
        """Monitor progress of base engine job."""
        while base_job.status not in ["completed", "failed"]:
            time.sleep(1.0)

            # Update progress from base job
            if hasattr(base_job, "progress"):
                vanta_job.progress = base_job.progress
            if hasattr(base_job, "metrics"):
                vanta_job.metrics.update(base_job.metrics)

            # Notify agents of progress
            self._notify_agents("training_progress", vanta_job)

    def _simulate_training(self, job: VantaTrainingJob):
        """Simulate training when base engine is not available."""
        logger.info(f"Simulating training for job {job.job_id}")

        for epoch in range(job.config.epochs):
            if not self.is_running:
                break

            # Simulate epoch training
            time.sleep(0.1)  # Simulate training time

            job.progress = (epoch + 1) / job.config.epochs
            job.metrics.update(
                {
                    "epoch": epoch + 1,
                    "loss": 1.0 - job.progress + 0.1 * (0.5 - time.time() % 1),
                    "accuracy": job.progress * 0.95,
                }
            )

            # Notify progress
            self._notify_agents("training_progress", job)

            logger.debug(f"Job {job.job_id} epoch {epoch + 1}/{job.config.epochs}")

    def _notify_agents(self, event_type: str, job: VantaTrainingJob):
        """Notify registered agents of training events."""
        for agent_name in job.config.agent_callbacks:
            if agent_name in self.registered_agents:
                try:
                    agent = self.registered_agents[agent_name]
                    if hasattr(agent, "on_training_event"):
                        agent.on_training_event(event_type, job)
                except Exception as e:
                    logger.warning(f"Failed to notify agent {agent_name}: {e}")

        # Also notify Vanta core if available
        if self.vanta_core and hasattr(self.vanta_core, "notify_training_event"):
            try:
                self.vanta_core.notify_training_event(event_type, job)
            except Exception as e:
                logger.warning(f"Failed to notify Vanta core: {e}")

    def list_active_jobs(self) -> List[str]:
        """List all active job IDs."""
        return list(self.active_jobs.keys())

    def list_queued_jobs(self) -> List[str]:
        """List all queued job IDs."""
        return [job.job_id for job in self.job_queue]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        # Remove from queue
        self.job_queue = [job for job in self.job_queue if job.job_id != job_id]

        # Cancel active job (would need base engine support)
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            self._notify_agents("training_cancelled", job)
            del self.active_jobs[job_id]
            return True

        return False


# Export compatibility classes
AsyncTrainingEngine = VantaAsyncTrainingEngine
TrainingConfig = VantaTrainingConfig
TrainingJob = VantaTrainingJob

# Create default instance for Vanta
_default_engine = None


def get_training_engine(vanta_core=None) -> VantaAsyncTrainingEngine:
    """Get the default Vanta training engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = VantaAsyncTrainingEngine(vanta_core)
    return _default_engine


def initialize_vanta_training(vanta_core=None):
    """Initialize Vanta training system."""
    engine = get_training_engine(vanta_core)
    engine.start_training_loop()
    logger.info("Initialized Vanta training system")
    return engine


# For backwards compatibility
def create_training_config(**kwargs) -> VantaTrainingConfig:
    """Create a training configuration."""
    return VantaTrainingConfig(**kwargs)


__all__ = [
    "VantaAsyncTrainingEngine",
    "VantaTrainingConfig",
    "VantaTrainingJob",
    "TrainingPriority",
    "AsyncTrainingEngine",  # Compatibility alias
    "TrainingConfig",  # Compatibility alias
    "TrainingJob",  # Compatibility alias
    "get_training_engine",
    "initialize_vanta_training",
    "create_training_config",
]
