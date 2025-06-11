"""
VantaCore Integration for Sigil GUI
Manages Vanta async components within the GUI environment
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Use standard path helper for imports
try:
    from utils.path_helper import add_project_root_to_path

    add_project_root_to_path()
except ImportError:
    # Fallback if path_helper isn't available
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    torch = None

try:
    from Vanta.async_training_engine import (
        AsyncTrainingEngine,
        TrainingConfig,
        TrainingJob,
    )
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore as VantaCore

    VANTA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vanta components not available: {e}")
    VANTA_AVAILABLE = False
    VantaCore = None
    AsyncTrainingEngine = None
    TrainingConfig = None
    TrainingJob = None


class VantaGUIIntegration:
    """
    Integration layer between VantaCore and the Sigil GUI.
    Manages async components and provides GUI-friendly interfaces.
    """

    def __init__(self, parent_gui):
        """Initialize VantaCore integration for GUI"""
        self.parent_gui = parent_gui
        self.logger = logging.getLogger("VantaGUI")

        # Component references
        self.vanta_core: Optional[VantaCore] = None
        self.training_engine: Optional[AsyncTrainingEngine] = None

        # Training state
        self.current_training_job: Optional[TrainingJob] = None
        self.training_status_callbacks = []

        # GPU monitoring
        self.gpu_monitor_thread = None
        self.gpu_status = {"available": False, "memory_used": 0, "memory_total": 0}

        # Initialize if Vanta is available
        if VANTA_AVAILABLE:
            self.initialize_vanta()
        else:
            self.logger.warning(
                "VantaCore not available - GUI will operate in fallback mode"
            )

    def initialize_vanta(self):
        """Initialize VantaCore and register components"""
        try:
            # Initialize VantaCore
            self.vanta_core = VantaCore()
            self.logger.info("VantaCore initialized successfully")

            # Setup event subscriptions
            self.setup_event_subscriptions()

            # Initialize training engine
            self.initialize_training_engine()

            # Start GPU monitoring
            self.start_gpu_monitoring()

            self.logger.info("Vanta integration setup complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize VantaCore: {e}")
            self.vanta_core = None

    def setup_event_subscriptions(self):
        """Setup event subscriptions for GUI updates"""
        if not self.vanta_core:
            return  # Subscribe to training events
        self.vanta_core.events.subscribe(
            "training.job_started", self.on_training_started
        )
        self.vanta_core.events.subscribe(
            "training.job_progress", self.on_training_progress
        )
        self.vanta_core.events.subscribe(
            "training.job_completed", self.on_training_completed
        )
        self.vanta_core.events.subscribe("training.job_failed", self.on_training_failed)
        self.vanta_core.events.subscribe("training.job_paused", self.on_training_paused)

        # Subscribe to GPU events
        self.vanta_core.events.subscribe("gpu.status_update", self.on_gpu_status_update)

    def initialize_training_engine(self):
        """Initialize the AsyncTrainingEngine"""
        if not self.vanta_core:
            return

        try:
            # Create default training config
            training_config = TrainingConfig(
                max_epochs=10,
                batch_size=8,
                learning_rate=1e-4,
                save_steps=100,
                eval_steps=50,
                warmup_steps=50,
                device="auto",
                output_dir=PROJECT_ROOT / "training_outputs",
                checkpoint_dir=PROJECT_ROOT / "checkpoints",
                tensorboard_log_dir=PROJECT_ROOT / "logs" / "tensorboard",
            )

            # Initialize training engine
            self.training_engine = AsyncTrainingEngine(self.vanta_core, training_config)

            # Register with VantaCore
            self.vanta_core.registry.register(
                "training_engine",
                self.training_engine,
                {"type": "AsyncTrainingEngine", "version": "1.0"},
            )

            self.logger.info("AsyncTrainingEngine initialized and registered")

        except Exception as e:
            self.logger.error(f"Failed to initialize training engine: {e}")
            self.training_engine = None

    def start_gpu_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        self.gpu_monitor_thread = threading.Thread(
            target=self._gpu_monitor_loop, daemon=True
        )
        self.gpu_monitor_thread.start()

    def _gpu_monitor_loop(self):
        """GPU monitoring loop"""
        while True:
            try:                # Check if torch and CUDA are available
                try:
                    if torch and torch.cuda.is_available():
                        device_count = torch.cuda.device_count()
                        if device_count > 0:
                            # Get memory info for primary GPU
                            memory_allocated = torch.cuda.memory_allocated(0)
                            memory_reserved = torch.cuda.memory_reserved(0)

                            # Get total memory (requires torch >= 1.4)
                            try:
                                total_memory = torch.cuda.get_device_properties(
                                    0
                                ).total_memory
                            except Exception:
                                total_memory = memory_reserved + (
                                    1024**3
                                )  # Fallback estimate

                            self.gpu_status = {
                                "available": True,
                                "device_count": device_count,
                                "memory_allocated": memory_allocated,
                                "memory_reserved": memory_reserved,
                                "memory_total": total_memory,
                                "device_name": torch.cuda.get_device_name(0),
                                "cuda_version": getattr(torch.version, 'cuda', 'unknown'),
                            }
                        else:
                            self.gpu_status = {
                                "available": False,
                                "reason": "No CUDA devices",
                            }
                    else:
                        self.gpu_status = {
                            "available": False,
                            "reason": "CUDA not available or PyTorch not installed",
                        }
                except Exception:
                    self.gpu_status = {
                        "available": False,
                        "reason": "PyTorch not available",
                    }# Publish GPU status update
                if self.vanta_core:
                    self.vanta_core.events.publish("gpu.status_update", self.gpu_status)

                # Update every 5 seconds
                time.sleep(5)

            except Exception as e:
                self.logger.error(f"GPU monitoring error: {e}")
                time.sleep(10)  # Wait longer on errors

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status"""
        return self.gpu_status.copy()

    def create_training_job(
        self, model_path: str, dataset_path: str, config_updates: Dict = None
    ) -> Optional[str]:
        """Create a new training job"""
        if not self.training_engine:
            self.logger.error("Training engine not available")
            return None

        try:
            # Create training config with updates
            config_dict = {
                "max_epochs": 10,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "output_dir": PROJECT_ROOT / "training_outputs",
                "checkpoint_dir": PROJECT_ROOT / "checkpoints",
            }

            if config_updates:
                config_dict.update(config_updates)

            config = TrainingConfig(**config_dict)

            # Create training job
            job_id = f"gui_training_{int(time.time())}"
            job = TrainingJob(
                job_id=job_id,
                model_name_or_path=model_path,
                dataset_name_or_path=dataset_path,
                config=config,
            )

            # Submit job to training engine
            success = self.training_engine.submit_job(job)

            if success:
                self.current_training_job = job
                self.logger.info(f"Training job {job_id} created successfully")
                return job_id
            else:
                self.logger.error("Failed to submit training job")
                return None

        except Exception as e:
            self.logger.error(f"Error creating training job: {e}")
            return None

    def start_training(self) -> bool:
        """Start training the current job"""
        if not self.training_engine or not self.current_training_job:
            return False

        try:
            return self.training_engine.start_job(self.current_training_job.job_id)
        except Exception as e:
            self.logger.error(f"Error starting training: {e}")
            return False

    def pause_training(self) -> bool:
        """Pause current training"""
        if not self.training_engine or not self.current_training_job:
            return False

        try:
            return self.training_engine.pause_job(self.current_training_job.job_id)
        except Exception as e:
            self.logger.error(f"Error pausing training: {e}")
            return False

    def stop_training(self) -> bool:
        """Stop current training"""
        if not self.training_engine or not self.current_training_job:
            return False

        try:
            return self.training_engine.stop_job(self.current_training_job.job_id)
        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
            return False

    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """Get current training status"""
        if not self.current_training_job:
            return None

        return {
            "job_id": self.current_training_job.job_id,
            "status": self.current_training_job.status,
            "progress": self.current_training_job.progress,
            "current_epoch": self.current_training_job.current_epoch,
            "current_step": self.current_training_job.current_step,
            "total_steps": self.current_training_job.total_steps,
            "loss": self.current_training_job.loss,
            "metrics": self.current_training_job.metrics,
        }

    def add_status_callback(self, callback):
        """Add a callback for training status updates"""
        self.training_status_callbacks.append(callback)

    # Event handlers
    def on_training_started(self, event):
        """Handle training started event"""
        self.logger.info(f"Training started: {event['data']}")
        self._notify_status_callbacks("started", event["data"])

    def on_training_progress(self, event):
        """Handle training progress event"""
        data = event["data"]
        if (
            self.current_training_job
            and data.get("job_id") == self.current_training_job.job_id
        ):
            # Update job status
            self.current_training_job.progress = data.get("progress", 0)
            self.current_training_job.current_epoch = data.get("epoch", 0)
            self.current_training_job.current_step = data.get("step", 0)
            self.current_training_job.loss = data.get("loss", None)

            self._notify_status_callbacks("progress", data)

    def on_training_completed(self, event):
        """Handle training completed event"""
        self.logger.info(f"Training completed: {event['data']}")
        if self.current_training_job:
            self.current_training_job.status = "completed"
        self._notify_status_callbacks("completed", event["data"])

    def on_training_failed(self, event):
        """Handle training failed event"""
        self.logger.error(f"Training failed: {event['data']}")
        if self.current_training_job:
            self.current_training_job.status = "failed"
            self.current_training_job.error = event["data"]
        self._notify_status_callbacks("failed", event["data"])

    def on_training_paused(self, event):
        """Handle training paused event"""
        self.logger.info(f"Training paused: {event['data']}")
        if self.current_training_job:
            self.current_training_job.status = "paused"
        self._notify_status_callbacks("paused", event["data"])

    def on_gpu_status_update(self, event):
        """Handle GPU status update event"""
        # GPU status is already updated in the monitoring loop
        # We can use this to notify any additional callbacks if needed
        data = event.get("data", {})
        self.logger.debug(f"GPU status update received: {data}")
        
        # Notify GUI components about GPU status changes
        for callback in self.training_status_callbacks:
            try:
                if hasattr(callback, "__call__"):
                    if hasattr(self.parent_gui, "root"):
                        self.parent_gui.root.after(0, lambda: callback("gpu_status", data))
                    else:
                        callback("gpu_status", data)
            except Exception as e:
                self.logger.error(f"Error in GPU status callback: {e}")

    def _notify_status_callbacks(self, status_type: str, data: Dict):
        """Notify all registered status callbacks"""
        for callback in self.training_status_callbacks:
            try:
                # Schedule callback on main thread
                if hasattr(self.parent_gui, "root"):
                    self.parent_gui.root.after(0, lambda cb=callback, st=status_type, d=data: cb(st, d))
                else:
                    callback(status_type, data)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")

    def shutdown(self):
        """Shutdown VantaCore integration"""
        if self.training_engine and self.current_training_job:
            self.stop_training()

        if self.vanta_core:
            self.vanta_core.shutdown()

        self.logger.info("VantaCore integration shutdown complete")
