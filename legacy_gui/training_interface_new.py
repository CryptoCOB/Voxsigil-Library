"""
Training Interface for Sigil GUI with VantaCore Integration

Provides a comprehensive training interface with real-time progress monitoring,
hyperparameter configuration, and integration with VantaCore's AsyncTrainingEngine.
"""

import logging
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, ttk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingInterface")

# Try to import VantaCore components
try:
    from Vanta.async_training_engine import (
        AsyncTrainingEngine,
        TrainingConfig,
        TrainingJob,
    )

    VANTA_AVAILABLE = True
    logger.info("VantaCore AsyncTrainingEngine available")
except ImportError:
    VANTA_AVAILABLE = False
    logger.warning("VantaCore AsyncTrainingEngine not available - using fallback mode")

    class TrainingConfig:
        """Simple configuration used by the fallback training engine."""

        def __init__(self, **kwargs):
            self.max_epochs = kwargs.get("max_epochs", 1)
            self.batch_size = kwargs.get("batch_size", 1)
            self.learning_rate = kwargs.get("learning_rate", 1e-4)
            self.device = kwargs.get("device", "cpu")
            self.mixed_precision = kwargs.get("mixed_precision", False)
            self.gradient_accumulation_steps = kwargs.get(
                "gradient_accumulation_steps", 1
            )

    class TrainingJob:
        """Simple asynchronous job wrapper."""

        def __init__(self, train_func):
            self.job_id = f"local_{int(time.time())}"
            self.status = "queued"
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._train_func = train_func

        def _run(self):
            try:
                self.status = "running"
                self._train_func()
                self.status = "completed"
            except Exception as e:
                self.status = "failed"
                logger.error(f"Training job failed: {e}")

        def start(self):
            self._thread.start()

    class AsyncTrainingEngine:
        """Fallback async engine running training in a background thread."""

        def submit_job(self, train_func, *_, **__):
            job = TrainingJob(train_func)
            job.start()
            return job

        def get_job_status(self, job):
            return {"status": job.status}


class VoxSigilTrainingInterface:
    """Training interface for the Sigil GUI with VantaCore integration"""

    def __init__(self, parent, data_loader, train_callback, save_callback):
        """Initialize the training interface"""
        self.parent = parent
        self.data_loader = data_loader
        self.train_callback = train_callback
        self.save_callback = save_callback

        # Training state variables
        self.is_training = False
        self.current_job = None
        self.training_data = None
        self.training_config = {}
        self.stop_requested = False

        # Setup the UI
        self._setup_ui()

        # Create a worker thread for progress updates
        self.progress_thread = None

    def _setup_ui(self):
        """Setup the training interface UI"""
        # Main container frame
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Split into left (config) and right (progress) panels
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Configuration
        config_frame = ttk.LabelFrame(paned, text="Training Configuration")
        paned.add(config_frame, weight=1)

        # Training configuration options
        config_inner = ttk.Frame(config_frame)
        config_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Dataset selection
        ttk.Label(config_inner, text="Dataset:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.dataset_var = tk.StringVar(value="ARC Training")
        dataset_combo = ttk.Combobox(config_inner, textvariable=self.dataset_var)
        dataset_combo["values"] = ("ARC Training", "Custom Dataset")
        dataset_combo.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Batch size
        ttk.Label(config_inner, text="Batch Size:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(
            config_inner, from_=1, to=256, textvariable=self.batch_size_var, width=10
        ).grid(row=1, column=1, sticky=tk.W, pady=5)

        # Learning rate
        ttk.Label(config_inner, text="Learning Rate:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(
            config_inner,
            from_=0.0001,
            to=0.1,
            increment=0.0001,
            textvariable=self.lr_var,
            width=10,
        ).grid(row=2, column=1, sticky=tk.W, pady=5)

        # Epochs
        ttk.Label(config_inner, text="Epochs:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(
            config_inner, from_=1, to=1000, textvariable=self.epochs_var, width=10
        ).grid(row=3, column=1, sticky=tk.W, pady=5)

        # Optimizer
        ttk.Label(config_inner, text="Optimizer:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.optimizer_var = tk.StringVar(value="Adam")
        optimizer_combo = ttk.Combobox(config_inner, textvariable=self.optimizer_var)
        optimizer_combo["values"] = ("Adam", "SGD", "AdamW")
        optimizer_combo.grid(row=4, column=1, sticky=tk.W, pady=5)

        # Device
        ttk.Label(config_inner, text="Device:").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        self.device_var = tk.StringVar(value="cuda" if VANTA_AVAILABLE else "cpu")
        device_combo = ttk.Combobox(config_inner, textvariable=self.device_var)
        device_combo["values"] = ("cuda", "cpu")
        device_combo.grid(row=5, column=1, sticky=tk.W, pady=5)

        # Training mode
        ttk.Label(config_inner, text="Training Mode:").grid(
            row=6, column=0, sticky=tk.W, pady=5
        )
        self.mode_var = tk.StringVar(value="Async" if VANTA_AVAILABLE else "Sync")
        mode_combo = ttk.Combobox(config_inner, textvariable=self.mode_var)
        mode_combo["values"] = ("Async", "Sync")
        mode_combo.grid(row=6, column=1, sticky=tk.W, pady=5)

        # Advanced options
        advanced_frame = ttk.LabelFrame(config_inner, text="Advanced Options")
        advanced_frame.grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # Mixed precision
        self.mixed_precision_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            advanced_frame, text="Mixed Precision", variable=self.mixed_precision_var
        ).pack(anchor=tk.W, pady=5)

        # Gradient accumulation
        ttk.Label(advanced_frame, text="Gradient Accumulation:").pack(
            anchor=tk.W, pady=5
        )
        self.grad_accum_var = tk.IntVar(value=1)
        ttk.Spinbox(
            advanced_frame, from_=1, to=32, textvariable=self.grad_accum_var, width=10
        ).pack(anchor=tk.W, pady=5)

        # Right panel - Progress
        progress_frame = ttk.LabelFrame(paned, text="Training Progress")
        paned.add(progress_frame, weight=2)

        # Progress display
        progress_inner = ttk.Frame(progress_frame)
        progress_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status
        ttk.Label(progress_inner, text="Status:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.status_var = tk.StringVar(value="Not started")
        ttk.Label(progress_inner, textvariable=self.status_var).grid(
            row=0, column=1, sticky=tk.W, pady=5
        )

        # Progress bar
        ttk.Label(progress_inner, text="Progress:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            progress_inner, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=1, column=1, sticky=tk.EW, pady=5)

        # Current epoch
        ttk.Label(progress_inner, text="Epoch:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.epoch_var = tk.StringVar(value="0/0")
        ttk.Label(progress_inner, textvariable=self.epoch_var).grid(
            row=2, column=1, sticky=tk.W, pady=5
        )

        # Loss
        ttk.Label(progress_inner, text="Loss:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.loss_var = tk.StringVar(value="N/A")
        ttk.Label(progress_inner, textvariable=self.loss_var).grid(
            row=3, column=1, sticky=tk.W, pady=5
        )

        # Accuracy
        ttk.Label(progress_inner, text="Accuracy:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.accuracy_var = tk.StringVar(value="N/A")
        ttk.Label(progress_inner, textvariable=self.accuracy_var).grid(
            row=4, column=1, sticky=tk.W, pady=5
        )

        # Time elapsed
        ttk.Label(progress_inner, text="Time Elapsed:").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        self.time_var = tk.StringVar(value="00:00:00")
        ttk.Label(progress_inner, textvariable=self.time_var).grid(
            row=5, column=1, sticky=tk.W, pady=5
        )

        # Training log
        ttk.Label(progress_inner, text="Training Log:").grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=5
        )
        self.log_text = tk.Text(progress_inner, height=10, width=50, wrap=tk.WORD)
        self.log_text.grid(row=7, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # Add scrollbar to log
        log_scrollbar = ttk.Scrollbar(
            progress_inner, orient=tk.VERTICAL, command=self.log_text.yview
        )
        log_scrollbar.grid(row=7, column=2, sticky=tk.NS)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        # Configure grid expansion
        progress_inner.columnconfigure(1, weight=1)
        progress_inner.rowconfigure(7, weight=1)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        # Start training button
        self.start_button = ttk.Button(
            control_frame, text="Start Training", command=self.start_training
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Stop training button
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self.stop_training,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Save model button
        self.save_button = ttk.Button(
            control_frame, text="Save Model", command=self.save_model, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.RIGHT, padx=5)

        # Load configuration button
        self.load_config_button = ttk.Button(
            control_frame, text="Load Config", command=self.load_config
        )
        self.load_config_button.pack(side=tk.RIGHT, padx=5)

        # Save configuration button
        self.save_config_button = ttk.Button(
            control_frame, text="Save Config", command=self.save_config
        )
        self.save_config_button.pack(side=tk.RIGHT, padx=5)

        # Show training status for VantaCore
        vanta_frame = ttk.LabelFrame(main_frame, text="VantaCore Status")
        vanta_frame.pack(fill=tk.X, pady=10)

        if VANTA_AVAILABLE:
            vanta_status = "✅ VantaCore AsyncTrainingEngine is available"
        else:
            vanta_status = (
                "⚠️ VantaCore AsyncTrainingEngine not available - using fallback mode"
            )

        ttk.Label(vanta_frame, text=vanta_status).pack(pady=5)

    def start_training(self):
        """Start the training process"""
        if self.is_training:
            return

        # Get configuration
        self.training_config = {
            "batch_size": self.batch_size_var.get(),
            "learning_rate": self.lr_var.get(),
            "epochs": self.epochs_var.get(),
            "optimizer": self.optimizer_var.get(),
            "device": self.device_var.get(),
            "mixed_precision": self.mixed_precision_var.get(),
            "gradient_accumulation_steps": self.grad_accum_var.get(),
            "mode": self.mode_var.get(),
        }

        # Update UI
        self.is_training = True
        self.stop_requested = False
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)

        # Log training start
        self._add_to_log(
            f"Starting training with configuration:\n{self.training_config}"
        )

        # Load training data
        try:
            self.training_data = self.data_loader.load_data("training")
            self._add_to_log(f"Loaded {len(self.training_data)} training samples")
        except Exception as e:
            self._add_to_log(f"Error loading training data: {str(e)}")
            self.is_training = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            return

        # Start progress monitoring thread
        self.progress_thread = threading.Thread(
            target=self._monitor_progress, daemon=True
        )
        self.progress_thread.start()

        # Start training in a separate thread
        training_thread = threading.Thread(target=self._run_training, daemon=True)
        training_thread.start()

    def stop_training(self):
        """Stop the training process"""
        if not self.is_training:
            return

        self._add_to_log("Stopping training...")
        self.stop_requested = True

    def save_model(self):
        """Save the trained model"""
        if self.save_callback:
            self.save_callback()

    def load_config(self):
        """Load a training configuration from file"""
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            import json

            with open(file_path, "r") as f:
                config = json.load(f)

            # Apply config to UI
            if "batch_size" in config:
                self.batch_size_var.set(config["batch_size"])
            if "learning_rate" in config:
                self.lr_var.set(config["learning_rate"])
            if "epochs" in config:
                self.epochs_var.set(config["epochs"])
            if "optimizer" in config:
                self.optimizer_var.set(config["optimizer"])
            if "device" in config:
                self.device_var.set(config["device"])
            if "mixed_precision" in config:
                self.mixed_precision_var.set(config["mixed_precision"])
            if "gradient_accumulation_steps" in config:
                self.grad_accum_var.set(config["gradient_accumulation_steps"])
            if "mode" in config:
                self.mode_var.set(config["mode"])

            self._add_to_log(f"Loaded configuration from {file_path}")
        except Exception as e:
            self._add_to_log(f"Error loading configuration: {str(e)}")

    def save_config(self):
        """Save the current training configuration to file"""
        config = {
            "batch_size": self.batch_size_var.get(),
            "learning_rate": self.lr_var.get(),
            "epochs": self.epochs_var.get(),
            "optimizer": self.optimizer_var.get(),
            "device": self.device_var.get(),
            "mixed_precision": self.mixed_precision_var.get(),
            "gradient_accumulation_steps": self.grad_accum_var.get(),
            "mode": self.mode_var.get(),
        }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            import json

            with open(file_path, "w") as f:
                json.dump(config, f, indent=2)

            self._add_to_log(f"Saved configuration to {file_path}")
        except Exception as e:
            self._add_to_log(f"Error saving configuration: {str(e)}")

    def _run_training(self):
        """Run the training process"""
        try:
            # Update status
            self.status_var.set("Training in progress")

            # Start time
            start_time = time.time()

            # Use AsyncTrainingEngine if available and selected
            if VANTA_AVAILABLE and self.training_config["mode"] == "Async":
                self._add_to_log("Using VantaCore AsyncTrainingEngine for training")
                self._run_async_training()
            else:
                self._add_to_log("Using synchronous training")
                self._run_sync_training()

            # Update UI when complete
            if not self.stop_requested:
                self.status_var.set("Training complete")
                self.save_button.config(state=tk.NORMAL)
                self._add_to_log("Training completed successfully")
            else:
                self.status_var.set("Training stopped")
                self._add_to_log("Training stopped by user")

        except Exception as e:
            self.status_var.set("Training failed")
            self._add_to_log(f"Error during training: {str(e)}")
            import traceback

            self._add_to_log(traceback.format_exc())

        finally:
            # Reset UI
            self.is_training = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _run_async_training(self):
        """Run training using AsyncTrainingEngine"""
        # Create training config
        config = TrainingConfig(
            max_epochs=self.training_config["epochs"],
            batch_size=self.training_config["batch_size"],
            learning_rate=self.training_config["learning_rate"],
            device=self.training_config["device"],
            mixed_precision=self.training_config["mixed_precision"],
            gradient_accumulation_steps=self.training_config[
                "gradient_accumulation_steps"
            ],
        )

        # Submit training job
        if self.train_callback:
            self.current_job = self.train_callback(self.training_data, config)

            if self.current_job:
                self._add_to_log(
                    f"Training job submitted with ID: {self.current_job.job_id}"
                )

                # Wait for job to complete or be stopped
                while not self.stop_requested:
                    status = self.current_job.status

                    if status in ["completed", "failed", "cancelled"]:
                        break

                    time.sleep(1)

                if self.stop_requested and self.current_job:
                    self._add_to_log("Cancelling training job...")
                    try:
                        if hasattr(self.train_callback, "stop_training_job"):
                            self.train_callback.stop_training_job(
                                self.current_job.job_id
                            )
                    except Exception as e:
                        self._add_to_log(f"Error cancelling job: {e}")

                self._add_to_log(
                    f"Training job finished with status: {self.current_job.status}"
                )
            else:
                self._add_to_log("Failed to submit training job")
        else:
            self._add_to_log("No training callback available")

    def _run_sync_training(self):
        """Run training synchronously using the provided callback"""
        def train_func():
            if self.train_callback:
                self.train_callback(self.training_data, self.training_config)

        if not VANTA_AVAILABLE:
            job = TrainingJob(train_func)
            job.start()
            while job.status == "running" and not self.stop_requested:
                time.sleep(0.5)
            status = job.status
        else:
            train_func()
            status = "completed"

        self._add_to_log(f"Training job finished with status: {status}")

    def _monitor_progress(self):
        """Monitor training progress and update UI"""
        start_time = time.time()

        while self.is_training and not self.stop_requested:
            # Update elapsed time
            elapsed = time.time() - start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            # Sleep briefly
            time.sleep(0.1)

    def _add_to_log(self, message):
        """Add a message to the training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Update UI from the main thread
        self.parent.after(0, self._update_log, log_message)

    def _update_log(self, message):
        """Update the log text (called from main thread)"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
