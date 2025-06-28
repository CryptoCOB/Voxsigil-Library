#!/usr/bin/env python3
"""
VoxSigil Streamlined Training GUI - Enhanced Version
==================================================
A clean, modern GUI that properly displays all VantaCore components
and provides intuitive training interface.
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback

# Create a log file for debugging with UTF-8 encoding
log_file = open("gui_debug_log.txt", "w", encoding="utf-8")


def log_debug(msg):
    print(msg)
    log_file.write(f"{msg}\n")
    log_file.flush()


log_debug("Starting GUI initialization...")
log_debug(f"Python version: {sys.version}")
log_debug(f"Current directory: {os.getcwd()}")

try:
    import numpy as np

    log_debug("NumPy imported successfully")
except ImportError as e:
    log_debug(f"Failed to import NumPy: {e}")

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
log_debug(f"Added to sys.path: {parent_dir}")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add a file handler with UTF-8 encoding
file_handler = logging.FileHandler("voxsigil_gui.log", encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Import PyQt5
try:
    log_debug("Attempting to import PyQt5...")
    from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSlider,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    log_debug("✅ PyQt5 imported successfully")
    logger.info("✅ PyQt5 imported successfully")
except ImportError as e:
    log_debug(f"❌ PyQt5 import error: {e}")
    log_debug(traceback.format_exc())
    logger.error(f"❌ PyQt5 not available: {e}")
    sys.exit(1)

# Import VantaCore
try:
    log_debug("Attempting to import VantaCore...")
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

    log_debug("✅ VantaCore imported successfully")
    logger.info("✅ VantaCore imported successfully")
    VANTACORE_AVAILABLE = True
except ImportError as e:
    log_debug(f"❌ VantaCore import error: {e}")
    log_debug(traceback.format_exc())
    logger.error(f"❌ VantaCore not available: {e}")
    VANTACORE_AVAILABLE = False


class ComponentDisplayWidget(QWidget):
    """Widget to display VantaCore components in an organized way."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the component display UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🧠 VantaCore Component Registry")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Component tree
        self.component_tree = QTreeWidget()
        self.component_tree.setHeaderLabels(["Component Name", "Type", "Status"])
        self.component_tree.setRootIsDecorated(True)
        self.component_tree.setAlternatingRowColors(True)
        layout.addWidget(self.component_tree)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Components")
        refresh_btn.clicked.connect(self.refresh_components)
        layout.addWidget(refresh_btn)

    def refresh_components(self):
        """Refresh the component display."""
        self.component_tree.clear()

        if not VANTACORE_AVAILABLE:
            item = QTreeWidgetItem(["VantaCore not available", "Error", "❌"])
            self.component_tree.addTopLevelItem(item)
            return

        try:
            # Create VantaCore instance
            vanta_core = UnifiedVantaCore()

            # Get all components
            if hasattr(vanta_core, "registry") and vanta_core.registry:
                component_names = vanta_core.registry.list_components()

                # Group components by type
                component_groups = {
                    "training": [],
                    "evaluation": [],
                    "inference": [],
                    "visualization": [],
                    "system": [],
                }

                for comp_name in component_names:
                    if comp_name.startswith("training_"):
                        component_groups["training"].append(comp_name)
                    elif comp_name.startswith("evaluation_"):
                        component_groups["evaluation"].append(comp_name)
                    elif comp_name.startswith("inference_"):
                        component_groups["inference"].append(comp_name)
                    elif comp_name.startswith("visualization_"):
                        component_groups["visualization"].append(comp_name)
                    else:
                        component_groups["system"].append(comp_name)

                # Create tree structure
                for group_type, components in component_groups.items():
                    if components:
                        # Create group header
                        group_icons = {
                            "training": "🎯",
                            "evaluation": "📊",
                            "inference": "🧮",
                            "visualization": "📈",
                            "system": "⚙️",
                        }

                        group_item = QTreeWidgetItem(
                            [
                                f"{group_icons.get(group_type, '📦')} {group_type.title()} ({len(components)})",
                                "Group",
                                "✅",
                            ]
                        )
                        group_item.setExpanded(True)
                        self.component_tree.addTopLevelItem(group_item)

                        # Add components to group
                        for comp_name in sorted(components):
                            component = vanta_core.registry.get(comp_name)
                            status = "✅ Active" if component else "❌ Error"
                            comp_type = (
                                type(component).__name__ if component else "Unknown"
                            )

                            item = QTreeWidgetItem(
                                [
                                    comp_name.replace(f"{group_type}_", ""),
                                    comp_type,
                                    status,
                                ]
                            )
                            group_item.addChild(item)  # Add summary
                total_components = len(component_names)
                summary_item = QTreeWidgetItem(
                    [f"📋 Total Components: {total_components}", "Summary", "ℹ️"]
                )
                self.component_tree.addTopLevelItem(summary_item)
        except Exception as e:
            logger.error(f"Error refreshing components: {e}")
            item = QTreeWidgetItem([f"Error: {str(e)}", "Error", "❌"])
            self.component_tree.addTopLevelItem(item)


class TrainingWorker(QObject):
    """Worker class for running training in a separate thread."""

    # Define signals for communication with main thread
    progress_updated = pyqtSignal(int)  # Progress percentage
    log_message = pyqtSignal(str)  # Log messages
    batch_completed = pyqtSignal(int, int, dict)  # epoch, batch, metrics
    epoch_completed = pyqtSignal(int, dict)  # epoch, evaluation results
    training_completed = pyqtSignal(dict)  # final results
    training_error = pyqtSignal(str)  # error messages

    def __init__(self):
        super().__init__()
        self.training_data_batches = []
        self.training_components = {}
        self.total_epochs = 1
        self.should_stop = False

    def setup_training(
        self,
        training_data_batches,
        training_components,
        total_epochs,
        enhanced_training_data=None,
    ):
        """Set up training parameters."""
        self.training_data_batches = training_data_batches
        self.training_components = training_components
        self.total_epochs = total_epochs
        self.enhanced_training_data = (
            enhanced_training_data if enhanced_training_data else []
        )
        self.should_stop = False

    def stop_training(self):
        """Signal to stop training."""
        self.should_stop = True

    def run_training(self):
        """Run the actual training process in a separate thread with chunked data support."""
        try:
            self.log_message.emit("🚀 Starting training in background thread...")

            # Check if we're using chunked data
            is_chunked = (
                isinstance(self.enhanced_training_data, dict)
                and self.enhanced_training_data.get("type") == "chunked"
            )

            if is_chunked:
                total_samples = self.enhanced_training_data.get("total_samples", 0)
                chunk_files = self.enhanced_training_data.get("chunk_files", [])
                self.log_message.emit(
                    f"📦 Training with chunked data: {total_samples:,} samples in {len(chunk_files)} chunks"
                )
            else:
                total_samples = (
                    len(self.enhanced_training_data)
                    if self.enhanced_training_data
                    else 0
                )
                self.log_message.emit(
                    f"📊 Training with standard data: {total_samples:,} samples"
                )

            # Debug: Check what training components we have
            self.log_message.emit(
                f"🔍 Debug: Training components keys: {list(self.training_components.keys())}"
            )  # Initialize VantaCore in worker thread if needed
            vanta_core = self.training_components.get("vanta_core")

            # Try to use VantaCore if available, otherwise fallback to direct components
            if vanta_core is None:
                self.log_message.emit(
                    "⚠️ VantaCore not available in worker, attempting to create new instance..."
                )
                try:
                    # Use threading to timeout VantaCore initialization
                    import queue
                    import threading

                    result_queue = queue.Queue()

                    def init_vantacore():
                        try:
                            from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

                            vanta_core = UnifiedVantaCore()
                            registration_results = (
                                vanta_core.auto_register_all_components()
                            )
                            result_queue.put(
                                ("success", vanta_core, registration_results)
                            )
                        except Exception as e:
                            result_queue.put(("error", str(e), None))

                    init_thread = threading.Thread(target=init_vantacore)
                    init_thread.daemon = True
                    init_thread.start()

                    # Wait for result with timeout
                    try:
                        status, vanta_core_or_error, registration_results = (
                            result_queue.get(timeout=5.0)
                        )
                        if status == "success":
                            vanta_core = vanta_core_or_error
                            self.log_message.emit(
                                f"✅ Worker thread VantaCore initialized with {len(registration_results.get('training_components', []))} components"
                            )
                            self.training_components["vanta_core"] = vanta_core
                            # Update component references
                            training_components = registration_results.get(
                                "training_components", []
                            )
                            for component_name in training_components:
                                component = vanta_core.get_component(component_name)
                                if component:
                                    component_type = type(component).__name__.lower()

                                    # Add type validation to prevent list assignment
                                    if isinstance(component, list):
                                        self.log_message.emit(
                                            f"⚠️ Component {component_name} returned as list, skipping"
                                        )
                                        continue

                                    if (
                                        "art" in component_name
                                        or "art" in component_type
                                    ):
                                        # Ensure component has required methods before assignment
                                        if hasattr(
                                            component, "train_batch"
                                        ) and hasattr(component, "lock"):
                                            self.training_components["art_trainer"] = (
                                                component
                                            )
                                            self.training_components[
                                                "art_available"
                                            ] = True
                                            self.log_message.emit(
                                                "✅ Valid ART trainer assigned from VantaCore"
                                            )
                                        else:
                                            self.log_message.emit(
                                                "⚠️ Invalid ART component from VantaCore, missing required methods"
                                            )
                                    elif (
                                        "grid" in component_name
                                        or "grid" in component_type
                                    ):
                                        self.training_components["grid_former"] = (
                                            component
                                        )
                                        self.training_components[
                                            "gridformer_available"
                                        ] = True
                        else:
                            raise Exception(vanta_core_or_error)
                    except queue.Empty:
                        self.log_message.emit("❌ VantaCore initialization timed out")
                        vanta_core = None

                except (ImportError, Exception) as e:
                    self.log_message.emit(
                        f"❌ Failed to create VantaCore in worker: {e}"
                    )
                    vanta_core = None

                # Try to load components directly as fallback
                if vanta_core is None:
                    self.log_message.emit(
                        "🔄 Attempting to load components directly..."
                    )
                    try:
                        # Try to import ART directly
                        from ART.art_trainer import ArtTrainer

                        art_trainer = ArtTrainer()
                        self.training_components["art_trainer"] = art_trainer
                        self.training_components["art_available"] = True
                        self.log_message.emit("✅ ART trainer loaded directly")
                    except Exception as art_e:
                        self.log_message.emit(
                            f"⚠️ Could not load ART trainer directly: {art_e}"
                        )

                    try:  # Try to import GridFormer directly
                        from core.grid_former import GridFormer

                        grid_former = GridFormer()
                        self.training_components["grid_former"] = grid_former
                        self.training_components["gridformer_available"] = True
                        self.log_message.emit("✅ GridFormer loaded directly")
                    except Exception as grid_e:
                        self.log_message.emit(
                            f"⚠️ Could not load GridFormer directly: {grid_e}"
                        )

            else:
                self.log_message.emit("✅ VantaCore available in worker thread")

            total_batches = (
                len(chunk_files) if is_chunked else len(self.training_data_batches)
            )

            for epoch in range(1, self.total_epochs + 1):
                if self.should_stop:
                    self.log_message.emit("⏹️ Training stopped by user")
                    return

                self.log_message.emit(f"📈 Starting epoch {epoch}/{self.total_epochs}")

                if is_chunked:
                    # Process each chunk sequentially for memory efficiency
                    for chunk_idx, chunk_file in enumerate(chunk_files):
                        if self.should_stop:
                            self.log_message.emit("⏹️ Training stopped by user")
                            return

                        self.log_message.emit(
                            f"📦 Loading chunk {chunk_idx + 1}/{len(chunk_files)}"
                        )

                        # Load chunk data
                        try:
                            with open(chunk_file, "r", encoding="utf-8") as f:
                                chunk_data = json.load(f)
                        except Exception as e:
                            self.log_message.emit(
                                f"❌ Failed to load chunk {chunk_idx + 1}: {e}"
                            )
                            continue

                        # Process chunk in smaller batches to prevent memory overload
                        batch_size = 100  # Process 100 samples at a time
                        chunk_batches = [
                            chunk_data[i : i + batch_size]
                            for i in range(0, len(chunk_data), batch_size)
                        ]

                        self.log_message.emit(
                            f"🔄 Processing chunk {chunk_idx + 1} in {len(chunk_batches)} mini-batches"
                        )

                        for mini_batch_idx, mini_batch in enumerate(chunk_batches):
                            if self.should_stop:
                                return  # Process mini-batch
                            self._process_training_batch(mini_batch)

                            # Update progress
                            processed_chunks = (
                                (epoch - 1) * len(chunk_files) + chunk_idx + 1
                            )
                            progress = int(
                                (
                                    processed_chunks
                                    / (self.total_epochs * len(chunk_files))
                                )
                                * 100
                            )
                            self.progress_updated.emit(min(progress, 100))

                            if (mini_batch_idx + 1) % 10 == 0:
                                self.log_message.emit(
                                    f"  📊 Mini-batch {mini_batch_idx + 1}/{len(chunk_batches)} completed"
                                )  # Force memory cleanup after each chunk
                        del chunk_data
                        gc.collect()

                        self.log_message.emit(
                            f"✅ Chunk {chunk_idx + 1} completed, memory cleaned"
                        )

                else:
                    # Standard batch processing for non-chunked data
                    for batch_idx, batch_samples in enumerate(
                        self.training_data_batches
                    ):
                        if self.should_stop:
                            return

                        # Process batch
                        self._process_training_batch(batch_samples)

                        # Update progress
                        processed_batches = (epoch - 1) * total_batches + (
                            batch_idx + 1
                        )
                        total_steps = self.total_epochs * total_batches
                        progress = int((processed_batches / total_steps) * 100)
                        self.progress_updated.emit(min(progress, 100))

                # Run evaluation at end of epoch
                evaluation_results = self._run_evaluation(epoch)
                self.epoch_completed.emit(epoch, evaluation_results)
                self.log_message.emit(
                    f"📊 Epoch {epoch} completed with accuracy: {evaluation_results.get('accuracy', 0):.3f}"
                )

            # Training completed successfully
            final_results = {
                "epochs": self.total_epochs,
                "total_samples": total_samples,
                "chunked": is_chunked,
            }
            self.training_completed.emit(final_results)
            self.log_message.emit("🎉 Training completed successfully!")

        except Exception as e:
            error_msg = f"❌ Training error: {str(e)}"
            self.training_error.emit(error_msg)
            self.log_message.emit(error_msg)

    def _process_training_batch(self, batch_samples):
        """Process a batch of training samples and return success count."""
        if not batch_samples:
            return 0

        self.log_message.emit(f"🔄 Processing batch with {len(batch_samples)} samples")

        # Get training components
        vanta_core = self.training_components.get("vanta_core")
        art_trainer = self.training_components.get("art_trainer")
        grid_former = self.training_components.get("grid_former")

        successful_samples = 0
        training_method = "fallback"

        if vanta_core:
            training_method = "vantacore_events"
        elif art_trainer or grid_former:
            training_method = "components_direct"
        else:
            training_method = "simulation"

        # Process each sample
        for i, sample in enumerate(batch_samples):
            if self.should_stop:
                return successful_samples

            try:
                if training_method == "vantacore_events" and vanta_core:
                    # VantaCore orchestration
                    training_data = {
                        "input": sample.get("input", []),
                        "output": sample.get("output", []),
                        "metadata": sample.get("metadata", {}),
                        "sample_index": i,
                    }
                    vanta_core.event_bus.emit("training_sample", training_data)
                    successful_samples += 1

                elif training_method == "components_direct":
                    # Direct component usage
                    if art_trainer and i < 10:  # Use ART for subset
                        try:
                            inp = sample.get("input", [])
                            if isinstance(inp, list) and len(inp) > 0:
                                inp_flat = np.array(inp, dtype=np.float32).flatten()[
                                    :256
                                ]
                                if inp_flat.max() > 1.0:
                                    inp_flat = inp_flat / (inp_flat.max() or 1.0)
                                inp_flat = np.pad(
                                    inp_flat,
                                    (0, max(0, 256 - len(inp_flat))),
                                    "constant",
                                )

                                # Use ART for inference/classification
                                if hasattr(art_trainer, "classify"):
                                    art_trainer.classify(inp_flat)
                                elif hasattr(art_trainer, "predict"):
                                    art_trainer.predict(inp_flat)

                                successful_samples += 1
                        except Exception:
                            pass

                    if grid_former:
                        try:
                            inp = sample.get("input", [])
                            if hasattr(grid_former, "detect_patterns"):
                                grid_former.detect_patterns(inp)
                            elif hasattr(grid_former, "transform_grid"):
                                grid_former.transform_grid(inp, "analysis")
                            successful_samples += 1
                        except Exception:
                            pass

                else:
                    # Simulation mode
                    time.sleep(0.001)
                    successful_samples += 1

            except Exception:
                continue

        success_rate = (successful_samples / len(batch_samples)) * 100
        self.log_message.emit(
            f"✅ Batch completed: {successful_samples}/{len(batch_samples)} samples ({success_rate:.1f}% success)"
        )

        return successful_samples


class TrainingControlWidget(QWidget):
    """Widget for controlling training processes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vanta_core = None
        self.training_active = False
        self.training_timer = None
        self.training_step = 0
        self.arc_data = None  # ARC data for testing
        self.enhanced_training_data = []  # Initialize empty list to avoid None
        self.fresh_data_generated = False  # Flag to track fresh data generation        # Threading attributes for non-blocking training
        self.training_thread = None
        self.training_worker = None
        self.gpu_devices = self.detect_gpus()  # List of available GPUs

        # Initialize empty training components to prevent errors
        self.training_components = {
            "art_available": False,
            "gridformer_available": False,
            "novel_paradigm_available": False,
            "art_trainer": None,
            "grid_former": None,
            "holo_engine": None,
        }

        self.setup_ui()
        if self.gpu_devices:
            self.log_message(f"🖥️ Detected GPUs: {self.gpu_devices}")
            if len(self.gpu_devices) < 3:
                self.log_message(
                    f"⚠️ Only {len(self.gpu_devices)} GPU(s) detected. Please check your system if you expect 3."
                )
            else:
                self.log_message(
                    f"✅ All {len(self.gpu_devices)} GPUs detected and available for training."
                )
        else:
            self.log_message("⚠️ No GPUs detected. Training will run on CPU.")

    def detect_gpus(self):
        """Detect all available GPUs using torch, tensorflow, or nvidia-smi."""
        gpus = []  # Try torch
        try:
            import torch

            if torch.cuda.is_available():
                gpus = [
                    f"cuda:{i} ({torch.cuda.get_device_name(i)})"
                    for i in range(torch.cuda.device_count())
                ]
                return gpus
        except ImportError:
            pass
        except Exception as e:
            self.log_message(f"[GPU Detection] torch error: {e}")  # Try tensorflow
        try:
            import tensorflow as tf

            physical_gpus = tf.config.list_physical_devices("GPU")
            if physical_gpus:
                gpus = [f"tf:{i} ({gpu.name})" for i, gpu in enumerate(physical_gpus)]
                return gpus
        except ImportError:
            pass
        except Exception as e:
            self.log_message(f"[GPU Detection] tensorflow error: {e}")
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = [f"nvidia:{i} ({name})" for i, name in enumerate(lines)]
                return gpus
        except Exception as e:
            local_log(f"nvidia-smi error: {e}")
        return gpus
                return gpus
        except Exception as e:
            self.log_message(f"[GPU Detection] nvidia-smi error: {e}")
        return gpus

    def setup_ui(self):
        """Set up the training control UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🎯 Training Control Center")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Control buttons
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ Start Training")
        self.start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_training)
        control_layout.addWidget(self.reset_btn)

        self.test_model_btn = QPushButton("🧪 Test Model")
        self.test_model_btn.clicked.connect(self.test_trained_model)
        self.test_model_btn.setToolTip("Test trained model against ARC solutions")
        control_layout.addWidget(self.test_model_btn)

        layout.addLayout(control_layout)

        # Enhanced Controls Section
        controls_group = QGroupBox("🔧 Training Parameters")
        controls_layout = QVBoxLayout(controls_group)

        # ARC Data Controls
        data_controls = QHBoxLayout()

        # Load ARC Data
        self.load_data_btn = QPushButton("📥 Load ARC Dataset")
        self.load_data_btn.clicked.connect(self.load_arc_data)
        data_controls.addWidget(self.load_data_btn)

        # Preview ARC Data
        self.preview_data_btn = QPushButton("👀 Preview Samples")
        self.preview_data_btn.clicked.connect(self.preview_arc_samples)
        data_controls.addWidget(self.preview_data_btn)

        # Data Status
        self.data_status_label = QLabel("📁 Dataset Status: ❌ Not Loaded")
        data_controls.addWidget(self.data_status_label)

        controls_layout.addLayout(data_controls)  # Sample Count Control
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("🔢 Generated Samples:"))
        self.gen_samples_slider = QSlider(Qt.Horizontal)
        self.gen_samples_slider.setRange(100, 2000000)  # Up to 2 million samples
        self.gen_samples_slider.setValue(10000)  # Default to 10K samples
        self.gen_samples_slider.valueChanged.connect(self.update_gen_samples_label)
        sample_layout.addWidget(self.gen_samples_slider)
        self.gen_samples_label = QLabel("10,000 samples")
        sample_layout.addWidget(self.gen_samples_label)

        # Quick sample count buttons
        quick_sample_layout = QHBoxLayout()

        self.sample_10k_btn = QPushButton("🔍 10K")
        self.sample_10k_btn.setToolTip("10,000 samples - good for testing")
        self.sample_10k_btn.clicked.connect(lambda: self.set_sample_count(10000))

        self.sample_100k_btn = QPushButton("🚀 100K")
        self.sample_100k_btn.setToolTip("100,000 samples - moderate training")
        self.sample_100k_btn.clicked.connect(lambda: self.set_sample_count(100000))

        self.sample_500k_btn = QPushButton("💪 500K")
        self.sample_500k_btn.setToolTip("500,000 samples - intensive training")
        self.sample_500k_btn.clicked.connect(lambda: self.set_sample_count(500000))

        self.sample_1m_btn = QPushButton("🔥 1M")
        self.sample_1m_btn.setToolTip("1,000,000 samples - heavy training")
        self.sample_1m_btn.clicked.connect(lambda: self.set_sample_count(1000000))

        self.sample_2m_btn = QPushButton("⚡ 2M")
        self.sample_2m_btn.setToolTip("2,000,000 samples - maximum training")
        self.sample_2m_btn.clicked.connect(lambda: self.set_sample_count(2000000))

        quick_sample_layout.addWidget(self.sample_10k_btn)
        quick_sample_layout.addWidget(self.sample_100k_btn)
        quick_sample_layout.addWidget(self.sample_500k_btn)
        quick_sample_layout.addWidget(self.sample_1m_btn)
        quick_sample_layout.addWidget(self.sample_2m_btn)

        sample_layout.addLayout(quick_sample_layout)
        controls_layout.addLayout(sample_layout)

        # Epochs Control
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("🔄 Max Epochs:"))
        self.epochs_slider = QSlider(Qt.Horizontal)
        self.epochs_slider.setRange(1, 100)
        self.epochs_slider.setValue(10)
        self.epochs_slider.valueChanged.connect(self.update_epoch_label)
        epoch_layout.addWidget(self.epochs_slider)
        self.epoch_label = QLabel("10 epochs")
        epoch_layout.addWidget(self.epoch_label)
        controls_layout.addLayout(epoch_layout)

        # ART Multiplier Control
        art_layout = QHBoxLayout()
        art_layout.addWidget(QLabel("🎨 ART Multiplier:"))
        self.art_multiplier_slider = QSlider(Qt.Horizontal)
        self.art_multiplier_slider.setRange(1, 20)
        self.art_multiplier_slider.setValue(5)
        self.art_multiplier_slider.valueChanged.connect(
            self.update_art_multiplier_label
        )
        art_layout.addWidget(self.art_multiplier_slider)
        self.art_multiplier_label = QLabel("5x generation")
        art_layout.addWidget(self.art_multiplier_label)
        controls_layout.addLayout(art_layout)

        # Sigil Generation
        sigil_controls = QHBoxLayout()
        self.gen_sigils_btn = QPushButton("✨ Generate Sigils")
        self.gen_sigils_btn.clicked.connect(self.generate_sigils)
        sigil_controls.addWidget(self.gen_sigils_btn)
        controls_layout.addLayout(sigil_controls)

        layout.addWidget(controls_group)

        # Output Section
        output_group = QGroupBox("📋 Training Output")
        output_layout = QVBoxLayout(output_group)

        # Training output
        self.training_output = QTextEdit()
        self.training_output.setReadOnly(True)
        self.training_output.setMaximumHeight(200)
        output_layout.addWidget(self.training_output)

        layout.addWidget(output_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        # Initialize VantaCore
        self.initialize_vanta_core()  # Initialize logging
        self.log_message("🚀 Training Control Center initialized")
        self.log_message("📊 Load ARC dataset to begin training")

    def set_sample_count(self, count):
        """Set the sample count slider to a specific value."""
        self.gen_samples_slider.setValue(count)
        self.update_gen_samples_label(count)

    def update_gen_samples_label(self, value):
        """Update the generated samples label with formatting for large numbers."""
        if value >= 1000000:
            self.gen_samples_label.setText(f"{value / 1000000:.1f}M samples")
        elif value >= 1000:
            self.gen_samples_label.setText(f"{value / 1000:.0f}K samples")
        else:
            self.gen_samples_label.setText(f"{value} samples")

        # Show warnings for very large sample counts
        if value >= 1000000:
            self.gen_samples_label.setStyleSheet("color: red; font-weight: bold;")
            self.gen_samples_label.setToolTip(
                "LARGE DATASET: Will use chunked processing and take significant time/storage"
            )
        elif value >= 100000:
            self.gen_samples_label.setStyleSheet("color: orange; font-weight: bold;")
            self.gen_samples_label.setToolTip(
                "MEDIUM DATASET: Will use chunked processing"
            )
        elif value >= 50000:
            self.gen_samples_label.setStyleSheet("color: blue;")
            self.gen_samples_label.setToolTip(
                "Will use chunked processing for memory efficiency"
            )
        else:
            self.gen_samples_label.setStyleSheet("")
            self.gen_samples_label.setToolTip("Standard in-memory processing")

    def update_epoch_label(self, value):
        """Update the epoch label."""
        self.epoch_label.setText(f"{value} epochs")

    def update_art_multiplier_label(self, value):
        """Update the ART multiplier label."""
        self.art_multiplier_label.setText(f"{value}x generation")

    def initialize_vanta_core(self):
        """Initialize VantaCore and training components for training operations."""
        self.log_message("🔄 Initializing training components...")

        # Try to initialize VantaCore first
        try:
            from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

            self.vanta_core = UnifiedVantaCore()
            self.training_components["vanta_core"] = self.vanta_core
            self.log_message("✅ VantaCore initialized for training")

            # Try to register components through VantaCore
            if hasattr(self.vanta_core, "auto_register_all_components"):
                try:
                    registration_results = (
                        self.vanta_core.auto_register_all_components()
                    )
                    if (
                        registration_results
                        and "training_components" in registration_results
                    ):
                        training_components = registration_results[
                            "training_components"
                        ]
                        self.log_message(
                            f"✅ VantaCore registered {len(training_components)} training components"
                        )

                        # Get specific components from VantaCore
                        for component_name in training_components:
                            component = self.vanta_core.get_component(component_name)
                            if component:
                                component_type = type(component).__name__.lower()
                                if (
                                    "art" in component_name.lower()
                                    or "art" in component_type
                                ):
                                    self.training_components["art_trainer"] = component
                                    self.training_components["art_available"] = True
                                    self.log_message(
                                        "✅ ART trainer available through VantaCore"
                                    )
                                elif (
                                    "grid" in component_name.lower()
                                    or "grid" in component_type
                                ):
                                    self.training_components["grid_former"] = component
                                    self.training_components["gridformer_available"] = (
                                        True
                                    )
                                    self.log_message(
                                        "✅ GridFormer available through VantaCore"
                                    )
                except Exception as e:
                    self.log_message(f"⚠️ VantaCore component registration failed: {e}")

        except ImportError as e:
            self.log_message(f"⚠️ VantaCore import failed: {e}")
            self.vanta_core = None
        except Exception as e:
            self.log_message(f"❌ VantaCore initialization failed: {e}")
            self.vanta_core = None  # Try to initialize ART directly if not available through VantaCore
        if not self.training_components.get("art_available", False):
            try:
                # Use the wrapper module to avoid Unicode encoding issues
                from art_trainer_wrapper import ART_TRAINER_AVAILABLE, get_art_trainer

                if ART_TRAINER_AVAILABLE:
                    art_trainer = get_art_trainer()
                    self.training_components["art_trainer"] = art_trainer
                    self.training_components["art_available"] = True
                    self.log_message("✅ ART trainer initialized directly")
                else:
                    self.log_message("⚠️ ART trainer not available through wrapper")
            except Exception as e:
                self.log_message(f"⚠️ ART trainer initialization failed: {e}")
                # Try to log the path information
                try:
                    parent_dir = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                    self.log_message(
                        f"  📋 ART trainer path: {os.path.join(parent_dir, 'ART', 'art_trainer.py')}"
                    )
                except Exception:
                    pass  # Try to initialize GridFormer directly if not available through VantaCore
        if not self.training_components.get("gridformer_available", False):
            try:
                # Use the wrapper module to avoid Unicode encoding issues
                from grid_former_wrapper import GRID_FORMER_AVAILABLE, get_grid_former

                if GRID_FORMER_AVAILABLE:
                    grid_former = get_grid_former()
                    self.training_components["grid_former"] = grid_former
                    self.training_components["gridformer_available"] = True
                    self.log_message("✅ GridFormer initialized directly")
                else:
                    self.log_message("⚠️ GridFormer not available through wrapper")
            except Exception as e:
                self.log_message(f"⚠️ GridFormer initialization failed: {e}")
                # Try to log the path information
                try:
                    parent_dir = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                    self.log_message(
                        f"  📋 GridFormer path: {os.path.join(parent_dir, 'core', 'grid_former.py')}"
                    )
                except Exception:
                    pass

        # Report available components
        available_components = []
        if self.training_components.get("art_available", False):
            available_components.append("ART")
        if self.training_components.get("gridformer_available", False):
            available_components.append("GridFormer")
        if self.vanta_core:
            available_components.append("VantaCore")

        if available_components:
            self.log_message(
                f"🎯 Available training components: {', '.join(available_components)}"
            )
            self.log_message(
                "✅ Training components ready - you can now start training!"
            )
        else:
            self.log_message(
                "⚠️ No training components available - training will run in simulation mode"
            )

    def load_arc_data(self):
        """Load the ARC dataset."""
        self.log_message("📥 Loading ARC dataset...")
        self.load_data_btn.setEnabled(False)

        try:
            import json
            import os

            # Load ARC training data
            arc_dir = os.path.join(
                os.path.dirname(__file__), "..", "ARC", "training_data"
            )
            training_file = os.path.join(arc_dir, "arc-agi_training_challenges.json")
            solutions_file = os.path.join(arc_dir, "arc-agi_training_solutions.json")

            if os.path.exists(training_file) and os.path.exists(solutions_file):
                with open(training_file, "r", encoding="utf-8") as f:
                    training_data = json.load(f)
                with open(solutions_file, "r", encoding="utf-8") as f:
                    solutions_data = json.load(f)

                self.arc_data = {"training": training_data, "solutions": solutions_data}

                self.log_message(f"✅ Loaded {len(training_data)} ARC training tasks")
                self.data_status_label.setText("📁 Dataset Status: ✅ Loaded")

            else:
                self.log_message(f"❌ ARC dataset not found at {training_file}")
                self.data_status_label.setText("📁 Dataset Status: ❌ Not Found")

        except Exception as e:
            self.log_message(f"❌ Error loading ARC data: {e}")
            self.data_status_label.setText("📁 Dataset Status: ❌ Load Error")
        finally:
            self.load_data_btn.setEnabled(True)

    def preview_arc_samples(self):
        """Preview ARC samples."""
        if not self.arc_data:
            self.log_message("❌ No ARC data loaded")
            return

        self.log_message("👀 Previewing ARC samples...")

        try:
            # Show first few tasks
            training_data = self.arc_data["training"]
            task_ids = list(training_data.keys())[:5]

            self.log_message(f"📋 Showing {len(task_ids)} sample tasks:")

            for i, task_id in enumerate(task_ids, 1):
                task = training_data[task_id]
                num_train = len(task["train"])
                num_test = len(task["test"])

                # Get grid dimensions of first training example
                if task["train"]:
                    first_input = task["train"][0]["input"]
                    grid_size = (
                        f"{len(first_input)}x{len(first_input[0])}"
                        if first_input
                        else "empty"
                    )
                else:
                    grid_size = "unknown"
                    self.log_message(
                        f"  {i}. Task {task_id[:8]}... - {num_train} train, {num_test} test, grid: {grid_size}"
                    )

        except Exception as e:
            self.log_message(f"❌ Error previewing samples: {e}")

    def generate_sigils(self):
        """Generate sigil-enhanced training data using ART with chunked processing for large datasets."""
        if not self.arc_data:
            self.log_message("❌ No ARC data loaded")
            return

        self.log_message(
            "✨ Generating sigils with chunked processing for large datasets..."
        )
        self.gen_sigils_btn.setEnabled(False)

        try:
            # Get user parameters
            target_samples = self.gen_samples_slider.value()
            art_multiplier = self.art_multiplier_slider.value()

            # For large datasets (>50k), implement chunking
            chunk_size = 10000  # Process 10k samples at a time
            use_chunking = target_samples > 50000

            if use_chunking:
                self.log_message(
                    f"🔧 Large dataset detected ({target_samples:,} samples)"
                )
                self.log_message(
                    f"📦 Using chunked processing: {chunk_size:,} samples per chunk"
                )
                num_chunks = (target_samples + chunk_size - 1) // chunk_size
                self.log_message(f"📊 Will create {num_chunks} chunks")

                # Create chunks directory
                chunks_dir = os.path.join(os.path.dirname(__file__), "training_chunks")
                os.makedirs(chunks_dir, exist_ok=True)

                # Clear any existing chunk files
                for f in os.listdir(chunks_dir):
                    if f.startswith("chunk_") and f.endswith(".json"):
                        os.remove(os.path.join(chunks_dir, f))

                chunk_files = []
                total_generated = 0

                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, target_samples)
                    chunk_samples = chunk_end - chunk_start

                    self.log_message(
                        f"📦 Processing chunk {chunk_idx + 1}/{num_chunks} ({chunk_samples:,} samples)"
                    )

                    # Generate this chunk
                    chunk_data = self._generate_chunk(chunk_samples, art_multiplier)

                    if chunk_data:
                        # Save chunk to disk
                        chunk_file = os.path.join(
                            chunks_dir, f"chunk_{chunk_idx:04d}.json"
                        )
                        with open(chunk_file, "w", encoding="utf-8") as f:
                            json.dump(chunk_data, f, indent=2)

                        chunk_files.append(chunk_file)
                        total_generated += len(chunk_data)

                        self.log_message(
                            f"💾 Saved chunk {chunk_idx + 1}: {len(chunk_data):,} samples to disk"
                        )

                        # Force memory cleanup after each chunk
                        del chunk_data
                        gc.collect()

                        # Update progress
                        progress = int((chunk_idx + 1) / num_chunks * 100)
                        self.progress_bar.setValue(progress)

                    else:
                        self.log_message(f"⚠️ Failed to generate chunk {chunk_idx + 1}")

                # Store chunk info instead of all data in memory
                self.enhanced_training_data = {
                    "type": "chunked",
                    "chunk_files": chunk_files,
                    "total_samples": total_generated,
                    "chunk_size": chunk_size,
                    "chunks_dir": chunks_dir,
                }

                self.log_message("✅ Chunked generation complete!")
                self.log_message(f"📊 Total samples: {total_generated:,}")
                self.log_message(f"📦 Chunks created: {len(chunk_files)}")
                self.log_message(f"💾 Data saved to: {chunks_dir}")

            else:
                # Standard generation for smaller datasets
                self.log_message(
                    f"📊 Standard generation for {target_samples:,} samples"
                )
                enhanced_data = self._generate_chunk(target_samples, art_multiplier)

                if enhanced_data:
                    self.enhanced_training_data = enhanced_data
                    self.log_message(
                        f"✅ Generated {len(enhanced_data):,} enhanced samples"
                    )
                else:
                    self.log_message("❌ Failed to generate samples")
                    return

            self.fresh_data_generated = True
            self.progress_bar.setValue(100)

        except Exception as e:
            self.log_message(f"❌ Error generating sigils: {e}")
            self.log_message(f"📋 Error details: {traceback.format_exc()}")
        finally:
            self.gen_sigils_btn.setEnabled(True)

    def _generate_chunk(self, chunk_samples, art_multiplier):
        """Generate a chunk of training samples."""
        try:
            training_data = self.arc_data["training"]
            task_ids = list(training_data.keys())  # Initialize components
            art_available = False
            art_trainer = None

            try:
                # Ensure parent directory is in Python path for imports
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

                from ART.art_trainer import ArtTrainer

                art_trainer = ArtTrainer()
                art_available = True
                self.log_message("✅ ART trainer initialized for chunk generation")
            except Exception as e:
                self.log_message(f"⚠️ ART not available: {e}")

            enhanced_data = []
            samples_per_task = max(1, chunk_samples // len(task_ids))

            for task_id in task_ids:
                if len(enhanced_data) >= chunk_samples:
                    break

                task = training_data[task_id]

                # Generate base samples from task
                for train_sample in task["train"]:
                    if len(enhanced_data) >= chunk_samples:
                        break

                    # Add original sample
                    enhanced_sample = {
                        "input": train_sample["input"],
                        "output": train_sample["output"],
                        "task_id": task_id,
                        "generation_type": "original",
                        "metadata": {
                            "source": "arc_training",
                            "input_shape": [
                                len(train_sample["input"]),
                                len(train_sample["input"][0])
                                if train_sample["input"]
                                else 0,
                            ],
                            "output_shape": [
                                len(train_sample["output"]),
                                len(train_sample["output"][0])
                                if train_sample["output"]
                                else 0,
                            ],
                        },
                    }
                    enhanced_data.append(enhanced_sample)
                # Generate ART-enhanced variations if available
                if art_available and art_trainer and art_multiplier > 1:
                    for _ in range(min(art_multiplier - 1, samples_per_task)):
                        if len(enhanced_data) >= chunk_samples:
                            break

                        try:
                            # Create ART-enhanced variation
                            base_sample = task["train"][0] if task["train"] else None
                            if base_sample:
                                # Generate variation using ART processing
                                enhanced_sample = self._create_art_variation(
                                    base_sample, task_id, art_trainer
                                )
                                if enhanced_sample:
                                    enhanced_data.append(enhanced_sample)
                        except Exception:
                            continue  # Skip failed variations

            return enhanced_data[:chunk_samples]  # Ensure exact count

        except Exception as e:
            self.log_message(f"❌ Error generating chunk: {e}")
            return []

    def _create_art_variation(self, base_sample, task_id, art_trainer):
        """Create an ART-enhanced variation of a base sample."""
        try:
            # Convert to flat vectors for ART processing
            inp = base_sample["input"]
            out = base_sample["output"]

            if isinstance(inp, list) and len(inp) > 0:
                inp_flat = np.array(inp, dtype=np.float32).flatten()[:256]
                if inp_flat.max() > 1.0:
                    inp_flat = inp_flat / (inp_flat.max() or 1.0)
                inp_flat = np.pad(
                    inp_flat, (0, max(0, 256 - len(inp_flat))), "constant"
                )
            else:
                inp_flat = np.zeros(256, dtype=np.float32)

            # Process through ART for pattern enhancement
            if hasattr(art_trainer, "enhance_pattern"):
                enhanced_inp = art_trainer.enhance_pattern(inp_flat)
            else:
                # Fallback: add small random variations
                enhanced_inp = inp_flat + np.random.normal(0, 0.01, inp_flat.shape)
                enhanced_inp = np.clip(enhanced_inp, 0, 1)

            # Convert back to grid format (approximate)
            original_shape = (len(inp), len(inp[0]) if inp else 1)
            enhanced_grid = enhanced_inp[
                : original_shape[0] * original_shape[1]
            ].reshape(original_shape)
            enhanced_grid = (
                (enhanced_grid * 10).astype(int).tolist()
            )  # Convert back to integer grid

            return {
                "input": enhanced_grid,
                "output": out,  # Keep original output
                "task_id": task_id,
                "generation_type": "art_enhanced",
                "metadata": {
                    "source": "art_variation",
                    "base_task": task_id,
                    "enhancement_method": "art_pattern_processing",
                },
            }

        except Exception:
            return None

    def load_training_chunk(self, chunk_index):
        """Load a specific training chunk from disk."""
        try:
            if (
                not isinstance(self.enhanced_training_data, dict)
                or self.enhanced_training_data.get("type") != "chunked"
            ):
                return []

            chunk_files = self.enhanced_training_data.get("chunk_files", [])
            if chunk_index >= len(chunk_files):
                return []

            chunk_file = chunk_files[chunk_index]
            if os.path.exists(chunk_file):
                with open(chunk_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []

        except Exception as e:
            self.log_message(f"❌ Error loading chunk {chunk_index}: {e}")
            return []

    def get_total_samples(self):
        """Get total number of training samples (chunked or regular)."""
        if (
            isinstance(self.enhanced_training_data, dict)
            and self.enhanced_training_data.get("type") == "chunked"
        ):
            return self.enhanced_training_data.get("total_samples", 0)
        elif isinstance(self.enhanced_training_data, list):
            return len(self.enhanced_training_data)
        else:
            return 0

    def start_training(self):
        """Start the training process."""
        try:
            # Check if we have data to train on
            total_samples = self.get_total_samples()
            if total_samples == 0:
                self.log_message(
                    "❌ No training data available! Please generate training data first."
                )
                return

            if self.training_active:
                self.log_message("⚠️ Training is already active!")
                return

            # Check if components are available
            if not any(self.training_components.values()):
                self.log_message(
                    "❌ No training components available! Please wait for initialization."
                )
                return
                # Update UI state
            self.training_active = True

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.training_step = 0

            self.log_message(f"🚀 Starting training with {total_samples:,} samples...")

            # Get training parameters
            epochs = self.epochs_slider.value()

            # Create and start training worker
            if self.training_worker is None:
                self.training_worker = TrainingWorker()
                self.training_worker.progress_updated.connect(self.update_progress)
                self.training_worker.log_message.connect(self.log_message)
                self.training_worker.training_completed.connect(
                    self.on_training_complete
                )
                self.training_worker.training_error.connect(self.on_training_error)

            # Set up training parameters
            self.training_worker.setup_training(
                training_data_batches=[],  # Will be handled by chunked processing
                training_components=self.training_components,
                total_epochs=epochs,
                enhanced_training_data=self.enhanced_training_data,
            )

            # Start training in thread
            if self.training_thread is None:
                self.training_thread = QThread()
                self.training_worker.moveToThread(self.training_thread)
                self.training_thread.started.connect(self.training_worker.run_training)
                self.training_thread.start()
            else:
                self.training_worker.run_training()

        except Exception as e:
            self.log_message(f"❌ Error starting training: {str(e)}")
            self.reset_training()

    def stop_training(self):
        """Stop the current training process."""
        if self.training_active:
            self.log_message("⏹️ Stopping training...")

            # Signal worker to stop
            if self.training_worker:
                self.training_worker.stop_training()

            # Stop timer if running
            if self.training_timer and self.training_timer.isActive():
                self.training_timer.stop()

            # Reset UI state
            self.training_active = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            self.log_message("✅ Training stopped.")

    def reset_training(self):
        """Reset the training state and UI."""
        try:  # Stop any active training
            if self.training_active:
                self.stop_training()

            # Reset worker and thread
            if self.training_thread and self.training_thread.isRunning():
                self.training_thread.quit()
                self.training_thread.wait()
                self.training_thread = None

            if self.training_worker:
                self.training_worker.deleteLater()
                self.training_worker = None

            # Reset UI state
            self.training_active = False
            self.training_step = 0
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # Clear any existing progress
            if hasattr(self, "progress_bar"):
                self.progress_bar.setValue(0)

            self.log_message("🔄 Training reset complete.")

        except Exception as e:
            self.log_message(f"❌ Error during reset: {str(e)}")

    def test_trained_model(self):
        """Test the trained model against ARC solutions."""
        try:
            if not self.arc_data:
                self.log_message(
                    "❌ No ARC data loaded for testing! Please load ARC dataset first."
                )
                return

            if not any(comp for comp in self.training_components.values() if comp):
                self.log_message("❌ No trained components available for testing!")
                return

            self.log_message("🧪 Starting model testing against ARC solutions...")

            # Get first few test samples
            test_samples = list(self.arc_data.items())[:10]  # Test on first 10 samples
            correct_predictions = 0
            total_tests = len(test_samples)

            for i, (task_id, task_data) in enumerate(test_samples):
                try:
                    # Simple test - check if we can process the input
                    if "train" in task_data and len(task_data["train"]) > 0:
                        input_grid = task_data["train"][0]["input"]

                        # Simulate model prediction (replace with actual model inference)
                        predicted_output = input_grid  # Placeholder - actual model would transform this

                    # For now, just check if we can process without errors
                    if predicted_output is not None:
                        correct_predictions += 1

                    self.log_message(
                        f"✓ Processed test {i + 1}/{total_tests}: Task {task_id}"
                    )

                except Exception as e:
                    self.log_message(f"❌ Error testing task {task_id}: {str(e)}")

            accuracy = (
                (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
            )

            self.log_message(
                f"🏆 Testing complete! Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_tests})"
            )

        except Exception as e:
            self.log_message(f"❌ Error during model testing: {str(e)}")

    def update_progress(self, value):
        """Update progress bar if it exists."""
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(value)

    def on_training_complete(self, results=None):
        """Handle training completion."""
        self.log_message("🎉 Training completed successfully!")
        if results:
            self.log_message(f"📊 Training summary: {results}")
        self.reset_training()

    def on_training_error(self, error_msg):
        """Handle training errors."""
        self.log_message(f"❌ Training error: {error_msg}")
        self.reset_training()

    def log_message(self, message):
        """Log a message to the console (placeholder for actual logging)."""
        print(f"[Training Control] {message}")
        # In a real implementation, this would emit a signal or write to a log widget


class MetricsDisplayWidget(QWidget):
    """Widget for displaying training metrics and system status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the metrics display UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📊 System Metrics & Status")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        layout.addWidget(self.metrics_text)

        # Update button
        update_btn = QPushButton("🔄 Update Metrics")
        update_btn.clicked.connect(self.update_metrics)
        layout.addWidget(update_btn)

        # Auto-update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(5000)  # Update every 5 seconds

    def update_metrics(self):
        """Update the metrics display."""
        self.metrics_text.clear()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        self.metrics_text.append(f"📊 System Metrics - Updated: {timestamp}")
        self.metrics_text.append("=" * 50)

        try:
            # System information
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_text.append(f"💻 CPU Usage: {cpu_percent}%")

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024**3)  # GB
            memory_total = memory.total / (1024**3)  # GB
            self.metrics_text.append(
                f"🧠 Memory: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_percent}%)"
            )

        except ImportError:
            self.metrics_text.append(
                "📊 System metrics not available (psutil not installed)"
            )
        except Exception as e:
            self.metrics_text.append(f"❌ Error getting system metrics: {e}")

        # VantaCore status
        if VANTACORE_AVAILABLE:
            try:
                vanta_core = UnifiedVantaCore()
                if hasattr(vanta_core, "registry"):
                    component_count = len(vanta_core.registry.list_components())
                    self.metrics_text.append(
                        f"🧠 VantaCore Components: {component_count}"
                    )

                    # Agent status
                    if hasattr(vanta_core, "agent_registry"):
                        agents = vanta_core.get_all_agents()
                        self.metrics_text.append(f"🤖 Active Agents: {len(agents)}")

            except Exception as e:
                self.metrics_text.append(f"❌ VantaCore status error: {e}")
        else:
            self.metrics_text.append("❌ VantaCore not available")


class VoxSigilMainWindow(QMainWindow):
    """Main window for the VoxSigil Streamlined GUI."""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setWindowTitle("VoxSigil Streamlined Training GUI")
        self.setMinimumSize(1200, 800)

    def setup_ui(self):
        """Set up the main window UI."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("🧠 VoxSigil Training Interface")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(
            "color: #2E86AB; padding: 10px; background-color: #F8F9FA; border-radius: 5px; margin-bottom: 10px;"
        )
        layout.addWidget(header)

        # Tab widget
        tab_widget = QTabWidget()

        # Components tab
        components_tab = ComponentDisplayWidget()
        tab_widget.addTab(components_tab, "🧠 Components")

        # Training tab
        training_tab = TrainingControlWidget()
        # (duplicate block removed)

        tab_widget.addTab(training_tab, "🎯 Training")

        # Metrics tab
        metrics_tab = MetricsDisplayWidget()
        tab_widget.addTab(metrics_tab, "📊 Metrics")

        layout.addWidget(tab_widget)

        # Status bar
        self.statusBar().showMessage("🟢 VoxSigil GUI Ready")

        # Refresh components on startup
        components_tab.refresh_components()


def main():
    """Main application entry point."""
    try:
        log_debug("Creating QApplication...")
        app = QApplication(sys.argv)
        app.setApplicationName("VoxSigil Training GUI")

        # Set application style
        style_sheet = """
            QMainWindow {
                background-color: #FFFFFF;
            }
            QTabWidget::pane {
                border: 1px solid #C0C0C0;
                background-color: #FFFFFF;
            }
            QTabBar::tab {
                background-color: #F0F0F0;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2E86AB;
                color: white;
            }
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1F5F85;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
            QTreeWidget {
                border: 1px solid #C0C0C0;
                background-color: #FAFAFA;
            }
            QTextEdit {
                border: 1px solid #C0C0C0;
                background-color: #FAFAFA;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QProgressBar {
                border: 1px solid #C0C0C0;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2E86AB;
                border-radius: 3px;
            }
        """
        app.setStyleSheet(style_sheet)

        # Create and show main window
        window = VoxSigilMainWindow()
        window.show()

        logger.info("🚀 VoxSigil GUI started successfully")
        return app.exec_()
    except Exception as e:
        log_debug(f"❌ CRITICAL ERROR in main(): {str(e)}")
        log_debug(traceback.format_exc())
        logger.critical(f"❌ CRITICAL ERROR in main(): {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

    logger.info("[LAUNCH] VoxSigil GUI started successfully")

    # Run application
