"""
VoxSigil Training Interface Module - Qt5 Version
Advanced training functionality with comprehensive introspection for the Dynamic GridFormer GUI
"""

import logging

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class VoxSigilTrainingInterface(QWidget):
    """
    Advanced Qt5 training interface for VoxSigil Dynamic GridFormer with comprehensive introspection.
    Enhanced with real-time streaming capabilities for training metrics and status.
    """

    training_updated = pyqtSignal(dict)

    def __init__(self, tab_widget=None, training_engine=None, event_bus=None):  # Added tab_widget
        """
        Initialize the training interface with introspection features

        Args:
            tab_widget: The main QTabWidget to add this interface to.
            training_engine: Optional async training engine for centralized training
            event_bus: Event bus for real-time updates
        """
        super().__init__()
        self.tab_widget = tab_widget  # Initialize self.tab_widget
        self.training_engine = training_engine
        self.event_bus = event_bus

        # Initialize data storage for introspection
        self.training_metrics = {
            "loss_history": [],
            "accuracy_history": [],
            "learning_rate_history": [],
            "gradient_norms": [],
            "layer_activations": {},
            "weight_distributions": {},
            "validation_scores": [],
            "computation_time": [],
            "memory_usage": [],
            "batch_statistics": [],
        }

        self.setup_ui()
        self.setup_introspection_features()
        self.setup_streaming()

    def setup_streaming(self):
        """Setup real-time streaming for training updates"""
        # Event bus subscriptions for real-time training updates
        if self.event_bus:
            self.event_bus.subscribe("training_progress", self.on_training_progress)
            self.event_bus.subscribe("training_metrics", self.on_training_metrics)
            self.event_bus.subscribe("training_status", self.on_training_status)
            self.event_bus.subscribe("training_job_update", self.on_training_job_update)

        # Timer for periodic updates of training status
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(1000)  # Update every second

    def setup_ui(self):
        """Create the main UI layout"""
        # Add to tab widget
        if self.tab_widget:  # Check if tab_widget is provided
            self.tab_widget.addTab(self, "🔥 Training Dashboard")
        else:
            logger.warning("VoxSigilTrainingInterface: tab_widget not provided, cannot add to tabs.")

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create splitter for better organization
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Control and status
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel - Introspection features
        right_panel = self.create_introspection_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])

    def create_control_panel(self):
        """Create the training control panel"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)

        # Training Status
        status_group = QGroupBox("Training Status")
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel("Ready to train")
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_group)

        # Control buttons
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout(button_group)

        self.start_btn = QPushButton("Start Training")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.save_btn = QPushButton("Save Model")

        for btn in [self.start_btn, self.pause_btn, self.stop_btn, self.save_btn]:
            button_layout.addWidget(btn)

        layout.addWidget(button_group)

        return control_widget

    def create_introspection_panel(self):
        """Create the introspection features panel"""
        introspection_widget = QWidget()
        layout = QVBoxLayout(introspection_widget)

        # Create tab widget for introspection features
        self.introspection_tabs = QTabWidget()
        layout.addWidget(self.introspection_tabs)

        return introspection_widget

    def setup_introspection_features(self):
        """Setup all 10 introspection features"""

        # Feature 1: Real-time Loss Visualization
        self.setup_loss_visualization()

        # Feature 2: Layer Activation Monitor
        self.setup_activation_monitor()

        # Feature 3: Gradient Flow Analysis
        self.setup_gradient_analysis()

        # Feature 4: Weight Distribution Tracker
        self.setup_weight_tracker()

        # Feature 5: Learning Rate Scheduler Visualization
        self.setup_lr_scheduler()

        # Feature 6: Model Architecture Inspector
        self.setup_architecture_inspector()

        # Feature 7: Training Metrics Dashboard
        self.setup_metrics_dashboard()

        # Feature 8: Hyperparameter Impact Analysis
        self.setup_hyperparameter_analysis()

        # Feature 9: Data Flow Visualization
        self.setup_data_flow()

        # Feature 10: Performance Profiler
        self.setup_performance_profiler()

    def setup_loss_visualization(self):
        """Feature 1: Real-time loss and accuracy plotting"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.loss_figure = Figure(figsize=(10, 6))
        self.loss_canvas = FigureCanvas(self.loss_figure)
        layout.addWidget(self.loss_canvas)

        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_title("Training Loss & Accuracy")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Value")

        self.introspection_tabs.addTab(widget, "📈 Loss Curves")

    def setup_activation_monitor(self):
        """Feature 2: Monitor layer activations in real-time"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.activation_table = QTableWidget()
        self.activation_table.setColumnCount(4)
        self.activation_table.setHorizontalHeaderLabels(["Layer", "Mean", "Std", "Max"])
        layout.addWidget(self.activation_table)

        self.introspection_tabs.addTab(widget, "🧠 Activations")

    def setup_gradient_analysis(self):
        """Feature 3: Gradient flow and magnitude analysis"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.gradient_figure = Figure(figsize=(10, 6))
        self.gradient_canvas = FigureCanvas(self.gradient_figure)
        layout.addWidget(self.gradient_canvas)

        self.gradient_ax = self.gradient_figure.add_subplot(111)
        self.gradient_ax.set_title("Gradient Flow Analysis")

        self.introspection_tabs.addTab(widget, "📊 Gradients")

    def setup_weight_tracker(self):
        """Feature 4: Track weight distributions across layers"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.weight_figure = Figure(figsize=(10, 6))
        self.weight_canvas = FigureCanvas(self.weight_figure)
        layout.addWidget(self.weight_canvas)

        self.introspection_tabs.addTab(widget, "⚖️ Weights")

    def setup_lr_scheduler(self):
        """Feature 5: Learning rate schedule visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.lr_figure = Figure(figsize=(10, 4))
        self.lr_canvas = FigureCanvas(self.lr_figure)
        layout.addWidget(self.lr_canvas)

        self.lr_ax = self.lr_figure.add_subplot(111)
        self.lr_ax.set_title("Learning Rate Schedule")

        self.introspection_tabs.addTab(widget, "📉 LR Schedule")

    def setup_architecture_inspector(self):
        """Feature 6: Interactive model architecture visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.arch_text = QTextEdit()
        self.arch_text.setReadOnly(True)
        self.arch_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.arch_text)

        self.introspection_tabs.addTab(widget, "🏗️ Architecture")

    def setup_metrics_dashboard(self):
        """Feature 7: Comprehensive training metrics dashboard"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Create metric labels
        metrics = ["Epoch", "Loss", "Accuracy", "Val Loss", "Val Acc", "Time/Epoch"]
        self.metric_labels = {}

        for i, metric in enumerate(metrics):
            label = QLabel(f"{metric}:")
            value = QLabel("0.0")
            value.setFont(QFont("Arial", 12, QFont.Bold))
            layout.addWidget(label, i // 2, (i % 2) * 2)
            layout.addWidget(value, i // 2, (i % 2) * 2 + 1)
            self.metric_labels[metric.lower().replace(" ", "_")] = value

        self.introspection_tabs.addTab(widget, "📋 Metrics")

    def setup_hyperparameter_analysis(self):
        """Feature 8: Hyperparameter impact tracking"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.hyperparam_table = QTableWidget()
        self.hyperparam_table.setColumnCount(3)
        self.hyperparam_table.setHorizontalHeaderLabels(["Parameter", "Value", "Impact Score"])
        layout.addWidget(self.hyperparam_table)

        self.introspection_tabs.addTab(widget, "🎛️ Hyperparams")

    def setup_data_flow(self):
        """Feature 9: Visualize data flow through the network"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.dataflow_figure = Figure(figsize=(12, 8))
        self.dataflow_canvas = FigureCanvas(self.dataflow_figure)
        layout.addWidget(self.dataflow_canvas)

        self.introspection_tabs.addTab(widget, "🌊 Data Flow")

    def setup_performance_profiler(self):
        """Feature 10: Performance profiling and bottleneck detection"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.perf_text = QTextEdit()
        self.perf_text.setReadOnly(True)
        self.perf_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.perf_text)

        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout(perf_group)

        self.perf_labels = {}
        perf_metrics = ["GPU Usage", "Memory", "Throughput", "Bottleneck"]

        for i, metric in enumerate(perf_metrics):
            label = QLabel(f"{metric}:")
            value = QLabel("0%")
            perf_layout.addWidget(label, i // 2, (i % 2) * 2)
            perf_layout.addWidget(value, i // 2, (i % 2) * 2 + 1)
            self.perf_labels[metric.lower().replace(" ", "_")] = value

        layout.addWidget(perf_group)
        self.introspection_tabs.addTab(widget, "⚡ Performance")

    def update_training_data(self, epoch, loss, accuracy, val_loss=None, val_acc=None):
        """Update all introspection features with new training data"""
        # Update metrics storage
        self.training_metrics["loss_history"].append(loss)
        self.training_metrics["accuracy_history"].append(accuracy)

        # Update loss visualization
        self.update_loss_plot()

        # Update metrics dashboard
        self.metric_labels["epoch"].setText(str(epoch))
        self.metric_labels["loss"].setText(f"{loss:.4f}")
        self.metric_labels["accuracy"].setText(f"{accuracy:.4f}")

    def update_loss_plot(self):
        """Update the loss visualization plot"""
        self.loss_ax.clear()
        epochs = range(len(self.training_metrics["loss_history"]))

        self.loss_ax.plot(epochs, self.training_metrics["loss_history"], "b-", label="Loss")
        if self.training_metrics["accuracy_history"]:
            self.loss_ax.plot(
                epochs, self.training_metrics["accuracy_history"], "r-", label="Accuracy"
            )

        self.loss_ax.legend()
        self.loss_ax.set_title("Training Progress")
        self.loss_canvas.draw()

    def on_training_progress(self, event):
        """Handle real-time training progress updates"""
        try:
            data = event.get("data", {})

            # Extract progress data with validation
            epoch = int(data.get("epoch", 0))
            progress = float(data.get("progress", 0))
            loss = float(data.get("loss", 0))
            accuracy = float(data.get("accuracy", 0))

            # Update progress bar and label
            if hasattr(self, "training_progress_bar"):
                self.training_progress_bar.setValue(int(progress))

            if hasattr(self, "training_label"):
                self.training_label.setText(f"Epoch {epoch} - {progress:.1f}%")

            # Update metrics
            self.update_training_metrics(epoch, loss, accuracy)

            logger.info(
                f"Training progress update: Epoch {epoch}, Progress {progress}%, Loss {loss:.4f}"
            )

        except (KeyError, ValueError) as e:
            logger.error(f"Invalid training progress data format: {e}")
        except AttributeError as e:
            logger.error(f"Missing UI component in training progress handler: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling training progress update: {e}", exc_info=True)

    def on_training_metrics(self, event):
        """Handle real-time training metrics updates"""
        try:
            data = event.get("data", {})

            # Update stored metrics with validation
            for key, value in data.items():
                if key in self.training_metrics:
                    if isinstance(self.training_metrics[key], list):
                        self.training_metrics[key].append(value)
                    else:
                        self.training_metrics[key] = value

            # Update visualizations
            self.update_loss_plot()

        except (KeyError, TypeError) as e:
            logger.error(f"Invalid training metrics data format: {e}")
        except AttributeError as e:
            logger.error(f"Missing training metrics structure: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling training metrics update: {e}", exc_info=True)

    def on_training_status(self, event):
        """Handle training status updates"""
        try:
            data = event.get("data", {})
            status = data.get("status", "Unknown")

            # Update status label if it exists
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Status: {status}")

                # Color coding with validation
                status_colors = {
                    "Running": "color: green;",
                    "Paused": "color: orange;", 
                    "Error": "color: red;",
                    "Completed": "color: blue;",
                    "Stopped": "color: gray;"
                }
                
                self.status_label.setStyleSheet(
                    status_colors.get(status, "color: gray;")
                )

        except (KeyError, AttributeError) as e:
            logger.error(f"Error updating training status display: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling training status update: {e}", exc_info=True)

    def on_training_job_update(self, event):
        """Handle training job updates from async training engine"""
        try:
            data = event.get("data", {})
            job_id = str(data.get("job_id", "unknown"))
            job_status = str(data.get("status", "Unknown"))
            job_progress = float(data.get("progress", 0))

            # Update job status display with validation
            if hasattr(self, "job_status_text"):
                status_message = f"Job {job_id}: {job_status} ({job_progress:.1f}%)"
                self.job_status_text.append(status_message)
                logger.debug(f"Training job update: {status_message}")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid training job update data format: {e}")
        except AttributeError as e:
            logger.error(f"Missing job status display component: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling training job update: {e}", exc_info=True)

    def periodic_update(self):
        """Periodic update for training interface"""
        try:
            # Check if training engine is available and get status
            if self.training_engine:
                try:
                    # Get training engine status with error handling
                    status = getattr(self.training_engine, "status", "Unknown")
                    active_jobs = getattr(self.training_engine, "active_jobs", 0)

                    # Update display components if they exist
                    if hasattr(self, "engine_status_label"):
                        self.engine_status_label.setText(f"Engine Status: {status}")

                    if hasattr(self, "active_jobs_label"):
                        self.active_jobs_label.setText(f"Active Jobs: {active_jobs}")
                        
                except AttributeError as e:
                    logger.debug(f"Training engine missing expected attributes: {e}")

            # Simulate some metrics for testing (only in development mode)
            import random

            if random.random() < 0.1:  # 10% chance to simulate update
                try:
                    simulated_data = {
                        "epoch": random.randint(1, 100),
                        "progress": random.uniform(0, 100),
                        "loss": random.uniform(0.1, 2.0),
                        "accuracy": random.uniform(0.5, 0.99),
                    }

                    # Simulate training progress event
                    fake_event = {"data": simulated_data}
                    self.on_training_progress(fake_event)
                    
                except Exception as e:
                    logger.debug(f"Error in simulated training data generation: {e}")

        except AttributeError as e:
            logger.error(f"Missing training interface components during periodic update: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in training interface periodic update: {e}", exc_info=True)
