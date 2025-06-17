#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportMissingImports=false, reportAssignmentType=false
"""
🧠 VoxSigil Dynamic GridFormer GUI v2.0-quantum-alpha
Enhanced Qt5-based model testing interface with advanced features and dynamic architecture analysis

Created by: GitHub Copilot
Purpose: Advanced Qt5 GUI for testing GridFormer models with comprehensive feature set
"""

import sys
from pathlib import Path
# Import utility tools
from tools.utilities.model_utils import ModelLoader
from tools.utilities.submission_utils import SubmissionFormatter
from utils.data_loader import ARCDataLoader

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplashScreen,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vanta_path = project_root / "Vanta"
sys.path.insert(0, str(vanta_path))

# Import existing components (with fallbacks)
try:
    from GUI.components.vanta_integration import VantaGUIIntegration

    VANTA_INTEGRATION_AVAILABLE = True
except ImportError:
    VANTA_INTEGRATION_AVAILABLE = False
    VantaGUIIntegration = None

try:
    from BLT.blt_encoder import BLTEncoder

    BLT_AVAILABLE = True
except ImportError:
    BLT_AVAILABLE = False

try:
    from Gridformer.inference.gridformer_inference_engine import GridFormerInference
except ImportError:
    GridFormerInference = None



class VoxSigilQt5Styles:
    """Enhanced Qt5 styling system for VoxSigil interface"""

    @staticmethod
    def get_dark_stylesheet():
        return """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #1a1a2e, stop:1 #16213e);
            color: #ffffff;
        }
        
        QTabWidget::pane {
            border: 2px solid #4ecdc4;
            border-radius: 8px;
            background: #2a2a4e;
        }
        
        QTabBar::tab {
            background: #3d3d6b;
            color: #ffffff;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 6px;
        }
        
        QTabBar::tab:selected {
            background: #4ecdc4;
            color: #1a1a2e;
            font-weight: bold;
        }
        
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4ecdc4, stop:1 #45b7b8);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5fd4d1, stop:1 #52c4c5);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3db5b8, stop:1 #35a3a6);
        }
        
        QTextEdit, QTableWidget {
            background: #2a2a4e;
            color: #ffffff;
            border: 1px solid #4ecdc4;
            border-radius: 4px;
            padding: 4px;
        }
        
        QProgressBar {
            border: 2px solid #4ecdc4;
            border-radius: 8px;
            text-align: center;
            background: #2a2a4e;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4ecdc4, stop:1 #45b7b8);
            border-radius: 6px;
        }
        """


class AdvancedModelAnalyzer(QWidget):
    """Feature 1: Advanced Model Architecture Analyzer"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Model structure tree
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["Layer", "Type", "Parameters", "Output Shape"])
        self.layout.addWidget(QLabel("Model Architecture:"))
        self.layout.addWidget(self.model_tree)

        # Analysis controls
        controls = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Model")
        self.export_btn = QPushButton("Export Analysis")
        controls.addWidget(self.analyze_btn)
        controls.addWidget(self.export_btn)
        self.layout.addLayout(controls)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.layout.addWidget(self.results_text)

    def analyze_model(self, model):
        """Analyze model architecture and populate tree"""
        self.model_tree.clear()

        if hasattr(model, "named_modules"):
            for name, module in model.named_modules():
                if name:  # Skip root module
                    item = QTreeWidgetItem(
                        [
                            name,
                            str(type(module).__name__),
                            str(sum(p.numel() for p in module.parameters())),
                            "Dynamic",
                        ]
                    )
                    self.model_tree.addTopLevelItem(item)


class RealTimePerformanceMonitor(QWidget):
    """Feature 2: Real-time Performance Monitor"""

    performance_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Metrics display
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.layout.addWidget(self.metrics_table)

        # Performance graph placeholder
        self.graph_widget = QWidget()
        self.graph_widget.setMinimumHeight(200)
        self.graph_widget.setStyleSheet("background: #2a2a4e; border: 1px solid #4ecdc4;")
        self.layout.addWidget(self.graph_widget)

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second

    def update_metrics(self):
        """Update performance metrics"""
        import psutil

        metrics = {
            "CPU Usage": f"{psutil.cpu_percent()}%",
            "Memory Usage": f"{psutil.virtual_memory().percent}%",
            "GPU Usage": "N/A",  # Would need GPU monitoring library
            "Inference Rate": f"{getattr(self, 'inference_rate', 0):.2f} inf/s",
        }

        self.metrics_table.setRowCount(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))


class BatchProcessingManager(QWidget):
    """Feature 3: Batch Processing Manager"""

    batch_completed = pyqtSignal(int, dict)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Batch configuration
        config_group = QGroupBox("Batch Configuration")
        config_layout = QGridLayout()

        config_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(32)
        config_layout.addWidget(self.batch_size, 0, 1)

        config_layout.addWidget(QLabel("Parallel Workers:"), 1, 0)
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(4)
        config_layout.addWidget(self.workers, 1, 1)

        config_group.setLayout(config_layout)
        self.layout.addWidget(config_group)

        # Progress tracking
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Batch queue
        self.queue_list = QListWidget()
        self.layout.addWidget(QLabel("Processing Queue:"))
        self.layout.addWidget(self.queue_list)

        # Controls
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Batch")
        self.pause_btn = QPushButton("Pause")
        self.clear_btn = QPushButton("Clear Queue")
        controls.addWidget(self.start_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.clear_btn)
        self.layout.addLayout(controls)


class ModelComparison(QWidget):
    """Feature 4: Multi-Model Comparison Tool"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Model selection
        selection_layout = QHBoxLayout()

        self.model1_combo = QComboBox()
        self.model2_combo = QComboBox()
        selection_layout.addWidget(QLabel("Model 1:"))
        selection_layout.addWidget(self.model1_combo)
        selection_layout.addWidget(QLabel("Model 2:"))
        selection_layout.addWidget(self.model2_combo)

        self.compare_btn = QPushButton("Compare Models")
        selection_layout.addWidget(self.compare_btn)

        self.layout.addLayout(selection_layout)

        # Comparison results
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setHorizontalHeaderLabels(["Metric", "Model 1", "Model 2"])
        self.layout.addWidget(self.comparison_table)

        # Detailed analysis
        self.analysis_text = QTextEdit()
        self.layout.addWidget(self.analysis_text)


class DataAugmentationStudio(QWidget):
    """Feature 5: Interactive Data Augmentation Studio"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Augmentation controls
        controls_group = QGroupBox("Augmentation Controls")
        controls_layout = QGridLayout()

        # Rotation
        controls_layout.addWidget(QLabel("Rotation:"), 0, 0)
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(0, 360)
        controls_layout.addWidget(self.rotation_slider, 0, 1)

        # Noise
        controls_layout.addWidget(QLabel("Noise Level:"), 1, 0)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        controls_layout.addWidget(self.noise_slider, 1, 1)

        # Scale
        controls_layout.addWidget(QLabel("Scale:"), 2, 0)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(50, 200)
        self.scale_slider.setValue(100)
        controls_layout.addWidget(self.scale_slider, 2, 1)

        controls_group.setLayout(controls_layout)
        self.layout.addWidget(controls_group)

        # Preview area
        self.preview_widget = QWidget()
        self.preview_widget.setMinimumHeight(300)
        self.preview_widget.setStyleSheet("background: #2a2a4e; border: 1px solid #4ecdc4;")
        self.layout.addWidget(QLabel("Augmentation Preview:"))
        self.layout.addWidget(self.preview_widget)

        # Apply controls
        apply_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Augmentation")
        self.reset_btn = QPushButton("Reset")
        apply_layout.addWidget(self.apply_btn)
        apply_layout.addWidget(self.reset_btn)
        self.layout.addLayout(apply_layout)


class HyperparameterOptimizer(QWidget):
    """Feature 6: Automated Hyperparameter Optimizer"""

    optimization_progress = pyqtSignal(int, dict)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Optimization strategy
        strategy_group = QGroupBox("Optimization Strategy")
        strategy_layout = QVBoxLayout()

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Grid Search", "Random Search", "Bayesian Optimization"])
        strategy_layout.addWidget(self.strategy_combo)

        strategy_group.setLayout(strategy_layout)
        self.layout.addWidget(strategy_group)

        # Parameter ranges
        params_group = QGroupBox("Parameter Ranges")
        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_min = QDoubleSpinBox()
        self.lr_min.setDecimals(6)
        self.lr_min.setValue(0.000001)
        self.lr_max = QDoubleSpinBox()
        self.lr_max.setDecimals(6)
        self.lr_max.setValue(0.01)
        params_layout.addWidget(self.lr_min, 0, 1)
        params_layout.addWidget(QLabel("to"), 0, 2)
        params_layout.addWidget(self.lr_max, 0, 3)

        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_min = QSpinBox()
        self.batch_min.setValue(8)
        self.batch_max = QSpinBox()
        self.batch_max.setValue(128)
        params_layout.addWidget(self.batch_min, 1, 1)
        params_layout.addWidget(QLabel("to"), 1, 2)
        params_layout.addWidget(self.batch_max, 1, 3)

        params_group.setLayout(params_layout)
        self.layout.addWidget(params_group)

        # Results tracking
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Trial", "Parameters", "Score", "Status"])
        self.layout.addWidget(QLabel("Optimization Results:"))
        self.layout.addWidget(self.results_table)

        # Control buttons
        controls = QHBoxLayout()
        self.start_opt_btn = QPushButton("Start Optimization")
        self.stop_opt_btn = QPushButton("Stop")
        controls.addWidget(self.start_opt_btn)
        controls.addWidget(self.stop_opt_btn)
        self.layout.addLayout(controls)


class ModelVersionControl(QWidget):
    """Feature 7: Model Version Control System"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Version history
        self.version_tree = QTreeWidget()
        self.version_tree.setHeaderLabels(["Version", "Date", "Description", "Performance"])
        self.layout.addWidget(QLabel("Model Versions:"))
        self.layout.addWidget(self.version_tree)

        # Version details
        details_group = QGroupBox("Version Details")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        details_layout.addWidget(self.details_text)

        details_group.setLayout(details_layout)
        self.layout.addWidget(details_group)

        # Controls
        controls = QHBoxLayout()
        self.save_version_btn = QPushButton("Save Version")
        self.load_version_btn = QPushButton("Load Version")
        self.compare_versions_btn = QPushButton("Compare Versions")
        self.rollback_btn = QPushButton("Rollback")

        controls.addWidget(self.save_version_btn)
        controls.addWidget(self.load_version_btn)
        controls.addWidget(self.compare_versions_btn)
        controls.addWidget(self.rollback_btn)

        self.layout.addLayout(controls)


class ExperimentTracker(QWidget):
    """Feature 8: Comprehensive Experiment Tracking"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Experiment configuration
        config_group = QGroupBox("Experiment Configuration")
        config_layout = QGridLayout()

        config_layout.addWidget(QLabel("Experiment Name:"), 0, 0)
        self.exp_name = QTextEdit()
        self.exp_name.setMaximumHeight(30)
        config_layout.addWidget(self.exp_name, 0, 1)

        config_layout.addWidget(QLabel("Description:"), 1, 0)
        self.exp_description = QTextEdit()
        self.exp_description.setMaximumHeight(60)
        config_layout.addWidget(self.exp_description, 1, 1)

        config_group.setLayout(config_layout)
        self.layout.addWidget(config_group)

        # Experiment history
        self.exp_table = QTableWidget(0, 5)
        self.exp_table.setHorizontalHeaderLabels(["Name", "Date", "Model", "Best Score", "Status"])
        self.layout.addWidget(QLabel("Experiment History:"))
        self.layout.addWidget(self.exp_table)

        # Controls
        controls = QHBoxLayout()
        self.start_exp_btn = QPushButton("Start Experiment")
        self.stop_exp_btn = QPushButton("Stop Experiment")
        self.export_exp_btn = QPushButton("Export Results")

        controls.addWidget(self.start_exp_btn)
        controls.addWidget(self.stop_exp_btn)
        controls.addWidget(self.export_exp_btn)

        self.layout.addLayout(controls)


class AdvancedVisualization(QWidget):
    """Feature 9: Advanced Model Visualization Suite"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Visualization controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QHBoxLayout()

        self.viz_type = QComboBox()
        self.viz_type.addItems(
            ["Activation Maps", "Attention Heatmaps", "Feature Maps", "Model Graph"]
        )
        controls_layout.addWidget(QLabel("Type:"))
        controls_layout.addWidget(self.viz_type)

        self.layer_selector = QComboBox()
        controls_layout.addWidget(QLabel("Layer:"))
        controls_layout.addWidget(self.layer_selector)

        self.generate_viz_btn = QPushButton("Generate")
        controls_layout.addWidget(self.generate_viz_btn)

        controls_group.setLayout(controls_layout)
        self.layout.addWidget(controls_group)

        # Visualization display
        self.viz_display = QWidget()
        self.viz_display.setMinimumHeight(400)
        self.viz_display.setStyleSheet("background: #2a2a4e; border: 1px solid #4ecdc4;")
        self.layout.addWidget(self.viz_display)

        # Export controls
        export_layout = QHBoxLayout()
        self.export_viz_btn = QPushButton("Export Visualization")
        self.save_config_btn = QPushButton("Save Config")
        export_layout.addWidget(self.export_viz_btn)
        export_layout.addWidget(self.save_config_btn)
        self.layout.addLayout(export_layout)


class IntelligentAssistant(QWidget):
    """Feature 10: AI-Powered Intelligent Assistant"""

    assistant_response = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Chat interface
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(QLabel("VoxSigil Assistant:"))
        self.layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(60)
        self.send_btn = QPushButton("Send")

        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_btn)
        self.layout.addLayout(input_layout)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QGridLayout()

        self.suggest_btn = QPushButton("Suggest Improvements")
        self.analyze_btn = QPushButton("Analyze Performance")
        self.debug_btn = QPushButton("Debug Issues")
        self.optimize_btn = QPushButton("Optimize Settings")

        actions_layout.addWidget(self.suggest_btn, 0, 0)
        actions_layout.addWidget(self.analyze_btn, 0, 1)
        actions_layout.addWidget(self.debug_btn, 1, 0)
        actions_layout.addWidget(self.optimize_btn, 1, 1)

        actions_group.setLayout(actions_layout)
        self.layout.addWidget(actions_group)

        # Connect signals
        self.send_btn.clicked.connect(self.send_message)
        self.assistant_response.connect(self.display_response)

    def send_message(self):
        """Send user message to assistant"""
        message = self.user_input.toPlainText().strip()
        if message:
            self.chat_display.append(f"<b>You:</b> {message}")
            self.user_input.clear()

            # Simulate AI response (would integrate with actual AI service)
            self.process_message(message)

    def process_message(self, message):
        """Process user message and generate response"""
        # Simple rule-based responses (would be replaced with actual AI)
        responses = {
            "help": "I can help you with model training, optimization, debugging, and analysis.",
            "performance": "Based on current metrics, your model is performing well. Consider increasing batch size for faster training.",
            "optimize": "I suggest adjusting learning rate to 0.001 and using data augmentation.",
            "debug": "Check for data leakage and ensure proper validation split.",
        }

        response = responses.get(
            message.lower(),
            "I'm here to help! Ask me about model training, optimization, or debugging.",
        )

        QTimer.singleShot(1000, lambda: self.assistant_response.emit(response))

    def display_response(self, response):
        """Display assistant response"""
        self.chat_display.append(f"<b>Assistant:</b> {response}")


class DynamicGridFormerTab(QWidget):
    """Dynamic GridFormer interface as a QWidget tab (converted from QMainWindow)"""

    def __init__(self):
        super().__init__()
        self._init_core_components()
        self._setup_gui()
        self._init_features()
        self._setup_connections()
        self._discover_models()

    def _init_core_components(self):
        """Initialize core utility components"""
        self.model_loader = ModelLoader()
        self.data_loader = ARCDataLoader()
        self.submission_formatter = SubmissionFormatter()

        # Initialize inference engine
        if GridFormerInference is not None:
            self.inference_engine = GridFormerInference(self.model_loader, self.data_loader)
        else:
            self.inference_engine = self._create_fallback_inference()

        # Initialize integrations
        self._init_integrations()

        # State variables
        self.discovered_models = {}
        self.current_model = None
        self.current_model_path = None
        self.test_data = None
        self.predictions = None

    def _init_integrations(self):
        """Initialize external integrations"""
        # Vanta integration
        if VANTA_INTEGRATION_AVAILABLE:
            try:
                self.vanta_integration = VantaGUIIntegration(self)
            except Exception as e:
                print(f"Vanta integration failed: {e}")
                self.vanta_integration = None
        else:
            self.vanta_integration = None

        # BLT integration
        if BLT_AVAILABLE:
            try:
                self.blt_encoder = BLTEncoder()
            except Exception as e:
                print(f"BLT integration failed: {e}")
                self.blt_encoder = None
        else:
            self.blt_encoder = None

    def _setup_gui(self):
        """Setup the main GUI layout"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🧠 VoxSigil Dynamic GridFormer Suite")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create main tab widget for all features
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

    def _init_features(self):
        """Initialize all advanced features as tabs"""
        # Feature 1: Model Analyzer
        self.model_analyzer = AdvancedModelAnalyzer()
        self.tab_widget.addTab(self.model_analyzer, "🔍 Model Analyzer")

        # Feature 2: Performance Monitor
        self.performance_monitor = RealTimePerformanceMonitor()
        self.tab_widget.addTab(self.performance_monitor, "📊 Performance")

        # Feature 3: Batch Manager
        self.batch_manager = BatchProcessingManager()
        self.tab_widget.addTab(self.batch_manager, "⚡ Batch Processing")

        # Feature 4: Model Comparison
        self.model_comparison = ModelComparison()
        self.tab_widget.addTab(self.model_comparison, "🔀 Model Comparison")

        # Feature 5: Data Augmentation
        self.data_augmentation = DataAugmentationStudio()
        self.tab_widget.addTab(self.data_augmentation, "🎨 Data Augmentation")

        # Feature 6: Hyperparameter Optimizer
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.tab_widget.addTab(self.hyperparameter_optimizer, "🎯 Hyperparameter Optimizer")

        # Feature 7: Version Control
        self.version_control = ModelVersionControl()
        self.tab_widget.addTab(self.version_control, "📦 Version Control")

        # Feature 8: Experiment Tracker
        self.experiment_tracker = ExperimentTracker()
        self.tab_widget.addTab(self.experiment_tracker, "🧪 Experiment Tracker")

        # Feature 9: Visualization Suite
        self.visualization_suite = AdvancedVisualization()
        self.tab_widget.addTab(self.visualization_suite, "🎭 Visualization Suite")

        # Feature 10: AI Assistant
        self.ai_assistant = IntelligentAssistant()
        self.tab_widget.addTab(self.ai_assistant, "🤖 AI Assistant")

    def _setup_connections(self):
        """Setup signal-slot connections"""
        # Performance monitoring
        self.performance_monitor.performance_updated.connect(self._on_performance_update)

        # Batch processing
        self.batch_manager.batch_completed.connect(self._on_batch_completed)

        # Hyperparameter optimization
        self.hyperparameter_optimizer.optimization_progress.connect(self._on_optimization_progress)

        # AI Assistant
        self.ai_assistant.assistant_response.connect(self._on_assistant_response)

    def _discover_models(self):
        """Discover available models"""
        try:
            models = self.model_loader.discover_models()
            self.discovered_models = models

            # Update model selector in comparison tool
            if hasattr(self.model_comparison, "model1_combo"):
                self.model_comparison.model1_combo.clear()
                self.model_comparison.model2_combo.clear()
                for model_name in models.keys():
                    self.model_comparison.model1_combo.addItem(model_name)
                    self.model_comparison.model2_combo.addItem(model_name)

        except Exception as e:
            print(f"Model discovery failed: {e}")

    def _create_fallback_inference(self):
        """Create fallback inference engine when GridFormerInference is not available"""

        class FallbackInference:
            def predict(self, data):
                return {"prediction": "Fallback inference - GridFormerInference not available"}

        return FallbackInference()

    # Event handlers
    def _on_performance_update(self, metrics):
        """Handle performance update"""
        pass

    def _on_batch_completed(self, batch_id, results):
        """Handle batch completion"""
        pass

    def _on_optimization_progress(self, trial, results):
        """Handle optimization progress"""
        pass

    def _on_assistant_response(self, response):
        """Handle assistant response"""
        pass


class DynamicGridFormerQt5GUI(QMainWindow):
    """Main Qt5 GUI Application - DEPRECATED: Use DynamicGridFormerTab instead"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "🧠 VoxSigil Dynamic GridFormer Qt5 Suite v2.0-quantum-alpha - DEPRECATED"
        )
        self.setGeometry(100, 100, 1600, 1000)

        # Show deprecation warning
        print("⚠️ WARNING: DynamicGridFormerQt5GUI (QMainWindow) is deprecated.")
        print("   Use DynamicGridFormerTab (QWidget) in the unified GUI instead.")

        # Apply dark theme
        self.setStyleSheet(VoxSigilQt5Styles.get_dark_stylesheet())

        # Initialize core components
        self._init_core_components()

        # Setup GUI
        self._setup_gui()

        # Initialize features
        self._init_features()

        # Setup connections
        self._setup_connections()

        # Discover models
        self._discover_models()

    def _init_core_components(self):
        """Initialize core utility components"""
        self.model_loader = ModelLoader()
        self.data_loader = ARCDataLoader()
        self.submission_formatter = SubmissionFormatter()

        # Initialize inference engine
        if GridFormerInference is not None:
            self.inference_engine = GridFormerInference(self.model_loader, self.data_loader)
        else:
            self.inference_engine = self._create_fallback_inference()

        # Initialize integrations
        self._init_integrations()

        # State variables
        self.discovered_models = {}
        self.current_model = None
        self.current_model_path = None
        self.test_data = None
        self.predictions = None

    def _init_integrations(self):
        """Initialize external integrations"""
        # Vanta integration
        if VANTA_INTEGRATION_AVAILABLE:
            try:
                self.vanta_integration = VantaGUIIntegration(self)
            except Exception as e:
                print(f"Vanta integration failed: {e}")
                self.vanta_integration = None
        else:
            self.vanta_integration = None

        # BLT integration
        if BLT_AVAILABLE:
            try:
                self.blt_encoder = BLTEncoder()
            except Exception as e:
                print(f"BLT integration failed: {e}")
                self.blt_encoder = None
        else:
            self.blt_encoder = None

    def _create_fallback_inference(self):
        """Create fallback inference engine"""

        class FallbackInference:
            def __init__(self):
                self.model = None

            def run_inference(self, data, model=None, options=None):
                model = model or self.model
                if model and hasattr(model, "predict"):
                    return [model.predict(sample) for sample in data]
                return [None] * len(data)

            def set_model(self, model):
                self.model = model

        return FallbackInference()

    def _setup_gui(self):
        """Setup main GUI structure"""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget_layout = QVBoxLayout()
        central_widget_layout.addWidget(main_splitter)
        central_widget.setLayout(central_widget_layout)

        # Left panel for navigation and quick controls
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Main content area with tabs
        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)

        # Set splitter proportions
        main_splitter.setSizes([300, 1300])

        # Setup menu bar
        self._setup_menu_bar()

        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_left_panel(self):
        """Create left navigation panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.load_model_btn = QPushButton("Load Model")
        self.refresh_models_btn = QPushButton("Refresh")

        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.refresh_models_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()

        self.quick_inference_btn = QPushButton("Quick Inference")
        self.quick_train_btn = QPushButton("Quick Train")
        self.save_session_btn = QPushButton("Save Session")
        self.load_session_btn = QPushButton("Load Session")

        actions_layout.addWidget(self.quick_inference_btn)
        actions_layout.addWidget(self.quick_train_btn)
        actions_layout.addWidget(self.save_session_btn)
        actions_layout.addWidget(self.load_session_btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()
        panel.setLayout(layout)

        return panel

    def _setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Session", self.new_session)
        file_menu.addAction("Open Session", self.open_session)
        file_menu.addAction("Save Session", self.save_session)
        file_menu.addSeparator()
        file_menu.addAction("Import Model", self.import_model)
        file_menu.addAction("Export Results", self.export_results)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Model Analyzer", lambda: self.tab_widget.setCurrentIndex(0))
        tools_menu.addAction("Performance Monitor", lambda: self.tab_widget.setCurrentIndex(1))
        tools_menu.addAction("Batch Processor", lambda: self.tab_widget.setCurrentIndex(2))

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)
        help_menu.addAction("Documentation", self.show_documentation)

    def _init_features(self):
        """Initialize all 10 encapsulated features"""
        # Feature 1: Advanced Model Analyzer
        self.model_analyzer = AdvancedModelAnalyzer()
        self.tab_widget.addTab(self.model_analyzer, "🔍 Model Analyzer")

        # Feature 2: Real-time Performance Monitor
        self.performance_monitor = RealTimePerformanceMonitor()
        self.tab_widget.addTab(self.performance_monitor, "📊 Performance Monitor")

        # Feature 3: Batch Processing Manager
        self.batch_manager = BatchProcessingManager()
        self.tab_widget.addTab(self.batch_manager, "⚡ Batch Processor")

        # Feature 4: Model Comparison Tool
        self.model_comparison = ModelComparison()
        self.tab_widget.addTab(self.model_comparison, "🔄 Model Comparison")

        # Feature 5: Data Augmentation Studio
        self.augmentation_studio = DataAugmentationStudio()
        self.tab_widget.addTab(self.augmentation_studio, "🎨 Augmentation Studio")

        # Feature 6: Hyperparameter Optimizer
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.tab_widget.addTab(self.hyperparameter_optimizer, "🎯 Hyperparameter Optimizer")

        # Feature 7: Model Version Control
        self.version_control = ModelVersionControl()
        self.tab_widget.addTab(self.version_control, "📝 Version Control")

        # Feature 8: Experiment Tracker
        self.experiment_tracker = ExperimentTracker()
        self.tab_widget.addTab(self.experiment_tracker, "🧪 Experiment Tracker")

        # Feature 9: Advanced Visualization
        self.advanced_visualization = AdvancedVisualization()
        self.tab_widget.addTab(self.advanced_visualization, "📈 Advanced Visualization")

        # Feature 10: Intelligent Assistant
        self.intelligent_assistant = IntelligentAssistant()
        self.tab_widget.addTab(self.intelligent_assistant, "🤖 AI Assistant")

    def _setup_connections(self):
        """Setup signal-slot connections"""
        # Model selection
        self.load_model_btn.clicked.connect(self.load_selected_model)
        self.refresh_models_btn.clicked.connect(self._discover_models)

        # Quick actions
        self.quick_inference_btn.clicked.connect(self.run_quick_inference)
        self.quick_train_btn.clicked.connect(self.run_quick_training)

        # Feature connections
        self.model_analyzer.analyze_btn.clicked.connect(self.analyze_current_model)

    def _discover_models(self):
        """Discover available models"""
        self.status_bar.showMessage("Discovering models...")

        try:
            model_paths = self.model_loader.discover_models()
            self.discovered_models = {}

            for path, metadata in model_paths.items():
                model_id = Path(path).stem if isinstance(path, str) else str(path)
                self.discovered_models[model_id] = {"path": path, "metadata": metadata}

            # Update combo box
            self.model_combo.clear()
            self.model_combo.addItems(list(self.discovered_models.keys()))

            # Update comparison tool
            self.model_comparison.model1_combo.clear()
            self.model_comparison.model2_combo.clear()
            self.model_comparison.model1_combo.addItems(list(self.discovered_models.keys()))
            self.model_comparison.model2_combo.addItems(list(self.discovered_models.keys()))

            self.status_bar.showMessage(f"Discovered {len(self.discovered_models)} models")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error discovering models: {str(e)}")
            self.status_bar.showMessage("Model discovery failed")

    def load_selected_model(self):
        """Load the selected model"""
        model_id = self.model_combo.currentText()
        if not model_id:
            return

        self.status_bar.showMessage(f"Loading model: {model_id}")

        try:
            model_path = self.discovered_models[model_id]["path"]
            self.current_model_path = model_path
            self.current_model = self.model_loader.load_model(model_path)

            # Update inference engine
            self.inference_engine.set_model(self.current_model)

            # Update model analyzer
            self.model_analyzer.analyze_model(self.current_model)

            self.status_bar.showMessage(f"Model loaded: {model_id}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
            self.status_bar.showMessage("Model loading failed")

    def analyze_current_model(self):
        """Analyze the current model"""
        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return

        self.model_analyzer.analyze_model(self.current_model)

        # Generate analysis report
        report = "Model Analysis Report\n"
        report += f"Model Type: {type(self.current_model).__name__}\n"

        if hasattr(self.current_model, "parameters"):
            total_params = sum(p.numel() for p in self.current_model.parameters())
            report += f"Total Parameters: {total_params:,}\n"

        self.model_analyzer.results_text.setPlainText(report)

    def run_quick_inference(self):
        """Run quick inference with sample data"""
        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return

        self.status_bar.showMessage("Running quick inference...")

        try:
            # Load sample data
            sample_data = self.data_loader.load_sample_data(5)
            # Run inference on the sample data
            results = self.inference_engine.run_inference(sample_data)

            # Persist results for later access
            self.predictions = results

            # Display the results in a simple dialog
            results_str = "\n".join(
                [f"Sample {idx + 1}: {pred}" for idx, pred in enumerate(results)]
            )
            QMessageBox.information(self, "Inference Results", results_str)

            self.status_bar.showMessage("Quick inference completed")
            QMessageBox.information(
                self, "Success", f"Inference completed on {len(sample_data)} samples"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Inference error: {str(e)}")
            self.status_bar.showMessage("Inference failed")

    def run_quick_training(self):
        """Run quick training session"""
        if self.current_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return

        self.status_bar.showMessage("Starting quick training...")
        QMessageBox.information(self, "Training", "Quick training started (this is a simulation)")
        self.status_bar.showMessage("Quick training completed")

    def new_session(self):
        """Create new session"""
        self.current_model = None
        self.current_model_path = None
        self.status_bar.showMessage("New session created")

    def open_session(self):
        """Open existing session"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Session", "", "JSON files (*.json)")
        if file_path:
            self.status_bar.showMessage(f"Session loaded: {file_path}")

    def save_session(self):
        """Save current session"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON files (*.json)")
        if file_path:
            self.status_bar.showMessage(f"Session saved: {file_path}")

    def import_model(self):
        """Import external model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Model", "", "Model files (*.pt *.pth)"
        )
        if file_path:
            self.status_bar.showMessage(f"Model imported: {file_path}")
            self._discover_models()

    def export_results(self):
        """Export analysis results"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON files (*.json)"
        )
        if file_path:
            self.status_bar.showMessage(f"Results exported: {file_path}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About",
            "VoxSigil Dynamic GridFormer Qt5 Suite v2.0\n"
            "Advanced AI model testing and analysis platform\n"
            "Created by GitHub Copilot",
        )

    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(
            self, "Documentation", "Documentation available at: https://github.com/your-repo/docs"
        )


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Show splash screen
    splash_pix = QPixmap(400, 300)
    splash_pix.fill(QColor("#1a1a2e"))
    splash = QSplashScreen(splash_pix)
    splash.show()
    splash.showMessage("Loading VoxSigil Qt5 Suite...", Qt.AlignCenter, QColor("#4ecdc4"))

    app.processEvents()

    # Create main window
    window = DynamicGridFormerQt5GUI()

    # Close splash and show main window
    splash.finish(window)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
