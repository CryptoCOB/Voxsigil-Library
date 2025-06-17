"""
Enhanced Model Discovery Tab with Development Mode Controls
Advanced model discovery, analysis, and management interface.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dev_config_manager import get_dev_config
from gui.components.dev_mode_panel import DevModeControlPanel

logger = logging.getLogger(__name__)


class ModelScanWorker(QThread):
    """Background worker for comprehensive model scanning."""

    scan_progress = pyqtSignal(int)
    model_discovered = pyqtSignal(dict)
    scan_complete = pyqtSignal(int)  # Total models found

    def __init__(self, scan_config: Dict[str, Any]):
        super().__init__()
        self.scan_config = scan_config

    def run(self):
        """Run comprehensive model scan."""
        try:
            models_found = 0
            search_paths = self.scan_config.get("paths", [])
            extensions = self.scan_config.get(
                "extensions", [".pth", ".pt", ".onnx", ".safetensors"]
            )
            deep_scan = self.scan_config.get("deep_scan", False)

            total_paths = len(search_paths)

            for i, path_str in enumerate(search_paths):
                path = Path(path_str)
                if path.exists():
                    if deep_scan:
                        models_found += self._deep_scan_directory(path, extensions)
                    else:
                        models_found += self._quick_scan_directory(path, extensions)

                self.scan_progress.emit(int((i + 1) / total_paths * 100))

            self.scan_complete.emit(models_found)

        except Exception as e:
            logger.error(f"Model scan error: {e}")

    def _quick_scan_directory(self, path: Path, extensions: List[str]) -> int:
        """Quick scan of directory for model files."""
        count = 0
        try:
            for ext in extensions:
                for model_file in path.rglob(f"*{ext}"):
                    model_info = self._extract_basic_info(model_file)
                    if model_info:
                        self.model_discovered.emit(model_info)
                        count += 1
        except Exception as e:
            logger.debug(f"Quick scan error in {path}: {e}")
        return count

    def _deep_scan_directory(self, path: Path, extensions: List[str]) -> int:
        """Deep scan with detailed analysis."""
        count = 0
        try:
            for ext in extensions:
                for model_file in path.rglob(f"*{ext}"):
                    model_info = self._extract_detailed_info(model_file)
                    if model_info:
                        self.model_discovered.emit(model_info)
                        count += 1
        except Exception as e:
            logger.debug(f"Deep scan error in {path}: {e}")
        return count

    def _extract_basic_info(self, model_file: Path) -> Optional[Dict[str, Any]]:
        """Extract basic model information."""
        try:
            stat = model_file.stat()
            return {
                "name": model_file.stem,
                "path": str(model_file),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": stat.st_mtime,
                "extension": model_file.suffix,
                "type": "Model",
                "framework": self._detect_framework(model_file),
                "analyzed": False,
            }
        except Exception as e:
            logger.debug(f"Basic info extraction error for {model_file}: {e}")
            return None

    def _extract_detailed_info(self, model_file: Path) -> Optional[Dict[str, Any]]:
        """Extract detailed model information with analysis."""
        basic_info = self._extract_basic_info(model_file)
        if not basic_info:
            return None

        try:
            # Try to analyze model contents
            if model_file.suffix in [".pth", ".pt"]:
                detailed_info = self._analyze_pytorch_model(model_file)
            elif model_file.suffix == ".onnx":
                detailed_info = self._analyze_onnx_model(model_file)
            elif model_file.suffix == ".safetensors":
                detailed_info = self._analyze_safetensors_model(model_file)
            else:
                detailed_info = {}

            basic_info.update(detailed_info)
            basic_info["analyzed"] = True
            return basic_info

        except Exception as e:
            logger.debug(f"Detailed analysis error for {model_file}: {e}")
            basic_info["analysis_error"] = str(e)
            return basic_info

    def _detect_framework(self, model_file: Path) -> str:
        """Detect the ML framework based on file extension and name."""
        ext = model_file.suffix.lower()
        name = model_file.name.lower()

        if ext in [".pth", ".pt"]:
            return "PyTorch"
        elif ext == ".onnx":
            return "ONNX"
        elif ext == ".safetensors":
            return "SafeTensors"
        elif ext in [".h5", ".hdf5"]:
            return "TensorFlow/Keras"
        elif "tensorflow" in name or "tf" in name:
            return "TensorFlow"
        else:
            return "Unknown"

    def _analyze_pytorch_model(self, model_file: Path) -> Dict[str, Any]:
        """Analyze PyTorch model file."""
        try:
            import torch

            checkpoint = torch.load(model_file, map_location="cpu")

            info = {}

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    info["metadata"] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    info["metadata"] = {k: v for k, v in checkpoint.items() if k != "model"}
                else:
                    state_dict = checkpoint
                    info["metadata"] = {}

                # Count parameters
                total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))
                info["parameters"] = total_params
                info["parameters_str"] = f"{total_params:,}"

                # Analyze architecture
                info["architecture"] = self._detect_architecture(list(state_dict.keys()))
                info["layers"] = len(state_dict)

            return info

        except Exception as e:
            return {"analysis_error": str(e)}

    def _analyze_onnx_model(self, model_file: Path) -> Dict[str, Any]:
        """Analyze ONNX model file."""
        try:
            # Placeholder for ONNX analysis
            return {"framework": "ONNX", "analysis": "ONNX analysis not implemented yet"}
        except Exception as e:
            return {"analysis_error": str(e)}

    def _analyze_safetensors_model(self, model_file: Path) -> Dict[str, Any]:
        """Analyze SafeTensors model file."""
        try:
            # Placeholder for SafeTensors analysis
            return {
                "framework": "SafeTensors",
                "analysis": "SafeTensors analysis not implemented yet",
            }
        except Exception as e:
            return {"analysis_error": str(e)}

    def _detect_architecture(self, layer_names: List[str]) -> str:
        """Detect model architecture from layer names."""
        layer_str = " ".join(layer_names).lower()

        if "transformer" in layer_str or "attention" in layer_str:
            return "Transformer"
        elif "conv" in layer_str:
            return "CNN"
        elif "lstm" in layer_str or "gru" in layer_str:
            return "RNN"
        elif "embedding" in layer_str:
            return "Embedding"
        elif "linear" in layer_str or "fc" in layer_str:
            return "MLP"
        else:
            return "Unknown"


class EnhancedModelDiscoveryTab(QWidget):
    """Enhanced Model Discovery Tab with comprehensive scanning and analysis."""

    def __init__(self):
        super().__init__()
        self.config = get_dev_config()
        self.discovered_models = []
        self.scan_worker = None

        self._init_ui()
        self._setup_connections()

        # Auto-scan timer
        self.auto_scan_timer = QTimer()
        self.auto_scan_timer.timeout.connect(self._auto_scan)

        if self.config.get_tab_config("model_discovery").auto_refresh:
            self.auto_scan_timer.start(
                self.config.get_tab_config("model_discovery").refresh_interval
            )

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🔍 Advanced Model Discovery & Analysis")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #2E86AB; padding: 10px;")
        layout.addWidget(title)  # Dev Mode Panel
        self.dev_panel = DevModeControlPanel("model_discovery")
        layout.addWidget(self.dev_panel)

        # Main tabs
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Discovery Tab
        discovery_tab = self._create_discovery_tab()
        tab_widget.addTab(discovery_tab, "🔍 Discovery")

        # Analysis Tab
        analysis_tab = self._create_analysis_tab()
        tab_widget.addTab(analysis_tab, "📊 Analysis")

        # Management Tab
        management_tab = self._create_management_tab()
        tab_widget.addTab(management_tab, "⚙️ Management")

    def _create_discovery_tab(self) -> QWidget:
        """Create the main discovery tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Discovery controls
        controls_group = QGroupBox("🎯 Discovery Controls")
        controls_layout = QVBoxLayout()

        # Scan configuration
        config_layout = QHBoxLayout()

        config_layout.addWidget(QLabel("Scan Type:"))
        self.scan_type = QComboBox()
        self.scan_type.addItems(["Quick Scan", "Deep Analysis", "Custom"])
        config_layout.addWidget(self.scan_type)

        config_layout.addWidget(QLabel("Extensions:"))
        self.extensions_combo = QComboBox()
        self.extensions_combo.addItems(
            ["All", "PyTorch (.pth, .pt)", "ONNX (.onnx)", "SafeTensors (.safetensors)"]
        )
        config_layout.addWidget(self.extensions_combo)

        controls_layout.addLayout(config_layout)

        # Action buttons
        actions_layout = QHBoxLayout()

        self.add_path_btn = QPushButton("📁 Add Path")
        self.add_path_btn.clicked.connect(self._add_scan_path)
        actions_layout.addWidget(self.add_path_btn)

        self.start_scan_btn = QPushButton("🚀 Start Scan")
        self.start_scan_btn.clicked.connect(self._start_scan)
        actions_layout.addWidget(self.start_scan_btn)

        self.stop_scan_btn = QPushButton("⏹️ Stop")
        self.stop_scan_btn.clicked.connect(self._stop_scan)
        self.stop_scan_btn.setEnabled(False)
        actions_layout.addWidget(self.stop_scan_btn)

        self.clear_btn = QPushButton("🧹 Clear")
        self.clear_btn.clicked.connect(self._clear_results)
        actions_layout.addWidget(self.clear_btn)

        controls_layout.addLayout(actions_layout)

        # Progress bar
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        controls_layout.addWidget(self.scan_progress)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Results display
        results_splitter = QSplitter()
        layout.addWidget(results_splitter)

        # Scan paths and results
        left_group = QGroupBox("📍 Scan Paths & Models")
        left_layout = QVBoxLayout()

        # Scan paths
        left_layout.addWidget(QLabel("Scan Paths:"))
        self.scan_paths_list = QListWidget()
        self.scan_paths_list.setMaximumHeight(80)
        # Add default paths
        self.scan_paths_list.addItem("./models")
        self.scan_paths_list.addItem("./checkpoints")
        self.scan_paths_list.addItem("./")
        left_layout.addWidget(self.scan_paths_list)

        # Results tree
        left_layout.addWidget(QLabel("Discovered Models:"))
        self.models_tree = QTreeWidget()
        self.models_tree.setHeaderLabels(["Name", "Size", "Type", "Framework"])
        self.models_tree.itemClicked.connect(self._on_model_selected)
        left_layout.addWidget(self.models_tree)

        left_group.setLayout(left_layout)
        results_splitter.addWidget(left_group)

        # Model details
        details_group = QGroupBox("📋 Model Details")
        details_layout = QVBoxLayout()

        self.model_details = QTextEdit()
        self.model_details.setReadOnly(True)
        self.model_details.setPlainText("Select a model to view details...")
        details_layout.addWidget(self.model_details)

        # Quick actions
        quick_actions = QHBoxLayout()

        self.analyze_btn = QPushButton("🔬 Analyze")
        self.analyze_btn.clicked.connect(self._analyze_selected)
        self.analyze_btn.setEnabled(False)
        quick_actions.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self._export_selected)
        self.export_btn.setEnabled(False)
        quick_actions.addWidget(self.export_btn)

        details_layout.addLayout(quick_actions)

        details_group.setLayout(details_layout)
        results_splitter.addWidget(details_group)

        return widget

    def _create_analysis_tab(self) -> QWidget:
        """Create the analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Analysis placeholder
        layout.addWidget(QLabel("📊 Model Analysis"))
        analysis_text = QTextEdit()
        analysis_text.setPlainText(
            "Model analysis features will be available here:\n\n"
            "• Parameter distribution analysis\n"
            "• Architecture visualization\n"
            "• Performance metrics\n"
            "• Compatibility checks\n"
            "• Benchmark results"
        )
        layout.addWidget(analysis_text)

        return widget

    def _create_management_tab(self) -> QWidget:
        """Create the management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Management placeholder
        layout.addWidget(QLabel("⚙️ Model Management"))
        management_text = QTextEdit()
        management_text.setPlainText(
            "Model management features will be available here:\n\n"
            "• Model organization\n"
            "• Version tracking\n"
            "• Storage optimization\n"
            "• Metadata management\n"
            "• Model deployment"
        )
        layout.addWidget(management_text)

        return widget

    def _setup_connections(self):
        """Setup signal connections."""
        self.dev_panel.dev_mode_toggled.connect(self._on_dev_mode_changed)
        self.dev_panel.refresh_triggered.connect(self._auto_scan)

    def _add_scan_path(self):
        """Add a new scan path."""
        path = QFileDialog.getExistingDirectory(self, "Select Directory to Scan")
        if path:
            self.scan_paths_list.addItem(path)

    def _start_scan(self):
        """Start model discovery scan."""
        if self.scan_worker and self.scan_worker.isRunning():
            return

        # Get scan configuration
        scan_paths = []
        for i in range(self.scan_paths_list.count()):
            scan_paths.append(self.scan_paths_list.item(i).text())

        if not scan_paths:
            QMessageBox.warning(self, "Warning", "No scan paths specified!")
            return

        # Configure scan
        scan_config = {
            "paths": scan_paths,
            "deep_scan": self.scan_type.currentText() == "Deep Analysis",
            "extensions": self._get_selected_extensions(),
        }

        # Clear previous results
        self.models_tree.clear()
        self.discovered_models.clear()

        # Start scan
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        self.start_scan_btn.setEnabled(False)
        self.stop_scan_btn.setEnabled(True)

        self.scan_worker = ModelScanWorker(scan_config)
        self.scan_worker.scan_progress.connect(self.scan_progress.setValue)
        self.scan_worker.model_discovered.connect(self._on_model_discovered)
        self.scan_worker.scan_complete.connect(self._on_scan_complete)
        self.scan_worker.start()

    def _get_selected_extensions(self) -> List[str]:
        """Get selected file extensions for scanning."""
        selection = self.extensions_combo.currentText()

        if selection == "All":
            return [".pth", ".pt", ".onnx", ".safetensors", ".h5", ".hdf5"]
        elif selection == "PyTorch (.pth, .pt)":
            return [".pth", ".pt"]
        elif selection == "ONNX (.onnx)":
            return [".onnx"]
        elif selection == "SafeTensors (.safetensors)":
            return [".safetensors"]
        else:
            return [".pth", ".pt"]

    def _stop_scan(self):
        """Stop the current scan."""
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.terminate()
            self.scan_worker.wait()

        self.scan_progress.setVisible(False)
        self.start_scan_btn.setEnabled(True)
        self.stop_scan_btn.setEnabled(False)

    def _clear_results(self):
        """Clear scan results."""
        self.models_tree.clear()
        self.discovered_models.clear()
        self.model_details.setPlainText("Select a model to view details...")

    def _on_model_discovered(self, model_info: Dict[str, Any]):
        """Handle discovered model."""
        self.discovered_models.append(model_info)

        # Add to tree
        item = QTreeWidgetItem(self.models_tree)
        item.setText(0, model_info["name"])
        item.setText(1, f"{model_info['size_mb']:.1f} MB")
        item.setText(2, model_info.get("type", "Model"))
        item.setText(3, model_info.get("framework", "Unknown"))
        item.setData(0, 0, model_info)

    def _on_scan_complete(self, total_models: int):
        """Handle scan completion."""
        self.scan_progress.setVisible(False)
        self.start_scan_btn.setEnabled(True)
        self.stop_scan_btn.setEnabled(False)

        self.model_details.setPlainText(f"Scan complete!\nFound {total_models} models.")

    def _on_model_selected(self, item: QTreeWidgetItem):
        """Handle model selection."""
        model_info = item.data(0, 0)
        if model_info:
            self._display_model_details(model_info)
            self.analyze_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

    def _display_model_details(self, model_info: Dict[str, Any]):
        """Display detailed model information."""
        details = f"Model: {model_info['name']}\n"
        details += f"Path: {model_info['path']}\n"
        details += f"Size: {model_info['size_mb']:.1f} MB\n"
        details += f"Framework: {model_info.get('framework', 'Unknown')}\n"
        details += f"Extension: {model_info.get('extension', 'Unknown')}\n"

        if model_info.get("analyzed", False):
            details += f"\nParameters: {model_info.get('parameters_str', 'Unknown')}\n"
            details += f"Architecture: {model_info.get('architecture', 'Unknown')}\n"
            details += f"Layers: {model_info.get('layers', 'Unknown')}\n"

            if "metadata" in model_info and model_info["metadata"]:
                details += "\nMetadata:\n"
                for key, value in model_info["metadata"].items():
                    details += f"  {key}: {value}\n"

        if "analysis_error" in model_info:
            details += f"\nAnalysis Error: {model_info['analysis_error']}\n"

        self.model_details.setPlainText(details)

    def _analyze_selected(self):
        """Analyze selected model."""
        current_item = self.models_tree.currentItem()
        if current_item:
            model_info = current_item.data(0, 0)
            QMessageBox.information(
                self,
                "Analysis",
                f"Detailed analysis for {model_info['name']} will be implemented here.",
            )

    def _export_selected(self):
        """Export selected model information."""
        current_item = self.models_tree.currentItem()
        if current_item:
            model_info = current_item.data(0, 0)

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Model Info",
                f"{model_info['name']}_discovery.json",
                "JSON Files (*.json)",
            )

            if file_path:
                with open(file_path, "w") as f:
                    json.dump(model_info, f, indent=2, default=str)

                QMessageBox.information(
                    self, "Export", f"Model information exported to:\n{file_path}"
                )

    def _auto_scan(self):
        """Perform automatic scan."""
        if self.config.get_tab_config("model_discovery").auto_refresh:
            self._start_scan()

    def _on_dev_mode_changed(self, enabled: bool):
        """Handle dev mode toggle."""
        self.config.update_tab_config("model_discovery", dev_mode=enabled)
        # Additional dev mode functionality can be added here
