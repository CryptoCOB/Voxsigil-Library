#!/usr/bin/env python3
"""
VoxSigil Model Discovery Tab Interface - Qt5 Version
Modular component for model discovery and selection functionality

Created by: GitHub Copilot
Purpose: Encapsulated model discovery interface for Dynamic GridFormer GUI
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QTextEdit, QGroupBox, QSplitter,
    QMessageBox, QFileDialog, QProgressBar, QComboBox,
    QCheckBox, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPen, QBrush
import torch
import json
import time


class ModelValidationWorker(QThread):
    """Background worker for model validation"""
    validation_complete = pyqtSignal(str, dict)
    progress_update = pyqtSignal(int)
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        
    def run(self):
        """Validate model in background"""
        try:
            self.progress_update.emit(25)
            
            # Basic file validation
            if not Path(self.model_path).exists():
                raise FileNotFoundError("Model file not found")
                
            self.progress_update.emit(50)
            
            # Try to load model
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            self.progress_update.emit(75)
            
            # Extract validation info
            validation_info = {
                "valid": True,
                "type": type(checkpoint).__name__,
                "size_mb": Path(self.model_path).stat().st_size / (1024 * 1024),
                "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
            }
            
            self.progress_update.emit(100)
            self.validation_complete.emit(self.model_path, validation_info)
            
        except Exception as e:
            validation_info = {"valid": False, "error": str(e)}
            self.validation_complete.emit(self.model_path, validation_info)


class ModelComparison:
    """Feature 1: Model Comparison System"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.comparison_models = []
        
    def create_comparison_widget(self):
        """Create model comparison interface"""
        comparison_group = QGroupBox("Model Comparison")
        layout = QVBoxLayout()
        
        # Comparison table
        self.comparison_table = QTableWidget(0, 5)
        self.comparison_table.setHorizontalHeaderLabels([
            "Model Name", "Size (MB)", "Parameters", "Type", "Performance"
        ])
        layout.addWidget(self.comparison_table)
        
        # Comparison controls
        controls_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add to Comparison")
        add_btn.clicked.connect(self.add_model_to_comparison)
        controls_layout.addWidget(add_btn)
        
        clear_btn = QPushButton("Clear Comparison")
        clear_btn.clicked.connect(self.clear_comparison)
        controls_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_comparison_report)
        controls_layout.addWidget(export_btn)
        
        layout.addLayout(controls_layout)
        comparison_group.setLayout(layout)
        return comparison_group
    
    def add_model_to_comparison(self):
        """Add selected model to comparison"""
        # Implementation would connect to main model selection
        pass
    
    def clear_comparison(self):
        """Clear comparison table"""
        self.comparison_table.setRowCount(0)
        self.comparison_models.clear()
    
    def export_comparison_report(self):
        """Export comparison as JSON report"""
        filename, _ = QFileDialog.getSaveFileName(
            self.parent, "Save Comparison Report", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.comparison_models, f, indent=2)


class ModelBookmarks:
    """Feature 2: Model Bookmarking System"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.bookmarks_file = Path("model_bookmarks.json")
        self.bookmarks = self.load_bookmarks()
        
    def create_bookmarks_widget(self):
        """Create bookmarks interface"""
        bookmarks_group = QGroupBox("Model Bookmarks")
        layout = QVBoxLayout()
        
        # Bookmarks list
        self.bookmarks_list = QListWidget()
        self.refresh_bookmarks_list()
        layout.addWidget(self.bookmarks_list)
        
        # Bookmark controls
        controls_layout = QHBoxLayout()
        
        add_bookmark_btn = QPushButton("📌 Bookmark Current")
        add_bookmark_btn.clicked.connect(self.add_bookmark)
        controls_layout.addWidget(add_bookmark_btn)
        
        remove_bookmark_btn = QPushButton("🗑️ Remove")
        remove_bookmark_btn.clicked.connect(self.remove_bookmark)
        controls_layout.addWidget(remove_bookmark_btn)
        
        load_bookmark_btn = QPushButton("📂 Load Bookmarked")
        load_bookmark_btn.clicked.connect(self.load_bookmarked_model)
        controls_layout.addWidget(load_bookmark_btn)
        
        layout.addLayout(controls_layout)
        bookmarks_group.setLayout(layout)
        return bookmarks_group
    
    def load_bookmarks(self):
        """Load bookmarks from file"""
        if self.bookmarks_file.exists():
            with open(self.bookmarks_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_bookmarks(self):
        """Save bookmarks to file"""
        with open(self.bookmarks_file, 'w') as f:
            json.dump(self.bookmarks, f, indent=2)
    
    def refresh_bookmarks_list(self):
        """Refresh the bookmarks list widget"""
        self.bookmarks_list.clear()
        for name, info in self.bookmarks.items():
            self.bookmarks_list.addItem(f"{name} - {info.get('path', 'Unknown')}")
    
    def add_bookmark(self):
        """Add current model to bookmarks"""
        # Would get current model from parent
        pass
    
    def remove_bookmark(self):
        """Remove selected bookmark"""
        current_item = self.bookmarks_list.currentItem()
        if current_item:
            bookmark_name = current_item.text().split(" - ")[0]
            del self.bookmarks[bookmark_name]
            self.save_bookmarks()
            self.refresh_bookmarks_list()
    
    def load_bookmarked_model(self):
        """Load selected bookmarked model"""
        current_item = self.bookmarks_list.currentItem()
        if current_item:
            bookmark_name = current_item.text().split(" - ")[0]
            model_path = self.bookmarks[bookmark_name]["path"]
            # Would trigger model loading in parent
            pass


class AutoModelScanner:
    """Feature 3: Automatic Model Scanning"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.perform_scan)
        self.scan_enabled = False
        
    def create_scanner_widget(self):
        """Create auto-scanner interface"""
        scanner_group = QGroupBox("Auto Model Scanner")
        layout = QVBoxLayout()
        
        # Scanner controls
        controls_layout = QHBoxLayout()
        
        self.enable_checkbox = QCheckBox("Enable Auto-Scan")
        self.enable_checkbox.toggled.connect(self.toggle_auto_scan)
        controls_layout.addWidget(self.enable_checkbox)
        
        interval_label = QLabel("Interval (minutes):")
        controls_layout.addWidget(interval_label)
        
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1, 60)
        self.interval_spinbox.setValue(5)
        self.interval_spinbox.valueChanged.connect(self.update_scan_interval)
        controls_layout.addWidget(self.interval_spinbox)
        
        layout.addLayout(controls_layout)
        
        # Scan status
        self.scan_status_label = QLabel("Scanner: Disabled")
        layout.addWidget(self.scan_status_label)
        
        # Last scan results
        self.scan_results = QTextEdit()
        self.scan_results.setMaximumHeight(100)
        self.scan_results.setPlainText("No scans performed yet...")
        layout.addWidget(self.scan_results)
        
        scanner_group.setLayout(layout)
        return scanner_group
    
    def toggle_auto_scan(self, enabled):
        """Toggle automatic scanning"""
        self.scan_enabled = enabled
        if enabled:
            self.scan_timer.start(self.interval_spinbox.value() * 60000)  # Convert to ms
            self.scan_status_label.setText("Scanner: Active")
        else:
            self.scan_timer.stop()
            self.scan_status_label.setText("Scanner: Disabled")
    
    def update_scan_interval(self, value):
        """Update scan interval"""
        if self.scan_enabled:
            self.scan_timer.start(value * 60000)
    
    def perform_scan(self):
        """Perform automatic model scan"""
        timestamp = time.strftime("%H:%M:%S")
        # Would trigger model discovery in parent
        self.scan_results.append(f"[{timestamp}] Auto-scan completed")


class ModelFilters:
    """Feature 4: Advanced Model Filtering"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.current_filters = {}
        
    def create_filters_widget(self):
        """Create filtering interface"""
        filters_group = QGroupBox("Model Filters")
        layout = QVBoxLayout()
        
        # Filter by type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["All", "PyTorch (.pt)", "Checkpoint (.ckpt)", "ONNX (.onnx)"])
        self.type_combo.currentTextChanged.connect(self.apply_filters)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Filter by size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Max Size (MB):"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(0, 10000)
        self.size_spinbox.setValue(1000)
        self.size_spinbox.valueChanged.connect(self.apply_filters)
        size_layout.addWidget(self.size_spinbox)
        layout.addLayout(size_layout)
        
        # Filter by recency
        recency_layout = QHBoxLayout()
        self.recent_checkbox = QCheckBox("Show only recent (last 7 days)")
        self.recent_checkbox.toggled.connect(self.apply_filters)
        recency_layout.addWidget(self.recent_checkbox)
        layout.addLayout(recency_layout)
        
        # Clear filters button
        clear_btn = QPushButton("Clear All Filters")
        clear_btn.clicked.connect(self.clear_filters)
        layout.addWidget(clear_btn)
        
        filters_group.setLayout(layout)
        return filters_group
    
    def apply_filters(self):
        """Apply current filters to model list"""
        self.current_filters = {
            "type": self.type_combo.currentText(),
            "max_size": self.size_spinbox.value(),
            "recent_only": self.recent_checkbox.isChecked()
        }
        # Would trigger filtering in parent model list
        
    def clear_filters(self):
        """Clear all filters"""
        self.type_combo.setCurrentText("All")
        self.size_spinbox.setValue(1000)
        self.recent_checkbox.setChecked(False)
        self.current_filters = {}


class ModelMetricsTracker:
    """Feature 5: Model Performance Metrics Tracking"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.metrics_history = {}
        
    def create_metrics_widget(self):
        """Create metrics tracking interface"""
        metrics_group = QGroupBox("Model Metrics Tracker")
        layout = QVBoxLayout()
        
        # Metrics table
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels([
            "Model", "Load Time (s)", "Memory Usage (MB)", "Last Used"
        ])
        layout.addWidget(self.metrics_table)
        
        # Metrics controls
        controls_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Metrics")
        refresh_btn.clicked.connect(self.refresh_metrics)
        controls_layout.addWidget(refresh_btn)
        
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self.clear_metrics)
        controls_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("Export Metrics")
        export_btn.clicked.connect(self.export_metrics)
        controls_layout.addWidget(export_btn)
        
        layout.addLayout(controls_layout)
        metrics_group.setLayout(layout)
        return metrics_group
    
    def record_model_load(self, model_name, load_time, memory_usage):
        """Record model loading metrics"""
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
        
        self.metrics_history[model_name].append({
            "timestamp": time.time(),
            "load_time": load_time,
            "memory_usage": memory_usage
        })
        self.refresh_metrics()
    
    def refresh_metrics(self):
        """Refresh metrics display"""
        self.metrics_table.setRowCount(len(self.metrics_history))
        
        for row, (model_name, history) in enumerate(self.metrics_history.items()):
            if history:
                latest = history[-1]
                self.metrics_table.setItem(row, 0, QTableWidgetItem(model_name))
                self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{latest['load_time']:.2f}"))
                self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{latest['memory_usage']:.1f}"))
                self.metrics_table.setItem(row, 3, QTableWidgetItem(
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(latest['timestamp']))
                ))
    
    def clear_metrics(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        self.metrics_table.setRowCount(0)
    
    def export_metrics(self):
        """Export metrics to JSON"""
        filename, _ = QFileDialog.getSaveFileName(
            self.parent, "Save Metrics", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)


class VoxSigilModelInterface(QWidget):
    """Qt5 Model discovery and selection interface for VoxSigil Dynamic GridFormer"""

    def __init__(self, parent_gui, tab_widget):
        super().__init__()
        self.parent_gui = parent_gui
        self.tab_widget = tab_widget

        # Model state
        self.discovered_models = {}
        self.current_model = None
        self.current_model_path = None

        # Initialize encapsulated features
        self.model_comparison = ModelComparison(self)
        self.model_bookmarks = ModelBookmarks(self)
        self.auto_scanner = AutoModelScanner(self)
        self.model_filters = ModelFilters(self)
        self.metrics_tracker = ModelMetricsTracker(self)

        # Create the interface
        self.setup_ui()
        self.apply_styles()

        # Add to tab widget
        self.tab_widget.addTab(self, "🔍 Model Discovery")

        # Discover models on initialization
        self.discover_models()

    def setup_ui(self):
        """Setup the Qt5 user interface"""
        main_layout = QVBoxLayout()
        
        # Create main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Discovery and filtering
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel: Model information and features
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([400, 600])
        main_layout.addWidget(main_splitter)
        
        # Bottom panel: Relationships graph
        bottom_panel = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel)
        
        self.setLayout(main_layout)

    def create_left_panel(self):
        """Create the left panel with discovery controls and filtering"""
        left_widget = QWidget()
        layout = QVBoxLayout()
        
        # Discovery controls
        discovery_group = QGroupBox("Model Discovery")
        discovery_layout = QVBoxLayout()
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        auto_discover_btn = QPushButton("🔍 Auto-Discover Models")
        auto_discover_btn.clicked.connect(self.discover_models)
        buttons_layout.addWidget(auto_discover_btn)
        
        load_custom_btn = QPushButton("📂 Load Custom Model")
        load_custom_btn.clicked.connect(self.load_custom_model)
        buttons_layout.addWidget(load_custom_btn)
        
        discovery_layout.addLayout(buttons_layout)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_select)
        discovery_layout.addWidget(self.model_list)
        
        # Load button
        load_btn = QPushButton("⚡ Load Selected Model")
        load_btn.clicked.connect(self.load_selected_model)
        discovery_layout.addWidget(load_btn)
        
        # Validation progress
        self.validation_progress = QProgressBar()
        self.validation_progress.setVisible(False)
        discovery_layout.addWidget(self.validation_progress)
        
        discovery_group.setLayout(discovery_layout)
        layout.addWidget(discovery_group)
        
        # Add filtering feature
        layout.addWidget(self.model_filters.create_filters_widget())
        
        # Add auto-scanner feature
        layout.addWidget(self.auto_scanner.create_scanner_widget())
        
        left_widget.setLayout(layout)
        return left_widget

    def create_right_panel(self):
        """Create the right panel with model info and features"""
        right_widget = QWidget()
        layout = QVBoxLayout()
        
        # Current model status
        status_group = QGroupBox("Current Model Status")
        status_layout = QVBoxLayout()
        
        self.current_model_label = QLabel("No model loaded")
        self.current_model_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_layout.addWidget(self.current_model_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Create tabbed features panel
        features_tabs = QTabWidget()
        
        # Model details tab
        details_widget = QWidget()
        details_layout = QVBoxLayout()
        
        self.model_details_text = QTextEdit()
        self.model_details_text.setPlainText("Select a model to view details...")
        self.model_details_text.setReadOnly(True)
        details_layout.addWidget(self.model_details_text)
        
        details_widget.setLayout(details_layout)
        features_tabs.addTab(details_widget, "📄 Details")
        
        # Add feature tabs
        features_tabs.addTab(self.model_comparison.create_comparison_widget(), "⚖️ Compare")
        features_tabs.addTab(self.model_bookmarks.create_bookmarks_widget(), "📌 Bookmarks")
        features_tabs.addTab(self.metrics_tracker.create_metrics_widget(), "📊 Metrics")
        
        layout.addWidget(features_tabs)
        right_widget.setLayout(layout)
        return right_widget

    def create_bottom_panel(self):
        """Create the bottom panel with relationship graph"""
        graph_group = QGroupBox("Agent–Component Relationships")
        layout = QVBoxLayout()
        
        # Graphics view for relationship visualization
        self.relationships_view = QGraphicsView()
        self.relationships_scene = QGraphicsScene()
        self.relationships_view.setScene(self.relationships_scene)
        self.relationships_view.setMaximumHeight(200)
        layout.addWidget(self.relationships_view)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Graph")
        refresh_btn.clicked.connect(self.draw_relationship_graph)
        layout.addWidget(refresh_btn)
        
        graph_group.setLayout(layout)
        
        # Initial draw
        self.draw_relationship_graph()
        
        return graph_group

    def apply_styles(self):
        """Apply VoxSigil-style dark theme"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #16213e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
            QPushButton:pressed {
                background-color: #533483;
            }
            QListWidget, QTextEdit, QTableWidget {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 4px;
                selection-background-color: #533483;
            }
            QTabWidget::pane {
                border: 1px solid #16213e;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #16213e;
                border: 1px solid #0f3460;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #533483;
            }
        """)

    def discover_models(self):
        """Discover available models in the workspace"""
        self.discovered_models.clear()
        self.model_list.clear()

        # Search patterns for model files
        search_paths = [
            Path("models"),
            Path("active/core"),
            Path("enhanced_training_project"),
            Path.cwd(),
        ]

        model_extensions = [".pt", ".pth", ".ckpt", ".model", ".onnx"]

        for search_path in search_paths:
            if search_path.exists():
                for ext in model_extensions:
                    for model_file in search_path.rglob(f"*{ext}"):
                        if model_file.is_file():
                            model_name = model_file.stem
                            self.discovered_models[model_name] = str(model_file)
                            self.model_list.addItem(f"{model_name} ({model_file.name})")

        # Update model details
        self.update_model_details(
            f"Discovered {len(self.discovered_models)} models:\n\n"
            + "\n".join(
                [f"• {name}: {path}" for name, path in self.discovered_models.items()]
            )
        )

        # Update auto-scanner results
        if hasattr(self.auto_scanner, 'scan_results'):
            timestamp = time.strftime("%H:%M:%S")
            self.auto_scanner.scan_results.append(
                f"[{timestamp}] Found {len(self.discovered_models)} models"
            )

    def load_custom_model(self):
        """Load a custom model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "PyTorch Models (*.pt *.pth);;Checkpoint Files (*.ckpt);;ONNX Models (*.onnx);;All Files (*.*)"
        )

        if file_path:
            model_name = Path(file_path).stem
            self.discovered_models[model_name] = file_path
            self.model_list.addItem(f"{model_name} (custom)")

    def on_model_select(self, item):
        """Handle model selection from list"""
        selected_text = item.text()
        model_name = selected_text.split(" (")[0]

        if model_name in self.discovered_models:
            model_path = self.discovered_models[model_name]
            self.show_model_info(model_name, model_path)
            
            # Start background validation
            self.start_model_validation(model_path)

    def start_model_validation(self, model_path):
        """Start background model validation"""
        self.validation_progress.setVisible(True)
        self.validation_progress.setValue(0)
        
        self.validation_worker = ModelValidationWorker(model_path)
        self.validation_worker.validation_complete.connect(self.on_validation_complete)
        self.validation_worker.progress_update.connect(self.validation_progress.setValue)
        self.validation_worker.start()

    def on_validation_complete(self, model_path, validation_info):
        """Handle validation completion"""
        self.validation_progress.setVisible(False)
        
        if validation_info.get("valid", False):
            self.model_details_text.append(f"\n✅ Validation: Model is valid")
        else:
            self.model_details_text.append(f"\n❌ Validation: {validation_info.get('error', 'Unknown error')}")

    def show_model_info(self, model_name, model_path):
        """Display detailed information about selected model"""
        details = f"Model: {model_name}\n"
        details += f"Path: {model_path}\n"
        details += f"File Size: {Path(model_path).stat().st_size / (1024 * 1024):.2f} MB\n\n"

        try:
            if model_path.endswith((".pt", ".pth")):
                checkpoint = torch.load(model_path, map_location="cpu")

                if isinstance(checkpoint, dict):
                    details += "Checkpoint Contents:\n"
                    for key in checkpoint.keys():
                        if key == "model_state_dict":
                            details += f"  • {key}: {len(checkpoint[key])} parameters\n"
                        elif key == "optimizer_state_dict":
                            details += f"  • {key}: Optimizer state\n"
                        else:
                            details += f"  • {key}: {type(checkpoint[key]).__name__}\n"

                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        details += f"\nModel Parameters: {len(state_dict)} layers\n"
                        details += "Layer Names:\n"
                        for i, layer_name in enumerate(list(state_dict.keys())[:10]):
                            details += f"  • {layer_name}\n"
                        if len(state_dict) > 10:
                            details += f"  ... and {len(state_dict) - 10} more layers\n"
                else:
                    details += f"Direct model object: {type(checkpoint).__name__}\n"

        except Exception as e:
            details += f"\nError loading model info: {str(e)}\n"

        self.update_model_details(details)

    def update_model_details(self, details):
        """Update the model details text widget"""
        self.model_details_text.setPlainText(details)

    def load_selected_model(self):
        """Load the currently selected model"""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a model to load.")
            return

        selected_text = current_item.text()
        model_name = selected_text.split(" (")[0]

        if model_name in self.discovered_models:
            model_path = self.discovered_models[model_name]

            try:
                # Record load start time for metrics
                start_time = time.time()
                
                # Use parent GUI's model loader
                success = self.parent_gui.model_loader.load_model(model_path)

                if success:
                    load_time = time.time() - start_time
                    
                    self.current_model = model_name
                    self.current_model_path = model_path
                    self.current_model_label.setText(f"✅ {model_name}")

                    # Update parent GUI state
                    self.parent_gui.current_model = self.parent_gui.model_loader.model
                    self.parent_gui.current_model_path = model_path

                    # Record metrics
                    memory_usage = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                    self.metrics_tracker.record_model_load(model_name, load_time, memory_usage)

                    QMessageBox.information(
                        self, "Success", f"Model '{model_name}' loaded successfully!"
                    )

                else:
                    QMessageBox.critical(
                        self, "Error", f"Failed to load model '{model_name}'"
                    )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
        else:
            QMessageBox.critical(
                self, "Error", "Selected model not found in discovered models."
            )

    def draw_relationship_graph(self):
        """Draw a simple graph of registered agents and their capabilities"""
        self.relationships_scene.clear()
        
        # Fetch agents from supervisor integration
        agents = []
        if (hasattr(self.parent_gui, "voxsigil_integration") 
            and self.parent_gui.voxsigil_integration):
            try:
                registry = self.parent_gui.voxsigil_integration.unified_core.agent_registry
                for name, agent in registry.get_all_agents():
                    meta = registry.agents.get(name, {}).get("metadata", {})
                    caps = meta.get("capabilities", [])
                    agent_type = meta.get("type", "Unknown")
                    status = meta.get("status", "Inactive")
                    agents.append((name, caps, agent_type, status))
            except Exception:
                agents = []

        # Layout agents in the scene
        x, y = 10, 10
        for i, (name, caps, agent_type, status) in enumerate(agents):
            # Draw agent node
            ellipse = QGraphicsEllipseItem(x, y, 20, 20)
            ellipse.setBrush(QBrush(QColor("#ffffff")))
            ellipse.setPen(QPen(QColor("#ffffff"), 2))
            self.relationships_scene.addItem(ellipse)
            
            # Add text
            cap_str = ", ".join(caps) if caps else "(no caps)"
            text = QGraphicsTextItem(f"{name} ({agent_type}, {status}): {cap_str}")
            text.setPos(x + 30, y)
            text.setDefaultTextColor(QColor("#ffffff"))
            self.relationships_scene.addItem(text)
            
            y += 30

        if not agents:
            text = QGraphicsTextItem("No agents registered")
            text.setPos(10, 10)
            text.setDefaultTextColor(QColor("#888888"))
            self.relationships_scene.addItem(text)

    def get_current_model_info(self):
        """Get information about the currently loaded model"""
        return {
            "name": self.current_model,
            "path": self.current_model_path,
            "model_object": getattr(self.parent_gui, "current_model", None),
        }
