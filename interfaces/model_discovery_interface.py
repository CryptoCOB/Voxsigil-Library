"""
Model Discovery Interface Module
Migrated to Qt5 with enhanced features for the Dynamic GridFormer GUI
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QSplitter, QTextEdit,
    QProgressBar, QComboBox, QCheckBox, QMessageBox, QFileDialog,
    QHeaderView, QFrame, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon


class ModelScannerThread(QThread):
    """Background thread for model scanning"""
    model_found = pyqtSignal(str, dict)  # model_name, metadata
    scan_complete = pyqtSignal(int)  # count of models found
    scan_progress = pyqtSignal(int)  # progress percentage
    
    def __init__(self, directory, model_loader=None):
        super().__init__()
        self.directory = directory
        self.model_loader = model_loader
        
    def run(self):
        """Run the model scanning process"""
        discovered_models = {}
        
        try:
            if self.model_loader and hasattr(self.model_loader, "discover_models"):
                discovered = self.model_loader.discover_models(self.directory)
                total = len(discovered)
                for i, (path, metadata) in enumerate(discovered.items()):
                    model_name = Path(path).stem if isinstance(path, str) else str(path)
                    self.model_found.emit(model_name, {
                        "path": path,
                        "metadata": metadata
                    })
                    self.scan_progress.emit(int((i + 1) / total * 100))
                    
            else:
                # Manual discovery
                model_extensions = [".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".sigil"]
                directory_path = Path(self.directory)
                
                if directory_path.exists():
                    all_files = []
                    for ext in model_extensions:
                        all_files.extend(list(directory_path.glob(f"**/*{ext}")))
                    
                    total = len(all_files)
                    for i, model_file in enumerate(all_files):
                        model_name = model_file.stem
                        size = model_file.stat().st_size
                        ext = model_file.suffix
                        
                        metadata = {
                            "type": "PyTorch" if ext in [".pt", ".pth"] else "Unknown",
                            "size": size,
                            "extension": ext,
                            "path": str(model_file)
                        }
                        
                        self.model_found.emit(model_name, {"path": str(model_file), "metadata": metadata})
                        self.scan_progress.emit(int((i + 1) / total * 100))
                        
        except Exception as e:
            print(f"Error during model scan: {e}")
            
        self.scan_complete.emit(len(discovered_models))


class ModelFilterWidget(QWidget):
    """Feature 1: Advanced Model Filtering"""
    filter_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Type filter
        layout.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "PyTorch", "SafeTensors", "Checkpoint", "Unknown"])
        self.type_filter.currentTextChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.type_filter)
        
        # Size filter
        layout.addWidget(QLabel("Min Size:"))
        self.size_filter = QComboBox()
        self.size_filter.addItems(["Any", "1MB+", "10MB+", "100MB+", "1GB+"])
        self.size_filter.currentTextChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.size_filter)
        
        # Name filter
        layout.addWidget(QLabel("Name:"))
        self.name_filter = QLineEdit()
        self.name_filter.setPlaceholderText("Filter by name...")
        self.name_filter.textChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.name_filter)
        
        self.setLayout(layout)
        
    def get_filters(self):
        """Get current filter settings"""
        return {
            "type": self.type_filter.currentText(),
            "size": self.size_filter.currentText(),
            "name": self.name_filter.text().lower()
        }


class ModelPreviewWidget(QWidget):
    """Feature 2: Model Preview and Details"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model details
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        layout.addWidget(QLabel("Model Details:"))
        layout.addWidget(self.details_text)
        
        # Quick actions
        actions_layout = QHBoxLayout()
        self.validate_btn = QPushButton("Validate Model")
        self.info_btn = QPushButton("Model Info")
        actions_layout.addWidget(self.validate_btn)
        actions_layout.addWidget(self.info_btn)
        layout.addLayout(actions_layout)
        
        self.setLayout(layout)
        
    def update_preview(self, model_name, metadata):
        """Update the preview with model information"""
        details = f"Model: {model_name}\n"
        if metadata:
            for key, value in metadata.items():
                if key == "size" and isinstance(value, (int, float)):
                    if value > 1024 * 1024 * 1024:
                        value = f"{value / 1024 / 1024 / 1024:.2f} GB"
                    elif value > 1024 * 1024:
                        value = f"{value / 1024 / 1024:.2f} MB"
                    elif value > 1024:
                        value = f"{value / 1024:.2f} KB"
                    else:
                        value = f"{value} B"
                details += f"{key.title()}: {value}\n"
        
        self.details_text.setText(details)


class ModelComparisonWidget(QWidget):
    """Feature 3: Model Comparison Tool"""
    
    def __init__(self):
        super().__init__()
        self.compared_models = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Model Comparison:"))
        
        # Comparison list
        self.comparison_tree = QTreeWidget()
        self.comparison_tree.setHeaderLabels(["Model", "Type", "Size", "Parameters"])
        layout.addWidget(self.comparison_tree)
        
        # Comparison actions
        actions_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add to Comparison")
        self.clear_btn = QPushButton("Clear Comparison")
        self.compare_btn = QPushButton("Generate Report")
        
        actions_layout.addWidget(self.add_btn)
        actions_layout.addWidget(self.clear_btn)
        actions_layout.addWidget(self.compare_btn)
        layout.addLayout(actions_layout)
        
        self.setLayout(layout)
        
    def add_model_to_comparison(self, model_name, metadata):
        """Add a model to the comparison list"""
        if model_name not in [item.text(0) for item in self.get_all_items()]:
            item = QTreeWidgetItem([
                model_name,
                metadata.get("type", "Unknown"),
                self.format_size(metadata.get("size", 0)),
                metadata.get("parameters", "Unknown")
            ])
            self.comparison_tree.addTopLevelItem(item)
            
    def get_all_items(self):
        """Get all items in the comparison tree"""
        items = []
        root = self.comparison_tree.invisibleRootItem()
        for i in range(root.childCount()):
            items.append(root.child(i))
        return items
        
    def format_size(self, size):
        """Format file size for display"""
        if isinstance(size, (int, float)):
            if size > 1024 * 1024 * 1024:
                return f"{size / 1024 / 1024 / 1024:.2f} GB"
            elif size > 1024 * 1024:
                return f"{size / 1024 / 1024:.2f} MB"
            else:
                return f"{size / 1024:.2f} KB"
        return str(size)


class ModelHistoryWidget(QWidget):
    """Feature 4: Model Loading History"""
    
    def __init__(self):
        super().__init__()
        self.history = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Recent Models:"))
        
        # History list
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["Model", "Loaded", "Status"])
        layout.addWidget(self.history_tree)
        
        # History actions
        actions_layout = QHBoxLayout()
        self.reload_btn = QPushButton("Reload Selected")
        self.clear_history_btn = QPushButton("Clear History")
        
        actions_layout.addWidget(self.reload_btn)
        actions_layout.addWidget(self.clear_history_btn)
        layout.addLayout(actions_layout)
        
        self.setLayout(layout)
        
    def add_to_history(self, model_name, status="Loaded"):
        """Add a model to the loading history"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        item = QTreeWidgetItem([model_name, timestamp, status])
        self.history_tree.insertTopLevelItem(0, item)  # Add to top
        
        # Limit history to 10 items
        if self.history_tree.topLevelItemCount() > 10:
            self.history_tree.takeTopLevelItem(10)


class ModelWatcherWidget(QWidget):
    """Feature 5: Auto-refresh and Model Watching"""
    
    def __init__(self, parent_interface):
        super().__init__()
        self.parent_interface = parent_interface
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_updates)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Auto-refresh settings
        refresh_layout = QHBoxLayout()
        self.auto_refresh_cb = QCheckBox("Auto-refresh")
        self.auto_refresh_cb.toggled.connect(self.toggle_auto_refresh)
        
        self.refresh_interval = QComboBox()
        self.refresh_interval.addItems(["30s", "1m", "5m", "10m"])
        self.refresh_interval.setCurrentText("5m")
        
        refresh_layout.addWidget(self.auto_refresh_cb)
        refresh_layout.addWidget(QLabel("Interval:"))
        refresh_layout.addWidget(self.refresh_interval)
        refresh_layout.addStretch()
        
        layout.addLayout(refresh_layout)
        
        # Manual refresh
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self.manual_refresh)
        layout.addWidget(self.refresh_btn)
        
        self.setLayout(layout)
        
    def toggle_auto_refresh(self, enabled):
        """Toggle auto-refresh functionality"""
        if enabled:
            interval_text = self.refresh_interval.currentText()
            interval_ms = self.parse_interval(interval_text)
            self.timer.start(interval_ms)
        else:
            self.timer.stop()
            
    def parse_interval(self, interval_text):
        """Parse interval string to milliseconds"""
        if interval_text.endswith('s'):
            return int(interval_text[:-1]) * 1000
        elif interval_text.endswith('m'):
            return int(interval_text[:-1]) * 60 * 1000
        return 5 * 60 * 1000  # Default 5 minutes
        
    def check_for_updates(self):
        """Check for model directory updates"""
        if hasattr(self.parent_interface, 'scan_models'):
            self.parent_interface.scan_models()
            
    def manual_refresh(self):
        """Manual refresh trigger"""
        self.check_for_updates()


class ModelDiscoveryInterface(QWidget):
    """Qt5 Model discovery and loading interface with enhanced features"""
    
    def __init__(self, parent_gui, model_loader=None, model_selected_callback=None):
        super().__init__()
        self.parent_gui = parent_gui
        self.model_loader = model_loader
        self.model_selected_callback = model_selected_callback
        self.discovered_models = {}
        self.scanner_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("🧠 GridFormer Model Discovery")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Discovery and list
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Directory selection
        dir_group = QGroupBox("Model Directory")
        dir_layout = QVBoxLayout()
        
        dir_select_layout = QHBoxLayout()
        self.model_dir_edit = QLineEdit("./models")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_model_dir)
        
        dir_select_layout.addWidget(self.model_dir_edit)
        dir_select_layout.addWidget(self.browse_btn)
        dir_layout.addLayout(dir_select_layout)
        
        # Scan controls
        scan_layout = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 Scan for Models")
        self.scan_btn.clicked.connect(self.scan_models)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        scan_layout.addWidget(self.scan_btn)
        scan_layout.addWidget(self.progress_bar)
        dir_layout.addLayout(scan_layout)
        
        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)
        
        # Model filtering
        filter_group = QGroupBox("Filters")
        self.filter_widget = ModelFilterWidget()
        self.filter_widget.filter_changed.connect(self.apply_filters)
        filter_layout = QVBoxLayout()
        filter_layout.addWidget(self.filter_widget)
        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)
        
        # Model list
        list_group = QGroupBox("Available Models")
        list_layout = QVBoxLayout()
        
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["Name", "Type", "Size", "Path"])
        self.model_tree.itemSelectionChanged.connect(self.on_model_selection_changed)
        list_layout.addWidget(self.model_tree)
        
        # Load button
        self.load_btn = QPushButton("Load Selected Model")
        self.load_btn.clicked.connect(self.load_selected_model)
        list_layout.addWidget(self.load_btn)
        
        list_group.setLayout(list_layout)
        left_layout.addWidget(list_group)
        
        left_panel.setLayout(left_layout)
        main_splitter.addWidget(left_panel)
        
        # Right panel - Enhanced features
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Create tab widget for features
        feature_tabs = QTabWidget()
        
        # Preview tab
        self.preview_widget = ModelPreviewWidget()
        feature_tabs.addTab(self.preview_widget, "Preview")
        
        # Comparison tab
        self.comparison_widget = ModelComparisonWidget()
        self.comparison_widget.add_btn.clicked.connect(self.add_to_comparison)
        self.comparison_widget.clear_btn.clicked.connect(self.clear_comparison)
        feature_tabs.addTab(self.comparison_widget, "Compare")
        
        # History tab
        self.history_widget = ModelHistoryWidget()
        self.history_widget.reload_btn.clicked.connect(self.reload_from_history)
        self.history_widget.clear_history_btn.clicked.connect(self.clear_history)
        feature_tabs.addTab(self.history_widget, "History")
        
        # Watcher tab
        self.watcher_widget = ModelWatcherWidget(self)
        feature_tabs.addTab(self.watcher_widget, "Auto-refresh")
        
        right_layout.addWidget(feature_tabs)
        right_panel.setLayout(right_layout)
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 400])
        layout.addWidget(main_splitter)
        
        self.setLayout(layout)
        
    def browse_model_dir(self):
        """Browse for model directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", "./models"
        )
        if directory:
            self.model_dir_edit.setText(directory)
            
    def scan_models(self):
        """Scan for models in the selected directory"""
        directory = self.model_dir_edit.text()
        if not directory:
            QMessageBox.warning(self, "Warning", "Please select a model directory first")
            return
            
        # Clear existing items
        self.model_tree.clear()
        self.discovered_models.clear()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.scan_btn.setEnabled(False)
        
        # Start background scanning
        self.scanner_thread = ModelScannerThread(directory, self.model_loader)
        self.scanner_thread.model_found.connect(self.on_model_found)
        self.scanner_thread.scan_complete.connect(self.on_scan_complete)
        self.scanner_thread.scan_progress.connect(self.progress_bar.setValue)
        self.scanner_thread.start()
        
    def on_model_found(self, model_name, model_info):
        """Handle when a model is found during scanning"""
        self.discovered_models[model_name] = model_info
        
        path = model_info.get("path", "N/A")
        metadata = model_info.get("metadata", {})
        model_type = metadata.get("type", "Unknown")
        size = metadata.get("size", "Unknown")
        
        # Format size
        if isinstance(size, (int, float)):
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
        else:
            size_str = str(size)
            
        item = QTreeWidgetItem([model_name, model_type, size_str, path])
        self.model_tree.addTopLevelItem(item)
        
    def on_scan_complete(self, count):
        """Handle scan completion"""
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        QMessageBox.information(self, "Scan Complete", f"Found {count} compatible models")
        
    def on_model_selection_changed(self):
        """Handle model selection change"""
        selected_items = self.model_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            model_name = item.text(0)
            if model_name in self.discovered_models:
                metadata = self.discovered_models[model_name].get("metadata", {})
                self.preview_widget.update_preview(model_name, metadata)
                
    def apply_filters(self):
        """Apply filters to the model list"""
        filters = self.filter_widget.get_filters()
        
        for i in range(self.model_tree.topLevelItemCount()):
            item = self.model_tree.topLevelItem(i)
            visible = True
            
            # Type filter
            if filters["type"] != "All" and item.text(1) != filters["type"]:
                visible = False
                
            # Name filter
            if filters["name"] and filters["name"] not in item.text(0).lower():
                visible = False
                
            item.setHidden(not visible)
            
    def add_to_comparison(self):
        """Add selected model to comparison"""
        selected_items = self.model_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            model_name = item.text(0)
            if model_name in self.discovered_models:
                metadata = self.discovered_models[model_name].get("metadata", {})
                self.comparison_widget.add_model_to_comparison(model_name, metadata)
                
    def clear_comparison(self):
        """Clear the comparison list"""
        self.comparison_widget.comparison_tree.clear()
        
    def reload_from_history(self):
        """Reload a model from history"""
        selected_items = self.history_widget.history_tree.selectedItems()
        if selected_items:
            model_name = selected_items[0].text(0)
            # Find and load the model
            for name, info in self.discovered_models.items():
                if name == model_name:
                    self.load_model(info["path"])
                    break
                    
    def clear_history(self):
        """Clear the loading history"""
        self.history_widget.history_tree.clear()
        
    def load_selected_model(self):
        """Load the selected model"""
        selected_items = self.model_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a model to load")
            return
            
        item = selected_items[0]
        model_path = item.text(3)
        model_name = item.text(0)
        
        # Add to history
        self.history_widget.add_to_history(model_name, "Loading...")
        
        try:
            # Load the model
            if hasattr(self.parent_gui, "load_model") and callable(self.parent_gui.load_model):
                self.parent_gui.load_model(model_path)
                self.history_widget.add_to_history(model_name, "Loaded")
            else:
                self.history_widget.add_to_history(model_name, "Failed")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.history_widget.add_to_history(model_name, "Error")
            
    def load_model(self, model_path):
        """Load a specific model by path"""
        if self.model_selected_callback:
            self.model_selected_callback(model_path)
