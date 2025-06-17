#!/usr/bin/env python
"""
Dataset Manager Panel

GUI widget for displaying dataset status, versions, and management controls.
"""

import time
from typing import Any, Dict

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class DatasetPanel(QtWidgets.QWidget):
    """
    Dataset management panel showing dataset status, versions, and controls.
    """

    def __init__(self, bus=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.datasets = {}

        self._init_ui()

        if self.bus:
            self.bus.subscribe("dataset.status", self.update_dataset_data)
            self.bus.subscribe("dataset.reindex.started", self.on_reindex_started)
            self.bus.subscribe("dataset.reindex.completed", self.on_reindex_completed)

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Header
        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("📊 Dataset Manager")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        header_layout.addWidget(title)

        # Last update time
        self.last_update_label = QtWidgets.QLabel("Last update: Never")
        self.last_update_label.setStyleSheet("color: #888;")
        header_layout.addWidget(self.last_update_label)
        header_layout.addStretch()

        # Refresh button
        refresh_button = QtWidgets.QPushButton("🔄 Refresh")
        refresh_button.clicked.connect(self._trigger_refresh)
        header_layout.addWidget(refresh_button)

        layout.addLayout(header_layout)

        # Summary cards
        summary_layout = QtWidgets.QHBoxLayout()

        # Total datasets card
        self.datasets_card = self._create_summary_card("Datasets", "0", "#2196F3")
        summary_layout.addWidget(self.datasets_card)

        # Total size card
        self.size_card = self._create_summary_card("Total Size", "0 MB", "#4CAF50")
        summary_layout.addWidget(self.size_card)

        # Formats card
        self.formats_card = self._create_summary_card("Formats", "0", "#FF9800")
        summary_layout.addWidget(self.formats_card)

        layout.addLayout(summary_layout)

        # Dataset tree view
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Left side - tree view
        tree_widget = QtWidgets.QWidget()
        tree_layout = QtWidgets.QVBoxLayout(tree_widget)

        tree_layout.addWidget(QtWidgets.QLabel("Dataset Tree"))
        self.dataset_tree = QtWidgets.QTreeWidget()
        self.dataset_tree.setHeaderLabels(["Name", "Type", "Size", "Modified"])
        self.dataset_tree.itemSelectionChanged.connect(self._on_dataset_selected)
        tree_layout.addWidget(self.dataset_tree)

        splitter.addWidget(tree_widget)

        # Right side - details and controls
        details_widget = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details_widget)

        details_layout.addWidget(QtWidgets.QLabel("Dataset Details"))

        # Details scroll area
        scroll_area = QtWidgets.QScrollArea()
        self.details_widget = QtWidgets.QWidget()
        self.details_layout = QtWidgets.QVBoxLayout(self.details_widget)

        # Dataset info labels
        self.info_labels = {
            "path": QtWidgets.QLabel("Path: -"),
            "type": QtWidgets.QLabel("Type: -"),
            "size": QtWidgets.QLabel("Size: -"),
            "license": QtWidgets.QLabel("License: -"),
            "version": QtWidgets.QLabel("Version: -"),
            "description": QtWidgets.QLabel("Description: -"),
            "rows": QtWidgets.QLabel("Rows/Files: -"),
            "modified": QtWidgets.QLabel("Last Modified: -"),
        }

        for label in self.info_labels.values():
            label.setWordWrap(True)
            self.details_layout.addWidget(label)

        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.reindex_button = QtWidgets.QPushButton("🔄 Re-index")
        self.reindex_button.clicked.connect(self._reindex_selected)
        self.reindex_button.setEnabled(False)
        button_layout.addWidget(self.reindex_button)

        self.explore_button = QtWidgets.QPushButton("📂 Explore")
        self.explore_button.clicked.connect(self._explore_selected)
        self.explore_button.setEnabled(False)
        button_layout.addWidget(self.explore_button)

        self.details_layout.addLayout(button_layout)
        self.details_layout.addStretch()

        scroll_area.setWidget(self.details_widget)
        details_layout.addWidget(scroll_area)

        splitter.addWidget(details_widget)
        splitter.setSizes([300, 200])

        layout.addWidget(splitter)

        # Status bar
        self.status_bar = QtWidgets.QLabel("Ready")
        self.status_bar.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.status_bar)

    def _create_summary_card(self, title: str, value: str, color: str) -> QtWidgets.QWidget:
        """Create a summary card widget."""
        card = QtWidgets.QFrame()
        card.setFrameStyle(QtWidgets.QFrame.Box)
        card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(33, 150, 243, 0.1);
                padding: 10px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(card)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)

        # Store reference for updates
        setattr(card, "value_label", value_label)

        return card

    def _trigger_refresh(self):
        """Trigger a manual dataset refresh."""
        if self.bus:
            self.bus.publish("dataset.refresh_request", {"manual": True})
            self.status_bar.setText("Refreshing datasets...")

    def _on_dataset_selected(self):
        """Handle dataset selection in tree."""
        selected_items = self.dataset_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            dataset_path = item.data(0, Qt.UserRole)
            if dataset_path and dataset_path in self.datasets:
                self._show_dataset_details(self.datasets[dataset_path])
                self.reindex_button.setEnabled(True)
                self.explore_button.setEnabled(True)
            else:
                self._clear_dataset_details()
                self.reindex_button.setEnabled(False)
                self.explore_button.setEnabled(False)
        else:
            self._clear_dataset_details()
            self.reindex_button.setEnabled(False)
            self.explore_button.setEnabled(False)

    def _show_dataset_details(self, dataset_info: Dict[str, Any]):
        """Show details for selected dataset."""
        self.info_labels["path"].setText(f"Path: {dataset_info.get('path', '-')}")
        self.info_labels["type"].setText(f"Type: {dataset_info.get('type', '-')}")

        size_mb = dataset_info.get("size_mb", 0)
        self.info_labels["size"].setText(f"Size: {size_mb} MB")

        self.info_labels["license"].setText(f"License: {dataset_info.get('license', 'unknown')}")
        self.info_labels["version"].setText(f"Version: {dataset_info.get('version', 'unknown')}")

        description = dataset_info.get("description", "No description available")
        if len(description) > 100:
            description = description[:100] + "..."
        self.info_labels["description"].setText(f"Description: {description}")

        if "rows" in dataset_info:
            self.info_labels["rows"].setText(f"Rows: {dataset_info['rows']}")
        elif "file_count" in dataset_info:
            self.info_labels["rows"].setText(f"Files: {dataset_info['file_count']}")
        else:
            self.info_labels["rows"].setText("Items: -")

        modified_time = dataset_info.get("last_modified", 0)
        if modified_time:
            modified_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_time))
            self.info_labels["modified"].setText(f"Last Modified: {modified_str}")
        else:
            self.info_labels["modified"].setText("Last Modified: -")

    def _clear_dataset_details(self):
        """Clear dataset details display."""
        for label in self.info_labels.values():
            label.setText(label.text().split(":")[0] + ": -")

    def _reindex_selected(self):
        """Re-index the selected dataset."""
        selected_items = self.dataset_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            dataset_path = item.data(0, Qt.UserRole)
            if dataset_path and self.bus:
                self.bus.publish("dataset.reindex_request", {"dataset_path": dataset_path})
                self.status_bar.setText(f"Re-indexing {dataset_path}...")

    def _explore_selected(self):
        """Open the selected dataset in file explorer."""
        selected_items = self.dataset_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            dataset_path = item.data(0, Qt.UserRole)
            if dataset_path:
                import os
                import subprocess

                try:
                    if os.name == "nt":  # Windows
                        subprocess.run(["explorer", dataset_path])
                    elif os.name == "posix":  # macOS and Linux
                        subprocess.run(
                            [
                                "open" if "darwin" in os.uname().sysname.lower() else "xdg-open",
                                dataset_path,
                            ]
                        )
                except Exception as e:
                    self.status_bar.setText(f"Error opening explorer: {e}")

    def update_dataset_data(self, payload: Dict[str, Any]):
        """Update the dataset panel with new data."""
        try:
            # Update last update time
            if "timestamp" in payload:
                update_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(payload["timestamp"])
                )
                self.last_update_label.setText(f"Last update: {update_time}")

            # Update datasets
            if "datasets" in payload:
                self.datasets = payload["datasets"]
                self._update_summary_cards(payload)
                self._update_dataset_tree()

            self.status_bar.setText("Ready")

        except Exception as e:
            print(f"Error updating dataset data: {e}")
            self.status_bar.setText(f"Error: {e}")

    def _update_summary_cards(self, payload: Dict[str, Any]):
        """Update summary cards."""
        # Total datasets
        total_datasets = payload.get("total_datasets", 0)
        self.datasets_card.value_label.setText(str(total_datasets))

        # Total size
        total_size = payload.get("total_size_mb", 0)
        if total_size > 1024:
            size_str = f"{total_size / 1024:.1f} GB"
        else:
            size_str = f"{total_size:.1f} MB"
        self.size_card.value_label.setText(size_str)

        # Formats count
        formats = set()
        for dataset_info in self.datasets.values():
            if dataset_info.get("type") == "file":
                formats.add(dataset_info.get("format", "unknown"))
            else:
                formats.add("directory")
        self.formats_card.value_label.setText(str(len(formats)))

    def _update_dataset_tree(self):
        """Update the dataset tree view."""
        self.dataset_tree.clear()

        # Group datasets by directory
        dataset_groups = {}

        for path, dataset_info in self.datasets.items():
            path_parts = path.split("/")
            if len(path_parts) > 1:
                group = path_parts[0]
                if group not in dataset_groups:
                    dataset_groups[group] = []
                dataset_groups[group].append((path, dataset_info))
            else:
                if "root" not in dataset_groups:
                    dataset_groups["root"] = []
                dataset_groups["root"].append((path, dataset_info))

        # Add items to tree
        for group_name, group_datasets in dataset_groups.items():
            group_item = QtWidgets.QTreeWidgetItem(self.dataset_tree)
            group_item.setText(0, group_name)
            group_item.setText(1, "group")
            group_item.setExpanded(True)

            for path, dataset_info in group_datasets:
                dataset_item = QtWidgets.QTreeWidgetItem(group_item)

                # Name (filename)
                name = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
                dataset_item.setText(0, name)
                dataset_item.setData(0, Qt.UserRole, path)

                # Type
                dataset_item.setText(1, dataset_info.get("type", "unknown"))

                # Size
                size_mb = dataset_info.get("size_mb", 0)
                if size_mb > 1024:
                    size_str = f"{size_mb / 1024:.1f} GB"
                else:
                    size_str = f"{size_mb:.1f} MB"
                dataset_item.setText(2, size_str)

                # Modified
                modified_time = dataset_info.get("last_modified", 0)
                if modified_time:
                    modified_str = time.strftime("%m/%d/%y", time.localtime(modified_time))
                    dataset_item.setText(3, modified_str)
                else:
                    dataset_item.setText(3, "-")

                # Color code by type
                if dataset_info.get("type") == "directory":
                    dataset_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
                else:
                    dataset_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))

        # Resize columns
        for i in range(4):
            self.dataset_tree.resizeColumnToContents(i)

    def on_reindex_started(self, payload: Dict[str, Any]):
        """Handle reindex started event."""
        dataset_path = payload.get("dataset_path", "")
        self.status_bar.setText(f"Re-indexing {dataset_path}...")

    def on_reindex_completed(self, payload: Dict[str, Any]):
        """Handle reindex completed event."""
        dataset_path = payload.get("dataset_path", "")
        success = payload.get("success", False)

        if success:
            self.status_bar.setText(f"Re-indexing completed: {dataset_path}")
        else:
            self.status_bar.setText(f"Re-indexing failed: {dataset_path}")


# Test the widget
if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = DatasetPanel()

    # Test with sample data
    test_data = {
        "timestamp": time.time(),
        "total_datasets": 3,
        "total_size_mb": 150.5,
        "datasets": {
            "data/train.json": {
                "path": "data/train.json",
                "type": "file",
                "format": ".json",
                "size_mb": 45.2,
                "last_modified": time.time() - 3600,
                "license": "MIT",
                "version": "1.0",
                "rows": 1000,
                "status": "ok",
            },
            "datasets/arc_tasks": {
                "path": "datasets/arc_tasks",
                "type": "directory",
                "size_mb": 105.3,
                "file_count": 400,
                "last_modified": time.time() - 7200,
                "license": "Apache-2.0",
                "version": "2.1",
                "description": "ARC challenge dataset with training and evaluation tasks",
                "status": "ok",
            },
        },
    }

    widget.update_dataset_data(test_data)
    widget.show()

    sys.exit(app.exec_())
