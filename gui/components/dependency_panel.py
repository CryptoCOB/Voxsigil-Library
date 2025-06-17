#!/usr/bin/env python
"""
Dependency Health Panel

GUI widget for displaying package status and dependency health.
"""

import time
from typing import Any, Dict

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class DependencyPanel(QtWidgets.QWidget):
    """
    Dependency health panel showing package status and system information.
    """

    def __init__(self, bus=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.packages = {}
        self.outdated_packages = {}
        self.system_info = {}

        self._init_ui()

        if self.bus:
            self.bus.subscribe("dependency.health", self.update_dependency_data)

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Header
        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("📦 Dependency Health Monitor")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF5722;")
        header_layout.addWidget(title)

        # Last check time
        self.last_check_label = QtWidgets.QLabel("Last check: Never")
        self.last_check_label.setStyleSheet("color: #888;")
        header_layout.addWidget(self.last_check_label)
        header_layout.addStretch()

        # Update button
        update_button = QtWidgets.QPushButton("🔄 Check Now")
        update_button.clicked.connect(self._trigger_check)
        header_layout.addWidget(update_button)

        layout.addLayout(header_layout)

        # Status cards
        status_layout = QtWidgets.QHBoxLayout()

        # Total packages card
        self.packages_card = self._create_status_card("Total Packages", "0", "#4CAF50")
        status_layout.addWidget(self.packages_card)

        # Outdated packages card
        self.outdated_card = self._create_status_card("Outdated", "0", "#FF9800")
        status_layout.addWidget(self.outdated_card)

        # System health card
        self.system_card = self._create_status_card("System", "OK", "#2196F3")
        status_layout.addWidget(self.system_card)

        layout.addLayout(status_layout)

        # Tabbed view
        self.tab_widget = QtWidgets.QTabWidget()

        # Outdated packages tab
        self.outdated_table = self._create_outdated_table()
        self.tab_widget.addTab(self.outdated_table, "⚠️ Outdated Packages")

        # All packages tab
        self.packages_table = self._create_packages_table()
        self.tab_widget.addTab(self.packages_table, "📋 All Packages")

        # System info tab
        self.system_widget = self._create_system_widget()
        self.tab_widget.addTab(self.system_widget, "🖥️ System Info")

        layout.addWidget(self.tab_widget)

    def _create_status_card(self, title: str, value: str, color: str) -> QtWidgets.QWidget:
        """Create a status card widget."""
        card = QtWidgets.QFrame()
        card.setFrameStyle(QtWidgets.QFrame.Box)
        card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(255, 87, 34, 0.1);
                padding: 10px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(card)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)

        # Store reference for updates
        setattr(card, "value_label", value_label)

        return card

    def _create_outdated_table(self) -> QtWidgets.QTableWidget:
        """Create outdated packages table."""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Package", "Current Version", "Latest Version", "Actions"])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)

        return table

    def _create_packages_table(self) -> QtWidgets.QTableWidget:
        """Create all packages table."""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Package", "Version"])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        # Enable sorting
        table.setSortingEnabled(True)

        return table

    def _create_system_widget(self) -> QtWidgets.QWidget:
        """Create system info widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # System info labels
        self.system_labels = {
            "python": QtWidgets.QLabel("Python: Unknown"),
            "pip": QtWidgets.QLabel("Pip: Unknown"),
            "cuda": QtWidgets.QLabel("CUDA: Unknown"),
            "gpu": QtWidgets.QLabel("GPU Driver: Unknown"),
        }

        for label in self.system_labels.values():
            label.setStyleSheet("font-family: monospace; padding: 5px;")
            layout.addWidget(label)

        layout.addStretch()

        # Actions
        actions_layout = QtWidgets.QHBoxLayout()

        upgrade_all_button = QtWidgets.QPushButton("⬆️ Upgrade All")
        upgrade_all_button.clicked.connect(self._upgrade_all_packages)
        actions_layout.addWidget(upgrade_all_button)

        install_package_button = QtWidgets.QPushButton("➕ Install Package")
        install_package_button.clicked.connect(self._install_package)
        actions_layout.addWidget(install_package_button)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        return widget

    def _trigger_check(self):
        """Trigger a manual dependency check."""
        if self.bus:
            self.bus.publish("dependency.check_request", {"manual": True})

    def _upgrade_all_packages(self):
        """Upgrade all outdated packages."""
        if self.outdated_packages:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Upgrade All",
                f"This will upgrade {len(self.outdated_packages)} packages. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )

            if reply == QtWidgets.QMessageBox.Yes:
                if self.bus:
                    self.bus.publish(
                        "dependency.upgrade_all", {"packages": list(self.outdated_packages.keys())}
                    )

    def _install_package(self):
        """Install a new package."""
        package_name, ok = QtWidgets.QInputDialog.getText(
            self, "Install Package", "Enter package name:"
        )

        if ok and package_name:
            if self.bus:
                self.bus.publish("dependency.install", {"package": package_name})

    def _upgrade_package(self, package_name: str):
        """Upgrade a specific package."""
        if self.bus:
            self.bus.publish("dependency.upgrade", {"package": package_name})

    def update_dependency_data(self, payload: Dict[str, Any]):
        """Update the dependency panel with new data."""
        try:
            # Update last check time
            if "timestamp" in payload:
                check_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(payload["timestamp"])
                )
                self.last_check_label.setText(f"Last check: {check_time}")

            # Update data
            if "packages" in payload:
                self.packages = payload["packages"]
                self._update_packages_table()

            if "outdated" in payload:
                self.outdated_packages = payload["outdated"]
                self._update_outdated_table()

            if "system" in payload:
                self.system_info = payload["system"]
                self._update_system_info()

            # Update status cards
            self._update_status_cards(payload)

        except Exception as e:
            print(f"Error updating dependency data: {e}")

    def _update_status_cards(self, payload: Dict[str, Any]):
        """Update status cards."""
        # Total packages
        total_packages = payload.get("total_packages", 0)
        self.packages_card.value_label.setText(str(total_packages))

        # Outdated packages
        outdated_count = payload.get("outdated_count", 0)
        self.outdated_card.value_label.setText(str(outdated_count))

        # Update outdated card color
        if outdated_count == 0:
            color = "#4CAF50"
        elif outdated_count < 5:
            color = "#FF9800"
        else:
            color = "#F44336"

        self.outdated_card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(255, 87, 34, 0.1);
                padding: 10px;
            }}
        """)

        # System health
        cuda_available = self.system_info.get("cuda_available", False)
        system_status = "OK" if cuda_available else "No CUDA"
        self.system_card.value_label.setText(system_status)

    def _update_packages_table(self):
        """Update all packages table."""
        self.packages_table.setRowCount(len(self.packages))

        for i, (package, version) in enumerate(sorted(self.packages.items())):
            # Package name
            self.packages_table.setItem(i, 0, QtWidgets.QTableWidgetItem(package))

            # Version
            self.packages_table.setItem(i, 1, QtWidgets.QTableWidgetItem(version))

    def _update_outdated_table(self):
        """Update outdated packages table."""
        self.outdated_table.setRowCount(len(self.outdated_packages))

        for i, (package, info) in enumerate(self.outdated_packages.items()):
            # Package name
            self.outdated_table.setItem(i, 0, QtWidgets.QTableWidgetItem(package))

            # Current version
            current_version = info.get("current", "unknown")
            self.outdated_table.setItem(i, 1, QtWidgets.QTableWidgetItem(current_version))

            # Latest version
            latest_version = info.get("latest", "unknown")
            self.outdated_table.setItem(i, 2, QtWidgets.QTableWidgetItem(latest_version))

            # Upgrade button
            upgrade_button = QtWidgets.QPushButton("⬆️ Upgrade")
            upgrade_button.clicked.connect(lambda checked, pkg=package: self._upgrade_package(pkg))
            self.outdated_table.setCellWidget(i, 3, upgrade_button)

    def _update_system_info(self):
        """Update system information display."""
        if not self.system_info:
            return

        # Python version
        python_version = self.system_info.get("python_version", "Unknown")
        self.system_labels["python"].setText(f"Python: {python_version}")

        # Pip version
        pip_version = self.system_info.get("pip_version", "Unknown")
        self.system_labels["pip"].setText(f"Pip: {pip_version}")

        # CUDA
        cuda_available = self.system_info.get("cuda_available", False)
        cuda_text = "Available" if cuda_available else "Not Available"
        cuda_color = "#4CAF50" if cuda_available else "#F44336"
        self.system_labels["cuda"].setText(f"CUDA: {cuda_text}")
        self.system_labels["cuda"].setStyleSheet(
            f"font-family: monospace; padding: 5px; color: {cuda_color};"
        )

        # GPU driver
        gpu_driver = self.system_info.get("gpu_driver", "Unknown")
        self.system_labels["gpu"].setText(f"GPU Driver: {gpu_driver}")


# Test the widget
if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = DependencyPanel()

    # Test with sample data
    test_data = {
        "timestamp": time.time(),
        "total_packages": 50,
        "outdated_count": 3,
        "packages": {"numpy": "1.21.0", "pandas": "1.3.0", "torch": "1.9.0", "PyQt5": "5.15.4"},
        "outdated": {
            "numpy": {"current": "1.21.0", "latest": "1.22.0"},
            "pandas": {"current": "1.3.0", "latest": "1.4.0"},
            "requests": {"current": "2.25.1", "latest": "2.27.1"},
        },
        "system": {
            "python_version": "Python 3.9.7",
            "pip_version": "pip 21.2.4",
            "cuda_available": True,
            "gpu_driver": "NVIDIA 470.86",
        },
    }

    widget.update_dependency_data(test_data)
    widget.show()

    sys.exit(app.exec_())
