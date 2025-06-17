#!/usr/bin/env python
"""
Security & Compliance Panel

GUI widget for displaying security scan results, vulnerabilities, and compliance status.
"""

import time
from typing import Any, Dict

from PyQt5 import QtCore, QtGui, QtWidgets


class SecurityPanel(QtWidgets.QWidget):
    """
    Security monitoring panel showing vulnerabilities, compliance status, and security alerts.
    """

    def __init__(self, bus=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.vulnerabilities = []
        self.compliance_data = {}

        self._init_ui()

        if self.bus:
            self.bus.subscribe("security.alert", self.update_security_data)

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Header
        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("🛡️ Security & Compliance Monitor")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")
        header_layout.addWidget(title)

        # Last scan time
        self.last_scan_label = QtWidgets.QLabel("Last scan: Never")
        self.last_scan_label.setStyleSheet("color: #888;")
        header_layout.addWidget(self.last_scan_label)
        header_layout.addStretch()

        # Manual scan button
        scan_button = QtWidgets.QPushButton("🔍 Run Scan")
        scan_button.clicked.connect(self._trigger_scan)
        header_layout.addWidget(scan_button)

        layout.addLayout(header_layout)

        # Status overview cards
        status_layout = QtWidgets.QHBoxLayout()

        # Vulnerability status card
        self.vuln_card = self._create_status_card("Vulnerabilities", "0", "#4CAF50")
        status_layout.addWidget(self.vuln_card)

        # Compliance status card
        self.compliance_card = self._create_status_card("Compliance", "✓", "#4CAF50")
        status_layout.addWidget(self.compliance_card)

        # Permission issues card
        self.permission_card = self._create_status_card("Permissions", "0", "#4CAF50")
        status_layout.addWidget(self.permission_card)

        layout.addLayout(status_layout)

        # Tabbed details
        self.tab_widget = QtWidgets.QTabWidget()

        # Vulnerabilities tab
        self.vuln_table = self._create_vulnerability_table()
        self.tab_widget.addTab(self.vuln_table, "🚨 Vulnerabilities")

        # Compliance tab
        self.compliance_widget = self._create_compliance_widget()
        self.tab_widget.addTab(self.compliance_widget, "📋 Compliance")

        # Permissions tab
        self.permission_table = self._create_permission_table()
        self.tab_widget.addTab(self.permission_table, "🔒 Permissions")

        layout.addWidget(self.tab_widget)

    def _create_status_card(self, title: str, value: str, color: str) -> QtWidgets.QWidget:
        """Create a status card widget."""
        card = QtWidgets.QFrame()
        card.setFrameStyle(QtWidgets.QFrame.Box)
        card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(76, 175, 80, 0.1);
                padding: 10px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(card)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)

        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        value_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(value_label)

        # Store references for updates
        setattr(card, "value_label", value_label)
        setattr(card, "title_label", title_label)

        return card

    def _create_vulnerability_table(self) -> QtWidgets.QTableWidget:
        """Create vulnerability table."""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Severity", "File", "Line", "Issue", "Confidence"])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)

        return table

    def _create_compliance_widget(self) -> QtWidgets.QWidget:
        """Create compliance status widget."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # GDPR status
        self.gdpr_status = QtWidgets.QLabel("GDPR Ready: ✓")
        self.gdpr_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.gdpr_status)

        # Voice consent status
        self.voice_consent_status = QtWidgets.QLabel("Voice Consent: ✓")
        self.voice_consent_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.voice_consent_status)

        # Data retention status
        self.data_retention_status = QtWidgets.QLabel("Data Retention: ✓")
        self.data_retention_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.data_retention_status)

        # Issues list
        layout.addWidget(QtWidgets.QLabel("Compliance Issues:"))
        self.compliance_issues_list = QtWidgets.QListWidget()
        layout.addWidget(self.compliance_issues_list)

        layout.addStretch()
        return widget

    def _create_permission_table(self) -> QtWidgets.QTableWidget:
        """Create permission issues table."""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Type", "File", "Mode", "Severity"])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)

        return table

    def _trigger_scan(self):
        """Trigger a manual security scan."""
        if self.bus:
            self.bus.publish("security.scan_request", {"manual": True})

    def update_security_data(self, payload: Dict[str, Any]):
        """Update the security panel with new data."""
        try:
            # Update last scan time
            if "timestamp" in payload:
                scan_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(payload["timestamp"]))
                self.last_scan_label.setText(f"Last scan: {scan_time}")

            # Update vulnerabilities
            if "vulnerabilities" in payload:
                self.vulnerabilities = payload["vulnerabilities"]
                self._update_vulnerability_table()
                self._update_vulnerability_card()

            # Update compliance
            if "compliance" in payload:
                self.compliance_data = payload["compliance"]
                self._update_compliance_widget()
                self._update_compliance_card()

            # Update permissions
            if "permission_issues" in payload:
                self.permission_issues = payload["permission_issues"]
                self._update_permission_table()
                self._update_permission_card()

        except Exception as e:
            print(f"Error updating security data: {e}")

    def _update_vulnerability_card(self):
        """Update vulnerability status card."""
        count = len(self.vulnerabilities)

        if count == 0:
            color = "#4CAF50"
            status = "✓"
        elif count < 5:
            color = "#FF9800"
            status = str(count)
        else:
            color = "#F44336"
            status = str(count)

        self.vuln_card.value_label.setText(status)
        self.vuln_card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(76, 175, 80, 0.1);
                padding: 10px;
            }}
        """)

    def _update_compliance_card(self):
        """Update compliance status card."""
        if not self.compliance_data:
            return

        issues = self.compliance_data.get("issues", [])

        if len(issues) == 0:
            color = "#4CAF50"
            status = "✓"
        else:
            color = "#F44336"
            status = "⚠"

        self.compliance_card.value_label.setText(status)
        self.compliance_card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(76, 175, 80, 0.1);
                padding: 10px;
            }}
        """)

    def _update_permission_card(self):
        """Update permission status card."""
        count = len(getattr(self, "permission_issues", []))

        if count == 0:
            color = "#4CAF50"
            status = "✓"
        else:
            color = "#F44336"
            status = str(count)

        self.permission_card.value_label.setText(status)
        self.permission_card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                background-color: rgba(76, 175, 80, 0.1);
                padding: 10px;
            }}
        """)

    def _update_vulnerability_table(self):
        """Update the vulnerability table."""
        self.vuln_table.setRowCount(len(self.vulnerabilities))

        for i, vuln in enumerate(self.vulnerabilities):
            # Severity
            severity = vuln.get("issue_severity", "UNKNOWN")
            severity_item = QtWidgets.QTableWidgetItem(severity)
            if severity == "HIGH":
                severity_item.setBackground(QtGui.QColor("#F44336"))
            elif severity == "MEDIUM":
                severity_item.setBackground(QtGui.QColor("#FF9800"))
            else:
                severity_item.setBackground(QtGui.QColor("#4CAF50"))
            self.vuln_table.setItem(i, 0, severity_item)

            # File
            filename = vuln.get("filename", "")
            self.vuln_table.setItem(i, 1, QtWidgets.QTableWidgetItem(filename))

            # Line
            line_number = str(vuln.get("line_number", ""))
            self.vuln_table.setItem(i, 2, QtWidgets.QTableWidgetItem(line_number))

            # Issue
            issue_text = vuln.get("issue_text", "")
            self.vuln_table.setItem(i, 3, QtWidgets.QTableWidgetItem(issue_text))

            # Confidence
            confidence = vuln.get("issue_confidence", "")
            self.vuln_table.setItem(i, 4, QtWidgets.QTableWidgetItem(confidence))

    def _update_compliance_widget(self):
        """Update compliance status widget."""
        if not self.compliance_data:
            return

        # Update status indicators
        gdpr_ready = self.compliance_data.get("gdpr_ready", False)
        self.gdpr_status.setText(f"GDPR Ready: {'✓' if gdpr_ready else '✗'}")
        self.gdpr_status.setStyleSheet(
            f"color: {'#4CAF50' if gdpr_ready else '#F44336'}; font-weight: bold;"
        )

        voice_consent = self.compliance_data.get("voice_consent", False)
        self.voice_consent_status.setText(f"Voice Consent: {'✓' if voice_consent else '✗'}")
        self.voice_consent_status.setStyleSheet(
            f"color: {'#4CAF50' if voice_consent else '#F44336'}; font-weight: bold;"
        )

        data_retention = self.compliance_data.get("data_retention", False)
        self.data_retention_status.setText(f"Data Retention: {'✓' if data_retention else '✗'}")
        self.data_retention_status.setStyleSheet(
            f"color: {'#4CAF50' if data_retention else '#F44336'}; font-weight: bold;"
        )

        # Update issues list
        self.compliance_issues_list.clear()
        issues = self.compliance_data.get("issues", [])
        for issue in issues:
            item_text = (
                f"{issue.get('type', 'Unknown')}: {issue.get('description', 'No description')}"
            )
            item = QtWidgets.QListWidgetItem(item_text)

            severity = issue.get("severity", "low")
            if severity == "high":
                item.setBackground(QtGui.QColor("#F44336"))
            elif severity == "medium":
                item.setBackground(QtGui.QColor("#FF9800"))

            self.compliance_issues_list.addItem(item)

    def _update_permission_table(self):
        """Update permission issues table."""
        permission_issues = getattr(self, "permission_issues", [])
        self.permission_table.setRowCount(len(permission_issues))

        for i, issue in enumerate(permission_issues):
            # Type
            issue_type = issue.get("type", "unknown")
            self.permission_table.setItem(i, 0, QtWidgets.QTableWidgetItem(issue_type))

            # File
            file_path = issue.get("path", "")
            self.permission_table.setItem(i, 1, QtWidgets.QTableWidgetItem(file_path))

            # Mode
            mode = issue.get("mode", "")
            self.permission_table.setItem(i, 2, QtWidgets.QTableWidgetItem(mode))

            # Severity
            severity = issue.get("severity", "low")
            severity_item = QtWidgets.QTableWidgetItem(severity)
            if severity == "high":
                severity_item.setBackground(QtGui.QColor("#F44336"))
            elif severity == "medium":
                severity_item.setBackground(QtGui.QColor("#FF9800"))
            else:
                severity_item.setBackground(QtGui.QColor("#4CAF50"))
            self.permission_table.setItem(i, 3, severity_item)


# Test the widget
if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = SecurityPanel()

    # Test with sample data
    test_data = {
        "timestamp": time.time(),
        "vulnerabilities": [
            {
                "issue_severity": "HIGH",
                "filename": "test.py",
                "line_number": 42,
                "issue_text": "Use of insecure MD5 hash",
                "issue_confidence": "HIGH",
            }
        ],
        "compliance": {
            "gdpr_ready": False,
            "voice_consent": True,
            "data_retention": True,
            "issues": [
                {
                    "type": "missing_gdpr_doc",
                    "description": "Missing PRIVACY.md",
                    "severity": "medium",
                }
            ],
        },
        "permission_issues": [
            {"type": "world_writable", "path": "/tmp/test.txt", "mode": "0o666", "severity": "high"}
        ],
    }

    widget.update_security_data(test_data)
    widget.show()

    sys.exit(app.exec_())
