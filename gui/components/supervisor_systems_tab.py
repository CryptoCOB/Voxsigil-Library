#!/usr/bin/env python3
"""
Supervisor Systems Tab - Real-time Supervisor and Orchestration Monitoring
Provides live monitoring of VoxSigil supervisor systems, orchestrators, and control systems.
"""

import logging
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class SupervisorStatusWidget(QWidget):
    """Widget displaying supervisor system status and health"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_status)
        self.update_timer.start(2000)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Supervisor Health Overview
        health_group = QGroupBox("Supervisor Health Overview")
        health_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        health_layout = QGridLayout(health_group)

        # Health metrics
        self.active_supervisors_label = VoxSigilWidgetFactory.create_label(
            "Active Supervisors: --", "info"
        )
        self.healthy_count_label = VoxSigilWidgetFactory.create_label("Healthy: --", "info")
        self.warning_count_label = VoxSigilWidgetFactory.create_label("Warnings: --", "info")
        self.critical_count_label = VoxSigilWidgetFactory.create_label("Critical: --", "info")

        # System load and performance
        self.system_load_label = VoxSigilWidgetFactory.create_label("System Load: --%", "info")
        self.response_time_label = VoxSigilWidgetFactory.create_label("Avg Response: -- ms", "info")

        self.load_progress = VoxSigilWidgetFactory.create_progress_bar()
        self.response_progress = VoxSigilWidgetFactory.create_progress_bar()

        health_layout.addWidget(self.active_supervisors_label, 0, 0)
        health_layout.addWidget(self.healthy_count_label, 0, 1)
        health_layout.addWidget(self.warning_count_label, 1, 0)
        health_layout.addWidget(self.critical_count_label, 1, 1)
        health_layout.addWidget(self.system_load_label, 2, 0)
        health_layout.addWidget(self.response_time_label, 2, 1)
        health_layout.addWidget(self.load_progress, 3, 0)
        health_layout.addWidget(self.response_progress, 3, 1)

        layout.addWidget(health_group)

        # Orchestration Statistics
        orch_group = QGroupBox("Orchestration Statistics")
        orch_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        orch_layout = QGridLayout(orch_group)

        self.active_workflows_label = VoxSigilWidgetFactory.create_label(
            "Active Workflows: --", "info"
        )
        self.queued_tasks_label = VoxSigilWidgetFactory.create_label("Queued Tasks: --", "info")
        self.completed_today_label = VoxSigilWidgetFactory.create_label(
            "Completed Today: --", "info"
        )
        self.failed_today_label = VoxSigilWidgetFactory.create_label("Failed Today: --", "info")

        orch_layout.addWidget(self.active_workflows_label, 0, 0)
        orch_layout.addWidget(self.queued_tasks_label, 0, 1)
        orch_layout.addWidget(self.completed_today_label, 1, 0)
        orch_layout.addWidget(self.failed_today_label, 1, 1)

        layout.addWidget(orch_group)

    def refresh_status(self):
        """Refresh supervisor status"""
        try:
            import random

            # Simulate supervisor metrics
            active_supervisors = random.randint(3, 8)
            healthy_count = random.randint(2, active_supervisors)
            warning_count = random.randint(0, 2)
            critical_count = max(0, active_supervisors - healthy_count - warning_count)

            system_load = random.randint(15, 85)
            response_time = random.randint(50, 500)

            self.active_supervisors_label.setText(f"Active Supervisors: {active_supervisors}")
            self.healthy_count_label.setText(f"Healthy: {healthy_count}")
            self.warning_count_label.setText(f"Warnings: {warning_count}")
            self.critical_count_label.setText(f"Critical: {critical_count}")

            self.system_load_label.setText(f"System Load: {system_load}%")
            self.response_time_label.setText(f"Avg Response: {response_time} ms")
            self.load_progress.setValue(system_load)
            self.response_progress.setValue(min(100, response_time // 5))

            # Orchestration stats
            active_workflows = random.randint(5, 25)
            queued_tasks = random.randint(0, 15)
            completed_today = random.randint(50, 200)
            failed_today = random.randint(0, 10)

            self.active_workflows_label.setText(f"Active Workflows: {active_workflows}")
            self.queued_tasks_label.setText(f"Queued Tasks: {queued_tasks}")
            self.completed_today_label.setText(f"Completed Today: {completed_today}")
            self.failed_today_label.setText(f"Failed Today: {failed_today}")

        except Exception as e:
            logger.error(f"Error updating supervisor status: {e}")


class SupervisorTree(QWidget):
    """Tree view of supervisor systems and their components"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_tree)
        self.refresh_timer.start(4000)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Supervisor", "Status", "Load", "Uptime", "Last Check"])
        self.tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        layout.addWidget(self.tree)
        self.refresh_tree()

    def refresh_tree(self):
        """Refresh the supervisor tree"""
        self.tree.clear()

        # Supervisor systems
        systems = {
            "Core Supervisors": [
                ("VantaSupervisor", "Healthy", "45%", "2d 14h 23m"),
                ("AgentOrchestrator", "Healthy", "67%", "2d 14h 23m"),
                ("TaskManager", "Warning", "89%", "1d 8h 45m"),
                ("WorkflowEngine", "Healthy", "34%", "2d 14h 23m"),
            ],
            "Monitoring Supervisors": [
                ("HealthMonitor", "Healthy", "23%", "2d 14h 23m"),
                ("MetricsCollector", "Healthy", "56%", "2d 14h 23m"),
                ("AlertManager", "Healthy", "12%", "2d 14h 23m"),
            ],
            "Resource Supervisors": [
                ("ResourceManager", "Healthy", "78%", "2d 14h 23m"),
                ("LoadBalancer", "Healthy", "45%", "2d 14h 23m"),
                ("SchedulerSupervisor", "Warning", "91%", "1d 3h 12m"),
            ],
            "Integration Supervisors": [
                ("EventBusSupervisor", "Healthy", "34%", "2d 14h 23m"),
                ("MessageQueueSupervisor", "Healthy", "67%", "2d 14h 23m"),
                ("DatabaseSupervisor", "Critical", "95%", "0d 2h 15m"),
            ],
        }

        for category, supervisors in systems.items():
            parent = QTreeWidgetItem([category, "", "", "", ""])
            parent.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
            parent.setExpanded(True)

            for name, status, load, uptime in supervisors:
                child = QTreeWidgetItem(
                    [name, status, load, uptime, datetime.now().strftime("%H:%M:%S")]
                )

                # Color code by status
                if status == "Healthy":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["success"]))
                elif status == "Warning":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["warning"]))
                elif status == "Critical":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["error"]))
                else:
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["text_secondary"]))

                parent.addChild(child)

            self.tree.addTopLevelItem(parent)


class SupervisorEventsLog(QWidget):
    """Log of supervisor events and decisions"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.add_sample_events()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = VoxSigilWidgetFactory.create_checkbox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)

        self.filter_level = VoxSigilWidgetFactory.create_button("Filter: ALL", "default")
        clear_btn = VoxSigilWidgetFactory.create_button("Clear Log", "default")
        clear_btn.clicked.connect(self.clear_log)

        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addWidget(self.filter_level)
        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)

        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.log_display.setReadOnly(True)

        layout.addWidget(self.log_display)

    def add_sample_events(self):
        """Add sample supervisor events"""
        events = [
            "[SUPERVISOR] VantaSupervisor started orchestration cycle",
            "[ORCHESTRATOR] New workflow assigned: sentiment-analysis-pipeline",
            "[TASK] Task queued: process_batch_documents (priority: high)",
            "[RESOURCE] Resource allocation: 4 CPU cores, 8GB RAM to training task",
            "[HEALTH] All agents responding within acceptable latency",
            "[SCHEDULER] Load balancing: redistributed 3 tasks to less loaded agents",
            "[ALERT] Warning: TaskManager load above 85% threshold",
            "[RECOVERY] Automatic restart initiated for failed DatabaseSupervisor",
            "[OPTIMIZER] Workflow optimization completed: 15% performance improvement",
            "[MONITOR] Heartbeat check completed: 7/8 supervisors healthy",
        ]

        for event in events:
            self.add_log_entry(event)

    def add_log_entry(self, message: str):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Color code by event type
        if "[ALERT]" in message or "Critical" in message:
            color = VoxSigilStyles.COLORS["error"]
        elif "[WARNING]" in message or "Warning" in message:
            color = VoxSigilStyles.COLORS["warning"]
        elif "[RECOVERY]" in message or "restart" in message.lower():
            color = VoxSigilStyles.COLORS["accent_coral"]
        elif "[HEALTH]" in message or "healthy" in message.lower():
            color = VoxSigilStyles.COLORS["success"]
        elif "[ORCHESTRATOR]" in message:
            color = VoxSigilStyles.COLORS["accent_cyan"]
        elif "[OPTIMIZER]" in message:
            color = VoxSigilStyles.COLORS["accent_mint"]
        else:
            color = VoxSigilStyles.COLORS["text_primary"]

        self.log_display.append(f'<span style="color: {color}">{formatted_message}</span>')

        if self.auto_scroll_checkbox.isChecked():
            self.log_display.moveCursor(self.log_display.textCursor().End)

    def clear_log(self):
        """Clear the log display"""
        self.log_display.clear()


class WorkflowVisualization(QWidget):
    """Visual representation of active workflows and dependencies"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Workflow tree
        self.workflow_tree = QTreeWidget()
        self.workflow_tree.setHeaderLabels(["Workflow", "Stage", "Progress", "Dependencies", "ETA"])
        self.workflow_tree.setStyleSheet(VoxSigilStyles.get_list_widget_stylesheet())

        # Add sample workflows
        workflows = [
            {
                "name": "Document Processing Pipeline",
                "stages": [
                    ("Data Ingestion", "Complete", "100%", "None", "Finished"),
                    ("Text Extraction", "Running", "67%", "Data Ingestion", "0h 15m"),
                    ("NLP Analysis", "Pending", "0%", "Text Extraction", "1h 30m"),
                    ("Result Aggregation", "Pending", "0%", "NLP Analysis", "2h 45m"),
                ],
            },
            {
                "name": "Model Training Workflow",
                "stages": [
                    ("Data Preprocessing", "Complete", "100%", "None", "Finished"),
                    ("Feature Engineering", "Complete", "100%", "Data Preprocessing", "Finished"),
                    ("Model Training", "Running", "34%", "Feature Engineering", "4h 20m"),
                    ("Model Validation", "Pending", "0%", "Model Training", "6h 15m"),
                ],
            },
        ]

        for workflow in workflows:
            parent = QTreeWidgetItem([workflow["name"], "", "", "", ""])
            parent.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
            parent.setExpanded(True)

            for stage, status, progress, deps, eta in workflow["stages"]:
                child = QTreeWidgetItem([stage, status, progress, deps, eta])

                # Color code by status
                if status == "Complete":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["success"]))
                elif status == "Running":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["info"]))
                elif status == "Pending":
                    child.setForeground(1, QColor(VoxSigilStyles.COLORS["text_secondary"]))

                parent.addChild(child)

            self.workflow_tree.addTopLevelItem(parent)

        layout.addWidget(self.workflow_tree)


class SupervisorSystemsTab(QWidget):
    """Main Supervisor Systems monitoring tab with streaming support"""

    # Signals for streaming data
    supervisor_update = pyqtSignal(dict)
    supervisor_event = pyqtSignal(str)
    workflow_update = pyqtSignal(dict)

    def __init__(self, event_bus=None):
        super().__init__()
        self.event_bus = event_bus
        self.setup_ui()
        self.setup_streaming()

    def setup_ui(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)

        # Title
        title = VoxSigilWidgetFactory.create_label("👑 Supervisor Systems Monitor", "title")
        layout.addWidget(title)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(VoxSigilStyles.get_splitter_stylesheet())

        # Left panel - Status and tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Supervisor status
        self.status_widget = SupervisorStatusWidget()
        left_layout.addWidget(self.status_widget)

        # Supervisor tree
        self.tree_widget = SupervisorTree()
        left_layout.addWidget(self.tree_widget)

        splitter.addWidget(left_panel)

        # Right panel - Events and workflows
        right_panel = QTabWidget()
        right_panel.setStyleSheet(VoxSigilStyles.get_tab_widget_stylesheet())

        # Events log tab
        self.events_log = SupervisorEventsLog()
        right_panel.addTab(self.events_log, "📜 Events")

        # Workflow visualization tab
        self.workflow_viz = WorkflowVisualization()
        right_panel.addTab(self.workflow_viz, "🔄 Workflows")

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

        # Status bar
        status_layout = QHBoxLayout()

        self.connection_status = VoxSigilWidgetFactory.create_label(
            "🔌 Connected to Supervisor Monitor", "info"
        )
        self.last_update = VoxSigilWidgetFactory.create_label("Last update: --:--:--", "info")

        status_layout.addWidget(self.connection_status)
        status_layout.addStretch()
        status_layout.addWidget(self.last_update)

        layout.addLayout(status_layout)

        # Apply dark theme
        self.setStyleSheet(VoxSigilStyles.get_base_stylesheet())

    def setup_streaming(self):
        """Setup event bus streaming subscriptions"""
        if self.event_bus:
            # Subscribe to supervisor-related events
            self.event_bus.subscribe("supervisor.status", self.on_supervisor_status)
            self.event_bus.subscribe("supervisor.event", self.on_supervisor_event)
            self.event_bus.subscribe("workflow.update", self.on_workflow_update)

            # Connect internal signals
            self.supervisor_update.connect(self.update_supervisor_display)
            self.supervisor_event.connect(self.events_log.add_log_entry)
            self.workflow_update.connect(self.update_workflow_display)

            logger.info("Supervisor Systems tab connected to streaming events")
            self.connection_status.setText("🔌 Connected to Supervisor Monitor")
        else:
            logger.warning("Supervisor Systems tab: No event bus available")
            self.connection_status.setText("⚠️ No Event Bus Connection")

    def on_supervisor_status(self, data):
        """Handle supervisor status updates"""
        try:
            self.supervisor_update.emit(data)
            self.last_update.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Error processing supervisor status: {e}")

    def on_supervisor_event(self, event):
        """Handle supervisor event notifications"""
        try:
            if isinstance(event, dict):
                message = event.get("message", str(event))
            else:
                message = str(event)
            self.supervisor_event.emit(message)
        except Exception as e:
            logger.error(f"Error processing supervisor event: {e}")

    def on_workflow_update(self, data):
        """Handle workflow update events"""
        try:
            self.workflow_update.emit(data)
        except Exception as e:
            logger.error(f"Error processing workflow update: {e}")

    def update_supervisor_display(self, data):
        """Update supervisor display with new data"""
        try:
            # Update would be handled by the status widget
            pass
        except Exception as e:
            logger.error(f"Error updating supervisor display: {e}")

    def update_workflow_display(self, data):
        """Update workflow display with new data"""
        try:
            # Update would be handled by the workflow widget
            pass
        except Exception as e:
            logger.error(f"Error updating workflow display: {e}")


def create_supervisor_systems_tab(event_bus=None) -> SupervisorSystemsTab:
    """Factory function to create Supervisor Systems tab"""
    return SupervisorSystemsTab(event_bus=event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    VoxSigilStyles.apply_dark_theme(app)

    tab = SupervisorSystemsTab()
    tab.show()

    sys.exit(app.exec_())
