#!/usr/bin/env python3
"""
Individual Agents Tab - extended version
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict

from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
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

logger = logging.getLogger(__name__)


class AgentInteractionWidget(QWidget):
    """Widget for interacting with a specific agent"""

    agent_interaction_requested = pyqtSignal(str, str)  # agent_name, command

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name
        self.auto_scroll = True
        self.setup_ui()

    # ------------- UI ----------------------------------------------------
    def setup_ui(self):
        layout = QVBoxLayout()

        info_label = QLabel(f"Agent: {self.agent_name}")
        info_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(info_label)

        self.alert_label = QLabel()  # CPU / memory alerts
        layout.addWidget(self.alert_label)

        self.status_label = QLabel("Status: Unknown")
        layout.addWidget(self.status_label)

        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout()

        metrics_layout.addWidget(QLabel("CPU Usage:"), 0, 0)
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        metrics_layout.addWidget(self.cpu_progress, 0, 1)

        metrics_layout.addWidget(QLabel("Memory:"), 1, 0)
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        metrics_layout.addWidget(self.memory_progress, 1, 1)

        metrics_layout.addWidget(QLabel("Avg Response:"), 2, 0)
        self.response_label = QLabel("0ms")
        metrics_layout.addWidget(self.response_label, 2, 1)

        metrics_layout.addWidget(QLabel("Active Tasks:"), 3, 0)
        self.tasks_label = QLabel("0")
        metrics_layout.addWidget(self.tasks_label, 3, 1)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        commands_group = QGroupBox("Quick Commands")
        cmds_layout = QHBoxLayout()
        for cmd in ["ping", "status", "reset", "pause", "resume"]:
            btn = QPushButton(cmd.capitalize())
            btn.clicked.connect(lambda _, c=cmd: self.send_command(c))
            cmds_layout.addWidget(btn)
        commands_group.setLayout(cmds_layout)
        layout.addWidget(commands_group)

        activity_group = QGroupBox("Recent Activity")
        act_layout = QVBoxLayout()
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.stateChanged.connect(lambda s: setattr(self, "auto_scroll", bool(s)))
        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(120)
        self.activity_log.setReadOnly(True)
        act_layout.addWidget(self.auto_scroll_cb)
        act_layout.addWidget(self.activity_log)
        activity_group.setLayout(act_layout)
        layout.addWidget(activity_group)

        self.setLayout(layout)

    # ------------- Helpers ----------------------------------------------
    def send_command(self, command: str):
        self.agent_interaction_requested.emit(self.agent_name, command)
        self._append_activity(f"Sent: {command}")

    def _append_activity(self, text: str):
        self.activity_log.append(text)
        if self.auto_scroll:
            self.activity_log.verticalScrollBar().setValue(
                self.activity_log.verticalScrollBar().maximum()
            )

    def update_status(self, status_data: Dict[str, Any]):
        try:
            status = status_data.get("status", "Unknown")
            cpu = status_data.get("cpu_usage", 0)
            memory = status_data.get("memory_usage", 0)
            resp = status_data.get("avg_response_time", 0)
            tasks = status_data.get("active_tasks", 0)

            self.status_label.setText(f"Status: {status}")
            self.cpu_progress.setValue(int(cpu))
            self.memory_progress.setValue(int(memory))
            self.response_label.setText(f"{resp:.1f}ms")
            self.tasks_label.setText(str(tasks))

            if cpu > 90 or memory > 95:
                self.alert_label.setText("⚠ High load")
                self.alert_label.setStyleSheet("color:red;")
            else:
                self.alert_label.clear()

            if cpu > 80 or memory > 90:
                self.status_label.setStyleSheet("color:red;")
            elif cpu > 60 or memory > 70:
                self.status_label.setStyleSheet("color:orange;")
            else:
                self.status_label.setStyleSheet("color:green;")
        except Exception as e:
            logger.error(f"Error updating status for {self.agent_name}: {e}")


class IndividualAgentsTab(QWidget):
    """Comprehensive individual agent monitoring and interaction tab"""

    agent_command_sent = pyqtSignal(str, str)  # agent_name, command

    def __init__(self, registry=None, event_bus=None):
        super().__init__()
        self.registry = registry
        self.event_bus = event_bus
        self.dark_theme = False
        self.agent_widgets: Dict[str, AgentInteractionWidget] = {}
        self.agent_status_data: Dict[str, Dict[str, Any]] = {}
        self.setup_ui()
        self.setup_streaming()

    # ------------- UI ----------------------------------------------------
    def setup_ui(self):
        v = QVBoxLayout()

        header = QLabel("Individual Agent Monitoring & Interaction")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        v.addWidget(header)

        # --- top controls -------------------------------------------------
        ctrl_bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search agent…")
        self.search_edit.textChanged.connect(self.filter_agents)
        ctrl_bar.addWidget(self.search_edit)

        self.export_btn = QPushButton("⬇ Export CSV")
        self.export_btn.clicked.connect(self.export_metrics)
        ctrl_bar.addWidget(self.export_btn)

        self.theme_btn = QPushButton("🌙 Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        ctrl_bar.addWidget(self.theme_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_timer)
        ctrl_bar.addWidget(self.pause_btn)

        self.jump_err_btn = QPushButton("⚡ First unhealthy")
        self.jump_err_btn.clicked.connect(self.select_unhealthy)
        ctrl_bar.addWidget(self.jump_err_btn)

        v.addLayout(ctrl_bar)

        # --- body ---------------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        v.addWidget(splitter)

        self.setLayout(v)

    def create_left_panel(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout()

        self.agent_tree = QTreeWidget()
        self.agent_tree.setHeaderLabels(["Agent", "Status", "Load"])
        self.agent_tree.itemClicked.connect(self.on_agent_selected)
        self.agent_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.agent_tree.customContextMenuRequested.connect(self.show_context_menu)
        l.addWidget(self.agent_tree)

        stats = QGroupBox("Quick Stats")
        stl = QVBoxLayout()
        self.total_agents_label = QLabel()
        self.active_agents_label = QLabel()
        self.idle_agents_label = QLabel()
        self.error_agents_label = QLabel()
        for lab in (
            self.total_agents_label,
            self.active_agents_label,
            self.idle_agents_label,
            self.error_agents_label,
        ):
            stl.addWidget(lab)
        stats.setLayout(stl)
        l.addWidget(stats)

        w.setLayout(l)
        return w

    def create_right_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout()
        self.agent_tabs = QTabWidget()
        v.addWidget(self.agent_tabs)
        w.setLayout(v)
        return w

    # ------------- Streaming --------------------------------------------
    def setup_streaming(self):
        if self.event_bus:
            self.event_bus.subscribe("agent_status_update", self.on_agent_status_update)
            self.event_bus.subscribe("agent_performance_update", self.on_agent_performance_update)
            self.event_bus.subscribe("agent_activity", self.on_agent_activity)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(2000)
        self.refresh_agents()

    # ------------- Agent management -------------------------------------
    def refresh_agents(self):
        self.agent_tree.clear()
        self.agent_tabs.clear()
        self.agent_widgets.clear()
        try:
            if self.registry:
                agents = self.registry.get_all_agents()
            else:
                agents = [
                    (n, None)
                    for n in [
                        "andy",
                        "astra",
                        "carla",
                        "dave",
                        "echo",
                        "gizmo",
                        "oracle",
                        "voxagent",
                    ]
                ]
            for name, obj in agents:
                self.add_agent(name, obj)
        except Exception as e:
            logger.error(f"Refreshing agents failed: {e}")
        self.update_quick_stats()

    def add_agent(self, name: str, agent=None):
        item = QTreeWidgetItem([name, "Unknown", "0%"])
        self.agent_tree.addTopLevelItem(item)
        widget = AgentInteractionWidget(name)
        widget.agent_interaction_requested.connect(self.send_agent_command)
        self.agent_widgets[name] = widget
        self.agent_tabs.addTab(widget, name)
        self.agent_status_data[name] = dict(
            status="Unknown", cpu_usage=0, memory_usage=0, avg_response_time=0, active_tasks=0
        )

    # ------------- UI helpers -------------------------------------------
    def on_agent_selected(self, item: QTreeWidgetItem, col: int):
        name = item.text(0)
        w = self.agent_widgets.get(name)
        idx = self.agent_tabs.indexOf(w)
        self.agent_tabs.setCurrentIndex(idx if idx >= 0 else 0)

    def filter_agents(self, text: str):
        for i in range(self.agent_tree.topLevelItemCount()):
            item = self.agent_tree.topLevelItem(i)
            item.setHidden(text.lower() not in item.text(0).lower())

    # Context menu with quick commands
    def show_context_menu(self, pos: QPoint):
        item = self.agent_tree.itemAt(pos)
        if not item:
            return
        name = item.text(0)
        menu = QMenu(self)
        for cmd in ["ping", "reset", "pause", "resume"]:
            act = QAction(cmd, self)
            act.triggered.connect(lambda _, c=cmd: self.send_agent_command(name, c))
            menu.addAction(act)
        menu.exec_(self.agent_tree.mapToGlobal(pos))

    def export_metrics(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(Path.home() / "agents.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["agent", "status", "cpu", "mem", "resp", "tasks"])
                for k, d in self.agent_status_data.items():
                    wr.writerow(
                        [
                            k,
                            d["status"],
                            d["cpu_usage"],
                            d["memory_usage"],
                            d["avg_response_time"],
                            d["active_tasks"],
                        ]
                    )
            QMessageBox.information(self, "Export", "CSV exported successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def toggle_theme(self):
        self.dark_theme = not self.dark_theme
        pal = QPalette()
        if self.dark_theme:
            pal.setColor(QPalette.Window, QColor(53, 53, 53))
            pal.setColor(QPalette.WindowText, Qt.white)
            pal.setColor(QPalette.Base, QColor(35, 35, 35))
            pal.setColor(QPalette.Text, Qt.white)
        self.setPalette(pal)
        self.theme_btn.setText("☀ Theme" if self.dark_theme else "🌙 Theme")

    def toggle_timer(self):
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.pause_btn.setText("▶ Resume")
        else:
            self.update_timer.start()
            self.pause_btn.setText("⏸ Pause")

    def select_unhealthy(self):
        for i in range(self.agent_tree.topLevelItemCount()):
            item = self.agent_tree.topLevelItem(i)
            name = item.text(0)
            data = self.agent_status_data.get(name, {})
            if data.get("cpu_usage", 0) > 90 or data.get("memory_usage", 0) > 95:
                self.on_agent_selected(item, 0)
                return

    # ------------- Event handlers ---------------------------------------
    def send_agent_command(self, agent_name: str, cmd: str):
        self.agent_command_sent.emit(agent_name, cmd)
        logger.info(f"Cmd {cmd} -> {agent_name}")

    def _update_agent(self, name: str, data: Dict[str, Any], update_tree=True):
        self.agent_status_data[name].update(data)
        self.agent_widgets[name].update_status(self.agent_status_data[name])
        if update_tree:
            self.update_tree_item(name)

    def on_agent_status_update(self, event):  # from event bus
        d = event.get("data", {})
        name = d.get("agent_name")
        if name in self.agent_widgets:
            self._update_agent(name, d)

    def on_agent_performance_update(self, event):
        self.on_agent_status_update(event)

    def on_agent_activity(self, event):
        d = event.get("data", {})
        name, act = d.get("agent_name"), d.get("activity", "")
        if name in self.agent_widgets:
            self.agent_widgets[name]._append_activity(f"Activity: {act}")

    # ------------- Stats -------------------------------------------------
    def update_tree_item(self, name: str):
        d = self.agent_status_data[name]
        status, cpu = d["status"], d["cpu_usage"]
        for i in range(self.agent_tree.topLevelItemCount()):
            it = self.agent_tree.topLevelItem(i)
            if it.text(0) == name:
                it.setText(1, status)
                it.setText(2, f"{cpu:.1f}%")
                break

    def update_quick_stats(self):
        tot = len(self.agent_status_data)
        active = sum(1 for d in self.agent_status_data.values() if d["status"] == "active")
        idle = sum(1 for d in self.agent_status_data.values() if d["status"] == "idle")
        errs = sum(1 for d in self.agent_status_data.values() if d["status"] == "error")
        self.total_agents_label.setText(f"Total: {tot}")
        self.active_agents_label.setText(f"Active: {active}")
        self.idle_agents_label.setText(f"Idle: {idle}")
        self.error_agents_label.setText(f"Errors: {errs}")

    # ------------- Simulation -------------------------------------------
    def periodic_update(self):
        try:
            import random

            for name in self.agent_status_data:
                self._update_agent(
                    name,
                    dict(
                        status=random.choice(["active", "idle", "busy"]),
                        cpu_usage=random.uniform(0, 100),
                        memory_usage=random.uniform(10, 98),
                        avg_response_time=random.uniform(1, 100),
                        active_tasks=random.randint(0, 15),
                    ),
                )
            self.update_quick_stats()
        except Exception as e:
            logger.error(f"Periodic update error: {e}")


def create_individual_agents_tab(registry=None, event_bus=None) -> IndividualAgentsTab:
    return IndividualAgentsTab(registry, event_bus)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    tab = create_individual_agents_tab()
    tab.show()
    sys.exit(app.exec_())
