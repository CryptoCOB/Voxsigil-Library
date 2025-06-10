from __future__ import annotations
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTabWidget,
    QPushButton,
    QScrollArea,
)

from .echo_log_panel import EchoLogPanel
from .mesh_map_panel import MeshMapPanel
from .agent_status_panel import AgentStatusPanel


class VoxSigilMainWindow(QMainWindow):
    """Simplified PyQt main window with placeholder tabs."""

    def __init__(self, registry=None, event_bus=None, async_bus=None):
        super().__init__()
        self.registry = registry
        self.event_bus = event_bus
        self.async_bus = async_bus
        self.echo_panel = None
        self.mesh_map_panel = None
        self.status_panel = None
        self.setWindowTitle("VoxSigil PyQt GUI")
        self.resize(1000, 800)
        self._init_ui()
        if self.event_bus:
            self.event_bus.subscribe("mesh_echo", self._on_mesh_echo)
            self.event_bus.subscribe("mesh_graph_update", self._on_mesh_graph)
            self.event_bus.subscribe("agent_status", self._on_status)
        if self.async_bus:
            try:
                self.async_bus.register_component("GUI")
                from Vanta.core.UnifiedAsyncBus import MessageType, AsyncMessage

                def _echo_cb(msg: AsyncMessage):
                    if self.echo_panel:
                        self.echo_panel.add_message(str(msg.content))

                self.async_bus.subscribe("GUI", MessageType.USER_INTERACTION, _echo_cb)
            except Exception:
                pass

    def _init_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_placeholder_tab("Models"), "Models")
        tabs.addTab(self._create_placeholder_tab("Testing"), "Testing")
        tabs.addTab(self._create_placeholder_tab("Training"), "Training")
        tabs.addTab(self._create_placeholder_tab("Visualization"), "Visualization")
        tabs.addTab(self._create_placeholder_tab("Performance"), "Performance")
        self.echo_panel = EchoLogPanel()
        self.mesh_map_panel = MeshMapPanel()
        self.status_panel = AgentStatusPanel()
        tabs.addTab(self.echo_panel, "Echo Log")
        tabs.addTab(self.mesh_map_panel, "Mesh Map")
        tabs.addTab(self.status_panel, "Agent Status")
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(tabs)
        if self.registry:
            layout.addWidget(self._create_agent_buttons())
        self.setCentralWidget(container)

    def _create_placeholder_tab(self, label: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"{label} tab not yet implemented."))
        return widget

    def _create_agent_buttons(self) -> QWidget:
        scroll = QScrollArea()
        container = QWidget()
        layout = QVBoxLayout(container)
        if not self.registry:
            layout.addWidget(QLabel("No agent registry available"))
        else:
            for name, agent in self.registry.get_all_agents():
                if hasattr(agent, "on_gui_call"):
                    btn = QPushButton(name)
                    btn.clicked.connect(agent.on_gui_call)
                    layout.addWidget(btn)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        return scroll

    def _on_mesh_echo(self, event):
        if self.echo_panel:
            msg = event.get("data")
            if isinstance(msg, dict):
                msg = msg.get("data")
            if msg is not None:
                self.echo_panel.add_message(str(msg))

    def _on_mesh_graph(self, event):
        if self.mesh_map_panel:
            graph = event.get("data")
            if graph is not None:
                self.mesh_map_panel.refresh(graph)

    def _on_status(self, event):
        if self.status_panel:
            msg = event.get("data")
            if msg:
                self.status_panel.add_status(str(msg))


def launch(registry=None, event_bus=None, async_bus=None):
    app = QApplication(sys.argv)
    win = VoxSigilMainWindow(registry, event_bus, async_bus)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    launch()
