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


class VoxSigilMainWindow(QMainWindow):
    """Simplified PyQt main window with placeholder tabs."""

    def __init__(self, registry=None):
        super().__init__()
        self.registry = registry
        self.setWindowTitle("VoxSigil PyQt GUI")
        self.resize(1000, 800)
        self._init_ui()

    def _init_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_placeholder_tab("Models"), "Models")
        tabs.addTab(self._create_placeholder_tab("Testing"), "Testing")
        tabs.addTab(self._create_placeholder_tab("Training"), "Training")
        tabs.addTab(self._create_placeholder_tab("Visualization"), "Visualization")
        tabs.addTab(self._create_placeholder_tab("Performance"), "Performance")
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


def launch(registry=None):
    app = QApplication(sys.argv)
    win = VoxSigilMainWindow(registry)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    launch()
