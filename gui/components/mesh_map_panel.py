from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class MeshMapPanel(QWidget):
    """Minimal panel displaying the current agent mesh graph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("No graph available")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def refresh(self, graph: dict) -> None:
        """Refresh panel contents with new graph data."""
        if not graph:
            self.label.setText("No graph available")
            return
        lines = []
        agents = graph.get("agents", {})
        for name, color in agents.items():
            lines.append(f'<span style="color:{color}">{name}</span>')
        self.label.setText("<br>".join(lines))
        self.label.setAlignment(Qt.AlignTop)
