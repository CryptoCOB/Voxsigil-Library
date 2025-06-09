from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class MeshMapPanel(QWidget):
    """Minimal panel displaying the current agent mesh graph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("No graph available")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def refresh(self, graph: dict) -> None:
        """Refresh panel contents with new graph data."""
        self.label.setText(str(graph))
