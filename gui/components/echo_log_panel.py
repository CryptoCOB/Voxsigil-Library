from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit

class EchoLogPanel(QWidget):
    """Simple scrolling panel that displays echo messages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._box = QPlainTextEdit()
        self._box.setReadOnly(True)
        layout.addWidget(self._box)

    def add_message(self, message: str) -> None:
        """Append a new message to the panel."""
        self._box.appendPlainText(message)
