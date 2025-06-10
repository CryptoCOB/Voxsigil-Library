from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit
from PyQt5.QtGui import QTextCursor


class AgentStatusPanel(QWidget):
    """Panel showing agent import and runtime status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._box = QPlainTextEdit()
        self._box.setReadOnly(True)
        layout.addWidget(self._box)

    def add_status(self, text: str) -> None:
        self._box.appendPlainText(text)
        self._box.moveCursor(QTextCursor.End)

