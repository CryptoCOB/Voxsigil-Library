import logging

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("AgentStatusPanel")


class AgentStatusPanel(QWidget):
    """Enhanced panel showing agent import, runtime status, and voice controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tts_integration = None
        self._init_tts()
        self._init_ui()
        self._setup_timer()

    def _init_tts(self):
        """Initialize TTS integration."""
        try:
            # Add project root to path if needed
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from core.neural_tts_integration import get_tts_integration

            self.tts_integration = get_tts_integration()
            logger.info("✅ TTS integration loaded in Agent Status Panel")
        except Exception as e:
            logger.warning(f"⚠️ TTS integration not available: {e}")
            self.tts_integration = None

    def _init_ui(self):
        """Initialize the enhanced UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📈 Agent Status & Voice Control")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet(
            "color: #2E86AB; padding: 5px; background: #F0F8FF; border-radius: 3px;"
        )
        layout.addWidget(title)  # Voice control section
        voice_group = QGroupBox("🎤 Quick Voice Controls")
        voice_layout = QVBoxLayout()

        # Agent selection and controls
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Agent:"))
        self.agent_combo = QComboBox()
        control_layout.addWidget(self.agent_combo)

        self.speak_status_btn = QPushButton("🎵 Speak Status")
        self.speak_status_btn.clicked.connect(self._speak_agent_status)
        control_layout.addWidget(self.speak_status_btn)

        self.greeting_btn = QPushButton("👋 Greeting")
        self.greeting_btn.clicked.connect(self._speak_agent_greeting)
        control_layout.addWidget(self.greeting_btn)

        voice_layout.addLayout(control_layout)

        # Voice status
        self.voice_status = QLabel("🔧 Voice system ready")
        self.voice_status.setStyleSheet("padding: 3px; background: #E8F5E8; border-radius: 3px;")
        voice_layout.addWidget(self.voice_status)

        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)  # Status text area
        status_group = QGroupBox("📊 System Status")
        status_layout = QVBoxLayout()

        self._box = QPlainTextEdit()
        self._box.setReadOnly(True)
        self._box.setMaximumHeight(300)
        status_layout.addWidget(self._box)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Initialize with system info
        self._update_system_status()

        # Populate agent combo after UI is set up
        self._populate_agent_combo()

    def _populate_agent_combo(self):
        """Populate the agent selection combo box."""
        if self.tts_integration and self.tts_integration.is_available():
            try:
                voices = self.tts_integration.get_available_voices()
                self.agent_combo.addItems(voices)
                self.voice_status.setText(f"✅ {len(voices)} agent voices available")
            except Exception as e:
                self.voice_status.setText(f"⚠️ Voice loading error: {str(e)}")
        else:
            self.agent_combo.addItems(["Nova", "Aria", "Kai", "Echo", "Sage"])
            self.voice_status.setText("❌ Neural TTS not available")

    def _setup_timer(self):
        """Setup periodic status updates."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_voice_status)
        self.update_timer.start(10000)  # Update every 10 seconds

    def _update_voice_status(self):
        """Update voice system status."""
        if self.tts_integration and self.tts_integration.is_available():
            try:
                voices = self.tts_integration.get_available_voices()
                self.voice_status.setText(f"✅ {len(voices)} agent voices operational")
            except Exception as e:
                self.voice_status.setText(f"⚠️ Voice status check failed: {str(e)}")

    def _speak_agent_status(self):
        """Make the selected agent speak their status."""
        agent_name = self.agent_combo.currentText()
        if not agent_name:
            return

        # Create a status message for the agent
        status_messages = {
            "Nova": "I am Nova, your professional AI assistant. All systems are operational and I'm ready to help.",
            "Aria": "I am Aria. My systems are running with elegance and precision. How may I assist you?",
            "Kai": "Hey! I'm Kai and I'm super excited to report that everything is running perfectly!",
            "Echo": "I am Echo... monitoring from the depths of the neural network... all systems are... operational...",
            "Sage": "I am Sage, your wise counselor. All systems are functioning properly under my guidance.",
        }

        message = status_messages.get(agent_name, f"I am {agent_name}. System status: operational.")

        if self.tts_integration:
            try:
                success = self.tts_integration.speak_for_agent(agent_name, message, blocking=False)
                if success:
                    self.add_status(f"🎤 {agent_name} spoke status update")
                else:
                    self.add_status(f"❌ Failed to make {agent_name} speak")
            except Exception as e:
                self.add_status(f"⚠️ Voice error for {agent_name}: {str(e)}")
        else:
            self.add_status(f"❌ TTS not available for {agent_name}")

    def _speak_agent_greeting(self):
        """Make the selected agent speak their greeting."""
        agent_name = self.agent_combo.currentText()
        if not agent_name:
            return

        try:
            from core.neural_tts_integration import generate_agent_greeting

            success = generate_agent_greeting(agent_name)
            if success:
                self.add_status(f"👋 {agent_name} played greeting")
            else:
                self.add_status(f"❌ Failed to play {agent_name} greeting")
        except Exception as e:
            self.add_status(f"⚠️ Greeting error for {agent_name}: {str(e)}")

    def _update_system_status(self):
        """Update the system status display."""
        self.add_status("🎙️ VoxSigil Neural TTS System - Agent Status Panel")
        self.add_status("=" * 50)

        if self.tts_integration and self.tts_integration.is_available():
            voices = self.tts_integration.get_available_voices()
            self.add_status(f"✅ Neural TTS Operational - {len(voices)} agent voices")
            for voice in voices:
                voice_info = self.tts_integration.get_agent_voice_info(voice)
                if voice_info:
                    self.add_status(
                        f"🎤 {voice}: {voice_info['speaking_style']} {voice_info['gender']}"
                    )
        else:
            self.add_status("❌ Neural TTS system not available")

        self.add_status("")
        self.add_status("📋 Instructions:")
        self.add_status("• Select an agent from the dropdown")
        self.add_status("• Click 'Speak Status' to hear agent status")
        self.add_status("• Click 'Greeting' to hear agent greeting")
        self.add_status("• Use Neural TTS tab for advanced controls")

    def add_status(self, text: str) -> None:
        """Add status text to the display."""
        self._box.appendPlainText(text)
        self._box.moveCursor(QTextCursor.End)
