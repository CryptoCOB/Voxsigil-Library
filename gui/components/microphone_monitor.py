"""
Microphone Monitor Component for VoxSigil GUI

This component provides visual feedback for microphone status,
voice activity detection, and STT functionality with agent voice recognition.
"""

import logging
from typing import Optional

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .gui_styles import VoxSigilStyles, VoxSigilWidgetFactory

logger = logging.getLogger(__name__)


class MicrophoneMonitor(QWidget):
    """Microphone monitoring widget with voice activity detection."""

    # Signals
    voice_detected = pyqtSignal(str)  # Emitted when voice is transcribed
    mic_status_changed = pyqtSignal(str)  # Emitted when mic status changes

    def __init__(self, vanta_core=None):
        super().__init__()
        self.vanta_core = vanta_core
        self.is_listening = False
        self.voice_level = 0
        self.setup_ui()
        self.setup_monitoring()

    def setup_ui(self):
        """Setup the microphone monitoring UI."""
        layout = QVBoxLayout(self)

        # Microphone Status Group
        mic_group = QGroupBox("🎤 Microphone Status")
        mic_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        mic_layout = QVBoxLayout(mic_group)

        # Status indicators
        status_layout = QHBoxLayout()

        self.mic_status_label = VoxSigilWidgetFactory.create_label("🔴 Microphone Offline", "error")
        self.listening_status_label = VoxSigilWidgetFactory.create_label("⏸️ Not Listening", "info")

        status_layout.addWidget(self.mic_status_label)
        status_layout.addWidget(self.listening_status_label)
        mic_layout.addLayout(status_layout)

        # Voice level indicator
        voice_level_layout = QHBoxLayout()
        voice_level_layout.addWidget(QLabel("Voice Level:"))

        self.voice_level_bar = VoxSigilWidgetFactory.create_progress_bar()
        self.voice_level_bar.setMaximum(100)
        self.voice_level_bar.setValue(0)
        voice_level_layout.addWidget(self.voice_level_bar)

        mic_layout.addLayout(voice_level_layout)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.listen_button = QPushButton("🎧 Start Listening")
        self.listen_button.setStyleSheet(VoxSigilStyles.get_button_stylesheet())
        self.listen_button.clicked.connect(self.toggle_listening)

        self.test_mic_button = QPushButton("🧪 Test Microphone")
        self.test_mic_button.setStyleSheet(VoxSigilStyles.get_button_stylesheet())
        self.test_mic_button.clicked.connect(self.test_microphone)

        controls_layout.addWidget(self.listen_button)
        controls_layout.addWidget(self.test_mic_button)
        mic_layout.addLayout(controls_layout)

        layout.addWidget(mic_group)

        # Voice Transcription Group
        transcription_group = QGroupBox("📝 Voice Transcription")
        transcription_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        transcription_layout = QVBoxLayout(transcription_group)

        # Last transcription
        self.last_transcription_label = VoxSigilWidgetFactory.create_label(
            "Last Command: None", "info"
        )
        transcription_layout.addWidget(self.last_transcription_label)

        # Transcription history
        self.transcription_history = QTextEdit()
        self.transcription_history.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        self.transcription_history.setMaximumHeight(150)
        self.transcription_history.setReadOnly(True)
        transcription_layout.addWidget(self.transcription_history)

        # Agent Recognition
        agent_layout = QHBoxLayout()
        agent_layout.addWidget(QLabel("Detected Agent:"))
        self.detected_agent_label = VoxSigilWidgetFactory.create_label("None", "info")
        agent_layout.addWidget(self.detected_agent_label)
        transcription_layout.addLayout(agent_layout)

        layout.addWidget(transcription_group)

        # Voice Command Status
        command_group = QGroupBox("⚡ Voice Commands")
        command_group.setStyleSheet(VoxSigilStyles.get_group_box_stylesheet())
        command_layout = QVBoxLayout(command_group)

        self.command_status_label = VoxSigilWidgetFactory.create_label(
            "Ready for commands", "success"
        )
        command_layout.addWidget(self.command_status_label)

        # Common commands help
        help_text = QLabel(
            "Try saying:\\n"
            "• 'Astra, navigate to...'\\n"
            "• 'Andy, compose output...'\\n"
            "• 'Voxka, analyze...'\\n"
            "• 'Oracle, what is...'\\n"
            "• 'Start training'\\n"
            "• 'Show status'"
        )
        help_text.setStyleSheet("color: #64B5F6; font-size: 11px;")
        help_text.setWordWrap(True)
        command_layout.addWidget(help_text)

        layout.addWidget(command_group)

    def setup_monitoring(self):
        """Setup microphone monitoring timers."""
        # Voice level monitoring timer
        self.voice_monitor_timer = QTimer()
        self.voice_monitor_timer.timeout.connect(self.update_voice_level)
        self.voice_monitor_timer.start(100)  # Update every 100ms

        # Microphone status check timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_microphone_status)
        self.status_timer.start(2000)  # Check every 2 seconds

        # Initial status check
        self.check_microphone_status()

    def check_microphone_status(self):
        """Check if microphone is available and working."""
        try:
            # Try to detect if STT system is available
            if self.vanta_core:
                stt_engine = self.vanta_core.get_component("async_stt_engine")
                if stt_engine:
                    self.mic_status_label.setText("🟢 Microphone Ready")
                    self.mic_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                    self.mic_status_changed.emit("ready")
                    return  # Check if sounddevice is available (basic check)
            try:
                import sounddevice as sd

                # Test if we can query devices
                devices = sd.query_devices()
                if devices:
                    self.mic_status_label.setText("🟡 Microphone Available")
                    self.mic_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                    self.mic_status_changed.emit("available")
                else:
                    self.mic_status_label.setText("🔴 No Microphone Detected")
                    self.mic_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                    self.mic_status_changed.emit("unavailable")

            except ImportError:
                self.mic_status_label.setText("🔴 Audio Libraries Missing")
                self.mic_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self.mic_status_changed.emit("missing_libs")

        except Exception as e:
            logger.error(f"Error checking microphone status: {e}")
            self.mic_status_label.setText("🔴 Microphone Error")
            self.mic_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self.mic_status_changed.emit("error")

    def update_voice_level(self):
        """Update voice level indicator."""
        try:
            if self.is_listening:
                # Simulate voice level for now - would connect to real audio input
                import random

                self.voice_level = max(0, self.voice_level + random.randint(-10, 15))
                self.voice_level = min(100, self.voice_level)

                self.voice_level_bar.setValue(self.voice_level)

                # Change color based on level
                if self.voice_level > 70:
                    self.voice_level_bar.setStyleSheet(
                        "QProgressBar::chunk { background-color: #4CAF50; }"
                    )
                elif self.voice_level > 30:
                    self.voice_level_bar.setStyleSheet(
                        "QProgressBar::chunk { background-color: #FF9800; }"
                    )
                else:
                    self.voice_level_bar.setStyleSheet(
                        "QProgressBar::chunk { background-color: #2196F3; }"
                    )
            else:
                self.voice_level = max(0, self.voice_level - 5)
                self.voice_level_bar.setValue(self.voice_level)

        except Exception as e:
            logger.error(f"Error updating voice level: {e}")

    def toggle_listening(self):
        """Toggle voice listening on/off."""
        try:
            if not self.is_listening:
                self.start_listening()
            else:
                self.stop_listening()
        except Exception as e:
            logger.error(f"Error toggling listening: {e}")

    def start_listening(self):
        """Start listening for voice commands."""
        try:
            self.is_listening = True
            self.listen_button.setText("🛑 Stop Listening")
            self.listening_status_label.setText("🎧 Listening...")
            self.listening_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.command_status_label.setText("Speak now...")

            # Start actual voice recognition if available
            if self.vanta_core:
                self.start_voice_recognition()

            logger.info("Voice listening started")

        except Exception as e:
            logger.error(f"Error starting voice listening: {e}")
            self.stop_listening()

    def stop_listening(self):
        """Stop listening for voice commands."""
        try:
            self.is_listening = False
            self.listen_button.setText("🎧 Start Listening")
            self.listening_status_label.setText("⏸️ Not Listening")
            self.listening_status_label.setStyleSheet("color: #757575; font-weight: normal;")
            self.command_status_label.setText("Ready for commands")

            logger.info("Voice listening stopped")

        except Exception as e:
            logger.error(f"Error stopping voice listening: {e}")

    def start_voice_recognition(self):
        """Start the actual voice recognition process."""
        try:
            # Try to get STT engine
            if self.vanta_core:
                stt_engine = self.vanta_core.get_component("async_stt_engine")
                if stt_engine:
                    # Start a background thread for continuous recognition
                    self.recognition_thread = VoiceRecognitionThread(stt_engine)
                    self.recognition_thread.transcription_ready.connect(
                        self.on_transcription_received
                    )
                    self.recognition_thread.start()
                    return

            # Fallback: simulate voice recognition for demo
            self.simulate_voice_recognition()

        except Exception as e:
            logger.error(f"Error starting voice recognition: {e}")

    def simulate_voice_recognition(self):
        """Simulate voice recognition for demonstration."""
        import random

        # Schedule simulated transcriptions
        sample_commands = [
            "Astra navigate to control center",
            "Andy compose status report",
            "Voxka analyze system performance",
            "Oracle show me the data",
            "Start training pipeline",
            "Show system status",
        ]

        def simulate_command():
            if self.is_listening:
                command = random.choice(sample_commands)
                self.on_transcription_received(command)
                # Schedule next simulation
                QTimer.singleShot(random.randint(5000, 15000), simulate_command)

        # Start first simulation
        QTimer.singleShot(3000, simulate_command)

    def on_transcription_received(self, text: str):
        """Handle received voice transcription."""
        try:
            if not text.strip():
                return

            # Update UI
            self.last_transcription_label.setText(f"Last Command: {text}")

            # Add to history
            timestamp = self.get_timestamp()
            self.transcription_history.append(f"[{timestamp}] {text}")

            # Detect agent from command
            agent_name = self.detect_agent_from_command(text)
            if agent_name:
                self.detected_agent_label.setText(agent_name)
                self.detected_agent_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                self.detected_agent_label.setText("General Command")
                self.detected_agent_label.setStyleSheet("color: #2196F3; font-weight: bold;")

            # Emit signal for other components
            self.voice_detected.emit(text)

            # Process the command
            self.process_voice_command(text, agent_name)

            logger.info(f"Voice command received: {text}")

        except Exception as e:
            logger.error(f"Error processing transcription: {e}")

    def detect_agent_from_command(self, text: str) -> Optional[str]:
        """Detect which agent is being addressed in the command."""
        text_lower = text.lower()

        # Known agent names and variations
        agent_keywords = {
            "Astra": ["astra", "navigation", "navigate"],
            "Andy": ["andy", "compose", "output"],
            "Voxka": ["voxka", "voice", "phi", "cognition"],
            "Oracle": ["oracle", "wisdom", "knowledge"],
            "Sam": ["sam", "help", "assist"],
            "Dave": ["dave", "data", "analyze"],
            "Dreamer": ["dreamer", "dream", "vision"],
            "Echo": ["echo", "communicate", "message"],
            "Warden": ["warden", "security", "protect"],
            "Carla": ["carla", "style", "creative"],
        }

        for agent_name, keywords in agent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return agent_name

        return None

    def process_voice_command(self, text: str, agent_name: Optional[str]):
        """Process the voice command and trigger appropriate actions."""
        try:
            text_lower = text.lower()

            # Route to specific agent if detected
            if agent_name and self.vanta_core:
                # Try to get the agent and make it respond
                agent = self.vanta_core.get_component(agent_name.lower())
                if agent and hasattr(agent, "speak"):
                    response = f"Command received: {text}"
                    agent.speak(response)

            # Handle general commands
            if "status" in text_lower:
                self.command_status_label.setText("Showing system status...")
            elif "training" in text_lower:
                self.command_status_label.setText("Initiating training...")
            elif "stop" in text_lower or "quit" in text_lower:
                self.stop_listening()
            else:
                self.command_status_label.setText(f"Processing: {text[:30]}...")

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")

    def test_microphone(self):
        """Test microphone functionality."""
        try:
            self.command_status_label.setText("Testing microphone...")

            # Basic microphone test
            try:
                import numpy as np
                import sounddevice as sd

                # Record a short sample
                duration = 1  # seconds
                sample_rate = 16000

                self.command_status_label.setText("Recording test sample...")
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                sd.wait()  # Wait until recording is finished

                # Calculate volume level
                volume = np.sqrt(np.mean(recording**2))

                if volume > 0.001:  # Threshold for detecting sound
                    self.command_status_label.setText(f"✅ Microphone working! Level: {volume:.3f}")
                    self.command_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                else:
                    self.command_status_label.setText("⚠️ Microphone detected but no sound")
                    self.command_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")

            except ImportError:
                self.command_status_label.setText("❌ Audio libraries not available")
                self.command_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            except Exception as e:
                self.command_status_label.setText(f"❌ Microphone test failed: {str(e)[:30]}")
                self.command_status_label.setStyleSheet("color: #F44336; font-weight: bold;")

        except Exception as e:
            logger.error(f"Error testing microphone: {e}")

    def get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")


class VoiceRecognitionThread(QThread):
    """Background thread for continuous voice recognition."""

    transcription_ready = pyqtSignal(str)

    def __init__(self, stt_engine):
        super().__init__()
        self.stt_engine = stt_engine
        self.running = True

    def run(self):
        """Run continuous voice recognition."""
        try:
            while self.running:
                # This would implement actual continuous recognition
                # For now, just a placeholder
                self.msleep(1000)  # Sleep for 1 second

        except Exception as e:
            logger.error(f"Error in voice recognition thread: {e}")

    def stop(self):
        """Stop the recognition thread."""
        self.running = False
        self.wait()
