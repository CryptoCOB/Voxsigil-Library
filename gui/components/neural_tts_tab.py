"""
Neural TTS Tab for VoxSigil GUI
Provides a complete interface for the neural TTS system with agent voice controls.
"""

import logging
import sys

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("NeuralTTSTab")


class TTSSynthesisThread(QThread):
    """Background thread for TTS synthesis to prevent GUI blocking."""

    synthesis_complete = pyqtSignal(bool, str)  # success, message
    progress_update = pyqtSignal(int)  # progress percentage

    def __init__(self, tts_engine, text, voice_profile, output_path=None):
        super().__init__()
        self.tts_engine = tts_engine
        self.text = text
        self.voice_profile = voice_profile
        self.output_path = output_path
        self.synthesis_type = "speak" if output_path is None else "file"

    def run(self):
        """Run TTS synthesis in background."""
        try:
            self.progress_update.emit(20)

            if self.synthesis_type == "speak":
                # Direct speech synthesis
                success = self.tts_engine.speak_text(self.text, self.voice_profile, blocking=True)
                self.progress_update.emit(80)

                if success:
                    self.synthesis_complete.emit(
                        True, f"Speech synthesis completed for {self.voice_profile}"
                    )
                else:
                    self.synthesis_complete.emit(
                        False, f"Speech synthesis failed for {self.voice_profile}"
                    )

            else:
                # File generation
                result_path = self.tts_engine.synthesize_speech(
                    text=self.text, voice_profile=self.voice_profile, output_path=self.output_path
                )
                self.progress_update.emit(80)

                if result_path:
                    self.synthesis_complete.emit(True, f"Audio file saved: {result_path}")
                else:
                    self.synthesis_complete.emit(False, "Failed to generate audio file")

            self.progress_update.emit(100)

        except Exception as e:
            self.synthesis_complete.emit(False, f"TTS Error: {str(e)}")


class AgentVoiceWidget(QWidget):
    """Widget for controlling an individual agent's voice."""

    def __init__(self, agent_name, voice_info, tts_engine):
        super().__init__()
        self.agent_name = agent_name
        self.voice_info = voice_info
        self.tts_engine = tts_engine
        self._init_ui()

    def _init_ui(self):
        """Initialize the agent voice control UI."""
        layout = QVBoxLayout()

        # Agent header
        header = QLabel(f"🎤 {self.agent_name}")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        header.setStyleSheet("color: #2E86AB; padding: 5px;")
        layout.addWidget(header)

        # Voice characteristics
        char_layout = QGridLayout()

        char_layout.addWidget(QLabel("Style:"), 0, 0)
        char_layout.addWidget(QLabel(self.voice_info["speaking_style"].title()), 0, 1)

        char_layout.addWidget(QLabel("Gender:"), 1, 0)
        char_layout.addWidget(QLabel(self.voice_info["gender"].title()), 1, 1)

        char_layout.addWidget(QLabel("Emotion:"), 2, 0)
        char_layout.addWidget(QLabel(self.voice_info["emotion"].title()), 2, 1)

        layout.addLayout(char_layout)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.test_btn = QPushButton("🎵 Test Voice")
        self.test_btn.clicked.connect(self._test_voice)
        btn_layout.addWidget(self.test_btn)

        self.greeting_btn = QPushButton("👋 Greeting")
        self.greeting_btn.clicked.connect(self._play_greeting)
        btn_layout.addWidget(self.greeting_btn)

        layout.addLayout(btn_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        self.setLayout(layout)

    def _test_voice(self):
        """Test the agent's voice."""
        test_text = f"Hello! I'm {self.agent_name}, and this is how I sound."
        try:
            from core.neural_tts_integration import agent_speak

            success = agent_speak(self.agent_name, test_text, blocking=False)
            if not success:
                QMessageBox.warning(
                    self, "TTS Error", f"Failed to test voice for {self.agent_name}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"TTS system error: {str(e)}")

    def _play_greeting(self):
        """Play the agent's greeting."""
        try:
            from core.neural_tts_integration import generate_agent_greeting

            success = generate_agent_greeting(self.agent_name)
            if not success:
                QMessageBox.warning(
                    self, "TTS Error", f"Failed to play greeting for {self.agent_name}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"TTS system error: {str(e)}")


class NeuralTTSTab(QWidget):
    """Main Neural TTS control tab for VoxSigil GUI."""

    def __init__(self):
        super().__init__()
        self.tts_engine = None
        self.synthesis_thread = None
        self.agent_widgets = {}
        self._init_neural_tts()
        self._init_ui()
        self._setup_timer()

    def _init_neural_tts(self):
        """Initialize the neural TTS engine."""
        try:
            # Add project root to path if needed
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from core.production_neural_tts import ProductionNeuralTTS

            self.tts_engine = ProductionNeuralTTS()
            logger.info("✅ Neural TTS engine initialized in GUI")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neural TTS in GUI: {e}")
            self.tts_engine = None

    def _init_ui(self):
        """Initialize the Neural TTS tab UI."""
        main_layout = QHBoxLayout()

        # Left panel - Agent Voice Controls
        left_panel = self._create_agent_panel()

        # Right panel - TTS Controls
        right_panel = self._create_control_panel()

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_agent_panel(self):
        """Create the agent voice control panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("🎭 Agent Voice Profiles")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(
            "color: #2E86AB; padding: 10px; background: #F0F8FF; border-radius: 5px;"
        )
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("🔧 Initializing TTS system...")
        layout.addWidget(self.status_label)

        # Scroll area for agent widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        agent_container = QWidget()
        agent_layout = QVBoxLayout()

        if self.tts_engine:
            try:
                voices = self.tts_engine.list_available_voices()
                self.status_label.setText(f"✅ {len(voices)} agent voices available")

                for voice_name in voices:
                    voice_info = self.tts_engine.get_voice_info(voice_name)
                    if voice_info:
                        agent_widget = AgentVoiceWidget(voice_name, voice_info, self.tts_engine)
                        agent_layout.addWidget(agent_widget)
                        self.agent_widgets[voice_name] = agent_widget

            except Exception as e:
                self.status_label.setText(f"❌ Error loading voices: {str(e)}")
        else:
            self.status_label.setText("❌ Neural TTS engine not available")

        agent_layout.addStretch()
        agent_container.setLayout(agent_layout)
        scroll.setWidget(agent_container)

        layout.addWidget(scroll)

        # Global controls
        global_group = QGroupBox("🌐 Global Controls")
        global_layout = QHBoxLayout()

        self.test_all_btn = QPushButton("🎵 Test All Voices")
        self.test_all_btn.clicked.connect(self._test_all_voices)
        global_layout.addWidget(self.test_all_btn)

        self.stop_all_btn = QPushButton("⏹️ Stop All")
        self.stop_all_btn.clicked.connect(self._stop_all_synthesis)
        global_layout.addWidget(self.stop_all_btn)

        global_group.setLayout(global_layout)
        layout.addWidget(global_group)

        panel.setLayout(layout)
        return panel

    def _create_control_panel(self):
        """Create the main TTS control panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("🎙️ Neural TTS Control Center")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(
            "color: #2E86AB; padding: 10px; background: #F0F8FF; border-radius: 5px;"
        )
        layout.addWidget(title)

        # Tab widget for different controls
        control_tabs = QTabWidget()

        # Speech synthesis tab
        speech_tab = self._create_speech_tab()
        control_tabs.addTab(speech_tab, "🗣️ Speech Synthesis")

        # File generation tab
        file_tab = self._create_file_generation_tab()
        control_tabs.addTab(file_tab, "💾 File Generation")

        # System info tab
        info_tab = self._create_system_info_tab()
        control_tabs.addTab(info_tab, "ℹ️ System Info")

        layout.addWidget(control_tabs)
        panel.setLayout(layout)
        return panel

    def _create_speech_tab(self):
        """Create the speech synthesis tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Text input
        text_group = QGroupBox("📝 Text Input")
        text_layout = QVBoxLayout()

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text for speech synthesis...")
        self.text_input.setMaximumHeight(100)
        text_layout.addWidget(self.text_input)

        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

        # Voice selection
        voice_group = QGroupBox("🎭 Voice Selection")
        voice_layout = QHBoxLayout()

        voice_layout.addWidget(QLabel("Agent:"))
        self.voice_combo = QComboBox()
        if self.tts_engine:
            self.voice_combo.addItems(self.tts_engine.list_available_voices())
        voice_layout.addWidget(self.voice_combo)

        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)

        # Synthesis controls
        synth_group = QGroupBox("🎚️ Synthesis Controls")
        synth_layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        synth_layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()

        self.speak_btn = QPushButton("🎵 Speak Text")
        self.speak_btn.clicked.connect(self._speak_text)
        btn_layout.addWidget(self.speak_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self._stop_synthesis)
        btn_layout.addWidget(self.stop_btn)

        synth_layout.addLayout(btn_layout)
        synth_group.setLayout(synth_layout)
        layout.addWidget(synth_group)

        # Status
        self.synthesis_status = QLabel("Ready for speech synthesis")
        self.synthesis_status.setStyleSheet(
            "padding: 5px; background: #E8F5E8; border-radius: 3px;"
        )
        layout.addWidget(self.synthesis_status)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_file_generation_tab(self):
        """Create the file generation tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Text input
        text_group = QGroupBox("📝 Text for Audio File")
        text_layout = QVBoxLayout()

        self.file_text_input = QTextEdit()
        self.file_text_input.setPlaceholderText("Enter text to generate audio file...")
        self.file_text_input.setMaximumHeight(100)
        text_layout.addWidget(self.file_text_input)

        text_group.setLayout(text_layout)
        layout.addWidget(text_group)

        # File settings
        file_group = QGroupBox("📁 File Settings")
        file_layout = QGridLayout()

        file_layout.addWidget(QLabel("Agent:"), 0, 0)
        self.file_voice_combo = QComboBox()
        if self.tts_engine:
            self.file_voice_combo.addItems(self.tts_engine.list_available_voices())
        file_layout.addWidget(self.file_voice_combo, 0, 1)

        file_layout.addWidget(QLabel("Output:"), 1, 0)
        self.output_path_label = QLabel("No file selected")
        file_layout.addWidget(self.output_path_label, 1, 1)

        self.browse_btn = QPushButton("📂 Browse...")
        self.browse_btn.clicked.connect(self._browse_output_file)
        file_layout.addWidget(self.browse_btn, 1, 2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Generation controls
        gen_group = QGroupBox("⚙️ Generation Controls")
        gen_layout = QVBoxLayout()

        # Progress bar
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setVisible(False)
        gen_layout.addWidget(self.file_progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()

        self.generate_btn = QPushButton("🎵 Generate Audio File")
        self.generate_btn.clicked.connect(self._generate_audio_file)
        btn_layout.addWidget(self.generate_btn)

        self.play_file_btn = QPushButton("▶️ Play Generated File")
        self.play_file_btn.clicked.connect(self._play_generated_file)
        self.play_file_btn.setEnabled(False)
        btn_layout.addWidget(self.play_file_btn)

        gen_layout.addLayout(btn_layout)
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)

        # Status
        self.file_status = QLabel("Ready for file generation")
        self.file_status.setStyleSheet("padding: 5px; background: #E8F5E8; border-radius: 3px;")
        layout.addWidget(self.file_status)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_system_info_tab(self):
        """Create the system information tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # System status
        status_group = QGroupBox("🔧 System Status")
        status_layout = QVBoxLayout()

        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(200)
        status_layout.addWidget(self.system_info)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Engine information
        engine_group = QGroupBox("🎯 TTS Engines")
        engine_layout = QVBoxLayout()

        self.engine_info = QTextEdit()
        self.engine_info.setReadOnly(True)
        self.engine_info.setMaximumHeight(150)
        engine_layout.addWidget(self.engine_info)

        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)

        # Voice profiles
        profiles_group = QGroupBox("🎭 Voice Profiles")
        profiles_layout = QVBoxLayout()

        self.profiles_info = QTextEdit()
        self.profiles_info.setReadOnly(True)
        profiles_layout.addWidget(self.profiles_info)

        profiles_group.setLayout(profiles_layout)
        layout.addWidget(profiles_group)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh System Info")
        refresh_btn.clicked.connect(self._refresh_system_info)
        layout.addWidget(refresh_btn)

        tab.setLayout(layout)

        # Load initial info
        self._refresh_system_info()

        return tab

    def _setup_timer(self):
        """Setup status update timer."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(5000)  # Update every 5 seconds

    def _update_status(self):
        """Update status information."""
        if self.tts_engine:
            try:
                engines = self.tts_engine.get_available_engines()
                voices = self.tts_engine.list_available_voices()
                self.status_label.setText(
                    f"✅ {len(engines)} engines, {len(voices)} voices available"
                )
            except Exception as e:
                self.status_label.setText(f"⚠️ Status update error: {str(e)}")

    def _speak_text(self):
        """Speak the input text."""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter text to speak")
            return

        voice = self.voice_combo.currentText()
        if not voice:
            QMessageBox.warning(self, "Voice Error", "Please select a voice")
            return

        self._start_synthesis(text, voice, synthesis_type="speak")

    def _generate_audio_file(self):
        """Generate an audio file."""
        text = self.file_text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter text for audio file generation")
            return

        voice = self.file_voice_combo.currentText()
        if not voice:
            QMessageBox.warning(self, "Voice Error", "Please select a voice")
            return

        output_path = self.output_path_label.text()
        if output_path == "No file selected":
            # Auto-generate filename
            import os

            output_path = os.path.join(os.getcwd(), f"tts_output_{voice.lower()}.wav")
            self.output_path_label.setText(output_path)

        self._start_synthesis(text, voice, synthesis_type="file", output_path=output_path)

    def _start_synthesis(self, text, voice, synthesis_type="speak", output_path=None):
        """Start TTS synthesis in background thread."""
        if self.synthesis_thread and self.synthesis_thread.isRunning():
            QMessageBox.warning(self, "TTS Busy", "TTS synthesis is already running")
            return

        if not self.tts_engine:
            QMessageBox.critical(self, "TTS Error", "Neural TTS engine is not available")
            return

        # Setup progress bar
        if synthesis_type == "speak":
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.synthesis_status.setText("🎵 Speaking...")
        else:
            self.file_progress_bar.setVisible(True)
            self.file_progress_bar.setValue(0)
            self.file_status.setText("🎵 Generating audio file...")
        # Start synthesis thread
        self.synthesis_thread = TTSSynthesisThread(self.tts_engine, text, voice, output_path)
        self.synthesis_thread.synthesis_complete.connect(self._on_synthesis_complete)
        self.synthesis_thread.progress_update.connect(self._on_progress_update)
        self.synthesis_thread.start()

    def _on_synthesis_complete(self, success, message):
        """Handle synthesis completion."""
        self.progress_bar.setVisible(False)
        self.file_progress_bar.setVisible(False)

        if success:
            self.synthesis_status.setText(f"✅ {message}")
            self.file_status.setText(f"✅ {message}")
            if "Audio file saved:" in message:
                self.play_file_btn.setEnabled(True)
                self.output_path_label.setText(message.split(": ", 1)[1])
        else:
            self.synthesis_status.setText(f"❌ {message}")
            self.file_status.setText(f"❌ {message}")

    def _on_progress_update(self, value):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.file_progress_bar.setValue(value)

    def _stop_synthesis(self):
        """Stop ongoing synthesis."""
        if self.synthesis_thread and self.synthesis_thread.isRunning():
            self.synthesis_thread.terminate()
            self.synthesis_thread.wait()
            self.progress_bar.setVisible(False)
            self.file_progress_bar.setVisible(False)
            self.synthesis_status.setText("⏹️ Synthesis stopped")
            self.file_status.setText("⏹️ Synthesis stopped")

    def _browse_output_file(self):
        """Browse for output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio File", "", "WAV Files (*.wav);;All Files (*)"
        )
        if file_path:
            self.output_path_label.setText(file_path)

    def _play_generated_file(self):
        """Play the generated audio file."""
        import os
        import subprocess

        file_path = self.output_path_label.text()
        if os.path.exists(file_path):
            try:
                # Try to open with default audio player
                if os.name == "nt":  # Windows
                    os.startfile(file_path)
                elif os.name == "posix":  # macOS and Linux
                    subprocess.call(["open" if sys.platform == "darwin" else "xdg-open", file_path])
            except Exception as e:
                QMessageBox.warning(self, "Playback Error", f"Could not open audio file: {str(e)}")
        else:
            QMessageBox.warning(self, "File Error", "Generated audio file not found")

    def _test_all_voices(self):
        """Test all available agent voices."""
        if not self.tts_engine:
            QMessageBox.critical(self, "TTS Error", "Neural TTS engine is not available")
            return

        try:
            from core.neural_tts_integration import VoxSigilTTSIntegration

            integration = VoxSigilTTSIntegration()

            # Run voice tests in background
            def run_tests():
                results = integration.test_all_agent_voices()
                success_count = sum(results.values())
                total_count = len(results)

                QMessageBox.information(
                    self,
                    "Voice Test Results",
                    f"Voice test completed!\n\nSuccessful: {success_count}/{total_count}\n\nCheck the audio output to hear each agent voice.",
                )

            # Run in separate thread to prevent GUI blocking
            import threading

            threading.Thread(target=run_tests, daemon=True).start()

        except Exception as e:
            QMessageBox.critical(self, "Test Error", f"Failed to test voices: {str(e)}")

    def _stop_all_synthesis(self):
        """Stop all TTS synthesis."""
        self._stop_synthesis()
        # Additional cleanup if needed

    def _refresh_system_info(self):
        """Refresh system information display."""
        if not self.tts_engine:
            self.system_info.setText("❌ Neural TTS engine not available")
            self.engine_info.setText("No TTS engines loaded")
            self.profiles_info.setText("No voice profiles available")
            return

        try:
            # System status
            engines = self.tts_engine.get_available_engines()
            voices = self.tts_engine.list_available_voices()

            system_text = f"""✅ Neural TTS System Status: OPERATIONAL
🎯 Available Engines: {len(engines)}
🎭 Voice Profiles: {len(voices)}
🔧 Device: {self.tts_engine.device}
📁 Cache Directory: {self.tts_engine.cache_dir}
🎵 Sample Rate: {self.tts_engine.sample_rate} Hz"""

            self.system_info.setText(system_text)

            # Engine information
            engine_text = "Available TTS Engines:\n\n"
            for i, engine in enumerate(engines, 1):
                engine_text += f"{i}. {engine.upper()}\n"
                if engine == "speecht5":
                    engine_text += "   - Microsoft SpeechT5 Neural TTS\n"
                    engine_text += "   - High-quality neural voice synthesis\n"
                elif engine == "pyttsx3":
                    engine_text += "   - System TTS with voice processing\n"
                    engine_text += "   - Reliable fallback engine\n"
                engine_text += "\n"

            self.engine_info.setText(engine_text)

            # Voice profiles
            profiles_text = "Agent Voice Profiles:\n\n"
            for voice_name in voices:
                voice_info = self.tts_engine.get_voice_info(voice_name)
                if voice_info:
                    profiles_text += f"🎤 {voice_name}\n"
                    profiles_text += f"   Style: {voice_info['speaking_style'].title()}\n"
                    profiles_text += f"   Gender: {voice_info['gender'].title()}\n"
                    profiles_text += f"   Emotion: {voice_info['emotion'].title()}\n"
                    profiles_text += f"   Speed: {voice_info['speed']}x\n"
                    profiles_text += f"   Energy: {voice_info['energy']}x\n\n"

            self.profiles_info.setText(profiles_text)

        except Exception as e:
            error_text = f"❌ Error loading system info: {str(e)}"
            self.system_info.setText(error_text)
            self.engine_info.setText(error_text)
            self.profiles_info.setText(error_text)
