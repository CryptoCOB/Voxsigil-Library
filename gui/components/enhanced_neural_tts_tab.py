"""
Enhanced Neural TTS Tab with Development Mode Controls
Comprehensive TTS interface with configurable dev mode options.
"""

import logging
from typing import Any, Optional

from PyQt5.QtCore import QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.dev_config_manager import get_dev_config
from gui.components.dev_mode_panel import DevModeControlPanel

# Import TTS components
try:
    from core.neural_tts_integration import get_tts_integration

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

logger = logging.getLogger("EnhancedNeuralTTSTab")


class TTSWorker(QThread):
    """Worker thread for TTS operations."""

    synthesis_started = pyqtSignal()
    synthesis_finished = pyqtSignal(bool, str)  # success, message
    progress_updated = pyqtSignal(int)  # progress percentage

    def __init__(self, text: str, agent: str, output_path: Optional[str] = None):
        super().__init__()
        self.text = text
        self.agent = agent
        self.output_path = output_path
        self.tts_integration = None

    def run(self):
        """Run TTS synthesis in background thread."""
        try:
            self.synthesis_started.emit()
            self.progress_updated.emit(10)

            if TTS_AVAILABLE:
                self.tts_integration = get_tts_integration()
                self.progress_updated.emit(30)

                if self.output_path:
                    # Generate audio file
                    result = self.tts_integration.generate_agent_audio_file(
                        self.agent, self.text, self.output_path
                    )
                    self.progress_updated.emit(80)

                    success = result is not None
                    message = f"Audio saved to {result}" if success else "Failed to generate audio"
                else:
                    # Direct speech
                    success = self.tts_integration.speak_for_agent(
                        self.agent, self.text, blocking=True
                    )
                    self.progress_updated.emit(80)
                    message = "Speech completed" if success else "Speech failed"

                self.progress_updated.emit(100)
                self.synthesis_finished.emit(success, message)
            else:
                self.synthesis_finished.emit(False, "TTS not available")

        except Exception as e:
            logger.error(f"TTS worker error: {e}")
            self.synthesis_finished.emit(False, f"Error: {e}")


class EnhancedNeuralTTSTab(QWidget):
    """
    Enhanced Neural TTS tab with comprehensive controls and dev mode support.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_dev_config()
        self.tts_integration = None
        self.current_worker = None

        self._init_ui()
        self._init_tts()
        self._setup_timers()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Development Mode Panel
        self.dev_panel = DevModeControlPanel("neural_tts", self)
        layout.addWidget(self.dev_panel)

        # Main content splitter
        splitter = QSplitter()
        layout.addWidget(splitter)

        # Left panel - Controls
        controls_widget = self._create_controls_panel()
        splitter.addWidget(controls_widget)

        # Right panel - Monitoring (dev mode)
        monitoring_widget = self._create_monitoring_panel()
        splitter.addWidget(monitoring_widget)

        # Set initial splitter ratio
        splitter.setSizes([300, 200])

        # Status bar
        self.status_label = QLabel("Neural TTS Ready")
        layout.addWidget(self.status_label)

    def _create_controls_panel(self) -> QWidget:
        """Create the main controls panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # TTS Engine Selection
        engine_group = QGroupBox("🎯 TTS Engine Configuration")
        engine_layout = QGridLayout(engine_group)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["auto", "speecht5", "pyttsx3"])
        engine_layout.addWidget(QLabel("Engine:"), 0, 0)
        engine_layout.addWidget(self.engine_combo, 0, 1)

        self.engine_status = QLabel("Checking...")
        engine_layout.addWidget(QLabel("Status:"), 1, 0)
        engine_layout.addWidget(self.engine_status, 1, 1)

        layout.addWidget(engine_group)

        # Agent Selection
        agent_group = QGroupBox("🎭 Agent Voice Selection")
        agent_layout = QGridLayout(agent_group)

        self.agent_combo = QComboBox()
        self.agent_combo.addItems(["Nova", "Aria", "Kai", "Echo", "Sage"])
        agent_layout.addWidget(QLabel("Agent:"), 0, 0)
        agent_layout.addWidget(self.agent_combo, 0, 1)

        self.voice_info_label = QLabel("Select an agent")
        agent_layout.addWidget(QLabel("Voice Info:"), 1, 0)
        agent_layout.addWidget(self.voice_info_label, 1, 1)

        layout.addWidget(agent_group)

        # Voice Parameters (dev mode)
        self.params_group = QGroupBox("🔧 Voice Parameters")
        params_layout = QGridLayout(self.params_group)

        # Speed control
        self.speed_slider = QSlider()
        self.speed_slider.setOrientation(1)  # Horizontal
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_label = QLabel("1.0x")
        params_layout.addWidget(QLabel("Speed:"), 0, 0)
        params_layout.addWidget(self.speed_slider, 0, 1)
        params_layout.addWidget(self.speed_label, 0, 2)

        # Energy control
        self.energy_slider = QSlider()
        self.energy_slider.setOrientation(1)  # Horizontal
        self.energy_slider.setRange(50, 150)
        self.energy_slider.setValue(100)
        self.energy_label = QLabel("1.0x")
        params_layout.addWidget(QLabel("Energy:"), 1, 0)
        params_layout.addWidget(self.energy_slider, 1, 1)
        params_layout.addWidget(self.energy_label, 1, 2)

        # Pitch control
        self.pitch_slider = QSlider()
        self.pitch_slider.setOrientation(1)  # Horizontal
        self.pitch_slider.setRange(-50, 50)
        self.pitch_slider.setValue(0)
        self.pitch_label = QLabel("0.0")
        params_layout.addWidget(QLabel("Pitch:"), 2, 0)
        params_layout.addWidget(self.pitch_slider, 2, 1)
        params_layout.addWidget(self.pitch_label, 2, 2)

        layout.addWidget(self.params_group)

        # Text Input
        text_group = QGroupBox("📝 Text Input")
        text_layout = QVBoxLayout(text_group)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to synthesize...")
        self.text_input.setMaximumHeight(100)
        text_layout.addWidget(self.text_input)

        # Quick text buttons
        quick_buttons = QHBoxLayout()
        self.greeting_btn = QPushButton("👋 Greeting")
        self.test_btn = QPushButton("🧪 Test Text")
        self.custom_btn = QPushButton("✏️ Custom")

        quick_buttons.addWidget(self.greeting_btn)
        quick_buttons.addWidget(self.test_btn)
        quick_buttons.addWidget(self.custom_btn)

        text_layout.addLayout(quick_buttons)
        layout.addWidget(text_group)

        # Action Buttons
        actions_group = QGroupBox("🎬 Actions")
        actions_layout = QGridLayout(actions_group)

        self.speak_btn = QPushButton("🔊 Speak")
        self.speak_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        self.save_audio_btn = QPushButton("💾 Save Audio")
        self.stop_btn = QPushButton("⏹️ Stop")
        self.test_all_btn = QPushButton("🎭 Test All Agents")

        actions_layout.addWidget(self.speak_btn, 0, 0)
        actions_layout.addWidget(self.save_audio_btn, 0, 1)
        actions_layout.addWidget(self.stop_btn, 1, 0)
        actions_layout.addWidget(self.test_all_btn, 1, 1)

        layout.addWidget(actions_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        return widget

    def _create_monitoring_panel(self) -> QWidget:
        """Create the monitoring panel for dev mode."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Monitoring tabs
        monitor_tabs = QTabWidget()

        # Engine Status Tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)

        self.engine_metrics = QTextEdit()
        self.engine_metrics.setMaximumHeight(150)
        self.engine_metrics.setReadOnly(True)
        self.engine_metrics.setStyleSheet(
            "background: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace;"
        )
        status_layout.addWidget(QLabel("Engine Metrics:"))
        status_layout.addWidget(self.engine_metrics)

        monitor_tabs.addTab(status_tab, "Engine")

        # Voice Analysis Tab
        voice_tab = QWidget()
        voice_layout = QVBoxLayout(voice_tab)

        self.voice_analysis = QTextEdit()
        self.voice_analysis.setMaximumHeight(150)
        self.voice_analysis.setReadOnly(True)
        self.voice_analysis.setStyleSheet(
            "background: #1e1e1e; color: #00aaff; font-family: 'Courier New', monospace;"
        )
        voice_layout.addWidget(QLabel("Voice Analysis:"))
        voice_layout.addWidget(self.voice_analysis)

        monitor_tabs.addTab(voice_tab, "Voice")

        # Performance Tab
        perf_tab = QWidget()
        perf_layout = QVBoxLayout(perf_tab)

        self.performance_metrics = QTextEdit()
        self.performance_metrics.setMaximumHeight(150)
        self.performance_metrics.setReadOnly(True)
        self.performance_metrics.setStyleSheet(
            "background: #1e1e1e; color: #ffaa00; font-family: 'Courier New', monospace;"
        )
        perf_layout.addWidget(QLabel("Performance Metrics:"))
        perf_layout.addWidget(self.performance_metrics)

        monitor_tabs.addTab(perf_tab, "Performance")

        layout.addWidget(monitor_tabs)

        # Dev mode controls
        dev_controls = QGroupBox("Development Controls")
        dev_layout = QGridLayout(dev_controls)

        self.debug_mode_cb = QCheckBox("Debug Mode")
        self.profile_cb = QCheckBox("Profile Performance")
        self.verbose_cb = QCheckBox("Verbose Logging")

        dev_layout.addWidget(self.debug_mode_cb, 0, 0)
        dev_layout.addWidget(self.profile_cb, 0, 1)
        dev_layout.addWidget(self.verbose_cb, 1, 0)

        layout.addWidget(dev_controls)

        return widget

    def _init_tts(self):
        """Initialize TTS integration."""
        if TTS_AVAILABLE:
            try:
                self.tts_integration = get_tts_integration()
                if self.tts_integration.is_available():
                    self.engine_status.setText("✅ Ready")
                    engines = self.tts_integration.neural_tts.get_available_engines()
                    self.engine_combo.clear()
                    self.engine_combo.addItems(["auto"] + engines)
                else:
                    self.engine_status.setText("❌ Not Available")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.engine_status.setText(f"❌ Error: {e}")
        else:
            self.engine_status.setText("❌ TTS Not Installed")

    def _setup_timers(self):
        """Setup update timers."""
        # Metrics update timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_metrics)

        # Start timer if auto-refresh is enabled
        tab_config = self.config.get_tab_config("neural_tts")
        if tab_config.auto_refresh:
            self.metrics_timer.start(tab_config.refresh_interval)

    def _connect_signals(self):
        """Connect all signals and slots."""
        # Agent selection
        self.agent_combo.currentTextChanged.connect(self._on_agent_changed)

        # Voice parameters
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.energy_slider.valueChanged.connect(self._on_energy_changed)
        self.pitch_slider.valueChanged.connect(self._on_pitch_changed)

        # Quick text buttons
        self.greeting_btn.clicked.connect(self._insert_greeting_text)
        self.test_btn.clicked.connect(self._insert_test_text)

        # Action buttons
        self.speak_btn.clicked.connect(self._on_speak_clicked)
        self.save_audio_btn.clicked.connect(self._on_save_audio_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.test_all_btn.clicked.connect(self._on_test_all_clicked)

        # Dev panel signals
        self.dev_panel.config_changed.connect(self._on_dev_config_changed)
        self.dev_panel.dev_mode_toggled.connect(self._on_dev_mode_toggled)

    @pyqtSlot(str)
    def _on_agent_changed(self, agent_name: str):
        """Handle agent selection change."""
        if self.tts_integration and agent_name:
            voice_info = self.tts_integration.get_agent_voice_info(agent_name)
            if voice_info:
                info_text = f"{voice_info['speaking_style']} {voice_info['gender']}"
                self.voice_info_label.setText(info_text)

                # Update sliders based on voice profile
                self.speed_slider.setValue(int(voice_info["speed"] * 100))
                self.energy_slider.setValue(int(voice_info["energy"] * 100))
                self.pitch_slider.setValue(int(voice_info["pitch"] * 100))

    def _on_speed_changed(self, value: int):
        """Handle speed slider change."""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.1f}x")

    def _on_energy_changed(self, value: int):
        """Handle energy slider change."""
        energy = value / 100.0
        self.energy_label.setText(f"{energy:.1f}x")

    def _on_pitch_changed(self, value: int):
        """Handle pitch slider change."""
        pitch = value / 100.0
        self.pitch_label.setText(f"{pitch:.1f}")

    def _insert_greeting_text(self):
        """Insert agent greeting text."""
        agent = self.agent_combo.currentText()
        greeting = f"Hello! I'm {agent}, your AI assistant. How can I help you today?"
        self.text_input.setText(greeting)

    def _insert_test_text(self):
        """Insert test text."""
        test_text = "This is a test of the VoxSigil neural text-to-speech system. The quick brown fox jumps over the lazy dog."
        self.text_input.setText(test_text)

    def _on_speak_clicked(self):
        """Handle speak button click."""
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_label.setText("Please enter text to speak")
            return

        agent = self.agent_combo.currentText()
        self._start_tts_worker(text, agent)

    def _on_save_audio_clicked(self):
        """Handle save audio button click."""
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_label.setText("Please enter text to save")
            return

        agent = self.agent_combo.currentText()
        output_path = f"tts_output_{agent.lower()}.wav"
        self._start_tts_worker(text, agent, output_path)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker = None
            self.progress_bar.setVisible(False)
            self.status_label.setText("Stopped")

    def _on_test_all_clicked(self):
        """Handle test all agents button click."""
        # This would cycle through all agents
        self.status_label.setText("Testing all agents...")
        # Implementation would test each agent voice

    def _start_tts_worker(self, text: str, agent: str, output_path: Optional[str] = None):
        """Start TTS worker thread."""
        if self.current_worker and self.current_worker.isRunning():
            return

        self.current_worker = TTSWorker(text, agent, output_path)
        self.current_worker.synthesis_started.connect(self._on_synthesis_started)
        self.current_worker.synthesis_finished.connect(self._on_synthesis_finished)
        self.current_worker.progress_updated.connect(self._on_progress_updated)
        self.current_worker.start()

    @pyqtSlot()
    def _on_synthesis_started(self):
        """Handle synthesis start."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.speak_btn.setEnabled(False)
        self.save_audio_btn.setEnabled(False)
        self.status_label.setText("Synthesizing...")

    @pyqtSlot(bool, str)
    def _on_synthesis_finished(self, success: bool, message: str):
        """Handle synthesis completion."""
        self.progress_bar.setVisible(False)
        self.speak_btn.setEnabled(True)
        self.save_audio_btn.setEnabled(True)
        self.status_label.setText(message)

        if success:
            self._log_to_dev_panel(f"Synthesis successful: {message}")
        else:
            self._log_to_dev_panel(f"Synthesis failed: {message}")

    @pyqtSlot(int)
    def _on_progress_updated(self, progress: int):
        """Handle progress update."""
        self.progress_bar.setValue(progress)

    def _on_dev_config_changed(self, setting_name: str, value: Any):
        """Handle dev configuration changes."""
        if setting_name == "auto_refresh":
            if value:
                tab_config = self.config.get_tab_config("neural_tts")
                self.metrics_timer.start(tab_config.refresh_interval)
            else:
                self.metrics_timer.stop()
        elif setting_name == "refresh_interval":
            if self.metrics_timer.isActive():
                self.metrics_timer.start(value)

    def _on_dev_mode_toggled(self, enabled: bool):
        """Handle dev mode toggle."""
        # Show/hide advanced controls
        self.params_group.setVisible(enabled)

        # Update monitoring visibility
        monitoring_widget = self.findChild(QWidget)  # Find monitoring panel
        if monitoring_widget:
            monitoring_widget.setVisible(enabled)

    def _update_metrics(self):
        """Update monitoring metrics."""
        if not self.dev_panel.is_dev_mode_enabled():
            return

        try:
            # Update engine metrics
            if self.tts_integration:
                engines = self.tts_integration.neural_tts.get_available_engines()
                metrics = f"Available Engines: {engines}\\n"
                metrics += f"Active Engine: {self.engine_combo.currentText()}\\n"
                metrics += f"Voice Profiles: {len(self.tts_integration.get_available_voices())}\\n"
                self.engine_metrics.setText(metrics)

            # Update voice analysis
            agent = self.agent_combo.currentText()
            if self.tts_integration and agent:
                voice_info = self.tts_integration.get_agent_voice_info(agent)
                if voice_info:
                    analysis = f"Agent: {agent}\\n"
                    analysis += f"Style: {voice_info['speaking_style']}\\n"
                    analysis += f"Gender: {voice_info['gender']}\\n"
                    analysis += f"Speed: {voice_info['speed']}x\\n"
                    analysis += f"Energy: {voice_info['energy']}x\\n"
                    self.voice_analysis.setText(analysis)

            # Update performance metrics
            import psutil

            perf = f"CPU Usage: {psutil.cpu_percent():.1f}%\\n"
            perf += f"Memory Usage: {psutil.virtual_memory().percent:.1f}%\\n"
            perf += f"Active Threads: {len(psutil.Process().threads())}\\n"
            self.performance_metrics.setText(perf)

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def _log_to_dev_panel(self, message: str):
        """Log message to dev panel."""
        self.dev_panel.log_message(message)
