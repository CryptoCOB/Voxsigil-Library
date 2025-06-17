#!/usr/bin/env python3
"""
Music Equalizer - Real-time audio spectrum visualization

Provides a live equalizer display showing:
- 32-band frequency spectrum from audio output
- Peak hold indicators
- Voice-only overlay (blue)
- Foley/ambience overlay (orange)

Visual style: Dark background with neon bars, VoxSigil color palette
"""

import logging
from typing import Any, Dict, List

import numpy as np

try:
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtWidgets

    HAVE_PYQTGRAPH = True
except ImportError:
    HAVE_PYQTGRAPH = False

    # Fallback for testing
    class QtWidgets:
        class QWidget:
            pass

        class QVBoxLayout:
            pass

        class QHBoxLayout:
            pass

        class QLabel:
            pass

        class QPushButton:
            pass

        class QSlider:
            pass


from core.base_core import BaseCore
from Vanta.core.vanta_registration import vanta_core_module
from Vanta.interfaces.cognitive_mesh import CognitiveMeshRole

logger = logging.getLogger(__name__)


@vanta_core_module(
    name="music_equalizer",
    subsystem="ui",
    mesh_role=CognitiveMeshRole.MONITOR,
    description="Real-time audio spectrum equalizer visualization",
    capabilities=["audio_visualization", "spectrum_analysis", "real_time_display"],
)
class MusicEqualizer(BaseCore, QtWidgets.QWidget):
    """
    Real-time music equalizer with spectrum visualization.

    Shows live audio spectrum in equalizer bar format:
    - 32-band frequency analysis
    - Peak hold indicators
    - Voice/music/foley overlays
    - Color-coded intensity (green->yellow->red)
    """

    def __init__(self, bus=None, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        BaseCore.__init__(self)

        self.bus = bus
        self.logger = logging.getLogger("VoxSigil.GUI.Equalizer")

        # Equalizer settings
        self.num_bands = 32
        self.peak_decay_rate = 0.95  # How fast peak indicators fall
        self.update_rate = 60  # Update every 60ms for smooth animation

        # Data storage
        self.frequency_data = np.zeros(self.num_bands)
        self.peak_data = np.zeros(self.num_bands)
        self.voice_overlay = np.zeros(self.num_bands)
        self.foley_overlay = np.zeros(self.num_bands)

        # Display options
        self.show_voice_overlay = True
        self.show_foley_overlay = True
        self.freeze_display = False
        self.colorblind_mode = False

        # Threading
        self.update_timer = None

        self._init_ui()

        # Subscribe to events
        if self.bus:
            self.bus.subscribe("audio.eq", self.update_spectrum)
            self.bus.subscribe("audio.voice", self.update_voice_overlay)
            self.bus.subscribe("audio.foley", self.update_foley_overlay)

        # Start update timer
        self._start_update_timer()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px 10px;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #0066cc;
            }
            QSlider::groove:horizontal {
                background: #333333;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ff00;
                width: 12px;
                border-radius: 6px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title and controls
        header_layout = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel("Music Equalizer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Control buttons
        self.voice_button = QtWidgets.QPushButton("Voice")
        self.voice_button.setCheckable(True)
        self.voice_button.setChecked(True)
        self.voice_button.clicked.connect(self.toggle_voice_overlay)
        header_layout.addWidget(self.voice_button)

        self.foley_button = QtWidgets.QPushButton("Foley")
        self.foley_button.setCheckable(True)
        self.foley_button.setChecked(True)
        self.foley_button.clicked.connect(self.toggle_foley_overlay)
        header_layout.addWidget(self.foley_button)

        self.freeze_button = QtWidgets.QPushButton("Freeze")
        self.freeze_button.setCheckable(True)
        self.freeze_button.clicked.connect(self.toggle_freeze)
        header_layout.addWidget(self.freeze_button)

        layout.addLayout(header_layout)

        if not HAVE_PYQTGRAPH:
            # Fallback display
            fallback = QtWidgets.QLabel(
                "PyQtGraph not available\nInstall with: pip install pyqtgraph"
            )
            fallback.setStyleSheet("color: #ff6666; font-size: 14px;")
            layout.addWidget(fallback)
            return

        # Create the equalizer plot
        self.plot = pg.PlotWidget(background="#000000")
        self.plot.setLabel("left", "Amplitude", units="dB")
        self.plot.setLabel("bottom", "Frequency Band")
        self.plot.setYRange(0, 100)
        self.plot.setXRange(0, self.num_bands)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)

        # Create bar graph items
        self._create_bar_items()

        layout.addWidget(self.plot)

        # Peak decay control
        control_layout = QtWidgets.QHBoxLayout()

        control_layout.addWidget(QtWidgets.QLabel("Peak Decay:"))

        self.decay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.decay_slider.setMinimum(85)
        self.decay_slider.setMaximum(99)
        self.decay_slider.setValue(int(self.peak_decay_rate * 100))
        self.decay_slider.valueChanged.connect(self.on_decay_changed)
        control_layout.addWidget(self.decay_slider)

        self.decay_label = QtWidgets.QLabel(f"{self.peak_decay_rate:.2f}")
        control_layout.addWidget(self.decay_label)

        control_layout.addStretch()

        # RMS level display
        self.rms_label = QtWidgets.QLabel("RMS: 0.0 dB")
        self.rms_label.setStyleSheet("color: #00ff00; font-family: monospace;")
        control_layout.addWidget(self.rms_label)

        layout.addLayout(control_layout)

        self.logger.info("Music equalizer UI initialized")

    def _create_bar_items(self):
        """Create bar graph items for the equalizer."""
        if not HAVE_PYQTGRAPH:
            return

        # Main frequency bars
        self.bars = pg.BarGraphItem(
            x=np.arange(self.num_bands),
            height=self.frequency_data,
            width=0.8,
            brush=pg.mkBrush("#44ff44"),
        )
        self.plot.addItem(self.bars)

        # Peak hold indicators
        self.peak_bars = pg.BarGraphItem(
            x=np.arange(self.num_bands),
            height=np.zeros(self.num_bands),
            width=0.8,
            brush=pg.mkBrush("#ffffff"),
            pen=pg.mkPen("#ffffff", width=1),
        )
        self.plot.addItem(self.peak_bars)

        # Voice overlay
        self.voice_bars = pg.PlotCurveItem(
            x=np.arange(self.num_bands), y=self.voice_overlay, pen=pg.mkPen("#4299e1", width=2)
        )
        self.plot.addItem(self.voice_bars)

        # Foley overlay
        self.foley_bars = pg.PlotCurveItem(
            x=np.arange(self.num_bands), y=self.foley_overlay, pen=pg.mkPen("#ffa500", width=2)
        )
        self.plot.addItem(self.foley_bars)

    def _start_update_timer(self):
        """Start the update timer for smooth animation."""
        if HAVE_PYQTGRAPH:
            self.update_timer = QtCore.QTimer()
            self.update_timer.timeout.connect(self._update_display)
            self.update_timer.start(self.update_rate)

    async def initialize_subsystem(self, core):
        """Initialize the equalizer subsystem."""
        self.core = core
        self.logger.info("Music equalizer subsystem initialized")

    def update_spectrum(self, payload: Dict[str, Any]):
        """Update the spectrum data from audio analysis."""
        if self.freeze_display:
            return

        try:
            bins = payload.get("bins", [])
            if len(bins) != self.num_bands:
                # Resample to fit our band count
                bins = np.interp(
                    np.linspace(0, len(bins) - 1, self.num_bands), np.arange(len(bins)), bins
                )

            # Convert to dB scale
            self.frequency_data = 20 * np.log10(np.maximum(np.array(bins), 1e-10))

            # Update peaks
            self.peak_data = np.maximum(self.peak_data * self.peak_decay_rate, self.frequency_data)

            # Calculate RMS level
            rms = np.sqrt(np.mean(np.array(bins) ** 2))
            rms_db = 20 * np.log10(max(rms, 1e-10))

            # Update RMS display
            if hasattr(self, "rms_label"):
                self.rms_label.setText(f"RMS: {rms_db:.1f} dB")

            self.logger.debug(f"Updated spectrum: RMS={rms_db:.1f}dB")

        except Exception as e:
            self.logger.error(f"Error updating spectrum: {e}")

    def update_voice_overlay(self, payload: Dict[str, Any]):
        """Update voice-only overlay data."""
        try:
            bins = payload.get("bins", [])
            if len(bins) != self.num_bands:
                bins = np.interp(
                    np.linspace(0, len(bins) - 1, self.num_bands), np.arange(len(bins)), bins
                )

            self.voice_overlay = 20 * np.log10(np.maximum(np.array(bins), 1e-10))

        except Exception as e:
            self.logger.error(f"Error updating voice overlay: {e}")

    def update_foley_overlay(self, payload: Dict[str, Any]):
        """Update foley/ambience overlay data."""
        try:
            bins = payload.get("bins", [])
            if len(bins) != self.num_bands:
                bins = np.interp(
                    np.linspace(0, len(bins) - 1, self.num_bands), np.arange(len(bins)), bins
                )

            self.foley_overlay = 20 * np.log10(np.maximum(np.array(bins), 1e-10))

        except Exception as e:
            self.logger.error(f"Error updating foley overlay: {e}")

    def _update_display(self):
        """Update the visual display (called by timer)."""
        if not HAVE_PYQTGRAPH or self.freeze_display:
            return

        try:
            # Update main bars with color coding
            colors = self._get_bar_colors(self.frequency_data)
            self.bars.setOpts(
                height=np.maximum(self.frequency_data, 0),
                brush=[pg.mkBrush(color) for color in colors],
            )

            # Update peak indicators
            self.peak_bars.setOpts(height=np.maximum(self.peak_data, 0))

            # Update overlays if enabled
            if self.show_voice_overlay:
                self.voice_bars.setData(
                    x=np.arange(self.num_bands), y=np.maximum(self.voice_overlay, 0)
                )
            else:
                self.voice_bars.setData(x=[], y=[])

            if self.show_foley_overlay:
                self.foley_bars.setData(
                    x=np.arange(self.num_bands), y=np.maximum(self.foley_overlay, 0)
                )
            else:
                self.foley_bars.setData(x=[], y=[])

        except Exception as e:
            self.logger.error(f"Error updating display: {e}")

    def _get_bar_colors(self, data: np.ndarray) -> List[str]:
        """Get colors for bars based on amplitude."""
        colors = []
        for value in data:
            if self.colorblind_mode:
                colors.append("#ffffff")
            elif value > 80:
                colors.append("#ff4444")  # Red for high
            elif value > 60:
                colors.append("#ffff44")  # Yellow for medium-high
            elif value > 40:
                colors.append("#ffa500")  # Orange for medium
            else:
                colors.append("#44ff44")  # Green for low
        return colors

    def toggle_voice_overlay(self):
        """Toggle voice overlay display."""
        self.show_voice_overlay = self.voice_button.isChecked()
        self.logger.info(f"Voice overlay: {'enabled' if self.show_voice_overlay else 'disabled'}")

    def toggle_foley_overlay(self):
        """Toggle foley overlay display."""
        self.show_foley_overlay = self.foley_button.isChecked()
        self.logger.info(f"Foley overlay: {'enabled' if self.show_foley_overlay else 'disabled'}")

    def toggle_freeze(self):
        """Toggle freeze frame."""
        self.freeze_display = self.freeze_button.isChecked()
        self.logger.info(f"Freeze frame: {'enabled' if self.freeze_display else 'disabled'}")

    def on_decay_changed(self, value: int):
        """Handle peak decay rate change."""
        self.peak_decay_rate = value / 100.0
        self.decay_label.setText(f"{self.peak_decay_rate:.2f}")

    def get_ui_spec(self) -> Dict[str, Any]:
        """Get UI specification for bridge registration."""
        return {
            "tab": "Music Equalizer",
            "widget": "MusicEqualizer",
            "stream": True,
            "stream_topic": "audio.eq",
            "priority": 8,
        }


def test_music_equalizer():
    """Test the music equalizer functionality."""
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)  # noqa: F841

    # Create mock bus
    class MockBus:
        def subscribe(self, topic, callback):
            pass

    equalizer = MusicEqualizer(bus=MockBus())
    equalizer.show()

    # Simulate spectrum data
    test_spectrum = {
        "bins": np.random.rand(32) * 0.5 + 0.1  # Random spectrum
    }
    equalizer.update_spectrum(test_spectrum)

    print("Music equalizer test completed!")
    return equalizer


if __name__ == "__main__":
    test_music_equalizer()
