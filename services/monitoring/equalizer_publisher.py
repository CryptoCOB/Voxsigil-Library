#!/usr/bin/env python3
"""
Equalizer Publisher - Generates audio spectrum data for the music equalizer

Processes audio streams and publishes FFT spectrum analysis:
- 32-band frequency analysis
- Peak hold data
- Voice-only separation
- Foley/ambience separation
"""

import asyncio
import logging
import time
from typing import Any, Dict

import numpy as np

from agents.base import BaseAgent
from Vanta.core.vanta_registration import vanta_core_module
from Vanta.interfaces.cognitive_mesh import CognitiveMeshRole

logger = logging.getLogger(__name__)


@vanta_core_module(
    name="equalizer_publisher",
    subsystem="monitoring",
    mesh_role=CognitiveMeshRole.PUBLISHER,
    description="Publishes real-time audio spectrum data for equalizer visualization",
    capabilities=["audio_analysis", "spectrum_analysis", "real_time_fft"],
)
class EqualizerPublisher(BaseAgent):
    """
    Publisher agent that generates audio spectrum data for the music equalizer.

    Processes audio streams and publishes:
    - 32-band frequency spectrum
    - Peak hold indicators
    - Voice-only overlay data
    - Foley/ambience overlay data
    """

    def __init__(self, bus=None, sample_rate: int = 44100, frame_size: int = 1024):
        super().__init__(bus=bus)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.logger = logging.getLogger("VoxSigil.Monitoring.Equalizer")

        # Spectrum analysis settings
        self.num_bands = 32
        self.update_interval = 0.06  # 60ms updates for smooth animation
        self.window = np.hanning(frame_size)

        # Data buffers
        self.audio_buffer = np.zeros(frame_size)
        self.voice_buffer = np.zeros(frame_size)
        self.foley_buffer = np.zeros(frame_size)

        # Peak tracking
        self.peak_data = np.zeros(self.num_bands)
        self.peak_decay = 0.95

        self.running = False

        # Subscribe to audio streams
        if self.bus:
            self.bus.subscribe("audio.pcm", self.on_audio_frame)
            self.bus.subscribe("audio.voice", self.on_voice_frame)
            self.bus.subscribe("audio.foley", self.on_foley_frame)

    async def initialize_agent(self):
        """Initialize the equalizer publisher."""
        self.logger.info("Equalizer publisher initialized")
        self.running = True

        # Start the analysis loop
        asyncio.create_task(self.analysis_loop())

    async def analysis_loop(self):
        """Main analysis and publishing loop."""
        while self.running:
            try:
                await self.analyze_and_publish()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in equalizer analysis loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def analyze_and_publish(self):
        """Analyze audio data and publish spectrum."""
        try:
            # Perform FFT analysis on main audio
            spectrum = self._analyze_spectrum(self.audio_buffer)

            # Analyze overlays
            voice_spectrum = self._analyze_spectrum(self.voice_buffer)
            foley_spectrum = self._analyze_spectrum(self.foley_buffer)

            # Update peaks
            self.peak_data = np.maximum(self.peak_data * self.peak_decay, spectrum)

            # Publish main spectrum
            if self.bus:
                self.bus.publish(
                    "audio.eq",
                    {
                        "bins": spectrum.tolist(),
                        "peak": self.peak_data.tolist(),
                        "timestamp": time.time(),
                    },
                )

                # Publish overlays
                if np.any(voice_spectrum > 0):
                    self.bus.publish("audio.voice", {"bins": voice_spectrum.tolist()})

                if np.any(foley_spectrum > 0):
                    self.bus.publish("audio.foley", {"bins": foley_spectrum.tolist()})

            # Calculate RMS for monitoring
            rms = np.sqrt(np.mean(spectrum**2))

            self.logger.debug(f"Published spectrum: RMS={rms:.3f}, Bands={len(spectrum)}")

        except Exception as e:
            self.logger.error(f"Error analyzing audio spectrum: {e}")

    def _analyze_spectrum(self, audio_data: np.ndarray) -> np.ndarray:
        """Analyze audio data and return frequency spectrum."""
        if len(audio_data) == 0 or np.all(audio_data == 0):
            return np.zeros(self.num_bands)

        try:
            # Apply window and perform FFT
            windowed = audio_data * self.window
            fft = np.abs(np.fft.rfft(windowed))

            # Convert to log-spaced frequency bands
            freqs = np.fft.rfftfreq(self.frame_size, 1 / self.sample_rate)

            # Create logarithmic frequency bins
            log_freqs = np.logspace(
                np.log10(freqs[1]),  # Start from first non-DC bin
                np.log10(freqs[-1]),  # End at Nyquist
                self.num_bands + 1,
            )

            # Bin the FFT data into logarithmic bands
            spectrum = np.zeros(self.num_bands)
            for i in range(self.num_bands):
                freq_mask = (freqs >= log_freqs[i]) & (freqs < log_freqs[i + 1])
                if np.any(freq_mask):
                    spectrum[i] = np.mean(fft[freq_mask])

            # Normalize to 0-1 range
            if np.max(spectrum) > 0:
                spectrum = spectrum / np.max(spectrum)

            return spectrum

        except Exception as e:
            self.logger.error(f"Error in spectrum analysis: {e}")
            return np.zeros(self.num_bands)

    def on_audio_frame(self, payload: Dict[str, Any]):
        """Handle incoming audio frame data."""
        try:
            audio_data = payload.get("data", [])
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)

            # Update buffer with latest frame
            if len(audio_data) == self.frame_size:
                self.audio_buffer = audio_data
            else:
                # Pad or truncate to frame size
                if len(audio_data) > self.frame_size:
                    self.audio_buffer = audio_data[: self.frame_size]
                else:
                    self.audio_buffer = np.pad(audio_data, (0, self.frame_size - len(audio_data)))

        except Exception as e:
            self.logger.error(f"Error processing audio frame: {e}")

    def on_voice_frame(self, payload: Dict[str, Any]):
        """Handle incoming voice-only audio frame."""
        try:
            audio_data = payload.get("data", [])
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)

            if len(audio_data) == self.frame_size:
                self.voice_buffer = audio_data
            else:
                if len(audio_data) > self.frame_size:
                    self.voice_buffer = audio_data[: self.frame_size]
                else:
                    self.voice_buffer = np.pad(audio_data, (0, self.frame_size - len(audio_data)))

        except Exception as e:
            self.logger.error(f"Error processing voice frame: {e}")

    def on_foley_frame(self, payload: Dict[str, Any]):
        """Handle incoming foley/ambience audio frame."""
        try:
            audio_data = payload.get("data", [])
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)

            if len(audio_data) == self.frame_size:
                self.foley_buffer = audio_data
            else:
                if len(audio_data) > self.frame_size:
                    self.foley_buffer = audio_data[: self.frame_size]
                else:
                    self.foley_buffer = np.pad(audio_data, (0, self.frame_size - len(audio_data)))

        except Exception as e:
            self.logger.error(f"Error processing foley frame: {e}")

    def generate_test_spectrum(self, frequencies: list = None) -> None:
        """Generate test spectrum data for development/testing."""
        if frequencies is None:
            frequencies = [440, 880, 1760]  # A4, A5, A6

        # Generate test audio with specified frequencies
        t = np.linspace(0, self.frame_size / self.sample_rate, self.frame_size)
        test_audio = np.zeros(self.frame_size)

        for freq in frequencies:
            test_audio += 0.3 * np.sin(2 * np.pi * freq * t)

        # Add some noise
        test_audio += 0.1 * np.random.normal(0, 1, self.frame_size)

        # Process as if it were real audio
        self.audio_buffer = test_audio

        self.logger.info(f"Generated test spectrum with frequencies: {frequencies}")

    async def shutdown(self):
        """Shutdown the publisher."""
        self.running = False
        self.logger.info("Equalizer publisher shutdown")

    def get_ui_spec(self) -> Dict[str, Any]:
        """Get UI specification."""
        return {
            "name": "EqualizerPublisher",
            "type": "publisher",
            "publishes": ["audio.eq", "audio.voice", "audio.foley"],
            "enabled": True,
        }


# Singleton instance for global access
_equalizer_publisher = None


def get_equalizer_publisher(bus=None) -> EqualizerPublisher:
    """Get or create the global equalizer publisher instance."""
    global _equalizer_publisher
    if _equalizer_publisher is None:
        _equalizer_publisher = EqualizerPublisher(bus=bus)
    return _equalizer_publisher


async def start_audio_monitoring(bus=None, sample_rate: int = 44100):
    """Start audio spectrum monitoring."""
    publisher = get_equalizer_publisher(bus)
    await publisher.initialize_agent()

    # Generate some test data for development
    publisher.generate_test_spectrum([220, 440, 880, 1760])  # Multiple test tones

    logger.info(f"Started audio spectrum monitoring at {sample_rate}Hz")
    return publisher
