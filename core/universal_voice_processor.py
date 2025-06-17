#!/usr/bin/env python3
"""
Universal Voice Fingerprinting and Noise Cancellation System
============================================================

Advanced audio processing for all VoxSigil agents including:
1. Voice fingerprinting and isolation
2. Real-time noise cancellation
3. Multi-agent voice separation
4. Environmental audio adaptation
5. Music agent audio enhancement
6. Cross-agent voice harmonization

This system is designed to be woven into all agents for unified audio processing.
"""

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of noise that can be detected and cancelled"""

    AMBIENT = "ambient"  # Background hum, AC, traffic
    SPEECH = "speech"  # Other people talking
    MUSIC = "music"  # Background music
    ELECTRONIC = "electronic"  # Computer fans, beeps
    MECHANICAL = "mechanical"  # Machinery, motors
    WIND = "wind"  # Wind noise
    CROWD = "crowd"  # Multiple people, crowds
    UNKNOWN = "unknown"  # Unclassified noise


class VoiceCharacteristic(Enum):
    """Voice characteristics for fingerprinting"""

    FUNDAMENTAL_FREQ = "f0"
    FORMANTS = "formants"
    SPECTRAL_CENTROID = "spectral_centroid"
    PITCH_VARIANCE = "pitch_variance"
    VOCAL_TRACT_LENGTH = "vocal_tract_length"
    BREATHINESS = "breathiness"
    ROUGHNESS = "roughness"
    SHIMMER = "shimmer"
    JITTER = "jitter"


@dataclass
class VoiceFingerprint:
    """Unique voice fingerprint for each agent"""

    agent_id: str

    # Core voice characteristics
    fundamental_frequency: float = 0.0  # Hz
    formant_frequencies: List[float] = field(default_factory=list)  # F1, F2, F3, F4
    spectral_centroid: float = 0.0  # Hz
    pitch_variance: float = 0.0  # Standard deviation of F0

    # Voice quality measures
    vocal_tract_length: float = 17.5  # cm
    breathiness_index: float = 0.0  # 0.0-1.0
    roughness_index: float = 0.0  # 0.0-1.0
    shimmer: float = 0.0  # Amplitude variation
    jitter: float = 0.0  # Frequency variation

    # Unique signature
    voice_signature: List[float] = field(default_factory=list)  # 128-dim fingerprint
    confidence_score: float = 0.0  # 0.0-1.0

    # Environmental adaptation
    noise_tolerance: float = 0.5  # How well voice cuts through noise
    clarity_threshold: float = 0.7  # Minimum clarity for processing

    # Creation metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class NoiseProfile:
    """Profile of detected environmental noise"""

    noise_type: NoiseType
    intensity_db: float = 0.0  # Decibel level
    frequency_range: Tuple[float, float] = (0.0, 0.0)  # Hz min, max
    spectral_characteristics: List[float] = field(default_factory=list)
    confidence: float = 0.0  # Detection confidence

    # Temporal characteristics
    duration: float = 0.0  # seconds
    intermittent: bool = False  # Is it continuous or intermittent?
    pattern: Optional[str] = None  # "periodic", "random", "constant"

    # Cancellation parameters
    reduction_factor: float = 0.8  # How much to reduce (0.0-1.0)
    filter_coefficients: List[float] = field(default_factory=list)


class UniversalVoiceProcessor:
    """Universal voice processing system for all agents"""

    def __init__(self):
        self.agent_fingerprints: Dict[str, VoiceFingerprint] = {}
        self.noise_profiles: List[NoiseProfile] = []
        self.audio_buffer = deque(maxlen=100)  # Store recent audio for analysis
        self.processing_lock = threading.Lock()

        # Initialize processing parameters
        self.sample_rate = 16000  # Hz
        self.frame_size = 1024
        self.hop_length = 512

        # Noise cancellation settings
        self.noise_gate_threshold = -40  # dB
        self.noise_reduction_factor = 0.85
        self.voice_enhancement_factor = 1.2

        logger.info("Universal Voice Processor initialized")

    def create_agent_fingerprint(
        self, agent_id: str, agent_type: str = "standard", voice_sample: Optional[np.ndarray] = None
    ) -> VoiceFingerprint:
        """Create a unique voice fingerprint for an agent"""

        # Base characteristics by agent type
        base_characteristics = self._get_base_characteristics(agent_type)

        # Generate unique variations for this specific agent
        agent_variations = self._generate_agent_variations(agent_id)

        # Combine base + variations
        fingerprint = VoiceFingerprint(
            agent_id=agent_id,
            fundamental_frequency=base_characteristics["f0"] * agent_variations["f0_mult"],
            formant_frequencies=[
                base_characteristics["f1"] * agent_variations["formant_mult"],
                base_characteristics["f2"] * agent_variations["formant_mult"],
                base_characteristics["f3"] * agent_variations["formant_mult"],
                base_characteristics["f4"] * agent_variations["formant_mult"],
            ],
            spectral_centroid=base_characteristics["centroid"] * agent_variations["centroid_mult"],
            vocal_tract_length=base_characteristics["tract_length"]
            * agent_variations["tract_mult"],
            breathiness_index=base_characteristics["breathiness"]
            * agent_variations["breathiness_mult"],
            roughness_index=base_characteristics["roughness"] * agent_variations["roughness_mult"],
            noise_tolerance=base_characteristics["noise_tolerance"],
            clarity_threshold=base_characteristics["clarity_threshold"],
        )

        # Generate unique voice signature
        fingerprint.voice_signature = self._generate_voice_signature(fingerprint)
        fingerprint.confidence_score = 0.95  # High confidence for generated fingerprints

        # Store fingerprint
        self.agent_fingerprints[agent_id] = fingerprint

        logger.info(f"Created voice fingerprint for {agent_id} ({agent_type})")
        logger.info(f"  F0: {fingerprint.fundamental_frequency:.1f}Hz")
        logger.info(f"  Formants: {[f'{f:.0f}Hz' for f in fingerprint.formant_frequencies]}")
        logger.info(f"  Signature: {len(fingerprint.voice_signature)}-dim vector")

        return fingerprint

    def _get_base_characteristics(self, agent_type: str) -> Dict[str, float]:
        """Get base voice characteristics by agent type"""

        characteristics = {
            "standard": {
                "f0": 150.0,
                "f1": 500.0,
                "f2": 1500.0,
                "f3": 2500.0,
                "f4": 3500.0,
                "centroid": 2000.0,
                "tract_length": 17.5,
                "breathiness": 0.1,
                "roughness": 0.05,
                "noise_tolerance": 0.6,
                "clarity_threshold": 0.7,
            },
            "music": {
                "f0": 220.0,
                "f1": 550.0,
                "f2": 1650.0,
                "f3": 2800.0,
                "f4": 3800.0,
                "centroid": 2500.0,
                "tract_length": 16.0,
                "breathiness": 0.05,
                "roughness": 0.02,
                "noise_tolerance": 0.9,
                "clarity_threshold": 0.8,
            },
            "analytical": {
                "f0": 140.0,
                "f1": 480.0,
                "f2": 1400.0,
                "f3": 2400.0,
                "f4": 3400.0,
                "centroid": 1800.0,
                "tract_length": 18.0,
                "breathiness": 0.08,
                "roughness": 0.03,
                "noise_tolerance": 0.7,
                "clarity_threshold": 0.85,
            },
            "creative": {
                "f0": 180.0,
                "f1": 520.0,
                "f2": 1600.0,
                "f3": 2700.0,
                "f4": 3700.0,
                "centroid": 2200.0,
                "tract_length": 16.5,
                "breathiness": 0.15,
                "roughness": 0.08,
                "noise_tolerance": 0.5,
                "clarity_threshold": 0.65,
            },
            "authoritative": {
                "f0": 120.0,
                "f1": 450.0,
                "f2": 1300.0,
                "f3": 2200.0,
                "f4": 3200.0,
                "centroid": 1600.0,
                "tract_length": 19.0,
                "breathiness": 0.03,
                "roughness": 0.01,
                "noise_tolerance": 0.8,
                "clarity_threshold": 0.9,
            },
        }

        return characteristics.get(agent_type, characteristics["standard"])

    def _generate_agent_variations(self, agent_id: str) -> Dict[str, float]:
        """Generate unique variations for a specific agent"""

        # Use agent ID as seed for consistent variations
        np.random.seed(hash(agent_id) % (2**32))

        variations = {
            "f0_mult": np.random.normal(1.0, 0.1),  # ±10% F0 variation
            "formant_mult": np.random.normal(1.0, 0.05),  # ±5% formant variation
            "centroid_mult": np.random.normal(1.0, 0.08),  # ±8% centroid variation
            "tract_mult": np.random.normal(1.0, 0.03),  # ±3% tract length variation
            "breathiness_mult": np.random.normal(1.0, 0.2),  # ±20% breathiness variation
            "roughness_mult": np.random.normal(1.0, 0.25),  # ±25% roughness variation
        }

        # Reset random seed
        np.random.seed()

        return variations

    def _generate_voice_signature(self, fingerprint: VoiceFingerprint) -> List[float]:
        """Generate 128-dimensional voice signature vector"""

        signature = []

        # F0 and harmonics (32 dimensions)
        f0 = fingerprint.fundamental_frequency
        for i in range(32):
            harmonic = f0 * (i + 1)
            signature.append(math.sin(harmonic * 0.001) * math.exp(-i * 0.1))

        # Formant characteristics (32 dimensions)
        for formant in fingerprint.formant_frequencies:
            for i in range(8):
                signature.append(math.cos(formant * 0.0005 + i) * (1.0 / (i + 1)))

        # Spectral and quality features (32 dimensions)
        for i in range(32):
            feature = (
                fingerprint.spectral_centroid * 0.0001
                + fingerprint.breathiness_index * 10
                + fingerprint.roughness_index * 5
                + i
            )
            signature.append(math.tanh(feature))

        # Unique agent characteristics (32 dimensions)
        agent_hash = hash(fingerprint.agent_id) % (2**32)
        for i in range(32):
            signature.append(math.sin((agent_hash + i) * 0.00001))

        return signature

    async def detect_noise_environment(
        self, audio_data: Optional[np.ndarray] = None
    ) -> List[NoiseProfile]:
        """Detect and profile environmental noise"""

        if audio_data is None:
            # Simulate noise detection for now
            return self._simulate_noise_detection()

        # Real noise detection would analyze the audio_data
        # For now, return simulated profiles
        return self._simulate_noise_detection()

    def _simulate_noise_detection(self) -> List[NoiseProfile]:
        """Simulate noise detection for demonstration"""

        # Common noise scenarios
        scenarios = [
            {
                "type": NoiseType.AMBIENT,
                "intensity": 35.0,
                "freq_range": (20.0, 200.0),
                "confidence": 0.8,
                "duration": 0.0,  # Continuous
                "intermittent": False,
                "pattern": "constant",
            },
            {
                "type": NoiseType.ELECTRONIC,
                "intensity": 25.0,
                "freq_range": (2000.0, 8000.0),
                "confidence": 0.6,
                "duration": 0.0,
                "intermittent": True,
                "pattern": "periodic",
            },
        ]

        profiles = []
        for scenario in scenarios[:1]:  # Return first scenario for demo
            profile = NoiseProfile(
                noise_type=scenario["type"],
                intensity_db=scenario["intensity"],
                frequency_range=scenario["freq_range"],
                confidence=scenario["confidence"],
                duration=scenario["duration"],
                intermittent=scenario["intermittent"],
                pattern=scenario["pattern"],
                reduction_factor=0.8,
            )
            profiles.append(profile)

        return profiles

    async def cancel_noise_for_agent(
        self, agent_id: str, audio_data: np.ndarray, noise_profiles: List[NoiseProfile]
    ) -> np.ndarray:
        """Apply noise cancellation specific to an agent's voice"""

        if agent_id not in self.agent_fingerprints:
            logger.warning(
                f"No fingerprint found for agent {agent_id}, using generic noise cancellation"
            )
            return self._generic_noise_cancellation(audio_data, noise_profiles)

        fingerprint = self.agent_fingerprints[agent_id]

        # Agent-specific noise cancellation
        processed_audio = audio_data.copy()

        for noise_profile in noise_profiles:
            processed_audio = self._apply_agent_specific_cancellation(
                processed_audio, fingerprint, noise_profile
            )

        # Enhance agent's voice characteristics
        processed_audio = self._enhance_agent_voice(processed_audio, fingerprint)

        return processed_audio

    def _apply_agent_specific_cancellation(
        self, audio_data: np.ndarray, fingerprint: VoiceFingerprint, noise_profile: NoiseProfile
    ) -> np.ndarray:
        """Apply noise cancellation that preserves agent's voice characteristics"""

        # Protect agent's fundamental frequency and formants
        protected_frequencies = [
            fingerprint.fundamental_frequency
        ] + fingerprint.formant_frequencies

        # Apply filtering that avoids agent's key frequencies
        processed = audio_data.copy()

        # Simple demonstration - in practice would use advanced signal processing
        noise_freq_min, noise_freq_max = noise_profile.frequency_range

        # If noise overlaps with agent's voice, use gentler reduction
        overlap_factor = self._calculate_frequency_overlap(protected_frequencies, noise_profile)
        reduction = noise_profile.reduction_factor * (1.0 - overlap_factor * 0.5)

        # Apply noise reduction (simulated)
        if noise_profile.noise_type in [NoiseType.AMBIENT, NoiseType.ELECTRONIC]:
            processed = processed * (1.0 - reduction * 0.3)  # Gentle reduction

        return processed

    def _calculate_frequency_overlap(
        self, agent_frequencies: List[float], noise_profile: NoiseProfile
    ) -> float:
        """Calculate how much agent's voice overlaps with noise frequencies"""

        noise_min, noise_max = noise_profile.frequency_range
        overlap_count = 0

        for freq in agent_frequencies:
            if noise_min <= freq <= noise_max:
                overlap_count += 1

        return overlap_count / len(agent_frequencies) if agent_frequencies else 0.0

    def _enhance_agent_voice(
        self, audio_data: np.ndarray, fingerprint: VoiceFingerprint
    ) -> np.ndarray:
        """Enhance agent's specific voice characteristics"""

        enhanced = audio_data.copy()

        # Boost agent's fundamental frequency and formants
        enhancement_factor = self.voice_enhancement_factor * fingerprint.noise_tolerance

        # Simple enhancement simulation
        enhanced = enhanced * enhancement_factor

        # Apply voice-specific processing based on fingerprint
        if fingerprint.breathiness_index > 0.1:
            # Add slight warmth for breathy voices
            enhanced = enhanced * 1.05

        if fingerprint.roughness_index < 0.05:
            # Enhance clarity for smooth voices
            enhanced = enhanced * 1.02

        return enhanced

    def _generic_noise_cancellation(
        self, audio_data: np.ndarray, noise_profiles: List[NoiseProfile]
    ) -> np.ndarray:
        """Generic noise cancellation when no agent fingerprint available"""

        processed = audio_data.copy()

        for noise_profile in noise_profiles:
            reduction = noise_profile.reduction_factor * self.noise_reduction_factor
            processed = processed * (1.0 - reduction * 0.2)

        return processed

    async def separate_agent_voices(
        self, mixed_audio: np.ndarray, agent_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Separate multiple agent voices from mixed audio"""

        separated = {}

        for agent_id in agent_ids:
            if agent_id in self.agent_fingerprints:
                # Use fingerprint to isolate this agent's voice
                fingerprint = self.agent_fingerprints[agent_id]
                isolated_audio = self._isolate_agent_voice(mixed_audio, fingerprint)
                separated[agent_id] = isolated_audio
            else:
                logger.warning(f"Cannot separate voice for {agent_id}: no fingerprint")
                separated[agent_id] = mixed_audio  # Return original

        return separated

    def _isolate_agent_voice(
        self, mixed_audio: np.ndarray, fingerprint: VoiceFingerprint
    ) -> np.ndarray:
        """Isolate a specific agent's voice from mixed audio"""

        # Use voice signature to identify and extract agent's voice
        # This is a simplified simulation - real implementation would use
        # advanced source separation techniques

        isolation_factor = fingerprint.confidence_score * 0.8
        isolated = mixed_audio * isolation_factor

        return isolated

    def get_agent_voice_settings(self, agent_id: str) -> Dict[str, Any]:
        """Get optimized voice settings for an agent based on fingerprint"""

        if agent_id not in self.agent_fingerprints:
            return self._get_default_voice_settings()

        fingerprint = self.agent_fingerprints[agent_id]

        settings = {
            "pitch_base": fingerprint.fundamental_frequency,
            "formant_f1": fingerprint.formant_frequencies[0]
            if fingerprint.formant_frequencies
            else 500,
            "formant_f2": fingerprint.formant_frequencies[1]
            if len(fingerprint.formant_frequencies) > 1
            else 1500,
            "breathiness": fingerprint.breathiness_index,
            "roughness": fingerprint.roughness_index,
            "noise_gate_threshold": self.noise_gate_threshold,
            "enhancement_factor": self.voice_enhancement_factor * fingerprint.noise_tolerance,
            "clarity_boost": fingerprint.clarity_threshold,
        }

        return settings

    def _get_default_voice_settings(self) -> Dict[str, Any]:
        """Default voice settings when no fingerprint available"""
        return {
            "pitch_base": 150.0,
            "formant_f1": 500.0,
            "formant_f2": 1500.0,
            "breathiness": 0.1,
            "roughness": 0.05,
            "noise_gate_threshold": self.noise_gate_threshold,
            "enhancement_factor": self.voice_enhancement_factor,
            "clarity_boost": 0.7,
        }

    async def adapt_to_environment(
        self, agent_id: str, environment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt agent voice settings to current environment"""

        # Detect current noise environment
        noise_profiles = await self.detect_noise_environment()

        # Get agent's base settings
        base_settings = self.get_agent_voice_settings(agent_id)

        # Adapt settings based on environment
        adapted_settings = base_settings.copy()

        # Calculate total noise intensity
        total_noise = sum(profile.intensity_db for profile in noise_profiles)

        if total_noise > 40:  # Noisy environment
            adapted_settings["enhancement_factor"] *= 1.3
            adapted_settings["clarity_boost"] *= 1.2
            adapted_settings["noise_gate_threshold"] += 5
        elif total_noise < 20:  # Quiet environment
            adapted_settings["enhancement_factor"] *= 0.9
            adapted_settings["breathiness"] *= 1.1  # Allow more natural breathing

        # Adapt to specific noise types
        for profile in noise_profiles:
            if profile.noise_type == NoiseType.SPEECH:
                # Other people talking - boost distinctiveness
                adapted_settings["pitch_base"] *= 1.05
                adapted_settings["enhancement_factor"] *= 1.2
            elif profile.noise_type == NoiseType.MUSIC:
                # Background music - adjust to complement rather than compete
                adapted_settings["pitch_base"] *= 0.98
                adapted_settings["breathiness"] *= 0.8

        return {
            "adapted_settings": adapted_settings,
            "noise_profiles": [
                {"type": p.noise_type.value, "intensity": p.intensity_db} for p in noise_profiles
            ],
            "adaptation_confidence": min(
                1.0,
                sum(p.confidence for p in noise_profiles) / len(noise_profiles)
                if noise_profiles
                else 0.5,
            ),
        }

    def create_agent_type_fingerprints(self) -> Dict[str, VoiceFingerprint]:
        """Create fingerprints for different types of agents"""

        agent_types = {
            "Astra": "analytical",
            "Phi": "analytical",
            "Oracle": "authoritative",
            "Echo": "creative",
            "MusicComposer": "music",
            "MusicSense": "music",
            "VoiceModulator": "music",
            "DataAgent": "analytical",
            "CreativeAgent": "creative",
            "SupervisorAgent": "authoritative",
        }

        fingerprints = {}
        for agent_name, agent_type in agent_types.items():
            fingerprint = self.create_agent_fingerprint(agent_name, agent_type)
            fingerprints[agent_name] = fingerprint

        return fingerprints

    def export_fingerprints(self, filepath: str):
        """Export agent fingerprints to file"""
        export_data = {}

        for agent_id, fingerprint in self.agent_fingerprints.items():
            export_data[agent_id] = {
                "fundamental_frequency": fingerprint.fundamental_frequency,
                "formant_frequencies": fingerprint.formant_frequencies,
                "spectral_centroid": fingerprint.spectral_centroid,
                "vocal_tract_length": fingerprint.vocal_tract_length,
                "breathiness_index": fingerprint.breathiness_index,
                "roughness_index": fingerprint.roughness_index,
                "voice_signature": fingerprint.voice_signature,
                "confidence_score": fingerprint.confidence_score,
                "noise_tolerance": fingerprint.noise_tolerance,
                "clarity_threshold": fingerprint.clarity_threshold,
            }

        try:
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Exported {len(export_data)} fingerprints to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export fingerprints: {e}")

    def import_fingerprints(self, filepath: str):
        """Import agent fingerprints from file"""
        try:
            with open(filepath, "r") as f:
                import_data = json.load(f)

            for agent_id, data in import_data.items():
                fingerprint = VoiceFingerprint(
                    agent_id=agent_id,
                    fundamental_frequency=data["fundamental_frequency"],
                    formant_frequencies=data["formant_frequencies"],
                    spectral_centroid=data["spectral_centroid"],
                    vocal_tract_length=data["vocal_tract_length"],
                    breathiness_index=data["breathiness_index"],
                    roughness_index=data["roughness_index"],
                    voice_signature=data["voice_signature"],
                    confidence_score=data["confidence_score"],
                    noise_tolerance=data["noise_tolerance"],
                    clarity_threshold=data["clarity_threshold"],
                )
                self.agent_fingerprints[agent_id] = fingerprint

            logger.info(f"Imported {len(import_data)} fingerprints from {filepath}")
        except Exception as e:
            logger.error(f"Failed to import fingerprints: {e}")


# Global voice processor instance
_voice_processor = None


def get_voice_processor() -> UniversalVoiceProcessor:
    """Get the global voice processor instance"""
    global _voice_processor
    if _voice_processor is None:
        _voice_processor = UniversalVoiceProcessor()
    return _voice_processor


async def demo_voice_fingerprinting():
    """Demonstrate voice fingerprinting and noise cancellation"""
    print("🎤 Voice Fingerprinting and Noise Cancellation Demo")
    print("=" * 60)

    processor = get_voice_processor()
    # Create fingerprints for various agents
    print("\n--- Creating Agent Voice Fingerprints ---")
    processor.create_agent_type_fingerprints()
    print(f"Created fingerprints for {len(processor.agent_fingerprints)} agents")

    # Demonstrate noise detection
    print("\n--- Environmental Noise Detection ---")
    noise_profiles = await processor.detect_noise_environment()
    for profile in noise_profiles:
        print(f"Detected: {profile.noise_type.value} at {profile.intensity_db:.1f}dB")
        print(
            f"  Frequency range: {profile.frequency_range[0]:.0f}-{profile.frequency_range[1]:.0f}Hz"
        )
        print(f"  Confidence: {profile.confidence:.2f}")

    # Demonstrate environment adaptation
    print("\n--- Environment Adaptation ---")
    for agent_name in ["Astra", "MusicComposer", "Oracle"]:
        adaptation = await processor.adapt_to_environment(agent_name, {})
        print(f"\n{agent_name} adaptation:")
        print(f"  Enhancement factor: {adaptation['adapted_settings']['enhancement_factor']:.2f}")
        print(f"  Clarity boost: {adaptation['adapted_settings']['clarity_boost']:.2f}")
        print(f"  Noise gate: {adaptation['adapted_settings']['noise_gate_threshold']:.0f}dB")

    # Demonstrate voice separation
    print("\n--- Multi-Agent Voice Separation ---")
    # Simulate mixed audio (normally would be real audio data)
    mixed_audio = np.random.randn(1000)  # Simulated audio
    agent_list = ["Astra", "Phi", "MusicComposer"]

    separated = await processor.separate_agent_voices(mixed_audio, agent_list)
    for agent_id, audio in separated.items():
        print(f"Separated {agent_id}: {len(audio)} samples")

    # Export fingerprints
    print("\n--- Exporting Voice Fingerprints ---")
    processor.export_fingerprints("agent_voice_fingerprints.json")

    print("\n✅ Voice fingerprinting and noise cancellation demo complete!")
    return processor


if __name__ == "__main__":
    asyncio.run(demo_voice_fingerprinting())
