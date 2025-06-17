"""Music processing agents for VoxSigil."""

from .music_composer_agent import CompositionRequest, MusicComposerAgent
from .music_sense_agent import MusicSenseAgent
from .voice_modulator_agent import VoiceModulationRequest, VoiceModulatorAgent

__all__ = [
    "MusicComposerAgent",
    "CompositionRequest",
    "MusicSenseAgent",
    "VoiceModulatorAgent",
    "VoiceModulationRequest",
]
