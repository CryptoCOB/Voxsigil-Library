"""Ensemble agents for VoxSigil."""

# Import music agents
try:
    from .music import (
        CompositionRequest,
        MusicComposerAgent,
        MusicSenseAgent,
        VoiceModulationRequest,
        VoiceModulatorAgent,
    )

    __all__ = [
        "MusicComposerAgent",
        "CompositionRequest",
        "MusicSenseAgent",
        "VoiceModulatorAgent",
        "VoiceModulationRequest",
    ]
except ImportError:
    __all__ = []
