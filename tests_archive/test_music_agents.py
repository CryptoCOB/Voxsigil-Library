#!/usr/bin/env python3
"""
Test script to verify music agents are importable and working.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_music_agent_imports():
    """Test importing music agents"""
    print("Testing music agent imports...")

    try:
        from agents.ensemble.music.music_composer_agent import (
            CompositionRequest,
            MusicComposerAgent,
        )

        print("✅ CompositionRequest and MusicComposerAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CompositionRequest/MusicComposerAgent: {e}")
        return False

    try:
        from agents.ensemble.music.music_sense_agent import MusicSenseAgent

        print("✅ MusicSenseAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MusicSenseAgent: {e}")
        return False

    try:
        from agents.ensemble.music.voice_modulator_agent import (
            VoiceModulationRequest,
            VoiceModulatorAgent,
        )

        print("✅ VoiceModulatorAgent and VoiceModulationRequest imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import VoiceModulatorAgent/VoiceModulationRequest: {e}")
        return False

    print("✅ All music agents imported successfully!")
    return True


def test_music_tab_availability():
    """Test if music tab will detect agents as available"""
    print("\nTesting music tab agent detection...")

    try:
        # Simulate the import check from music_tab.py
        from agents.ensemble.music.music_composer_agent import (
            CompositionRequest,
            MusicComposerAgent,
        )
        from agents.ensemble.music.music_sense_agent import MusicSenseAgent
        from agents.ensemble.music.voice_modulator_agent import (
            VoiceModulationRequest,
            VoiceModulatorAgent,
        )

        MUSIC_AGENTS_AVAILABLE = True
        print("✅ Music agents are available! No more simulation mode.")
        return True

    except ImportError:
        print("❌ Music agents still not available - will run in simulation mode")
        return False


if __name__ == "__main__":
    success = test_music_agent_imports()
    if success:
        test_music_tab_availability()
    else:
        print("\n❌ Music agent import test failed")
        sys.exit(1)
