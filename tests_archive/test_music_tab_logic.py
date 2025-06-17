#!/usr/bin/env python3
"""Test music tab agent detection logic"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Replicate the import logic from music_tab.py
try:
    from agents.ensemble.music.music_composer_agent import CompositionRequest, MusicComposerAgent
    from agents.ensemble.music.music_sense_agent import MusicSenseAgent
    from agents.ensemble.music.voice_modulator_agent import (
        VoiceModulationRequest,
        VoiceModulatorAgent,
    )

    MUSIC_AGENTS_AVAILABLE = True
    print("✅ SUCCESS: Music agents are now available!")
    print("✅ MUSIC_AGENTS_AVAILABLE = True")
    print("✅ The 'Music agents not available, running in simulation mode' warning should be GONE!")

except ImportError as e:
    MUSIC_AGENTS_AVAILABLE = False
    print(f"❌ FAILED: Music agents still not available: {e}")
    print("❌ MUSIC_AGENTS_AVAILABLE = False")
    print("❌ GUI will still show 'Music agents not available, running in simulation mode'")

print(f"\nResult: MUSIC_AGENTS_AVAILABLE = {MUSIC_AGENTS_AVAILABLE}")
