#!/usr/bin/env python3
"""Simple test to check music agent availability"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.ensemble.music.music_composer_agent import CompositionRequest, MusicComposerAgent

    print("✅ SUCCESS: Music agents are now available!")
    print("✅ CompositionRequest and MusicComposerAgent imported successfully")
    print("✅ The 'Music agents not available' warning should be fixed")
except ImportError as e:
    print(f"❌ FAILED: Music agents still not available: {e}")
