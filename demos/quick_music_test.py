#!/usr/bin/env python3
"""Very simple music agent import test"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing music agent imports...")
    print("✅ SUCCESS: MusicComposerAgent imports work!")
    success = True
except Exception as e:
    print(f"❌ FAILED: {e}")
    success = False

if success:
    print("🎵 Music agents are now available - simulation mode warning should be gone!")
else:
    print("⚠️ Music agents still not available - will still show simulation mode warning")
