#!/usr/bin/env python3
"""
TTS/STT Status Check for VoxSigil
=================================
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check TTS/STT dependencies"""
    print("TTS/STT DEPENDENCY CHECK")
    print("=" * 30)

    deps = {
        "edge_tts": False,
        "pyttsx3": False,
        "vosk": False,
        "sounddevice": False,
        "numpy": False,
    }

    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")

    return deps


def test_agent_voice_profiles():
    """Test agent voice profiles"""
    print("\nAGENT VOICE PROFILES")
    print("=" * 30)

    try:
        from core.agent_voice_system import AgentVoiceSystem

        voice_system = AgentVoiceSystem()
        print(f"Voice profiles loaded: {len(voice_system.voice_profiles)}")

        # Show sample profiles
        for name, profile in list(voice_system.voice_profiles.items())[:5]:
            print(f"  {name}: {profile.voice_id} ({profile.tone})")

        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def test_simple_speak():
    """Test simple agent speak"""
    print("\nAGENT SPEAK TEST")
    print("=" * 30)

    try:
        # Import a simple agent
        from agents.base import BaseAgent

        class TestAgent(BaseAgent):
            def __init__(self):
                self.sigil = "TEST"

        agent = TestAgent()

        # Test speak method (this should work even if TTS is not fully set up)
        print("Testing agent.speak() method...")
        result = agent.speak("This is a test message from VoxSigil.", add_signature=False)

        print(f"Speak method result: {result}")
        return True

    except Exception as e:
        print(f"Failed: {e}")
        return False


def main():
    """Main test"""
    print("VOXSIGIL TTS/STT STATUS CHECK")
    print("=" * 40)

    # Check dependencies
    deps = check_dependencies()

    # Test voice system
    voice_ok = test_agent_voice_profiles()

    # Test speak method
    speak_ok = test_simple_speak()

    # Summary
    print("\nSUMMARY")
    print("=" * 30)

    tts_available = deps["edge_tts"] or deps["pyttsx3"]
    stt_available = deps["vosk"] and deps["sounddevice"]

    print(f"TTS Available: {'✅ Yes' if tts_available else '❌ No'}")
    print(f"STT Available: {'✅ Yes' if stt_available else '❌ No'}")
    print(f"Voice System: {'✅ Working' if voice_ok else '❌ Failed'}")
    print(f"Agent Speak: {'✅ Working' if speak_ok else '❌ Failed'}")

    if tts_available and voice_ok:
        print("\n🎉 TTS system is ready!")
        print("Agents can speak with unique voices.")
    else:
        print("\n⚠️  TTS system needs setup.")


if __name__ == "__main__":
    main()
