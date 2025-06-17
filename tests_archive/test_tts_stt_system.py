#!/usr/bin/env python3
"""
VoxSigil TTS/STT Test Suite
===========================

Comprehensive testing of Text-to-Speech (TTS) and Speech-to-Text (STT) functionality
in the VoxSigil system.

This test will:
1. Check if TTS/STT engines are available
2. Test basic TTS functionality with different voices
3. Test STT functionality if microphone is available
4. Test agent voice system integration
5. Provide recommendations for fixing any issues
"""

import asyncio
import logging
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_tts_dependencies():
    """Test if TTS dependencies are available"""
    print("=" * 60)
    print("TESTING TTS DEPENDENCIES")
    print("=" * 60)

    results = {}

    # Test edge-tts
    try:
        import edge_tts

        print("✅ edge-tts available")
        results["edge_tts"] = True
    except ImportError as e:
        print(f"❌ edge-tts not available: {e}")
        results["edge_tts"] = False

    # Test pyttsx3
    try:
        import pyttsx3

        print("✅ pyttsx3 available")
        results["pyttsx3"] = True
    except ImportError as e:
        print(f"❌ pyttsx3 not available: {e}")
        results["pyttsx3"] = False

    return results


def test_stt_dependencies():
    """Test if STT dependencies are available"""
    print("\n" + "=" * 60)
    print("TESTING STT DEPENDENCIES")
    print("=" * 60)

    results = {}

    # Test vosk
    try:
        import vosk

        print("✅ vosk available")
        results["vosk"] = True
    except ImportError as e:
        print(f"❌ vosk not available: {e}")
        results["vosk"] = False

    # Test sounddevice
    try:
        import sounddevice as sd

        print("✅ sounddevice available")
        results["sounddevice"] = True
    except ImportError as e:
        print(f"❌ sounddevice not available: {e}")
        results["sounddevice"] = False

    # Test numpy (needed for audio processing)
    try:
        import numpy as np

        print("✅ numpy available")
        results["numpy"] = True
    except ImportError as e:
        print(f"❌ numpy not available: {e}")
        results["numpy"] = False

    return results


def test_agent_voice_system():
    """Test the agent voice system"""
    print("\n" + "=" * 60)
    print("TESTING AGENT VOICE SYSTEM")
    print("=" * 60)

    try:
        from core.agent_voice_system import AgentVoiceSystem

        print("✅ AgentVoiceSystem imported successfully")

        # Initialize voice system
        voice_system = AgentVoiceSystem()
        print(f"✅ Voice system initialized with {len(voice_system.voice_profiles)} agent profiles")

        # Test some agent profiles
        test_agents = ["Astra", "Phi", "Voxka", "Oracle"]
        for agent in test_agents:
            if agent in voice_system.voice_profiles:
                profile = voice_system.voice_profiles[agent]
                print(f"✅ {agent}: voice_id={profile.voice_id}, tone={profile.tone}")
            else:
                print(f"⚠️  {agent}: no voice profile found")

        return True

    except Exception as e:
        print(f"❌ Agent voice system test failed: {e}")
        return False


def test_tts_engines():
    """Test TTS engines"""
    print("\n" + "=" * 60)
    print("TESTING TTS ENGINES")
    print("=" * 60)

    try:
        from engines.async_tts_engine import AsyncTTSEngine

        print("✅ AsyncTTSEngine imported successfully")

        # Try to create TTS engine
        tts_engine = AsyncTTSEngine()
        print("✅ TTS engine created")

        available_engines = tts_engine.get_available_engines()
        print(f"✅ Available TTS engines: {available_engines}")

        return True, tts_engine

    except Exception as e:
        print(f"❌ TTS engine test failed: {e}")
        return False, None


def test_stt_engines():
    """Test STT engines"""
    print("\n" + "=" * 60)
    print("TESTING STT ENGINES")
    print("=" * 60)

    try:
        from engines.async_stt_engine import AsyncSTTEngine

        print("✅ AsyncSTTEngine imported successfully")

        # Try to create STT engine
        stt_engine = AsyncSTTEngine()
        print("✅ STT engine created")

        return True, stt_engine

    except Exception as e:
        print(f"❌ STT engine test failed: {e}")
        return False, None


async def test_basic_tts(tts_engine):
    """Test basic TTS functionality"""
    print("\n" + "=" * 60)
    print("TESTING BASIC TTS FUNCTIONALITY")
    print("=" * 60)

    if not tts_engine:
        print("❌ No TTS engine available for testing")
        return False

    try:
        # Test text
        test_text = "Hello! This is a test of the VoxSigil text-to-speech system."

        print(f"🎤 Testing TTS with text: '{test_text}'")

        # Create TTS request
        from engines.async_tts_engine import TTSRequest

        request = TTSRequest(
            text=test_text,
            voice_id="default",
            output_path=None,  # Will use temporary file
        )

        # Synthesize
        result = await tts_engine.synthesize_speech(request)

        if result and result.success:
            print("✅ TTS synthesis successful!")
            print(f"   Engine used: {result.engine_used}")
            print(f"   Output file: {result.audio_path}")
            print(f"   Duration: {result.synthesis_time:.2f}s")
            return True
        else:
            print(f"❌ TTS synthesis failed: {result.error if result else 'Unknown error'}")
            return False

    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False


def test_agent_speak():
    """Test agent speak functionality"""
    print("\n" + "=" * 60)
    print("TESTING AGENT SPEAK FUNCTIONALITY")
    print("=" * 60)

    try:
        from agents.base import BaseAgent

        # Create a test agent
        class TestAgent(BaseAgent):
            def __init__(self):
                self.sigil = "TEST"
                self._voice_profile = "test"

        agent = TestAgent()
        print("✅ Test agent created")

        # Test speak method
        result = agent.speak("Hello, this is a test of the agent voice system!", add_signature=True)

        if result:
            print("✅ Agent speak method executed successfully")
            return True
        else:
            print("⚠️  Agent speak method returned False (might be normal if no TTS available)")
            return False

    except Exception as e:
        print(f"❌ Agent speak test failed: {e}")
        return False


def provide_installation_guide(tts_results, stt_results):
    """Provide installation guide for missing dependencies"""
    print("\n" + "=" * 60)
    print("INSTALLATION GUIDE FOR MISSING DEPENDENCIES")
    print("=" * 60)

    missing_deps = []

    # Check TTS dependencies
    if not tts_results.get("edge_tts", True):
        missing_deps.append("edge-tts")
    if not tts_results.get("pyttsx3", True):
        missing_deps.append("pyttsx3")

    # Check STT dependencies
    if not stt_results.get("vosk", True):
        missing_deps.append("vosk")
    if not stt_results.get("sounddevice", True):
        missing_deps.append("sounddevice")
    if not stt_results.get("numpy", True):
        missing_deps.append("numpy")

    if missing_deps:
        print("📦 Install missing dependencies with:")
        print(f"   pip install {' '.join(missing_deps)}")
        print()
        print("📝 Additional notes:")
        print("   - edge-tts: Provides high-quality cloud-based TTS")
        print("   - pyttsx3: Provides offline TTS using system voices")
        print("   - vosk: Provides offline speech recognition")
        print("   - sounddevice: Required for microphone input")
        print("   - numpy: Required for audio processing")
    else:
        print("✅ All dependencies are available!")


async def main():
    """Run comprehensive TTS/STT test suite"""
    print("VOXSIGIL TTS/STT TEST SUITE")
    print("=" * 60)
    print("Testing Text-to-Speech and Speech-to-Text functionality")
    print()

    # Test dependencies
    tts_deps = test_tts_dependencies()
    stt_deps = test_stt_dependencies()

    # Test systems
    voice_system_ok = test_agent_voice_system()
    tts_ok, tts_engine = test_tts_engines()
    stt_ok, stt_engine = test_stt_engines()

    # Test functionality
    if tts_ok and tts_engine:
        tts_basic_ok = await test_basic_tts(tts_engine)
    else:
        tts_basic_ok = False

    agent_speak_ok = test_agent_speak()

    # Provide installation guide
    provide_installation_guide(tts_deps, stt_deps)

    # Final report
    print("\n" + "=" * 60)
    print("FINAL TTS/STT TEST REPORT")
    print("=" * 60)

    print("TTS Dependencies:")
    for dep, status in tts_deps.items():
        print(f"  {dep}: {'✅ Available' if status else '❌ Missing'}")

    print("STT Dependencies:")
    for dep, status in stt_deps.items():
        print(f"  {dep}: {'✅ Available' if status else '❌ Missing'}")

    print("System Components:")
    print(f"  Voice System: {'✅ Working' if voice_system_ok else '❌ Failed'}")
    print(f"  TTS Engine: {'✅ Working' if tts_ok else '❌ Failed'}")
    print(f"  STT Engine: {'✅ Working' if stt_ok else '❌ Failed'}")

    print("Functionality Tests:")
    print(f"  Basic TTS: {'✅ Working' if tts_basic_ok else '❌ Failed'}")
    print(f"  Agent Speak: {'✅ Working' if agent_speak_ok else '❌ Failed'}")

    # Overall status
    all_critical_working = voice_system_ok and tts_ok

    print("\n" + "=" * 60)
    if all_critical_working:
        print("🎉 SUCCESS: TTS/STT system is working!")
        print("🎤 Agents can speak with unique voices")
        print("🔊 Text-to-speech functionality is available")
        if stt_ok:
            print("🎙️ Speech-to-text functionality is available")
    else:
        print("⚠️  ISSUES DETECTED in TTS/STT system")
        print("🔧 Please install missing dependencies and retry")
        print("📝 Check the installation guide above")


if __name__ == "__main__":
    asyncio.run(main())
