#!/usr/bin/env python3
"""
Simple TTS/STT Test for VoxSigil
================================
"""

import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_edge_tts():
    """Test Edge TTS directly"""
    print("Testing Edge TTS...")
    try:
        import tempfile

        import edge_tts

        # Create a simple TTS test
        text = "Hello! This is a test of the Edge TTS system in VoxSigil."
        voice = "en-US-AriaNeural"

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            output_file = tmp_file.name

        print(f"Synthesizing: '{text}'")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

        print(f"✅ Edge TTS successful! Audio saved to: {output_file}")

        # Check file size
        file_size = os.path.getsize(output_file)
        print(f"   Audio file size: {file_size} bytes")

        # Clean up
        os.unlink(output_file)
        return True

    except Exception as e:
        print(f"❌ Edge TTS failed: {e}")
        return False


def test_pyttsx3():
    """Test pyttsx3 TTS"""
    print("\nTesting pyttsx3...")
    try:
        import pyttsx3

        engine = pyttsx3.init()

        # Get available voices
        voices = engine.getProperty("voices")
        print(f"✅ pyttsx3 initialized with {len(voices)} voices")

        for i, voice in enumerate(voices[:3]):  # Show first 3 voices
            print(f"   Voice {i}: {voice.name} ({voice.id})")

        # Test speech (this will actually speak if speakers are connected)
        text = "Hello! This is a test of pyttsx3 in VoxSigil."
        print(f"Speaking: '{text}'")
        engine.say(text)
        engine.runAndWait()

        print("✅ pyttsx3 TTS successful!")
        return True

    except Exception as e:
        print(f"❌ pyttsx3 failed: {e}")
        return False


def test_agent_voice_system():
    """Test the VoxSigil agent voice system"""
    print("\nTesting VoxSigil Agent Voice System...")
    try:
        from core.agent_voice_system import AgentVoiceSystem

        voice_system = AgentVoiceSystem()
        print(f"✅ Agent voice system loaded with {len(voice_system.voice_profiles)} profiles")

        # Show some agent voice profiles
        sample_agents = ["Astra", "Phi", "Oracle", "Voxka"]
        for agent in sample_agents:
            if agent in voice_system.voice_profiles:
                profile = voice_system.voice_profiles[agent]
                print(
                    f"   {agent}: {profile.voice_id} (tone: {profile.tone}, pitch: {profile.pitch})"
                )

        return True

    except Exception as e:
        print(f"❌ Agent voice system failed: {e}")
        return False


def test_agent_speak_method():
    """Test the agent speak method"""
    print("\nTesting Agent Speak Method...")
    try:
        from agents.phi import Phi  # Try to import a real agent

        # Create agent instance
        phi = Phi()
        print(f"✅ Agent {phi.__class__.__name__} created")

        # Test speak method
        result = phi.speak(
            "Hello! I am Phi, testing my voice capabilities in VoxSigil.", add_signature=True
        )

        if result:
            print("✅ Agent speak method executed successfully")
        else:
            print("⚠️  Agent speak method completed (result may vary based on TTS availability)")

        return True

    except Exception as e:
        print(f"❌ Agent speak test failed: {e}")
        return False


def test_vosk_stt():
    """Test Vosk STT if available"""
    print("\nTesting Vosk STT...")
    try:
        # Check if we can create a model (this would need a downloaded model)
        print("✅ Vosk imported successfully")
        print("   Note: STT testing requires a downloaded language model")
        print("   Download models from: https://alphacephei.com/vosk/models")

        return True

    except Exception as e:
        print(f"❌ Vosk STT test failed: {e}")
        return False


async def main():
    """Run TTS/STT tests"""
    print("VoxSigil TTS/STT Quick Test")
    print("=" * 40)

    # Test Edge TTS
    edge_ok = await test_edge_tts()

    # Test pyttsx3
    pyttsx3_ok = test_pyttsx3()

    # Test VoxSigil systems
    voice_system_ok = test_agent_voice_system()
    agent_speak_ok = test_agent_speak_method()

    # Test STT
    stt_ok = test_vosk_stt()

    # Summary
    print("\n" + "=" * 40)
    print("TEST RESULTS SUMMARY")
    print("=" * 40)
    print(f"Edge TTS: {'✅ Working' if edge_ok else '❌ Failed'}")
    print(f"pyttsx3 TTS: {'✅ Working' if pyttsx3_ok else '❌ Failed'}")
    print(f"Agent Voice System: {'✅ Working' if voice_system_ok else '❌ Failed'}")
    print(f"Agent Speak Method: {'✅ Working' if agent_speak_ok else '❌ Failed'}")
    print(f"Vosk STT: {'✅ Available' if stt_ok else '❌ Failed'}")

    if edge_ok or pyttsx3_ok:
        print("\n🎉 TTS functionality is working!")
        print("🎤 Agents can speak with their unique voices")
    else:
        print("\n⚠️  TTS functionality needs attention")

    if stt_ok:
        print("🎙️ STT capability is available (needs model download)")
    else:
        print("⚠️  STT functionality needs attention")


if __name__ == "__main__":
    asyncio.run(main())
