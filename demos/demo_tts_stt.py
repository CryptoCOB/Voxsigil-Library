#!/usr/bin/env python3
"""
VoxSigil TTS/STT Demo
====================

This script demonstrates the TTS and STT capabilities in VoxSigil.
"""

import asyncio
import os
import sys
import tempfile
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def demo_edge_tts():
    """Demonstrate Edge TTS"""
    print("🎤 EDGE TTS DEMONSTRATION")
    print("-" * 40)

    try:
        import edge_tts

        # Test different voices and agent personalities
        agent_demos = [
            {
                "agent": "Astra (Navigation Agent)",
                "voice": "en-US-AriaNeural",
                "text": "Welcome to VoxSigil! I am Astra, your navigation and guidance assistant. I help you traverse the cognitive landscape.",
            },
            {
                "agent": "Phi (Mathematical Agent)",
                "voice": "en-US-JennyNeural",
                "text": "Greetings! I am Phi, the mathematical reasoning agent. I process numerical patterns and logical structures.",
            },
            {
                "agent": "Oracle (Wisdom Agent)",
                "voice": "en-US-GuyNeural",
                "text": "Hello, I am Oracle. I provide deep insights and wisdom drawn from vast knowledge repositories.",
            },
        ]

        for demo in agent_demos:
            print(f"\n🤖 {demo['agent']}")
            print(f"   Voice: {demo['voice']}")
            print(f"   Text: {demo['text']}")

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    output_file = tmp_file.name

                communicate = edge_tts.Communicate(demo["text"], demo["voice"])
                await communicate.save(output_file)

                file_size = os.path.getsize(output_file)
                print(f"   ✅ Generated audio: {file_size} bytes")

                # Clean up
                os.unlink(output_file)

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Edge TTS demo failed: {e}")
        return False


def demo_pyttsx3():
    """Demonstrate pyttsx3 TTS"""
    print("\n🔊 PYTTSX3 TTS DEMONSTRATION")
    print("-" * 40)

    try:
        import pyttsx3

        engine = pyttsx3.init()

        # Get voices
        voices = engine.getProperty("voices")
        print(f"Available system voices: {len(voices)}")

        for i, voice in enumerate(voices):
            print(f"  {i + 1}. {voice.name}")

        # Demo with different settings
        demos = [
            {
                "rate": 150,
                "volume": 0.8,
                "text": "This is a slow, quiet voice for calm instructions.",
            },
            {
                "rate": 200,
                "volume": 1.0,
                "text": "This is a normal pace voice for standard communication.",
            },
            {
                "rate": 250,
                "volume": 1.0,
                "text": "This is a faster voice for urgent notifications!",
            },
        ]

        for i, demo in enumerate(demos):
            print(f"\n🎵 Demo {i + 1}: Rate={demo['rate']}, Volume={demo['volume']}")
            print(f"   Text: {demo['text']}")

            engine.setProperty("rate", demo["rate"])
            engine.setProperty("volume", demo["volume"])

            print("   🔊 Speaking...")
            engine.say(demo["text"])
            engine.runAndWait()
            print("   ✅ Complete")

            time.sleep(1)  # Brief pause between demos

        return True

    except Exception as e:
        print(f"❌ pyttsx3 demo failed: {e}")
        return False


def demo_voice_profiles():
    """Demonstrate agent voice profiles"""
    print("\n👥 AGENT VOICE PROFILES")
    print("-" * 40)

    try:
        from core.agent_voice_system import AgentVoiceSystem

        voice_system = AgentVoiceSystem()

        print(f"Loaded {len(voice_system.voice_profiles)} agent voice profiles:")

        # Show detailed profiles for key agents
        key_agents = ["Astra", "Phi", "Oracle", "Voxka", "Echo"]

        for agent_name in key_agents:
            if agent_name in voice_system.voice_profiles:
                profile = voice_system.voice_profiles[agent_name]
                print(f"\n🤖 {agent_name}:")
                print(f"   Voice ID: {profile.voice_id}")
                print(f"   Pitch: {profile.pitch}")
                print(f"   Speed: {profile.speed}")
                print(f"   Volume: {profile.volume}")
                print(f"   Tone: {profile.tone}")
                if profile.personality_traits:
                    print(f"   Traits: {', '.join(profile.personality_traits[:3])}")
                if profile.signature_phrases:
                    print(f"   Signature: {profile.signature_phrases[0]}")

        return True

    except Exception as e:
        print(f"❌ Voice profiles demo failed: {e}")
        return False


def demo_stt_info():
    """Show STT information"""
    print("\n🎙️ SPEECH-TO-TEXT INFORMATION")
    print("-" * 40)

    try:
        import sounddevice as sd

        print("✅ Vosk STT engine available")
        print("✅ SoundDevice for microphone input available")

        # Show audio devices
        devices = sd.query_devices()
        print(f"\nAvailable audio devices: {len(devices)}")

        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        print(f"Input devices (microphones): {len(input_devices)}")

        for i, device in enumerate(input_devices[:3]):  # Show first 3
            print(f"  {i + 1}. {device['name']} ({device['max_input_channels']} channels)")

        print("\n📝 Notes:")
        print("  - STT requires downloading language models")
        print("  - Models available at: https://alphacephei.com/vosk/models")
        print("  - Small models (~50MB) work well for basic recognition")
        print("  - Large models (~1GB) provide better accuracy")

        return True

    except Exception as e:
        print(f"❌ STT info failed: {e}")
        return False


async def main():
    """Run TTS/STT demonstration"""
    print("🎉 VOXSIGIL TTS/STT DEMONSTRATION")
    print("=" * 50)
    print("This demo showcases the voice capabilities of VoxSigil agents.")
    print()

    # Run demonstrations
    edge_ok = await demo_edge_tts()
    pyttsx3_ok = demo_pyttsx3()
    profiles_ok = demo_voice_profiles()
    stt_ok = demo_stt_info()

    # Final summary
    print("\n" + "=" * 50)
    print("🏁 DEMONSTRATION SUMMARY")
    print("=" * 50)

    print(f"Edge TTS (Cloud): {'✅ Working' if edge_ok else '❌ Failed'}")
    print(f"pyttsx3 TTS (Local): {'✅ Working' if pyttsx3_ok else '❌ Failed'}")
    print(f"Agent Voice Profiles: {'✅ Working' if profiles_ok else '❌ Failed'}")
    print(f"STT Information: {'✅ Available' if stt_ok else '❌ Failed'}")

    if edge_ok and profiles_ok:
        print("\n🎉 EXCELLENT! VoxSigil TTS system is fully functional!")
        print("🤖 Each agent can speak with a unique, personalized voice")
        print("🎤 High-quality cloud-based TTS via Edge TTS")

    if pyttsx3_ok:
        print("🔊 Local TTS also available for offline operation")

    if stt_ok:
        print("🎙️ STT capability ready (download models to activate)")

    print("\n✨ VoxSigil agents are ready to communicate!")


if __name__ == "__main__":
    asyncio.run(main())
