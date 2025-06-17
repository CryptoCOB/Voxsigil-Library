#!/usr/bin/env python3
"""
Quick Agent Voice Test - Play Sample Agent Voices
================================================

Quick test to play a few agent voices to hear their characteristics.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the VoxSigil library to the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from core.agent_voice_system import AgentVoiceSystem

    voice_available = True
except ImportError as e:
    logger.error(f"Voice system not available: {e}")
    voice_available = False

try:
    import pyttsx3

    tts_available = True
except ImportError:
    tts_available = False

try:
    import edge_tts

    edge_available = True
except ImportError:
    edge_available = False


async def quick_voice_test():
    """Quick test of agent voices"""

    if not voice_available:
        print("❌ Voice system not available")
        return

    print("🎤 Quick Agent Voice Test")
    print("=" * 40)

    voice_system = AgentVoiceSystem()

    # Test a few key agents
    test_agents = ["Astra", "Phi", "Echo", "Oracle", "Voxka"]

    for agent_name in test_agents:
        profile = voice_system.get_voice_profile(agent_name)
        if not profile:
            print(f"❌ No profile for {agent_name}")
            continue

        print(f"\n--- {agent_name} ---")
        print(f"Personality: {', '.join(profile.personality_traits)}")
        print(f"Tone: {profile.tone}")
        print(f"Voice settings: Pitch {profile.pitch:+.1f}st, Speed {profile.speed:.1f}x")

        # Get signature phrase
        signature = voice_system.get_signature_phrase(agent_name)
        text = f"Hello! I'm {agent_name}. {signature}"
        print(f'Speaking: "{text}"')

        # Try to speak with pyttsx3 (simpler fallback)
        if tts_available:
            try:
                engine = pyttsx3.init()

                # Adjust voice properties
                rate = int(200 * profile.speed)
                engine.setProperty("rate", rate)
                engine.setProperty("volume", profile.volume)

                # Try to select appropriate voice
                voices = engine.getProperty("voices")
                if voices and len(voices) > 1:
                    if "female" in profile.voice_id:
                        # Try to find female voice
                        for voice in voices:
                            if "female" in voice.name.lower():
                                engine.setProperty("voice", voice.id)
                                break
                    else:
                        # Use male voice
                        for voice in voices:
                            if "male" in voice.name.lower():
                                engine.setProperty("voice", voice.id)
                                break

                print("🔊 Playing voice...")
                engine.say(text)
                engine.runAndWait()

            except Exception as e:
                print(f"❌ TTS error: {e}")
        else:
            print("🔇 No TTS engine available - voice characteristics defined but cannot play")

        # Wait between agents
        await asyncio.sleep(1)

    print(f"\n✅ Voice test complete! Tested {len(test_agents)} agents.")

    # Show all available agents
    all_profiles = voice_system.get_all_profiles()
    print(f"\n📋 All available agents ({len(all_profiles)}):")
    for i, (name, profile) in enumerate(all_profiles.items(), 1):
        print(f"{i:2d}. {name} - {profile.tone} ({', '.join(profile.personality_traits[:2])})")


async def main():
    """Main function"""

    print("🔧 System Check:")
    print(f"Voice System: {'✅' if voice_available else '❌'}")
    print(f"pyttsx3 TTS: {'✅' if tts_available else '❌'}")
    print(f"Edge TTS: {'✅' if edge_available else '❌'}")
    print()

    await quick_voice_test()


if __name__ == "__main__":
    asyncio.run(main())
