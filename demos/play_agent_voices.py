#!/usr/bin/env python3
"""
Agent Voice Demonstration - Play All Agent Voices
================================================

This script will play the voices of all VoxSigil agents so you can hear
their unique characteristics and personality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the VoxSigil library to the path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from core.agent_voice_system import AgentVoiceSystem

    voice_system_available = True
    logger.info("✅ Agent voice system available")
except ImportError as e:
    logger.warning(f"Voice system not available: {e}")
    voice_system_available = False

try:
    import asyncio
    import os
    import tempfile

    import edge_tts
    import pygame

    edge_tts_available = True
    logger.info("✅ Edge TTS available for voice playback")
except ImportError as e:
    logger.warning(f"Edge TTS not available: {e}")
    edge_tts_available = False

try:
    import pyttsx3

    pyttsx3_available = True
    logger.info("✅ pyttsx3 available for voice playback")
except ImportError as e:
    logger.warning(f"pyttsx3 not available: {e}")
    pyttsx3_available = False


class AgentVoicePlayer:
    """Play agent voices with their unique characteristics"""

    def __init__(self):
        self.voice_system = AgentVoiceSystem() if voice_system_available else None
        self.tts_engine = None
        self.edge_voices = {
            "en-us-female-1": "en-US-AriaNeural",
            "en-us-male-1": "en-US-GuyNeural",
            "en-us-male-2": "en-US-DavisNeural",
            "en-us-male-3": "en-US-JasonNeural",
            "en-us-female-2": "en-US-JennyNeural",
            "en-us-male-4": "en-US-TonyNeural",
            "en-us-female-3": "en-US-NancyNeural",
            "en-us-female-4": "en-US-SaraNeural",
            "en-us-male-5": "en-US-BrianNeural",
            "en-us-male-6": "en-US-ChristopherNeural",
            "en-us-female-5": "en-US-ElizabethNeural",
            "en-us-male-7": "en-US-EricNeural",
            "en-us-female-6": "en-US-MichelleNeural",
            "en-us-male-8": "en-US-RogerNeural",
            "en-us-female-7": "en-US-SteffanNeural",
            "en-us-female-8": "en-US-AIGenerated1Neural",
        }
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
            self.pygame_available = True
        except Exception:
            self.pygame_available = False
            logger.warning("pygame not available for audio playback")

    async def play_agent_voice_edge_tts(self, agent_name: str, text: str, voice_config: dict):
        """Play agent voice using Edge TTS"""
        try:
            # Get Edge TTS voice
            voice_id = voice_config.get("voice_id", "en-us-neutral")
            edge_voice = self.edge_voices.get(voice_id, "en-US-AriaNeural")

            # Apply pitch and rate from voice config
            pitch = voice_config.get("pitch", 0)
            speed = voice_config.get("speed", 1.0)

            # Create SSML with voice characteristics
            ssml_text = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <prosody pitch="{pitch:+.0f}st" rate="{speed:.1f}">
                    {text}
                </prosody>
            </speak>
            """

            # Generate speech
            communicate = edge_tts.Communicate(ssml_text, edge_voice)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                await communicate.save(tmp_file.name)

                # Play the audio file
                if self.pygame_available:
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()

                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)

                    print(f'🔊 Played {agent_name}\'s voice: "{text}"')
                else:
                    print(f'🔇 Generated audio for {agent_name}: "{text}" (no playback available)')
                # Clean up
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass

        except Exception as e:
            print(f"❌ Error playing {agent_name}'s voice: {e}")

    def play_agent_voice_pyttsx3(self, agent_name: str, text: str, voice_config: dict):
        """Play agent voice using pyttsx3"""
        try:
            if not pyttsx3_available:
                return

            engine = pyttsx3.init()

            # Apply voice characteristics
            rate = int(200 * voice_config.get("speed", 1.0))
            volume = voice_config.get("volume", 0.8)

            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            # Get available voices and try to match gender/style
            voices = engine.getProperty("voices")
            voice_id = voice_config.get("voice_id", "")

            if voices:
                if "female" in voice_id and len(voices) > 0:
                    # Try to find a female voice
                    for voice in voices:
                        if "female" in voice.name.lower() or "woman" in voice.name.lower():
                            engine.setProperty("voice", voice.id)
                            break
                    else:
                        engine.setProperty("voice", voices[0].id)
                elif "male" in voice_id and len(voices) > 1:
                    # Try to find a male voice
                    for voice in voices:
                        if "male" in voice.name.lower() or "man" in voice.name.lower():
                            engine.setProperty("voice", voice.id)
                            break
                    else:
                        if len(voices) > 1:
                            engine.setProperty("voice", voices[1].id)

            print(f'🔊 Playing {agent_name}\'s voice: "{text}"')
            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print(f"❌ Error playing {agent_name}'s voice with pyttsx3: {e}")

    async def demo_all_agent_voices(self):
        """Play voices for all agents with their unique characteristics"""
        if not self.voice_system:
            print("❌ Voice system not available")
            return

        print("🎤 VoxSigil Agent Voice Demonstration")
        print("=" * 60)
        print("Playing all agent voices with their unique characteristics...")
        print()

        # Get all agent profiles
        profiles = self.voice_system.get_all_profiles()

        for i, (agent_name, profile) in enumerate(profiles.items(), 1):
            print(f"\n--- Agent {i}: {agent_name} ---")
            print(f"Personality: {', '.join(profile.personality_traits)}")
            print(f"Tone: {profile.tone}")
            print(f"Voice ID: {profile.voice_id}")
            print(
                f"Pitch: {profile.pitch:+.1f}st, Speed: {profile.speed:.1f}x, Volume: {profile.volume:.1f}"
            )

            # Get signature phrase
            signature_phrase = self.voice_system.get_signature_phrase(agent_name)

            # Get voice config
            voice_config = self.voice_system.get_tts_config(agent_name)

            # Create introduction text
            intro_text = f"Hello! I'm {agent_name}. {signature_phrase}"

            print(f'Speaking: "{intro_text}"')

            # Try Edge TTS first, fallback to pyttsx3
            if edge_tts_available:
                await self.play_agent_voice_edge_tts(agent_name, intro_text, voice_config)
            elif pyttsx3_available:
                self.play_agent_voice_pyttsx3(agent_name, intro_text, voice_config)
            else:
                print(f"🔇 No TTS engine available to play {agent_name}'s voice")

            # Wait between agents
            await asyncio.sleep(1)

        print(f"\n🎉 Demonstration complete! Played voices for {len(profiles)} agents.")

    async def demo_specific_agents(self, agent_names: list):
        """Play voices for specific agents"""
        if not self.voice_system:
            print("❌ Voice system not available")
            return

        print(f"🎤 Playing voices for: {', '.join(agent_names)}")
        print("=" * 60)

        for agent_name in agent_names:
            profile = self.voice_system.get_voice_profile(agent_name)
            if not profile:
                print(f"❌ No voice profile found for {agent_name}")
                continue

            print(f"\n--- {agent_name} ---")
            signature_phrase = self.voice_system.get_signature_phrase(agent_name)
            voice_config = self.voice_system.get_tts_config(agent_name)

            intro_text = f"Greetings! I am {agent_name}. {signature_phrase}"

            if edge_tts_available:
                await self.play_agent_voice_edge_tts(agent_name, intro_text, voice_config)
            elif pyttsx3_available:
                self.play_agent_voice_pyttsx3(agent_name, intro_text, voice_config)
            else:
                print("🔇 No TTS engine available")

            await asyncio.sleep(0.5)

    async def interactive_voice_demo(self):
        """Interactive demo where user can select agents to hear"""
        if not self.voice_system:
            print("❌ Voice system not available")
            return

        profiles = self.voice_system.get_all_profiles()
        agent_list = list(profiles.keys())

        while True:
            print("\n🎤 Agent Voice Selection")
            print("=" * 40)
            print("Available agents:")

            for i, agent_name in enumerate(agent_list, 1):
                profile = profiles[agent_name]
                print(f"{i:2d}. {agent_name} ({profile.tone})")

            print(f"{len(agent_list) + 1:2d}. Play all agents")
            print(f"{len(agent_list) + 2:2d}. Exit")

            try:
                choice = input("\nSelect agent number (or 'q' to quit): ").strip()

                if choice.lower() in ["q", "quit", "exit"]:
                    break

                choice_num = int(choice)

                if 1 <= choice_num <= len(agent_list):
                    agent_name = agent_list[choice_num - 1]
                    await self.demo_specific_agents([agent_name])
                elif choice_num == len(agent_list) + 1:
                    await self.demo_all_agent_voices()
                elif choice_num == len(agent_list) + 2:
                    break
                else:
                    print("❌ Invalid choice")

            except (ValueError, KeyboardInterrupt):
                break

        print("👋 Goodbye!")


async def main():
    """Main function"""
    player = AgentVoicePlayer()

    # Check what's available
    print("🔧 System Check:")
    print(f"Voice System: {'✅' if voice_system_available else '❌'}")
    print(f"Edge TTS: {'✅' if edge_tts_available else '❌'}")
    print(f"pyttsx3: {'✅' if pyttsx3_available else '❌'}")
    print(f"Audio Playback: {'✅' if player.pygame_available else '❌'}")
    print()

    if not voice_system_available:
        print("❌ Cannot run voice demo without voice system")
        return

    if not (edge_tts_available or pyttsx3_available):
        print("❌ No TTS engines available")
        return

    # Play all agent voices automatically
    await player.demo_all_agent_voices()

    # Optionally run interactive demo
    print("\n" + "=" * 60)
    print("Would you like to run the interactive voice selection demo?")
    choice = input("Press Enter to continue with interactive demo, or 'q' to quit: ").strip()

    if choice.lower() not in ["q", "quit"]:
        await player.interactive_voice_demo()


if __name__ == "__main__":
    asyncio.run(main())
