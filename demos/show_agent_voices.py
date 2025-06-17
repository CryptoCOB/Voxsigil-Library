#!/usr/bin/env python3
"""
Agent Voice Characteristics Display
==================================

Shows the voice characteristics of all VoxSigil agents without requiring TTS libraries.
"""

import sys
from pathlib import Path

# Add the VoxSigil library to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.agent_voice_system import AgentVoiceSystem

    voice_available = True
except ImportError as e:
    print(f"❌ Voice system not available: {e}")
    voice_available = False


def display_agent_voices():
    """Display all agent voice characteristics"""

    if not voice_available:
        print("❌ Cannot load voice system")
        return

    print("🎤 VoxSigil Agent Voice Characteristics")
    print("=" * 70)
    print("Here are all the unique agent voices in VoxSigil:")
    print()

    voice_system = AgentVoiceSystem()
    all_profiles = voice_system.get_all_profiles()

    # Group agents by tone/personality for better presentation
    agent_groups = {
        "Leadership & Authority": [],
        "Technical & Analytical": [],
        "Creative & Artistic": [],
        "Wisdom & Mystical": [],
        "Supportive & Communication": [],
        "Specialized": [],
    }
    # Categorize agents
    for name, profile in all_profiles.items():
        tone = profile.tone.lower()

        if any(word in tone for word in ["authoritative", "protective", "cosmic"]):
            agent_groups["Leadership & Authority"].append((name, profile))
        elif any(word in tone for word in ["analytical", "technical", "mathematical"]):
            agent_groups["Technical & Analytical"].append((name, profile))
        elif any(word in tone for word in ["artistic", "creative", "ethereal"]):
            agent_groups["Creative & Artistic"].append((name, profile))
        elif any(word in tone for word in ["mystical", "philosophical", "scholarly"]):
            agent_groups["Wisdom & Mystical"].append((name, profile))
        elif any(word in tone for word in ["friendly", "communicative", "helpful"]):
            agent_groups["Supportive & Communication"].append((name, profile))
        else:
            agent_groups["Specialized"].append((name, profile))

    # Display each group
    for group_name, agents in agent_groups.items():
        if not agents:
            continue

        print(f"\n🔸 {group_name}")
        print("-" * 50)

        for name, profile in agents:
            # Get signature phrase
            signature = voice_system.get_signature_phrase(name)

            print(f"\n  👤 {name}")
            print(f"     Tone: {profile.tone.title()}")
            print(f"     Voice: {profile.voice_id}")
            print(
                f"     Settings: Pitch {profile.pitch:+.1f}st | Speed {profile.speed:.1f}x | Volume {profile.volume:.1f}"
            )
            print(f"     Personality: {', '.join(profile.personality_traits)}")
            print(f'     Signature: "{signature}"')

            # Show what they would say
            intro_text = f"Hello! I'm {name}. {signature}"
            print(f'     Would say: "{intro_text}"')

    print("\n🎯 Summary")
    print("=" * 40)
    print(f"Total agents: {len(all_profiles)}")
    print("Voice variety:")
    print("  • Pitch range: -8.0st to +8.0st")
    print("  • Speed range: 0.7x to 1.25x")
    print("  • Different tones: authoritative, cheerful, philosophical, protective, etc.")
    print("  • Unique personality traits for each agent")
    print("  • Custom signature phrases")

    print("\n📢 To hear these voices:")
    print("  1. Install TTS libraries: pip install edge-tts pyttsx3 pygame")
    print("  2. Run: python play_agent_voices.py")
    print("  3. Or use the quick test: python quick_agent_voice_test.py")


def show_voice_technology():
    """Show the voice technology details"""

    print("\n🔬 Voice Technology Features")
    print("=" * 50)
    print("VoxSigil uses advanced voice synthesis with:")
    print()

    print("📡 TTS Engine Support:")
    print("  • Edge TTS (Microsoft Neural Voices)")
    print("  • pyttsx3 (Local system voices)")
    print("  • Custom voice characteristics per agent")
    print()

    print("🎛️ Voice Modulation:")
    print("  • Pitch adjustment (-50 to +50 semitones)")
    print("  • Speed control (0.5x to 2.0x)")
    print("  • Volume normalization")
    print("  • Emotional tone mapping")
    print()

    print("🧠 Personality Integration:")
    print("  • Each agent has unique personality traits")
    print("  • Custom signature phrases")
    print("  • Tone matches agent function")
    print("  • Voice characteristics reflect agent role")
    print()

    print("🔮 Advanced Features (Available):")
    print("  • SSML generation for natural speech")
    print("  • Emotional state detection")
    print("  • Breathing simulation")
    print("  • Neuromorphic voice synthesis")
    print("  • Voice fingerprinting")
    print("  • Noise cancellation")


def main():
    """Main function"""
    display_agent_voices()
    show_voice_technology()

    print("\n🎤 Ready to hear the voices?")
    print("Install the TTS libraries and run the voice player scripts!")


if __name__ == "__main__":
    main()
