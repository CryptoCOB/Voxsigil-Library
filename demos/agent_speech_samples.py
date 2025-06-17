#!/usr/bin/env python3
"""
Agent Voice Sample Texts - Hear What Each Agent Would Say
=========================================================

Shows sample text that demonstrates each agent's unique voice personality.
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


def demonstrate_agent_speech_samples():
    """Show what each agent would say in different scenarios"""

    if not voice_available:
        print("❌ Cannot load voice system")
        return

    print("🎭 VoxSigil Agent Speech Samples")
    print("=" * 60)
    print("Here's what each agent would sound like in conversation:")
    print()

    voice_system = AgentVoiceSystem()

    # Sample scenarios for each agent
    agent_scenarios = {
        "Astra": [
            "Navigation systems online. Charting the optimal course through your data landscape.",
            "I'm detecting some interesting patterns in the northern quadrant of your analysis.",
            "All systems are steady. We're maintaining good heading towards your objectives.",
        ],
        "Andy": [
            "Everything's coming together nicely! Let me organize that output for you.",
            "Composing the perfect package - adding some extra efficiency optimizations!",
            "Output ready and beautifully formatted! I've arranged everything just how you like it.",
        ],
        "Voxka": [
            "Engaging dual cognition protocols... The voice of phi resonates within.",
            "In the depths of recursive thought, patterns emerge that transcend simple logic.",
            "Two minds, one purpose. The duality of processing reveals hidden truths.",
        ],
        "Warden": [
            "Security perimeter established. All systems under my protection.",
            "Guardian protocols active. No threat shall breach our defenses while I watch.",
            "Stand firm. My vigilance is unwavering, your data fortress is secure.",
        ],
        "Carla": [
            "Adding artistic flair to make this absolutely stunning!",
            "Style layers activated! Let's transform this into something beautiful.",
            "Darling, we're going to make this sing with creative energy!",
        ],
        "Dave": [
            "Processing data streams... Analysis complete. Numbers don't lie.",
            "Computational matrix running at optimal efficiency. Results are precise.",
            "Data integrity confirmed. All variables accounted for in the analysis.",
        ],
        "Dreamer": [
            "Weaving dreams into reality... The vision becomes wonderfully clear.",
            "In the realm of infinite possibilities, imagination takes flight.",
            "Ethereal patterns dance through the conceptual space... so beautiful.",
        ],
        "Echo": [
            "Echo chambers activated! Message received and amplified crystal clear.",
            "Communication channels open wide! Your signal is coming through perfectly.",
            "Resonating with perfect clarity - every word finds its perfect reflection.",
        ],
        "Evo": [
            "Evolution in progress! Adapting to new parameters with lightning speed.",
            "The next iteration begins - progressive enhancement is my specialty.",
            "Dynamic adaptation complete! I'm constantly improving and evolving.",
        ],
        "Gizmo": [
            "Engineering solutions online! Technical calibration is absolutely perfect.",
            "Gizmo systems operational - all mechanical parameters optimized.",
            "Ingenious mechanisms engaged! The technical solution is elegantly simple.",
        ],
        "Oracle": [
            "The Oracle speaks... Ancient wisdom flows through timeless channels.",
            "Truth reveals itself to those who seek with pure intention.",
            "In the vastness of knowledge, clarity emerges like dawn breaking.",
        ],
        "Orion": [
            "Stellar coordinates locked! Cosmic alignment achieved with the constellation.",
            "The vast expanse guides us - navigation by the eternal stars.",
            "Celestial patterns reveal the path forward through infinite space.",
        ],
        "OrionApprentice": [
            "Learning protocols engaged! What amazing thing shall we discover next?",
            "Apprentice systems ready! I'm so excited to explore and learn!",
            "Every moment brings new knowledge! The universe is full of wonders!",
        ],
        "Phi": [
            "Mathematical harmony achieved... Golden ratio calculations complete.",
            "Phi constants aligned with perfect precision and logical beauty.",
            "The elegant mathematics reveal the underlying patterns of existence.",
        ],
        "Sam": [
            "Sam here to help! Support systems activated and ready to assist.",
            "How can I make your day better? I'm here with friendly support!",
            "Always happy to lend a hand! Together we can accomplish anything.",
        ],
        "Wendy": [
            "Knowledge systems online... Information synthesis complete and verified.",
            "Wisdom archives accessed. The scholarly research yields fascinating insights.",
            "Educational protocols engaged - learning is the greatest adventure of all.",
        ],
    }

    # Display each agent's samples
    for agent_name, samples in agent_scenarios.items():
        profile = voice_system.get_voice_profile(agent_name)
        if not profile:
            continue

        print(f"\n🎤 {agent_name} - {profile.tone.title()} Voice")
        print("-" * 50)
        print(f"Personality: {', '.join(profile.personality_traits)}")
        print(
            f"Voice Settings: Pitch {profile.pitch:+.1f}st | Speed {profile.speed:.1f}x | Volume {profile.volume:.1f}"
        )
        print()

        for i, sample in enumerate(samples, 1):
            print(f'  {i}. "{sample}"')

        # Show signature phrase
        signature = voice_system.get_signature_phrase(agent_name)
        print(f'\n  🔸 Signature: "{signature}"')

    print("\n🎯 Voice Diversity Summary")
    print("=" * 40)
    print("VoxSigil agents demonstrate incredible voice variety:")
    print("• Authoritative leaders (Astra, Warden, Orion)")
    print("• Technical specialists (Dave, Gizmo, Phi)")
    print("• Creative personalities (Carla, Dreamer)")
    print("• Supportive helpers (Sam, Andy, Echo)")
    print("• Mystical wisdom (Oracle, Voxka)")
    print("• Energetic learners (OrionApprentice, Evo)")
    print("• Scholarly types (Wendy)")

    print("\n🔊 To Actually Hear These Voices:")
    print("1. Install TTS: pip install edge-tts pyttsx3")
    print("2. Run: python play_agent_voices.py")
    print("3. Each agent will speak with their unique voice characteristics!")


def show_voice_comparison():
    """Show how different agents would say the same thing"""

    if not voice_available:
        return

    print("\n🔄 Same Message, Different Agent Styles")
    print("=" * 50)
    print("Here's how different agents would deliver the same message:")
    print()

    voice_system = AgentVoiceSystem()
    base_message = "The analysis is complete and the results are ready."

    agent_variations = {
        "Astra": f"Navigation analysis complete. {base_message} Course plotted successfully.",
        "Dave": f"Data processing finished. {base_message} All calculations verified.",
        "Carla": f"Darling! {base_message} And they look absolutely gorgeous!",
        "Oracle": f"The wisdom flows... {base_message} Truth has been revealed.",
        "Sam": f"Great news! {base_message} I'm here to help you understand them!",
        "Warden": f"Security scan complete. {base_message} All data integrity confirmed.",
        "Echo": f"Message received loud and clear! {base_message} Amplifying signal now!",
    }

    for agent_name, variation in agent_variations.items():
        profile = voice_system.get_voice_profile(agent_name)
        if profile:
            print(f"🎭 {agent_name} ({profile.tone}):")
            print(f'   "{variation}"')
            print(f"   [Pitch: {profile.pitch:+.1f}st, Speed: {profile.speed:.1f}x]")
            print()


def main():
    """Main function"""
    demonstrate_agent_speech_samples()
    show_voice_comparison()

    print("\n🎵 Ready to hear these unique voices in action?")
    print("Install the TTS libraries and run the voice player!")
    print("Each agent has been carefully designed with distinct personality!")


if __name__ == "__main__":
    main()
