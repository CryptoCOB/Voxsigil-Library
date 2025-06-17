#!/usr/bin/env python3
"""
Advanced TTS Techniques Demo for VoxSigil - Simple Version
==========================================================

This script demonstrates the cutting-edge human-like TTS techniques
available in VoxSigil.
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
    from core.human_like_tts_enhancement import (
        AdvancedVoiceProcessor,
        EmotionalState,
        SpeechContext,
    )

    basic_available = True
    logger.info("✅ Basic human-like TTS available")
except ImportError as e:
    logger.warning(f"Basic TTS not available: {e}")
    basic_available = False

try:
    from core.advanced_human_tts_techniques import (
        AdvancedHumanTTSProcessor,
        create_advanced_agent_profiles,
    )

    advanced_available = True
    logger.info("✅ Advanced neuromorphic TTS available")
except ImportError as e:
    logger.warning(f"Advanced TTS not available: {e}")
    advanced_available = False

try:
    from engines.enhanced_human_tts_engine import EnhancedTTSEngine

    enhanced_available = True
    logger.info("✅ Enhanced TTS engine available")
except ImportError as e:
    logger.warning(f"Enhanced engine not available: {e}")
    enhanced_available = False


async def demo_basic_tts():
    """Demonstrate basic human-like TTS"""
    if not basic_available:
        print("❌ Basic TTS not available")
        return

    print("\n" + "=" * 50)
    print("BASIC HUMAN-LIKE TTS DEMONSTRATION")
    print("=" * 50)

    processor = AdvancedVoiceProcessor()

    demos = [
        {
            "agent": "Astra",
            "text": "Hello! I'm excited to help you analyze your data patterns.",
            "emotion": EmotionalState.EXCITED,
            "context": SpeechContext.GREETING,
        },
        {
            "agent": "Phi",
            "text": "Let me think through this complex problem step by step.",
            "emotion": EmotionalState.NEUTRAL,
            "context": SpeechContext.THINKING,
        },
    ]

    for demo in demos:
        try:
            print(f"\n--- {demo['agent']} ---")
            print(f"Text: {demo['text']}")

            result = await processor.generate_human_like_speech(
                agent_name=demo["agent"],
                text=demo["text"],
                emotion=demo["emotion"],
                context=demo["context"],
            )

            print(f"✅ Generated SSML with {len(result['breathing_pattern'])} breath events")
            print(
                f"Rate: {result['prosody_params'].rate:.2f}x, Pitch: {result['prosody_params'].pitch:+.1f}st"
            )

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_advanced_tts():
    """Demonstrate advanced neuromorphic TTS"""
    if not advanced_available:
        print("❌ Advanced TTS not available")
        return

    print("\n" + "=" * 50)
    print("ADVANCED NEUROMORPHIC TTS DEMONSTRATION")
    print("=" * 50)

    processor = AdvancedHumanTTSProcessor()

    # Setup profiles
    agent_profiles = create_advanced_agent_profiles()
    for agent_name, characteristics in agent_profiles.items():
        processor.create_neuromorphic_profile(agent_name, characteristics)
        print(f"Created neuromorphic profile for {agent_name}")

    # Demo neuromorphic synthesis
    scenarios = [
        {
            "agent": "Astra",
            "text": "I understand your concerns about these data anomalies.",
            "context": {"partner_emotion": {"concern": 0.8}, "flow": "building"},
        },
        {
            "agent": "Echo",
            "text": "Oh wow! That's really exciting news!",
            "context": {"partner_emotion": {"excitement": 0.9}, "flow": "interruption"},
        },
    ]

    for scenario in scenarios:
        try:
            print(f"\n--- {scenario['agent']} (Neuromorphic) ---")
            print(f"Text: {scenario['text']}")

            result = await processor.synthesize_neuromorphic_speech(
                agent_name=scenario["agent"],
                text=scenario["text"],
                conversation_context=scenario["context"],
            )

            print("✅ Generated neuromorphic speech with:")
            print(f"  - Emotional state: {list(result['emotional_state'].keys())[:3]}")
            print(f"  - Vocal tract modeling: F1={result['vocal_parameters']['formant_f1']:.0f}Hz")
            print(f"  - Breathing events: {len(result['breathing_pattern'])}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_voice_profiles():
    """Show unique agent voice characteristics"""
    if not advanced_available:
        print("❌ Advanced TTS not available for voice profiles demo")
        return

    print("\n" + "=" * 50)
    print("AGENT VOICE PROFILES")
    print("=" * 50)

    processor = AdvancedHumanTTSProcessor()
    agent_profiles = create_advanced_agent_profiles()

    for agent_name, characteristics in agent_profiles.items():
        profile = processor.create_neuromorphic_profile(agent_name, characteristics)

        print(f"\n--- {agent_name} ---")
        print(f"Neural embedding: {len(profile.neural_embedding)} dimensions")
        print(f"Formant F1 (jaw): {profile.formant_frequencies[0]:.0f}Hz")
        print(f"Formant F2 (tongue): {profile.formant_frequencies[1]:.0f}Hz")
        print(f"Empathy level: {profile.empathy_level:.2f}")
        print(f"Adaptation rate: {profile.adaptation_rate:.3f}")
        print(f"Breath control: {profile.breath_control:.2f}")


async def demo_performance():
    """Compare performance of different TTS levels"""
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    test_text = "This is a performance test of VoxSigil TTS enhancement levels."
    agent = "Astra"

    import time

    # Test basic TTS
    if basic_available:
        processor = AdvancedVoiceProcessor()
        start_time = time.time()
        try:
            await processor.generate_human_like_speech(agent_name=agent, text=test_text)
            basic_time = (time.time() - start_time) * 1000
            print(f"✅ Basic TTS: {basic_time:.1f}ms")
        except Exception as e:
            print(f"❌ Basic TTS failed: {e}")

    # Test advanced TTS
    if advanced_available:
        processor = AdvancedHumanTTSProcessor()
        agent_profiles = create_advanced_agent_profiles()
        processor.create_neuromorphic_profile(agent, agent_profiles[agent])

        start_time = time.time()
        try:
            await processor.synthesize_neuromorphic_speech(agent_name=agent, text=test_text)
            advanced_time = (time.time() - start_time) * 1000
            print(f"✅ Advanced TTS: {advanced_time:.1f}ms")
        except Exception as e:
            print(f"❌ Advanced TTS failed: {e}")

    # Test enhanced engine
    if enhanced_available:
        engine = EnhancedTTSEngine()
        start_time = time.time()
        try:
            await engine.synthesize_human_like_speech(agent_name=agent, text=test_text)
            engine_time = (time.time() - start_time) * 1000
            print(f"✅ Enhanced Engine: {engine_time:.1f}ms")
        except Exception as e:
            print(f"❌ Enhanced Engine failed: {e}")


async def main():
    """Run all demonstrations"""
    print("🎤 VoxSigil Advanced TTS Techniques Demonstration")
    print("=" * 60)

    await demo_basic_tts()
    await demo_advanced_tts()
    await demo_voice_profiles()
    await demo_performance()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("✅ All available TTS enhancement techniques demonstrated")
    print("📖 See ADVANCED_TTS_TECHNIQUES_GUIDE.md for implementation details")


if __name__ == "__main__":
    asyncio.run(main())
