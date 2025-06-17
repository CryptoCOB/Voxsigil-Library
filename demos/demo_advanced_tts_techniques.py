#!/usr/bin/env python3
"""
Advanced TTS Techniques Demo for VoxSigil
=========================================

This script demonstrates the cutting-edge human-like TTS techniques
available in VoxSigil, from basic enhancements to neuromorphic synthesis.
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
    from core.human_like_tts_enhancement import AdvancedVoiceProcessor, EmotionalState, SpeechContext
    from core.advanced_human_tts_techniques import AdvancedHumanTTSProcessor, create_advanced_agent_profiles
    from engines.enhanced_human_tts_engine import EnhancedTTSEngine
    basic_available = True
    advanced_available = True
except ImportError as e:
    logger.warning(f"Advanced TTS modules not available: {e}")
    basic_available = False
    advanced_available = False

try:
    from core.agent_voice_system import AgentVoiceSystem
    agent_voice_available = True
except ImportError:
    logger.warning("Agent voice system not available")
    agent_voice_available = False


class TTSTechniqueDemonstrator:
    """Demonstrates various TTS enhancement techniques"""
    
    def __init__(self):
        self.basic_processor = None
        self.advanced_processor = None
        self.enhanced_engine = None
        self.agent_voice_system = None
        
        # Initialize available processors
        if basic_available:
            self.basic_processor = AdvancedVoiceProcessor()
            logger.info("✅ Basic human-like TTS processor initialized")
        
        if advanced_available:
            self.advanced_processor = AdvancedHumanTTSProcessor()
            self._setup_advanced_profiles()
            logger.info("✅ Advanced neuromorphic TTS processor initialized")
        
        try:
            self.enhanced_engine = EnhancedTTSEngine()
            logger.info("✅ Enhanced TTS engine initialized")
        except Exception as e:
            logger.warning(f"Enhanced TTS engine not available: {e}")
        
        if agent_voice_available:
            self.agent_voice_system = AgentVoiceSystem()
            logger.info("✅ Agent voice system initialized")
    
    def _setup_advanced_profiles(self):
        """Setup advanced neuromorphic profiles for demo agents"""
        if not self.advanced_processor:
            return
        
        agent_profiles = create_advanced_agent_profiles()
        for agent_name, characteristics in agent_profiles.items():
            try:
                profile = self.advanced_processor.create_neuromorphic_profile(agent_name, characteristics)
                logger.info(f"Created neuromorphic profile for {agent_name}")
                logger.info(f"  - Neural embedding size: {len(profile.neural_embedding)}")
                logger.info(f"  - Formant frequencies: {[f'{f:.0f}Hz' for f in profile.formant_frequencies]}")
                logger.info(f"  - Empathy level: {profile.empathy_level}")
                logger.info(f"  - Adaptation rate: {profile.adaptation_rate}")
            except Exception as e:
                logger.error(f"Failed to create profile for {agent_name}: {e}")
    
    async def demo_basic_enhancements(self):
        """Demonstrate basic human-like TTS enhancements"""
        print("\n" + "="*60)
        print("BASIC HUMAN-LIKE TTS ENHANCEMENTS")
        print("="*60)
        
        if not self.basic_processor:
            print("❌ Basic TTS processor not available")
            return
        
        demo_texts = [
            {
                "agent": "Astra",
                "text": "Hello! I'm Astra, your analytical assistant. I'm excited to help you understand your data.",
                "emotion": EmotionalState.EXCITED,
                "context": SpeechContext.GREETING
            },
            {
                "agent": "Phi",
                "text": "Hmm, that's a complex problem. Let me think through this step by step.",
                "emotion": EmotionalState.NEUTRAL,
                "context": SpeechContext.THINKING
            },
            {
                "agent": "Oracle",
                "text": "The key insight here is that we must consider the long-term implications.",
                "emotion": EmotionalState.SERIOUS,
                "context": SpeechContext.EXPLANATION
            }
        ]
        
        for demo in demo_texts:
            try:
                print(f"\n--- {demo['agent']} ---")
                print(f"Text: {demo['text']}")
                print(f"Emotion: {demo['emotion'].value}")
                print(f"Context: {demo['context'].value}")
                
                result = await self.basic_processor.generate_human_like_speech(
                    agent_name=demo["agent"],
                    text=demo["text"],
                    emotion=demo["emotion"],
                    context=demo["context"]
                )
                
                print(f"Generated SSML: {result['ssml'][:100]}...")
                print(f"Breathing pattern: {len(result['breathing_pattern'])} breaths")
                print(f"Prosody adjustments: Rate={result['prosody_params'].rate:.2f}, Pitch={result['prosody_params'].pitch:+.1f}st")
                
            except Exception as e:
                print(f"❌ Error generating speech for {demo['agent']}: {e}")
    
    async def demo_neuromorphic_synthesis(self):
        """Demonstrate advanced neuromorphic voice synthesis"""
        print("\n" + "="*60)
        print("ADVANCED NEUROMORPHIC VOICE SYNTHESIS")
        print("="*60)
        
        if not self.advanced_processor:
            print("❌ Advanced TTS processor not available")
            return
        
        conversation_scenarios = [
            {
                "agent": "Astra",
                "text": "I understand you're concerned about these data anomalies.",
                "context": {
                    "partner_emotion": {"concern": 0.8, "uncertainty": 0.6},
                    "flow": "building",
                    "adaptation_signal": 0.1
                }
            },
            {
                "agent": "Echo",
                "text": "Oh wow! That's actually really exciting news!",
                "context": {
                    "partner_emotion": {"excitement": 0.9, "joy": 0.7},
                    "flow": "interruption",
                    "adaptation_signal": 0.2
                }
            },
            {
                "agent": "Oracle",
                "text": "In conclusion, we must proceed with measured wisdom.",
                "context": {
                    "partner_emotion": {"contemplation": 0.5, "respect": 0.7},
                    "flow": "resolution",
                    "adaptation_signal": 0.05
                }
            }
        ]
        
        for scenario in conversation_scenarios:
            try:
                print(f"\n--- {scenario['agent']} (Neuromorphic) ---")
                print(f"Text: {scenario['text']}")
                print(f"Partner emotion: {scenario['context']['partner_emotion']}")
                print(f"Conversation flow: {scenario['context']['flow']}")
                
                result = await self.advanced_processor.synthesize_neuromorphic_speech(
                    agent_name=scenario["agent"],
                    text=scenario["text"],
                    conversation_context=scenario["context"]
                )
                
                print(f"Generated SSML: {result['ssml'][:150]}...")
                print(f"Emotional state: {result['emotional_state']}")
                print("Vocal tract parameters:")
                for param, value in result['vocal_parameters'].items():
                    if isinstance(value, float):
                        print(f"  - {param}: {value:.1f}")
                    else:
                        print(f"  - {param}: {value}")
                
                print(f"Breathing pattern: {len(result['breathing_pattern'])} events")
                for breath in result['breathing_pattern'][:2]:  # Show first 2 breaths
                    print(f"  - {breath['type']} at word {breath['position']}, duration {breath['duration']:.2f}s")
                
            except Exception as e:
                print(f"❌ Error in neuromorphic synthesis for {scenario['agent']}: {e}")
    
    async def demo_voice_characteristics(self):
        """Demonstrate unique agent voice characteristics"""
        print("\n" + "="*60)
        print("AGENT VOICE CHARACTERISTICS")
        print("="*60)
        
        if not self.advanced_processor:
            print("❌ Advanced processor not available for voice characteristics demo")
            return
        
        for agent_name, profile in self.advanced_processor.voice_profiles.items():
            print(f"\n--- {agent_name} Voice Profile ---")
            print(f"Neural embedding dimension: {len(profile.neural_embedding)}")
            print("Formant frequencies:")
            formant_labels = ["F1 (jaw)", "F2 (tongue)", "F3 (lips)", "F4 (tract)"]
            for i, freq in enumerate(profile.formant_frequencies):
                print(f"  - {formant_labels[i]}: {freq:.0f} Hz")
            
            print("Personality traits:")
            print(f"  - Empathy level: {profile.empathy_level:.2f}")
            print(f"  - Emotional stability: {profile.emotional_stability:.2f}")
            print(f"  - Adaptation rate: {profile.adaptation_rate:.3f}")
            print(f"  - Breath control: {profile.breath_control:.2f}")
            
            print("Naturalness features:")
            print(f"  - Micro-expressions: {profile.micro_expressions_audio}")
            print(f"  - Subliminal warmth: {profile.subliminal_warmth:.2f}")
            print(f"  - Conversational anchoring: {profile.conversational_anchoring}")
    
    async def demo_emotional_contagion(self):
        """Demonstrate emotional contagion system"""
        print("\n" + "="*60)
        print("EMOTIONAL CONTAGION DEMONSTRATION")
        print("="*60)
        
        if not self.advanced_processor:
            print("❌ Advanced processor not available for emotional contagion demo")
            return
        
        # Simulate a conversation where partner emotions change
        conversation_flow = [
            {
                "text": "Hi there! How are you today?",
                "partner_emotion": {"joy": 0.8, "excitement": 0.6},
                "description": "Partner is happy and excited"
            },
            {
                "text": "I'm a bit worried about the project deadline.",
                "partner_emotion": {"concern": 0.7, "stress": 0.5},
                "description": "Partner becomes concerned and stressed"
            },
            {
                "text": "Actually, I think we can solve this together!",
                "partner_emotion": {"confidence": 0.8, "optimism": 0.7},
                "description": "Partner regains confidence and optimism"
            }
        ]
        
        agent = "Echo"  # Most empathetic agent
        print(f"Using {agent} (high empathy agent) for demonstration")
        
        for i, turn in enumerate(conversation_flow):
            try:
                print(f"\nConversation Turn {i+1}:")
                print(f"Context: {turn['description']}")
                print(f"Text: {turn['text']}")
                print(f"Partner emotion: {turn['partner_emotion']}")
                
                result = await self.advanced_processor.synthesize_neuromorphic_speech(
                    agent_name=agent,
                    text=turn["text"],
                    conversation_context={
                        "partner_emotion": turn["partner_emotion"],
                        "flow": "building"
                    }
                )
                
                print(f"Agent's emotional response: {result['emotional_state']}")
                print(f"Voice adaptation applied: {result['vocal_parameters']['breath_support']:.2f}")
                
                # Show how the agent's voice adapts to partner emotions
                if i > 0:
                    print("🔄 Voice adaptation due to emotional contagion")
                  except Exception as e:
                print(f"❌ Error in emotional contagion demo: {e}")
    
    async def demo_performance_comparison(self):
        """Compare performance of different TTS enhancement levels"""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        test_text = "This is a test of the various TTS enhancement levels available in VoxSigil."
        agent = "Astra"
        
        import time
        
        # Test basic processor
        if self.basic_processor:
            start_time = time.time()
            try:
                await self.basic_processor.generate_human_like_speech(
                    agent_name=agent,
                    text=test_text
                )
                basic_time = (time.time() - start_time) * 1000
                print(f"✅ Basic TTS: {basic_time:.1f}ms")
                print("   Features: SSML, prosody, breathing, emotion detection")
            except Exception as e:
                print(f"❌ Basic TTS failed: {e}")
        
        # Test advanced processor
        if self.advanced_processor:
            start_time = time.time()
            try:
                await self.advanced_processor.synthesize_neuromorphic_speech(
                    agent_name=agent,
                    text=test_text
                )
                advanced_time = (time.time() - start_time) * 1000
                print(f"✅ Advanced TTS: {advanced_time:.1f}ms")
                print("   Features: + neuromorphic synthesis, vocal tract modeling, emotional contagion")
            except Exception as e:
                print(f"❌ Advanced TTS failed: {e}")
        
        # Test enhanced engine
        if self.enhanced_engine:
            start_time = time.time()
            try:
                await self.enhanced_engine.synthesize_human_like_speech(
                    agent_name=agent,
                    text=test_text
                )
                engine_time = (time.time() - start_time) * 1000
                print(f"✅ Enhanced Engine: {engine_time:.1f}ms")
                print("   Features: + audio enhancement, caching, multi-engine support")
            except Exception as e:
                print(f"❌ Enhanced engine failed: {e}")
    
    async def run_full_demo(self):
        """Run the complete TTS techniques demonstration"""
        print("🎤 VoxSigil Advanced TTS Techniques Demonstration")
        print("=" * 60)
        
        try:
            await self.demo_basic_enhancements()
            await self.demo_neuromorphic_synthesis()
            await self.demo_voice_characteristics()
            await self.demo_emotional_contagion()
            await self.demo_performance_comparison()
            
            print("\n" + "="*60)
            print("DEMONSTRATION COMPLETE")
            print("="*60)
            print("✅ All available TTS enhancement techniques have been demonstrated")
            print("📖 See ADVANCED_TTS_TECHNIQUES_GUIDE.md for implementation details")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")


async def main():
    """Main demo function"""
    demonstrator = TTSTechniqueDemonstrator()
    await demonstrator.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())
