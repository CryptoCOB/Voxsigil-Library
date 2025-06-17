#!/usr/bin/env python3
"""
VoxSigil Agent Enhanced Voice Integration
========================================

Integrates the enhanced human-like TTS system with VoxSigil agents,
providing natural, contextual speech capabilities.

Key Features:
1. Automatic emotion detection from agent context
2. Dynamic voice adaptation based on agent personality
3. Contextual speech patterns (greetings, explanations, warnings)
4. Real-time voice enhancement
5. Multi-agent voice coordination
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Import enhanced TTS system
try:
    from core.human_like_tts_enhancement import EmotionalState, SpeechContext
    from engines.enhanced_human_tts_engine import (
        EnhancedTTSEngine,
        NeuralTTSConfig,
        create_enhanced_tts_engine,
    )

    ENHANCED_TTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced TTS not available: {e}")
    ENHANCED_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AgentSpeechContext:
    """Context information for agent speech"""

    agent_name: str
    situation: str = "normal"  # normal, urgent, celebration, problem
    interaction_type: str = "explanation"  # greeting, explanation, question, instruction
    emotional_state: str = "neutral"  # neutral, excited, calm, confident, concerned
    priority: str = "normal"  # low, normal, high, urgent
    audience: str = "user"  # user, agent, system, broadcast


class VoxSigilVoiceManager:
    """Manages enhanced voice capabilities for VoxSigil agents"""

    def __init__(self):
        self.enhanced_engine = None
        self.voice_history = {}  # Track recent voice interactions
        self.agent_contexts = {}  # Track agent emotional contexts

        if ENHANCED_TTS_AVAILABLE:
            self._initialize_enhanced_engine()
        else:
            logger.warning("Enhanced TTS not available, using fallback methods")

    def _initialize_enhanced_engine(self):
        """Initialize the enhanced TTS engine"""
        try:
            config = NeuralTTSConfig(
                engine="edge",
                quality="high",
                enable_emotion_detection=True,
                enable_context_analysis=True,
                enable_breathing_simulation=True,
                enable_natural_hesitations=True,
                enable_prosody_variation=True,
                enable_real_time_adaptation=True,
                enable_audio_enhancement=True,
            )

            self.enhanced_engine = create_enhanced_tts_engine(config)
            logger.info("Enhanced TTS engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced TTS engine: {e}")
            self.enhanced_engine = None

    async def speak_as_agent(
        self,
        agent_name: str,
        text: str,
        context: Optional[AgentSpeechContext] = None,
        add_signature: bool = False,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """
        Make an agent speak with enhanced human-like characteristics

        Args:
            agent_name: Name of the VoxSigil agent
            text: Text to speak
            context: Speech context information
            add_signature: Whether to add agent signature
            priority: Speech priority level

        Returns:
            Dictionary with synthesis results and metadata
        """

        if context is None:
            context = AgentSpeechContext(agent_name=agent_name)

        # Add agent signature if requested
        if add_signature:
            signature = self._generate_agent_signature(agent_name)
            text = f"{signature} {text}"

        # Detect emotional context from text and situation
        emotion = self._detect_emotional_context(text, context)
        speech_context = self._determine_speech_context(text, context)

        # Use enhanced TTS if available
        if self.enhanced_engine:
            result = await self._synthesize_enhanced(
                agent_name, text, emotion, speech_context, context
            )
        else:
            result = await self._synthesize_fallback(agent_name, text, context)

        # Update agent context history
        self._update_agent_context(agent_name, context, result)

        return result

    def _generate_agent_signature(self, agent_name: str) -> str:
        """Generate a natural agent signature"""

        signatures = {
            "Astra": "This is Astra.",
            "Phi": "Phi here.",
            "Oracle": "Oracle speaking.",
            "Echo": "Hey, it's Echo!",
            "Voxka": "Voxka reporting.",
            "Andy": "Andy at your service.",
        }

        return signatures.get(agent_name, f"This is {agent_name}.")

    def _detect_emotional_context(self, text: str, context: AgentSpeechContext) -> str:
        """Detect appropriate emotional state for the speech"""

        text_lower = text.lower()

        # Priority-based emotion detection
        if context.priority == "urgent":
            return "urgent"
        elif context.situation == "celebration":
            return "excited"
        elif context.situation == "problem":
            return "concerned"
        elif context.situation == "urgent":
            return "urgent"

        # Text-based emotion detection
        if any(word in text_lower for word in ["!", "amazing", "fantastic", "excellent", "great"]):
            return "excited"
        elif any(word in text_lower for word in ["error", "problem", "failed", "wrong"]):
            return "concerned"
        elif any(word in text_lower for word in ["welcome", "hello", "greetings"]):
            return "friendly"
        elif any(word in text_lower for word in ["definitely", "absolutely", "certain"]):
            return "confident"

        # Agent-specific default emotions
        agent_emotions = {
            "Astra": "confident",
            "Phi": "excited",
            "Oracle": "serious",
            "Echo": "happy",
            "Voxka": "neutral",
        }

        return agent_emotions.get(context.agent_name, context.emotional_state)

    def _determine_speech_context(self, text: str, context: AgentSpeechContext) -> str:
        """Determine appropriate speech context"""

        text_lower = text.lower().strip()

        # Explicit context mapping
        context_mapping = {
            "greeting": "greeting",
            "explanation": "explanation",
            "question": "question",
            "instruction": "instruction",
            "warning": "warning",
        }

        if context.interaction_type in context_mapping:
            return context_mapping[context.interaction_type]

        # Text-based context detection
        if text_lower.endswith("?") or text_lower.startswith(
            ("what", "how", "why", "when", "where", "who")
        ):
            return "question"
        elif any(word in text_lower for word in ["hello", "hi", "greetings", "welcome"]):
            return "greeting"
        elif any(word in text_lower for word in ["warning", "alert", "danger", "caution"]):
            return "warning"
        elif any(word in text_lower for word in ["please", "first", "next", "then", "step"]):
            return "instruction"
        elif len(text.split()) > 15:
            return "explanation"

        return "explanation"  # Default

    async def _synthesize_enhanced(
        self,
        agent_name: str,
        text: str,
        emotion: str,
        speech_context: str,
        context: AgentSpeechContext,
    ) -> Dict[str, Any]:
        """Synthesize using enhanced TTS engine"""

        try:
            result = await self.enhanced_engine.synthesize_human_like_speech(
                agent_name=agent_name, text=text, emotion=emotion, context=speech_context
            )

            # Add VoxSigil-specific metadata
            result.update(
                {
                    "synthesis_type": "enhanced",
                    "agent_name": agent_name,
                    "speech_context": context,
                    "natural_features_used": True,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Enhanced synthesis failed for {agent_name}: {e}")
            return await self._synthesize_fallback(agent_name, text, context)

    async def _synthesize_fallback(
        self, agent_name: str, text: str, context: AgentSpeechContext
    ) -> Dict[str, Any]:
        """Fallback synthesis using basic TTS"""

        try:
            # Try Edge TTS as fallback
            import os
            import tempfile

            import edge_tts

            # Map agents to voices
            voice_mapping = {
                "Astra": "en-US-AriaNeural",
                "Phi": "en-US-JennyNeural",
                "Oracle": "en-US-GuyNeural",
                "Echo": "en-US-AriaNeural",
                "Voxka": "en-US-DavisNeural",
            }

            voice_id = voice_mapping.get(agent_name, "en-US-AriaNeural")

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                output_path = tmp_file.name

            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(output_path)

            return {
                "success": True,
                "audio_path": output_path,
                "synthesis_type": "fallback",
                "agent_name": agent_name,
                "voice_id": voice_id,
                "file_size": os.path.getsize(output_path),
            }

        except Exception as e:
            logger.error(f"Fallback synthesis failed for {agent_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent_name,
                "synthesis_type": "failed",
            }

    def _update_agent_context(
        self, agent_name: str, context: AgentSpeechContext, result: Dict[str, Any]
    ):
        """Update agent context history for future interactions"""

        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = []

        context_entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "context": context,
            "result": result,
            "success": result.get("success", False),
        }

        self.agent_contexts[agent_name].append(context_entry)

        # Keep only recent contexts (last 10)
        if len(self.agent_contexts[agent_name]) > 10:
            self.agent_contexts[agent_name] = self.agent_contexts[agent_name][-10:]

    def get_agent_voice_status(self, agent_name: str) -> Dict[str, Any]:
        """Get current voice status for an agent"""

        if not ENHANCED_TTS_AVAILABLE:
            return {
                "agent_name": agent_name,
                "enhanced_tts": False,
                "fallback_available": True,
                "recent_interactions": 0,
            }

        recent_interactions = len(self.agent_contexts.get(agent_name, []))

        voice_info = {}
        if self.enhanced_engine:
            voice_info = await self.enhanced_engine.get_agent_voice_info(agent_name)

        return {
            "agent_name": agent_name,
            "enhanced_tts": True,
            "voice_info": voice_info,
            "recent_interactions": recent_interactions,
            "context_history": self.agent_contexts.get(agent_name, [])[-3:],  # Last 3
        }

    def list_available_features(self) -> Dict[str, Any]:
        """List all available voice features"""

        if not ENHANCED_TTS_AVAILABLE or not self.enhanced_engine:
            return {
                "enhanced_tts": False,
                "fallback_tts": True,
                "available_agents": ["Astra", "Phi", "Oracle", "Echo", "Voxka"],
                "basic_features": ["text_to_speech", "voice_selection"],
            }

        features = self.enhanced_engine.list_available_features()
        features.update(
            {
                "voxsigil_integration": True,
                "context_awareness": True,
                "agent_signatures": True,
                "emotional_adaptation": True,
            }
        )

        return features


# Global voice manager instance
_voice_manager = None


def get_voice_manager() -> VoxSigilVoiceManager:
    """Get the global voice manager instance"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoxSigilVoiceManager()
    return _voice_manager


async def agent_speak_enhanced(
    agent_name: str,
    text: str,
    situation: str = "normal",
    interaction_type: str = "explanation",
    add_signature: bool = False,
    priority: str = "normal",
) -> Dict[str, Any]:
    """
    Convenience function for enhanced agent speech

    Args:
        agent_name: Name of the VoxSigil agent
        text: Text to speak
        situation: Current situation context
        interaction_type: Type of interaction
        add_signature: Whether to add agent signature
        priority: Speech priority

    Returns:
        Speech synthesis result
    """

    voice_manager = get_voice_manager()

    context = AgentSpeechContext(
        agent_name=agent_name,
        situation=situation,
        interaction_type=interaction_type,
        priority=priority,
    )

    return await voice_manager.speak_as_agent(agent_name, text, context, add_signature, priority)


# Demo function
async def demo_enhanced_agent_voices():
    """Demonstrate enhanced agent voice capabilities"""

    print("🎤 VoxSigil Enhanced Agent Voice Demo")
    print("=" * 50)

    voice_manager = get_voice_manager()

    # Show available features
    features = voice_manager.list_available_features()
    print(f"Enhanced TTS Available: {features.get('enhanced_tts', False)}")
    print(f"Available Engines: {features.get('engines', ['fallback'])}")

    # Demo different agents and contexts
    demos = [
        {
            "agent": "Astra",
            "text": "Welcome to VoxSigil! I'm here to guide you through our cognitive landscape.",
            "situation": "normal",
            "interaction_type": "greeting",
            "add_signature": True,
        },
        {
            "agent": "Phi",
            "text": "I've discovered an interesting mathematical pattern in the data! The results show a 23.7% improvement.",
            "situation": "celebration",
            "interaction_type": "explanation",
            "add_signature": False,
        },
        {
            "agent": "Oracle",
            "text": "We must consider the deeper implications of this decision. What consequences might arise?",
            "situation": "normal",
            "interaction_type": "question",
            "add_signature": True,
        },
        {
            "agent": "Echo",
            "text": "Oh wow, that's totally amazing! The system is running so smoothly now!",
            "situation": "celebration",
            "interaction_type": "explanation",
            "add_signature": False,
        },
    ]

    for i, demo in enumerate(demos, 1):
        print(f"\n🤖 Demo {i}: {demo['agent']} - {demo['situation']} - {demo['interaction_type']}")
        print(f"Text: {demo['text']}")

        result = await agent_speak_enhanced(
            agent_name=demo["agent"],
            text=demo["text"],
            situation=demo["situation"],
            interaction_type=demo["interaction_type"],
            add_signature=demo["add_signature"],
        )

        if result.get("success"):
            print("✅ Speech synthesis successful!")
            print(f"   Type: {result.get('synthesis_type', 'unknown')}")
            print(f"   Audio file: {result.get('audio_path', 'N/A')}")
            print(f"   Natural features: {result.get('natural_features_used', False)}")
        else:
            print(f"❌ Speech synthesis failed: {result.get('error', 'Unknown error')}")

    print("\n🎉 Enhanced agent voice demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_agent_voices())
