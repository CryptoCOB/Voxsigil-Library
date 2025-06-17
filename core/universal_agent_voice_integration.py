#!/usr/bin/env python3
"""
Universal Agent Voice Integration System
========================================

This system weaves voice fingerprinting and noise cancellation into ALL VoxSigil agents:
- Standard agents (Astra, Phi, Oracle, Echo)
- Music agents (Composer, Sense, Modulator)
- Specialized agents (Data, Creative, Supervisor)
- Custom agents

Provides unified voice processing across the entire agent ecosystem.
"""

import asyncio
import logging
from typing import Any, Dict, List

# Import core systems
from .universal_voice_processor import get_voice_processor

logger = logging.getLogger(__name__)


class VoiceEnabledAgentMixin:
    """Mixin to add voice processing capabilities to any agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voice_processor = get_voice_processor()
        self.agent_voice_settings = None
        self._voice_initialized = False

    async def initialize_voice(self, agent_type: str = "standard"):
        """Initialize voice processing for this agent"""
        if not self._voice_initialized:
            # Create unique voice fingerprint
            self.voice_fingerprint = self.voice_processor.create_agent_fingerprint(
                agent_id=self.get_agent_id(), agent_type=agent_type
            )

            # Get optimized voice settings
            self.agent_voice_settings = self.voice_processor.get_agent_voice_settings(
                self.get_agent_id()
            )

            self._voice_initialized = True
            logger.info(f"Voice initialized for agent {self.get_agent_id()}")

    def get_agent_id(self) -> str:
        """Get unique agent identifier - override in agent classes"""
        return (
            getattr(self, "name", None)
            or getattr(self, "agent_id", None)
            or self.__class__.__name__
        )

    async def speak_with_voice_processing(self, text: str, **kwargs) -> Dict[str, Any]:
        """Enhanced speak method with voice processing"""
        if not self._voice_initialized:
            await self.initialize_voice()

        # Adapt to current environment
        environment_adaptation = await self.voice_processor.adapt_to_environment(
            self.get_agent_id(), kwargs.get("environment_data", {})
        )

        # Apply voice processing
        voice_params = {
            **self.agent_voice_settings,
            **environment_adaptation["adapted_settings"],
            **kwargs.get("voice_params", {}),
        }

        # Generate speech with enhanced settings
        result = await self._generate_enhanced_speech(text, voice_params)

        # Add voice processing metadata
        result.update(
            {
                "agent_id": self.get_agent_id(),
                "voice_fingerprint_confidence": self.voice_fingerprint.confidence_score,
                "noise_adaptation": environment_adaptation,
                "voice_processing_enabled": True,
            }
        )

        return result

    async def _generate_enhanced_speech(
        self, text: str, voice_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate speech with enhanced voice parameters - override in subclasses"""
        # Default implementation - agents should override this
        return {
            "text": text,
            "voice_params": voice_params,
            "audio_data": None,  # Would contain actual audio
            "processing_time": 0.0,
        }

    async def handle_noisy_environment(self, audio_data, noise_level: float = None) -> Any:
        """Process audio in noisy environments"""
        if not self._voice_initialized:
            await self.initialize_voice()

        # Detect noise environment
        noise_profiles = await self.voice_processor.detect_noise_environment(audio_data)

        # Apply agent-specific noise cancellation
        clean_audio = await self.voice_processor.cancel_noise_for_agent(
            self.get_agent_id(), audio_data, noise_profiles
        )

        return clean_audio

    def get_voice_characteristics(self) -> Dict[str, Any]:
        """Get this agent's voice characteristics"""
        if not self._voice_initialized or not self.voice_fingerprint:
            return {}

        return {
            "fundamental_frequency": self.voice_fingerprint.fundamental_frequency,
            "formant_frequencies": self.voice_fingerprint.formant_frequencies,
            "breathiness": self.voice_fingerprint.breathiness_index,
            "roughness": self.voice_fingerprint.roughness_index,
            "noise_tolerance": self.voice_fingerprint.noise_tolerance,
            "clarity_threshold": self.voice_fingerprint.clarity_threshold,
        }


class EnhancedStandardAgent(VoiceEnabledAgentMixin):
    """Enhanced standard agent with voice processing"""

    def __init__(self, name: str, personality: Dict[str, Any] = None):
        self.name = name
        self.personality = personality or {}
        super().__init__()

    async def initialize_voice(self, agent_type: str = "standard"):
        await super().initialize_voice(agent_type)

    async def speak(self, text: str, **kwargs) -> Dict[str, Any]:
        """Standard agent speak method with voice processing"""
        return await self.speak_with_voice_processing(text, **kwargs)

    async def _generate_enhanced_speech(
        self, text: str, voice_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate speech for standard agents"""
        # Integration with existing TTS system
        try:
            from ..human_like_tts_enhancement import (
                AdvancedVoiceProcessor,
                EmotionalState,
                SpeechContext,
            )

            processor = AdvancedVoiceProcessor()

            # Use voice parameters to enhance TTS
            emotion = EmotionalState.NEUTRAL  # Default emotion
            context = SpeechContext.EXPLANATION  # Default context

            result = await processor.generate_human_like_speech(
                agent_name=self.name, text=text, emotion=emotion, context=context
            )

            # Apply voice fingerprint adjustments
            result["voice_params"].update(voice_params)

            return result

        except ImportError:
            # Fallback if TTS system not available
            return {
                "text": text,
                "voice_params": voice_params,
                "ssml": f'<speak><voice name="{self.name}">{text}</voice></speak>',
                "processing_time": 0.1,
            }


class EnhancedMusicAgent(VoiceEnabledAgentMixin):
    """Enhanced music agent with specialized voice processing"""

    def __init__(self, name: str, music_type: str = "general"):
        self.name = name
        self.music_type = music_type
        super().__init__()

    async def initialize_voice(self, agent_type: str = "music"):
        await super().initialize_voice(agent_type)

    async def speak(self, text: str, **kwargs) -> Dict[str, Any]:
        """Music agent speak method with enhanced voice processing"""
        return await self.speak_with_voice_processing(text, **kwargs)

    async def sing(
        self, lyrics: str, melody_data: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Enhanced singing with voice processing"""
        if not self._voice_initialized:
            await self.initialize_voice("music")

        # Special singing voice parameters
        singing_params = {
            **self.agent_voice_settings,
            "pitch_range_extended": True,
            "vibrato_enabled": True,
            "breath_control_enhanced": True,
            "resonance_boost": 1.3,
        }

        # Generate singing voice
        result = await self._generate_singing_voice(lyrics, melody_data, singing_params)

        return result

    async def _generate_enhanced_speech(
        self, text: str, voice_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate speech optimized for music agents"""
        # Music agents have enhanced voice quality
        enhanced_params = voice_params.copy()
        enhanced_params.update(
            {"pitch_accuracy": 0.95, "harmonic_richness": 1.2, "dynamic_range": 1.5}
        )

        return {
            "text": text,
            "voice_params": enhanced_params,
            "audio_quality": "high",
            "music_optimized": True,
            "processing_time": 0.15,
        }

    async def _generate_singing_voice(
        self, lyrics: str, melody_data: Dict[str, Any], singing_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate singing voice with melody"""
        return {
            "lyrics": lyrics,
            "melody_data": melody_data,
            "singing_params": singing_params,
            "voice_type": self.music_type,
            "agent_id": self.name,
            "audio_quality": "premium",
            "processing_time": 0.3,
        }

    async def harmonize_with_agents(
        self, other_agents: List[str], audio_data: Any
    ) -> Dict[str, Any]:
        """Harmonize voice with other music agents"""
        if not self._voice_initialized:
            await self.initialize_voice("music")

        # Separate voices
        separated_voices = await self.voice_processor.separate_agent_voices(
            audio_data, [self.name] + other_agents
        )

        # Create harmony based on voice characteristics
        harmony_result = {
            "primary_voice": separated_voices.get(self.name),
            "harmony_voices": {agent: separated_voices.get(agent) for agent in other_agents},
            "harmony_type": "adaptive",
            "blend_factor": 0.7,
        }

        return harmony_result


class AgentVoiceOrchestrator:
    """Orchestrates voice processing across all agent types"""

    def __init__(self):
        self.voice_processor = get_voice_processor()
        self.registered_agents: Dict[str, VoiceEnabledAgentMixin] = {}
        self.agent_groups: Dict[str, List[str]] = {
            "standard": [],
            "music": [],
            "analytical": [],
            "creative": [],
            "authoritative": [],
        }

    def register_agent(self, agent: VoiceEnabledAgentMixin, group: str = "standard"):
        """Register an agent for voice processing"""
        agent_id = agent.get_agent_id()
        self.registered_agents[agent_id] = agent

        if group in self.agent_groups:
            self.agent_groups[group].append(agent_id)

        logger.info(f"Registered agent {agent_id} in group {group}")

    async def initialize_all_agents(self):
        """Initialize voice processing for all registered agents"""
        for group, agent_ids in self.agent_groups.items():
            for agent_id in agent_ids:
                if agent_id in self.registered_agents:
                    agent = self.registered_agents[agent_id]
                    await agent.initialize_voice(group)

    async def coordinate_multi_agent_speech(
        self, agents_and_texts: Dict[str, str]
    ) -> Dict[str, Any]:
        """Coordinate speech from multiple agents to avoid conflicts"""
        results = {}

        # Process agents in sequence to avoid voice conflicts
        for agent_id, text in agents_and_texts.items():
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                result = await agent.speak(text)
                results[agent_id] = result

        return results

    async def create_agent_chorus(self, agent_ids: List[str], text: str) -> Dict[str, Any]:
        """Create harmonized speech from multiple agents"""
        if not agent_ids:
            return {}

        # Generate individual voices
        individual_voices = {}
        for agent_id in agent_ids:
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                voice_result = await agent.speak(text)
                individual_voices[agent_id] = voice_result

        # Harmonize voices (if music agents involved)
        music_agents = [aid for aid in agent_ids if aid in self.agent_groups["music"]]
        if music_agents:
            # Enhanced harmony for music agents
            harmony_data = await self._create_voice_harmony(music_agents, individual_voices)
        else:
            harmony_data = individual_voices

        return {
            "chorus_text": text,
            "participating_agents": agent_ids,
            "individual_voices": individual_voices,
            "harmony_data": harmony_data,
            "chorus_type": "music" if music_agents else "speech",
        }

    async def _create_voice_harmony(
        self, music_agent_ids: List[str], voices: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create harmonic blend of music agent voices"""
        # Get voice characteristics for harmony calculation
        characteristics = {}
        for agent_id in music_agent_ids:
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                characteristics[agent_id] = agent.get_voice_characteristics()

        # Calculate harmony intervals based on voice characteristics
        harmony_structure = self._calculate_harmony_intervals(characteristics)

        return {
            "harmony_structure": harmony_structure,
            "blend_weights": {aid: 1.0 / len(music_agent_ids) for aid in music_agent_ids},
            "harmony_type": "adaptive_agent_blend",
        }

    def _calculate_harmony_intervals(
        self, characteristics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate harmony intervals based on agent voice characteristics"""
        harmony = {}

        if not characteristics:
            return harmony

        # Use fundamental frequencies to create pleasing harmony
        base_freq = None
        for agent_id, chars in characteristics.items():
            f0 = chars.get("fundamental_frequency", 150.0)
            if base_freq is None:
                base_freq = f0
                harmony[agent_id] = 1.0  # Base frequency
            else:
                # Calculate harmonic ratio
                ratio = f0 / base_freq
                harmony[agent_id] = ratio

        return harmony

    async def handle_environment_change(self, environment_data: Dict[str, Any]):
        """Adapt all agents to environment change"""
        adaptations = {}

        for agent_id, agent in self.registered_agents.items():
            if hasattr(agent, "voice_processor"):
                adaptation = await agent.voice_processor.adapt_to_environment(
                    agent_id, environment_data
                )
                adaptations[agent_id] = adaptation

        logger.info(f"Adapted {len(adaptations)} agents to environment change")
        return adaptations

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of voice processing system"""
        return {
            "total_agents": len(self.registered_agents),
            "agent_groups": {group: len(agents) for group, agents in self.agent_groups.items()},
            "voice_processor_active": self.voice_processor is not None,
            "fingerprints_created": len(self.voice_processor.agent_fingerprints),
            "initialized_agents": sum(
                1
                for agent in self.registered_agents.values()
                if getattr(agent, "_voice_initialized", False)
            ),
        }


# Global orchestrator instance
_orchestrator = None


def get_voice_orchestrator() -> AgentVoiceOrchestrator:
    """Get the global voice orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentVoiceOrchestrator()
    return _orchestrator


# Convenience functions for integrating with existing agents
async def enhance_existing_agent(agent_instance, agent_type: str = "standard"):
    """Add voice processing to an existing agent instance"""

    # Dynamically add voice capabilities
    if not hasattr(agent_instance, "voice_processor"):
        # Add mixin capabilities
        original_class = agent_instance.__class__

        class EnhancedAgent(VoiceEnabledAgentMixin, original_class):
            pass

        # Transform the instance
        agent_instance.__class__ = EnhancedAgent

        # Initialize voice processing
        await agent_instance.initialize_voice(agent_type)

        # Register with orchestrator
        orchestrator = get_voice_orchestrator()
        orchestrator.register_agent(agent_instance, agent_type)

        logger.info(f"Enhanced existing agent {agent_instance} with voice processing")

    return agent_instance


async def demo_agent_voice_integration():
    """Demonstrate voice integration across different agent types"""
    print("🎼 Universal Agent Voice Integration Demo")
    print("=" * 60)

    orchestrator = get_voice_orchestrator()

    # Create different types of agents
    print("\n--- Creating Enhanced Agents ---")

    # Standard agents
    astra = EnhancedStandardAgent("Astra", {"analytical": True, "precise": True})
    phi = EnhancedStandardAgent("Phi", {"thoughtful": True, "logical": True})

    # Music agents
    composer = EnhancedMusicAgent("MusicComposer", "orchestral")
    voice_mod = EnhancedMusicAgent("VoiceModulator", "electronic")

    # Register agents
    orchestrator.register_agent(astra, "analytical")
    orchestrator.register_agent(phi, "analytical")
    orchestrator.register_agent(composer, "music")
    orchestrator.register_agent(voice_mod, "music")

    print(f"Registered {len(orchestrator.registered_agents)} agents")

    # Initialize all agents
    print("\n--- Initializing Voice Processing ---")
    await orchestrator.initialize_all_agents()

    # Demonstrate individual agent speech
    print("\n--- Individual Agent Speech ---")
    astra_speech = await astra.speak("I'm analyzing the data patterns now.")
    print(f"Astra: {astra_speech['text']}")
    print(f"  Voice confidence: {astra_speech.get('voice_fingerprint_confidence', 0):.2f}")

    composer_speech = await composer.speak("Let me compose a melody for you.")
    print(f"Composer: {composer_speech['text']}")
    print(f"  Music optimized: {composer_speech.get('music_optimized', False)}")

    # Demonstrate multi-agent coordination
    print("\n--- Multi-Agent Coordination ---")
    multi_speech = await orchestrator.coordinate_multi_agent_speech(
        {
            "Astra": "The analysis shows interesting trends.",
            "MusicComposer": "I can create music to represent this data.",
            "Phi": "The logical implications are significant.",
        }
    )

    for agent_id, result in multi_speech.items():
        print(
            f"{agent_id}: Generated speech with {len(result.get('voice_params', {}))} voice parameters"
        )

    # Demonstrate agent chorus
    print("\n--- Agent Chorus ---")
    chorus = await orchestrator.create_agent_chorus(
        ["Astra", "Phi", "MusicComposer"], "Welcome to VoxSigil!"
    )
    print(f"Chorus with {len(chorus['participating_agents'])} agents")
    print(f"Chorus type: {chorus['chorus_type']}")

    # Show system status
    print("\n--- System Status ---")
    status = orchestrator.get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")

    print("\n✅ Universal voice integration demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_agent_voice_integration())
