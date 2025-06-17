"""
Agent Voice System - Unique Voice Characteristics for Each Agent

This module assigns unique voice characteristics to each VoxSigil agent,
creating distinct personalities through voice modulation, pitch, speed, and tone.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Voice profile for an agent with unique characteristics."""

    agent_name: str
    voice_id: str  # TTS voice identifier
    pitch: float  # -50 to 50 (semitones)
    speed: float  # 0.5 to 2.0 (multiplier)
    volume: float  # 0.0 to 1.0
    tone: str  # emotional tone
    accent: Optional[str] = None
    personality_traits: List[str] = None
    signature_phrases: List[str] = None

    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = []
        if self.signature_phrases is None:
            self.signature_phrases = []


class AgentVoiceSystem:
    """Manages unique voice profiles for all VoxSigil agents."""

    def __init__(self):
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self._initialize_default_profiles()

    def _initialize_default_profiles(self):
        """Initialize unique voice profiles for each agent."""

        # Core Navigation Agent - Calm, steady, authoritative
        self.voice_profiles["Astra"] = VoiceProfile(
            agent_name="Astra",
            voice_id="en-us-female-1",
            pitch=0.0,
            speed=0.9,
            volume=0.8,
            tone="authoritative",
            personality_traits=["steady", "wise", "navigational"],
            signature_phrases=[
                "Charting the course...",
                "Navigation systems online.",
                "Setting coordinates for optimal path.",
            ],
        )

        # Output Composer - Cheerful, efficient, organized
        self.voice_profiles["Andy"] = VoiceProfile(
            agent_name="Andy",
            voice_id="en-us-male-1",
            pitch=5.0,
            speed=1.1,
            volume=0.85,
            tone="cheerful",
            personality_traits=["organized", "efficient", "helpful"],
            signature_phrases=[
                "Composing output package...",
                "Everything's coming together nicely!",
                "Let me organize that for you.",
            ],
        )

        # Dual Cognition Core - Deep, resonant, philosophical
        self.voice_profiles["Voxka"] = VoiceProfile(
            agent_name="Voxka",
            voice_id="en-us-male-2",
            pitch=-8.0,
            speed=0.8,
            volume=0.9,
            tone="philosophical",
            personality_traits=["deep", "contemplative", "dual-natured"],
            signature_phrases=[
                "Engaging dual cognition protocols...",
                "The voice of phi resonates...",
                "Recursive thoughts converging...",
            ],
        )

        # Guardian/Security Agent - Strong, protective, alert
        self.voice_profiles["Warden"] = VoiceProfile(
            agent_name="Warden",
            voice_id="en-us-male-3",
            pitch=-3.0,
            speed=0.95,
            volume=0.95,
            tone="protective",
            personality_traits=["vigilant", "strong", "protective"],
            signature_phrases=[
                "Security perimeter established.",
                "All systems under protection.",
                "Guardian protocols active.",
            ],
        )

        # Creative/Stylizer Agent - Artistic, expressive, dynamic
        self.voice_profiles["Carla"] = VoiceProfile(
            agent_name="Carla",
            voice_id="en-us-female-2",
            pitch=3.0,
            speed=1.05,
            volume=0.8,
            tone="artistic",
            personality_traits=["creative", "expressive", "stylish"],
            signature_phrases=[
                "Adding artistic flair...",
                "Style layers activated!",
                "Let's make this beautiful.",
            ],
        )

        # Data Processing Agent - Precise, analytical, quick
        self.voice_profiles["Dave"] = VoiceProfile(
            agent_name="Dave",
            voice_id="en-us-male-4",
            pitch=2.0,
            speed=1.2,
            volume=0.75,
            tone="analytical",
            personality_traits=["precise", "analytical", "data-focused"],
            signature_phrases=[
                "Processing data streams...",
                "Analysis complete.",
                "Numbers don't lie.",
            ],
        )

        # Dream/Vision Agent - Ethereal, imaginative, flowing
        self.voice_profiles["Dreamer"] = VoiceProfile(
            agent_name="Dreamer",
            voice_id="en-us-female-3",
            pitch=4.0,
            speed=0.85,
            volume=0.7,
            tone="ethereal",
            personality_traits=["imaginative", "visionary", "ethereal"],
            signature_phrases=[
                "Weaving dreams into reality...",
                "The vision becomes clear...",
                "In the realm of possibilities...",
            ],
        )

        # Echo/Communication Agent - Clear, communicative, resonant
        self.voice_profiles["Echo"] = VoiceProfile(
            agent_name="Echo",
            voice_id="en-us-female-4",
            pitch=1.0,
            speed=1.0,
            volume=0.85,
            tone="communicative",
            personality_traits=["clear", "resonant", "communicative"],
            signature_phrases=[
                "Echo chambers activated...",
                "Message received and amplified.",
                "Communication channels open.",
            ],
        )

        # Evolutionary Agent - Adaptive, dynamic, progressive
        self.voice_profiles["Evo"] = VoiceProfile(
            agent_name="Evo",
            voice_id="en-us-male-5",
            pitch=6.0,
            speed=1.15,
            volume=0.8,
            tone="progressive",
            personality_traits=["adaptive", "evolving", "progressive"],
            signature_phrases=[
                "Evolution in progress...",
                "Adapting to new parameters.",
                "The next iteration begins.",
            ],
        )

        # Engineering Agent - Technical, precise, systematic
        self.voice_profiles["Gizmo"] = VoiceProfile(
            agent_name="Gizmo",
            voice_id="en-us-male-6",
            pitch=-1.0,
            speed=1.05,
            volume=0.9,
            tone="technical",
            personality_traits=["technical", "systematic", "ingenious"],
            signature_phrases=[
                "Engineering solutions online...",
                "Gizmo systems operational.",
                "Technical calibration complete.",
            ],
        )

        # Wisdom/Oracle Agent - Ancient, wise, mystical
        self.voice_profiles["Oracle"] = VoiceProfile(
            agent_name="Oracle",
            voice_id="en-us-female-5",
            pitch=-5.0,
            speed=0.7,
            volume=0.9,
            tone="mystical",
            personality_traits=["wise", "ancient", "mystical"],
            signature_phrases=[
                "The Oracle speaks...",
                "Ancient wisdom flows...",
                "Truth reveals itself.",
            ],
        )

        # Space/Cosmic Agent - Vast, cosmic, stellar
        self.voice_profiles["Orion"] = VoiceProfile(
            agent_name="Orion",
            voice_id="en-us-male-7",
            pitch=-6.0,
            speed=0.9,
            volume=0.85,
            tone="cosmic",
            personality_traits=["vast", "stellar", "cosmic"],
            signature_phrases=[
                "Stellar coordinates locked...",
                "Cosmic alignment achieved.",
                "The constellation guides us.",
            ],
        )

        # Learning Agent - Curious, energetic, inquisitive
        self.voice_profiles["OrionApprentice"] = VoiceProfile(
            agent_name="OrionApprentice",
            voice_id="en-us-female-6",
            pitch=8.0,
            speed=1.25,
            volume=0.8,
            tone="curious",
            personality_traits=["curious", "learning", "energetic"],
            signature_phrases=[
                "Learning protocols engaged!",
                "Apprentice systems ready!",
                "What shall we discover next?",
            ],
        )

        # Mathematical/Abstract Agent - Logical, precise, mathematical
        self.voice_profiles["Phi"] = VoiceProfile(
            agent_name="Phi",
            voice_id="en-us-male-8",
            pitch=-2.0,
            speed=0.95,
            volume=0.8,
            tone="mathematical",
            personality_traits=["logical", "mathematical", "precise"],
            signature_phrases=[
                "Mathematical harmony achieved...",
                "Golden ratio calculations complete.",
                "Phi constants aligned.",
            ],
        )

        # Assistant Agent - Friendly, helpful, supportive
        self.voice_profiles["Sam"] = VoiceProfile(
            agent_name="Sam",
            voice_id="en-us-female-7",
            pitch=2.0,
            speed=1.1,
            volume=0.85,
            tone="friendly",
            personality_traits=["helpful", "supportive", "friendly"],
            signature_phrases=[
                "Sam here to help!",
                "Support systems activated.",
                "How can I assist you today?",
            ],
        )

        # Knowledge Agent - Scholarly, informative, wise
        self.voice_profiles["Wendy"] = VoiceProfile(
            agent_name="Wendy",
            voice_id="en-us-female-8",
            pitch=0.5,
            speed=1.0,
            volume=0.8,
            tone="scholarly",
            personality_traits=["knowledgeable", "scholarly", "informative"],
            signature_phrases=[
                "Knowledge systems online...",
                "Information synthesis complete.",
                "Wisdom archives accessed.",
            ],
        )

    def get_voice_profile(self, agent_name: str) -> Optional[VoiceProfile]:
        """Get the voice profile for a specific agent."""
        return self.voice_profiles.get(agent_name)

    def get_all_profiles(self) -> Dict[str, VoiceProfile]:
        """Get all voice profiles."""
        return self.voice_profiles.copy()

    def add_custom_profile(self, profile: VoiceProfile):
        """Add a custom voice profile for an agent."""
        self.voice_profiles[profile.agent_name] = profile

    def get_signature_phrase(self, agent_name: str) -> str:
        """Get a random signature phrase for an agent."""
        profile = self.get_voice_profile(agent_name)
        if profile and profile.signature_phrases:
            return random.choice(profile.signature_phrases)
        return f"{agent_name} online."

    def get_tts_config(self, agent_name: str) -> Dict[str, Any]:
        """Get TTS configuration for an agent."""
        profile = self.get_voice_profile(agent_name)
        if not profile:
            # Default configuration
            return {"voice_id": "en-us-neutral", "pitch": 0.0, "speed": 1.0, "volume": 0.8}

        return {
            "voice_id": profile.voice_id,
            "pitch": profile.pitch,
            "speed": profile.speed,
            "volume": profile.volume,
            "tone": profile.tone,
        }

    def speak_with_personality(
        self, agent_name: str, text: str, add_signature: bool = False
    ) -> Dict[str, Any]:
        """
        Get speech configuration with agent personality.

        Args:
            agent_name: Name of the agent
            text: Text to speak
            add_signature: Whether to add a signature phrase

        Returns:
            Dictionary with text and TTS configuration
        """
        profile = self.get_voice_profile(agent_name)

        # Prepare the text
        speech_text = text
        if add_signature and profile:
            signature = random.choice(profile.signature_phrases)
            speech_text = f"{signature} {text}"

        return {
            "text": speech_text,
            "config": self.get_tts_config(agent_name),
            "personality": profile.personality_traits if profile else [],
        }


# Global instance
_agent_voice_system = None


def get_agent_voice_system() -> AgentVoiceSystem:
    """Get the global agent voice system instance."""
    global _agent_voice_system
    if _agent_voice_system is None:
        _agent_voice_system = AgentVoiceSystem()
    return _agent_voice_system


def speak_as_agent(agent_name: str, text: str, add_signature: bool = False) -> Dict[str, Any]:
    """
    Convenience function to get speech configuration for an agent.

    Args:
        agent_name: Name of the agent
        text: Text to speak
        add_signature: Whether to add a signature phrase

    Returns:
        Dictionary with text and TTS configuration
    """
    voice_system = get_agent_voice_system()
    return voice_system.speak_with_personality(agent_name, text, add_signature)
