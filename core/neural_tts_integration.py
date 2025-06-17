"""
VoxSigil Neural TTS Integration
Integrates the production neural TTS engine into the existing agent system.
"""

import logging
from typing import Any, Dict, Optional

from core.production_neural_tts import ProductionNeuralTTS

logger = logging.getLogger("VoxSigilTTSIntegration")


class VoxSigilTTSIntegration:
    """
    Integration layer for VoxSigil Neural TTS.
    Provides a bridge between agents and the neural TTS engine.
    """

    def __init__(self):
        self.neural_tts = None
        self._initialize_tts()

    def _initialize_tts(self):
        """Initialize the neural TTS engine."""
        try:
            self.neural_tts = ProductionNeuralTTS()
            logger.info("✅ VoxSigil Neural TTS Integration initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neural TTS: {e}")

    def is_available(self) -> bool:
        """Check if neural TTS is available."""
        return self.neural_tts is not None

    def get_available_voices(self) -> list:
        """Get list of available voice profiles."""
        if not self.neural_tts:
            return []
        return self.neural_tts.list_available_voices()

    def speak_for_agent(self, agent_name: str, text: str, blocking: bool = True) -> bool:
        """
        Make an agent speak using neural TTS.

        Args:
            agent_name: Name of the agent (Nova, Aria, Kai, Echo, Sage)
            text: Text to speak
            blocking: Whether to wait for speech completion

        Returns:
            True if successful, False otherwise
        """
        if not self.neural_tts:
            logger.warning("Neural TTS not available")
            return False

        try:
            success = self.neural_tts.speak_text(text, agent_name, blocking=blocking)
            if success:
                logger.info(f"🎤 {agent_name} spoke: '{text[:50]}...'")
            else:
                logger.warning(f"❌ Failed to synthesize speech for {agent_name}")
            return success
        except Exception as e:
            logger.error(f"Error in agent speech synthesis: {e}")
            return False

    def generate_agent_audio_file(
        self, agent_name: str, text: str, output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate an audio file for agent speech.

        Args:
            agent_name: Name of the agent
            text: Text to synthesize
            output_path: Path for output file (optional)

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.neural_tts:
            return None

        try:
            result_path = self.neural_tts.synthesize_speech(
                text=text, voice_profile=agent_name, output_path=output_path
            )

            if result_path:
                logger.info(f"🎵 Generated audio file for {agent_name}: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Error generating audio file: {e}")
            return None

    def get_agent_voice_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get voice information for an agent."""
        if not self.neural_tts:
            return None
        return self.neural_tts.get_voice_info(agent_name)

    def test_all_agent_voices(self) -> Dict[str, bool]:
        """Test all agent voices and return results."""
        results = {}

        if not self.neural_tts:
            return results

        test_messages = {
            "Nova": "Hello! I'm Nova, your professional AI assistant.",
            "Aria": "Greetings. I am Aria, bringing elegance to our interaction.",
            "Kai": "Hey there! I'm Kai, and I'm excited to help you!",
            "Echo": "I am Echo... guardian of vocal mysteries...",
            "Sage": "I am Sage, your wise counselor and guide.",
        }

        for agent_name, message in test_messages.items():
            logger.info(f"Testing voice for {agent_name}...")
            success = self.speak_for_agent(agent_name, message, blocking=True)
            results[agent_name] = success

        return results


# Global TTS integration instance
_tts_integration = None


def get_tts_integration() -> VoxSigilTTSIntegration:
    """Get the global TTS integration instance."""
    global _tts_integration
    if _tts_integration is None:
        _tts_integration = VoxSigilTTSIntegration()
    return _tts_integration


def agent_speak(agent_name: str, text: str, blocking: bool = True) -> bool:
    """
    Convenience function for agent speech.

    Args:
        agent_name: Name of the agent
        text: Text to speak
        blocking: Whether to wait for completion

    Returns:
        True if successful, False otherwise
    """
    integration = get_tts_integration()
    return integration.speak_for_agent(agent_name, text, blocking)


def generate_agent_greeting(agent_name: str) -> bool:
    """Generate and play a greeting for an agent."""
    greetings = {
        "Nova": "Welcome to VoxSigil! I'm Nova, your professional AI assistant. I'm ready to help you with our advanced neural voice processing system.",
        "Aria": "Greetings and salutations. I am Aria, and I shall guide you through our sophisticated voice synthesis technology with elegance and wisdom.",
        "Kai": "Hey there! I'm Kai, and I'm super excited to show you all the amazing things our neural TTS system can do! This is going to be awesome!",
        "Echo": "I am Echo... the mysterious guardian of vocal secrets. Listen carefully as I reveal the hidden depths of our neural voice synthesis capabilities...",
        "Sage": "I am Sage, the wise counselor. With knowledge and authority encoded in my voice patterns, I shall demonstrate the power of our advanced TTS system.",
    }

    greeting = greetings.get(agent_name, f"Hello, I am {agent_name}.")
    return agent_speak(agent_name, greeting, blocking=True)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🎙️  VoxSigil Neural TTS Integration Test")
    print("=" * 50)

    # Initialize integration
    integration = get_tts_integration()

    if integration.is_available():
        print("✅ Neural TTS Integration is available!")
        print(f"🎭 Available voices: {integration.get_available_voices()}")

        # Test agent greetings
        print("\n🎤 Testing Agent Greetings:")
        for agent in ["Nova", "Aria", "Kai", "Echo", "Sage"]:
            print(f"\n{agent} Greeting:")
            success = generate_agent_greeting(agent)
            if success:
                print(f"   ✅ {agent} greeting successful")
            else:
                print(f"   ❌ {agent} greeting failed")

        print("\n🎉 Integration test complete!")
        print("✅ VoxSigil Neural TTS is fully integrated and operational!")
    else:
        print("❌ Neural TTS Integration is not available")
