"""
Production Neural TTS Engine for VoxSigil
A robust, free, open-source TTS system using available models and fallbacks.

Features:
- SpeechT5: Microsoft's neural TTS (transformers-based)
- Enhanced pyttsx3 with voice processing
- Voice profiles and personality mapping
- Real-time audio processing
- Emotion and prosody control
- Multi-speaker support
"""

import logging
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Union

# Audio processing
try:
    import torch
    import torchaudio

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Transformers-based TTS
try:
    from datasets import load_dataset
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

# Audio utilities (used conditionally)
try:
    import audioop
    import wave

    import numpy as np

    HAVE_AUDIO = True
except ImportError:
    HAVE_AUDIO = False

# Fallback TTS
try:
    import pyttsx3

    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False

logger = logging.getLogger("ProductionNeuralTTS")


class VoiceProfile:
    """Enhanced voice profile for neural TTS."""

    def __init__(
        self,
        name: str,
        engine: str = "auto",
        voice_id: Optional[str] = None,
        gender: str = "neutral",
        age: str = "adult",
        emotion: str = "neutral",
        speed: float = 1.0,
        pitch: float = 0.0,
        energy: float = 1.0,
        accent: str = "neutral",
        personality_traits: Optional[Dict[str, float]] = None,
        speaking_style: str = "conversational",
    ):
        self.name = name
        self.engine = engine
        self.voice_id = voice_id
        self.gender = gender
        self.age = age
        self.emotion = emotion
        self.speed = speed
        self.pitch = pitch
        self.energy = energy
        self.accent = accent
        self.personality_traits = personality_traits or {}
        self.speaking_style = speaking_style


class ProductionNeuralTTS:
    """
    Production-ready Neural TTS Engine using free/open-source models.
    Focuses on reliability and quality with available libraries.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        prefer_neural: bool = True,
    ):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "voxsigil_production_tts")
        self.device = device or ("cuda" if torch.cuda.is_available() and HAVE_TORCH else "cpu")
        self.prefer_neural = prefer_neural

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Engine instances
        self.speecht5_processor = None
        self.speecht5_model = None
        self.speecht5_vocoder = None
        self.speecht5_speaker_embeddings = None
        self.pyttsx3_engine = None

        # Voice profiles
        self.voice_profiles: Dict[str, VoiceProfile] = {}

        # Audio processing
        self.sample_rate = 16000

        # Thread safety
        self.lock = threading.Lock()

        # Initialize engines
        self._initialize_engines()
        self._create_agent_voice_profiles()

        logger.info(f"Production Neural TTS initialized on {self.device}")
        logger.info(f"Available engines: {self.get_available_engines()}")

    def _initialize_engines(self):
        """Initialize available TTS engines."""

        # Initialize SpeechT5 (Microsoft's neural TTS)
        if HAVE_TRANSFORMERS and self.prefer_neural:
            try:
                logger.info("Loading SpeechT5 neural TTS model...")
                self.speecht5_processor = SpeechT5Processor.from_pretrained(
                    "microsoft/speecht5_tts"
                )
                self.speecht5_model = SpeechT5ForTextToSpeech.from_pretrained(
                    "microsoft/speecht5_tts"
                )
                self.speecht5_vocoder = SpeechT5HifiGan.from_pretrained(
                    "microsoft/speecht5_hifigan"
                )

                # Load speaker embeddings dataset
                embeddings_dataset = load_dataset(
                    "Matthijs/cmu-arctic-xvectors", split="validation"
                )
                self.speecht5_speaker_embeddings = torch.tensor(
                    embeddings_dataset[7306]["xvector"]
                ).unsqueeze(0)

                if self.device == "cuda" and torch.cuda.is_available():
                    self.speecht5_model = self.speecht5_model.to(self.device)
                    self.speecht5_vocoder = self.speecht5_vocoder.to(self.device)
                    self.speecht5_speaker_embeddings = self.speecht5_speaker_embeddings.to(
                        self.device
                    )

                logger.info("✅ SpeechT5 neural TTS loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SpeechT5: {e}")
                logger.info("Falling back to pyttsx3...")

        # Initialize pyttsx3 (fallback or primary)
        if HAVE_PYTTSX3:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                # Configure pyttsx3 for better quality
                self._configure_pyttsx3()
                logger.info("✅ pyttsx3 TTS initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")

    def _configure_pyttsx3(self):
        """Configure pyttsx3 for optimal quality."""
        if not self.pyttsx3_engine:
            return

        try:
            # Set reasonable defaults
            self.pyttsx3_engine.setProperty("rate", 180)  # Words per minute
            self.pyttsx3_engine.setProperty("volume", 0.9)

            # Try to select a good voice
            voices = self.pyttsx3_engine.getProperty("voices")
            if voices:
                # Prefer female voices for variety
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.name.lower():
                        self.pyttsx3_engine.setProperty("voice", voice.id)
                        break
        except Exception as e:
            logger.warning(f"Could not fully configure pyttsx3: {e}")

    def _create_agent_voice_profiles(self):
        """Create voice profiles for VoxSigil agents."""

        # Define agent voice characteristics
        agent_voices = {
            "Nova": VoiceProfile(
                name="Nova",
                gender="female",
                age="young_adult",
                emotion="friendly",
                speed=1.1,
                pitch=0.2,
                energy=1.2,
                accent="american",
                personality_traits={"confidence": 0.8, "warmth": 0.9, "intelligence": 0.9},
                speaking_style="professional",
            ),
            "Aria": VoiceProfile(
                name="Aria",
                gender="female",
                age="adult",
                emotion="calm",
                speed=0.95,
                pitch=0.0,
                energy=1.0,
                accent="british",
                personality_traits={"elegance": 0.9, "wisdom": 0.8, "patience": 0.9},
                speaking_style="refined",
            ),
            "Kai": VoiceProfile(
                name="Kai",
                gender="male",
                age="young_adult",
                emotion="energetic",
                speed=1.15,
                pitch=-0.1,
                energy=1.3,
                accent="neutral",
                personality_traits={"enthusiasm": 0.9, "curiosity": 0.8, "innovation": 0.9},
                speaking_style="casual",
            ),
            "Echo": VoiceProfile(
                name="Echo",
                gender="neutral",
                age="adult",
                emotion="mysterious",
                speed=0.9,
                pitch=-0.05,
                energy=0.9,
                accent="subtle",
                personality_traits={"mystery": 0.8, "depth": 0.9, "intrigue": 0.7},
                speaking_style="thoughtful",
            ),
            "Sage": VoiceProfile(
                name="Sage",
                gender="male",
                age="mature",
                emotion="wise",
                speed=0.85,
                pitch=-0.2,
                energy=0.8,
                accent="neutral",
                personality_traits={"wisdom": 0.95, "authority": 0.8, "guidance": 0.9},
                speaking_style="authoritative",
            ),
        }

        for agent_name, profile in agent_voices.items():
            self.voice_profiles[agent_name] = profile
            logger.info(
                f"Created voice profile for {agent_name}: {profile.speaking_style} {profile.gender}"
            )

    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines."""
        engines = []
        if self.speecht5_model is not None:
            engines.append("speecht5")
        if self.pyttsx3_engine is not None:
            engines.append("pyttsx3")
        return engines

    def add_voice_profile(self, profile: VoiceProfile):
        """Add a custom voice profile."""
        self.voice_profiles[profile.name] = profile
        logger.info(f"Added voice profile: {profile.name}")

    def _enhance_text_for_speech(self, text: str, profile: VoiceProfile) -> str:
        """Enhance text with speech markers and personality."""

        enhanced = text

        # Add personality-based speech patterns
        if profile.personality_traits.get("enthusiasm", 0) > 0.7:
            # Add excitement markers for enthusiastic characters
            enhanced = re.sub(r"([.!?])", r"\1 ", enhanced)
            enhanced = re.sub(r"\!", "!", enhanced)

        if profile.personality_traits.get("wisdom", 0) > 0.8:
            # Add thoughtful pauses for wise characters
            enhanced = re.sub(r"([.:])", r"\1... ", enhanced)

        if profile.personality_traits.get("mystery", 0) > 0.7:
            # Add dramatic pauses for mysterious characters
            enhanced = re.sub(r"([,;])", r"\1.. ", enhanced)

        # Add emotion-based modifications
        if profile.emotion == "excited":
            enhanced = enhanced.replace(".", "!")
        elif profile.emotion == "calm":
            enhanced = enhanced.replace("!", ".")

        return enhanced.strip()

    def synthesize_speech(
        self,
        text: str,
        voice_profile: Optional[Union[str, VoiceProfile]] = None,
        output_path: Optional[str] = None,
        enhance_text: bool = True,
    ) -> Optional[str]:
        """
        Synthesize speech from text using the best available engine.

        Args:
            text: Text to synthesize
            voice_profile: Voice profile name or VoiceProfile object
            output_path: Path to save audio file (optional)
            enhance_text: Whether to enhance text with personality markers

        Returns:
            Path to generated audio file if successful, None otherwise
        """

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        # Get voice profile
        if isinstance(voice_profile, str):
            profile = self.voice_profiles.get(voice_profile)
            if not profile:
                logger.warning(f"Voice profile '{voice_profile}' not found, using default")
                profile = (
                    list(self.voice_profiles.values())[0]
                    if self.voice_profiles
                    else VoiceProfile("default")
                )
        elif isinstance(voice_profile, VoiceProfile):
            profile = voice_profile
        else:
            profile = (
                list(self.voice_profiles.values())[0]
                if self.voice_profiles
                else VoiceProfile("default")
            )

        # Enhance text based on personality
        if enhance_text:
            text = self._enhance_text_for_speech(text, profile)

        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time() * 1000)
            output_path = os.path.join(self.cache_dir, f"speech_{profile.name}_{timestamp}.wav")

        # Try neural TTS first if available
        if self.speecht5_model and profile.engine != "pyttsx3":
            try:
                return self._synthesize_with_speecht5(text, profile, output_path)
            except Exception as e:
                logger.warning(f"SpeechT5 synthesis failed: {e}, falling back to pyttsx3")

        # Fall back to pyttsx3
        if self.pyttsx3_engine:
            try:
                return self._synthesize_with_pyttsx3(text, profile, output_path)
            except Exception as e:
                logger.error(f"pyttsx3 synthesis failed: {e}")

        logger.error("No working TTS engine available")
        return None

    def _synthesize_with_speecht5(self, text: str, profile: VoiceProfile, output_path: str) -> str:
        """Synthesize speech using SpeechT5."""

        with self.lock:
            # Prepare input
            inputs = self.speecht5_processor(text=text, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate speech
            with torch.no_grad():
                speech = self.speecht5_model.generate_speech(
                    inputs["input_ids"],
                    self.speecht5_speaker_embeddings,
                    vocoder=self.speecht5_vocoder,
                )

            # Move to CPU for saving
            if speech.is_cuda:
                speech = speech.cpu()

            # Apply voice profile modifications
            speech_np = speech.numpy()

            # Adjust speed
            if profile.speed != 1.0:
                import scipy.signal

                speech_np = scipy.signal.resample(speech_np, int(len(speech_np) / profile.speed))

            # Save audio
            torchaudio.save(output_path, torch.tensor(speech_np).unsqueeze(0), self.sample_rate)

            logger.info(f"Generated speech using SpeechT5: {output_path}")
            return output_path

    def _synthesize_with_pyttsx3(self, text: str, profile: VoiceProfile, output_path: str) -> str:
        """Synthesize speech using pyttsx3."""

        with self.lock:
            # Configure engine based on profile
            rate = int(180 * profile.speed)  # Base rate * speed multiplier
            self.pyttsx3_engine.setProperty("rate", max(50, min(300, rate)))

            # Adjust pitch if possible (limited in pyttsx3)
            volume = max(0.1, min(1.0, 0.9 * profile.energy))
            self.pyttsx3_engine.setProperty("volume", volume)

            # Select voice based on gender preference
            voices = self.pyttsx3_engine.getProperty("voices")
            if voices:
                if profile.gender == "female":
                    for voice in voices:
                        if any(term in voice.name.lower() for term in ["female", "zira", "hazel"]):
                            self.pyttsx3_engine.setProperty("voice", voice.id)
                            break
                elif profile.gender == "male":
                    for voice in voices:
                        if any(term in voice.name.lower() for term in ["male", "david", "mark"]):
                            self.pyttsx3_engine.setProperty("voice", voice.id)
                            break

            # Save to file
            self.pyttsx3_engine.save_to_file(text, output_path)
            self.pyttsx3_engine.runAndWait()

            logger.info(f"Generated speech using pyttsx3: {output_path}")
            return output_path

    def speak_text(
        self,
        text: str,
        voice_profile: Optional[Union[str, VoiceProfile]] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Speak text directly without saving to file.

        Args:
            text: Text to speak
            voice_profile: Voice profile to use
            blocking: Whether to wait for speech to complete

        Returns:
            True if successful, False otherwise
        """

        if not self.pyttsx3_engine:
            logger.error("No TTS engine available for direct speech")
            return False

        try:
            # Get voice profile
            if isinstance(voice_profile, str):
                profile = self.voice_profiles.get(voice_profile)
            elif isinstance(voice_profile, VoiceProfile):
                profile = voice_profile
            else:
                profile = (
                    list(self.voice_profiles.values())[0]
                    if self.voice_profiles
                    else VoiceProfile("default")
                )

            if not profile:
                profile = VoiceProfile("default")

            # Enhance text
            enhanced_text = self._enhance_text_for_speech(text, profile)

            with self.lock:
                # Configure engine
                rate = int(180 * profile.speed)
                self.pyttsx3_engine.setProperty("rate", max(50, min(300, rate)))
                volume = max(0.1, min(1.0, 0.9 * profile.energy))
                self.pyttsx3_engine.setProperty("volume", volume)

                # Speak
                self.pyttsx3_engine.say(enhanced_text)
                if blocking:
                    self.pyttsx3_engine.runAndWait()

            logger.info(f"Speaking text with voice {profile.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False

    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a voice profile."""
        profile = self.voice_profiles.get(voice_name)
        if not profile:
            return None

        return {
            "name": profile.name,
            "gender": profile.gender,
            "age": profile.age,
            "emotion": profile.emotion,
            "accent": profile.accent,
            "speaking_style": profile.speaking_style,
            "personality_traits": profile.personality_traits,
            "speed": profile.speed,
            "pitch": profile.pitch,
            "energy": profile.energy,
        }

    def list_available_voices(self) -> List[str]:
        """List all available voice profiles."""
        return list(self.voice_profiles.keys())

    def test_voice(self, voice_name: str, test_text: Optional[str] = None) -> bool:
        """Test a voice by speaking a sample text."""
        if test_text is None:
            test_text = f"Hello! I'm {voice_name}, and this is how I sound."

        return self.speak_text(test_text, voice_name, blocking=True)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize TTS engine
    tts = ProductionNeuralTTS()

    # Test each agent voice
    for voice_name in tts.list_available_voices():
        print(f"\\nTesting voice: {voice_name}")
        voice_info = tts.get_voice_info(voice_name)
        print(f"Style: {voice_info['speaking_style']}, Gender: {voice_info['gender']}")

        # Test speech
        tts.test_voice(voice_name)
        time.sleep(1)  # Brief pause between voices
