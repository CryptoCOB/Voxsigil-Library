"""
Advanced Neural TTS Engine for VoxSigil
Uses state-of-the-art free/open-source models for human-like speech synthesis.

Features:
- Coqui XTTS-v2: Real-time voice cloning and neural synthesis
- Bark: Transformer-based TTS with emotions and sound effects
- SpeechT5: Microsoft's neural TTS model
- Voice cloning from reference audio
- Emotion and prosody control
- Multi-speaker support
"""

import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Union

# Audio processing
try:
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Neural TTS engines
try:
    from TTS.api import TTS

    HAVE_COQUI_TTS = True
except ImportError:
    HAVE_COQUI_TTS = False

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models

    HAVE_BARK = True
except ImportError:
    HAVE_BARK = False

try:
    from datasets import load_dataset
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    HAVE_SPEECHT5 = True
except ImportError:
    HAVE_SPEECHT5 = False

# Fallback TTS
try:
    import pyttsx3

    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False

logger = logging.getLogger("AdvancedNeuralTTS")


class VoiceProfile:
    """Advanced voice profile with neural TTS parameters."""

    def __init__(
        self,
        name: str,
        engine: str = "auto",
        voice_id: Optional[str] = None,
        reference_audio: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        pitch: float = 0.0,
        energy: float = 1.0,
        personality_traits: Optional[Dict[str, float]] = None,
    ):
        self.name = name
        self.engine = engine
        self.voice_id = voice_id
        self.reference_audio = reference_audio
        self.emotion = emotion
        self.speed = speed
        self.pitch = pitch
        self.energy = energy
        self.personality_traits = personality_traits or {}


class AdvancedNeuralTTS:
    """
    Advanced Neural TTS Engine using free/open-source models.
    Provides human-like speech synthesis with voice cloning capabilities.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        models_to_load: Optional[List[str]] = None,
    ):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "voxsigil_tts")
        self.device = device or ("cuda" if torch.cuda.is_available() and HAVE_TORCH else "cpu")
        self.models_to_load = models_to_load or ["xtts", "bark", "speecht5"]

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Engine instances
        self.coqui_tts = None
        self.bark_loaded = False
        self.speecht5_processor = None
        self.speecht5_model = None
        self.speecht5_vocoder = None

        # Voice profiles
        self.voice_profiles: Dict[str, VoiceProfile] = {}

        # Thread safety
        self.lock = threading.Lock()

        # Initialize engines
        self._initialize_engines()
        self._create_default_voice_profiles()

        logger.info(f"Advanced Neural TTS initialized on {self.device}")
        logger.info(f"Available engines: {self.get_available_engines()}")

    def _initialize_engines(self):
        """Initialize available TTS engines."""

        # Initialize Coqui TTS (XTTS-v2)
        if HAVE_COQUI_TTS and "xtts" in self.models_to_load:
            try:
                logger.info("Loading Coqui XTTS-v2 model...")
                # Use the latest multilingual model
                self.coqui_tts = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True
                )
                if torch.cuda.is_available():
                    self.coqui_tts.to(self.device)
                logger.info("✅ Coqui XTTS-v2 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Coqui TTS: {e}")
                self.coqui_tts = None

        # Initialize Bark
        if HAVE_BARK and "bark" in self.models_to_load:
            try:
                logger.info("Preloading Bark models...")
                preload_models()
                self.bark_loaded = True
                logger.info("✅ Bark models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Bark: {e}")
                self.bark_loaded = False

        # Initialize SpeechT5
        if HAVE_SPEECHT5 and "speecht5" in self.models_to_load:
            try:
                logger.info("Loading SpeechT5 models...")
                self.speecht5_processor = SpeechT5Processor.from_pretrained(
                    "microsoft/speecht5_tts"
                )
                self.speecht5_model = SpeechT5ForTextToSpeech.from_pretrained(
                    "microsoft/speecht5_tts"
                )
                self.speecht5_vocoder = SpeechT5HifiGan.from_pretrained(
                    "microsoft/speecht5_hifigan"
                )

                if torch.cuda.is_available():
                    self.speecht5_model.to(self.device)
                    self.speecht5_vocoder.to(self.device)

                logger.info("✅ SpeechT5 models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SpeechT5: {e}")
                self.speecht5_processor = None

    def _create_default_voice_profiles(self):
        """Create default voice profiles for VoxSigil agents."""

        # Define voice profiles with neural TTS characteristics
        profiles = [
            VoiceProfile("Astra", "xtts", emotion="authoritative", speed=0.9, pitch=-2.0),
            VoiceProfile("Andy", "bark", emotion="cheerful", speed=1.1, pitch=2.0),
            VoiceProfile("Voxka", "xtts", emotion="philosophical", speed=0.8, pitch=-1.0),
            VoiceProfile("Warden", "bark", emotion="protective", speed=0.85, pitch=-4.0),
            VoiceProfile("Carla", "xtts", emotion="artistic", speed=1.0, pitch=3.0),
            VoiceProfile("Dave", "speecht5", emotion="analytical", speed=1.2, pitch=1.0),
            VoiceProfile("Dreamer", "bark", emotion="ethereal", speed=0.7, pitch=4.0),
            VoiceProfile("Echo", "xtts", emotion="communicative", speed=1.0, pitch=0.5),
            VoiceProfile("Evo", "bark", emotion="progressive", speed=1.1, pitch=2.5),
            VoiceProfile("Gizmo", "xtts", emotion="technical", speed=1.3, pitch=1.5),
            VoiceProfile("Oracle", "bark", emotion="mystical", speed=0.6, pitch=-1.5),
            VoiceProfile("Orion", "xtts", emotion="cosmic", speed=0.9, pitch=0.0),
            VoiceProfile("OrionApprentice", "speecht5", emotion="curious", speed=1.2, pitch=3.0),
            VoiceProfile("Phi", "xtts", emotion="mathematical", speed=0.9, pitch=-1.0),
            VoiceProfile("Sam", "bark", emotion="friendly", speed=1.0, pitch=1.0),
            VoiceProfile("Wendy", "speecht5", emotion="scholarly", speed=1.0, pitch=0.5),
        ]

        for profile in profiles:
            self.voice_profiles[profile.name] = profile

    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines."""
        engines = []

        if self.coqui_tts:
            engines.append("xtts")
        if self.bark_loaded:
            engines.append("bark")
        if self.speecht5_processor:
            engines.append("speecht5")
        if HAVE_PYTTSX3:
            engines.append("pyttsx3")

        return engines

    def synthesize_speech(
        self, text: str, voice_profile: Union[str, VoiceProfile], output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Synthesize speech using advanced neural TTS.

        Args:
            text: Text to synthesize
            voice_profile: Voice profile name or VoiceProfile object
            output_path: Output file path (optional)

        Returns:
            Path to generated audio file or None if failed
        """

        if isinstance(voice_profile, str):
            if voice_profile not in self.voice_profiles:
                logger.error(f"Voice profile '{voice_profile}' not found")
                return None
            profile = self.voice_profiles[voice_profile]
        else:
            profile = voice_profile

        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time() * 1000)
            output_path = os.path.join(self.cache_dir, f"{profile.name}_{timestamp}.wav")

        try:
            with self.lock:
                # Choose engine based on profile or availability
                engine = profile.engine
                if engine == "auto":
                    available = self.get_available_engines()
                    engine = available[0] if available else "pyttsx3"

                # Synthesize using selected engine
                if engine == "xtts" and self.coqui_tts:
                    return self._synthesize_xtts(text, profile, output_path)
                elif engine == "bark" and self.bark_loaded:
                    return self._synthesize_bark(text, profile, output_path)
                elif engine == "speecht5" and self.speecht5_processor:
                    return self._synthesize_speecht5(text, profile, output_path)
                else:
                    # Fallback to pyttsx3
                    return self._synthesize_pyttsx3(text, profile, output_path)

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None

    def _synthesize_xtts(self, text: str, profile: VoiceProfile, output_path: str) -> Optional[str]:
        """Synthesize speech using Coqui XTTS-v2."""
        try:
            # Use default speaker or reference audio
            speaker_wav = profile.reference_audio or self._get_default_speaker_audio(profile)

            # Generate audio
            audio = self.coqui_tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language="en",
                emotion=profile.emotion,
                speed=profile.speed,
            )

            # Apply pitch adjustment if needed
            if profile.pitch != 0.0:
                audio = self._adjust_pitch(audio, profile.pitch, 22050)

            # Save audio
            torchaudio.save(output_path, torch.tensor(audio).unsqueeze(0), 22050)
            logger.info(f"✅ XTTS synthesis complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}")
            return None

    def _synthesize_bark(self, text: str, profile: VoiceProfile, output_path: str) -> Optional[str]:
        """Synthesize speech using Bark."""
        try:
            # Format text with emotion and speaker
            speaker_id = self._get_bark_speaker(profile)

            # Add emotion markers for Bark
            emotion_text = self._format_bark_emotion(text, profile.emotion)

            # Generate audio
            audio_array = generate_audio(emotion_text, history_prompt=speaker_id)

            # Apply speed adjustment
            if profile.speed != 1.0:
                audio_array = librosa.effects.time_stretch(audio_array, rate=1.0 / profile.speed)

            # Apply pitch adjustment
            if profile.pitch != 0.0:
                audio_array = self._adjust_pitch(audio_array, profile.pitch, SAMPLE_RATE)

            # Save audio
            sf.write(output_path, audio_array, SAMPLE_RATE)
            logger.info(f"✅ Bark synthesis complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Bark synthesis failed: {e}")
            return None

    def _synthesize_speecht5(
        self, text: str, profile: VoiceProfile, output_path: str
    ) -> Optional[str]:
        """Synthesize speech using SpeechT5."""
        try:
            # Prepare inputs
            inputs = self.speecht5_processor(text=text, return_tensors="pt")

            # Load speaker embeddings
            speaker_embeddings = self._get_speecht5_speaker_embeddings(profile)

            # Generate speech
            with torch.no_grad():
                speech = self.speecht5_model.generate_speech(
                    inputs["input_ids"].to(self.device),
                    speaker_embeddings.to(self.device),
                    vocoder=self.speecht5_vocoder,
                )

            # Convert to numpy and apply effects
            audio_array = speech.cpu().numpy()

            # Apply speed adjustment
            if profile.speed != 1.0:
                audio_array = librosa.effects.time_stretch(audio_array, rate=1.0 / profile.speed)

            # Apply pitch adjustment
            if profile.pitch != 0.0:
                audio_array = self._adjust_pitch(audio_array, profile.pitch, 16000)

            # Save audio
            sf.write(output_path, audio_array, 16000)
            logger.info(f"✅ SpeechT5 synthesis complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"SpeechT5 synthesis failed: {e}")
            return None

    def _synthesize_pyttsx3(
        self, text: str, profile: VoiceProfile, output_path: str
    ) -> Optional[str]:
        """Fallback synthesis using pyttsx3."""
        if not HAVE_PYTTSX3:
            return None

        try:
            engine = pyttsx3.init()

            # Set voice properties
            voices = engine.getProperty("voices")
            if voices:
                # Choose voice based on profile
                voice_index = hash(profile.name) % len(voices)
                engine.setProperty("voice", voices[voice_index].id)

            # Set speed
            rate = engine.getProperty("rate")
            engine.setProperty("rate", int(rate * profile.speed))

            # Set volume
            engine.setProperty("volume", profile.energy)

            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()

            logger.info(f"✅ pyttsx3 synthesis complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return None

    def _get_default_speaker_audio(self, profile: VoiceProfile) -> str:
        """Get default speaker audio for XTTS voice cloning."""
        # In a real implementation, you'd have reference audio files
        # For now, return a placeholder path
        return "default_speaker.wav"  # This would be a real reference audio file

    def _get_bark_speaker(self, profile: VoiceProfile) -> str:
        """Get Bark speaker ID based on profile."""
        # Map personality to Bark speaker presets
        speaker_map = {
            "authoritative": "v2/en_speaker_6",
            "cheerful": "v2/en_speaker_9",
            "philosophical": "v2/en_speaker_4",
            "protective": "v2/en_speaker_7",
            "artistic": "v2/en_speaker_1",
            "analytical": "v2/en_speaker_5",
            "ethereal": "v2/en_speaker_2",
            "communicative": "v2/en_speaker_3",
            "progressive": "v2/en_speaker_8",
            "technical": "v2/en_speaker_0",
            "mystical": "v2/en_speaker_4",
            "cosmic": "v2/en_speaker_6",
            "curious": "v2/en_speaker_9",
            "mathematical": "v2/en_speaker_5",
            "friendly": "v2/en_speaker_1",
            "scholarly": "v2/en_speaker_3",
        }

        return speaker_map.get(profile.emotion, "v2/en_speaker_0")

    def _format_bark_emotion(self, text: str, emotion: str) -> str:
        """Format text with Bark emotion markers."""
        emotion_markers = {
            "cheerful": "[laughs] ",
            "sad": "[sighs] ",
            "excited": "♪ ",
            "whisper": "[whispers] ",
            "authoritative": "",
            "mystical": "♪ ",
            "ethereal": "♪ ",
        }

        marker = emotion_markers.get(emotion, "")
        return f"{marker}{text}"

    def _get_speecht5_speaker_embeddings(self, profile: VoiceProfile) -> torch.Tensor:
        """Get speaker embeddings for SpeechT5."""
        try:
            # Load default speaker embeddings dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

            # Choose speaker based on profile hash
            speaker_idx = hash(profile.name) % len(embeddings_dataset)
            speaker_embeddings = torch.tensor(embeddings_dataset[speaker_idx]["xvector"]).unsqueeze(
                0
            )

            return speaker_embeddings
        except Exception:
            # Fallback to random embeddings
            return torch.randn(1, 512)

    def _adjust_pitch(self, audio: np.ndarray, pitch_shift: float, sample_rate: int) -> np.ndarray:
        """Adjust pitch of audio."""
        try:
            if HAVE_TORCH and librosa:
                return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
            else:
                return audio
        except Exception:
            return audio
            return audio

    def clone_voice(
        self, reference_audio: str, text: str, output_path: Optional[str] = None
    ) -> Optional[str]:
        """Clone a voice from reference audio using XTTS."""
        if not self.coqui_tts:
            logger.error("Voice cloning requires Coqui XTTS")
            return None

        try:
            if not output_path:
                timestamp = int(time.time() * 1000)
                output_path = os.path.join(self.cache_dir, f"cloned_voice_{timestamp}.wav")

            # Generate audio with voice cloning
            audio = self.coqui_tts.tts(text=text, speaker_wav=reference_audio, language="en")

            # Save audio
            torchaudio.save(output_path, torch.tensor(audio).unsqueeze(0), 22050)
            logger.info(f"✅ Voice cloning complete: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return None

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities."""
        return {
            "device": self.device,
            "available_engines": self.get_available_engines(),
            "voice_profiles": list(self.voice_profiles.keys()),
            "capabilities": {
                "voice_cloning": bool(self.coqui_tts),
                "emotion_synthesis": bool(self.bark_loaded),
                "neural_tts": bool(self.coqui_tts or self.speecht5_processor),
                "real_time": True,
            },
            "dependencies": {
                "torch": HAVE_TORCH,
                "coqui_tts": HAVE_COQUI_TTS,
                "bark": HAVE_BARK,
                "speecht5": HAVE_SPEECHT5,
                "pyttsx3": HAVE_PYTTSX3,
            },
        }


# Global instance
_neural_tts_instance = None
_instance_lock = threading.Lock()


def get_neural_tts(
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> AdvancedNeuralTTS:
    """Get singleton neural TTS instance."""
    global _neural_tts_instance

    if _neural_tts_instance is None:
        with _instance_lock:
            if _neural_tts_instance is None:
                _neural_tts_instance = AdvancedNeuralTTS(cache_dir, device, models)

    return _neural_tts_instance


if __name__ == "__main__":
    # Demo usage
    print("🚀 Advanced Neural TTS Demo")
    print("=" * 50)

    # Initialize TTS engine
    tts = get_neural_tts()

    # Print system info
    info = tts.get_system_info()
    print(f"Device: {info['device']}")
    print(f"Available engines: {info['available_engines']}")
    print(f"Voice profiles: {len(info['voice_profiles'])}")
    print()

    # Test synthesis
    test_text = (
        "Hello! This is advanced neural text-to-speech synthesis using free, open-source models."
    )

    for voice_name in ["Astra", "Andy", "Voxka"]:
        print(f"🎤 Testing {voice_name}...")
        audio_path = tts.synthesize_speech(test_text, voice_name)
        if audio_path:
            print(f"✅ Generated: {audio_path}")
        else:
            print(f"❌ Failed to generate audio for {voice_name}")
        print()
