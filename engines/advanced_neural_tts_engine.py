"""
Advanced Neural TTS Engine for VoxSigil
Supports multiple cutting-edge TTS backends with human-like quality.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import pygame

logger = logging.getLogger("VoxSigil.AdvancedTTS")

# Advanced TTS Engine Imports (with fallbacks)
try:
    import elevenlabs

    HAVE_ELEVENLABS = True
except ImportError:
    HAVE_ELEVENLABS = False
    logger.warning("ElevenLabs not available - install with: uv pip install elevenlabs")

try:
    import openai

    HAVE_OPENAI_TTS = True
except ImportError:
    HAVE_OPENAI_TTS = False
    logger.warning("OpenAI TTS not available - install with: uv pip install openai")

try:
    import azure.cognitiveservices.speech as speechsdk

    HAVE_AZURE_TTS = True
except ImportError:
    HAVE_AZURE_TTS = False
    logger.warning(
        "Azure TTS not available - install with: uv pip install azure-cognitiveservices-speech"
    )

try:
    from TTS.api import TTS as CoquiTTS

    HAVE_COQUI_TTS = True
except ImportError:
    HAVE_COQUI_TTS = False
    logger.warning("Coqui TTS not available - install with: uv pip install TTS")

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models

    HAVE_BARK = True
except ImportError:
    HAVE_BARK = False
    logger.warning("Bark TTS not available - install with: uv pip install bark")

# Fallback imports
try:
    import pyttsx3

    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False

try:
    import edge_tts

    HAVE_EDGE_TTS = True
except ImportError:
    HAVE_EDGE_TTS = False


class AdvancedTTSEngine:
    """
    Advanced Neural TTS Engine supporting multiple backends:
    - ElevenLabs (Premium AI voices)
    - OpenAI TTS (GPT-powered voices)
    - Azure Neural TTS (Microsoft)
    - Coqui TTS (Local neural)
    - Bark (Transformer-based)
    - Edge TTS (Fallback)
    - pyttsx3 (Final fallback)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize pygame for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_ready = True
        except Exception as e:
            logger.warning(f"Failed to initialize pygame audio: {e}")
            self.audio_ready = False

        # Engine priority order (best to worst)
        self.engine_priority = [
            "elevenlabs",
            "openai",
            "azure",
            "coqui",
            "bark",
            "edge_tts",
            "pyttsx3",
        ]

        # Voice mappings for different engines
        self.voice_mappings = {
            "elevenlabs": {
                "male-1": "Adam",
                "male-2": "Antoni",
                "male-3": "Arnold",
                "male-4": "Josh",
                "male-5": "Sam",
                "female-1": "Bella",
                "female-2": "Elli",
                "female-3": "Rachel",
                "female-4": "Domi",
                "female-5": "Freya",
            },
            "openai": {
                "male-1": "alloy",
                "male-2": "echo",
                "male-3": "fable",
                "male-4": "onyx",
                "female-1": "nova",
                "female-2": "shimmer",
            },
            "azure": {
                "male-1": "en-US-BrianNeural",
                "male-2": "en-US-ChristopherNeural",
                "male-3": "en-US-EricNeural",
                "male-4": "en-US-GuyNeural",
                "male-5": "en-US-RogerNeural",
                "female-1": "en-US-AriaNeural",
                "female-2": "en-US-AvaNeural",
                "female-3": "en-US-EmmaNeural",
                "female-4": "en-US-JennyNeural",
                "female-5": "en-US-MichelleNeural",
            },
        }

        # Initialize available engines
        self._init_engines()

        logger.info(f"AdvancedTTSEngine initialized with {len(self.available_engines)} engines")

    def _init_engines(self):
        """Initialize available TTS engines."""
        self.available_engines = []

        # Check ElevenLabs
        if HAVE_ELEVENLABS:
            api_key = self.config.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                try:
                    elevenlabs.set_api_key(api_key)
                    self.available_engines.append("elevenlabs")
                    logger.info("✅ ElevenLabs TTS initialized")
                except Exception as e:
                    logger.warning(f"ElevenLabs initialization failed: {e}")
            else:
                logger.warning("ElevenLabs available but no API key provided")

        # Check OpenAI TTS
        if HAVE_OPENAI_TTS:
            api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = openai.OpenAI(api_key=api_key)
                    self.available_engines.append("openai")
                    logger.info("✅ OpenAI TTS initialized")
                except Exception as e:
                    logger.warning(f"OpenAI TTS initialization failed: {e}")
            else:
                logger.warning("OpenAI TTS available but no API key provided")

        # Check Azure TTS
        if HAVE_AZURE_TTS:
            speech_key = self.config.get("azure_speech_key") or os.getenv("AZURE_SPEECH_KEY")
            region = self.config.get("azure_region") or os.getenv("AZURE_REGION", "eastus")
            if speech_key:
                try:
                    self.azure_config = speechsdk.SpeechConfig(
                        subscription=speech_key, region=region
                    )
                    self.available_engines.append("azure")
                    logger.info("✅ Azure TTS initialized")
                except Exception as e:
                    logger.warning(f"Azure TTS initialization failed: {e}")
            else:
                logger.warning("Azure TTS available but no speech key provided")

        # Check Coqui TTS (local)
        if HAVE_COQUI_TTS:
            try:
                # Initialize with a lightweight model first
                self.coqui_tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
                self.available_engines.append("coqui")
                logger.info("✅ Coqui TTS initialized")
            except Exception as e:
                logger.warning(f"Coqui TTS initialization failed: {e}")

        # Check Bark (local)
        if HAVE_BARK:
            try:
                # Preload models in background thread
                threading.Thread(target=self._preload_bark, daemon=True).start()
                self.available_engines.append("bark")
                logger.info("✅ Bark TTS initialized (loading models...)")
            except Exception as e:
                logger.warning(f"Bark TTS initialization failed: {e}")

        # Add fallback engines
        if HAVE_EDGE_TTS:
            self.available_engines.append("edge_tts")
            logger.info("✅ Edge TTS (fallback) available")

        if HAVE_PYTTSX3:
            self.available_engines.append("pyttsx3")
            logger.info("✅ pyttsx3 TTS (final fallback) available")

    def _preload_bark(self):
        """Preload Bark models in background."""
        try:
            preload_models()
            logger.info("Bark models preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload Bark models: {e}")

    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines."""
        return self.available_engines.copy()

    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about available engines."""
        info = {}

        for engine in self.available_engines:
            if engine == "elevenlabs":
                info[engine] = {
                    "name": "ElevenLabs AI",
                    "type": "cloud",
                    "quality": "premium",
                    "latency": "medium",
                    "voices": list(self.voice_mappings["elevenlabs"].keys()),
                    "features": ["emotion", "voice_cloning", "premium_quality"],
                }
            elif engine == "openai":
                info[engine] = {
                    "name": "OpenAI TTS",
                    "type": "cloud",
                    "quality": "high",
                    "latency": "low",
                    "voices": list(self.voice_mappings["openai"].keys()),
                    "features": ["fast", "natural", "gpt_powered"],
                }
            elif engine == "azure":
                info[engine] = {
                    "name": "Azure Neural TTS",
                    "type": "cloud",
                    "quality": "high",
                    "latency": "medium",
                    "voices": list(self.voice_mappings["azure"].keys()),
                    "features": ["neural", "ssml", "emotions"],
                }
            elif engine == "coqui":
                info[engine] = {
                    "name": "Coqui TTS",
                    "type": "local",
                    "quality": "good",
                    "latency": "medium",
                    "voices": ["default"],
                    "features": ["local", "no_api_key", "open_source"],
                }
            elif engine == "bark":
                info[engine] = {
                    "name": "Bark (Suno)",
                    "type": "local",
                    "quality": "good",
                    "latency": "high",
                    "voices": ["default"],
                    "features": ["local", "transformer", "music_capable"],
                }
            elif engine == "edge_tts":
                info[engine] = {
                    "name": "Edge TTS",
                    "type": "cloud",
                    "quality": "medium",
                    "latency": "low",
                    "voices": ["basic"],
                    "features": ["free", "fast", "basic"],
                }
            elif engine == "pyttsx3":
                info[engine] = {
                    "name": "pyttsx3",
                    "type": "local",
                    "quality": "basic",
                    "latency": "low",
                    "voices": ["system"],
                    "features": ["offline", "basic", "reliable"],
                }

        return info

    async def speak_async(self, text: str, voice_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Synthesize and play speech using the best available engine.

        Args:
            text: Text to synthesize
            voice_config: Voice configuration (voice_id, pitch, speed, etc.)

        Returns:
            bool: True if successful, False otherwise
        """
        if not text.strip():
            return False

        voice_config = voice_config or {}

        # Try engines in priority order
        for engine_name in self.engine_priority:
            if engine_name not in self.available_engines:
                continue

            try:
                logger.info(f"🎤 Attempting TTS with {engine_name}")
                success = await self._synthesize_with_engine(engine_name, text, voice_config)
                if success:
                    logger.info(f"✅ Successfully used {engine_name} for TTS")
                    return True
                else:
                    logger.warning(f"❌ {engine_name} failed, trying next engine...")
            except Exception as e:
                logger.error(f"❌ Error with {engine_name}: {e}")
                continue

        logger.error("❌ All TTS engines failed!")
        return False

    def speak(self, text: str, voice_config: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous wrapper for speak_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.speak_async(text, voice_config))
                    return future.result()
            else:
                return loop.run_until_complete(self.speak_async(text, voice_config))
        except Exception as e:
            logger.error(f"Error in synchronous speak: {e}")
            return False

    async def _synthesize_with_engine(
        self, engine_name: str, text: str, voice_config: Dict[str, Any]
    ) -> bool:
        """Synthesize speech with a specific engine."""
        try:
            if engine_name == "elevenlabs":
                return await self._synthesize_elevenlabs(text, voice_config)
            elif engine_name == "openai":
                return await self._synthesize_openai(text, voice_config)
            elif engine_name == "azure":
                return await self._synthesize_azure(text, voice_config)
            elif engine_name == "coqui":
                return await self._synthesize_coqui(text, voice_config)
            elif engine_name == "bark":
                return await self._synthesize_bark(text, voice_config)
            elif engine_name == "edge_tts":
                return await self._synthesize_edge_tts(text, voice_config)
            elif engine_name == "pyttsx3":
                return await self._synthesize_pyttsx3(text, voice_config)
            else:
                return False
        except Exception as e:
            logger.error(f"Engine {engine_name} synthesis error: {e}")
            return False

    async def _synthesize_elevenlabs(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with ElevenLabs."""
        if not HAVE_ELEVENLABS:
            return False

        voice_id = voice_config.get("voice_id", "male-1")
        elevenlabs_voice = self.voice_mappings["elevenlabs"].get(voice_id, "Adam")

        try:
            audio = elevenlabs.generate(
                text=text, voice=elevenlabs_voice, model="eleven_monolingual_v1"
            )

            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_file.write(audio)
                tmp_path = tmp_file.name

            success = self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return success

        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            return False

    async def _synthesize_openai(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with OpenAI TTS."""
        if not HAVE_OPENAI_TTS:
            return False

        voice_id = voice_config.get("voice_id", "male-1")
        openai_voice = self.voice_mappings["openai"].get(voice_id, "alloy")

        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1-hd",  # High quality model
                voice=openai_voice,
                input=text,
            )

            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            success = self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return success

        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            return False

    async def _synthesize_azure(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with Azure Neural TTS."""
        if not HAVE_AZURE_TTS:
            return False

        voice_id = voice_config.get("voice_id", "female-1")
        azure_voice = self.voice_mappings["azure"].get(voice_id, "en-US-AriaNeural")

        try:
            self.azure_config.speech_synthesis_voice_name = azure_voice
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.azure_config)

            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Save to temporary file and play
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(result.audio_data)
                    tmp_path = tmp_file.name

                success = self._play_audio_file(tmp_path)
                os.unlink(tmp_path)
                return success
            else:
                logger.error(f"Azure synthesis failed: {result.reason}")
                return False

        except Exception as e:
            logger.error(f"Azure TTS synthesis failed: {e}")
            return False

    async def _synthesize_coqui(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with Coqui TTS."""
        if not HAVE_COQUI_TTS:
            return False

        try:
            # Generate audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            self.coqui_tts.tts_to_file(text=text, file_path=tmp_path)

            success = self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return success

        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            return False

    async def _synthesize_bark(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with Bark."""
        if not HAVE_BARK:
            return False

        try:
            # Generate audio
            audio_array = generate_audio(text)

            # Convert to audio file and play
            import scipy.io.wavfile as wavfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            wavfile.write(tmp_path, SAMPLE_RATE, audio_array)

            success = self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return success

        except Exception as e:
            logger.error(f"Bark synthesis failed: {e}")
            return False

    async def _synthesize_edge_tts(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with Edge TTS (fallback)."""
        if not HAVE_EDGE_TTS:
            return False

        try:
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            await communicate.save(tmp_path)

            success = self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return success

        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            return False

    async def _synthesize_pyttsx3(self, text: str, voice_config: Dict[str, Any]) -> bool:
        """Synthesize with pyttsx3 (final fallback)."""
        if not HAVE_PYTTSX3:
            return False

        try:
            engine = pyttsx3.init()

            # Apply voice settings
            rate = int(voice_config.get("speed", 1.0) * 200)
            volume = voice_config.get("volume", 0.8)

            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            # Save to file and play (since pyttsx3 blocking behavior varies)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            success = self._play_audio_file(tmp_path)
            os.unlink(tmp_path)
            return success

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return False

    def _play_audio_file(self, file_path: str) -> bool:
        """Play audio file using pygame."""
        if not self.audio_ready:
            logger.warning("Audio system not ready")
            return False

        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            return True

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False

    def test_engines(self) -> Dict[str, bool]:
        """Test all available engines with a sample phrase."""
        test_text = "Hello, this is a test of the VoxSigil advanced TTS system."
        results = {}

        for engine in self.available_engines:
            logger.info(f"🧪 Testing {engine}...")
            try:
                success = asyncio.run(self._synthesize_with_engine(engine, test_text, {}))
                results[engine] = success
                status = "✅ PASS" if success else "❌ FAIL"
                logger.info(f"{status} {engine}")
            except Exception as e:
                results[engine] = False
                logger.error(f"❌ FAIL {engine}: {e}")

        return results


# Demo function
async def demo_advanced_tts():
    """Demonstrate the advanced TTS engine capabilities."""
    print("🚀 VoxSigil Advanced Neural TTS Demo")
    print("=" * 50)

    # Initialize engine
    tts = AdvancedTTSEngine()

    print(f"\n📊 Available Engines: {len(tts.get_available_engines())}")
    for engine in tts.get_available_engines():
        print(f"  ✅ {engine}")

    print("\n🔧 Engine Information:")
    engine_info = tts.get_engine_info()
    for name, info in engine_info.items():
        print(f"  🎤 {info['name']} ({info['type']}) - Quality: {info['quality']}")

    print("\n🧪 Testing Engines:")
    test_results = tts.test_engines()

    working_engines = [engine for engine, success in test_results.items() if success]
    print(f"\n✅ Working Engines: {len(working_engines)}")
    for engine in working_engines:
        print(f"  🎵 {engine}")

    if working_engines:
        print(f"\n🎤 Speaking with best engine ({working_engines[0]})...")
        await tts.speak_async(
            "Welcome to VoxSigil's advanced neural text-to-speech system. "
            "This is significantly more natural and human-like than traditional TTS engines.",
            {"voice_id": "female-1", "speed": 1.0, "volume": 0.8},
        )

    print("\n🎉 Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_advanced_tts())
