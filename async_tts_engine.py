"""
DEPRECATED: All async TTS (text-to-speech) requests should be routed through UnifiedVantaCore's async bus.
Do not instantiate or use this engine directly. Use UnifiedVantaCore and its async bus for all orchestration.
"""

# Async Text-to-Speech (TTS) Engine for Vanta
# Handles text-to-speech synthesis asynchronously with multiple engine support

import asyncio
import glob
import hashlib
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Vanta.core.UnifiedAsyncBus import (
    MessageType,  # Import MessageType for async bus integration
)

# TTS Engine Dependencies
try:
    import edge_tts

    HAVE_EDGE_TTS = True
except ImportError:
    HAVE_EDGE_TTS = False
    edge_tts = None

try:
    import pyttsx3

    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False
    pyttsx3 = None

logger = logging.getLogger("Vanta.AsyncTTS")


@dataclass
class TTSConfig:
    """Configuration for the Async TTS Engine."""

    engine: str = "edge"
    voice: str = "en-US-AriaNeural"
    rate: int = 150
    volume: float = 0.9
    output_format: str = "mp3"
    max_concurrent_synthesis: int = 3
    synthesis_timeout: float = 30.0
    cache_enabled: bool = True
    cache_max_size: int = 100


@dataclass
class SynthesisRequest:
    """Request for text synthesis."""

    text: str
    voice: Optional[str] = None
    rate: Optional[int] = None
    volume: Optional[float] = None
    priority: int = 5


@dataclass
class SynthesisResult:
    """Result from text synthesis."""

    success: bool
    text: str
    audio_path: Optional[str] = None
    duration_ms: float = 0.0
    engine_used: str = ""
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AsyncTTSEngine:
    """Async Text-to-Speech Engine for Vanta."""

    COMPONENT_NAME = "async_tts_engine"

    def __init__(self, vanta_core, config: Optional[TTSConfig] = None):
        """Initialize the Async TTS Engine."""
        self.vanta_core = vanta_core
        self.config = config or TTSConfig()

        # Engine initialization
        self._engines = {}
        self._current_engine = None
        self._initialize_engines()

        # Queue management
        self._synthesis_queue: asyncio.Queue = asyncio.Queue()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_synthesis)

        # Caching
        self._audio_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
        }

        # Background processing
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Register with VantaCore
        self.vanta_core.register_component(
            self.COMPONENT_NAME,
            self,
            {
                "type": "async_tts",
                "engine": self.config.engine,
                "voice": self.config.voice,
                "rate": self.config.rate,
                "engines_available": list(self._engines.keys()),
            },
        )
        logger.info(f"{self.COMPONENT_NAME} registered with VantaCore")

        # === Unified Async Bus Integration ===
        # Register with UnifiedVantaCore's async bus and subscribe handler
        if hasattr(self.vanta_core, "async_bus"):
            self.vanta_core.async_bus.register_component("tts_engine")
            self.vanta_core.async_bus.subscribe(
                "tts_engine",
                MessageType.TEXT_TO_SPEECH,
                self.handle_speech_request,
            )
            logger.info(
                "tts_engine registered and subscribed to async bus (TEXT_TO_SPEECH)"
            )
        else:
            logger.warning(
                "UnifiedVantaCore async bus not available; async bus integration skipped."
            )

        # Start background processing
        self._start_background_processing()

    def _initialize_engines(self):
        """Initialize available TTS engines."""
        logger.info("Initializing TTS engines...")

        # Edge TTS
        if HAVE_EDGE_TTS:
            self._engines["edge"] = {
                "available": True,
                "module": edge_tts,
                "supports_async": True,
                "supports_ssml": True,
            }
            logger.info("Edge TTS engine available")

        # pyttsx3
        if HAVE_PYTTSX3 and pyttsx3:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.config.rate)
                engine.setProperty("volume", self.config.volume)
                self._engines["pyttsx3"] = {
                    "available": True,
                    "engine": engine,
                    "supports_async": False,
                    "supports_ssml": False,
                }
                logger.info("pyttsx3 TTS engine available")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")

        # System TTS (Windows SAPI)
        self._engines["system"] = {
            "available": True,
            "supports_async": True,
            "supports_ssml": False,
        }
        logger.info("System TTS engine available")

        # Set current engine
        if self.config.engine in self._engines:
            self._current_engine = self.config.engine
            logger.info(f"Using TTS engine: {self._current_engine}")
        else:
            # Fallback to first available engine
            self._current_engine = (
                next(iter(self._engines.keys())) if self._engines else None
            )
            logger.warning(
                f"Configured engine '{self.config.engine}' not available, using: {self._current_engine}"
            )

    def _start_background_processing(self):
        """Start background queue processing."""
        try:
            loop = asyncio.get_event_loop()
            self._processing_task = loop.create_task(self._process_queue())
            logger.info("Background TTS processing started")
        except RuntimeError:
            logger.warning("No event loop available for background processing")

    async def _process_queue(self):
        """Background task to process TTS synthesis queue."""
        logger.info("TTS queue processing started")

        while not self._shutdown_event.is_set():
            try:
                request = await asyncio.wait_for(
                    self._synthesis_queue.get(), timeout=1.0
                )
                await self._process_synthesis_request(request)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in TTS queue processing: {e}")
                await asyncio.sleep(1.0)

        logger.info("TTS queue processing stopped")

    async def _process_synthesis_request(self, request: SynthesisRequest):
        """Process a single synthesis request."""
        start_time = time.time()

        async with self._semaphore:
            try:
                # Check cache first
                if self.config.cache_enabled:
                    cached_path = self._check_cache(request.text)
                    if cached_path:
                        self._stats["cache_hits"] += 1
                        result = SynthesisResult(
                            success=True,
                            text=request.text,
                            audio_path=cached_path,
                            duration_ms=(time.time() - start_time) * 1000,
                            engine_used="cache",
                        )
                        self._publish_result(result)
                        return

                # Perform synthesis
                result = await self._synthesize_text(request)

                # Cache the result if successful
                if result.success and result.audio_path and self.config.cache_enabled:
                    self._cache_audio(request.text, result.audio_path)

                # Update statistics
                self._update_stats(result, time.time() - start_time)

                # Publish result
                self._publish_result(result)

            except Exception as e:
                logger.error(f"Error processing synthesis request: {e}")
                result = SynthesisResult(
                    success=False,
                    text=request.text,
                    duration_ms=(time.time() - start_time) * 1000,
                    engine_used=self._current_engine or "unknown",
                    error=str(e),
                )
                self._publish_result(result)

    async def _synthesize_text(self, request: SynthesisRequest) -> SynthesisResult:
        """Perform text synthesis using the configured engine."""
        start_time = time.time()

        try:
            if self._current_engine == "edge" and HAVE_EDGE_TTS:
                return await self._synthesize_with_edge(request)
            elif self._current_engine == "pyttsx3" and HAVE_PYTTSX3:
                return await self._synthesize_with_pyttsx3(request)
            elif self._current_engine == "system":
                return await self._synthesize_with_system(request)
            else:
                return SynthesisResult(
                    success=False,
                    text=request.text,
                    duration_ms=(time.time() - start_time) * 1000,
                    engine_used="none",
                    error="No suitable TTS engine available",
                )
        except Exception as e:
            return SynthesisResult(
                success=False,
                text=request.text,
                duration_ms=(time.time() - start_time) * 1000,
                engine_used=self._current_engine or "unknown",
                error=str(e),
            )

    async def _synthesize_with_edge(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize using Edge TTS."""
        start_time = time.time()

        try:
            voice = request.voice or self.config.voice

            # Create TTS object
            if edge_tts is not None:
                communicate = edge_tts.Communicate(request.text, voice)

                # Generate audio to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{self.config.output_format}"
                )
                temp_path = temp_file.name
                temp_file.close()

                # Save audio
                await communicate.save(temp_path)

                return SynthesisResult(
                    success=True,
                    text=request.text,
                    audio_path=temp_path,
                    duration_ms=(time.time() - start_time) * 1000,
                    engine_used="edge",
                    metadata={"voice": voice, "format": self.config.output_format},
                )
            else:
                raise Exception("Edge TTS module not available")

        except Exception as e:
            return SynthesisResult(
                success=False,
                text=request.text,
                duration_ms=(time.time() - start_time) * 1000,
                engine_used="edge",
                error=str(e),
            )

    async def _synthesize_with_pyttsx3(
        self, request: SynthesisRequest
    ) -> SynthesisResult:
        """Synthesize using pyttsx3."""
        start_time = time.time()

        try:
            engine = self._engines["pyttsx3"]["engine"]

            # Apply request-specific settings
            if request.rate:
                engine.setProperty("rate", request.rate)
            if request.volume:
                engine.setProperty("volume", request.volume)

            # Generate audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_path = temp_file.name
            temp_file.close()

            # Save to file
            engine.save_to_file(request.text, temp_path)

            # Run synthesis in thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, engine.runAndWait)

            return SynthesisResult(
                success=True,
                text=request.text,
                audio_path=temp_path,
                duration_ms=(time.time() - start_time) * 1000,
                engine_used="pyttsx3",
            )

        except Exception as e:
            return SynthesisResult(
                success=False,
                text=request.text,
                duration_ms=(time.time() - start_time) * 1000,
                engine_used="pyttsx3",
                error=str(e),
            )

    async def _synthesize_with_system(
        self, request: SynthesisRequest
    ) -> SynthesisResult:
        """Synthesize using system TTS (Windows SAPI)."""
        start_time = time.time()

        try:
            # Generate audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_path = temp_file.name
            temp_file.close()

            # Use Windows SAPI via command line
            escaped_text = request.text.replace("'", "''")
            sapi_command = f"powershell -Command \"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.SetOutputToWaveFile('{temp_path}'); $synth.Speak('{escaped_text}'); $synth.Dispose()\""

            # Run in executor to avoid blocking
            process = await asyncio.create_subprocess_shell(
                sapi_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and os.path.exists(temp_path):
                return SynthesisResult(
                    success=True,
                    text=request.text,
                    audio_path=temp_path,
                    duration_ms=(time.time() - start_time) * 1000,
                    engine_used="system",
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"System TTS failed: {error_msg}")

        except Exception as e:
            return SynthesisResult(
                success=False,
                text=request.text,
                duration_ms=(time.time() - start_time) * 1000,
                engine_used="system",
                error=str(e),
            )

    def _check_cache(self, text: str) -> Optional[str]:
        """Check if audio for text is cached."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        with self._cache_lock:
            if text_hash in self._audio_cache:
                path = self._audio_cache[text_hash]
                if os.path.exists(path):
                    return path
                else:
                    del self._audio_cache[text_hash]

        return None

    def _cache_audio(self, text: str, audio_path: str):
        """Cache audio file for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        with self._cache_lock:
            if len(self._audio_cache) >= self.config.cache_max_size:
                oldest_key = next(iter(self._audio_cache))
                old_path = self._audio_cache.pop(oldest_key)
                try:
                    os.remove(old_path)
                except OSError:
                    pass

            self._audio_cache[text_hash] = audio_path

    def _update_stats(self, result: SynthesisResult, processing_time: float):
        """Update processing statistics."""
        self._stats["total_requests"] += 1

        if result.success:
            self._stats["successful_syntheses"] += 1
        else:
            self._stats["failed_syntheses"] += 1

        self._stats["total_processing_time"] += processing_time
        self._stats["average_processing_time"] = (
            self._stats["total_processing_time"] / self._stats["total_requests"]
        )

    def _publish_result(self, result: SynthesisResult):
        """Publish synthesis result to the Vanta ecosystem."""
        try:
            self.vanta_core.publish_event(
                "tts_synthesis_complete",
                {
                    "component": self.COMPONENT_NAME,
                    "result": {
                        "success": result.success,
                        "text": result.text,
                        "audio_path": result.audio_path,
                        "duration_ms": result.duration_ms,
                        "engine_used": result.engine_used,
                        "error": result.error,
                        "metadata": result.metadata,
                    },
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish TTS result: {e}")

    async def handle_speech_request(self, message):
        """
        Async bus handler for TEXT_TO_SPEECH requests.
        Use this method for all async TTS requests via UnifiedVantaCore's async bus.
        Args:
            message: AsyncMessage instance with text and metadata
        Returns:
            dict: Synthesis result
        """
        # Example: message.content should contain text to synthesize
        # Implement actual synthesis logic here
        # ...
        return {"error": "Not implemented", "success": False}

    # Public API methods

    async def synthesize_text_async(self, text: str, **kwargs) -> SynthesisResult:
        """Synthesize text asynchronously."""
        request = SynthesisRequest(text=text, **kwargs)
        return await self._synthesize_text(request)

    def synthesize_text(self, text: str, **kwargs):
        """Queue text for synthesis (non-blocking)."""
        request = SynthesisRequest(text=text, **kwargs)

        try:
            self._synthesis_queue.put_nowait(request)
            logger.debug(f"Queued TTS request: {text[:50]}...")
        except asyncio.QueueFull:
            logger.warning("TTS synthesis queue is full")

    def get_available_voices(self) -> List[str]:
        """Get list of available voices for current engine."""
        if self._current_engine == "edge" and HAVE_EDGE_TTS:
            return [
                "en-US-AriaNeural",
                "en-US-JennyNeural",
                "en-US-GuyNeural",
                "en-GB-SoniaNeural",
                "en-GB-RyanNeural",
                "en-AU-NatashaNeural",
            ]
        elif self._current_engine == "pyttsx3" and HAVE_PYTTSX3:
            try:
                engine = self._engines["pyttsx3"]["engine"]
                voices = engine.getProperty("voices")
                return [voice.id for voice in voices] if voices else []
            except Exception:
                return []
        else:
            return ["system_default"]

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()

    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        return {
            "component": self.COMPONENT_NAME,
            "status": "healthy" if self._current_engine else "degraded",
            "current_engine": self._current_engine,
            "available_engines": list(self._engines.keys()),
            "queue_size": self._synthesis_queue.qsize(),
            "active_tasks": len(self._active_tasks),
            "cache_size": len(self._audio_cache),
            "stats": self.get_stats(),
        }

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up AsyncTTSEngine...")

        self._shutdown_event.set()

        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()

        for task in self._active_tasks.values():
            if not task.done():
                task.cancel()

        with self._cache_lock:
            for audio_path in self._audio_cache.values():
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            self._audio_cache.clear()

        if "pyttsx3" in self._engines and "engine" in self._engines["pyttsx3"]:
            try:
                self._engines["pyttsx3"]["engine"].stop()
            except Exception:
                pass

        logger.info("AsyncTTSEngine cleanup completed")

    @staticmethod
    def cleanup_all():
        """Static method to clean up any TTS resources."""
        logger.info("Static cleanup of all AsyncTTSEngine resources")
        try:
            # Clean up pyttsx3 resources which may persist across instances
            # Attempt to stop any running engines
            if HAVE_PYTTSX3 and pyttsx3:
                try:
                    engine = pyttsx3.init()
                    engine.stop()
                    logger.info("Stopped pyttsx3 engine")
                except Exception as e:
                    logger.warning(f"Could not stop pyttsx3 engine: {e}")
            # Clean up any temporary audio files
            temp_dir = tempfile.gettempdir()
            for pattern in ["tts_*.mp3", "tts_*.wav"]:
                for file in glob.glob(os.path.join(temp_dir, pattern)):
                    try:
                        os.remove(file)
                        logger.debug(f"Removed temporary file: {file}")
                    except OSError:
                        pass
            logger.info("Static TTS cleanup complete")
        except Exception as e:
            logger.warning(f"Static TTS cleanup failed: {e}")

    async def shutdown(self):
        """Async shutdown for interface consistency."""
        self.cleanup()
        self._shutdown_event.set()
        logger.info("AsyncTTSEngine shutdown complete.")


def create_async_tts_engine(
    vanta_core, config: Optional[TTSConfig] = None
) -> AsyncTTSEngine:
    """Create and return an AsyncTTSEngine instance."""
    return AsyncTTSEngine(vanta_core, config)
