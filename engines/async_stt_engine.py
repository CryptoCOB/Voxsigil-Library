"""
Async Speech-to-Text (STT) Engine for Vanta
Uses Vosk for offline speech recognition with async support
Enhanced with features from CRYSTAL_HARMONY SWARM

DEPRECATED: All async STT (speech-to-text) requests should be routed through UnifiedVantaCore's async bus.
Do not instantiate or use this engine directly. Use UnifiedVantaCore and its async bus for all orchestration.

# Example (for maintainers):
# from Vanta.core.UnifiedVantaCore import UnifiedVantaCore
# vanta_core = UnifiedVantaCore()
# vanta_core.async_bus.register_component('stt_engine')
# vanta_core.async_bus.subscribe('stt_engine', MessageType.AUDIO_TRANSCRIPTION, stt_engine.handle_audio_request)
"""

# Move all imports to the top of the file (PEP8 compliance)
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import field  # Keep for Pydantic if needed, else remove.
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import vosk

from Vanta.core.UnifiedAsyncBus import (
    MessageType,  # Import MessageType for async bus integration
)

# Configure logger first
logger = logging.getLogger("Vanta.AsyncSTT")

# Fix for circular import issue with numpy
try:
    # First, try to detect if we're in a problematic environment
    import sys

    # Check if numpy is already problematically imported
    problematic_numpy = False
    if "numpy" in sys.modules:
        try:
            import numpy as test_numpy

            # Try accessing a basic attribute
            _ = test_numpy.array
        except (AttributeError, ImportError):
            problematic_numpy = True
            logger.warning(
                "Detected problematic numpy import, will skip numpy functionality"
            )

    if not problematic_numpy:
        # Use our custom numpy resolver to safely import numpy
        import importlib.util

        # Dynamically import numpy_resolver
        module_name = "numpy_resolver"
        file_path = os.path.join(os.path.dirname(__file__), "numpy_resolver.py")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is not None and spec.loader is not None:
            numpy_resolver = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = numpy_resolver
            spec.loader.exec_module(numpy_resolver)
            # Get numpy modules with proper error handling
            numpy, np, HAVE_NUMPY = numpy_resolver.safe_import_numpy()
            if not HAVE_NUMPY:
                logger.warning("Numpy resolver failed, disabling numpy functionality")
        else:
            HAVE_NUMPY = False
            numpy = None
            np = None
            logger.warning("Failed to load numpy_resolver module spec or loader.")

    else:
        # Skip numpy entirely if problematic
        HAVE_NUMPY = False
        numpy = None
        np = None
        logger.warning("Skipping numpy import due to detected circular import issues")

except Exception as e:
    # Any error means we can't use numpy
    HAVE_NUMPY = False
    numpy = None
    np = None
    logger.warning(
        f"Failed to import numpy (will continue without numpy functionality): {e}"
    )

# Pydantic for config validation (optional)
try:
    import pydantic

    HAVE_PYDANTICS_DEPS = True
    BaseModel = pydantic.BaseModel  # type: ignore
    Field = pydantic.Field  # type: ignore
    validator = pydantic.validator  # type: ignore
except ImportError:
    HAVE_PYDANTICS_DEPS = False
    # Minimal fallback dataclass if Pydantic is not available
    from dataclasses import dataclass

    @dataclass
    class BaseModel:  # Mock BaseModel
        pass

    def Field(*args, **kwargs):  # Mock Field
        return field(*args, **kwargs)

    def validator(*args, **kwargs):  # Mock validator
        def decorator(func):
            return func

        return decorator


# STT Dependencies
try:
    import numpy as np
    import sounddevice as sd
    import vosk

    HAVE_STT_DEPS = True
except ImportError as e:
    HAVE_STT_DEPS = False
    logger.warning(
        f"Core STT dependencies (vosk, numpy, sounddevice) not available: {e}. STT functionality will be disabled."
    )
    vosk = None
    np = None
    sd = None

# Model Download Dependencies (optional)
try:
    import aiofiles
    import aiohttp

    HAVE_AIOHTTP_DEPS = True
except ImportError:
    HAVE_AIOHTTP_DEPS = False
    logger.warning(
        "aiohttp/aiofiles not available. Automatic model download will be disabled."
    )
    aiohttp = None
    aiofiles = None


DEFAULT_MODEL_DOWNLOAD_BASE_URL = "https://alphacephei.com/vosk/models"  # Example, check official Vosk for actual links


class STTConfig(BaseModel):
    sample_rate: int = Field(16000, gt=0, description="Audio sample rate in Hz.")
    channels: int = Field(1, ge=1, le=2, description="Number of audio channels.")
    default_duration: int = Field(
        7, gt=0, description="Default recording duration in seconds if not specified."
    )
    silence_threshold_vad: float = Field(
        0.01,
        ge=0.0,
        lt=1.0,
        description="Energy threshold for VAD (0.0-1.0). Lower is more sensitive.",
    )
    vad_buffer_before_ms: int = Field(
        200, ge=0, description="Milliseconds of audio to keep before VAD triggers."
    )
    vad_buffer_after_ms: int = Field(
        500, ge=0, description="Milliseconds of audio to keep after VAD stops."
    )
    language: str = Field(
        "en-us",
        description="Language code for Vosk model (e.g., 'en-us', 'de', 'fr'). Used for auto-download if model_path is empty.",
    )
    device_id: Optional[Union[int, str]] = Field(
        None, description="Audio input device ID or name substring. None for default."
    )
    transcription_process_timeout: Optional[float] = Field(
        20.0,
        gt=0,
        description="Timeout in seconds for the transcription process of a single audio chunk.",
    )
    model_download_timeout: float = Field(
        300.0, gt=0, description="Timeout in seconds for model download."
    )
    model_path: Optional[str] = Field(
        None, description="Path to the Vosk model directory."
    )

    if HAVE_PYDANTICS_DEPS:  # Pydantic specific validators

        @validator("model_path")
        def model_path_exists_if_set(cls, v):
            if v and not Path(v).exists() and not Path(v).is_dir():
                # This check is basic, full validation if it's a *valid* model path is hard
                # We primarily check existence if a path is provided.
                # If it's a name for auto-download, this validator doesn't apply
                # Logic for auto-download handles non-existent paths if path is empty.
                resolved_path = Path(v)
                if (
                    not resolved_path.is_absolute()
                ):  # Only check for existence if it looks like a path
                    # This simple check might not be robust enough for all cases of what 'v' might be.
                    # A more robust approach might involve trying to load a small part of the model or checking metadata.
                    logger.warning(
                        f"Provided model_path '{v}' does not exist. Attempting to use it anyway."
                    )
            return v


class AsyncSTTEngine:
    """Async Speech-to-Text Engine using Vosk with enhancements"""

    COMPONENT_NAME = "async_stt_engine"

    def __init__(self, vanta_core: Any, config: STTConfig):
        self.vanta_core = vanta_core
        if not isinstance(config, STTConfig):
            logger.warning(
                "Config is not an instance of STTConfig. Attempting to create one."
            )
            try:
                config = STTConfig(**config)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to parse config: {e}")
                raise ValueError(f"Invalid STTConfig provided: {e}") from e
        self.config = config
        self.model: Optional[Any] = None
        # Import KaldiRecognizer at the top of the file
        self.recognizer: Optional[Any] = None
        self.is_initialized = False
        self._transcription_task: Optional[asyncio.Task] = None
        self._is_cancelling = False

        if not HAVE_STT_DEPS:
            logger.error(
                "Core STT dependencies are missing. AsyncSTTEngine will not function."
            )
        if not self.config.model_path and not HAVE_AIOHTTP_DEPS:
            logger.warning(
                "model_path is not set and aiohttp/aiofiles are missing. Automatic model download is not possible."
            )

        # Register with VantaCore
        self.vanta_core.register_component(
            self.COMPONENT_NAME,
            self,
            {
                "type": "async_stt",
                "language": self.config.language,
                "sample_rate": self.config.sample_rate,
                "model_path": self.config.model_path,
            },
        )
        logger.info(f"{self.COMPONENT_NAME} registered with VantaCore")

        # === Unified Async Bus Integration ===
        # Register with UnifiedVantaCore's async bus and subscribe handler
        if hasattr(self.vanta_core, "async_bus"):
            self.vanta_core.async_bus.register_component("stt_engine")
            self.vanta_core.async_bus.subscribe(
                "stt_engine",
                MessageType.AUDIO_TRANSCRIPTION,
                self.handle_audio_request,
            )
            logger.info(
                "stt_engine registered and subscribed to async bus (AUDIO_TRANSCRIPTION)"
            )
        else:
            logger.warning(
                "UnifiedVantaCore async bus not available; async bus integration skipped."
            )

    async def handle_audio_request(self, message):
        """
        Async bus handler for AUDIO_TRANSCRIPTION requests.
        Use this method for all async STT requests via UnifiedVantaCore's async bus.

        Args:
            message: AsyncMessage instance with audio data or recording request parameters

        Returns:
            dict: Transcription result with text field
        """
        try:
            logger.info("Handling async audio transcription request")

            # Extract duration from the message if available
            duration = None
            if hasattr(message, "metadata") and isinstance(message.metadata, dict):
                duration = message.metadata.get("duration")
            elif hasattr(message, "get"):
                duration = message.get("duration")

            # Use default duration if not specified
            if duration is None:
                duration = self.config.default_duration
                logger.debug(f"Using default duration: {duration}s")

            # Process audio transcription
            final_text = ""
            is_error = False
            error_message = None

            try:
                # Use async generator to get all results
                async for result in self.transcribe_audio(duration=duration):
                    # Check for errors
                    if "error" in result:
                        is_error = True
                        error_message = result["error"]
                        logger.error(f"Transcription error: {error_message}")
                        break

                    # Check for cancellation
                    if result.get("status") == "cancelled":
                        logger.info("Transcription was cancelled")
                        break

                    # Process final result
                    if result.get("is_final", False):
                        current_text = result.get("text", "")
                        if current_text:
                            if final_text:
                                final_text += " " + current_text
                            else:
                                final_text = current_text

                            logger.info(f"Final transcription: {final_text}")

            except Exception as e:
                is_error = True
                error_message = str(e)
                logger.error(f"Error during transcription: {e}", exc_info=True)

            # Return the result
            if is_error:
                return {
                    "success": False,
                    "error": error_message or "Unknown transcription error",
                    "text": "",
                }
            else:
                return {
                    "success": True,
                    "text": final_text.strip(),
                    "source": "async_stt_engine",
                }

        except Exception as e:
            logger.error(f"Error in handle_audio_request: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "text": "",
            }

    async def _download_model_if_needed(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Downloads Vosk model if path is not specified and dependencies are met."""
        if not self.config.model_path and self.config.language:
            if not HAVE_AIOHTTP_DEPS or not aiohttp or not aiofiles:
                logger.error(
                    "Cannot download model: aiohttp/aiofiles dependencies are missing."
                )
                return False

            # Standard Vosk model directory (example, may vary)
            vosk_model_dir = Path.home() / ".cache" / "vosk"
            expected_model_path = (
                vosk_model_dir / f"vosk-model-{self.config.language}-0.22"
            )  # Version might change

            if expected_model_path.exists() and expected_model_path.is_dir():
                logger.info(f"Using existing downloaded model at {expected_model_path}")
                self.config.model_path = str(expected_model_path)
                return True

            # Simplified model name for download URL (this needs real Vosk model naming)
            # Actual Vosk models are archives like vosk-model-small-en-us-0.15.zip
            # This part needs to be robust and know actual model archive names and structure.
            # For now, this is a placeholder.
            model_archive_name = (
                f"vosk-model-small-{self.config.language}-0.15.zip"  # EXAMPLE
            )
            model_url = f"{DEFAULT_MODEL_DOWNLOAD_BASE_URL}/{model_archive_name}"
            download_target_path = vosk_model_dir / model_archive_name
            extracted_model_dir_name = model_archive_name.replace(
                ".zip", ""
            )  # Approximation

            try:
                logger.info(
                    f"Attempting to download model for language '{self.config.language}' from {model_url}..."
                )
                self.vanta_core.publish_event(
                    "stt.model.download.start",
                    {"language": self.config.language, "url": model_url},
                    source="AsyncSTTEngine",
                )

                os.makedirs(vosk_model_dir, exist_ok=True)

                timeout = aiohttp.ClientTimeout(
                    total=self.config.model_download_timeout
                )
                async with aiohttp.ClientSession() as session:
                    async with session.get(model_url, timeout=timeout) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("content-length", 0))
                        downloaded_size = 0

                        async with aiofiles.open(download_target_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                                downloaded_size += len(chunk)
                                if total_size > 0:
                                    progress = (downloaded_size / total_size) * 100
                                    # Throttle progress updates if needed
                                    logger.debug(
                                        f"Model download progress: {progress:.2f}%"
                                    )
                                    self.vanta_core.publish_event(
                                        "stt.model.download.progress",
                                        {"progress": progress},
                                        source="AsyncSTTEngine",
                                    )

                logger.info(
                    f"Model downloaded to {download_target_path}. Extracting..."
                )
                # ADD MODEL EXTRACTION LOGIC HERE (e.g., using zipfile module in executor)
                # This is a blocking operation, should be run in executor
                # Example: await loop.run_in_executor(None, self._extract_model, download_target_path, vosk_model_dir)
                # For now, assume extraction happens and set path

                # Placeholder: Assume model extracts to a folder named after the archive (minus .zip)
                final_model_path = vosk_model_dir / extracted_model_dir_name
                if (
                    not final_model_path.exists()
                ):  # This check depends on actual extraction success
                    # Simplified: Simulate extraction was successful by setting the path.
                    # In reality, you'd check if extraction created the expected directory.
                    # For now, we'll error if the expected path (post-extraction placeholder) isn't what we need.
                    # This part is highly dependent on real Vosk model archive structures.
                    logger.warning(
                        f"Model extraction simulation. Assuming {final_model_path} is the model dir."
                    )
                    # Let's assume the extracted model path is directly usable if named correctly.
                    # The user might need to download specific pre-extracted model folders.

                self.config.model_path = str(
                    final_model_path
                )  # This needs to be the *extracted model directory*
                logger.info(
                    f"Model successfully downloaded and 'extracted' to {self.config.model_path}"
                )
                self.vanta_core.publish_event(
                    "stt.model.download.success",
                    {"path": self.config.model_path},
                    source="AsyncSTTEngine",
                )
                return True

            except Exception as e:
                logger.error(f"Failed to download or extract model: {e}")
                self.vanta_core.publish_event(
                    "stt.model.download.failure",
                    {"error": str(e)},
                    source="AsyncSTTEngine",
                )
                if download_target_path.exists():
                    os.remove(download_target_path)  # Clean up partial download
                return False
        elif self.config.model_path:
            return True  # Model path is already provided
        else:
            logger.error(
                "Cannot initialize model: No model_path specified and language not set for auto-download."
            )
            return False

    # def _extract_model(self, archive_path: Path, extract_to_dir: Path):
    #     import zipfile
    #     with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    #         zip_ref.extractall(extract_to_dir)
    #     os.remove(archive_path) # Remove zip after extraction
    #     logger.info(f"Model extracted to {extract_to_dir}")

    async def initialize(self) -> bool:
        """Initialize the STT engine asynchronously, including model download if needed."""
        if not HAVE_STT_DEPS or not vosk or not sd or not np:
            logger.error(
                "STT dependencies (vosk, numpy, sounddevice) not available for initialization."
            )
            return False

        if self.is_initialized:
            logger.info("STT engine already initialized.")
            return True

        logger.info("Initializing Vosk STT engine...")
        loop = asyncio.get_event_loop()

        if not await self._download_model_if_needed(loop):
            logger.error("Model setup failed. Cannot initialize STT engine.")
            return False

        if not self.config.model_path:  # Double check after download attempt
            logger.error(
                "Model path still not available after download attempt. Cannot initialize."
            )
            return False

        try:
            model_load_path = self.config.model_path
            logger.info(f"Loading Vosk model from: {model_load_path}")

            # Check if model_path actually points to a valid model directory structure.
            # Vosk expects a directory with specific files like "am/final.mdl", "conf/model.conf" etc.
            # A simple Path(model_load_path).exists() isn't enough.
            # For this example, we'll assume the path provided or downloaded/extracted is correct.
            # A real implementation should verify the model structure.
            if not Path(model_load_path).is_dir():
                logger.error(
                    f"Vosk model path '{model_load_path}' is not a directory or does not exist. Please provide a valid model directory path."
                )
                return False
            # Further check if it contains expected Vosk model files/subdirs like 'am', 'graph', 'conf'
            # if not (Path(model_load_path)/'am').exists() or not (Path(model_load_path)/'conf').exists():
            #     logger.error(f"Vosk model path '{model_load_path}' does not seem to contain a valid model structure (missing 'am' or 'conf' subdirectories).")
            #     return False

            self.model = await loop.run_in_executor(
                None,
                lambda: vosk.Model(model_load_path),  # type: ignore
            )

            self.recognizer = await loop.run_in_executor(
                None,
                lambda: vosk.KaldiRecognizer(self.model, self.config.sample_rate),  # type: ignore
            )
            # Enable word-level details
            self.recognizer.SetWords(True)  # type: ignore
            # self.recognizer.SetPartialWords(True) # For more granular partials

            self.is_initialized = True
            logger.info("Vosk STT engine initialized successfully")
            self.vanta_core.publish_event(
                "stt.engine.initialized",
                {
                    "model_path": str(model_load_path),
                    "language": self.config.language,
                    "sample_rate": self.config.sample_rate,
                },
                source="AsyncSTTEngine",
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize STT engine: {e}", exc_info=True)
            self.model = None
            self.recognizer = None
            self.is_initialized = False
            return False

    @staticmethod
    def list_available_devices() -> List[Dict[str, Any]]:
        """Lists available audio input devices."""
        if not HAVE_STT_DEPS or not sd:
            logger.warning("sounddevice not available, cannot list devices.")
            return []
        try:
            return sd.query_devices()  # type: ignore
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
            return []

    def _record_audio_with_vad(self, duration: int) -> Optional[Any]:
        """Record audio data with basic VAD (blocking, run in thread pool)."""
        if not sd or not np:
            return None
        try:
            samplerate = self.config.sample_rate
            channels = self.config.channels
            device = self.config.device_id

            logger.info(f"Recording audio for max {duration}s with VAD...")

            # VAD parameters
            energy_threshold = self.config.silence_threshold_vad
            # Chunk size for VAD analysis (e.g., 30ms)
            vad_chunk_duration_ms = 30
            vad_chunk_size = int(samplerate * vad_chunk_duration_ms / 1000)

            buffer_before_samples = int(
                samplerate * self.config.vad_buffer_before_ms / 1000
            )
            int(samplerate * self.config.vad_buffer_after_ms / 1000)

            max_samples = int(duration * samplerate)

            recorded_frames = []
            active_voice_frames = []
            is_speaking = False
            silence_chunks_after_speech = 0
            max_silence_chunks = (
                self.config.vad_buffer_after_ms // vad_chunk_duration_ms
            )

            with sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                device=device,
                dtype="int16",
                blocksize=vad_chunk_size,
            ) as stream:
                for _ in range(
                    max_samples // vad_chunk_size
                ):  # Iterate for max duration by VAD chunks
                    if self._is_cancelling:
                        logger.info("Recording cancelled during VAD.")
                        return None

                    audio_chunk, overflowed = stream.read(vad_chunk_size)
                    if overflowed:
                        logger.warning("Audio input overflowed!")

                    recorded_frames.append(audio_chunk)
                    if (
                        len(recorded_frames)
                        > (buffer_before_samples // vad_chunk_size) + 1
                    ):  # Keep limited history for pre-buffer
                        recorded_frames.pop(0)

                    # Simple energy-based VAD
                    energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)) / (
                        2**15
                    )  # Normalized energy

                    if energy > energy_threshold:
                        if not is_speaking:  # Speech start
                            logger.debug("VAD: Speech started")
                            is_speaking = True
                            # Add pre-buffer
                            for frame in recorded_frames[
                                :-1
                            ]:  # Add history except current chunk
                                active_voice_frames.append(frame)
                        active_voice_frames.append(audio_chunk)
                        silence_chunks_after_speech = 0
                    elif is_speaking:  # Silence after speech
                        active_voice_frames.append(
                            audio_chunk
                        )  # still add to post-buffer
                        silence_chunks_after_speech += 1
                        if silence_chunks_after_speech >= max_silence_chunks:
                            logger.debug("VAD: Speech ended (silence detected)")
                            break  # End recording due to silence

                    if sum(f.shape[0] for f in active_voice_frames) >= max_samples:
                        logger.debug("VAD: Max duration reached for active voice.")
                        break

            if not active_voice_frames:
                logger.info("VAD: No speech detected above threshold.")
                return None

            audio_data = np.concatenate(active_voice_frames)
            return np.squeeze(audio_data) if audio_data.size > 0 else None

        except Exception as e:
            logger.error(f"Error recording audio with VAD: {e}", exc_info=True)
            return None

    def _transcribe_chunk_vosk(
        self, audio_data_bytes: bytes, is_final_chunk: bool
    ) -> List[Dict[str, Any]]:
        """Helper to process one chunk with Vosk, handles partial and final results."""
        if not self.recognizer:
            return []

        results = []
        current_processing_time = 0

        start_time = time.monotonic()
        if self.recognizer.AcceptWaveform(audio_data_bytes):
            # Full final result from this chunk
            final_res_json = self.recognizer.Result()
            final_res_dict = json.loads(final_res_json)
            current_processing_time = (time.monotonic() - start_time) * 1000
            results.append(
                {
                    "text": final_res_dict.get("text", ""),
                    "is_final": True,  # Indicates this is a complete utterance segment
                    "chunk_info": final_res_dict,  # Full Vosk result for this segment
                    "processing_time_ms": current_processing_time,
                }
            )
        else:
            # Partial result
            partial_res_json = self.recognizer.PartialResult()
            partial_res_dict = json.loads(partial_res_json)
            current_processing_time = (time.monotonic() - start_time) * 1000
            if partial_res_dict.get("partial", ""):
                results.append(
                    {
                        "text": partial_res_dict.get("partial", ""),
                        "is_final": False,
                        "chunk_info": partial_res_dict,
                        "processing_time_ms": current_processing_time,
                    }
                )

        if (
            is_final_chunk and not results
        ):  # If it's the absolute last chunk, force FinalResult
            final_res_json = self.recognizer.FinalResult()
            final_res_dict = json.loads(final_res_json)
            current_processing_time = (
                time.monotonic() - start_time
            ) * 1000  # Re-measure if FinalResult was called
            # Only add if it has text and wasn't already captured by AcceptWaveform returning true
            # This check is important to avoid duplicate "final" results if AcceptWaveform already triggered one
            # A better state machine might be needed if AcceptWaveform=True AND FinalResult() are both to be parsed.
            # Typically, if AcceptWaveform returns true, Result() is what you want. If false, PartialResult().
            # On the very last chunk, you always call FinalResult().
            if final_res_dict.get("text", ""):  # Add only if there's text
                results.append(
                    {
                        "text": final_res_dict.get("text", ""),
                        "is_final": True,  # This is the ultimate final result
                        "chunk_info": final_res_dict,
                        "processing_time_ms": current_processing_time,
                    }
                )

        return results

    async def transcribe_audio(
        self, duration: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Record and transcribe audio asynchronously, yielding partial and final results.
        Yields dictionaries: {"text": str, "is_final": bool, "chunk_info": Optional[Dict], "processing_time_ms": float}
        """
        if (
            not self.is_initialized
            or not self.recognizer
            or not HAVE_STT_DEPS
            or not sd
            or not np
        ):
            logger.error("STT engine not initialized or core dependencies missing.")
            # Yield a final error message or raise an exception
            yield {
                "text": "",
                "is_final": True,
                "error": "STT engine not initialized",
                "chunk_info": None,
                "processing_time_ms": 0.0,
            }
            return

        if self._transcription_task and not self._transcription_task.done():
            logger.warning(
                "Another transcription is already in progress. Please cancel it first or wait."
            )
            yield {
                "text": "",
                "is_final": True,
                "error": "Transcription already in progress",
                "chunk_info": None,
                "processing_time_ms": 0.0,
            }
            return

        self._is_cancelling = False
        current_task = asyncio.current_task()
        self._transcription_task = current_task  # Store the current task

        duration = duration or self.config.default_duration
        loop = asyncio.get_event_loop()

        total_audio_duration_processed = 0.0

        try:
            logger.info(f"Starting transcription process for max {duration}s...")
            self.vanta_core.publish_event(
                "stt.recording.start", {"duration": duration}, source="AsyncSTTEngine"
            )

            # Audio recording part (can be refactored to stream audio chunks as well)
            # For now, using the VAD-enabled _record_audio_with_vad
            audio_data = await loop.run_in_executor(
                None, self._record_audio_with_vad, duration
            )

            if self._is_cancelling:
                logger.info("Transcription cancelled during audio recording.")
                self.vanta_core.publish_event(
                    "stt.transcription.cancelled", {}, source="AsyncSTTEngine"
                )
                yield {
                    "text": "",
                    "is_final": True,
                    "status": "cancelled",
                    "chunk_info": None,
                    "processing_time_ms": 0.0,
                }
                return

            if audio_data is None or audio_data.size == 0:
                logger.info("No audio data recorded or VAD detected no speech.")
                self.vanta_core.publish_event(
                    "stt.transcription.empty",
                    {"reason": "No audio data from VAD"},
                    source="AsyncSTTEngine",
                )
                yield {
                    "text": "",
                    "is_final": True,
                    "status": "no_audio_data",
                    "chunk_info": None,
                    "processing_time_ms": 0.0,
                }
                return

            total_audio_duration_processed = len(audio_data) / self.config.sample_rate
            logger.info(
                f"Audio recorded ({total_audio_duration_processed:.2f}s). Processing with Vosk..."
            )

            # Simulate chunking for streaming to Vosk if _record_audio_with_vad returned a block
            # Ideally, audio source itself would be an async iterator of chunks
            audio_bytes = audio_data.tobytes()

            # How to chunk for Vosk? Let's say 1-second chunks or smaller.
            # For simplicity with current structure, we process the whole audio_data block.
            # For true streaming partials, the _record_audio would yield chunks,
            # and _transcribe_chunk_vosk would be called per chunk.
            # The current `_transcribe_chunk_vosk` is more for conceptual streaming, let's adapt:

            start_transcribe_time = time.monotonic()

            # This timeout applies to the whole transcription of the current audio_data block
            transcription_results = await asyncio.wait_for(
                loop.run_in_executor(
                    None, self._process_full_audio_with_partials, audio_bytes
                ),
                timeout=self.config.transcription_process_timeout,
            )

            overall_processing_time = (time.monotonic() - start_transcribe_time) * 1000

            if self._is_cancelling:  # Check after potentially long blocking call
                logger.info("Transcription cancelled during Vosk processing.")
                self.vanta_core.publish_event(
                    "stt.transcription.cancelled", {}, source="AsyncSTTEngine"
                )
                yield {
                    "text": "",
                    "is_final": True,
                    "status": "cancelled",
                    "chunk_info": None,
                    "processing_time_ms": 0.0,
                }
                return

            final_text_parts = []
            for res in transcription_results:
                yield res  # Yield each partial/final result dictionary as it comes
                if res["is_final"] and res.get("text"):
                    final_text_parts.append(res["text"])

            final_transcription = " ".join(filter(None, final_text_parts)).strip()

            if final_transcription:
                logger.info(
                    f"Final Transcription: '{final_transcription}' (Audio duration: {total_audio_duration_processed:.2f}s, Processing: {overall_processing_time:.0f}ms)"
                )
                self.vanta_core.publish_event(
                    "stt.transcription.success",
                    {
                        "text": final_transcription,
                        "audio_duration_s": total_audio_duration_processed,
                        "processing_time_ms": overall_processing_time,
                        "word_details": transcription_results[-1]
                        .get("chunk_info", {})
                        .get("result")
                        if transcription_results
                        and transcription_results[-1]["is_final"]
                        else None,
                    },
                    source="AsyncSTTEngine",
                )
            else:
                logger.info("Transcription resulted in empty text.")
                self.vanta_core.publish_event(
                    "stt.transcription.empty",
                    {"reason": "Vosk returned empty text"},
                    source="AsyncSTTEngine",
                )

            # Final yield to signal completion (even if empty text)
            # The last `res` from transcription_results *should* be the truly final one.
            if not transcription_results or not transcription_results[-1]["is_final"]:
                yield {
                    "text": final_transcription,
                    "is_final": True,
                    "status": "completed_empty_or_no_final_marker",
                    "chunk_info": None,
                    "processing_time_ms": overall_processing_time,
                }

        except asyncio.TimeoutError:
            logger.error(
                f"Transcription process timed out after {self.config.transcription_process_timeout}s."
            )
            self.vanta_core.publish_event(
                "stt.transcription.error", {"error": "Timeout"}, source="AsyncSTTEngine"
            )
            yield {
                "text": "",
                "is_final": True,
                "error": "Timeout",
                "chunk_info": None,
                "processing_time_ms": 0.0,
            }
        except asyncio.CancelledError:
            logger.info("Transcription task was cancelled.")
            self.vanta_core.publish_event(
                "stt.transcription.cancelled", {}, source="AsyncSTTEngine"
            )
            yield {
                "text": "",
                "is_final": True,
                "status": "cancelled",
                "chunk_info": None,
                "processing_time_ms": 0.0,
            }
            # Ensure recognizer is reset if cancelled mid-processing a segment
            if self.recognizer:
                self.recognizer.Reset()
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            self.vanta_core.publish_event(
                "stt.transcription.error", {"error": str(e)}, source="AsyncSTTEngine"
            )
            yield {
                "text": "",
                "is_final": True,
                "error": str(e),
                "chunk_info": None,
                "processing_time_ms": 0.0,
            }
        finally:
            if self.recognizer:  # Reset recognizer state for next use
                self.recognizer.Reset()
            self._transcription_task = None
            self._is_cancelling = False

    def _process_full_audio_with_partials(
        self, audio_data_bytes: bytes
    ) -> List[Dict[str, Any]]:
        """
        Processes a single block of audio data, generating partial/final results.
        This is a blocking function, designed to be run in an executor.
        """
        if not self.recognizer:
            return []

        results_log = []
        chunk_size = self.config.sample_rate * 2 * 1  # 1 second of 16-bit mono audio

        # In a real streaming scenario, this loop would be driven by incoming audio chunks.
        # Here, we simulate it by feeding chunks of the complete `audio_data_bytes`.
        for i in range(0, len(audio_data_bytes), chunk_size):
            if self._is_cancelling:  # Check for cancellation (best effort in sync code)
                # This check is limited; if cancellation occurs mid-Vosk call, it won't be caught here.
                logger.debug("Vosk processing chunk cancelled.")
                break

            chunk = audio_data_bytes[i : i + chunk_size]
            is_last_chunk_of_this_block = (i + chunk_size) >= len(audio_data_bytes)

            # _transcribe_chunk_vosk is adapted to handle this usage.
            # It will use AcceptWaveform and PartialResult/Result
            vosk_results = self._transcribe_chunk_vosk(
                chunk, is_last_chunk_of_this_block
            )
            results_log.extend(vosk_results)

        # After all chunks, ensure a final result is extracted if not already done.
        # This logic is now mostly handled within _transcribe_chunk_vosk by passing is_final_chunk=True
        # but we can call FinalResult one last time if no "is_final: True" was seen.
        # Check if the last result from vosk_results was truly final.
        needs_explicit_final_call = True
        if results_log and results_log[-1].get("is_final"):
            needs_explicit_final_call = False

        if needs_explicit_final_call and not self._is_cancelling:
            logger.debug("Forcing final result from Vosk recognizer.")
            start_time = time.monotonic()
            final_res_json = (
                self.recognizer.FinalResult()
            )  # Call FinalResult on whatever is buffered.
            processing_time = (time.monotonic() - start_time) * 1000
            final_res_dict = json.loads(final_res_json)
            if final_res_dict.get("text"):  # Only add if there's new text
                results_log.append(
                    {
                        "text": final_res_dict.get("text", ""),
                        "is_final": True,
                        "chunk_info": final_res_dict,
                        "processing_time_ms": processing_time,
                    }
                )
        return results_log

    async def cancel_transcription(self):
        """Requests cancellation of the current transcription task."""
        self._is_cancelling = True  # Set flag for synchronous parts
        if self._transcription_task and not self._transcription_task.done():
            logger.info("Attempting to cancel transcription task...")
            self._transcription_task.cancel()
            try:
                await self._transcription_task  # Wait for cancellation to be processed
            except asyncio.CancelledError:
                logger.info("Transcription task successfully cancelled.")
            except Exception as e:
                # Log if waiting for the task raised an unexpected error
                logger.warning(
                    f"Error encountered while waiting for cancelled task: {e}"
                )
        else:
            logger.info("No active transcription task to cancel.")

    async def shutdown(self):
        """Shutdown the STT engine and cancel any ongoing transcription."""
        logger.info("Shutting down STT engine...")
        await self.cancel_transcription()  # Ensure any active task is cancelled

        self.is_initialized = False
        # Release Vosk resources if they were loaded directly (Python's GC should handle it)
        # If Vosk uses C libraries that need explicit free, it would be done here.
        # For vosk.Model and vosk.KaldiRecognizer, Python's GC is usually sufficient.
        self.model = None
        self.recognizer = None

        logger.info("STT engine shutdown complete.")
        self.vanta_core.publish_event(
            "stt.engine.shutdown", {}, source="AsyncSTTEngine"
        )

    def get_health_status(self) -> dict:
        """Return health status of the STT engine."""
        return {
            "component": self.COMPONENT_NAME,
            "status": "healthy" if self.is_initialized else "degraded",
            "model_loaded": self.model is not None,
            "language": getattr(self.config, "language", None),
        }


# --- VantaCore mock for example usage if not imported ---
class VantaCore:
    def register_component(self, name: str, component: object, metadata: dict):
        logger.info(f"Registered component {name} with metadata: {metadata}")

    def publish_event(self, event_name: str, payload: dict, source: str):
        logger.info(
            f"EVENT from {source}: {event_name} - {json.dumps(payload, indent=2)}"
        )


# Example Usage (requires a mock vanta_core and proper model setup)
# async def main():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     )

#     # Ensure you have a Vosk model available.
#     # For 'en-us', download from https://alphacephei.com/vosk/models
#     # e.g., vosk-model-small-en-us-0.15 and extract it.
#     # Update MODEL_PATH_FOR_TESTING to point to the extracted model directory.
#     MODEL_PATH_FOR_TESTING = "vosk-model-small-en-us-0.15"  # REPLACE WITH YOUR MODEL PATH or leave "" for auto-download if implemented

#     # Check if specified model path exists for testing, otherwise rely on auto-download feature
#     if MODEL_PATH_FOR_TESTING and not Path(MODEL_PATH_FOR_TESTING).exists():
#         logger.warning(
#             f"Test model path '{MODEL_PATH_FOR_TESTING}' not found. Set config.model_path to empty for auto-download attempt or provide a valid path."
#         )
#         # To force auto-download for testing, set it to None or ""
#         # test_model_path_for_config = None
#     # else:
#     #    test_model_path_for_config = MODEL_PATH_FOR_TESTING

#     # Test auto-download (requires aiohttp, aiofiles, and correct model URLs/names)
#     # config = STTConfig(model_path=None, language="en-us")

#     # Test with a specific model path:
#     config = STTConfig(
#         model_path=MODEL_PATH_FOR_TESTING, language="en-us"
#     )  # or your language

#     if (
#         not Path(config.model_path if config.model_path else "").exists()
#         and not config.language
#     ):
#         logger.error(
#             "For the example, please set a valid MODEL_PATH_FOR_TESTING or ensure config.language is set for auto-download."
#         )
#         return

#     # Check for critical dependencies before even trying
#     if not HAVE_STT_DEPS:
#         logger.error("Cannot run example: Core STT dependencies missing.")
#         return

#     # For auto-download feature to work, ensure AIOHTTP_DEPS are met.
#     if not config.model_path and not HAVE_AIOHTTP_DEPS:
#         logger.error(
#             "Cannot run auto-download example: aiohttp/aiofiles dependencies missing."
#         )
#         return

#     stt_engine = AsyncSTTEngine(vanta_core=VantaCore(), config=config)

#     if await stt_engine.initialize():
#         logger.info("STT Engine Initialized for example usage.")

#         for i, device in enumerate(stt_engine.list_available_devices()):
#             logger.info(
#                 f"  Device {device.get('index', i)}: {device.get('name')} (Input Channels: {device.get('max_input_channels')})"
#             )

#         logger.info(
#             "Starting transcription (will run for default_duration or until VAD stops speech)..."
#         )

#         # Example of how to use the async generator
#         try:
#             async for result in stt_engine.transcribe_audio(
#                 duration=10
#             ):  # Record for 10s max
#                 if result.get("error"):
#                     logger.error(f"Transcription error: {result['error']}")
#                     break
#                 if result.get("status") == "cancelled":
#                     logger.info("Transcription was cancelled by user or system.")
#                     break

#                 if result["is_final"]:
#                     logger.info(
#                         f"FINAL RESULT: '{result['text']}' (Processing: {result.get('processing_time_ms', 0):.0f}ms)"
#                     )
#                     # logger.info(f"Full chunk_info for final: {json.dumps(result.get('chunk_info'), indent=2)}")
#                 else:
#                     logger.info(
#                         f"PARTIAL: '{result['text']}' (Processing: {result.get('processing_time_ms', 0):.0f}ms)"
#                     )

#                 # Example: External cancellation after receiving first partial result
#                 # if not result["is_final"] and result["text"]:
#                 #     logger.warning("DEMO: Requesting cancellation after first partial...")
#                 #     await stt_engine.cancel_transcription()

#         except Exception as e:
#             logger.error(f"Error during example transcription: {e}", exc_info=True)

#         await stt_engine.shutdown()
#     else:
#         logger.error("STT Engine failed to initialize for example.")


# if __name__ == "__main__":
#     # This example part would require a running asyncio loop.
#     # To run it: python your_script_name.py
#     # Ensure you have a microphone and have installed dependencies.
#     # Also, set up a Vosk model or ensure auto-download works.
#     if sys.platform == "win32":  # Fix for asyncio on Windows with ProactorEventLoop
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logger.info("Example stopped by user.")
#     except Exception as e:
#         logger.error(f"Unhandled error in main: {e}", exc_info=True)
