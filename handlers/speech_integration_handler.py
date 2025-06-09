"""
Speech Integration Handler for VantaCore

This module provides the integration between TTS/STT engines and VantaCore,
ensuring that speech functionality is properly registered with VantaCore
and available to other components, including the GUI.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Import async_bus related components
from Vanta.core.UnifiedAsyncBus import MessageType
from Vanta.core.UnifiedVantaCore import UnifiedVantaCore, get_vanta_core

# Import TTS/STT engines
try:
    from Vanta.async_tts_engine import (
        AsyncTTSEngine,
        TTSConfig,
        create_async_tts_engine,
    )

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    AsyncTTSEngine = None
    TTSConfig = None
    create_async_tts_engine = None

try:
    from Vanta.async_stt_engine import AsyncSTTEngine, STTConfig

    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    AsyncSTTEngine = None
    STTConfig = None

logger = logging.getLogger("VantaCore.Speech")


@dataclass
class SpeechConfig:
    """Configuration for speech components."""

    tts_config: Optional[Dict[str, Any]] = None
    stt_config: Optional[Dict[str, Any]] = None
    enable_tts: bool = True
    enable_stt: bool = True
    register_with_async_bus: bool = True


class SpeechIntegrationHandler:
    """
    Handles integration of TTS and STT engines with VantaCore.
    This ensures that speech functionality is properly registered with VantaCore
    and available for use by other components, including the GUI.
    """

    def __init__(self, vanta_core: Optional[UnifiedVantaCore] = None):
        """Initialize with optional VantaCore instance."""
        self.vanta_core = vanta_core if vanta_core else get_vanta_core()
        self.tts_engine = None
        self.stt_engine = None
        self._tts_initialized = False
        self._stt_initialized = False

    def initialize_speech_engines(
        self, config: Optional[SpeechConfig] = None
    ) -> Dict[str, bool]:
        """
        Initialize TTS and STT engines based on the provided configuration.

        Args:
            config: Configuration for speech engines

        Returns:
            Dict indicating initialization status of each engine
        """
        if config is None:
            config = SpeechConfig()

        results = {"tts_initialized": False, "stt_initialized": False}

        # Initialize TTS engine if enabled
        if config.enable_tts:
            results["tts_initialized"] = self.initialize_tts_engine(config.tts_config)

        # Initialize STT engine if enabled
        if config.enable_stt:
            results["stt_initialized"] = self.initialize_stt_engine(config.stt_config)

        # Register with async bus if requested and available
        if config.register_with_async_bus and hasattr(self.vanta_core, "async_bus"):
            self._register_with_async_bus()

        return results

    def initialize_tts_engine(
        self, tts_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initialize the TTS engine.

        Args:
            tts_config: Configuration for the TTS engine

        Returns:
            True if initialization was successful, False otherwise
        """
        # First check if an engine is already registered
        existing_engine = self.vanta_core.get_component("async_tts_engine")
        if existing_engine:
            logger.info(f"Using existing TTS engine: {type(existing_engine).__name__}")
            self.tts_engine = existing_engine
            self._tts_initialized = True
            return True

        # Check if TTS is available
        if (
            not TTS_AVAILABLE
            or AsyncTTSEngine is None
            or create_async_tts_engine is None
        ):
            logger.warning("TTS engine not available")
            self._tts_initialized = False
            return False

        try:
            # Create TTS config if provided
            engine_config = None
            if tts_config and TTSConfig is not None:
                engine_config = TTSConfig(**tts_config)

            # Create and initialize the TTS engine
            self.tts_engine = create_async_tts_engine(self.vanta_core, engine_config)
            logger.info("TTS engine initialized successfully")

            # Engine self-registers with VantaCore in its __init__ method
            self._tts_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self._tts_initialized = False
            return False

    def initialize_stt_engine(
        self, stt_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initialize the STT engine.

        Args:
            stt_config: Configuration for the STT engine

        Returns:
            True if initialization was successful, False otherwise
        """
        # First check if an engine is already registered
        existing_engine = self.vanta_core.get_component("async_stt_engine")
        if existing_engine:
            logger.info(f"Using existing STT engine: {type(existing_engine).__name__}")
            self.stt_engine = existing_engine
            self._stt_initialized = True
            return True

        # Check if STT is available
        if not STT_AVAILABLE or AsyncSTTEngine is None:
            logger.warning("STT engine not available")
            self._stt_initialized = False
            return False

        try:
            # Create STT config if provided
            engine_config = None
            if stt_config and STTConfig is not None:
                engine_config = STTConfig(**stt_config)

            # Create and initialize the STT engine
            self.stt_engine = AsyncSTTEngine(self.vanta_core, engine_config)

            # Register with VantaCore
            self.vanta_core.register_component(
                "async_stt_engine",
                self.stt_engine,
                {
                    "type": "speech_to_text",
                    "engine": "vosk",
                    "language": getattr(engine_config, "language", "en-us")
                    if engine_config
                    else "en-us",
                },
            )

            # Initialize the engine (this loads models etc.)
            if hasattr(self.stt_engine, "initialize"):
                import asyncio

                loop = asyncio.get_event_loop()
                initialized = loop.run_until_complete(self.stt_engine.initialize())
                if not initialized:
                    logger.error("Failed to initialize STT engine")
                    self._stt_initialized = False
                    return False

            logger.info("STT engine initialized successfully")
            self._stt_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize STT engine: {e}")
            self._stt_initialized = False
            return False

    def _register_with_async_bus(self) -> None:
        """Register TTS and STT engines with the async bus."""
        if not hasattr(self.vanta_core, "async_bus"):
            logger.warning("Async bus not available, skipping registration")
            return

        # Register TTS engine with async bus if initialized
        if self._tts_initialized and self.tts_engine:
            # Check if engine has the required handler method
            if not hasattr(self.tts_engine, "handle_speech_request"):
                logger.warning(
                    "TTS engine does not have handle_speech_request method, implementing basic handler"
                )

                # Add a basic handler if missing
                async def handle_speech_request(message):
                    """Basic handler for TEXT_TO_SPEECH requests."""
                    if not hasattr(self.tts_engine, "synthesize_text"):
                        return {
                            "error": "TTS engine does not support synthesize_text",
                            "success": False,
                        }

                    try:
                        text = (
                            message.content
                            if hasattr(message, "content")
                            else message.get("text", "")
                        )
                        self.tts_engine.synthesize_text(text)
                        return {"success": True, "text": text}
                    except Exception as e:
                        logger.error(f"Error in TTS handler: {e}")
                        return {"error": str(e), "success": False}

                setattr(self.tts_engine, "handle_speech_request", handle_speech_request)

            try:
                self.vanta_core.async_bus.register_component("tts_engine")
                self.vanta_core.async_bus.subscribe(
                    "tts_engine",
                    MessageType.TEXT_TO_SPEECH,
                    self.tts_engine.handle_speech_request,
                )
                logger.info("TTS engine registered with async bus")
            except Exception as e:
                logger.error(f"Failed to register TTS engine with async bus: {e}")

        # Register STT engine with async bus if initialized
        if self._stt_initialized and self.stt_engine:
            # Check if engine has the required handler method
            if not hasattr(self.stt_engine, "handle_audio_request"):
                logger.warning(
                    "STT engine does not have handle_audio_request method, implementing basic handler"
                )

                # Add a basic handler if missing
                async def handle_audio_request(message):
                    """Basic handler for AUDIO_TRANSCRIPTION requests."""
                    if not hasattr(self.stt_engine, "transcribe_audio"):
                        return {
                            "error": "STT engine does not support transcribe_audio",
                            "success": False,
                        }

                    try:
                        # Get the maximum duration from the message if available
                        duration = (
                            message.get("duration") if hasattr(message, "get") else None
                        )

                        # Call the transcribe_audio method and get the final result
                        final_text = ""
                        async for result in self.stt_engine.transcribe_audio(
                            duration=duration
                        ):
                            if result.get("is_final", False) and result.get("text"):
                                final_text = result["text"]

                        return {"success": True, "text": final_text}
                    except Exception as e:
                        logger.error(f"Error in STT handler: {e}")
                        return {"error": str(e), "success": False}

                setattr(self.stt_engine, "handle_audio_request", handle_audio_request)

            try:
                self.vanta_core.async_bus.register_component("stt_engine")
                self.vanta_core.async_bus.subscribe(
                    "stt_engine",
                    MessageType.AUDIO_TRANSCRIPTION,
                    self.stt_engine.handle_audio_request,
                )
                logger.info("STT engine registered with async bus")
            except Exception as e:
                logger.error(f"Failed to register STT engine with async bus: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get status of speech engines."""
        status = {
            "tts_initialized": self._tts_initialized,
            "stt_initialized": self._stt_initialized,
            "tts_engine_type": type(self.tts_engine).__name__
            if self.tts_engine
            else None,
            "stt_engine_type": type(self.stt_engine).__name__
            if self.stt_engine
            else None,
            "tts_available": TTS_AVAILABLE,
            "stt_available": STT_AVAILABLE,
        }

        # Add engine-specific health status if available
        if self.tts_engine and hasattr(self.tts_engine, "get_health_status"):
            status["tts_health"] = self.tts_engine.get_health_status()

        if self.stt_engine and hasattr(self.stt_engine, "get_health_status"):
            status["stt_health"] = self.stt_engine.get_health_status()

        return status


def initialize_speech_system(
    vanta_core: Optional[UnifiedVantaCore] = None,
    tts_config: Optional[Dict[str, Any]] = None,
    stt_config: Optional[Dict[str, Any]] = None,
    enable_tts: bool = True,
    enable_stt: bool = True,
    register_with_async_bus: bool = True,
) -> SpeechIntegrationHandler:
    """
    Initialize the speech system with VantaCore.

    Args:
        vanta_core: Optional VantaCore instance
        tts_config: Configuration for TTS engine
        stt_config: Configuration for STT engine
        enable_tts: Whether to enable TTS
        enable_stt: Whether to enable STT
        register_with_async_bus: Whether to register with async bus

    Returns:
        The initialized speech integration handler
    """
    config = SpeechConfig(
        tts_config=tts_config,
        stt_config=stt_config,
        enable_tts=enable_tts,
        enable_stt=enable_stt,
        register_with_async_bus=register_with_async_bus,
    )

    handler = SpeechIntegrationHandler(vanta_core)
    handler.initialize_speech_engines(config)
    return handler
