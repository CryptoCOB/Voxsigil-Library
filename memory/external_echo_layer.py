# File: vanta_external_echo_layer.py (Refactored Revision with MemoryBraid)
"""
External Echo Layer for VantaCore with Direct MemoryBraid Integration.

Bridges internal echo system with external I/O, logs events to VantaCore,
and imprints significant external interactions into a MemoryBraid.
"""

import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

# Import unified MemoryBraidInterface
from Vanta.interfaces.protocol_interfaces import MemoryBraidInterface
from VoxSigilRag.voxsigil_rag_compression import RAGCompressionEngine

_default_compressor = RAGCompressionEngine()

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VantaCore.ExternalEchoLayer")


# --- Configuration Class ---
class ExternalEchoLayerConfig:
    def __init__(
        self,
        log_level: str = "INFO",
        record_seconds: int = 5,  # Potentially for future audio buffering
        heartbeat_interval_s: int = 10,  # Increased default
        echo_stream_component_name: str = "echo_stream_service",
        meta_reflex_component_name: str = "meta_reflex_service",
        memory_braid_component_name: str = "memory_braid_service",  # For MemoryBraid
        default_event_ttl_s: int = 3600,  # Default TTL for events imprinted to MemoryBraid
    ):
        self.log_level = log_level
        self.record_seconds = record_seconds
        self.heartbeat_interval_s = heartbeat_interval_s
        self.echo_stream_component_name = echo_stream_component_name
        self.meta_reflex_component_name = meta_reflex_component_name
        self.memory_braid_component_name = memory_braid_component_name
        self.default_event_ttl_s = default_event_ttl_s

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


# --- Interface Definitions (Protocols) ---
@runtime_checkable
class EchoStreamInterface(Protocol):
    def add_source(self, callback: Callable[[str, dict[str, Any]], dict[str, Any]]) -> None: ...
    def add_sink(self, callback: Callable[[str, dict[str, Any]], None]) -> None: ...
    def emit(
        self, channel: str, text: Optional[str], metadata: Optional[dict[str, Any]]
    ) -> None: ...


@runtime_checkable
class MetaReflexLayerInterface(Protocol):
    def process_echo(self, echo_data: dict[str, Any]) -> None: ...


# --- Minimal Fallback Implementations ---
# Use minimal implementations instead of complex stubs


class _FallbackEchoStream:
    """Minimal fallback for EchoStreamInterface."""

    def add_source(self, cb):
        pass

    def add_sink(self, cb):
        pass

    def emit(self, ch, txt, meta):
        pass


class _FallbackMetaReflexLayer:
    """Minimal fallback for MetaReflexLayerInterface."""

    def process_echo(self, data):
        pass


class _FallbackMemoryBraid:
    """Minimal fallback for MemoryBraidInterface."""

    def imprint(self, key: str, value: Any, ttl: Optional[int] = None):
        pass


class ExternalEchoLayer:
    COMPONENT_NAME = "external_echo_layer"

    def __init__(
        self,
        vanta_core: "UnifiedVantaCore",
        config: ExternalEchoLayerConfig,
        transcription_handler: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.vanta_core = vanta_core
        self.config = config
        self.transcription_handler = transcription_handler

        self.component_id: str = f"{self.COMPONENT_NAME}_{uuid.uuid4().hex[:8]}"
        self.active: bool = False
        self.lock = threading.RLock()
        self.output_handlers: list[Callable[[dict[str, Any]], None]] = []

        self.blt = RAGCompressionEngine()

        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_active: bool = False  # Fetch or create default dependencies
        self.echo_stream: EchoStreamInterface = self.vanta_core.get_component(
            self.config.echo_stream_component_name, _FallbackEchoStream()
        )
        self.meta_reflex_layer: Optional[MetaReflexLayerInterface] = self.vanta_core.get_component(
            self.config.meta_reflex_component_name, _FallbackMetaReflexLayer()
        )
        self.memory_braid: Optional[MemoryBraidInterface] = (
            self.vanta_core.get_component(  # Get MemoryBraid
                self.config.memory_braid_component_name, _FallbackMemoryBraid()
            )
        )

        self._connect_to_internal_echo_stream()

        self.vanta_core.register_component(self.COMPONENT_NAME, self, {"id": self.component_id})
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.initialized",
            self._get_init_event_data(),
            source=self.COMPONENT_NAME,
        )
        self._start_heartbeat_thread()
        logger.info(
            f"ExternalEchoLayer initialized: ID={self.component_id}. MemoryBraid Type: {type(self.memory_braid).__name__}"
        )

    def _get_init_event_data(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "record_seconds": self.config.record_seconds,
            "has_handler": self.transcription_handler is not None,
            "echo_connected": not isinstance(self.echo_stream, _FallbackEchoStream),
            "reflex_connected": self.meta_reflex_layer is not None
            and not isinstance(self.meta_reflex_layer, _FallbackMetaReflexLayer),
            "memory_braid_connected": self.memory_braid is not None
            and not isinstance(self.memory_braid, _FallbackMemoryBraid),
        }

    def _connect_to_internal_echo_stream(self):
        if self.echo_stream and not isinstance(
            self.echo_stream, _FallbackEchoStream
        ):  # Only connect if real
            try:
                self.echo_stream.add_source(self._echo_source_callback)
                self.echo_stream.add_sink(self._echo_sink_callback)
                logger.info(
                    f"{self.COMPONENT_NAME} registered with EchoStream: {type(self.echo_stream).__name__}."
                )
            except Exception as e:
                logger.error(f"Error connecting {self.COMPONENT_NAME} to EchoStream: {e}")
        elif isinstance(self.echo_stream, _FallbackEchoStream):
            logger.info(
                f"{self.COMPONENT_NAME} using _FallbackEchoStream. Callbacks registered nominally."
            )
            self.echo_stream.add_source(self._echo_source_callback)  # Call on stub is fine
            self.echo_stream.add_sink(self._echo_sink_callback)
        else:
            logger.warning(f"{self.COMPONENT_NAME}: EchoStream not available for connection.")

    def _start_heartbeat_thread(self):
        if self.heartbeat_active:
            return
        self.heartbeat_active = True

        def heartbeat_loop():
            while self.heartbeat_active:
                try:
                    pulse = 0.6 if self.active else 0.2  # Adjusted pulse values
                    status_str = "processing_external_io" if self.active else "idle"
                    heartbeat_data = {
                        "component_id": self.component_id,
                        "status": status_str,
                        "pulse_value": pulse,
                        "output_handlers": len(self.output_handlers),
                        "echo_stream_type": type(self.echo_stream).__name__,
                    }
                    self.vanta_core.publish_event(
                        f"{self.COMPONENT_NAME}.heartbeat",
                        heartbeat_data,
                        source=self.COMPONENT_NAME,
                    )
                except Exception as e:
                    logger.error(f"EEL Heartbeat error: {e}")
                time.sleep(self.config.heartbeat_interval_s)
            logger.info(f"{self.COMPONENT_NAME} heartbeat thread concluded.")

        self.heartbeat_thread = threading.Thread(
            target=heartbeat_loop, name=f"{self.COMPONENT_NAME}_Heartbeat", daemon=True
        )
        self.heartbeat_thread.start()
        logger.info(
            f"{self.COMPONENT_NAME} heartbeat thread initiated (interval: {self.config.heartbeat_interval_s}s)."
        )

    def _echo_source_callback(self, channel: str, message: dict[str, Any]) -> dict[str, Any]:
        # logger.debug(f"EEL EchoSource on '{channel}': {str(message)[:100]}...") # Too verbose for default
        return message

    def _echo_sink_callback(self, channel: str, message: dict[str, Any]) -> None:
        # logger.debug(f"EEL EchoSink on '{channel}': {str(message)[:100]}...") # Too verbose for default
        # If message comes from a specific channel EEL should react to by sending to output handlers
        if (
            channel == self.config.echo_stream_component_name + ".output" and self.active
        ):  # Example channel naming
            output_data = {
                "text": message.get("text", message.get("content", "")),
                "channel_origin": channel,
                "timestamp": message.get("timestamp", time.time()),
                "metadata": message.get("metadata", {}),
                "source_echo_message_id": message.get("id"),
            }
            self.handle_output(output_data)

        if (
            self.meta_reflex_layer
            and hasattr(self.meta_reflex_layer, "process_echo")
            and not isinstance(self.meta_reflex_layer, _FallbackMetaReflexLayer)
        ):
            try:
                self.meta_reflex_layer.process_echo(message)
            except Exception as e:
                logger.error(f"Error forwarding echo to MetaReflexLayer: {e}")

    def start(self):  # Same as before, event published
        with self.lock:
            if self.active:
                logger.info(f"{self.COMPONENT_NAME} already active.")
                return
            self.active = True
        logger.info(f"{self.COMPONENT_NAME} started.")
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.started", {}, source=self.COMPONENT_NAME
        )

    def stop(self):  # Same as before, event published
        with self.lock:
            if not self.active:
                logger.info(f"{self.COMPONENT_NAME} already inactive.")
                return
            self.active = False
        logger.info(f"{self.COMPONENT_NAME} stopped.")
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.stopped", {}, source=self.COMPONENT_NAME
        )

    def terminate(self):  # Same as before, event published
        logger.info(f"{self.COMPONENT_NAME} terminating...")
        self.stop()
        if self.heartbeat_active:
            self.heartbeat_active = False
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=1.0)
        logger.info(f"{self.COMPONENT_NAME} terminated.")
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.terminated", {}, source=self.COMPONENT_NAME
        )

    def set_transcription_handler(
        self, handler: Callable[[dict[str, Any]], None]
    ):  # Same as before
        if not callable(handler):
            logger.warning("Invalid transcription handler.")
            return
        self.transcription_handler = handler
        logger.info(f"{self.COMPONENT_NAME}: Transcription handler set/updated.")

    def add_output_handler(self, handler: Callable[[dict[str, Any]], None]):  # Same as before
        if not callable(handler):
            logger.warning("Invalid output handler.")
            return
        with self.lock:
            if handler not in self.output_handlers:
                self.output_handlers.append(handler)
        logger.info(
            f"{self.COMPONENT_NAME}: Output handler added: {getattr(handler, '__name__', 'unnamed_handler')}"
        )

    def remove_output_handler(self, handler: Callable[[dict[str, Any]], None]):  # Same as before
        with self.lock:
            if handler in self.output_handlers:
                self.output_handlers.remove(handler)
        logger.info(
            f"{self.COMPONENT_NAME}: Output handler removed: {getattr(handler, '__name__', 'unnamed_handler')}"
        )

    def handle_output(self, output_data: dict[str, Any]):
        if not self.active:
            logger.warning(f"Output handling skipped: {self.COMPONENT_NAME} not active.")
            return
        text_preview = str(output_data.get("text", output_data.get("content", "N/A_text")))[:50]
        logger.debug(f"{self.COMPONENT_NAME} forwarding output: {text_preview}...")
        with self.lock:
            handlers_copy = list(self.output_handlers)

        for handler in handlers_copy:
            try:
                handler(output_data)
            except Exception as e:
                logger.error(
                    f"Error in output handler '{getattr(handler, '__name__', 'unknown')}': {e}"
                )  # Imprint this output to MemoryBraid
        if self.memory_braid and not isinstance(self.memory_braid, _FallbackMemoryBraid):
            try:
                braid_key = (
                    f"eel:output:{output_data.get('timestamp', time.time())}:{uuid.uuid4().hex[:6]}"
                )
                self.memory_braid.imprint(
                    braid_key, output_data, ttl_seconds=self.config.default_event_ttl_s
                )
                logger.debug(f"Imprinted EEL output to MemoryBraid with key '{braid_key}'.")
            except Exception as e:
                logger.error(f"Failed to imprint EEL output to MemoryBraid: {e}")

        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.output.sent",
            {"text_preview": text_preview},
            source=self.COMPONENT_NAME,
        )

    def get_health(self) -> dict[str, Any]:  # Updated health status
        return {
            "status": "active_healthy" if self.active else "inactive",
            "component_id": self.component_id,
            "echo_stream_type": type(self.echo_stream).__name__,
            "meta_reflex_type": type(self.meta_reflex_layer).__name__
            if self.meta_reflex_layer
            else "None",
            "memory_braid_type": type(self.memory_braid).__name__ if self.memory_braid else "None",
            "output_handlers": len(self.output_handlers),
            "timestamp": time.time(),
        }

    def simulate_transcription(
        self, text: str, confidence: float = 0.9, source: str = "simulated_mic_input"
    ):
        if not self.active:
            logger.warning(f"Transcription simulation skipped: {self.COMPONENT_NAME} inactive.")
            return
        if not self.transcription_handler:
            logger.warning(
                f"Transcription simulation skipped: No handler registered for {self.COMPONENT_NAME}."
            )
            return

        transcription_data = {
            "text": text,
            "confidence": confidence,
            "timestamp": time.time(),
            "source": source,
            "id": f"trans_{uuid.uuid4().hex[:8]}",
        }
        try:
            self.transcription_handler(transcription_data)
            logger.info(f"{self.COMPONENT_NAME} processed simulated transcription: {text[:50]}...")

            event_data_for_vanta = transcription_data.copy()
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.transcription.processed",
                event_data_for_vanta,
                source=self.COMPONENT_NAME,
            )  # Imprint transcription to MemoryBraid
            if self.memory_braid and not isinstance(self.memory_braid, _FallbackMemoryBraid):
                try:
                    # Create a more structured key for MemoryBraid
                    braid_key = f"eel:transcription:{transcription_data['timestamp']}:{transcription_data['id']}"
                    self.memory_braid.imprint(
                        braid_key,
                        transcription_data,
                        ttl_seconds=self.config.default_event_ttl_s,
                    )
                    logger.debug(f"Imprinted transcription to MemoryBraid with key '{braid_key}'.")
                except Exception as e:
                    logger.error(
                        f"Failed to imprint transcription to MemoryBraid: {e}"
                    )  # If echo_stream is "real", also emit the raw/processed data to it
            if self.echo_stream and not isinstance(self.echo_stream, _FallbackEchoStream):
                self.echo_stream.emit(
                    channel=f"{self.config.echo_stream_component_name}.transcription_input",
                    text=text,
                    metadata=transcription_data,
                )

        except Exception as e:
            logger.error(f"Error in transcription handler during simulation: {e}")

    def connect_to_external_system(
        self, system_name: str, connection_config: Optional[dict[str, Any]] = None
    ):  # No change needed
        logger.info(
            f"{self.COMPONENT_NAME} VantaCore: Simulating connection to external system: {system_name}"
        )
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.external_system.connect_attempt",
            {"system_name": system_name, "config": connection_config},
            source=self.COMPONENT_NAME,
        )
        return True

    def disconnect_from_external_system(self, system_name: str):  # No change needed
        logger.info(
            f"{self.COMPONENT_NAME} VantaCore: Simulating disconnection from external system: {system_name}"
        )
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.external_system.disconnect_attempt",
            {"system_name": system_name},
            source=self.COMPONENT_NAME,
        )
        return True

    def emit_to_configured_echo_stream(
        self,
        text: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
        channel_suffix: str = "data_out",
    ):
        """Emit to the configured echo_stream component."""
        if not self.echo_stream or isinstance(self.echo_stream, _FallbackEchoStream):
            logger.warning(
                f"{self.COMPONENT_NAME}: Cannot emit. EchoStream is a stub or not configured."
            )
            return False
        try:
            full_metadata = metadata or {}
            full_metadata.update({"_vanta_eel_source_id": self.component_id})
            target_channel = f"{self.config.echo_stream_component_name}.{channel_suffix}"
            compressed = self.blt.compress(text) if text is not None else None
            self.echo_stream.emit(channel=target_channel, text=compressed, metadata=full_metadata)
            logger.debug(
                f"{self.COMPONENT_NAME} emitted to configured EchoStream on channel '{target_channel}'."
            )
            return True
        except Exception as e:
            logger.error(f"Error emitting to configured EchoStream from {self.COMPONENT_NAME}: {e}")
            return False


# --- Singleton Accessor ---
_vanta_external_echo_layer_instance: Optional[ExternalEchoLayer] = None
_vanta_instance_lock = threading.Lock()


def get_vanta_external_echo_layer(
    vanta_core: Optional["UnifiedVantaCore"] = None,
    config: Optional[ExternalEchoLayerConfig] = None,
) -> ExternalEchoLayer:
    global _vanta_external_echo_layer_instance
    if _vanta_external_echo_layer_instance is None:
        with _vanta_instance_lock:
            if _vanta_external_echo_layer_instance is None:
                if vanta_core is None or config is None:
                    msg = "ExternalEchoLayer for VantaCore must be initialized with VantaCore and Config instances on first call."
                    logger.error(msg)
                    raise ValueError(msg)
                _vanta_external_echo_layer_instance = ExternalEchoLayer(
                    vanta_core=vanta_core, config=config
                )
    return _vanta_external_echo_layer_instance


# --- Example Usage (Adapted for VantaCore) ---
if __name__ == "__main__":
    main_logger_eel_v = logging.getLogger("VantaExternalEchoExample")
    main_logger_eel_v.setLevel(
        logging.DEBUG
    )  # Main example logger    main_logger_eel_v.info("--- Starting Vanta External Echo Layer Example ---")

    # Lazy import to avoid circular dependency
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

    vanta_system_eel_v = UnifiedVantaCore()  # Initialize VantaCore

    # Example: Register a mock MemoryBraid for EEL to find and use
    class MyMemoryBraid(MemoryBraidInterface):
        def imprint(self, key: str, value: Any, ttl: Optional[int] = None):
            main_logger_eel_v.info(
                f"MyMemoryBraid: Imprint key='{key}', value='{str(value)[:50]}...'"
            )

    eel_conf = ExternalEchoLayerConfig(
        log_level="DEBUG",
        heartbeat_interval_s=3,
        memory_braid_component_name="global_memory_braid",
    )
    vanta_system_eel_v.register_component("global_memory_braid", MyMemoryBraid())  # Register it

    try:  # Use try-except for first singleton get, as it now requires params
        eel = get_vanta_external_echo_layer(vanta_core=vanta_system_eel_v, config=eel_conf)
    except ValueError as e:
        main_logger_eel_v.critical(f"Could not start EEL example: {e}")
        exit()

    def simple_handler(data: dict[str, Any]):
        main_logger_eel_v.info(
            f"SIMPLE HANDLER GOT: Text='{data.get('text')}' from source='{data.get('source')}'"
        )

    eel.set_transcription_handler(simple_handler)
    eel.start()
    main_logger_eel_v.info("Vanta ExternalEchoLayer running...")

    try:
        eel.simulate_transcription("Hello VantaCore world, this is a test.")
        time.sleep(1)
        eel.handle_output(
            {
                "text": "This is an output EEL is handling.",
                "source": "internal_system_A",
            }
        )
        time.sleep(eel_conf.heartbeat_interval_s * 1.5)  # Let heartbeat run
        main_logger_eel_v.info(
            f"Current EEL Health: {json.dumps(eel.get_health(), indent=2, default=str)}"
        )
    except KeyboardInterrupt:
        main_logger_eel_v.info("Interrupted.")
    finally:
        main_logger_eel_v.info("Terminating Vanta ExternalEchoLayer.")
        eel.terminate()
    main_logger_eel_v.info("--- Vanta External Echo Layer Example Finished ---")
