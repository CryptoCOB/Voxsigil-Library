"""
Vanta Interaction Manager (formerly CheckInManager)

Handles user state tracking, conversation context, and integrates
Speech-to-Text (STT) via Vosk and Text-to-Speech (TTS) for voice interaction
with the VantaCore ecosystem.
"""

# HOLO-1.5 Recursive Symbolic Cognition Mesh imports
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from .base import BaseCore, CognitiveMeshRole, vanta_core_module

logger = logging.getLogger("VantaInteractionManager")

# VantaCore Integration
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

    HAVE_VANTA_CORE = True
    VantaCoreType = Type[UnifiedVantaCore]
except ImportError:
    HAVE_VANTA_CORE = False
    UnifiedVantaCore = None  # Minimal VantaCore stub for standalone functionality / testing

    class VantaCore:
        def get_component(self, name: str, default: Any = None):
            logger.warning(f"[VantaCoreStubEIM] get_component for '{name}' returning default.")
            return default

        def register_component(self, name: str, comp: Any, meta: Optional[dict] = None):
            logger.info(f"[VantaCoreStubEIM] Component '{name}' registered (stub).")

        def publish_event(
            self, etype: str, data: Optional[dict] = None, source: Optional[str] = None
        ):
            logger.debug(f"[VantaCoreStubEIM] Event: {etype}, Data: {data}, Src: {source}")

    VantaCoreType = Type[VantaCore]
    logger.debug("VantaCore class not found, using a stub. Full integration may be affected.")

# STT (Vosk) and Audio Input Dependencies
try:
    import vosk

    HAVE_VOSK = True
except ImportError:
    HAVE_VOSK = False
    vosk = None
    logging.getLogger("VantaInteractionManager").warning(
        "Vosk library not found. STT features will be disabled. Install with: uv add vosk"
    )

try:
    import numpy as np
    import sounddevice as sd

    HAVE_SOUNDDEVICE = True
except ImportError:
    HAVE_SOUNDDEVICE = False
    sd = None
    np = None
    logging.getLogger("VantaInteractionManager").warning(
        "Sounddevice library not found. Audio recording for STT will be disabled. Install with: uv add sounddevice numpy"
    )

# TTS (pyttsx3) Dependencies
try:
    import pyttsx3

    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False
    pyttsx3 = None
    logging.getLogger("VantaInteractionManager").warning(
        "pyttsx3 library not found. TTS features will be disabled. Install with: uv add pyttsx3"
    )


@dataclass
class VantaInteractionManagerConfig:
    log_level: str = "INFO"
    # Check-in related (can be kept for passive monitoring or adapted)
    checkin_interval_s: int = 180
    inactive_checkin_interval_s: int = 600
    forced_checkin_interval_s: int = 1800  # Max time before a check-in event
    # User activity thresholds
    idle_threshold_s: int = 300
    away_threshold_s: int = 1800
    sleep_threshold_s: int = 28800  # 8 hours
    # STT/Vosk config
    vosk_model_path: str = ""  # Path to Vosk model directory (empty = auto-download)
    stt_default_listen_duration_s: int = 7
    stt_silence_threshold_s: float = 2.0  # Seconds of silence to stop early
    # TTS/pyttsx3 config
    tts_default_rate: int = 150  # Words per minute
    tts_default_volume: float = 0.9  # 0.0 to 1.0
    # Conversation context
    conversation_prune_interval_s: int = 3600  # 1 hour
    stale_conversation_max_age_s: int = 86400  # 24 hours


@dataclass
class ConversationThread:
    topic: str = ""
    last_intent: str = ""
    incomplete: bool = True
    last_active: float = field(default_factory=time.time)
    loop_tag: str = ""
    meta_state: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "last_intent": self.last_intent,
            "incomplete": self.incomplete,
            "last_active": self.last_active,
            "loop_tag": self.loop_tag,
            "meta_state": self.meta_state,
            "message_count": len(self.messages),
        }


@vanta_core_module(
    name="vanta_interaction_manager",
    subsystem="system_management",
    mesh_role=CognitiveMeshRole.MANAGER,
    description="User interaction manager with voice I/O, conversation tracking, and state monitoring",
    capabilities=[
        "voice_input",
        "voice_output",
        "state_tracking",
        "conversation_management",
        "user_monitoring",
        "stt_processing",
        "tts_synthesis",
    ],
    cognitive_load=2.5,
    symbolic_depth=2,
    collaboration_patterns=["user_feedback", "state_coordination", "voice_interaction"],
)
class VantaInteractionManager(BaseCore):
    COMPONENT_NAME = "interaction_manager"

    def __init__(
        self,
        vanta_core: Any,  # Changed from VantaCore to Any to fix type issues
        config: VantaInteractionManagerConfig,
        model_manager: Optional[Any] = None,
        passive_state_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.vanta_core = vanta_core
        self.config = config
        self.model_manager = model_manager
        self.passive_state_update_callback = passive_state_update_callback

        numeric_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # User state tracking
        self.last_user_interaction_ts: float = time.time()
        self.last_checkin_event_ts: float = 0.0
        self.explicit_user_state: str = "unknown"
        self.interaction_count: int = 0
        self.active_conversations: Dict[str, ConversationThread] = {}
        self.current_topic: str = "general"  # STT/Vosk state
        self._vosk_model: Optional[Any] = None
        self._vosk_recognizer: Optional[Any] = None
        if HAVE_VOSK and vosk:
            try:
                if self.config.vosk_model_path:
                    logger.info(f"Loading Vosk model from: {self.config.vosk_model_path}...")
                    self._vosk_model = vosk.Model(self.config.vosk_model_path)
                else:
                    # Use default small model - will auto-download if needed
                    logger.info("Loading default Vosk model (en-us)...")
                    self._vosk_model = vosk.Model(lang="en-us")

                self._vosk_recognizer = vosk.KaldiRecognizer(self._vosk_model, 16000)
                logger.info("Vosk model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Vosk model: {e}")
                self._vosk_model = None
                self._vosk_recognizer = None  # TTS/pyttsx3 state
        self._tts_engine: Optional[Any] = None
        if HAVE_PYTTSX3 and pyttsx3:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", self.config.tts_default_rate)
                self._tts_engine.setProperty("volume", self.config.tts_default_volume)
                logger.info("pyttsx3 TTS engine initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3 TTS engine: {e}")

        # Thread for passive monitoring
        self.running_passive_loop: bool = False
        self.passive_loop_thread: Optional[threading.Thread] = None
        self.thread_lock = threading.RLock()

        self.vanta_core.register_component(
            self.COMPONENT_NAME, self, {"type": "user_io_voice_manager"}
        )
        logger.info(
            f"{self.COMPONENT_NAME} initialized. STT: {'Vosk' if self._vosk_recognizer else 'Disabled'}, TTS: {'pyttsx3' if self._tts_engine else 'Disabled'}"
        )

    async def initialize(self) -> bool:
        """Initialize the VantaInteractionManager for HOLO-1.5 BaseCore compliance."""
        try:
            # Initialize audio system check
            stt_available = self._vosk_recognizer is not None
            tts_available = self._tts_engine is not None

            logger.info("VantaInteractionManager initialized with HOLO-1.5 enhancement")
            logger.info(f"Voice capabilities - STT: {stt_available}, TTS: {tts_available}")

            # Start passive monitoring if callback is available
            if self.passive_state_update_callback:
                self.start_passive_monitoring()
                logger.info("Passive monitoring started automatically")

            return True

        except Exception as e:
            logger.error(f"Error initializing VantaInteractionManager: {e}")
            return False

    def start_passive_monitoring(self) -> None:
        """Starts the passive monitoring loop if a callback is provided."""
        if not self.passive_state_update_callback:
            logger.info(
                "No passive_state_update_callback set. Passive monitoring loop not started."
            )
            return

        with self.thread_lock:
            if self.running_passive_loop:
                logger.warning(f"{self.COMPONENT_NAME} passive monitoring loop already running.")
                return
            self.running_passive_loop = True
            self.passive_loop_thread = threading.Thread(
                target=self._run_passive_loop,
                daemon=True,
                name="VantaInteractionPassiveLoop",
            )
            self.passive_loop_thread.start()
            logger.info(f"{self.COMPONENT_NAME} passive monitoring loop started.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.passive_monitoring.started",
                {},
                source=self.COMPONENT_NAME,
            )

    def stop_passive_monitoring(self) -> None:
        """Stops the passive monitoring loop."""
        with self.thread_lock:
            if not self.running_passive_loop:
                logger.info(f"{self.COMPONENT_NAME} passive monitoring loop not running.")
                return
            self.running_passive_loop = False

        if self.passive_loop_thread and self.passive_loop_thread.is_alive():
            logger.info("Stopping VantaInteractionManager passive loop...")
            try:
                self.passive_loop_thread.join(timeout=5.0)
            except Exception as e:
                logger.error(f"Error joining passive loop thread: {e}")
            if self.passive_loop_thread.is_alive():
                logger.warning("Passive loop thread did not stop cleanly.")

        self.passive_loop_thread = None
        logger.info(f"{self.COMPONENT_NAME} passive monitoring loop stopped.")
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.passive_monitoring.stopped",
            {},
            source=self.COMPONENT_NAME,
        )

    def _run_passive_loop(self) -> None:
        """Runs the passive monitoring loop for check-ins or state updates."""
        logger.info(f"{self.COMPONENT_NAME} passive monitoring loop entering run state.")
        last_prune_time = time.monotonic()

        while self.running_passive_loop:
            loop_start_time = time.monotonic()
            try:
                now_mono = time.monotonic()

                if self._should_run_passive_checkin(now_mono):
                    self._perform_passive_checkin_event()
                    self.last_checkin_event_ts = now_mono

                if now_mono - last_prune_time > self.config.conversation_prune_interval_s:
                    self.prune_stale_conversations()
                    last_prune_time = now_mono
            except Exception as e:
                logger.error(f"Error in {self.COMPONENT_NAME} passive loop: {e}", exc_info=True)

            elapsed = time.monotonic() - loop_start_time
            sleep_interval = min(
                10,
                self.config.checkin_interval_s / 10,
                self.config.conversation_prune_interval_s / 10,
            )
            sleep_time = max(1.0, sleep_interval - elapsed)
            if self.running_passive_loop:
                time.sleep(sleep_time)
        logger.info(f"{self.COMPONENT_NAME} passive monitoring loop exited.")

    def _should_run_passive_checkin(self, current_monotonic_time: float) -> bool:
        if not self.passive_state_update_callback:
            return False

        time_since_last_event = current_monotonic_time - self.last_checkin_event_ts
        if self.last_checkin_event_ts == 0.0:
            time_since_last_event = float("inf")

        if time_since_last_event >= self.config.forced_checkin_interval_s:
            logger.debug(
                f"Forcing passive check-in event due to max interval ({self.config.forced_checkin_interval_s}s)."
            )
            return True

        user_state = self.get_user_activity_state()
        interval_for_state = (
            self.config.checkin_interval_s
            if user_state == "active"
            else self.config.inactive_checkin_interval_s
        )

        return time_since_last_event >= interval_for_state

    def _perform_passive_checkin_event(self) -> None:
        if not self.passive_state_update_callback:
            return

        checkin_data = {
            "event_origin": self.COMPONENT_NAME,
            "type": "passive_user_state_update",
            "timestamp": time.time(),
            "user_activity_state": self.get_user_activity_state(),
            "last_interaction_timestamp": self.last_user_interaction_ts,
            "current_interaction_count": self.interaction_count,
            "active_conversations_summary": {
                t: cd.to_dict() for t, cd in self.active_conversations.items() if cd.incomplete
            },
            "current_topic": self.current_topic,
        }
        try:
            logger.debug(
                f"Performing passive check-in event: User state {checkin_data['user_activity_state']}"
            )
            self.passive_state_update_callback(checkin_data)
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.passive_checkin",
                checkin_data,
                source=self.COMPONENT_NAME,
            )
        except Exception as e:
            logger.error(f"Error calling passive_state_update_callback: {e}")

    def update_user_interaction(
        self,
        interaction_type: str = "unspecified_voice_or_text",
        topic: Optional[str] = None,
        message_text: Optional[str] = None,
        message_role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        explicit_state_override: Optional[str] = None,
    ) -> None:
        """Tracks user interaction, updating state and conversation context."""
        now = time.time()
        current_metadata = metadata or {}
        self.last_user_interaction_ts = now
        self.interaction_count += 1

        if explicit_state_override:
            self.set_explicit_user_state(explicit_state_override)
        elif self.explicit_user_state != "active":
            self.set_explicit_user_state("active")

        effective_topic = topic or self.current_topic or "general"
        if not self.current_topic and topic:
            self.current_topic = topic
        elif not topic and not self.current_topic:
            self.current_topic = "general"

        conv_thread = self.active_conversations.get(effective_topic)
        if not conv_thread:
            conv_thread = ConversationThread(topic=effective_topic)
            self.active_conversations[effective_topic] = conv_thread

        conv_thread.last_active = now
        if "intent" in current_metadata:
            conv_thread.last_intent = current_metadata["intent"]
        if "loop_tag" in current_metadata:
            conv_thread.loop_tag = current_metadata["loop_tag"]
        if "meta_state" in current_metadata:
            conv_thread.meta_state = current_metadata["meta_state"]
        if current_metadata.get("conversation_completed", False):
            conv_thread.incomplete = False
        else:
            conv_thread.incomplete = True

        if message_text:
            conv_thread.messages.append(
                {"role": message_role, "content": message_text, "timestamp": now}
            )

        interaction_event_data = {
            "type": interaction_type,
            "topic": effective_topic,
            "timestamp": now,
            "new_user_state": self.explicit_user_state,
            "message_role": message_role,
            "message_preview": message_text[:50] if message_text else None,
            **current_metadata,
        }
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.user_interaction",
            interaction_event_data,
            source=self.COMPONENT_NAME,
        )
        logger.debug(
            f"User interaction tracked: Type='{interaction_type}', Topic='{effective_topic}', State='{self.explicit_user_state}'"
        )

    def get_user_activity_state(self) -> str:
        """Determines user activity state based on thresholds and explicit state."""
        if self.explicit_user_state not in ["unknown", "active"]:
            return self.explicit_user_state

        time_since_last_interaction = time.time() - self.last_user_interaction_ts
        if time_since_last_interaction > self.config.sleep_threshold_s:
            return "sleeping"
        if time_since_last_interaction > self.config.away_threshold_s:
            return "away"
        if time_since_last_interaction > self.config.idle_threshold_s:
            return "idle"
        return "active"

    def set_explicit_user_state(self, state: str) -> bool:
        """Allows external setting of the user's state."""
        valid_states = ["unknown", "active", "idle", "away", "sleeping"]
        if state.lower() in valid_states:
            old_state = self.explicit_user_state
            self.explicit_user_state = state.lower()
            if old_state != self.explicit_user_state:
                logger.info(
                    f"User state explicitly set to: {self.explicit_user_state} (was: {old_state})"
                )
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.user_state.set",
                    {
                        "state": self.explicit_user_state,
                        "previous_state": old_state,
                        "source": "explicit_set",
                    },
                    source=self.COMPONENT_NAME,
                )
            return True
        logger.warning(f"Attempt to set invalid user state: {state}")
        return False  # --- STT (Speech-to-Text) Methods ---

    def listen_and_transcribe(
        self, duration_s: Optional[int] = None, silence_timeout_s: Optional[float] = None
    ) -> Optional[str]:
        """Listens to microphone input and returns transcribed text using Vosk."""
        if not self._vosk_recognizer or not self._vosk_model:
            logger.error("Vosk STT model not loaded. Cannot listen and transcribe.")
            return None
        if not HAVE_SOUNDDEVICE or sd is None or np is None:
            logger.error("Sounddevice/Numpy not available. Cannot record audio for STT.")
            return None

        listen_duration = (
            duration_s if duration_s is not None else self.config.stt_default_listen_duration_s
        )
        effective_silence_timeout = (
            silence_timeout_s
            if silence_timeout_s is not None
            else self.config.stt_silence_threshold_s
        )

        sample_rate = 16000  # Vosk prefers 16kHz
        logger.info(
            f"Listening for command ({listen_duration}s, silence timeout {effective_silence_timeout}s)..."
        )
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.stt.listening_start",
            {"duration_s": listen_duration},
            source=self.COMPONENT_NAME,
        )

        try:
            # Record audio
            audio_data = sd.rec(
                int(listen_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            audio_np = np.squeeze(audio_data)

            if audio_np.size == 0:
                logger.info("No audio data recorded.")
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.stt.no_audio_data",
                    {},
                    source=self.COMPONENT_NAME,
                )
                return None

            logger.info(
                f"Audio recorded ({len(audio_np) / sample_rate:.2f}s). Transcribing with Vosk..."
            )
            stt_start_time = time.monotonic()

            # Process audio with Vosk
            self._vosk_recognizer.AcceptWaveform(audio_np.tobytes())
            result = self._vosk_recognizer.FinalResult()
            stt_duration_ms = (time.monotonic() - stt_start_time) * 1000

            # Parse Vosk result
            result_dict = json.loads(result)
            transcribed_text: str = result_dict.get("text", "").strip()

            if transcribed_text:
                logger.info(
                    f"Vosk STT Result: '{transcribed_text}' (Processed in {stt_duration_ms:.0f}ms)"
                )
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.stt.transcription_success",
                    {
                        "text": transcribed_text,
                        "model": "vosk",
                        "processing_ms": stt_duration_ms,
                        "audio_duration_s": len(audio_np) / sample_rate,
                    },
                    source=self.COMPONENT_NAME,
                )
                self.update_user_interaction(
                    interaction_type="voice_command_stt",
                    message_text=transcribed_text,
                    message_role="user",
                    metadata={"stt_confidence": result_dict.get("confidence", 0.0)},
                )
                return transcribed_text
            else:
                logger.info("Vosk STT: No speech detected or empty transcription.")
                self.vanta_core.publish_event(
                    f"{self.COMPONENT_NAME}.stt.no_speech_detected",
                    {"model": "vosk"},
                    source=self.COMPONENT_NAME,
                )
                return None
        except Exception as e:
            logger.error(f"Error during STT listen/transcribe: {e}", exc_info=True)
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.stt.error",
                {"error": str(e)},
                source=self.COMPONENT_NAME,
            )
            return None

    # --- TTS (Text-to-Speech) Methods ---
    def speak_text(self, text_to_speak: str, voice_config: Optional[Dict[str, Any]] = None) -> bool:
        """Speaks the given text using the configured TTS engine."""
        if not self._tts_engine:
            logger.error("pyttsx3 TTS engine not initialized. Cannot speak.")
            return False
        if not text_to_speak:
            logger.warning("TTS: No text provided to speak.")
            return False

        try:
            logger.info(f"TTS Speaking: '{text_to_speak[:70]}...'")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.tts.speaking_start",
                {"text_preview": text_to_speak[:70]},
                source=self.COMPONENT_NAME,
            )

            # Apply voice config if provided
            current_rate = self._tts_engine.getProperty("rate")
            current_volume = self._tts_engine.getProperty("volume")

            if voice_config:
                rate = voice_config.get("rate", self.config.tts_default_rate)
                volume = voice_config.get("volume", self.config.tts_default_volume)
                self._tts_engine.setProperty("rate", rate)
                self._tts_engine.setProperty("volume", volume)

            self._tts_engine.say(text_to_speak)
            self._tts_engine.runAndWait()

            # Restore original settings if they were changed
            if voice_config:
                self._tts_engine.setProperty("rate", current_rate)
                self._tts_engine.setProperty("volume", current_volume)

            logger.info("TTS Speaking finished.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.tts.speaking_end",
                {"text_length": len(text_to_speak)},
                source=self.COMPONENT_NAME,
            )
            self.update_user_interaction(
                interaction_type="voice_response_tts",
                message_text=text_to_speak,
                message_role="assistant",
                metadata={"tts_engine": "pyttsx3"},
            )
            return True
        except Exception as e:
            logger.error(f"Error during TTS speak_text: {e}", exc_info=True)
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.tts.error",
                {"error": str(e)},
                source=self.COMPONENT_NAME,
            )
            return False

    # --- Conversation Management ---
    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        return {topic: thread.to_dict() for topic, thread in self.active_conversations.items()}

    def prune_stale_conversations(self) -> int:
        max_age = self.config.stale_conversation_max_age_s
        now = time.time()
        pruned_count = 0
        topics_to_prune = [
            topic
            for topic, thread in self.active_conversations.items()
            if now - thread.last_active > max_age
        ]
        for topic in topics_to_prune:
            del self.active_conversations[topic]
            pruned_count += 1
            logger.info(f"Pruned stale conversation topic: {topic}")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.conversation.pruned",
                {"topic": topic},
                source=self.COMPONENT_NAME,
            )
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} stale conversations.")
        return pruned_count

    def mark_conversation_complete(self, topic: str) -> bool:
        if topic in self.active_conversations:
            self.active_conversations[topic].incomplete = False
            logger.info(f"Conversation topic '{topic}' marked complete.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.conversation.completed",
                {"topic": topic},
                source=self.COMPONENT_NAME,
            )
            return True
        return False

    def get_conversation_history(
        self, topic: str, limit: int = 20
    ) -> Optional[List[Dict[str, Any]]]:
        thread = self.active_conversations.get(topic)
        return thread.messages[-limit:] if thread else None

    def get_current_context(self) -> Dict[str, Any]:
        """Provides current interaction state as context."""
        user_act_state = self.get_user_activity_state()
        active_convs = {
            t: th.to_dict() for t, th in self.active_conversations.items() if th.incomplete
        }
        return {
            "user_activity_state": user_act_state,
            "last_interaction_timestamp": self.last_user_interaction_ts,
            "time_since_last_interaction_s": time.time() - self.last_user_interaction_ts,
            "current_interaction_count_session": self.interaction_count,
            "active_incomplete_conversations": active_convs,
            "current_topic_focus": self.current_topic,
            "timestamp": time.time(),
            "_source_component": self.COMPONENT_NAME,
        }

    def get_provider_name(self) -> str:
        return self.COMPONENT_NAME


if __name__ == "__main__":
    example_logger = logging.getLogger("VantaInteractionManagerExample")
    example_logger.setLevel(logging.DEBUG)

    example_logger.info("--- Vanta Interaction Manager Example ---")  # 1. Initialize VantaCore
    if HAVE_VANTA_CORE and UnifiedVantaCore is not None:
        vanta_system_im = UnifiedVantaCore()
    else:
        vanta_system_im = VantaCore()

    # 2. Configuration
    im_config = VantaInteractionManagerConfig(
        log_level="DEBUG",
        checkin_interval_s=5,
        vosk_model_path="",
        stt_default_listen_duration_s=4,
    )

    # 3. Passive state update callback (optional)
    def print_passive_update(data: Dict[str, Any]):
        example_logger.info(
            f"Passive Update Received: State='{data.get('user_activity_state')}', Topic='{data.get('current_topic')}'"
        )

    # 4. Instantiate Manager
    interaction_mgr = VantaInteractionManager(
        vanta_core=vanta_system_im,
        config=im_config,
        passive_state_update_callback=print_passive_update,
    )
    interaction_mgr.start_passive_monitoring()

    # 5. Simulate usage
    example_logger.info("Simulating user interaction...")
    interaction_mgr.update_user_interaction(
        interaction_type="gui_click",
        topic="gridformer_status",
        message_text="User clicked status button.",
        message_role="user",
    )
    current_context_for_tot = interaction_mgr.get_current_context()
    example_logger.info(
        f"Current Context for ToT: {json.dumps(current_context_for_tot, indent=2, default=str)}"
    )

    if interaction_mgr._tts_engine:
        interaction_mgr.speak_text(
            "Hello! I am Vanta. How can I assist you with GRID-Former today?"
        )
    else:
        example_logger.warning("TTS engine not available, skipping speak_text demo.")

    if interaction_mgr._vosk_recognizer:
        example_logger.info(
            "Please speak a command for Vanta (e.g., 'Vanta check training logs')... You have 4 seconds."
        )
        transcribed_command = interaction_mgr.listen_and_transcribe()
        if transcribed_command:
            example_logger.info(f"YOU SAID (Transcribed): '{transcribed_command}'")
            interaction_mgr.update_user_interaction(
                interaction_type="voice_command_received",
                message_text=transcribed_command,
                message_role="user",
                topic="vanta_command",
            )
            if "training" in transcribed_command.lower():
                interaction_mgr.speak_text(
                    f"Okay, I will check the training logs for GRID-Former based on your command: {transcribed_command}"
                )
            else:
                interaction_mgr.speak_text(
                    f"I understood your command as: {transcribed_command}. I will process it."
                )
        else:
            example_logger.info("No command transcribed or an error occurred.")
            interaction_mgr.speak_text("I didn't catch that. Could you please repeat?")
    else:
        example_logger.warning("Vosk model not available, skipping listen_and_transcribe demo.")

    try:
        time.sleep(im_config.checkin_interval_s * 2.5)
    except KeyboardInterrupt:
        example_logger.info("Interrupted by user.")
    finally:
        example_logger.info("Stopping Vanta Interaction Manager passive monitoring...")
        interaction_mgr.stop_passive_monitoring()
        example_logger.info("--- Vanta Interaction Manager Example Finished ---")
