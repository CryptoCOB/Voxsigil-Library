# sleep_time_compute_core.py
"""
SleepTimeCompute (Cognitive Rhythm Core) - Standalone Version

This module implements the SleepTimeCompute core, which manages cognitive rhythms
for efficient background processing during rest phases. It provides capabilities for:
- Rest phase replay for memory processing.
- Pattern compression simulation.
- Reflective learning simulation.
- Dream-based scenario simulation.
- Memory consolidation simulation.

This version is self-contained and does not rely on an external SDK.
"""

import argparse
import json
import logging
import random
import threading
import time
from collections import deque  # Removed defaultdict as it wasn't used
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Union  # Added Deque

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType  # add import

# --- Logger Setup ---
# Ensures the logger is configured. In a larger application, this might be done at the application root.
logger_stc = logging.getLogger("SleepTimeComputeCore")
if (
    not logger_stc.hasHandlers() and not logging.getLogger().hasHandlers()
):  # Check root logger too
    handler = logging.StreamHandler()
    # More detailed formatter for production-grade
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s Tid:%(thread)d %(funcName)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger_stc.addHandler(handler)
    logger_stc.setLevel(logging.INFO)  # Default level, can be changed by application


# --- Enums (as before, good for clarity) ---
class CognitiveState(Enum):
    """Possible cognitive states for the SleepTimeCompute."""

    ACTIVE = "active"
    REST = "rest"
    DEEP_REST = "deep_rest"
    DREAMING = "dreaming"  # (Sub-state of DEEP_REST or REST)
    LEARNING = "learning"  # (Sub-state of DEEP_REST or REST)
    MAINTENANCE = "maintenance"  # For internal tasks like pruning
    ERROR = "error"


class ProcessingPriority(Enum):
    """Priority levels for background processing tasks."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


# --- Main Class ---
class SleepTimeCompute:
    """
    Manages cognitive rhythms for background processing during simulated rest phases.
    Handles memory consolidation, pattern compression, and reflective learning simulations.
    """

    DEFAULT_CONFIG = {
        "memory_queue_size": 100,
        "pattern_queue_size": 50,
        "reflection_queue_size": 30,
        "simulation_queue_size": 20,
        "min_rest_cycle_duration_s": 5.0,
        "max_rest_cycle_duration_s": 30.0,
        "default_replay_duration_s": 5.0,
        "pattern_compression_target_ratio": 0.5,
        "dream_simulation_outcomes_range": (1, 3),  # Min/max outcomes per dream
        "thread_join_timeout_s": 5.0,  # Timeout for waiting on the processing thread
    }

    def __init__(
        self,
        vanta_core=None,  # UnifiedVantaCore instance for registration
        config: Optional[dict[str, Any]] = None,
        external_memory_interface: Optional[Any] = None,
    ):  # For MemoryBraid or similar
        """
        Initialize the SleepTimeCompute core.

        Args:
            config: Optional configuration dictionary to override defaults.
            external_memory_interface: Optional instance of a memory system (e.g., MemoryBraid)
                                       that SleepTimeCompute can interact with.
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        logger_stc.info(
            f"Initializing SleepTimeCompute with effective config: {self.config}"
        )

        self._current_state = CognitiveState.ACTIVE
        self._previous_state = CognitiveState.ACTIVE
        self._state_transition_time = (
            time.monotonic()
        )  # Use monotonic for internal timing
        self._lock = threading.RLock()  # For thread safety

        # Processing queues
        self._memory_consolidation_queue: deque[dict[str, Any]] = deque(
            maxlen=self.config["memory_queue_size"]
        )
        self._pattern_compression_queue: deque[dict[str, Any]] = deque(
            maxlen=self.config["pattern_queue_size"]
        )
        self._reflection_queue: deque[dict[str, Any]] = deque(
            maxlen=self.config["reflection_queue_size"]
        )
        self._simulation_queue: deque[dict[str, Any]] = deque(
            maxlen=self.config["simulation_queue_size"]
        )

        # Statistics
        self._stats = self._get_initial_stats()

        # Processing control
        self._processing_active = False
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_processing_event = threading.Event()  # For graceful thread shutdown

        # External Memory System
        self.memory_interface = external_memory_interface
        if self.memory_interface:
            logger_stc.info(
                f"External memory interface ({type(self.memory_interface).__name__}) provided."
            )
        else:
            logger_stc.info(
                "No external memory interface provided. Memory operations will be simulated."
            )

        # Handler Registries
        self._rest_phase_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._pattern_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._reflection_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._dream_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._consolidation_handlers: list[Callable[[dict[str, Any]], None]] = []

        logger_stc.info("SleepTimeCompute initialized.")
        self.vanta_core = vanta_core
        # Register with UnifiedVantaCore
        if self.vanta_core:
            try:
                self.vanta_core.register_component(
                    "sleep_time_compute", self, {"type": "sleep_time_compute"}
                )
                if hasattr(self.vanta_core, "async_bus"):
                    self.vanta_core.async_bus.register_component("sleep_time_compute")
                    self.vanta_core.async_bus.subscribe(
                        "sleep_time_compute",
                        MessageType.MEMORY_OPERATION,
                        self.handle_memory_operation,
                    )
            except Exception as e:
                logger_stc.warning(f"Failed to register SleepTimeCompute: {e}")

    def handle_memory_operation(self, message: AsyncMessage):
        """Consume memory operations and enqueue for consolidation"""
        try:
            data = message.content or {}
            self._memory_consolidation_queue.append(data)
        except Exception as e:
            logger_stc.error(f"SleepTimeCompute failed to handle memory op: {e}")

    def _get_initial_stats(self) -> dict[str, Any]:
        return {
            "memory_items_processed_total": 0,
            "patterns_compressed_total": 0,
            "reflections_generated_total": 0,
            "simulations_run_total": 0,
            "cumulative_rest_duration_s": 0.0,
            "rest_cycles_completed_total": 0,
            "last_rest_cycle_details": None,
            "errors_encountered": 0,
        }

    def _change_state(self, new_state: CognitiveState) -> None:
        with self._lock:
            if self._current_state == new_state:
                return
            self._previous_state = self._current_state
            self._current_state = new_state
            self._state_transition_time = time.monotonic()
            logger_stc.info(
                f"Cognitive state changed from {self._previous_state.value} to {self._current_state.value}"
            )
            # Here, you could emit an internal event or call a state change handler if needed.

    # --- Public API for Queueing Tasks ---
    def queue_task(
        self,
        queue_name: str,
        item_data: dict[str, Any],
        priority: Union[str, ProcessingPriority] = ProcessingPriority.MEDIUM,
    ) -> bool:
        """
        Generic method to queue a task for background processing.

        Args:
            queue_name: Name of the queue (e.g., "memory", "pattern", "reflection", "simulation").
            item_data: The data for the task item.
            priority: Priority of the task.

        Returns:
            True if queued successfully, False otherwise.
        """
        if not isinstance(queue_name, str) or not item_data:
            logger_stc.error(
                f"Invalid queue_name ('{queue_name}') or item_data for queue_task."
            )
            return False

        try:
            eff_priority = (
                priority
                if isinstance(priority, ProcessingPriority)
                else ProcessingPriority[priority.upper()]
            )
        except (KeyError, AttributeError):
            logger_stc.warning(
                f"Invalid priority '{priority}' for task. Defaulting to MEDIUM."
            )
            eff_priority = ProcessingPriority.MEDIUM

        queue_item = {
            "data": item_data,
            "priority": eff_priority,  # Store enum for easier sorting if needed later
            "queued_at_monotonic": time.monotonic(),
            "queued_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        queue: Optional[deque[dict[str, Any]]] = None
        if queue_name == "memory_consolidation":
            queue = self._memory_consolidation_queue
        elif queue_name == "pattern_compression":
            queue = self._pattern_compression_queue
        elif queue_name == "reflection":
            queue = self._reflection_queue
        elif queue_name == "simulation":
            queue = self._simulation_queue
        else:
            logger_stc.error(f"Unknown queue name: '{queue_name}'. Task not queued.")
            return False

        with self._lock:
            # Potentially insert based on priority if queues were sorted lists,
            # but for deques, simple append is typical. Maxlen handles overflow.
            queue.append(queue_item)

        logger_stc.debug(
            f"Queued item for '{queue_name}' processing. Priority: {eff_priority.name}. Data snippet: {str(item_data)[:80]}..."
        )
        return True

    # Convenience wrappers for specific queues
    def queue_memory_consolidation(
        self,
        memory_item: dict[str, Any],
        priority: Union[str, ProcessingPriority] = "MEDIUM",
    ) -> bool:
        return self.queue_task(
            "memory_consolidation", {"memory_item": memory_item}, priority
        )

    def queue_pattern_compression(
        self,
        pattern_data: dict[str, Any],
        priority: Union[str, ProcessingPriority] = "MEDIUM",
    ) -> bool:
        return self.queue_task(
            "pattern_compression", {"pattern_data": pattern_data}, priority
        )

    def queue_reflection_topic(
        self,
        topic: str,
        context: Optional[dict[str, Any]] = None,
        priority: Union[str, ProcessingPriority] = "MEDIUM",
    ) -> bool:
        return self.queue_task(
            "reflection", {"topic": topic, "context": context or {}}, priority
        )

    def queue_dream_scenario(
        self,
        scenario_data: dict[str, Any],
        priority: Union[str, ProcessingPriority] = "MEDIUM",
    ) -> bool:
        return self.queue_task("simulation", {"scenario_data": scenario_data}, priority)

    # --- Core Processing Methods (Simulated) ---
    # These methods would contain complex logic in a real system. Here they are placeholders.
    def _process_rest_phase_replay(self, duration_s: float) -> dict[str, Any]:
        logger_stc.info(f"Starting rest phase replay for ~{duration_s:.1f}s...")
        start_process_time = time.monotonic()
        processed_count = 0
        items_from_queue: list[dict[str, Any]] = []
        with self._lock:
            while self._memory_consolidation_queue and (
                time.monotonic() - start_process_time < duration_s
            ):
                items_from_queue.append(self._memory_consolidation_queue.popleft())

        insights = []
        for item in items_from_queue:
            memory_item = item.get("data", {}).get("memory_item", {})
            memory_id = memory_item.get("id", "unknown_id")
            logger_stc.debug(
                f"Replaying/strengthening memory_item: {memory_id} (Data: {str(memory_item)[:50]})"
            )
            if self.memory_interface and hasattr(
                self.memory_interface, "strengthen_memory"
            ):
                self.memory_interface.strengthen_memory(
                    memory_id=memory_id, factor=1.1
                )  # Example
            processed_count += 1
            if random.random() < 0.05:  # Small chance of insight
                insights.append(f"Insight related to memory {memory_id}")
            if time.monotonic() - start_process_time >= duration_s:
                break  # Check time frequently

        elapsed = time.monotonic() - start_process_time
        self._stats["memory_items_processed_total"] += processed_count
        logger_stc.info(
            f"Rest phase replay finished. Processed {processed_count} items in {elapsed:.2f}s."
        )
        return {
            "type": "replay",
            "processed_count": processed_count,
            "duration_s": elapsed,
            "insights": insights,
        }

    def _process_pattern_compression(self) -> dict[str, Any]:
        logger_stc.info("Starting pattern compression...")
        start_process_time = time.monotonic()
        processed_count = 0
        items_from_queue: list[dict[str, Any]] = []
        with self._lock:  # Get all items for this cycle
            items_from_queue.extend(self._pattern_compression_queue)
            self._pattern_compression_queue.clear()

        target_ratio = self.config["pattern_compression_target_ratio"]
        for item in items_from_queue:
            pattern_data = item.get("data", {}).get("pattern_data", {})
            logger_stc.debug(
                f"Compressing pattern (Target ratio: {target_ratio}): {str(pattern_data)[:80]}"
            )
            # Actual compression logic would be here
            processed_count += 1

        elapsed = time.monotonic() - start_process_time
        self._stats["patterns_compressed_total"] += processed_count
        logger_stc.info(
            f"Pattern compression finished. Compressed {processed_count} patterns in {elapsed:.2f}s."
        )
        return {
            "type": "compression",
            "processed_count": processed_count,
            "duration_s": elapsed,
            "avg_ratio_achieved": target_ratio * 0.9,
        }  # Simulated

    def _process_reflective_learning(self) -> dict[str, Any]:
        logger_stc.info("Starting reflective learning...")
        start_process_time = time.monotonic()
        processed_count = 0
        items_from_queue: list[dict[str, Any]] = []
        with self._lock:
            items_from_queue.extend(self._reflection_queue)
            self._reflection_queue.clear()

        insights = []
        for item in items_from_queue:
            topic = item.get("data", {}).get("topic", "general")
            context = item.get("data", {}).get("context", {})
            logger_stc.debug(
                f"Reflecting on topic '{topic}' with context: {str(context)[:50]}"
            )
            # Actual reflection logic here
            if random.random() < 0.3:
                insights.append(f"Reflection insight on '{topic}'")
            processed_count += 1

        elapsed = time.monotonic() - start_process_time
        self._stats["reflections_generated_total"] += processed_count
        logger_stc.info(
            f"Reflective learning finished. Generated {processed_count} reflections in {elapsed:.2f}s."
        )
        return {
            "type": "reflection",
            "processed_count": processed_count,
            "duration_s": elapsed,
            "insights_generated": insights,
        }

    def _process_dream_simulation(self) -> dict[str, Any]:
        logger_stc.info("Starting dream simulation...")
        start_process_time = time.monotonic()
        processed_count = 0
        items_from_queue: list[dict[str, Any]] = []
        with self._lock:
            items_from_queue.extend(self._simulation_queue)
            self._simulation_queue.clear()

        outcomes = []
        min_o, max_o = self.config["dream_simulation_outcomes_range"]
        for item in items_from_queue:
            scenario = item.get("data", {}).get("scenario_data", {})
            logger_stc.debug(f"Simulating dream scenario: {str(scenario)[:80]}")
            # Actual simulation logic here
            num_outcomes = random.randint(min_o, max_o)
            for _ in range(num_outcomes):
                outcomes.append(f"Simulated outcome for scenario: {str(scenario)[:30]}")
            processed_count += 1

        elapsed = time.monotonic() - start_process_time
        self._stats["simulations_run_total"] += processed_count
        logger_stc.info(
            f"Dream simulation finished. Ran {processed_count} simulations in {elapsed:.2f}s."
        )
        return {
            "type": "dream_simulation",
            "simulations_run": processed_count,
            "duration_s": elapsed,
            "outcomes_generated_count": len(outcomes),
        }

    def _process_memory_consolidation(self) -> dict[str, Any]:
        logger_stc.info(
            "Starting memory consolidation (final pass for this rest cycle)..."
        )
        # This is a simplified consolidation. A real one would interact more deeply with memory_interface.
        # For this pass, we'll assume items in _memory_consolidation_queue are already replayed/strengthened.
        # This stage could be for moving items to a more permanent/optimized store, indexing, etc.
        start_process_time = time.monotonic()
        processed_count = 0
        # Items might have been added to queue *during* this rest cycle's earlier phases.
        # This ensures they also get a chance to be "consolidated" (even if just noted).
        items_from_queue: list[dict[str, Any]] = []
        with self._lock:
            while self._memory_consolidation_queue:  # Process remaining
                items_from_queue.append(self._memory_consolidation_queue.popleft())

        for item in items_from_queue:
            memory_item = item.get("data", {}).get("memory_item", {})
            memory_id = memory_item.get("id", "unknown_id")
            logger_stc.debug(f"Consolidating memory_item: {memory_id}")
            if self.memory_interface and hasattr(
                self.memory_interface, "mark_as_consolidated"
            ):
                self.memory_interface.mark_as_consolidated(memory_id)
            processed_count += 1

        elapsed = time.monotonic() - start_process_time
        # This stat might double-count if _process_rest_phase_replay also updates it for the same items.
        # Consider if "memory_items_processed_total" should only be incremented in one place per item.
        # For now, it represents "processing touches".
        self._stats["memory_items_processed_total"] += processed_count
        logger_stc.info(
            f"Memory consolidation pass finished. Finalized {processed_count} items in {elapsed:.2f}s."
        )
        return {
            "type": "consolidation_pass",
            "finalized_count": processed_count,
            "duration_s": elapsed,
        }

    def should_consolidate(self) -> bool:
        """
        Determine if memory consolidation should be performed based on queue size,
        elapsed time since last consolidation, and system state.

        Returns:
            bool: True if consolidation should be performed, False otherwise.
        """
        with self._lock:
            # Check if we have enough items in memory queue to justify consolidation
            if len(self._memory_consolidation_queue) >= self.config.get(
                "consolidation_threshold", 5
            ):
                logger_stc.info(
                    f"Memory queue size {len(self._memory_consolidation_queue)} exceeds threshold - consolidation needed"
                )
                return True

            # Check if we're in a suitable state for consolidation
            if self._current_state != CognitiveState.ACTIVE:
                logger_stc.debug(
                    f"Current state {self._current_state.value} is not suitable for immediate consolidation"
                )
                return False

            # Check time since last consolidation (if tracked in stats)
            last_consolidation = self._stats.get("last_consolidation_time", 0)
            time_since_last = time.monotonic() - last_consolidation
            min_interval = self.config.get(
                "min_consolidation_interval_s", 600
            )  # Default: 10 minutes

            if (
                time_since_last > min_interval
                and len(self._memory_consolidation_queue) > 0
            ):
                logger_stc.info(
                    f"Time since last consolidation ({time_since_last:.1f}s) exceeds minimum interval - consolidation needed"
                )
                return True

        return False

    def consolidate_memories(self) -> dict[str, Any]:
        """
        Perform memory consolidation by processing the memory queue.
        This initiates a rest phase for memory processing.

        Returns:
            Dict[str, Any]: Results of the consolidation process.
        """
        logger_stc.info("Starting memory consolidation process")

        # Record consolidation time
        with self._lock:
            self._stats["last_consolidation_time"] = time.monotonic()

        # Use the existing rest phase processing mechanism
        if (
            self._current_state == CognitiveState.REST
            or self._current_state == CognitiveState.DEEP_REST
        ):
            logger_stc.info("Already in REST state, continuing with current rest phase")
            return {"status": "in_progress", "message": "Already in rest phase"}

        # Determine duration based on queue size
        queue_size = len(self._memory_consolidation_queue)
        duration_s = min(
            max(
                self.config.get("min_rest_cycle_duration_s", 5.0),
                queue_size * 0.5,  # 0.5 seconds per item
            ),
            self.config.get("max_rest_cycle_duration_s", 30.0),
        )  # Start a rest phase for consolidation
        try:
            success = self.enter_rest_mode(duration_s=duration_s, depth="normal")

            if not success:
                logger_stc.warning("Failed to enter rest mode for consolidation")
                return {
                    "status": "error",
                    "error": "Failed to enter rest mode",
                    "timestamp": time.time(),
                }

            # Use the stats to determine items processed
            processed_count = len(self._memory_consolidation_queue)
            logger_stc.info(
                f"Memory consolidation initiated: processing {processed_count} items"
            )
            return {
                "status": "success",
                "duration": duration_s,
                "items_processed": processed_count,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger_stc.error(f"Error during memory consolidation: {e}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}

    # --- Rest Cycle Management ---
    def enter_rest_mode(
        self, duration_s: Optional[float] = None, depth: str = "normal"
    ) -> bool:
        """
        Initiates a rest cycle for background processing.

        Args:
            duration_s: Optional duration for the rest cycle in seconds.
                        If None, a random duration between min/max config values is used.
            depth: "normal" or "deep". Deep rest might allow more intensive processing.

        Returns:
            True if rest mode was successfully initiated, False otherwise.
        """
        with self._lock:
            if self._processing_active:
                logger_stc.info(
                    "Rest mode already active. Request to enter ignored or could extend duration."
                )
                return False  # Or logic to extend current cycle

            self._processing_active = True
            self._stop_processing_event.clear()  # Ensure stop event is clear

        eff_duration_s = duration_s
        if eff_duration_s is None:
            eff_duration_s = random.uniform(
                self.config["min_rest_cycle_duration_s"],
                self.config["max_rest_cycle_duration_s"],
            )
        eff_duration_s = max(0.1, eff_duration_s)  # Ensure positive duration

        rest_state = (
            CognitiveState.DEEP_REST if depth.lower() == "deep" else CognitiveState.REST
        )
        self._change_state(rest_state)
        logger_stc.info(
            f"Entering {rest_state.value} mode for approximately {eff_duration_s:.2f} seconds."
        )

        self._processing_thread = threading.Thread(
            target=self._rest_processing_loop,
            args=(eff_duration_s, rest_state),
            name="SleepTimeCompute-Worker",
            daemon=True,
        )
        self._processing_thread.start()
        return True

    def _rest_processing_loop(
        self, target_duration_s: float, initial_rest_state: CognitiveState
    ) -> None:
        """Background thread for processing tasks during a rest cycle."""
        cycle_start_time = time.monotonic()
        logger_stc.info(
            f"Rest processing loop started. Target duration: {target_duration_s:.2f}s. State: {initial_rest_state.value}"
        )

        # Define processing stages. In deep rest, all might run. In normal, it could be prioritized.
        # This is a simple sequential execution. A more complex scheduler could be used.
        processing_stages_deep = [
            self._process_rest_phase_replay,
            self._process_pattern_compression,
            self._process_reflective_learning,
            self._process_dream_simulation,
            self._process_memory_consolidation,  # Final pass
        ]
        # For normal rest, maybe fewer or shorter stages based on queue priorities
        processing_stages_normal = [
            self._process_rest_phase_replay,  # Prioritize memory replay
            self._process_pattern_compression,  # Then patterns
            self._process_memory_consolidation,
        ]

        stages_to_run = (
            processing_stages_deep
            if initial_rest_state == CognitiveState.DEEP_REST
            else processing_stages_normal
        )

        stage_reports = []

        for stage_func in stages_to_run:
            if self._stop_processing_event.is_set() or (
                time.monotonic() - cycle_start_time >= target_duration_s
            ):
                logger_stc.info("Rest cycle interrupted (stop event or timeout).")
                break

            # Allocate a portion of the remaining time to each stage (simple division)
            # More sophisticated would be dynamic based on queue sizes or priorities
            remaining_time = target_duration_s - (time.monotonic() - cycle_start_time)
            time_for_stage = (
                remaining_time / (len(stages_to_run) - stages_to_run.index(stage_func))
                if len(stages_to_run) > stages_to_run.index(stage_func)
                else remaining_time
            )
            time_for_stage = max(0.1, time_for_stage)  # Min time for a stage

            try:
                if (
                    stage_func == self._process_rest_phase_replay
                ):  # This stage takes a duration argument
                    report = stage_func(
                        min(time_for_stage, self.config["default_replay_duration_s"])
                    )
                else:
                    report = stage_func()
                stage_reports.append(report)
            except Exception as e:
                logger_stc.error(
                    f"Error during stage {stage_func.__name__}: {e}", exc_info=True
                )
                self._stats["errors_encountered"] += 1
                stage_reports.append({"type": stage_func.__name__, "error": str(e)})

            time.sleep(0.05)  # Small yield to allow other threads

        # --- Cycle finished ---
        cycle_actual_duration_s = time.monotonic() - cycle_start_time
        with self._lock:
            self._stats["cumulative_rest_duration_s"] += cycle_actual_duration_s
            self._stats["rest_cycles_completed_total"] += 1
            self._stats["last_rest_cycle_details"] = {
                "start_utc": datetime.fromtimestamp(
                    cycle_start_time - time.monotonic() + time.time(), tz=timezone.utc
                ).isoformat(),  # Approx UTC start
                "duration_s": round(cycle_actual_duration_s, 2),
                "target_duration_s": round(target_duration_s, 2),
                "initial_state": initial_rest_state.value,
                "stages_executed_reports": stage_reports,
                "interrupted": self._stop_processing_event.is_set(),
            }
            self._processing_active = False  # Mark as no longer processing

        self._change_state(CognitiveState.ACTIVE)  # Transition back to active
        logger_stc.info(
            f"Rest processing loop finished. Actual duration: {cycle_actual_duration_s:.2f}s."
        )

    def stop_rest_mode(self, graceful: bool = True) -> None:
        """Stops an ongoing rest cycle."""
        if not self._processing_active or not self._processing_thread:
            logger_stc.info("No active rest mode to stop.")
            return

        logger_stc.info(
            f"Attempting to {'gracefully ' if graceful else ''}stop rest mode..."
        )
        self._stop_processing_event.set()  # Signal the loop to stop

        if graceful and self._processing_thread.is_alive():
            timeout = self.config["thread_join_timeout_s"]
            self._processing_thread.join(timeout=timeout)
            if self._processing_thread.is_alive():
                logger_stc.warning(
                    f"Processing thread did not terminate gracefully within {timeout}s. It might be stuck."
                )
            else:
                logger_stc.info("Processing thread stopped gracefully.")
        # If not graceful, the thread will eventually exit when it checks _stop_processing_event.
        # Setting self._processing_active = False is done at the end of _rest_processing_loop.
        # The state change to ACTIVE also happens there.

    # --- Handler Management ---
    def register_handler(
        self, phase_type: str, handler_func: Callable[[dict[str, Any]], None]
    ) -> bool:
        """
        Registers a callback handler for a specific processing phase.
        Handlers are called *after* a phase is completed.
        The handler will receive a dictionary with details about the completed phase.

        Args:
            phase_type: "replay", "compression", "reflection", "dream", "consolidation".
            handler_func: The function to call. It will receive a dict with phase results.
        """
        if not callable(handler_func):
            logger_stc.error(f"Invalid handler: not callable for phase '{phase_type}'.")
            return False

        registry_map = {
            "replay": self._rest_phase_handlers,
            "compression": self._pattern_handlers,
            "reflection": self._reflection_handlers,
            "dream": self._dream_handlers,
            "consolidation": self._consolidation_handlers,
        }
        if phase_type in registry_map:
            registry_map[phase_type].append(handler_func)
            logger_stc.info(
                f"Registered handler for '{phase_type}' phase: {handler_func.__name__}"
            )
            return True
        else:
            logger_stc.warning(
                f"Unknown phase_type '{phase_type}' for handler registration."
            )
            return False

    # --- Getters ---
    def get_current_state(self) -> str:
        return self._current_state.value

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            # Return a deep copy to prevent external modification
            return json.loads(json.dumps(self._stats))

    def get_queue_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "memory_consolidation_queue": len(self._memory_consolidation_queue),
                "pattern_compression_queue": len(self._pattern_compression_queue),
                "reflection_queue": len(self._reflection_queue),
                "simulation_queue": len(self._simulation_queue),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = self._get_initial_stats()
        logger_stc.info("SleepTimeCompute statistics have been reset.")

    def shutdown(self, graceful_stop_timeout_s: Optional[float] = None) -> None:
        """Initiates shutdown of the SleepTimeCompute, stopping any active processing."""
        logger_stc.info("SleepTimeCompute shutdown initiated.")
        if self._processing_active:
            timeout = (
                graceful_stop_timeout_s
                if graceful_stop_timeout_s is not None
                else self.config["thread_join_timeout_s"]
            )
            self.stop_rest_mode(
                graceful=True
            )  # Uses its own timeout logic from stop_rest_mode
            # Wait for thread to actually finish after signaling stop
            if self._processing_thread and self._processing_thread.is_alive():
                logger_stc.info(
                    f"Waiting up to {timeout}s for processing thread to complete shutdown..."
                )
                self._processing_thread.join(timeout=timeout)
                if self._processing_thread.is_alive():
                    logger_stc.warning(
                        "Processing thread still alive after shutdown join attempt."
                    )
        logger_stc.info("SleepTimeCompute shutdown complete.")


# --- Example Usage (if run as a script) ---
if __name__ == "__main__":
    # Configure root logger for more verbose output during testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s %(funcName)s:%(lineno)d - %(message)s",
    )
    logger_stc.setLevel(logging.DEBUG)  # Set this module's logger to DEBUG as well

    print("=" * 20 + " Testing SleepTimeCompute Standalone " + "=" * 20)

    # Example external memory (can be your MemoryBraid)
    class MyMemorySystem:
        def strengthen_memory(self, memory_id: str, factor: float):
            logger_stc.debug(
                f"[MyMemorySystem] Strengthening memory '{memory_id}' by factor {factor:.1f}"
            )

        def mark_as_consolidated(self, memory_id: str):
            logger_stc.debug(
                f"[MyMemorySystem] Marking memory '{memory_id}' as consolidated."
            )

    my_mem = MyMemorySystem()
    stc = SleepTimeCompute(
        config={"min_rest_cycle_duration_s": 1.0, "max_rest_cycle_duration_s": 2.0},
        external_memory_interface=my_mem,
    )

    # Example handler
    def my_replay_handler(report: dict[str, Any]):
        logger_stc.info(
            f"HANDLER - Replay Report: Processed {report.get('processed_count')}, Duration {report.get('duration_s'):.2f}s"
        )

    stc.register_handler("replay", my_replay_handler)

    # Queue some tasks
    stc.queue_memory_consolidation(
        {"memory_item": {"id": "mem1", "content": "First memory"}}
    )
    stc.queue_pattern_compression(
        {"pattern_data": {"type": "spatial", "grid": [[1, 0], [0, 1]]}}
    )
    stc.queue_reflection_topic("recent_performance", {"accuracy": 0.8})
    stc.queue_dream_scenario(
        {"scenario_data": {"type": "what_if", "condition": "power_loss"}}
    )
    stc.queue_memory_consolidation(
        {"memory_item": {"id": "mem2", "content": "Second memory"}}
    )

    print("\n--- Initial Queue Stats ---")
    print(stc.get_queue_stats())

    print("\n--- Entering Rest Mode (Short Cycle) ---")
    stc.enter_rest_mode(duration_s=1.5, depth="normal")  # Short cycle for testing

    # Wait for the rest cycle to likely complete.
    # In a real app, you wouldn't block like this but let the thread run.
    time.sleep(2.5)  # Give it time to finish + a bit more

    if stc._processing_active and stc._processing_thread:  # Check if it's still running
        logger_stc.warning(
            "Rest cycle might still be active. Forcing stop for test completion."
        )
        stc.stop_rest_mode(
            graceful=False
        )  # Force stop if needed for testing script exit
        time.sleep(0.5)

    print("\n--- Stats After Rest Cycle ---")
    print(json.dumps(stc.get_stats(), indent=2))

    print("\n--- Queue Stats After Rest Cycle ---")
    print(stc.get_queue_stats())  # Should be empty or reduced

    print("\n--- Entering DEEP Rest Mode (Longer Cycle, queue more items) ---")
    stc.queue_memory_consolidation(
        {"memory_item": {"id": "mem3_deep", "content": "Deep memory 1"}}
    )
    stc.queue_memory_consolidation(
        {"memory_item": {"id": "mem4_deep", "content": "Deep memory 2"}}
    )
    stc.queue_pattern_compression(
        {"pattern_data": {"type": "temporal_deep", "sequence": [1, 1, 2, 3, 5]}}
    )
    stc.enter_rest_mode(duration_s=3, depth="deep")
    time.sleep(4)

    if stc._processing_active and stc._processing_thread:
        logger_stc.warning("Deep rest cycle might still be active. Forcing stop.")
        stc.stop_rest_mode(graceful=False)
        time.sleep(0.5)

    print("\n--- Stats After DEEP Rest Cycle ---")
    print(json.dumps(stc.get_stats(), indent=2))

    print("\n--- Final Queue Stats ---")
    print(stc.get_queue_stats())

    print("\n--- Shutting Down ---")
    stc.shutdown()

    print("\n=" * 20 + " Test Finished " + "=" * 20)

# --- CLI/API for Sleep/Dream Cycle and Dream Signature Management ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SleepTimeCompute CLI for sleep/dream cycle and dream signature management."
    )
    parser.add_argument(
        "--rest", action="store_true", help="Trigger a rest (sleep) cycle."
    )
    parser.add_argument(
        "--deep", action="store_true", help="Trigger a deep rest (dream) cycle."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        help="Duration of the rest cycle in seconds.",
    )
    parser.add_argument(
        "--export-dream",
        type=str,
        help="Export dream signature to the given file path.",
    )
    parser.add_argument(
        "--import-dream",
        type=str,
        help="Import dream signature from the given file path.",
    )
    args = parser.parse_args()

    # Setup memory system and SleepTimeCompute
    my_mem = MyMemorySystem()
    stc = SleepTimeCompute(
        config={"min_rest_cycle_duration_s": 1.0, "max_rest_cycle_duration_s": 2.0},
        external_memory_interface=my_mem,
    )

    if args.rest or args.deep:
        depth = "deep" if args.deep else "normal"
        stc.enter_rest_mode(duration_s=args.duration, depth=depth)
        time.sleep(args.duration + 1)
        print(json.dumps(stc.get_stats(), indent=2))

    if args.export_dream:
        # Simulate export (replace with real RAGCompression if available)
        dream_signature = json.dumps(stc.get_stats())
        with open(args.export_dream, "w") as f:
            f.write(dream_signature)
        print(f"Dream signature exported to {args.export_dream}")

    if args.import_dream:
        with open(args.import_dream, "r") as f:
            dream_signature = f.read()
        print(f"Dream signature imported from {args.import_dream}:")
        print(dream_signature)
