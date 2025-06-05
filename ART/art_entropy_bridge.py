"""
ART-Entropy Bridge Module

This module implements a bridge between an ARTController and an EntropyGuardian
(or similar system entropy monitoring component), enabling adaptive resonance
adjustments based on system entropy and cognitive state.

Core functions:
1. Connects ARTController to an entropy monitoring component.
2. Adjusts ART vigilance parameters based on system entropy levels.
3. Provides resonance pattern insight to entropy monitoring.
4. Enables coordinated cognitive stability across pattern recognition and external state.
"""

import logging
import time
import threading
import numpy as np  # For variance calculation
from typing import Any, Optional

# Assuming art_logger is in the same directory or properly pathed
try:
    from .art_logger import get_art_logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _default_logger = logging.getLogger("voxsigil.art.entropy_bridge_fallback")

    from typing import Optional

    def get_art_logger(
        name: Optional[str] = None,
        level: Optional[int] = logging.INFO,
        log_file: Optional[str] = None,
        base_logger_name: Optional[str] = None,
    ):
        return _default_logger


# Note: The above import assumes the art_logger module is in the same directory.
# The ArtEntropyBridge now expects concrete instances to be passed during initialization.


class ArtEntropyBridge:
    """
    Bridge component that connects an ARTController with an EntropyGuardian system (or similar).
    This component enables adaptive vigilance adjustment and pattern resonance feedback.
    """

    def __init__(
        self,
        art_controller: Any,
        entropy_guardian: Any,
        config: Optional[dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ArtEntropyBridge.

        Args:
            art_controller: An instance of ARTController (or a compatible class).
            entropy_guardian: An instance of an entropy monitoring component (e.g., EntropyGuardian).
            config: Optional configuration parameters.
            logger_instance: Optional logger instance.
        """
        self.logger = (
            logger_instance
            if logger_instance
            else get_art_logger("voxsigil.art.entropy_bridge")
        )
        self.config = config or {}
        self.art_controller = art_controller
        self.entropy_guardian = entropy_guardian

        # Critical check for actual components
        if not self.art_controller:
            self.logger.critical(
                "ArtEntropyBridge initialized without a valid ARTController. The bridge will be disabled."
            )
            self.enabled = False  # Disable if no ARTController
        if not self.entropy_guardian:
            self.logger.critical(
                "ArtEntropyBridge initialized without a valid EntropyGuardian. The bridge will be disabled."
            )
            self.enabled = False  # Disable if no EntropyGuardian

        self.enabled = (
            self.config.get("enabled", True) if self.enabled else False
        )  # Ensure self.enabled reflects component availability
        self.active = False
        self.lock = (
            threading.RLock()
        )  # Changed to RLock for re-entrant scenarios if any

        self.adaptation_interval = self.config.get("adaptation_interval_seconds", 60)
        self.min_vigilance = self.config.get("min_vigilance", 0.3)
        self.max_vigilance = self.config.get("max_vigilance", 0.9)
        self.entropy_sensitivity = self.config.get("entropy_sensitivity", 0.7)
        self.significant_change_threshold = self.config.get(
            "significant_vigilance_change_threshold", 0.03
        )

        self.stats = {
            "vigilance_adjustments": 0,
            "resonance_reports": 0,
            "last_entropy_level": 0.0,
            "last_adaptation_time": 0,
            "total_resonance_samples": 0,
            "start_time": time.time(),
            "errors_in_adaptation": 0,
        }

        self.logger.info(
            f"ArtEntropyBridge initialized. ART: {type(self.art_controller).__name__}, Entropy: {type(self.entropy_guardian).__name__}"
        )

        if self.enabled:
            self._start_adaptation_thread()
            # Auto-activate only if components are valid and enabled
            if (
                self.art_controller and self.entropy_guardian
            ):  # Redundant check if self.enabled is set correctly above, but safe
                self.activate()

    def _start_adaptation_thread(self):
        if (
            not hasattr(self, "adaptation_thread")
            or not self.adaptation_thread.is_alive()
        ):
            self.adaptation_thread = threading.Thread(
                target=self._adaptation_loop, daemon=True
            )
            self.adaptation_thread.start()
            self.logger.debug("ArtEntropyBridge adaptation thread started.")
        else:
            self.logger.debug("ArtEntropyBridge adaptation thread already running.")

    def _adaptation_loop(self):
        self.logger.info("ArtEntropyBridge adaptation loop started.")
        while self.enabled:  # Loop as long as the bridge is enabled
            try:
                if self.active:
                    self._perform_adaptation_cycle()
                time.sleep(self.adaptation_interval)
            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {e}", exc_info=True)
                self.stats["errors_in_adaptation"] += 1
                time.sleep(
                    max(10, self.adaptation_interval / 2)
                )  # Sleep longer on error but not indefinitely
        self.logger.info("ArtEntropyBridge adaptation loop stopped.")

    def activate(self) -> bool:
        with self.lock:
            if self.active:
                self.logger.warning("ArtEntropyBridge already active.")
                return True

            # Check if components are valid; they must be provided externally now
            if not self.art_controller or not self.entropy_guardian:
                self.logger.error(
                    "Cannot activate bridge: missing ARTController or EntropyGuardian. Ensure valid instances are provided during initialization."
                )
                self.enabled = False  # Ensure bridge is marked as not enabled
                return False

            # Connect EntropyGuardian to ARTController if such a method exists
            if hasattr(self.entropy_guardian, "connect_to_art_controller") and callable(
                getattr(self.entropy_guardian, "connect_to_art_controller")
            ):
                try:
                    self.entropy_guardian.connect_to_art_controller(self.art_controller)
                    self.logger.info(
                        "Entropy monitor successfully connected to ARTController."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to connect EntropyGuardian to ARTController: {e}",
                        exc_info=True,
                    )
            else:
                self.logger.info(
                    "EntropyGuardian does not have a 'connect_to_art_controller' method. Skipping direct connection."
                )

            try:
                self._perform_adaptation_cycle()  # Perform initial adaptation
                self.logger.info("Initial adaptation cycle completed upon activation.")
            except Exception as e:
                self.logger.error(
                    f"Error during initial adaptation cycle: {e}", exc_info=True
                )
                self.stats["errors_in_adaptation"] += 1

            self.active = True
            self.logger.info("ArtEntropyBridge activated.")
            return True

    def deactivate(self) -> bool:
        with self.lock:
            if not self.active:
                self.logger.warning("ArtEntropyBridge already inactive.")
                return True
            self.active = False
            # self.enabled = False # Consider if deactivating should also disable the thread loop
            self.logger.info("ArtEntropyBridge deactivated.")
            # Note: The adaptation thread will exit its loop if self.enabled is set to False.
            # If only self.active is False, the thread continues but _perform_adaptation_cycle is skipped.
            return True

    def shutdown(self):  # New method to gracefully stop the thread
        self.logger.info("ArtEntropyBridge shutting down...")
        self.enabled = False  # This will stop the adaptation_loop
        self.deactivate()
        if hasattr(self, "adaptation_thread") and self.adaptation_thread.is_alive():
            self.adaptation_thread.join(
                timeout=self.adaptation_interval + 5
            )  # Wait for thread to finish
            if self.adaptation_thread.is_alive():
                self.logger.warning("Adaptation thread did not terminate gracefully.")
        self.logger.info("ArtEntropyBridge shut down complete.")

    def _perform_adaptation_cycle(self):
        if not self.active:
            return

        # Ensure components are still valid
        if not self.art_controller or not self.entropy_guardian:
            self.logger.warning(
                "Adaptation cycle skipped: ARTController or EntropyGuardian is missing."
            )
            # Deactivate or disable if components become invalid? For now, just skip cycle.
            # self.active = False
            # self.enabled = False
            return

        try:
            entropy_state = self.entropy_guardian.get_status()
            total_entropy = entropy_state.get(
                "total_entropy", 0.5
            )  # Default to neutral if not found
            critical_dimension = entropy_state.get("critical_dimension", "focus")

            art_stats = self.art_controller.get_statistics()
            current_vigilance = art_stats.get(
                "vigilance",
                self.art_controller.vigilance
                if hasattr(self.art_controller, "vigilance")
                else 0.5,
            )
            avg_resonance = art_stats.get(
                "avg_resonance",
                self.art_controller.avg_resonance
                if hasattr(self.art_controller, "avg_resonance")
                else 0.5,
            )

            self.stats["last_entropy_level"] = total_entropy
            self.stats["last_adaptation_time"] = time.time()

            dimension_factor = self._get_dimension_factor(critical_dimension)
            target_vigilance_adjustment = 0.0

            if total_entropy > 0.7:
                target_vigilance_adjustment = (total_entropy - 0.7) * dimension_factor
            elif total_entropy < 0.3:
                target_vigilance_adjustment = -(0.3 - total_entropy) * dimension_factor
            else:
                entropy_delta = total_entropy - 0.5  # -0.2 to 0.2
                target_vigilance_adjustment = entropy_delta * dimension_factor * 0.5

            target_vigilance = current_vigilance * (1 + target_vigilance_adjustment)
            target_vigilance = max(
                self.min_vigilance, min(self.max_vigilance, target_vigilance)
            )

            if (
                abs(target_vigilance - current_vigilance)
                > self.significant_change_threshold
            ):
                self.art_controller.set_vigilance(target_vigilance)
                self.stats["vigilance_adjustments"] += 1
                adjustment_info = {
                    "old_vigilance": round(current_vigilance, 4),
                    "new_vigilance": round(target_vigilance, 4),
                    "entropy_level": round(total_entropy, 4),
                    "critical_dimension": critical_dimension,
                    "timestamp": time.time(),
                }
                self.logger.info(
                    f"Adjusted ART vigilance: {adjustment_info['old_vigilance']} -> {adjustment_info['new_vigilance']} (entropy: {adjustment_info['entropy_level']}, dimension: {critical_dimension})"
                )
                # self.logger.debug("Vigilance adjustment details", extra=adjustment_info) # Example of structured logging

            # Report resonance patterns to EntropyGuardian
            # Ensure self.entropy_guardian has the method before calling
            if hasattr(self.entropy_guardian, "update_system_entropy") and callable(
                getattr(self.entropy_guardian, "update_system_entropy")
            ):
                self._report_resonance_patterns(avg_resonance)
            else:
                self.logger.debug(
                    "EntropyGuardian does not have 'update_system_entropy' method. Skipping report."
                )

        except Exception as e:
            self.logger.error(f"Error during adaptation cycle: {e}", exc_info=True)
            self.stats["errors_in_adaptation"] += 1

    def _get_dimension_factor(self, dimension: str) -> float:
        dimension_sensitivity = {
            "focus": 1.0,
            "semantic": 0.8,
            "goal": 0.5,
            "belief": 0.7,
            "temporal": 0.4,
            "default": 0.6,
        }
        factor = (
            dimension_sensitivity.get(
                dimension.lower(), dimension_sensitivity["default"]
            )
            * self.entropy_sensitivity
        )
        return factor

    def _report_resonance_patterns(self, avg_resonance: float):
        # Check if entropy_guardian is valid and has the required method
        if (
            not self.entropy_guardian
            or not hasattr(self.entropy_guardian, "update_system_entropy")
            or not callable(getattr(self.entropy_guardian, "update_system_entropy"))
        ):
            self.logger.debug(
                "Cannot report resonance patterns: EntropyGuardian is invalid or missing 'update_system_entropy' method."
            )
            return

        recent_resonance = []
        resonance_variance = 0.1

        if hasattr(self.art_controller, "recent_resonance") and isinstance(
            self.art_controller.recent_resonance, list
        ):
            recent_resonance = list(self.art_controller.recent_resonance)  # Make a copy
            if len(recent_resonance) > 1:
                resonance_variance = float(
                    np.var(recent_resonance)
                )  # np import needed at top

        try:
            self.entropy_guardian.update_system_entropy(
                efficiency_metrics={
                    "task_completion_rate": round(avg_resonance, 4),
                    "resource_utilization": round(
                        max(0.1, min(0.9, 0.5 + resonance_variance)), 4
                    ),
                    "execution_time_factor": round(
                        self._calculate_resonance_trend(recent_resonance), 4
                    ),
                }
            )
            self.stats["resonance_reports"] += 1
            self.stats["total_resonance_samples"] += len(recent_resonance)
        except Exception as e:
            self.logger.error(
                f"Error reporting resonance patterns to EntropyGuardian: {e}",
                exc_info=True,
            )

    def _calculate_resonance_trend(self, resonance_history: list[float]) -> float:
        if not resonance_history or len(resonance_history) < 2:
            return 1.0

        half_point = len(resonance_history) // 2
        if half_point < 1:
            return 1.0

        # Ensure no division by zero if slices are empty (though half_point < 1 check helps)
        recent_slice = resonance_history[half_point:]
        older_slice = resonance_history[:half_point]

        if not recent_slice or not older_slice:
            return 1.0

        recent_avg = sum(recent_slice) / len(recent_slice)
        older_avg = sum(older_slice) / len(older_slice)

        trend_diff = (
            recent_avg - older_avg
        )  # Positive if improving, negative if declining
        # Convert to a factor: <1 improving, >1 declining
        # Max change capped to e.g., 0.5 (factor 0.5 or 1.5)
        trend_factor = 1.0 - np.clip(trend_diff, -0.5, 0.5)
        return trend_factor

    def notify_goal_change(self, goal_data: dict[str, Any]) -> None:
        if (
            not self.active or not self.art_controller
        ):  # Check for art_controller directly
            self.logger.debug(
                "Goal change notification ignored: bridge inactive or ARTController invalid."
            )
            return

        goal_id = goal_data.get("goal_id")
        drift = float(goal_data.get("drift", 0.0))
        # priority = float(goal_data.get("priority", 0.5))

        if not goal_id:
            self.logger.warning("Goal change notification received without goal_id.")
            return

        try:
            art_stats = self.art_controller.get_statistics()
            current_vigilance = art_stats.get(
                "vigilance",
                self.art_controller.vigilance
                if hasattr(self.art_controller, "vigilance")
                else 0.5,
            )

            if drift > 0.4:  # Significant drift
                # Increase vigilance proportionally to drift, capped
                vigilance_increase_factor = min(
                    drift * 0.2, 0.1
                )  # Max 10% increase from this factor
                adjusted_vigilance = min(
                    self.max_vigilance,
                    current_vigilance * (1 + vigilance_increase_factor),
                )

                if (
                    abs(adjusted_vigilance - current_vigilance)
                    > self.significant_change_threshold / 2
                ):  # More sensitive for goal changes
                    self.art_controller.set_vigilance(adjusted_vigilance)
                    self.stats["vigilance_adjustments"] += 1
                    log_info = {
                        "goal_id": goal_id,
                        "drift": round(drift, 4),
                        "old_vigilance": round(current_vigilance, 4),
                        "new_vigilance": round(adjusted_vigilance, 4),
                    }
                    self.logger.info(
                        f"Adjusted ART vigilance to {log_info['new_vigilance']} due to goal drift ({log_info['drift']}) on goal '{goal_id}'. Old: {log_info['old_vigilance']}"
                    )
        except Exception as e:
            self.logger.error(
                f"Error adjusting ART parameters for goal change '{goal_id}': {e}",
                exc_info=True,
            )

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            uptime = time.time() - self.stats["start_time"]
            status = {
                "active": self.active,
                "enabled": self.enabled,
                "art_controller_type": type(self.art_controller).__name__
                if self.art_controller
                else "None",
                "entropy_guardian_type": type(self.entropy_guardian).__name__
                if self.entropy_guardian
                else "None",
                "last_entropy_level": round(self.stats["last_entropy_level"], 4),
                "vigilance_adjustments": self.stats["vigilance_adjustments"],
                "resonance_reports": self.stats["resonance_reports"],
                "errors_in_adaptation": self.stats["errors_in_adaptation"],
                "uptime_seconds": round(uptime, 2),
                "last_adaptation_time_ago_seconds": round(
                    time.time() - self.stats["last_adaptation_time"], 2
                )
                if self.stats["last_adaptation_time"]
                else None,
                "adaptation_interval_seconds": self.adaptation_interval,
            }
            if (
                self.art_controller
                and hasattr(self.art_controller, "get_statistics")
                and callable(getattr(self.art_controller, "get_statistics"))
            ):  # Check art_controller directly
                try:
                    art_stats = self.art_controller.get_statistics()
                    status["current_vigilance"] = round(art_stats.get("vigilance"), 4)
                    status["current_avg_resonance"] = round(
                        art_stats.get("avg_resonance"), 4
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not retrieve full ART stats for status: {e}"
                    )
            return status


# Singleton instance management (optional, can be handled by the application)
_bridge_instance: Optional[ArtEntropyBridge] = None
_bridge_lock = threading.Lock()


def get_art_entropy_bridge_instance(
    art_controller: Any,
    entropy_guardian: Any,
    config: Optional[dict[str, Any]] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> ArtEntropyBridge:
    global _bridge_instance
    if _bridge_instance is None:
        with _bridge_lock:
            if _bridge_instance is None:  # Double-check locking
                _bridge_instance = ArtEntropyBridge(
                    art_controller, entropy_guardian, config, logger_instance
                )
    else:  # If instance exists, update components if they are different (or log a warning)
        if art_controller and _bridge_instance.art_controller is not art_controller:
            _bridge_instance.logger.warning(
                "ArtEntropyBridge already exists with a different ARTController. Re-configuring."
            )
            _bridge_instance.art_controller = art_controller
        if (
            entropy_guardian
            and _bridge_instance.entropy_guardian is not entropy_guardian
        ):
            _bridge_instance.logger.warning(
                "ArtEntropyBridge already exists with a different EntropyGuardian. Re-configuring."
            )
            _bridge_instance.entropy_guardian = entropy_guardian
        if config:
            _bridge_instance.config.update(config)  # Allow config updates
        if logger_instance and _bridge_instance.logger is not logger_instance:
            _bridge_instance.logger = logger_instance

    return _bridge_instance


# Example Usage
if __name__ == "__main__":
    main_logger = get_art_logger("art_entropy_bridge_example")
    main_logger.setLevel(logging.DEBUG)

    main_logger.info("--- ArtEntropyBridge Example --- ")

    # Define Mock ARTController and EntropyGuardian locally for the example
    class MockARTController:
        def __init__(self, logger_instance=None):
            self.logger = logger_instance or get_art_logger("MockARTControllerExample")
            self.vigilance = 0.6
            self.avg_resonance = 0.75
            self.recent_resonance = [0.7, 0.75, 0.8]
            self.logger.info(
                f"MockARTController (example) initialized with vigilance: {self.vigilance}"
            )

        def get_statistics(self) -> dict[str, Any]:
            return {
                "vigilance": self.vigilance,
                "avg_resonance": self.avg_resonance,
                "num_categories": 5,
            }

        def set_vigilance(self, new_vigilance: float) -> None:
            self.logger.info(
                f"MockARTController (example): Vigilance set from {self.vigilance} to {new_vigilance}"
            )
            self.vigilance = new_vigilance

        def get_config(self) -> dict[str, Any]:
            return {"vigilance": self.vigilance, "input_dim": 20}

    class MockEntropyGuardian:
        def __init__(self, logger_instance=None):
            self.logger = logger_instance or get_art_logger(
                "MockEntropyGuardianExample"
            )
            self.total_entropy = 0.4
            self.critical_dimension = "semantic"
            self.logger.info("MockEntropyGuardian (example) initialized.")

        def get_status(self) -> dict[str, Any]:
            return {
                "total_entropy": self.total_entropy,
                "critical_dimension": self.critical_dimension,
            }

        def update_system_entropy(self, efficiency_metrics: dict[str, Any]) -> None:
            self.logger.info(
                f"MockEntropyGuardian (example): Received efficiency metrics: {efficiency_metrics}"
            )

        def connect_to_art_controller(self, art_controller: Any) -> None:
            self.logger.info(
                f"MockEntropyGuardian (example): Connected to ART controller: {type(art_controller).__name__}"
            )

    # Instantiate mock components
    mock_art = MockARTController(logger_instance=main_logger)
    mock_entropy = MockEntropyGuardian(logger_instance=main_logger)

    bridge_config = {
        "enabled": True,
        "adaptation_interval_seconds": 5,  # Short interval for example
        "min_vigilance": 0.2,
        "max_vigilance": 0.95,
        "entropy_sensitivity": 0.8,
    }

    # Create bridge instance (not using singleton getter for direct control in example)
    bridge = ArtEntropyBridge(
        mock_art, mock_entropy, config=bridge_config, logger_instance=main_logger
    )
    # bridge.activate() # activate is called in __init__ if components are valid

    if not bridge.active:
        main_logger.error("Bridge did not activate. Check component setup.")
    else:
        main_logger.info(f"Bridge activated. Initial status: {bridge.get_status()}")

        # Simulate entropy changes and goal notifications
        try:
            for i in range(3):
                time.sleep(bridge.adaptation_interval + 1)
                # Simulate entropy change
                mock_entropy.total_entropy = (
                    np.random.rand()
                )  # Random entropy between 0 and 1
                mock_entropy.critical_dimension = np.random.choice(
                    ["focus", "semantic", "goal"]
                )
                main_logger.info(
                    f"Simulated entropy change: {mock_entropy.total_entropy:.2f}, dim: {mock_entropy.critical_dimension}"
                )

                if i == 1:
                    goal_info = {
                        "goal_id": "test_goal_123",
                        "drift": 0.65,
                        "priority": 0.9,
                    }
                    main_logger.info(f"Simulating goal change: {goal_info}")
                    bridge.notify_goal_change(goal_info)

                main_logger.info(f"Bridge status: {bridge.get_status()}")

        except KeyboardInterrupt:
            main_logger.info("Example interrupted by user.")
        finally:
            main_logger.info("Shutting down bridge...")
            bridge.shutdown()
            main_logger.info("Bridge example finished.")
