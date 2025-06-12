"""
Internal dialogue management for VantaCore ecosystem.

This module handles:
1. Self-reflexive communication
2. Component cross-talk orchestration
3. Dialogue context tracking
4. Multi-agent conversation facilitation
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, List, Optional

# Import VantaCore framework
try:
    from Vanta.core.UnifiedVantaCore import (
        generate_internal_dialogue_message,
        get_component,
        publish_event,
        register_component,
        safe_component_call,
        subscribe_to_event,
        trace_event,
    )
except ImportError:
    # Local fallback for development/testing
    from .UnifiedVantaCore import (
        generate_internal_dialogue_message,
        get_component,
        publish_event,
        register_component,
        safe_component_call,
        subscribe_to_event,
        trace_event,
    )

# HOLO-1.5 Mesh Infrastructure
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voxsigil.supervisor.dialogue")


@vanta_core_module(
    name="dialogue_manager",
    subsystem="cognitive_processing",
    mesh_role=CognitiveMeshRole.ORCHESTRATOR,
    description="Internal dialogue management for VantaCore ecosystem with cross-component communication",
    capabilities=["dialogue_orchestration", "component_communication", "conversation_tracking", "multi_agent_facilitation", "context_management"],
    cognitive_load=2.5,
    symbolic_depth=3,
    collaboration_patterns=["cross_component_dialogue", "orchestrated_communication", "context_aware_facilitation"]
)
class DialogueManager(BaseCore):
    """
    Manages internal dialogues between components within the VantaCore ecosystem.
    """

    def __init__(self, vanta_core, config: Dict[str, Any], model_manager=None):
        """
        Initialize the dialogue manager.

        Args:
            vanta_core: VantaCore instance
            config: Configuration dictionary
            model_manager: Optional model manager for dialogue analysis
        """
        # Initialize BaseCore with HOLO-1.5 mesh capabilities
        super().__init__(vanta_core, config)
        
        self.config = config
        self.model_manager = model_manager

        # Dialogue configuration
        self.dialogue_timeout_s = config.get("dialogue_timeout_s", 30)
        self.max_dialogue_turns = config.get("max_dialogue_turns", 5)
        self.max_dialogue_history = config.get("max_dialogue_history", 20)
        self.min_dialogue_interval_s = config.get("min_dialogue_interval_s", 60)

        # Dialogue state
        self.active_dialogues: Dict[str, Dict[str, Any]] = {}
        self.dialogue_history: List[Dict[str, Any]] = []
        self.last_dialogue_timestamp = 0        # Component registry for dialogue participants
        self.participating_components = set(
            [
                "vox_agent",
                "omega3_agent",
                "meta_core",
                "system_monitor",
                "memory_cluster",
                "art_controller",
                "entropy_guardian",
            ]
        )

        self.connected_to = []  # Register this component with VantaCore
        register_component("dialogue_manager", self)

        # Subscribe to relevant eventssubscribe_to_event("component_registered", self._on_component_registered)
        subscribe_to_event("dialogue_request", self._on_dialogue_request)

    async def initialize(self) -> bool:
        """Initialize the dialogue manager with HOLO-1.5 capabilities."""
        try:
            # Set up dialogue logger
            self._setup_dialogue_logger()
            self.dialogue_logger.info("DialogueManager HOLO-1.5 initialized")
            
            # Mark as initialized
            self.is_initialized = True
            logger.info("🗣️ DialogueManager initialized with HOLO-1.5 mesh capabilities")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DialogueManager: {e}")
            return False

    def _setup_dialogue_logger(self):
        """Set up a dedicated logger for dialogues with daily rotation."""
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
        )
        os.makedirs(logs_dir, exist_ok=True)

        # Define log file path
        log_file = os.path.join(logs_dir, "vox_dialogues.log")

        # Create a dedicated logger
        self.dialogue_logger = logging.getLogger("metaconsciousness.vox.dialogue.file")
        self.dialogue_logger.setLevel(logging.DEBUG)

        # Prevent duplicating logs if handler already exists
        if not self.dialogue_logger.handlers:
            # Create file handler with daily rotation and keeping 7 days of logs
            file_handler = TimedRotatingFileHandler(
                log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
            )

            # Define formatter with more detail for file logs
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)

            # Add handler to logger
            self.dialogue_logger.addHandler(file_handler)

            # Set propagate to False to prevent double logging
            self.dialogue_logger.propagate = False

            self.dialogue_logger.info("===== Dialogue Logger Initialized =====")

    def _on_component_registered(self, event_data: Dict[str, Any]) -> None:
        """
        Handle component registration events from VantaCore.

        Args:
            event_data: Event data containing component information
        """
        component_name = event_data.get("component_name")
        if component_name:
            self.participating_components.add(component_name)
            self.dialogue_logger.info(
                f"Component '{component_name}' registered for dialogue participation"
            )

    def _on_dialogue_request(self, event_data: Dict[str, Any]) -> None:
        """
        Handle dialogue request events from other components.

        Args:
            event_data: Event data containing dialogue request information
        """
        initial_prompt = event_data.get("prompt", "")
        context_data = event_data.get("context", {})
        dialogue_type = event_data.get("type", "generic")
        initiator = event_data.get("initiator", "unknown")
        participants = event_data.get("participants")

        if initial_prompt:
            dialogue_id = self.trigger_dialogue(
                initial_prompt=initial_prompt,
                context_data=context_data,
                dialogue_type=dialogue_type,
                initiator=initiator,
                participants=participants,
            )

            if dialogue_id:
                # Publish event about successful dialogue creation
                publish_event(
                    "dialogue_started",
                    {
                        "dialogue_id": dialogue_id,
                        "initiator": initiator,
                        "type": dialogue_type,
                    },
                )
                self.dialogue_logger.info(
                    f"Started dialogue '{dialogue_id}' from request by '{initiator}'"
                )
            else:
                self.dialogue_logger.warning(
                    f"Failed to start dialogue from request by '{initiator}'"
                )

    def trigger_dialogue(
        self,
        initial_prompt: str,
        context_data: Optional[Dict[str, Any]] = None,
        dialogue_type: str = "generic",
        initiator: str = "vox_agent",
        participants: Optional[List[str]] = None,
    ) -> str:
        """
        Trigger a new internal dialogue.

        Args:
            initial_prompt: The prompt to start the dialogue
            context_data: Additional context data
            dialogue_type: Type of dialogue (generic, entropy, pattern, memory, etc.)
            initiator: Component initiating the dialogue
            participants: Optional specific participants

        Returns:
            Dialogue ID if successful, empty string otherwise
        """
        now = time.time()

        # Check if we're triggering dialogues too frequently
        if now - self.last_dialogue_timestamp < self.min_dialogue_interval_s:
            self.dialogue_logger.debug(
                f"Dialogue requested too soon after previous dialogue (interval: {now - self.last_dialogue_timestamp:.2f}s)"
            )
            logger.debug("Dialogue requested too soon after previous dialogue")
            return ""

        logger.info(f"Triggering internal dialogue: {dialogue_type}")
        self.dialogue_logger.info(
            f'NEW DIALOGUE: Type={dialogue_type}, Initiator={initiator}, Prompt="{initial_prompt[:100]}..."'
        )

        # Generate dialogue ID
        dialogue_id = f"dialogue_{dialogue_type}_{int(now)}"

        # Determine participants if not provided
        if not participants:
            participants = self._select_participants_for_dialogue(dialogue_type)

        # Create dialogue context
        dialogue = {
            "id": dialogue_id,
            "type": dialogue_type,
            "initiator": initiator,
            "participants": participants,
            "start_time": now,
            "last_update": now,
            "is_active": True,
            "turns": [],
            "prompt": initial_prompt,
            "context_data": context_data or {},
            "outcome": None,
        }

        # Store in active dialogues
        self.active_dialogues[dialogue_id] = dialogue

        # Start the dialogue with initial prompt
        self._add_dialogue_turn(dialogue_id, initiator, initial_prompt, is_system=True)

        # Process first turn to get responses from participants
        success = self._process_dialogue_turn(dialogue_id)

        if success:
            self.last_dialogue_timestamp = now

            trace_event(
                "dialogue_started",
                {
                    "dialogue_id": dialogue_id,
                    "dialogue_type": dialogue_type,
                    "participants": participants,
                    "prompt": initial_prompt[:100] + "..."
                    if len(initial_prompt) > 100
                    else initial_prompt,
                },
            )

            self.dialogue_logger.info(
                f"Dialogue {dialogue_id} started successfully with {len(participants)} participants"
            )
            return dialogue_id
        else:
            # Clean up failed dialogue
            del self.active_dialogues[dialogue_id]
            self.dialogue_logger.warning(
                f"Failed to start dialogue of type {dialogue_type}"
            )
            return ""

    def _select_participants_for_dialogue(self, dialogue_type: str) -> List[str]:
        """Select appropriate participants based on dialogue type."""
        # Base participants for all dialogues
        participants = ["vox_agent"]

        if dialogue_type == "entropy":
            participants.extend(["entropy_guardian", "meta_core"])
        elif dialogue_type == "pattern":
            participants.extend(["art_controller", "memory_cluster"])
        elif dialogue_type == "memory":
            participants.extend(["memory_cluster", "meta_learning_engine"])
        elif dialogue_type == "system":
            participants.extend(["system_monitor", "meta_core"])
        elif dialogue_type == "emotional":
            participants.extend(["omega3_agent", "art_controller"])
        else:  # Generic dialogue - select some random participants
            available = list(self.participating_components - set(participants))
            if available:
                # Add 1-3 random participants
                count = min(3, len(available))
                participants.extend(random.sample(available, count))

        return list(set(participants))  # Ensure uniqueness

    def _add_dialogue_turn(
        self, dialogue_id: str, sender: str, content: str, is_system: bool = False
    ) -> bool:
        """Add a turn to an active dialogue."""
        if dialogue_id not in self.active_dialogues:
            logger.warning(f"Cannot add turn to unknown dialogue: {dialogue_id}")
            return False

        dialogue = self.active_dialogues[dialogue_id]

        # Check if dialogue has timed out
        if time.time() - dialogue["last_update"] > self.dialogue_timeout_s:
            logger.warning(f"Dialogue {dialogue_id} has timed out")
            self._conclude_dialogue(dialogue_id, "timeout")
            return False

        # Check if max turns reached
        if len(dialogue["turns"]) >= self.max_dialogue_turns:
            logger.info(f"Dialogue {dialogue_id} reached max turns")
            self._conclude_dialogue(dialogue_id, "max_turns_reached")
            return False

        # Add the turn
        turn = {
            "timestamp": time.time(),
            "sender": sender,
            "content": content,
            "is_system": is_system,
        }

        dialogue["turns"].append(turn)
        dialogue["last_update"] = time.time()

        return True

    def _process_dialogue_turn(self, dialogue_id: str) -> bool:
        """Process the latest turn in a dialogue, getting responses from participants."""
        if dialogue_id not in self.active_dialogues:
            return False

        dialogue = self.active_dialogues[dialogue_id]

        if not dialogue["turns"]:
            logger.warning(f"Cannot process empty dialogue: {dialogue_id}")
            return False

        # Get the most recent turn
        latest_turn = dialogue["turns"][-1]
        sender = latest_turn["sender"]
        content = latest_turn["content"]

        # Send to each participant except the sender
        success = False
        for participant in dialogue["participants"]:
            if participant == sender:
                continue

            response = self._request_participant_response(
                participant,
                dialogue_id,
                sender,
                content,
                dialogue["type"],
                dialogue["context_data"],
            )

            if response:
                success = (
                    self._add_dialogue_turn(dialogue_id, participant, response)
                    or success
                )

        return success

    def _request_participant_response(
        self,
        participant: str,
        dialogue_id: str,
        sender: str,
        content: str,
        dialogue_type: str,
        context_data: Dict[str, Any],
    ) -> Optional[str]:
        """Request a response from a dialogue participant."""
        logger.debug(
            f"Requesting response from {participant} for dialogue {dialogue_id}"
        )  # Create the message for the component
        message = generate_internal_dialogue_message(
            content,
            metadata={
                "sender": "dialogue_manager",
                "target": participant,
                "action": "request_dialogue_response",
                "dialogue_id": dialogue_id,
                "dialogue_type": dialogue_type,
                "original_sender": sender,
                "context_data": context_data,
            },
        )

        # Send to the component
        try:
            response = safe_component_call(
                participant, "handle_internal_message", message
            )

            if isinstance(response, dict) and "response" in response:
                return response["response"]

            logger.warning(f"Invalid response format from {participant}")
            return None

        except Exception as e:
            logger.error(f"Error getting response from {participant}: {e}")
            return None

    def _conclude_dialogue(self, dialogue_id: str, reason: str) -> None:
        """Conclude an active dialogue."""
        if dialogue_id not in self.active_dialogues:
            return

        dialogue = self.active_dialogues[dialogue_id]

        # Mark as inactive and add conclusion
        dialogue["is_active"] = False
        dialogue["end_time"] = time.time()
        dialogue["conclusion_reason"] = reason

        # Generate outcome if possible
        outcome = self._generate_dialogue_outcome(dialogue)
        dialogue["outcome"] = outcome

        # Move to history
        self.dialogue_history.append(dialogue)
        while len(self.dialogue_history) > self.max_dialogue_history:
            self.dialogue_history.pop(0)

        # Remove from active dialogues
        del self.active_dialogues[dialogue_id]

        trace_event(
            "dialogue_concluded",
            {
                "dialogue_id": dialogue_id,
                "reason": reason,
                "duration": dialogue["end_time"] - dialogue["start_time"],
                "turns": len(dialogue["turns"]),
                "outcome": outcome[:100] + "..."
                if outcome and len(outcome) > 100
                else outcome,
            },
        )

    def _generate_dialogue_outcome(self, dialogue: Dict[str, Any]) -> Optional[str]:
        """Generate an outcome summary from the dialogue."""
        # If no turns, nothing to summarize
        if not dialogue["turns"]:
            return None  # Try to get decision_router to summarize if available (placeholder)
        router = get_component("decision_router")
        if router and hasattr(router, "query"):
            try:
                # Compile turns into a single text
                turns_text = "\n".join(
                    [
                        f"{turn['sender']}: {turn['content']}"
                        for turn in dialogue["turns"]
                    ]
                )

                # Send to LLM for summarization
                summary = router.query(
                    f"Summarize the key insights and conclusions from this internal dialogue:\n{turns_text}",
                    max_tokens=200,
                    temperature=0.3,
                )

                if summary:
                    return summary

            except Exception as e:
                logger.error(f"Error generating dialogue outcome: {e}")

        # Simple fallback summary
        last_turn = dialogue["turns"][-1]
        return f"Last message from {last_turn['sender']}: {last_turn['content']}"

    def continue_dialogue(self, dialogue_id: str, content: str, sender: str) -> bool:
        """
        Continue an existing dialogue with a new message.

        Args:
            dialogue_id: ID of the dialogue
            content: Message content
            sender: Sender component name

        Returns:
            Success flag
        """
        if dialogue_id not in self.active_dialogues:
            logger.warning(f"Cannot continue unknown dialogue: {dialogue_id}")
            return False

        dialogue = self.active_dialogues[dialogue_id]

        # Check if sender is a participant
        if sender not in dialogue["participants"]:
            logger.warning(f"{sender} is not a participant in dialogue {dialogue_id}")
            return False

        # Add the turn
        success = self._add_dialogue_turn(dialogue_id, sender, content)
        if success:
            # Process the turn to get responses
            return self._process_dialogue_turn(dialogue_id)

        return False

    def get_dialogue_state(self, dialogue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a dialogue.

        Args:
            dialogue_id: ID of the dialogue

        Returns:
            Dialogue state dictionary or None if not found
        """
        # Check active dialogues first
        if dialogue_id in self.active_dialogues:
            return self.active_dialogues[dialogue_id]

        # Check dialogue history
        for dialogue in self.dialogue_history:
            if dialogue["id"] == dialogue_id:
                return dialogue

        return None

    def get_active_dialogues(self) -> List[Dict[str, Any]]:
        """
        Get all active dialogues.

        Returns:
            List of active dialogue dictionaries
        """
        return list(self.active_dialogues.values())

    def get_dialogue_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent dialogue history.

        Args:
            limit: Maximum number of dialogues to return

        Returns:
            List of past dialogue dictionaries
        """
        return self.dialogue_history[-limit:]

    def register_component(self, component_name: str) -> bool:
        """
        Register a component as a dialogue participant.

        Args:
            component_name: Name of the component

        Returns:
            Success flag
        """
        # Check if component exists
        component = get_component(component_name)
        if not component:
            logger.warning(f"Cannot register nonexistent component: {component_name}")
            return False

        # Check if component supports dialogue
        if not hasattr(component, "handle_internal_message"):
            logger.warning(
                f"Component {component_name} does not support internal messages"
            )
            return False

        self.participating_components.add(component_name)
        logger.info(f"Registered {component_name} for dialogue participation")
        return True

    def get_component_participation(self, component_name: str) -> Dict[str, int]:
        """
        Get participation statistics for a component.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with participation statistics
        """
        stats = {
            "active_dialogues": 0,
            "historical_dialogues": 0,
            "total_turns": 0,
            "initiated_dialogues": 0,
        }

        # Count in active dialogues
        for dialogue in self.active_dialogues.values():
            if component_name in dialogue["participants"]:
                stats["active_dialogues"] += 1

            if dialogue["initiator"] == component_name:
                stats["initiated_dialogues"] += 1

            # Count turns
            for turn in dialogue["turns"]:
                if turn["sender"] == component_name:
                    stats["total_turns"] += 1

        # Count in historical dialogues
        for dialogue in self.dialogue_history:
            if component_name in dialogue["participants"]:
                stats["historical_dialogues"] += 1

            if dialogue["initiator"] == component_name:
                stats["initiated_dialogues"] += 1

            # Count turns
            for turn in dialogue["turns"]:
                if turn["sender"] == component_name:
                    stats["total_turns"] += 1

        return stats

    def find_dialogues_by_type(
        self, dialogue_type: str, include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find dialogues of a specific type.

        Args:
            dialogue_type: Type of dialogue to find
            include_inactive: Whether to include inactive dialogues

        Returns:
            List of matching dialogue dictionaries
        """
        result = []

        # Add active dialogues of this type
        for dialogue in self.active_dialogues.values():
            if dialogue["type"] == dialogue_type:
                result.append(dialogue)

        # Add historical dialogues if requested
        if include_inactive:
            for dialogue in self.dialogue_history:
                if dialogue["type"] == dialogue_type:
                    result.append(dialogue)

        return result

    def find_dialogues_by_participant(
        self, participant: str, include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find dialogues involving a specific participant.

        Args:
            participant: Name of the participant
            include_inactive: Whether to include inactive dialogues

        Returns:
            List of matching dialogue dictionaries
        """
        result = []

        # Add active dialogues with this participant
        for dialogue in self.active_dialogues.values():
            if participant in dialogue["participants"]:
                result.append(dialogue)

        # Add historical dialogues if requested
        if include_inactive:
            for dialogue in self.dialogue_history:
                if participant in dialogue["participants"]:
                    result.append(dialogue)

        return result

    def get_dialogue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about dialogues.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "active_count": len(self.active_dialogues),
            "historical_count": len(self.dialogue_history),
            "dialogue_types": {},
            "participant_counts": {},
            "avg_turns_per_dialogue": 0,
            "avg_dialogue_duration": 0,
        }

        # Collect stats from all dialogues (active and historical)
        all_dialogues = list(self.active_dialogues.values()) + self.dialogue_history
        total_turns = 0
        total_duration = 0
        dialogue_count = len(all_dialogues)

        for dialogue in all_dialogues:
            # Count dialogue types
            dialogue_type = dialogue["type"]
            if dialogue_type in stats["dialogue_types"]:
                stats["dialogue_types"][dialogue_type] += 1
            else:
                stats["dialogue_types"][dialogue_type] = 1

            # Count participant occurrences
            for participant in dialogue["participants"]:
                if participant in stats["participant_counts"]:
                    stats["participant_counts"][participant] += 1
                else:
                    stats["participant_counts"][participant] = 1

            # Add turns
            total_turns += len(dialogue["turns"])

            # Add duration for completed dialogues
            if "end_time" in dialogue:
                duration = dialogue["end_time"] - dialogue["start_time"]
                total_duration += duration

        # Calculate averages
        if dialogue_count > 0:
            stats["avg_turns_per_dialogue"] = total_turns / dialogue_count

            # Only count dialogues with end_time for duration average
            completed_dialogues = sum(1 for d in all_dialogues if "end_time" in d)
            if completed_dialogues > 0:
                stats["avg_dialogue_duration"] = total_duration / completed_dialogues

        return stats

    def get_dialogue_logs_for_vox(self, days: int = 1, max_entries: int = 1000) -> str:
        """
        Retrieve dialogue logs for Vox to analyze.

        Args:
            days: Number of days of logs to retrieve (1-7)
            max_entries: Maximum number of log entries to return

        Returns:
            String containing formatted dialogue logs
        """
        try:
            # Limit days to valid range (we only keep 7 days of logs)
            days = max(1, min(7, days))
            self.dialogue_logger.info(
                f"Retrieving {days} days of dialogue logs for Vox analysis"
            )

            # Determine log files to read
            logs_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
            )
            log_file = os.path.join(logs_dir, "vox_dialogues.log")

            # List of files to read (current log + rotated logs if needed)
            files_to_read = [log_file]

            # Add rotated log files if requesting more than today
            if days > 1:
                today = datetime.now().date()
                for i in range(1, days):
                    # TimedRotatingFileHandler uses the format filename.YYYY-MM-DD
                    previous_date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                    rotated_log = f"{log_file}.{previous_date}"
                    if os.path.exists(rotated_log):
                        files_to_read.append(rotated_log)

            # Collect logs from files
            logs = []
            entries_read = 0

            for file_path in files_to_read:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        for line in file:
                            logs.append(line.strip())
                            entries_read += 1
                            if entries_read >= max_entries:
                                break

                if entries_read >= max_entries:
                    break

            # Format logs for Vox analysis
            self.dialogue_logger.info(
                f"Retrieved {len(logs)} log entries for Vox analysis"
            )
            return "\n".join(logs)

        except Exception as e:
            error_msg = f"Error retrieving dialogue logs: {e}"
            self.dialogue_logger.error(error_msg)
            logger.error(error_msg)
            return f"Error retrieving logs: {e}"

    def analyze_logs_with_vox(self, days: int = 1) -> Dict[str, Any]:
        """
        Analyze dialogue logs using Vox's model capabilities.

        Args:
            days: Number of days of logs to analyze (1-7)

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Get logs for analysis
            logs = self.get_dialogue_logs_for_vox(days)

            if logs.startswith("Error"):
                return {"error": logs}

            # Check if we have access to models
            if not self.model_manager:
                return {"error": "Model manager not available for analysis"}

            self.dialogue_logger.info(
                f"Analyzing {days} days of dialogue logs with Vox models"
            )

            # Create analysis prompt
            analysis_prompt = f"""Analyze the following system dialogue logs and provide insights:
1. Identify common dialogue patterns and topics
2. Detect any anomalies or unusual exchanges
3. Summarize the most significant dialogues
4. Provide recommendations for improving dialogue efficiency

LOGS:
{logs[:50000]}  # Limit to 50k characters to avoid context length issues

Provide a structured analysis with sections for each of the above points.
"""

            # Use model manager to generate analysis
            analysis = self.model_manager.generate_text(
                prompt=analysis_prompt,
                model_name=self.model_manager.default_reasoning_model,
                temperature=0.3,
                max_tokens=2000,
            )

            if not analysis:
                return {"error": "Failed to generate analysis"}

            # Log analysis completion
            self.dialogue_logger.info(
                f"Completed analysis of {days} days of dialogue logs"
            )

            # Create structured result
            result = {
                "timestamp": time.time(),
                "days_analyzed": days,
                "analysis": analysis,
                "log_entries_count": logs.count("\n") + 1,
            }

            # Store analysis in dialogue history for future reference
            self._store_log_analysis(result)

            return result

        except Exception as e:
            error_msg = f"Error analyzing dialogue logs: {e}"
            self.dialogue_logger.error(error_msg)
            logger.error(error_msg)
            return {"error": error_msg}

    def _store_log_analysis(self, analysis_result: Dict[str, Any]) -> None:
        """Store log analysis results for future reference."""
        try:
            # Create directory for analytics if it doesn't exist
            logs_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
            )
            analytics_dir = os.path.join(logs_dir, "analytics")
            os.makedirs(analytics_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.fromtimestamp(analysis_result["timestamp"]).strftime(
                "%Y%m%d_%H%M%S"
            )
            filename = os.path.join(
                analytics_dir, f"dialogue_analysis_{timestamp}.json"
            )

            # Save analysis to file
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, indent=2)

            self.dialogue_logger.info(f"Stored dialogue analysis to {filename}")

        except Exception as e:
            self.dialogue_logger.error(f"Error storing dialogue analysis: {e}")

    def get_health(self) -> str:
        """
        Get the health status of the dialogue manager.

        Returns:
            Health status string
        """
        # Check for potential issues
        if len(self.active_dialogues) > 10:
            self.dialogue_logger.warning(
                f"High number of active dialogues: {len(self.active_dialogues)}"
            )
            return "warning"

        # Check for timed out dialogues that weren't properly closed
        now = time.time()
        stale_dialogues = 0
        for dialogue_id, dialogue in self.active_dialogues.items():
            if now - dialogue["last_update"] > self.dialogue_timeout_s * 2:
                stale_dialogues += 1

        if stale_dialogues > 0:
            self.dialogue_logger.warning(f"Found {stale_dialogues} stale dialogues")
            return "warning"

        return "ok"

    def cleanup_stale_dialogues(self) -> int:
        """
        Clean up stale dialogues that haven't been properly closed.

        Returns:
            Number of dialogues cleaned up
        """
        now = time.time()
        stale_dialogues = []

        for dialogue_id, dialogue in self.active_dialogues.items():
            if now - dialogue["last_update"] > self.dialogue_timeout_s * 2:
                stale_dialogues.append(dialogue_id)

        for dialogue_id in stale_dialogues:
            self.dialogue_logger.info(f"Cleaning up stale dialogue: {dialogue_id}")
            self._conclude_dialogue(dialogue_id, "cleanup_stale")

        return len(stale_dialogues)
