#!/usr/bin/env python
"""
UnifiedAsyncBus - Asynchronous Communication Bus for Vanta Components

Provides a unified asynchronous communication infrastructure for different
components within the Vanta system, particularly focused on handling
communications between speech components (STT, TTS), memory systems, and
processing engines.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class MessagePriority(Enum):
    """Priority levels for messages in the async bus."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class MessageType(Enum):
    """Types of messages that can be sent through the async bus."""

    AUDIO_TRANSCRIPTION = "audio_transcription"
    TEXT_TO_SPEECH = "text_to_speech"
    MEMORY_OPERATION = "memory_operation"
    MEMORY_RESULT = "memory_result"
    PROCESSING_REQUEST = "processing_request"
    PROCESSING_RESPONSE = "processing_response"
    COMPONENT_STATUS = "component_status"
    SYSTEM_COMMAND = "system_command"
    USER_INTERACTION = "user_interaction"
    # CAT Engine message types
    CLASSIFICATION_REQUEST = "classification_request"
    PATTERN_ANALYSIS = "pattern_analysis"
    # Proactive Intelligence message types
    ACTION_EVALUATION = "action_evaluation"
    PRIORITY_UPDATE = "priority_update"
    # Hybrid Cognition Engine message types
    REASONING_REQUEST = "reasoning_request"
    BRANCH_UPDATE = "branch_update"
    # ToT Engine message types
    THOUGHT_REQUEST = "thought_request"
    BRANCH_EVALUATION = "branch_evaluation"


class AsyncMessage:
    """Message object for communication across the async bus."""

    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        content: Any,
        target_ids: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.message_type = message_type
        self.sender_id = sender_id
        self.content = content
        self.target_ids = target_ids or []
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.message_id = f"{sender_id}_{self.timestamp}_{id(self)}"


class UnifiedAsyncBus:
    """
    Unified asynchronous communication bus for Vanta components.

    This bus enables asynchronous communication between different components
    of the Vanta system, particularly focused on:
    - STT (Speech-to-Text) operations
    - TTS (Text-to-Speech) operations
    - Memory systems (Echo Memory, Memory Braid)
    - Processing engines

    It follows a publish-subscribe pattern with support for targeted messages
    and priority-based processing.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the async bus with optional logger."""
        self.logger = logger or logging.getLogger("UnifiedAsyncBus")
        self.subscriptions: Dict[MessageType, Dict[str, Callable]] = {
            message_type: {} for message_type in MessageType
        }
        self.component_ids: Set[str] = set()
        self.message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running = False
        self.processing_task = None
        self.logger.info("UnifiedAsyncBus initialized")

    async def start(self):
        """Start the async bus message processing loop."""
        if self.running:
            return

        self.running = True
        self.processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("AsyncBus message processing started")

    async def stop(self):
        """Stop the async bus message processing."""
        if not self.running:
            return

        self.running = False
        if self.processing_task:
            await self.processing_task
            self.processing_task = None

        self.logger.info("AsyncBus message processing stopped")

    def register_component(self, component_id: str) -> bool:
        """Register a component with the async bus."""
        if component_id in self.component_ids:
            self.logger.warning(f"Component '{component_id}' already registered")
            return False

        self.component_ids.add(component_id)
        self.logger.info(f"Component '{component_id}' registered with async bus")
        return True

    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from the async bus."""
        if component_id not in self.component_ids:
            self.logger.warning(f"Component '{component_id}' not registered")
            return False

        self.component_ids.remove(component_id)

        # Remove all subscriptions for this component
        for message_type in MessageType:
            if component_id in self.subscriptions[message_type]:
                del self.subscriptions[message_type][component_id]

        self.logger.info(f"Component '{component_id}' unregistered from async bus")
        return True

    def subscribe(
        self, component_id: str, message_type: MessageType, callback: Callable
    ) -> bool:
        """Subscribe a component to a specific message type."""
        if component_id not in self.component_ids:
            self.logger.warning(
                f"Cannot subscribe: Component '{component_id}' not registered"
            )
            return False

        self.subscriptions[message_type][component_id] = callback
        self.logger.info(
            f"Component '{component_id}' subscribed to '{message_type.value}' messages"
        )
        return True

    def unsubscribe(self, component_id: str, message_type: MessageType) -> bool:
        """Unsubscribe a component from a specific message type."""
        if component_id not in self.component_ids:
            return False

        if component_id in self.subscriptions[message_type]:
            del self.subscriptions[message_type][component_id]
            self.logger.info(
                f"Component '{component_id}' unsubscribed from '{message_type.value}' messages"
            )
            return True

        return False

    async def publish(self, message: AsyncMessage) -> bool:
        """Publish a message to the async bus."""
        if message.sender_id not in self.component_ids:
            self.logger.warning(
                f"Cannot publish: Sender '{message.sender_id}' not registered"
            )
            return False

        # Add to priority queue with priority value as first item for sorting
        await self.message_queue.put((message.priority.value, message))

        self.logger.debug(
            f"Message from '{message.sender_id}' of type '{message.message_type.value}' published"
        )
        return True

    async def _process_messages(self):
        """Process messages from the queue based on priority."""
        while self.running:
            try:
                # Wait for a message with a timeout to allow for clean shutdown
                try:
                    priority, message = await asyncio.wait_for(
                        self.message_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                await self._deliver_message(message)
                self.message_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    async def _deliver_message(self, message: AsyncMessage):
        """Deliver a message to all subscribed components."""
        delivered = False

        # If target_ids is specified, only deliver to those components
        if message.target_ids:
            for target_id in message.target_ids:
                if target_id in self.subscriptions[message.message_type]:
                    callback = self.subscriptions[message.message_type][target_id]
                    try:
                        await self._call_callback(callback, message)
                        delivered = True
                    except Exception as e:
                        self.logger.error(
                            f"Error delivering targeted message to '{target_id}': {e}"
                        )
        else:
            # Broadcast to all subscribers of this message type
            for component_id, callback in self.subscriptions[
                message.message_type
            ].items():
                if component_id != message.sender_id:  # Don't send back to sender
                    try:
                        await self._call_callback(callback, message)
                        delivered = True
                    except Exception as e:
                        self.logger.error(
                            f"Error delivering broadcast message to '{component_id}': {e}"
                        )

        if not delivered:
            self.logger.debug(
                f"No recipients for message type '{message.message_type.value}' from '{message.sender_id}'"
            )

    async def _call_callback(self, callback: Callable, message: AsyncMessage):
        """Call a callback with proper handling of async vs sync functions."""
        if asyncio.iscoroutinefunction(callback):
            await callback(message)
        else:
            callback(message)
