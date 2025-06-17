#!/usr/bin/env python3
"""
Conversation Orchestrator - Routes and processes user commands
==============================================================

Handles routing of user commands from the Control Center to appropriate
subsystems and orchestrates responses back to the user interface.
"""

import logging
import time
from typing import Callable, Dict

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Orchestrates conversations between user interface and system components.

    Routes commands to appropriate handlers and manages response flow.
    """

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.command_handlers = {}
        self.conversation_history = []
        self.active_contexts = {}

        # Register built-in handlers
        self.register_builtin_handlers()

        # Setup event subscriptions
        if self.event_bus:
            self.event_bus.subscribe("user.command", self.handle_user_command)

    def register_builtin_handlers(self):
        """Register built-in command handlers"""

        def ping_handler(command: str, context: Dict) -> str:
            return "pong"

        def status_handler(command: str, context: Dict) -> str:
            return f"System status: Active. Time: {time.strftime('%H:%M:%S')}"

        def help_handler(command: str, context: Dict) -> str:
            handlers = list(self.command_handlers.keys())
            return f"Available commands: {', '.join(handlers)}"

        def echo_handler(command: str, context: Dict) -> str:
            parts = command.split(None, 1)
            if len(parts) > 1:
                return parts[1]
            return "echo requires text to echo"

        def context_handler(command: str, context: Dict) -> str:
            return f"Current context: {len(self.active_contexts)} active sessions"

        # Register handlers
        self.register_handler("ping", ping_handler)
        self.register_handler("status", status_handler)
        self.register_handler("help", help_handler)
        self.register_handler("echo", echo_handler)
        self.register_handler("context", context_handler)

    def register_handler(self, command: str, handler: Callable[[str, Dict], str]):
        """Register a command handler"""
        self.command_handlers[command.lower()] = handler
        logger.info(f"Registered command handler: {command}")

    def handle_user_command(self, event):
        """Handle user command events from the event bus"""
        try:
            data = event.get("data", {})
            command = data.get("text", "").strip()
            context = data.get("context", {})
            user_id = context.get("user_id", "default")

            if not command:
                return

            logger.info(f"Processing command from user {user_id}: {command}")

            # Add to conversation history
            self.conversation_history.append(
                {
                    "timestamp": time.time(),
                    "user_id": user_id,
                    "command": command,
                    "type": "user_input",
                }
            )

            # Process command
            response = self.process_command(command, context)

            # Add response to history
            self.conversation_history.append(
                {
                    "timestamp": time.time(),
                    "user_id": user_id,
                    "response": response,
                    "type": "system_response",
                }
            )

            # Send response back through event bus
            if self.event_bus:
                self.event_bus.publish(
                    "command.reply",
                    {"text": response, "context": context, "timestamp": time.time()},
                )

        except Exception as e:
            logger.error(f"Error handling user command: {e}")
            if self.event_bus:
                self.event_bus.publish(
                    "command.reply",
                    {"text": f"Error processing command: {e}", "timestamp": time.time()},
                )

    def process_command(self, command: str, context: Dict) -> str:
        """Process a command and return response"""
        try:
            # Parse command
            if command.startswith("/"):
                # Slash command
                parts = command[1:].split()
                cmd = parts[0].lower() if parts else ""

                if cmd in self.command_handlers:
                    return self.command_handlers[cmd](command, context)
                else:
                    return f"Unknown command: {cmd}. Type /help for available commands."

            else:
                # Natural language or other processing
                return self.process_natural_language(command, context)

        except Exception as e:
            logger.error(f"Error processing command '{command}': {e}")
            return f"Error processing command: {e}"

    def process_natural_language(self, text: str, context: Dict) -> str:
        """Process natural language input"""
        # This could be enhanced with AI/LLM integration

        # Simple keyword-based responses for now
        text_lower = text.lower()

        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm the VoxSigil assistant. How can I help you today?"

        elif any(word in text_lower for word in ["status", "health", "system"]):
            return "System is running normally. All core components are active."

        elif any(word in text_lower for word in ["train", "training"]):
            return "Training system is available. Use /training commands for more options."

        elif any(word in text_lower for word in ["help", "assist", "support"]):
            return "I can help with system commands, status checks, and basic queries. Type /help for command list."

        elif any(word in text_lower for word in ["tab", "tabs", "interface"]):
            return "Multiple tabs are available for different system components. Check the tab bar above."

        else:
            return f"I understand you said: '{text}'. I'm still learning - try slash commands like /help for now."

    def update_context(self, user_id: str, context_updates: Dict):
        """Update context for a user session"""
        if user_id not in self.active_contexts:
            self.active_contexts[user_id] = {}

        self.active_contexts[user_id].update(context_updates)

        # Publish context update
        if self.event_bus:
            self.event_bus.publish(
                "context.updated", {"user_id": user_id, "context": self.active_contexts[user_id]}
            )

    def get_conversation_history(self, user_id: str = None, limit: int = 50) -> list:
        """Get conversation history for a user or all users"""
        history = self.conversation_history

        if user_id:
            history = [item for item in history if item.get("user_id") == user_id]

        return history[-limit:] if limit else history

    def clear_history(self, user_id: str = None):
        """Clear conversation history"""
        if user_id:
            self.conversation_history = [
                item for item in self.conversation_history if item.get("user_id") != user_id
            ]
        else:
            self.conversation_history.clear()

        logger.info(f"Cleared conversation history for user: {user_id or 'all users'}")


# For testing
def test_orchestrator():
    """Test the conversation orchestrator"""

    class MockEventBus:
        def __init__(self):
            self.subscribers = {}
            self.published = []

        def subscribe(self, topic, handler):
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(handler)

        def publish(self, topic, data):
            self.published.append((topic, data))
            if topic in self.subscribers:
                for handler in self.subscribers[topic]:
                    handler({"data": data})

    # Create orchestrator with mock bus
    bus = MockEventBus()
    _ = ConversationOrchestrator(bus)  # instantiate orchestrator (kept alive by bus subscriptions)

    # Test commands
    test_commands = [
        "/ping",
        "/help",
        "/status",
        "hello there",
        "what is the system status?",
        "/unknown_command",
    ]

    for cmd in test_commands:
        print(f"\nCommand: {cmd}")
        bus.publish("user.command", {"text": cmd, "context": {"user_id": "test"}})

        # Check responses
        for topic, data in bus.published:
            if topic == "command.reply":
                print(f"Response: {data['text']}")

        bus.published.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_orchestrator()
