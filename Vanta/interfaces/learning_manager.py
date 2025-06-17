"""
Learning Manager interface for Vanta
"""

class LearningManager:
    """Manages learning processes and adaptation."""
    
    def __init__(self, learning_rate=0.01, adaptation_enabled=True):
        self.learning_rate = learning_rate
        self.adaptation_enabled = adaptation_enabled
        self.learning_history = []
        self.current_learning_state = "idle"
    
    def start_learning(self, task_type="general"):
        """Start a learning session."""
        self.current_learning_state = "active"
        logger.info(f"Started learning session for task: {task_type}")
        return True
    
    def stop_learning(self):
        """Stop the current learning session."""
        self.current_learning_state = "idle"
        logger.info("Stopped learning session")
        return True
    
    def get_learning_status(self):
        """Get current learning status."""
        return {
            "state": self.current_learning_state,
            "learning_rate": self.learning_rate,
            "adaptation_enabled": self.adaptation_enabled,
            "history_length": len(self.learning_history)
        }
    
    def update_learning_rate(self, new_rate):
        """Update the learning rate."""
        self.learning_rate = max(0.001, min(1.0, new_rate))
    
    def add_learning_experience(self, experience):
        """Add a learning experience to history."""
        self.learning_history.append(experience)
        if len(self.learning_history) > 1000:  # Keep history manageable
            self.learning_history = self.learning_history[-1000:]
    
    def get_recommendations(self):
        """Get learning-based recommendations."""
        return {"recommendation": "continue_current_approach", "confidence": 0.8}

import logging
logger = logging.getLogger(__name__)
