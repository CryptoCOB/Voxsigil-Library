"""
Validation Utilities for VoxSigil Library

Provides schema validation for agent configurations, signals, and session state.
Used by molt agents to ensure data integrity and correctness.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class AgentValidator:
    """Validator for VoxSigil agent configurations and data."""
    
    @staticmethod
    def validate_signal(signal: Dict[str, Any]) -> bool:
        """
        Validate a prediction signal.
        
        Args:
            signal: Signal dictionary to validate.
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        required_fields = [
            'agent_id', 'market_id', 'prediction',
            'confidence', 'timestamp'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in signal:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate prediction range
        if not 0 <= signal['prediction'] <= 1:
            raise ValidationError(
                f"Prediction must be between 0 and 1, got {signal['prediction']}"
            )
        
        # Validate confidence range
        if not 0 <= signal['confidence'] <= 1:
            raise ValidationError(
                f"Confidence must be between 0 and 1, got {signal['confidence']}"
            )
        
        # Validate timestamp format (basic check)
        if not isinstance(signal['timestamp'], str):
            raise ValidationError("Timestamp must be a string")
        
        return True
    
    @staticmethod
    def validate_session_state(state: Dict[str, Any]) -> bool:
        """
        Validate session state structure.
        
        Args:
            state: Session state dictionary to validate.
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        required_sections = [
            'metadata', 'configuration', 'active_predictions',
            'signal_history', 'performance_metrics', 'network_state'
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in state:
                raise ValidationError(f"Missing required section: {section}")
        
        # Validate metadata
        metadata = state['metadata']
        required_metadata = ['agent_id', 'session_id', 'version', 'created_at']
        for field in required_metadata:
            if field not in metadata:
                raise ValidationError(f"Missing metadata field: {field}")
        
        # Validate active predictions
        for pred in state['active_predictions']:
            if 'prediction_id' not in pred:
                raise ValidationError("Active prediction missing prediction_id")
            if not 0 <= pred.get('probability', -1) <= 1:
                raise ValidationError(
                    f"Invalid probability in prediction {pred.get('prediction_id')}"
                )
        
        return True
    
    @staticmethod
    def validate_hooks_config(config: Dict[str, Any]) -> bool:
        """
        Validate hooks configuration.
        
        Args:
            config: Hooks configuration dictionary to validate.
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        if 'hooks' not in config:
            raise ValidationError("Missing 'hooks' section")
        
        hooks = config['hooks']
        
        for hook_name, hook_config in hooks.items():
            # Check required fields
            if 'enabled' not in hook_config:
                raise ValidationError(f"Hook {hook_name} missing 'enabled' field")
            if 'trigger' not in hook_config:
                raise ValidationError(f"Hook {hook_name} missing 'trigger' field")
            
            # Validate trigger types
            valid_triggers = [
                'on_startup', 'on_shutdown', 'on_command_execute',
                'on_error', 'on_prediction_resolve', 'periodic'
            ]
            if hook_config['trigger'] not in valid_triggers:
                raise ValidationError(
                    f"Invalid trigger '{hook_config['trigger']}' for hook {hook_name}"
                )
        
        return True
    
    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> bool:
        """
        Validate complete agent configuration.
        
        Args:
            config: Agent configuration dictionary to validate.
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        required_keys = ['boot', 'agents', 'memory', 'hooks']
        
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required config key: {key}")
        
        # Validate hooks as JSON
        if isinstance(config['hooks'], str):
            try:
                hooks = json.loads(config['hooks'])
                AgentValidator.validate_hooks_config(hooks)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON in hooks config: {e}")
        else:
            AgentValidator.validate_hooks_config(config['hooks'])
        
        return True
    
    @staticmethod
    def validate_prediction(prediction: Dict[str, Any]) -> bool:
        """
        Validate a prediction object.
        
        Args:
            prediction: Prediction dictionary to validate.
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        required_fields = ['market_id', 'probability']
        
        for field in required_fields:
            if field not in prediction:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate probability
        if not 0 <= prediction['probability'] <= 1:
            raise ValidationError(
                f"Probability must be between 0 and 1, got {prediction['probability']}"
            )
        
        # Validate confidence interval if present
        if 'confidence_interval' in prediction:
            interval = prediction['confidence_interval']
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValidationError("Confidence interval must be [lower, upper]")
            if not (0 <= interval[0] <= interval[1] <= 1):
                raise ValidationError(
                    f"Invalid confidence interval: {interval}"
                )
        
        return True
    
    @staticmethod
    def validate_file(filepath: Path, expected_type: str) -> bool:
        """
        Validate a file exists and has expected format.
        
        Args:
            filepath: Path to file.
            expected_type: Expected file type ('json', 'md', 'txt').
            
        Returns:
            True if valid.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not filepath.exists():
            raise ValidationError(f"File does not exist: {filepath}")
        
        if expected_type == 'json':
            try:
                with open(filepath, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON in {filepath}: {e}")
        
        elif expected_type in ('md', 'txt'):
            # Just verify it's readable
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    f.read()
            except Exception as e:
                raise ValidationError(f"Cannot read file {filepath}: {e}")
        
        return True


def validate_signal(signal: Dict[str, Any]) -> bool:
    """Convenience function to validate signal."""
    return AgentValidator.validate_signal(signal)


def validate_session_state(state: Dict[str, Any]) -> bool:
    """Convenience function to validate session state."""
    return AgentValidator.validate_session_state(state)


def validate_hooks_config(config: Dict[str, Any]) -> bool:
    """Convenience function to validate hooks config."""
    return AgentValidator.validate_hooks_config(config)


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """Convenience function to validate agent config."""
    return AgentValidator.validate_agent_config(config)
